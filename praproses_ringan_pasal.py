"""
Preprocessing ringan PDF regulasi Indonesia untuk RAG, chunk per Pasal.

Extractor:
  - PyMuPDF: cepat, cocok untuk batch besar.
  - pdfplumber: fallback untuk layout/character mapping yang lebih rapi.

Tidak memakai Docling, OCR, Transformers, atau model download.

Setup:
    pip install pymupdf pdfplumber

Contoh:
    python praproses_ringan_pasal.py
    python praproses_ringan_pasal.py --limit 5 --no-move-failed
    python praproses_ringan_pasal.py --pdf data/pdf/peraturan-menteri/PER-11-MEN-VI-2005.pdf --extractor auto
"""

import argparse
import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import fitz

DEFAULT_PDF_DIR = Path("data") / "pdf"
DEFAULT_OUTPUT_PATH = Path("data") / "processed_chunks_ringan_pasal.json"
DEFAULT_OCR_DIR = Path("data") / "pdf_butuh_ocr"
DEFAULT_RAW_TEXT_DIR = Path("data") / "raw_text_ringan"

MIN_EXTRACTED_CHARS = 300
MIN_PASAL_COUNT = 1
MAX_NOISE_RATIO = 0.18
MAX_OCR_ARTIFACT_COUNT = 500

REGULATION_HIERARCHY = {
    "Undang-Undang": 1,
    "Peraturan Pemerintah Pengganti Undang-Undang": 1,
    "Peraturan Pemerintah": 2,
    "Peraturan Presiden": 3,
    "Peraturan Menteri Ketenagakerjaan": 4,
    "Peraturan Menteri": 4,
    "Keputusan Presiden": 4,
    "Keputusan Menteri": 5,
    "Instruksi Presiden": 5,
    "Putusan MK": 1,
    "Putusan MA": 3,
    "Putusan": 3,
    "Unknown": 99,
}


@dataclass
class PageExtraction:
    page_number: int
    text: str
    extractor: str
    score: float


@dataclass
class SectionContext:
    bab: str = ""
    bab_title: str = ""
    bagian: str = ""
    bagian_title: str = ""
    paragraf: str = ""
    paragraf_title: str = ""


def compact_spaces(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def normalize_ocr_number_token(token: str) -> str:
    return token.translate(str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "L": "1", "|": "1"}))


def _fix_split_numlist(match: re.Match) -> str:
    leading, d1, d2 = match.group(1), match.group(2), match.group(3)
    return f"{leading}{d2}. " if d1 == d2 else f"{leading}{d1}{d2}. "


def normalize_legal_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    text = text.replace("\uf0b7", "-").replace("\xad\n", "").replace("\xad", "")
    text = re.sub(r"(?<=\d)[Oo](?=\d)|(?<=\d)[Il|](?=\d)", lambda m: normalize_ocr_number_token(m.group(0)), text)
    text = re.sub(r"\b(1|2)[0Oo][0-9OoIlL|]{2}\b", lambda m: normalize_ocr_number_token(m.group(0)), text)
    text = re.sub(r"(?m)^\s*\((\d{1,2})[Il1]\)\s*", r"(\1) ", text)
    text = re.sub(r"(?m)^\s*\((\d{1,2})[Il1]\s+(?=[A-Z])", r"(\1) ", text)
    text = re.sub(r"(?m)^\s*\((\d{1,2})\s+(?=[A-Z])", r"(\1) ", text)
    text = re.sub(r"(?m)^(\s*)(\d)\s+(\d{1,2})\.\s+", _fix_split_numlist, text)
    text = re.sub(r"\bP\s*a\s*s\s*a\s*l\b", "Pasal", text, flags=re.IGNORECASE)
    text = re.sub(r"\bPasa[lI1]\b", "Pasal", text, flags=re.IGNORECASE)
    text = re.sub(r"\bPas\.?il\b", "Pasal", text, flags=re.IGNORECASE)
    text = re.sub(r"\bB\s*A\s*B\b", "BAB", text, flags=re.IGNORECASE)
    text = re.sub(r"\bT\s*E\s*N\s*T\s*A\s*N\s*G\b", "TENTANG", text, flags=re.IGNORECASE)
    text = re.sub(r"\bM\s*E\s*M\s*U\s*T\s*U\s*S\s*K\s*A\s*N\b", "MEMUTUSKAN", text, flags=re.IGNORECASE)
    return text


def remove_noise_lines(text: str) -> str:
    kept = []
    for raw_line in text.splitlines():
        line = compact_spaces(raw_line)
        if not line:
            kept.append("")
            continue
        upper = line.upper()
        if re.fullmatch(r"-\s*\d+\s*-", line):
            continue
        if re.fullmatch(r"PRES\s*I?DEN", upper) or re.fullmatch(r"REPUBLIK\s+INDONESIA", upper):
            continue
        if re.search(r"\bJDIH\b|WWW\.", upper):
            continue
        if re.fullmatch(r"(SK|SL|S1|SI)[(<]?\s*NO\s+[A-Z0-9\s./|'-]+", upper):
            continue
        kept.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()


def clean_extracted_text(text: str) -> str:
    text = normalize_legal_text(text)
    text = remove_noise_lines(text)
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def score_text_quality(text: str) -> float:
    if not text:
        return -999.0
    words = re.findall(r"\S+", text)
    word_count = max(len(words), 1)
    pasal_count = len(re.findall(r"(?m)^\s*Pasal\s+(?:\d+[A-Z]?|[IVXLCDM]+)\b", text, re.IGNORECASE))
    legal_terms = len(re.findall(r"\b(Pasal|ayat|huruf|BAB|Bagian|Paragraf|Menimbang|Mengingat|MEMUTUSKAN|TENTANG)\b", text, re.IGNORECASE))
    weird_chars = len(re.findall(r"[^\w\s.,;:!?()\[\]{}'\"/\-+%=<>@#&*]", text, flags=re.UNICODE))
    mojibake = len(re.findall(r"Â|â|�|\\x[0-9a-fA-F]{2}", text))
    no_vowel_words = sum(
        1 for word in words
        if len(re.sub(r"[^A-Za-z]", "", word)) >= 5 and not re.search(r"[aeiouAEIOU]", word)
    )
    ocr_artifacts = len(re.findall(
        r"\b(scbagai|scbagaimana|dalarn|dimak\s*sud|avat|ava\s*t|liuruf|pcr|kcb|sctiap|tcr|dcn|clan|h\.?uus|udarn)\b",
        text,
        re.IGNORECASE,
    ))
    return (
        min(len(text), 5000) / 200
        + pasal_count * 30
        + legal_terms * 1.5
        - weird_chars * 0.6
        - mojibake * 4
        - (no_vowel_words / word_count) * 80
        - ocr_artifacts * 2
    )


def extraction_quality(text: str) -> Dict:
    words = re.findall(r"\S+", text)
    pasal_count = len(re.findall(r"(?m)^\s*Pasal\s+(?:\d+[A-Z]?|[IVXLCDM]+)\b", text, re.IGNORECASE))
    weird_chars = len(re.findall(r"[^\w\s.,;:!?()\[\]{}'\"/\-+%=<>@#&*]", text, flags=re.UNICODE))
    mojibake = len(re.findall(r"Â|â|�|\\x[0-9a-fA-F]{2}", text))
    ocr_artifacts = len(re.findall(
        r"\b(scbagai|scbagaimana|dalarn|dimak\s*sud|avat|ava\s*t|liuruf|pcr|kcb|sctiap|tcr|dcn|clan|h\.?uus|udarn)\b",
        text,
        re.IGNORECASE,
    ))
    noise_ratio = (weird_chars + mojibake * 3 + ocr_artifacts * 5) / max(len(words), 1)
    return {
        "char_count": len(text),
        "word_count": len(words),
        "pasal_count": pasal_count,
        "noise_ratio": round(noise_ratio, 4),
        "ocr_artifact_count": ocr_artifacts,
    }


def is_extraction_usable(
    text: str,
    min_chars: int,
    min_pasal_count: int,
    max_noise_ratio: float,
    max_ocr_artifact_count: int,
) -> Tuple[bool, str, Dict]:
    quality = extraction_quality(text)
    if quality["char_count"] < min_chars:
        return False, f"teks terlalu pendek ({quality['char_count']} char)", quality
    if quality["pasal_count"] < min_pasal_count:
        return False, f"Pasal terdeteksi kurang dari {min_pasal_count}", quality
    if quality["noise_ratio"] > max_noise_ratio:
        return False, f"noise terlalu tinggi ({quality['noise_ratio']})", quality
    if quality["ocr_artifact_count"] > max_ocr_artifact_count:
        return False, f"artefak OCR terlalu banyak ({quality['ocr_artifact_count']})", quality
    return True, "ok", quality


def extract_pymupdf_pages(pdf_path: Path) -> List[str]:
    pages = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            pages.append(page.get_text("text", sort=True).strip())
    return pages


def load_pdfplumber():
    try:
        import pdfplumber
    except ImportError as exc:
        raise RuntimeError("pdfplumber belum terinstall. Jalankan: pip install pdfplumber") from exc
    return pdfplumber


def extract_pdfplumber_pages(pdf_path: Path, layout: bool = True) -> List[str]:
    pdfplumber = load_pdfplumber()
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text(
                layout=layout,
                x_tolerance=1.5,
                y_tolerance=3,
                keep_blank_chars=False,
                # Text-flow order follows the PDF's internal character stream.
                # Many scanned/legal PDFs expose a selectable text layer whose
                # stream order differs from the visual layout, causing chunks to
                # start mid-page or read bottom-to-top. Position order is more
                # stable for regulation text.
                use_text_flow=False,
            )
            pages.append((text or "").strip())
    return pages


def page_extraction(page_number: int, raw_text: str, extractor: str) -> PageExtraction:
    text = clean_extracted_text(raw_text)
    return PageExtraction(page_number, text, extractor, score_text_quality(text))


def choose_best_pages(pymupdf_pages: List[str], pdfplumber_pages: List[str]) -> List[PageExtraction]:
    page_count = max(len(pymupdf_pages), len(pdfplumber_pages))
    chosen = []
    for idx in range(page_count):
        candidates = []
        if idx < len(pymupdf_pages):
            candidates.append(page_extraction(idx + 1, pymupdf_pages[idx], "pymupdf"))
        if idx < len(pdfplumber_pages):
            candidates.append(page_extraction(idx + 1, pdfplumber_pages[idx], "pdfplumber"))
        best = max(candidates, key=lambda item: item.score) if candidates else PageExtraction(idx + 1, "", "none", -999.0)
        chosen.append(best)
    return chosen


def extract_and_clean_text(pdf_path: Path, extractor: str) -> Tuple[str, Dict]:
    pymupdf_pages: List[str] = []
    pdfplumber_pages: List[str] = []

    if extractor in ("auto", "pymupdf"):
        pymupdf_pages = extract_pymupdf_pages(pdf_path)
    if extractor in ("auto", "pdfplumber"):
        pdfplumber_pages = extract_pdfplumber_pages(pdf_path)

    if extractor == "pymupdf":
        chosen = [
            page_extraction(i + 1, text, "pymupdf")
            for i, text in enumerate(pymupdf_pages)
        ]
    elif extractor == "pdfplumber":
        chosen = [
            page_extraction(i + 1, text, "pdfplumber")
            for i, text in enumerate(pdfplumber_pages)
        ]
    else:
        chosen = choose_best_pages(pymupdf_pages, pdfplumber_pages)

    text = "\n\n".join(page.text for page in chosen if page.text).strip()
    text = clean_extracted_text(text)
    extractor_counts: Dict[str, int] = {}
    for page in chosen:
        extractor_counts[page.extractor] = extractor_counts.get(page.extractor, 0) + 1
    return text, {"page_count": len(chosen), "extractor_counts": extractor_counts}


REGULATION_PATTERNS: List[Tuple[str, str]] = [
    (r"\bUNDANG[-\s]?UNDANG\b|\bUU\b", "Undang-Undang"),
    (r"\bPERATURAN\s+PEMERINTAH\s+PENGGANTI\s+UNDANG[-\s]?UNDANG\b|\bPERPPU\b", "Peraturan Pemerintah Pengganti Undang-Undang"),
    (r"\bPERATURAN\s+PEMERINTAH\b|\bPP\b", "Peraturan Pemerintah"),
    (r"\bPERATURAN\s+PRESIDEN\b|\bPERPRES\b", "Peraturan Presiden"),
    (r"\bPERATURAN\s+MENTERI\s+KETENAGAKERJAAN\b|\bPERMENAKER\b|\bPMNAKER\b", "Peraturan Menteri Ketenagakerjaan"),
    (r"\bPER[.\-\s]?\d{1,4}[/.\-\s]+MEN\b", "Peraturan Menteri"),
    (r"\bPERATURAN\s+MENTERI\b|\bPERMEN\b", "Peraturan Menteri"),
    (r"\bKEPUTUSAN\s+MENTERI\b|\bKEPMEN\b", "Keputusan Menteri"),
    (r"\bKEPUTUSAN\s+PRESIDEN\b|\bKEPPRES\b", "Keputusan Presiden"),
    (r"\bINSTRUKSI\s+PRESIDEN\b|\bINPRES\b", "Instruksi Presiden"),
    (r"\bPUTUSAN\s+MAHKAMAH\s+KONSTITUSI\b|\bPUTUSAN\s+MK\b", "Putusan MK"),
    (r"\bPUTUSAN\s+MAHKAMAH\s+AGUNG\b|\bPUTUSAN\s+MA\b", "Putusan MA"),
    (r"\bPUTUSAN\b", "Putusan"),
]


def detect_regulation_type(filename: str, head_text: str) -> str:
    if re.search(r"\bPERATURAN\s*MENTERI\s*KETENAGAKERJAAN\b.*?\bNOMOR\b", head_text, re.IGNORECASE):
        return "Peraturan Menteri Ketenagakerjaan"
    if re.search(r"\bPERATURAN\s*MENTERI\b.*?\bNOMOR\b", head_text, re.IGNORECASE):
        return "Peraturan Menteri"
    title_match = re.search(
        r"\b(UNDANG[-\s]?UNDANG|PERATURAN\s+PEMERINTAH\s+PENGGANTI\s+UNDANG[-\s]?UNDANG|"
        r"PERATURAN\s+PEMERINTAH|PERATURAN\s+PRESIDEN|PERATURAN\s+MENTERI\s+KETENAGAKERJAAN|"
        r"PERATURAN\s+MENTERI|KEPUTUSAN\s+MENTERI|KEPUTUSAN\s+PRESIDEN|INSTRUKSI\s+PRESIDEN|"
        r"PUTUSAN\s+MAHKAMAH\s+KONSTITUSI|PUTUSAN\s+MAHKAMAH\s+AGUNG)"
        r"(?:\s+REPUBLIK\s+INDONESIA)?\s+(?:NOMOR|NO\.?)\b",
        head_text,
        re.IGNORECASE,
    )
    if title_match:
        title = title_match.group(1)
        for pattern, label in REGULATION_PATTERNS:
            if re.search(pattern, title, re.IGNORECASE):
                return label
    filename_text = filename.replace("_", " ").replace("-", " ")
    filename_text = re.sub(r"(?i)\b(PERMEN)(?=\d)", r"\1 ", filename_text)
    for pattern, label in REGULATION_PATTERNS:
        if re.search(pattern, filename_text, re.IGNORECASE):
            return label
    for pattern, label in REGULATION_PATTERNS:
        if re.search(pattern, head_text, re.IGNORECASE):
            return label
    return "Unknown"


def extract_nomor_year_from_filename(filename: str) -> Tuple[Optional[str], Optional[int]]:
    raw_name = Path(filename).stem
    filename_text = re.sub(r"[_\-\s]+", " ", raw_name)
    old_permen_match = re.search(
        r"\bPER[.\-\s]?(?P<no>\d{1,4})[-_/.\s]+MEN[-_/.\s]+(?P<month>[IVX]{1,5})[-_/.\s]+(?P<year>19\d{2}|20\d{2})\b",
        raw_name,
        re.IGNORECASE,
    )
    if old_permen_match:
        no = int(old_permen_match.group("no"))
        month = old_permen_match.group("month").upper()
        year = int(old_permen_match.group("year"))
        return f"PER.{no}/MEN/{month}/{year}", year

    name = normalize_ocr_number_token(filename_text)
    complex_match = re.search(r"\b(?P<no>[A-Za-z0-9.-]+(?:/[A-Za-z0-9.-]+){1,3}/(?P<year>19\d{2}|20\d{2}))\b", name)
    if complex_match:
        return complex_match.group("no"), int(complex_match.group("year"))
    compact_match = re.search(r"\b(?:PERPRES|PERMEN|PP|UU)?\D*(?P<no>\d{1,3})(?P<year>19\d{2}|20\d{2})\b", name, re.IGNORECASE)
    if compact_match:
        return str(int(compact_match.group("no"))), int(compact_match.group("year"))
    for pattern in [
        r"(?P<year>19\d{2}|20\d{2})\D*(?P<no>\d{1,4})\b",
        r"\b(?:NOMOR|NO)?\D*(?P<no>\d{1,4})\D*(?:TAHUN|TH)\D*(?P<year>19\d{2}|20\d{2})\b",
        r"\bNO(?:MOR)?\D*(?P<no>\d{1,4})\D*(?P<year>19\d{2}|20\d{2})\b",
    ]:
        match = re.search(pattern, name, re.IGNORECASE)
        if match:
            return str(int(match.group("no"))), int(match.group("year"))
    return None, None


def extract_tentang(head_text: str) -> Optional[str]:
    normalized = re.sub(r"\s+", " ", head_text)

    title_zone = re.split(
        r"\b(Menimbang|Mengingat|MEMUTUSKAN|Menetapkan|Pasal\s+1)\b",
        normalized,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    title_match = re.search(
        r"\bNOMOR\b.{0,160}?\bTENTANG\s+(?P<about>.*?)(?=\bDENGAN\s+RAH\s*MAT\b|\bPRESIDEN\s+REPUBLIK\b|$)",
        title_zone,
        flags=re.IGNORECASE,
    )
    if title_match:
        about = compact_spaces(title_match.group("about")).strip(" .;:")
        if 5 <= len(about) <= 700:
            return about.upper()

    menetapkan_match = re.search(
        r"\bMenetap\w*\s*:\s*.*?\bTENTANG\s+(?P<about>.*?)(?=\.?\s*\b(?:BAB\s+[IVXLCDM]+|Pasal\s+1)\b|\.|$)",
        normalized,
        flags=re.IGNORECASE,
    )
    if menetapkan_match:
        about = compact_spaces(menetapkan_match.group("about")).strip(" .;:")
        if 5 <= len(about) <= 700:
            return about.upper()

    candidates = []
    for pattern in [
        r"\bTENTANG\s+(?P<about>.*?)(?=\bDENGAN\s+RAH\s*MAT\b|\bPRESIDEN\s+REPUBLIK\b|\bMenimbang\b|\bMengingat\b|\bMEMUTUSKAN\b|\bBAB\s+[IVXLCDM]+\b|\bPasal\s+1\b)",
    ]:
        for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
            about = compact_spaces(match.group("about"))
            about = re.sub(r"^(?:PERATURAN|UNDANG-UNDANG|KEPUTUSAN|INSTRUKSI).*?\bTENTANG\s+", "", about, flags=re.IGNORECASE)
            about = about.strip(" .;:")
            if 5 <= len(about) <= 700:
                candidates.append(about)
    return max(candidates, key=len).upper() if candidates else None


def extract_metadata(filepath: Path, text_content: str) -> Dict:
    filename = filepath.name
    head_text = re.sub(r"\s+", " ", normalize_legal_text(text_content[:10000]))
    title_zone = re.split(
        r"\b(Menimbang|Mengingat|MEMUTUSKAN|Menetapkan|Pasal\s+1)\b",
        head_text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    regulation_type = detect_regulation_type(filename, head_text)
    nomor, year = None, None
    complex_match = re.search(
        r"\b(?:NOMOR|NO\.?)\s*:?\s*(?P<no>[A-Za-z0-9.-]+(?:/[A-Za-z0-9.-]+){1,3}/(?P<year>19\d{2}|20\d{2}))\b",
        title_zone,
        re.IGNORECASE,
    )
    if complex_match:
        nomor = complex_match.group("no").strip("./- ")
        year = int(complex_match.group("year"))
    else:
        title_match = re.search(
            r"\b(?:NOMOR|NO\.?)\s*(?P<no>[0-9A-Za-z./-]+)\s*(?:TAHUN|TH\.?)\s*(?P<year>19\d{2}|20\d{2})\b",
            title_zone,
            re.IGNORECASE,
        )
        if title_match:
            nomor = title_match.group("no").strip("./- ")
            year = int(title_match.group("year"))
        else:
            nomor, year = extract_nomor_year_from_filename(filename)

    if nomor and re.search(r"\bPER[.\-/]?\d{1,4}.*\bMEN\b", nomor, re.IGNORECASE):
        regulation_type = "Peraturan Menteri"
    if re.search(r"\bMENTERI\s+(TENAGA\s+KERJA|KETENAGAKERJAAN|TENAGA\s+KERJA\s+DAN\s+TRANSMIGRASI)\b", head_text, re.IGNORECASE):
        if regulation_type in ("Unknown", "Undang-Undang", "Peraturan Menteri"):
            regulation_type = "Peraturan Menteri Ketenagakerjaan"

    about = extract_tentang(head_text)
    hierarchy = REGULATION_HIERARCHY.get(regulation_type, 99)
    return {
        "regulation_type": regulation_type,
        "regulation_hierarchy": hierarchy,
        "nomor": nomor or "Unknown",
        "tentang": about or "Unknown",
        "year": year or 0,
        "publication_year": year or 0,
        "active_status": "Berlaku",
        "source_file": filename,
        "source_path": str(filepath),
        "extractor": "pymupdf_pdfplumber",
        "chunk_level": "pasal",
    }


def canonical_section_title(lines: List[str], idx: int) -> str:
    for next_line in lines[idx + 1 : idx + 4]:
        candidate = compact_spaces(next_line)
        if candidate and not re.match(r"^(BAB|Bagian|Paragraf|Pasal)\b", candidate, re.IGNORECASE):
            return candidate.title()
    return ""


def build_section_index(text: str) -> List[Tuple[int, SectionContext]]:
    lines = text.splitlines()
    index: List[Tuple[int, SectionContext]] = []
    offset = 0
    ctx = SectionContext()
    for idx, line in enumerate(lines):
        clean = compact_spaces(line)
        if re.fullmatch(r"BAB\s+([IVXLCDM]+|\d+)", clean, re.IGNORECASE):
            ctx = SectionContext(bab=clean.upper(), bab_title=canonical_section_title(lines, idx))
            index.append((offset, SectionContext(**ctx.__dict__)))
        elif re.fullmatch(r"Bagian\s+Ke\w+", clean, re.IGNORECASE):
            ctx.bagian = clean.title()
            ctx.bagian_title = canonical_section_title(lines, idx)
            index.append((offset, SectionContext(**ctx.__dict__)))
        elif re.fullmatch(r"Paragraf\s+(\d+|\w+)", clean, re.IGNORECASE):
            ctx.paragraf = clean.title()
            ctx.paragraf_title = canonical_section_title(lines, idx)
            index.append((offset, SectionContext(**ctx.__dict__)))
        offset += len(line) + 1
    return index


def context_at(index: List[Tuple[int, SectionContext]], position: int) -> SectionContext:
    current = SectionContext()
    for offset, ctx in index:
        if offset <= position:
            current = ctx
        else:
            break
    return current


def strip_before_body(text: str) -> str:
    marker = re.search(r"(?:^|\n)(BAB\s+I\b|Pasal\s+1\b|PERTAMA\b|KESATU\b)", text, re.IGNORECASE)
    return text[marker.start():].strip() if marker else text


def strip_after_explanation(text: str) -> str:
    return re.split(r"\bPENJELASAN\s+ATAS\b", text, maxsplit=1, flags=re.IGNORECASE)[0].strip()


def strip_after_appendix(text: str) -> str:
    # Lampiran headers are often OCR-damaged in scanned legal PDFs, e.g.
    # "LAMPil~A:'J I" followed by "PERATURAN MENTERI ...".
    # Do not cut on ordinary body references like "tercantum dalam Lampiran I";
    # only cut when the candidate line is followed by appendix-title markers.
    strict = re.search(
        r"(?m)^\s*LAMPIRAN\s+(?:PERATURAN|UNDANG[-\s]?UNDANG|KEPUTUSAN|INSTRUKSI)\b",
        text,
        flags=re.IGNORECASE,
    )
    if strict:
        return text[: strict.start()].strip()

    inline_noisy = re.search(
        r"\bL\s*A\s*M\s*P[^\n]{0,40}?\b(?:PERATURAN|UNDANG[-\s]?UNDANG|KEPUTUSAN|INSTRUKSI)\b"
        r".{0,300}?\b(?:NOMOR|TENTANG)\b",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if inline_noisy:
        return text[: inline_noisy.start()].strip()

    noisy_header = re.compile(
        r"(?mi)^\s*L\s*A\s*M\s*P[^\n]{0,20}(?:\b[IVXLCDM]+\b|\b\d+\b)?\s*$"
    )
    for match in noisy_header.finditer(text):
        lookahead = text[match.end() : match.end() + 1200]
        if re.search(
            r"\b(PERATURAN|UNDANG[-\s]?UNDANG|KEPUTUSAN|INSTRUKSI)\b.{0,250}\b(NOMOR|TENTANG)\b",
            lookahead,
            flags=re.IGNORECASE | re.DOTALL,
        ):
            return text[: match.start()].strip()
    return text.strip()


def normalize_pasal_id(raw_pasal_id: str) -> str:
    normalized = re.sub(r"\s+", " ", raw_pasal_id).strip()
    normalized = re.sub(r"(?<=\d)[Oo](?=\d|\b)", "0", normalized)
    match = re.fullmatch(r"Pasal\s+([IlL|])\s*(\d{1,3}[A-Z]?)", normalized, flags=re.IGNORECASE)
    if match:
        return f"Pasal 1{match.group(2)}"
    split_number = re.fullmatch(r"Pasal\s+([0-9]+(?:\s+[0-9]+){1,3}[A-Z]?)", normalized, flags=re.IGNORECASE)
    if split_number:
        return f"Pasal {''.join(split_number.group(1).split())}"
    return normalized.title()


def split_pasals(text: str) -> Iterable[Tuple[int, str, str]]:
    pasal_pattern = r"(?m)^[ \t]*(Pasal\s+(?:[IlL|]\s*\d{1,3}[A-Z]?|\d+[Oo]?(?:\s+\d+[Oo]?){1,3}[A-Z]?|\d+[A-Z]?|[IVXLCDM]+))[ \t.:]*(?=$)"
    matches = list(re.finditer(pasal_pattern, text, re.IGNORECASE))
    if not matches:
        diktum_pattern = (
            r"(?m)^\s*(PERTAMA|KESATU|KEDUA|KETIGA|KEEMPAT|KELIMA|KEENAM|KETUJUH|"
            r"KEDELAPAN|KESEMBILAN|KESEPULUH|KESEBELAS|KEDUA\s*BELAS|KETIGA\s*BELAS)[\s.:]*"
        )
        matches = list(re.finditer(diktum_pattern, text, re.IGNORECASE))
    if not matches:
        if text.strip():
            yield 0, "Isi Dokumen", text.strip()
        return
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        yield start, normalize_pasal_id(match.group(1)), text[match.end():end].strip()


def strip_trailing_section_headers(content: str) -> str:
    lines = content.strip().splitlines()
    for idx in range(len(lines) - 1, -1, -1):
        line = compact_spaces(lines[idx])
        if re.fullmatch(r"BAB\s+([IVXLCDM]+|\d+)", line, re.IGNORECASE):
            return "\n".join(lines[:idx]).strip()
        if re.fullmatch(r"Bagian\s+Ke\w+", line, re.IGNORECASE):
            return "\n".join(lines[:idx]).strip()
        if re.fullmatch(r"Paragraf\s+(\d+|\w+)", line, re.IGNORECASE):
            return "\n".join(lines[:idx]).strip()
    return content.strip()


def estimate_token_count(text: str) -> int:
    return max(1, int(len(re.findall(r"\S+", text)) * 1.35))


def build_display_text(metadata: Dict, pasal_id: str, content: str, ctx: SectionContext) -> str:
    header = f"{metadata['regulation_type']} No. {metadata['nomor']} Tahun {metadata['publication_year']}"
    if metadata.get("tentang") and metadata["tentang"] != "Unknown":
        header += f" tentang {metadata['tentang']}"
    parts = [header]
    if ctx.bab:
        parts.append(f"{ctx.bab}" + (f" - {ctx.bab_title}" if ctx.bab_title else ""))
    if ctx.bagian:
        parts.append(f"{ctx.bagian}" + (f" - {ctx.bagian_title}" if ctx.bagian_title else ""))
    if ctx.paragraf:
        parts.append(f"{ctx.paragraf}" + (f" - {ctx.paragraf_title}" if ctx.paragraf_title else ""))
    parts.extend([pasal_id, content])
    return "\n".join(parts)


def build_citation_text(metadata: Dict, pasal_id: str) -> str:
    reg = f"{metadata['regulation_type']} No. {metadata['nomor']} Tahun {metadata['publication_year']}"
    return f"{reg}, {pasal_id}"


def build_chunk_metadata(metadata: Dict, ctx: SectionContext, pasal_id: str, kind: str, chunk_index: int, token_count: int) -> Dict:
    meta = {
        **metadata,
        "pasal_id": pasal_id,
        "chunk_kind": kind,
        "chunk_index": chunk_index,
        "token_count_estimate": token_count,
        "bab": ctx.bab,
        "bab_title": ctx.bab_title,
        "bagian": ctx.bagian,
        "bagian_title": ctx.bagian_title,
        "paragraf": ctx.paragraf,
        "paragraf_title": ctx.paragraf_title,
    }
    return {key: ("" if value is None else value) for key, value in meta.items()}


def structural_chunking_per_pasal(text: str, metadata: Dict) -> List[Dict]:
    text = strip_after_appendix(strip_after_explanation(strip_before_body(text)))
    section_index = build_section_index(text)
    chunks = []
    for pasal_pos, pasal_id, raw_content in split_pasals(text):
        ctx = context_at(section_index, pasal_pos)
        content = re.sub(
            r"^\s*Dalam\s+(?:Peraturan|Undang-Undang|Keputusan|Instruksi|Peraturan\s+\w+)"
            r"(?:\s+\w+){0,8}\s+ini\s+yang\s+dimaksud\s+dengan\s*:\s*",
            "",
            raw_content,
            flags=re.IGNORECASE | re.DOTALL,
        ).strip()
        if not content:
            continue
        content = strip_trailing_section_headers(content)
        if not content:
            continue
        content = re.sub(r"(?<!\n)\n(?!\n)", " ", content)
        content = re.sub(r"\n{2,}", "\n", content)
        content = compact_spaces(content)
        chunk_index = len(chunks) + 1
        token_count = estimate_token_count(content)
        meta = build_chunk_metadata(metadata, ctx, pasal_id, "pasal", chunk_index, token_count)
        display_text = build_display_text(metadata, pasal_id, content, ctx)
        citation_text = build_citation_text(metadata, pasal_id)
        chunk_id = hashlib.sha1(f"{metadata['source_file']}::{pasal_id}".encode("utf-8")).hexdigest()
        chunks.append({
            "id": chunk_id,
            "text": content,
            "display_text": display_text,
            "embedding_text": display_text,
            "citation_text": citation_text,
            "metadata": meta,
        })
    return chunks


def iter_pdf_files(pdf_directory: Path) -> Iterable[Path]:
    for root, _, files in os.walk(pdf_directory):
        for file in sorted(files):
            if file.lower().endswith(".pdf"):
                yield Path(root) / file


def unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    counter = 2
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def move_or_copy_failed_pdf(pdf_path: Path, ocr_dir: Path, reason: str, copy_failed: bool, move_failed: bool) -> str:
    if not move_failed and not copy_failed:
        return ""
    ocr_dir.mkdir(parents=True, exist_ok=True)
    destination = unique_destination(ocr_dir / pdf_path.name)
    action = "copy" if copy_failed else "move"
    if copy_failed:
        shutil.copy2(pdf_path, destination)
    else:
        shutil.move(str(pdf_path), str(destination))
    manifest_path = ocr_dir / "failed_extraction_manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps({"source": str(pdf_path), "destination": str(destination), "reason": reason, "action": action}, ensure_ascii=False) + "\n")
    return str(destination)


def save_raw_text(raw_text_dir: Optional[Path], pdf_path: Path, text: str) -> None:
    if raw_text_dir is None:
        return
    raw_text_dir.mkdir(parents=True, exist_ok=True)
    (raw_text_dir / f"{pdf_path.stem}.txt").write_text(text, encoding="utf-8")


def process_pdf(filepath: Path, args: argparse.Namespace) -> Tuple[List[Dict], Optional[Dict]]:
    print(f"Memproses: {filepath.name}", flush=True)
    try:
        text, extraction_info = extract_and_clean_text(filepath, args.extractor)
    except Exception as exc:
        reason = f"ekstraksi gagal: {exc}"
        moved_to = move_or_copy_failed_pdf(filepath, args.ocr_dir, reason, args.copy_failed, args.move_failed)
        print(f"  -> gagal, {reason}")
        return [], {"source_file": filepath.name, "status": "failed", "reason": reason, "moved_to": moved_to}

    usable, reason, quality = is_extraction_usable(
        text,
        args.min_chars,
        args.min_pasal_count,
        args.max_noise_ratio,
        args.max_ocr_artifact_count,
    )
    if not usable:
        moved_to = move_or_copy_failed_pdf(filepath, args.ocr_dir, reason, args.copy_failed, args.move_failed)
        print(f"  -> butuh OCR ({reason})")
        return [], {"source_file": filepath.name, "status": "needs_ocr", "reason": reason, "quality": quality, "moved_to": moved_to}

    save_raw_text(args.raw_text_dir, filepath, text)
    metadata = extract_metadata(filepath, text)
    metadata.update({
        "extraction_page_count": extraction_info["page_count"],
        "extraction_counts": json.dumps(extraction_info["extractor_counts"], ensure_ascii=False),
        "extraction_char_count": quality["char_count"],
        "extraction_word_count": quality["word_count"],
        "extraction_pasal_count": quality["pasal_count"],
        "extraction_noise_ratio": quality["noise_ratio"],
        "extraction_ocr_artifact_count": quality["ocr_artifact_count"],
    })
    chunks = structural_chunking_per_pasal(text, metadata)
    if not chunks:
        reason = "tidak ada chunk Pasal yang terbentuk"
        moved_to = move_or_copy_failed_pdf(filepath, args.ocr_dir, reason, args.copy_failed, args.move_failed)
        print(f"  -> butuh OCR ({reason})")
        return [], {"source_file": filepath.name, "status": "needs_ocr", "reason": reason, "quality": quality, "moved_to": moved_to}

    print(
        f"  -> {metadata['regulation_type']} No {metadata['nomor']} Th {metadata['publication_year']} | "
        f"{len(chunks)} Pasal | {metadata['extraction_counts']}"
    )
    return chunks, None


def process_corpus(pdf_directory: Path, args: argparse.Namespace) -> Tuple[List[Dict], List[Dict]]:
    all_chunks = []
    failures = []
    for index, filepath in enumerate(iter_pdf_files(pdf_directory), start=1):
        if args.limit is not None and index > args.limit:
            break
        chunks, failure = process_pdf(filepath, args)
        all_chunks.extend(chunks)
        if failure:
            failures.append(failure)
    return all_chunks, failures


def save_json(data, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess ringan PDF regulasi menjadi JSON chunk per Pasal.")
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR, help="Folder berisi PDF.")
    parser.add_argument("--pdf", type=Path, help="Proses satu PDF saja.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path output JSON.")
    parser.add_argument("--ocr-dir", type=Path, default=DEFAULT_OCR_DIR, help="Folder PDF yang butuh OCR.")
    parser.add_argument("--raw-text-dir", type=Path, default=DEFAULT_RAW_TEXT_DIR, help="Folder simpan raw text.")
    parser.add_argument("--no-save-raw-text", dest="raw_text_dir", action="store_const", const=None)
    parser.add_argument("--extractor", choices=["auto", "pymupdf", "pdfplumber"], default="auto", help="Extractor teks.")
    parser.add_argument("--limit", type=int, help="Batasi jumlah PDF untuk preview.")
    parser.add_argument("--min-chars", type=int, default=MIN_EXTRACTED_CHARS)
    parser.add_argument("--min-pasal-count", type=int, default=MIN_PASAL_COUNT)
    parser.add_argument("--max-noise-ratio", type=float, default=MAX_NOISE_RATIO)
    parser.add_argument("--max-ocr-artifact-count", type=int, default=MAX_OCR_ARTIFACT_COUNT)
    parser.add_argument("--copy-failed", action="store_true", help="Salin PDF gagal ke OCR dir, bukan dipindah.")
    parser.add_argument("--no-move-failed", dest="move_failed", action="store_false", help="Jangan pindahkan PDF gagal.")
    parser.set_defaults(move_failed=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.copy_failed:
        args.move_failed = False

    if args.pdf:
        chunks, failure = process_pdf(args.pdf, args)
        failures = [failure] if failure else []
    else:
        chunks, failures = process_corpus(args.pdf_dir, args)

    save_json(chunks, args.output)
    if failures:
        failure_output = args.output.with_suffix(".failures.json")
        save_json(failures, failure_output)
        print(f"Failure report: {failure_output}")
    print(f"Total chunk: {len(chunks)}")
    print(f"Disimpan ke: {args.output}")
    if chunks:
        print("Contoh chunk pertama:")
        print(json.dumps(chunks[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
