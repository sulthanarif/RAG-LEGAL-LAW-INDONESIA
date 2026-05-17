"""
Preprocessing PDF regulasi Indonesia dengan Docling, chunk per Pasal.

Bedanya dengan praproses-final.py:
  - Ekstraksi utama memakai Docling, bukan PyMuPDF/Tesseract.
  - OCR Docling dimatikan secara default. File scan/ekstraksi buruk dipindahkan
    ke folder "data/pdf_butuh_ocr" supaya diproses OCR terpisah.
  - Chunk default benar-benar per Pasal. Pasal panjang tidak dipecah kecuali
    memakai flag --split-long-pasal.

Setup:
    pip install docling
    pip install transformers  # opsional, hanya untuk --split-long-pasal

Contoh:
    python praproses_docling_pasal.py --pdf-dir data/pdf --output data/processed_chunks_docling_pasal.json
    python praproses_docling_pasal.py --pdf path/ke/file.pdf --no-move-failed
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

TOKENIZER_NAME = "intfloat/multilingual-e5-base"
DEFAULT_PDF_DIR = Path("data") / "pdf"
DEFAULT_OUTPUT_PATH = Path("data") / "processed_chunks_docling_pasal.json"
DEFAULT_OCR_DIR = Path("data") / "pdf_butuh_ocr"
DEFAULT_RAW_TEXT_DIR = Path("data") / "docling_text"

CHUNK_SIZE_LIMIT = 500
CHUNK_OVERLAP = 60
INCLUDE_PREAMBLE = False

MIN_EXTRACTED_CHARS = 300
MAX_NOISE_RATIO = 0.12
MIN_PASAL_COUNT = 1

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

_TOKENIZER = None


@dataclass
class SectionContext:
    bab: Optional[str] = None
    bab_title: Optional[str] = None
    bagian: Optional[str] = None
    bagian_title: Optional[str] = None
    paragraf: Optional[str] = None
    paragraf_title: Optional[str] = None


def compact_spaces(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            from transformers import AutoTokenizer, logging as transformers_logging
        except ImportError as exc:
            raise RuntimeError(
                "Transformers belum terinstall. Install `transformers` atau jalankan tanpa --split-long-pasal."
            ) from exc
        transformers_logging.set_verbosity_error()
        _TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    return _TOKENIZER


def estimate_token_count(text: str) -> int:
    # Estimasi ringan supaya startup tidak perlu import transformers/HF tokenizer.
    words = re.findall(r"\S+", text)
    return max(1, int(len(words) * 1.35))


def normalize_ocr_number_token(token: str) -> str:
    return token.translate(str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "|": "1"}))


def normalize_legal_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
    text = re.sub(r"(?<=\d)[Oo](?=\d)|(?<=\d)[Il|](?=\d)", lambda m: normalize_ocr_number_token(m.group(0)), text)
    text = re.sub(r"\b(1|2)[0Oo][0-9OoIl|]{2}\b", lambda m: normalize_ocr_number_token(m.group(0)), text)
    text = re.sub(r"(?m)^\s*\((\d{1,2})[Il1]\)\s*", r"(\1) ", text)
    text = re.sub(r"(?m)^\s*\((\d{1,2})[Il1]\s+(?=[A-Z])", r"(\1) ", text)
    text = re.sub(r"(?m)^\s*\((\d{1,2})\s+(?=[A-Z])", r"(\1) ", text)
    text = re.sub(r"(?m)^(\s*)(\d)\s+(\d{1,2})\.\s+", _fix_split_numlist, text)
    text = re.sub(r"(?m)^(\s*)(\d{1,2})[Il]\.\s+", lambda m: f"{m.group(1)}{m.group(2)}1. ", text)
    text = re.sub(r"\bP\s*a\s*s\s*a\s*l\b", "Pasal", text, flags=re.IGNORECASE)
    text = re.sub(r"\bB\s*A\s*B\b", "BAB", text, flags=re.IGNORECASE)
    text = re.sub(r"\bT\s*E\s*N\s*T\s*A\s*N\s*G\b", "TENTANG", text, flags=re.IGNORECASE)
    text = re.sub(r"\bM\s*E\s*M\s*U\s*T\s*U\s*S\s*K\s*A\s*N\b", "MEMUTUSKAN", text, flags=re.IGNORECASE)
    return text


def _fix_split_numlist(match: re.Match) -> str:
    leading, d1, d2 = match.group(1), match.group(2), match.group(3)
    if d1 == d2:
        return f"{leading}{d2}. "
    return f"{leading}{d1}{d2}. "


def remove_markdown_noise(markdown: str) -> str:
    lines = []
    for raw in markdown.splitlines():
        line = raw.strip()
        if not line:
            lines.append("")
            continue
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"^\s*[-*]\s+(?=(BAB|Bagian|Paragraf|Pasal|\(\d+\)|[a-z]\.|\d+\.))", "", line, flags=re.IGNORECASE)
        line = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
        line = re.sub(r"__([^_]+)__", r"\1", line)
        line = re.sub(r"`([^`]+)`", r"\1", line)
        if re.fullmatch(r"[-=]{3,}", line):
            continue
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


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


def clean_docling_text(markdown: str) -> str:
    text = remove_markdown_noise(markdown)
    text = normalize_legal_text(text)
    text = remove_noise_lines(text)
    return text.strip()


def load_docling_converter(enable_ocr: bool = False):
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:
        raise RuntimeError(
            "Docling belum terinstall. Jalankan: pip install docling"
        ) from exc

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = enable_ocr
    pipeline_options.do_table_structure = True

    return DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)},
    )


def extract_text_docling(pdf_path: Path, converter) -> Tuple[str, Dict]:
    result = converter.convert(pdf_path)
    document = getattr(result, "document", None)
    if document is None:
        status = str(getattr(result, "status", "unknown"))
        raise RuntimeError(f"Docling tidak menghasilkan document. Status: {status}")

    markdown = document.export_to_markdown()
    text = clean_docling_text(markdown)
    info = {
        "docling_status": str(getattr(result, "status", "")),
        "raw_markdown_chars": len(markdown),
        "clean_text_chars": len(text),
    }
    return text, info


def extraction_quality(text: str) -> Dict:
    words = re.findall(r"\S+", text)
    pasal_count = len(re.findall(r"(?m)^\s*Pasal\s+(?:\d+[A-Z]?|[IVXLCDM]+)\b", text, re.IGNORECASE))
    replacement_chars = text.count("\ufffd")
    suspicious = re.findall(r"[^\w\s.,;:!?()\[\]{}'\"/\-+%=<>@#&*|\\]", text, flags=re.UNICODE)
    non_latin_words = [w for w in words if len(w) > 3 and not re.search(r"[A-Za-z]", w)]
    noise_ratio = (replacement_chars + len(suspicious) + len(non_latin_words)) / max(len(words), 1)
    return {
        "char_count": len(text),
        "word_count": len(words),
        "pasal_count": pasal_count,
        "noise_ratio": round(noise_ratio, 4),
    }


def is_extraction_usable(text: str, min_chars: int, min_pasal_count: int, max_noise_ratio: float) -> Tuple[bool, str, Dict]:
    quality = extraction_quality(text)
    if quality["char_count"] < min_chars:
        return False, f"teks terlalu pendek ({quality['char_count']} char)", quality
    if quality["noise_ratio"] > max_noise_ratio:
        return False, f"noise terlalu tinggi ({quality['noise_ratio']})", quality
    if quality["pasal_count"] < min_pasal_count:
        return False, f"Pasal terdeteksi kurang dari {min_pasal_count}", quality
    return True, "ok", quality


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
    for pattern, label in REGULATION_PATTERNS:
        if re.search(pattern, filename_text, re.IGNORECASE):
            return label
    for pattern, label in REGULATION_PATTERNS:
        if re.search(pattern, head_text, re.IGNORECASE):
            return label
    return "Unknown"


def extract_nomor_year_from_filename(filename: str) -> Tuple[Optional[str], Optional[int]]:
    raw_name = Path(filename).stem
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

    name = normalize_ocr_number_token(raw_name)
    complex_match = re.search(r"\b(?P<no>[A-Za-z0-9.-]+(?:/[A-Za-z0-9.-]+){1,3}/(?P<year>19\d{2}|20\d{2}))\b", name)
    if complex_match:
        return complex_match.group("no"), int(complex_match.group("year"))
    for pattern in [
        r"(?P<year>19\d{2}|20\d{2})\D*(?P<no>\d{1,4})\b",
        r"\b(?P<no>\d{1,4})\D*(?:TAHUN|TH)\D*(?P<year>19\d{2}|20\d{2})\b",
        r"\bNO(?:MOR)?\D*(?P<no>\d{1,4})\D*(?P<year>19\d{2}|20\d{2})\b",
    ]:
        match = re.search(pattern, name, re.IGNORECASE)
        if match:
            return str(int(match.group("no"))), int(match.group("year"))
    return None, None


def extract_tentang(head_text: str) -> Optional[str]:
    normalized = re.sub(r"\s+", " ", head_text)
    candidates = []
    for pattern in [
        r"Menetap\w*\s*:\s*.*?\bTENTANG\s+(?P<about>.*?)(?=\bBAB\s+[IVXLCDM]+\b|\bPasal\s+1\b|$)",
        r"\bTENTANG\s+(?P<about>.*?)(?=\bDENGAN\s+RAHMAT\b|\bPRESIDEN\s+REPUBLIK\b|\bMenimbang\b|\bMengingat\b|\bMEMUTUSKAN\b|\bBAB\s+[IVXLCDM]+\b|\bPasal\s+1\b)",
    ]:
        for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
            about = compact_spaces(match.group("about"))
            about = re.sub(r"^(?:PERATURAN|UNDANG-UNDANG|KEPUTUSAN|INSTRUKSI).*?\bTENTANG\s+", "", about, flags=re.IGNORECASE)
            about = about.strip(" .;:")
            if 5 <= len(about) <= 240:
                candidates.append(about)
    return max(candidates, key=len).upper() if candidates else None


def extract_metadata(filepath: Path, text_content: str) -> Dict:
    filename = filepath.name
    head_text = re.sub(r"\s+", " ", normalize_legal_text(text_content[:10000]))
    regulation_type = detect_regulation_type(filename, head_text)
    nomor, year = None, None

    complex_match = re.search(
        r"\b(?:NOMOR|NO\.?)\s*:?\s*(?P<no>[A-Za-z0-9.-]+(?:/[A-Za-z0-9.-]+){1,3}/(?P<year>19\d{2}|20\d{2}))\b",
        head_text,
        re.IGNORECASE,
    )
    if complex_match:
        nomor = normalize_ocr_number_token(complex_match.group("no")).strip("./- ")
        year = int(normalize_ocr_number_token(complex_match.group("year")))
    else:
        title_match = re.search(
            r"\b(?:NOMOR|NO\.?)\s+(?P<no>[0-9A-Za-z./-]+)\s+(?:TAHUN|TH\.?)\s+(?P<year>19\d{2}|20\d{2})\b",
            head_text,
            re.IGNORECASE,
        )
        if title_match:
            nomor = normalize_ocr_number_token(title_match.group("no")).strip("./- ")
            year = int(normalize_ocr_number_token(title_match.group("year")))
        else:
            nomor, year = extract_nomor_year_from_filename(filename)

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
        "extractor": "docling",
        "chunk_level": "pasal",
    }


def canonical_section_title(lines: List[str], idx: int) -> Optional[str]:
    for next_line in lines[idx + 1 : idx + 4]:
        candidate = compact_spaces(next_line)
        if candidate and not re.match(r"^(BAB|Bagian|Paragraf|Pasal)\b", candidate, re.IGNORECASE):
            return candidate.title()
    return None


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
    if INCLUDE_PREAMBLE:
        return text
    marker = re.search(r"(?:^|\n)(BAB\s+I\b|Pasal\s+1\b|PERTAMA\b|KESATU\b)", text, re.IGNORECASE)
    return text[marker.start():].strip() if marker else text


def strip_after_explanation(text: str) -> str:
    return re.split(r"\bPENJELASAN\s+ATAS\b", text, maxsplit=1, flags=re.IGNORECASE)[0].strip()


def split_pasals(text: str) -> Iterable[Tuple[int, str, str]]:
    pasal_pattern = r"(?m)^\s*(Pasal\s+(?:\d+[A-Z]?|[IVXLCDM]+))[\s.:]*$"
    matches = list(re.finditer(pasal_pattern, text, re.IGNORECASE))

    if not matches:
        pasal_pattern = r"(?m)^\s*(Pasal\s+(?:\d+[A-Z]?|[IVXLCDM]+))[\s.:]*"
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
        pasal_id = compact_spaces(match.group(1)).title()
        yield start, pasal_id, text[match.end():end].strip()


def build_display_text(metadata: Dict, pasal_id: str, content: str, ctx: SectionContext) -> str:
    header = f"{metadata['regulation_type']} No. {metadata['nomor']} Tahun {metadata['publication_year']}"
    parts = [header]
    if ctx.bab:
        parts.append(f"{ctx.bab}" + (f" - {ctx.bab_title}" if ctx.bab_title else ""))
    if ctx.bagian:
        parts.append(f"{ctx.bagian}" + (f" - {ctx.bagian_title}" if ctx.bagian_title else ""))
    if ctx.paragraf:
        parts.append(f"{ctx.paragraf}" + (f" - {ctx.paragraf_title}" if ctx.paragraf_title else ""))
    parts.append(pasal_id)
    parts.append(content)
    return "\n".join(parts)


def build_citation_text(metadata: Dict, pasal_id: str) -> str:
    reg = f"{metadata['regulation_type']} No. {metadata['nomor']} Tahun {metadata['publication_year']}"
    return f"{reg}, {pasal_id}"


def build_chunk_metadata(metadata: Dict, ctx: SectionContext, pasal_id: str, kind: str, chunk_index: int) -> Dict:
    clean_meta = {
        **metadata,
        "pasal_id": pasal_id,
        "chunk_kind": kind,
        "chunk_index": chunk_index,
        "bab": ctx.bab or "",
        "bab_title": ctx.bab_title or "",
        "bagian": ctx.bagian or "",
        "bagian_title": ctx.bagian_title or "",
        "paragraf": ctx.paragraf or "",
        "paragraf_title": ctx.paragraf_title or "",
    }
    return {key: ("" if val is None else val) for key, val in clean_meta.items()}


def split_text_by_tokens(text: str, max_tokens: int = CHUNK_SIZE_LIMIT, overlap: int = CHUNK_OVERLAP) -> List[str]:
    tokenizer = get_tokenizer()
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return [text]
    chunks = []
    stride = max_tokens - overlap
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunks.append(tokenizer.decode(tokens[start:end], skip_special_tokens=True).strip())
        if end == len(tokens):
            break
        start += stride
    return [chunk for chunk in chunks if chunk]


def structural_chunking_per_pasal(text: str, metadata: Dict, split_long_pasal: bool = False) -> List[Dict]:
    text = strip_after_explanation(text)
    text = strip_before_body(text)
    section_index = build_section_index(text)
    chunks: List[Dict] = []

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

        content = re.sub(r"(?<!\n)\n(?!\n)", " ", content)
        content = re.sub(r"\n{2,}", "\n", content)
        content = compact_spaces(content)
        token_count = estimate_token_count(content)

        parts = split_text_by_tokens(content) if split_long_pasal and token_count > CHUNK_SIZE_LIMIT else [content]
        for part_idx, part in enumerate(parts, start=1):
            ci = len(chunks) + 1
            kind = "pasal_split" if len(parts) > 1 else "pasal"
            meta = build_chunk_metadata(metadata, ctx, pasal_id, kind, ci)
            if len(parts) > 1:
                meta["split_part"] = part_idx
            meta["token_count"] = estimate_token_count(part)
            display_text = build_display_text(metadata, pasal_id, part, ctx)
            citation_text = build_citation_text(metadata, pasal_id)
            chunk_id = hashlib.sha1(f"{metadata['source_file']}::{pasal_id}::{part_idx}".encode("utf-8")).hexdigest()
            chunks.append(
                {
                    "id": chunk_id,
                    "text": part,
                    "display_text": display_text,
                    "embedding_text": display_text,
                    "citation_text": citation_text,
                    "metadata": meta,
                }
            )
    return chunks


def iter_pdf_files(pdf_directory: Path) -> Iterable[Path]:
    for root, _, files in os.walk(pdf_directory):
        for file in sorted(files):
            if file.lower().endswith(".pdf"):
                yield Path(root) / file


def unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    counter = 2
    while True:
        candidate = path.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def move_or_copy_failed_pdf(pdf_path: Path, ocr_dir: Path, reason: str, copy_failed: bool, move_failed: bool) -> Optional[Path]:
    if not move_failed and not copy_failed:
        return None
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
    return destination


def save_raw_text(raw_text_dir: Optional[Path], pdf_path: Path, text: str) -> None:
    if raw_text_dir is None:
        return
    raw_text_dir.mkdir(parents=True, exist_ok=True)
    output = raw_text_dir / f"{pdf_path.stem}.txt"
    output.write_text(text, encoding="utf-8")


def process_pdf(
    filepath: Path,
    converter,
    args: argparse.Namespace,
) -> Tuple[List[Dict], Optional[Dict]]:
    print(f"Memproses: {filepath.name}")
    try:
        text, docling_info = extract_text_docling(filepath, converter)
    except Exception as exc:
        reason = f"Docling gagal: {exc}"
        moved_to = move_or_copy_failed_pdf(filepath, args.ocr_dir, reason, args.copy_failed, args.move_failed)
        print(f"  -> gagal ekstraksi, {'dipindah/disalin ke ' + str(moved_to) if moved_to else 'tidak dipindah'}")
        return [], {"source_file": filepath.name, "status": "failed", "reason": reason, "moved_to": str(moved_to) if moved_to else ""}

    usable, reason, quality = is_extraction_usable(text, args.min_chars, args.min_pasal_count, args.max_noise_ratio)
    if not usable:
        moved_to = move_or_copy_failed_pdf(filepath, args.ocr_dir, reason, args.copy_failed, args.move_failed)
        print(f"  -> ekstraksi buruk ({reason}), {'dipindah/disalin ke ' + str(moved_to) if moved_to else 'tidak dipindah'}")
        return [], {
            "source_file": filepath.name,
            "status": "needs_ocr",
            "reason": reason,
            "quality": quality,
            "docling": docling_info,
            "moved_to": str(moved_to) if moved_to else "",
        }

    save_raw_text(args.raw_text_dir, filepath, text)
    metadata = extract_metadata(filepath, text)
    metadata.update(
        {
            "extraction_char_count": quality["char_count"],
            "extraction_word_count": quality["word_count"],
            "extraction_pasal_count": quality["pasal_count"],
            "extraction_noise_ratio": quality["noise_ratio"],
            **docling_info,
        }
    )
    chunks = structural_chunking_per_pasal(text, metadata, split_long_pasal=args.split_long_pasal)
    if not chunks:
        reason = "tidak ada chunk Pasal yang terbentuk"
        moved_to = move_or_copy_failed_pdf(filepath, args.ocr_dir, reason, args.copy_failed, args.move_failed)
        print(f"  -> {reason}, {'dipindah/disalin ke ' + str(moved_to) if moved_to else 'tidak dipindah'}")
        return [], {"source_file": filepath.name, "status": "needs_ocr", "reason": reason, "quality": quality, "moved_to": str(moved_to) if moved_to else ""}

    print(
        f"  -> {metadata['regulation_type']} No {metadata['nomor']} Th {metadata['publication_year']} | "
        f"{len(chunks)} chunks | Pasal terdeteksi {quality['pasal_count']}"
    )
    return chunks, None


def process_corpus(pdf_directory: Path, converter, args: argparse.Namespace) -> Tuple[List[Dict], List[Dict]]:
    all_chunks: List[Dict] = []
    failures: List[Dict] = []
    for index, filepath in enumerate(iter_pdf_files(pdf_directory), start=1):
        if args.limit is not None and index > args.limit:
            break
        chunks, failure = process_pdf(filepath, converter, args)
        all_chunks.extend(chunks)
        if failure:
            failures.append(failure)
    return all_chunks, failures


def save_json(data, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess PDF regulasi dengan Docling menjadi chunk JSON per Pasal.")
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR, help="Folder berisi PDF.")
    parser.add_argument("--pdf", type=Path, help="Proses satu PDF saja.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path output JSON chunks.")
    parser.add_argument("--ocr-dir", type=Path, default=DEFAULT_OCR_DIR, help="Folder tujuan PDF yang butuh OCR.")
    parser.add_argument("--raw-text-dir", type=Path, default=DEFAULT_RAW_TEXT_DIR, help="Folder simpan teks hasil Docling. Pakai --no-save-raw-text untuk mematikan.")
    parser.add_argument("--no-save-raw-text", dest="raw_text_dir", action="store_const", const=None)
    parser.add_argument("--limit", type=int, help="Batasi jumlah PDF untuk preview.")
    parser.add_argument("--enable-docling-ocr", action="store_true", help="Aktifkan OCR bawaan Docling. Default mati.")
    parser.add_argument("--split-long-pasal", action="store_true", help="Pecah Pasal yang melebihi batas token. Default mati.")
    parser.add_argument("--min-chars", type=int, default=MIN_EXTRACTED_CHARS, help="Minimal karakter hasil ekstraksi agar dianggap layak.")
    parser.add_argument("--min-pasal-count", type=int, default=MIN_PASAL_COUNT, help="Minimal jumlah Pasal terdeteksi agar dianggap layak.")
    parser.add_argument("--max-noise-ratio", type=float, default=MAX_NOISE_RATIO, help="Maksimal rasio noise ekstraksi.")
    parser.add_argument("--copy-failed", action="store_true", help="Salin PDF gagal ke OCR dir, bukan dipindah.")
    parser.add_argument("--no-move-failed", dest="move_failed", action="store_false", help="Jangan pindahkan PDF gagal.")
    parser.set_defaults(move_failed=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    converter = load_docling_converter(enable_ocr=args.enable_docling_ocr)

    if args.copy_failed:
        args.move_failed = False

    if args.pdf:
        chunks, failure = process_pdf(args.pdf, converter, args)
        failures = [failure] if failure else []
    else:
        chunks, failures = process_corpus(args.pdf_dir, converter, args)

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
