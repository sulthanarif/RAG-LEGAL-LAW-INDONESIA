"""
pipeline_legal_rag_indonesia.py
================================
Pipeline lengkap: PDF regulasi Indonesia → JSON chunks untuk RAG.

Fitur:
  - Ekstraksi teks native PyMuPDF
  - Advanced OCR (deskew, shadow removal, adaptive threshold) via Tesseract
  - Normalisasi teks legal + fix artefak OCR
  - Deteksi metadata otomatis (tipe regulasi, nomor, tahun, judul)
  - Structural chunking per Pasal
  - Post-correction OCR via Qwen2.5-7B-Instruct lokal (HuggingFace)

Setup:
    pip install pymupdf pillow pytesseract opencv-python numpy transformers \
                accelerate bitsandbytes huggingface_hub

    # Tesseract (Windows)
    # Download installer: https://github.com/UB-Mannheim/tesseract/wiki
    # Pastikan language pack 'ind' ikut terinstall

Jalankan:
    # Proses semua PDF di folder data/pdf/
    python pipeline_legal_rag_indonesia.py

    # Proses satu PDF saja
    python pipeline_legal_rag_indonesia.py --pdf path/ke/file.pdf

    # Tanpa koreksi OCR (lebih cepat)
    python pipeline_legal_rag_indonesia.py --no-ocr-correction

    # Dengan koreksi OCR (default, butuh GPU + ~15GB download model pertama kali)
    python pipeline_legal_rag_indonesia.py --ocr-correction
"""

import argparse
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import fitz
import numpy as np
import pytesseract
from PIL import Image
from transformers import logging as transformers_logging

transformers_logging.set_verbosity_error()

# ─────────────────────────────────────────────────────────────────────────────
# KONFIGURASI GLOBAL
# ─────────────────────────────────────────────────────────────────────────────
TOKENIZER_NAME      = "intfloat/multilingual-e5-base"
CHUNK_SIZE_LIMIT    = 500
CHUNK_OVERLAP       = 60
INCLUDE_PREAMBLE    = False
DEFAULT_PDF_DIR     = os.path.join("data", "pdf")
DEFAULT_OUTPUT_PATH = os.path.join("data", "processed_chunks.json")

# OCR correction via Qwen2.5-7B
OCR_CORRECTION_MODEL  = "Qwen/Qwen2.5-7B-Instruct"
OCR_CORRECTION_4BIT   = True       # bitsandbytes 4-bit, ~4.3GB VRAM
OCR_CORRECTION_BATCH  = 3          # chunk per inferensi; turunkan ke 1 jika OOM
OCR_BAD_THRESHOLD     = 0.08       # rasio noise; di atas ini → kirim ke model

# ─────────────────────────────────────────────────────────────────────────────
# TOKENIZER
# ─────────────────────────────────────────────────────────────────────────────
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)


# ─────────────────────────────────────────────────────────────────────────────
# TESSERACT SETUP
# ─────────────────────────────────────────────────────────────────────────────
def configure_tesseract() -> bool:
    cmd = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
    if not cmd:
        win_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        for path in win_paths:
            if os.path.exists(path):
                cmd = path
                break
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
        print(f"✅ Tesseract: {cmd}")
    else:
        print("❌ Tesseract tidak ditemukan.")
        return False
    try:
        version = pytesseract.get_tesseract_version()
        print(f"📦 Tesseract version: {version}")
        return True
    except Exception as e:
        print(f"❌ Gagal load Tesseract: {e}")
        return False


TESSERACT_READY = configure_tesseract()


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASS
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SectionContext:
    bab: Optional[str] = None
    bab_title: Optional[str] = None
    bagian: Optional[str] = None
    bagian_title: Optional[str] = None
    paragraf: Optional[str] = None
    paragraf_title: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# TEXT UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def compact_spaces(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def normalize_ocr_number_token(token: str) -> str:
    return token.translate(str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1", "|": "1"}))


def _fix_split_numlist(m: re.Match) -> str:
    """
    Fix OCR artefak angka terpisah di awal baris.
    '2 2. Teks' → '2. Teks'   (digit pertama duplikat OCR)
    '1 0. Teks' → '10. Teks'  (angka 2 digit yang terbelah)
    """
    leading, d1, d2 = m.group(1), m.group(2), m.group(3)
    if d1 == d2:
        return f"{leading}{d2}. "
    return f"{leading}{d1}{d2}. "


def normalize_legal_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    # Fix OCR confuse O/0 dan I/l/1 di dalam angka
    text = re.sub(
        r"(?<=\d)[Oo](?=\d)|(?<=\d)[Il|](?=\d)",
        lambda m: normalize_ocr_number_token(m.group(0)),
        text,
    )
    text = re.sub(
        r"\b(1|2)[0Oo][0-9OoIl|]{2}\b",
        lambda m: normalize_ocr_number_token(m.group(0)),
        text,
    )
    # Fix nomor ayat yang OCR-nya pisah: '(1l)' atau '(1 '
    text = re.sub(r"(?m)^\s*\((\d{1,2})[Il1]\)\s*", r"(\1) ", text)
    text = re.sub(r"(?m)^\s*\((\d{1,2})[Il1]\s+(?=[A-Z])", r"(\1) ", text)
    text = re.sub(r"(?m)^\s*\((\d{1,2})\s+(?=[A-Z])", r"(\1) ", text)
    # Fix angka nomor urut terbelah: '2 2.' → '2.' atau '1 0.' → '10.'
    text = re.sub(r"(?m)^(\s*)(\d)\s+(\d{1,2})\.\s+", _fix_split_numlist, text)
    text = re.sub(
        r"(?m)^(\s*)(\d{1,2})[Il]\.\s+",
        lambda m: f"{m.group(1)}{m.group(2)}1. ",
        text,
    )
    # Fix kata kunci legal yang hurufnya tersebar oleh OCR
    text = re.sub(r"\bP\s*a\s*s\s*a\s*l\b",            "Pasal",      text, flags=re.IGNORECASE)
    text = re.sub(r"\bB\s*A\s*B\b",                     "BAB",        text, flags=re.IGNORECASE)
    text = re.sub(r"\bM\s*e\s*n\s*i\s*m\s*b\s*a\s*n\s*g\b", "Menimbang", text, flags=re.IGNORECASE)
    text = re.sub(r"\bM\s*e\s*n\s*g\s*i\s*n\s*g\s*a\s*t\b", "Mengingat",  text, flags=re.IGNORECASE)
    text = re.sub(r"\bM\s*E\s*M\s*U\s*T\s*U\s*S\s*K\s*A\s*N\b", "MEMUTUSKAN", text, flags=re.IGNORECASE)
    text = re.sub(r"\bT\s*E\s*N\s*T\s*A\s*N\s*G\b",    "TENTANG",    text, flags=re.IGNORECASE)
    return text


def strip_table_garbage(text: str) -> str:
    lines = text.split("\n")
    clean = []
    for line in lines:
        s = line.strip()
        if not s:
            clean.append("")
            continue
        if s.count("|") >= 2 or s.count("─") >= 3 or re.match(r"^[\s\-─|═╬+]+$", s):
            continue
        if len(s) > 20 and sum(c.isdigit() for c in s) / len(s) > 0.45 and s.count(".") < 1:
            continue
        clean.append(line)
    return "\n".join(clean)


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
        if len(line) <= 18 and re.search(r"[<>{}=*']", line):
            continue
        kept.append(line)
    cleaned = "\n".join(kept)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


# ─────────────────────────────────────────────────────────────────────────────
# ADVANCED OCR PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def _deskew(image: np.ndarray) -> np.ndarray:
    """Koreksi kemiringan halaman scan."""
    coords = np.column_stack(np.where(image < 128))
    if len(coords) < 100:
        return image
    angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
    if angle < -45:
        angle = 90 + angle
    if abs(angle) < 0.3:
        return image
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _remove_shadow(gray: np.ndarray) -> np.ndarray:
    """Normalisasi pencahayaan tidak rata (shadow removal) via dilasi + divide."""
    dilated = cv2.dilate(gray, np.ones((21, 21), np.uint8))
    bg = cv2.GaussianBlur(dilated, (21, 21), 0)
    norm = cv2.divide(gray.astype(np.float32), bg.astype(np.float32) + 1e-6)
    return np.clip(norm * 255, 0, 255).astype(np.uint8)


def _binarize(gray: np.ndarray) -> np.ndarray:
    """Binarisasi adaptif + Otsu digabung (AND) → tahan terhadap ketidakrataan cahaya."""
    adapt = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=31, C=15
    )
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.bitwise_and(adapt, otsu)


def _denoise(binary: np.ndarray) -> np.ndarray:
    """Hapus noise titik kecil sambil jaga stroke teks."""
    kernel = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    return cv2.fastNlMeansDenoising(opened, None, h=10, templateWindowSize=7, searchWindowSize=21)


def preprocess_for_ocr(page: fitz.Page, zoom: float = 3.5) -> np.ndarray:
    """
    Full preprocessing pipeline untuk halaman scan:
    render → grayscale → shadow removal → deskew → binarize → denoise
    """
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    if pix.n == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    gray   = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray   = _remove_shadow(gray)
    gray   = _deskew(gray)
    binary = _binarize(gray)
    clean  = _denoise(binary)
    return clean


def ocr_with_fallback(img: np.ndarray) -> str:
    """
    Coba PSM 6 (blok seragam) dan PSM 4 (kolom campuran),
    pilih hasil dengan word count latin terbanyak.
    """
    configs = [
        r"--oem 3 --psm 6 -l ind+eng",
        r"--oem 3 --psm 4 -l ind+eng",
    ]
    pil_img = Image.fromarray(img)
    results = []
    for cfg in configs:
        try:
            txt = pytesseract.image_to_string(pil_img, config=cfg).strip()
            results.append(txt)
        except Exception:
            results.append("")

    def score(t: str) -> int:
        return sum(1 for w in t.split() if re.search(r"[a-zA-Z]", w))

    return max(results, key=score)


def extract_page_text(page: fitz.Page, use_ocr: bool, page_number: int) -> str:
    native_text  = page.get_text("text", sort=True).strip()
    char_count   = len(native_text)
    newline_ratio = native_text.count("\n") / max(char_count, 1)
    native_ok    = char_count >= 80 and newline_ratio <= 0.35

    if native_ok:
        return native_text

    if not use_ocr:
        print(f"    ⚠️  Halaman {page_number}: teks kurang ({char_count} char), OCR tidak tersedia.")
        return native_text

    print(f"    🔬 OCR Halaman {page_number}: native teks miskin ({char_count} char), jalankan OCR...")
    try:
        preprocessed = preprocess_for_ocr(page)
        result = ocr_with_fallback(preprocessed)
        print(f"    ✅ OCR selesai: {len(result)} char")
        return result
    except Exception as e:
        print(f"    ❌ OCR Error halaman {page_number}: {e}")
        return native_text


def extract_and_clean_text(pdf_path: str) -> str:
    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        print(f"Error membuka {pdf_path}: {exc}")
        return ""

    page_texts = []
    use_ocr = TESSERACT_READY
    for page_number, page in enumerate(doc, start=1):
        page_texts.append(extract_page_text(page, use_ocr, page_number))
    doc.close()

    text = "\n\n".join(page_texts)
    text = normalize_legal_text(text)
    text = remove_noise_lines(text)
    text = strip_table_garbage(text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# METADATA EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

# Urutan penting: paling spesifik di atas, fallback paling bawah
REGULATION_PATTERNS: List[Tuple[str, str]] = [
    (r"\bUNDANG[-\s]?UNDANG\b",                                                      "Undang-Undang"),
    (r"\bPERATURAN\s+PEMERINTAH\b",                                                  "Peraturan Pemerintah"),
    (r"\bPERATURAN\s+PRESIDEN\b|\bPERPRES\b",                                        "Peraturan Presiden"),
    (r"\bPERATURAN\s+MENTERI\s+KETENAGAKERJAAN\b|\bPERMENAKER\b|\bPMNAKER\b",       "Peraturan Menteri Ketenagakerjaan"),
    (r"\bPERATURAN\s+MENTERI\b|\bPERMEN\b",                                          "Peraturan Menteri"),
    (r"\bKEPUTUSAN\s+MENTERI\b|\bKEPMEN\b",                                          "Keputusan Menteri"),
    (r"\bPUTUSAN\b",                                                                  "Putusan"),
    # Fallback singkatan — paling bawah supaya tidak false-positive
    (r"\bPP\b",                                                                       "Peraturan Pemerintah"),
    (r"\bUU\b",                                                                       "Undang-Undang"),
]


def detect_regulation_type(filename: str, head_text: str) -> str:
    # 1. Coba dari judul formal dengan pola NOMOR (paling akurat)
    title_match = re.search(
        r"\b(UNDANG[-\s]?UNDANG|PERATURAN\s+PEMERINTAH|PERATURAN\s+PRESIDEN|"
        r"PERATURAN\s+MENTERI\s+KETENAGAKERJAAN|PERATURAN\s+MENTERI|KEPUTUSAN\s+MENTERI)"
        r"(?:\s+REPUBLIK\s+INDONESIA)?\s+(?:NOMOR|NO\.?)\b",
        head_text,
        re.IGNORECASE,
    )
    if title_match:
        for pattern, label in REGULATION_PATTERNS:
            if re.search(pattern, title_match.group(1), re.IGNORECASE):
                return label

    # 2. Scan 2000 karakter pertama head_text (skip singkatan pendek — terlalu berisik)
    head_snippet = head_text[:2000]
    for pattern, label in REGULATION_PATTERNS:
        if pattern in (r"\bPP\b", r"\bUU\b"):
            continue
        if re.search(pattern, head_snippet, re.IGNORECASE):
            return label

    # 3. Fallback ke nama file
    fn_upper  = filename.upper()
    name_only = os.path.splitext(fn_upper)[0]
    if re.search(r"\bUNDANG\b",           name_only): return "Undang-Undang"
    if re.search(r"\bPERPRES\b",          name_only): return "Peraturan Presiden"
    if re.search(r"\bPP\b(?:\.\d|\d)",    name_only): return "Peraturan Pemerintah"
    if re.search(r"PERMEN|PER\.|[-/]MEN[-/]", name_only): return "Peraturan Menteri"
    if re.search(r"KEPMEN|KEP\.",         name_only): return "Keputusan Menteri"

    filename_text = filename.replace("_", " ").replace("-", " ")
    for pattern, label in REGULATION_PATTERNS:
        if re.search(pattern, filename_text, re.IGNORECASE):
            return label

    # 4. Last resort: singkatan pendek dari head_text
    for pattern in (r"\bUU\b", r"\bPP\b"):
        label = next(lbl for pat, lbl in REGULATION_PATTERNS if pat == pattern)
        if re.search(pattern, head_snippet):
            return label

    return "Unknown"


def extract_nomor_year_from_text(header_text: str) -> Tuple[Optional[str], Optional[int]]:
    patterns = [
        # Format lawas: PER.12/MEN/X/2011
        (r"\b(?:NOMOR|NO\.?)\s*:?\s*"
         r"(?P<no>(?:PER|KEP)[\.-]?[A-Za-z0-9]+[-/][A-Za-z0-9.-]+"
         r"(?:[-/][A-Za-z0-9.-]+)*[-/](?P<year>19\d{2}|20\d{2}))\b"),
        # Format dengan garis miring
        (r"\b(?:NOMOR|NO\.?)\s*:?\s*"
         r"(?P<no>[A-Za-z0-9.-]+(?:[-/][A-Za-z0-9.-]+)+[-/](?P<year>19\d{2}|20\d{2}))\b"),
        # Format standar baru: Nomor 13 Tahun 2003
        (r"\b(?:NOMOR|NO\.?)\s*:?\s*(?P<no>[0-9A-Za-z./-]+)"
         r"\s+(?:TAHUN|TH\.?)\s+(?P<year>19\d{2}|20\d{2})\b"),
    ]
    for pat in patterns:
        m = re.search(pat, header_text, re.IGNORECASE)
        if m:
            return m.group("no").strip("./- "), int(m.group("year"))
    return None, None


def extract_nomor_year_from_filename(filename: str) -> Tuple[Optional[str], Optional[int]]:
    name       = os.path.splitext(filename)[0]
    normalized = normalize_ocr_number_token(name)

    complex_match = re.search(
        r"\b(?P<no>[A-Za-z0-9.-]+(?:[-/][A-Za-z0-9.-]+){1,3}[-/](?P<year>19\d{2}|20\d{2}))\b",
        normalized,
        re.IGNORECASE,
    )
    if complex_match:
        return complex_match.group("no"), int(complex_match.group("year"))

    for pat in [
        r"(?P<year>19\d{2}|20\d{2})\D*(?P<no>\d{1,4})\b",
        r"\b(?P<no>\d{1,4})\D*(?:TAHUN|TH)\D*(?P<year>19\d{2}|20\d{2})\b",
        r"\bNO(?:MOR)?\D*(?P<no>\d{1,4})\D*(?P<year>19\d{2}|20\d{2})\b",
    ]:
        m = re.search(pat, normalized, re.IGNORECASE)
        if m:
            return str(int(m.group("no"))), int(m.group("year"))
    return None, None


def extract_tentang(head_text: str) -> Optional[str]:
    normalized = re.sub(r"\s+", " ", head_text)
    candidates = []
    for pattern in [
        r"Menetap\w*\s*:\s*.*?\bTENTANG\s+(?P<about>.*?)(?=\bBAB\s+[IVXLCDM]+\b|\bPasal\s+1\b|$)",
        r"\bTENTANG\s+(?P<about>.*?)(?=\bDENGAN\s+RAHMAT\b|\bPRESIDEN\s+REPUBLIK\b"
        r"|\bMenimbang\b|\bMengingat\b|\bMEMUTUSKAN\b|\bBAB\s+[IVXLCDM]+\b)",
    ]:
        for match in re.finditer(pattern, normalized, flags=re.IGNORECASE | re.DOTALL):
            about = compact_spaces(match.group("about"))
            about = re.sub(
                r"^(?:PERATURAN|UNDANG-UNDANG).*?\bTENTANG\s+", "", about, flags=re.IGNORECASE
            )
            about = about.strip(" .;:")
            if 5 <= len(about) <= 220:
                candidates.append(about)
    return max(candidates, key=len).upper() if candidates else None


def extract_metadata(filepath: str, text_content: str) -> Dict:
    filename  = os.path.basename(filepath)
    head_text = re.sub(r"\s+", " ", normalize_legal_text(text_content[:8000]))

    # Potong bagian sebelum Menimbang/Mengingat/MEMUTUSKAN
    # Pattern cover OCR yang hasilkan spasi di dalam kata kunci
    split_pattern = (
        r"\b(?:Menimbang|Mengingat|MEMUTUSKAN"
        r"|M\s*e\s*n\s*i\s*m\s*b\s*a\s*n\s*g"
        r"|M\s*e\s*n\s*g\s*i\s*n\s*g\s*a\s*t)\b"
    )
    header_parts = re.split(split_pattern, head_text, maxsplit=1, flags=re.IGNORECASE)
    header_only  = header_parts[0].strip() if header_parts[0].strip() else head_text

    regulation_type = detect_regulation_type(filename, header_only)

    nomor, year = extract_nomor_year_from_text(header_only)
    if not nomor:
        nomor, year = extract_nomor_year_from_filename(filename)

    about = extract_tentang(head_text)
    return {
        "regulation_type": regulation_type,
        "nomor":            nomor or "Unknown",
        "tentang":          about or "Unknown",
        "year":             year or 0,
        "publication_year": year or 0,
        "active_status":    "Berlaku",
        "source_file":      filename,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────────────────────────────────────
def split_text_by_tokens(
    text: str,
    max_tokens: int = CHUNK_SIZE_LIMIT,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return [text]
    chunks = []
    start  = 0
    stride = max_tokens - overlap
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunks.append(tokenizer.decode(tokens[start:end], skip_special_tokens=True).strip())
        if end == len(tokens):
            break
        start += stride
    return [c for c in chunks if c]


def canonical_section_title(lines: List[str], idx: int) -> Optional[str]:
    for next_line in lines[idx + 1: idx + 4]:
        candidate = compact_spaces(next_line)
        if candidate and not re.match(r"^(BAB|Bagian|Paragraf|Pasal)\b", candidate, re.IGNORECASE):
            return candidate.title()
    return None


def build_section_index(text: str) -> List[Tuple[int, SectionContext]]:
    lines  = text.splitlines()
    index: List[Tuple[int, SectionContext]] = []
    offset = 0
    ctx    = SectionContext()
    for idx, line in enumerate(lines):
        clean = compact_spaces(line)
        if re.fullmatch(r"BAB\s+[IVXLCDM]+", clean, re.IGNORECASE):
            ctx = SectionContext(bab=clean.upper(), bab_title=canonical_section_title(lines, idx))
            index.append((offset, SectionContext(**ctx.__dict__)))
        elif re.fullmatch(r"Bagian\s+Ke\w+", clean, re.IGNORECASE):
            ctx.bagian       = clean.title()
            ctx.bagian_title = canonical_section_title(lines, idx)
            index.append((offset, SectionContext(**ctx.__dict__)))
        elif re.fullmatch(r"Paragraf\s+\d+", clean, re.IGNORECASE):
            ctx.paragraf       = clean.title()
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
    marker = re.search(
        r"(?:^|\n)(BAB\s+I\b|Pasal\s+1\b|PERTAMA\b|KESATU\b)", text, re.IGNORECASE
    )
    return text[marker.start():].strip() if marker else text


def strip_after_explanation(text: str) -> str:
    return re.split(r"\bPENJELASAN\s+ATAS\b", text, maxsplit=1, flags=re.IGNORECASE)[0].strip()


def split_pasals(text: str) -> Iterable[Tuple[int, str, str]]:
    pasal_pattern = r"(?m)^\s*(Pasal\s+(?:\d+[A-Z]?|[IVXLCDM]+))[\s.:]*"
    matches       = list(re.finditer(pasal_pattern, text, re.IGNORECASE))

    if not matches:
        diktum_pattern = (
            r"(?m)^\s*(PERTAMA|KESATU|KEDUA|KETIGA|KEEMPAT|KELIMA|KEENAM"
            r"|KETUJUH|KEDELAPAN|KESEMBILAN|KESEPULUH|KESEBELAS"
            r"|KEDUA\s*BELAS|KETIGA\s*BELAS)[\s.:]*"
        )
        matches = list(re.finditer(diktum_pattern, text, re.IGNORECASE))

    if not matches:
        if text.strip():
            yield 0, "Isi Dokumen", text.strip()
        return

    for idx, match in enumerate(matches):
        start = match.start()
        end   = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        yield start, match.group(1).strip().title(), text[match.end():end].strip()


def build_chunk_metadata(
    metadata: Dict, ctx: SectionContext, pasal_id: str, kind: str, chunk_index: int
) -> Dict:
    return {
        **metadata,
        "pasal_id":      pasal_id,
        "chunk_kind":    kind,
        "chunk_index":   chunk_index,
        "bab":           ctx.bab,
        "bab_title":     ctx.bab_title,
        "bagian":        ctx.bagian,
        "bagian_title":  ctx.bagian_title,
        "paragraf":      ctx.paragraf,
        "paragraf_title": ctx.paragraf_title,
    }


def _strip_definition_preamble(content: str) -> str:
    """Hapus kalimat pembuka Pasal 1 definisi dengan pattern luwes."""
    return re.sub(
        r"^\s*Dalam\s+(?:Peraturan|Undang-Undang|Keputusan|Peraturan\s+\w+)"
        r"(?:\s+\w+){0,6}\s+ini\s+yang\s+dimaksud\s+dengan\s*:\s*",
        "",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()


def structural_chunking(text: str, metadata: Dict) -> List[Dict]:
    text = strip_after_explanation(text)
    text = strip_before_body(text)
    section_index = build_section_index(text)
    chunks: List[Dict] = []

    for pasal_pos, pasal_id, raw_content in split_pasals(text):
        ctx     = context_at(section_index, pasal_pos)
        content = _strip_definition_preamble(raw_content)
        if not content:
            continue

        # Single newline → spasi, multi-newline → satu newline
        cleaned = re.sub(r"(?<!\n)\n(?!\n)", " ", content)
        cleaned = re.sub(r"\n{2,}", "\n", cleaned)
        cleaned = compact_spaces(cleaned)

        token_count = len(tokenizer.encode(cleaned, add_special_tokens=False))

        if token_count > CHUNK_SIZE_LIMIT:
            parts  = re.split(r"(?<=\.)\s+", cleaned)
            buffer = ""
            sub_idx = 1
            for part in parts:
                candidate = (buffer + " " + part).strip() if buffer else part
                if (
                    len(tokenizer.encode(candidate, add_special_tokens=False)) > CHUNK_SIZE_LIMIT
                    and buffer
                ):
                    ci   = len(chunks) + 1
                    meta = build_chunk_metadata(metadata, ctx, pasal_id, "pasal_split", ci)
                    meta["split_part"] = sub_idx
                    chunks.append({"text": buffer.strip(), "metadata": meta})
                    buffer  = part
                    sub_idx += 1
                else:
                    buffer = candidate
            if buffer.strip():
                ci   = len(chunks) + 1
                meta = build_chunk_metadata(metadata, ctx, pasal_id, "pasal_split", ci)
                meta["split_part"] = sub_idx
                chunks.append({"text": buffer.strip(), "metadata": meta})
        else:
            ci   = len(chunks) + 1
            meta = build_chunk_metadata(metadata, ctx, pasal_id, "pasal", ci)
            chunks.append({"text": cleaned, "metadata": meta})

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# OCR POST-CORRECTION via Qwen2.5-7B-Instruct (HuggingFace)
# ─────────────────────────────────────────────────────────────────────────────

_OCR_CORRECTION_SYSTEM = (
    "Kamu adalah sistem koreksi OCR untuk dokumen hukum Indonesia. "
    "Perbaiki karakter yang rusak akibat scan buruk. "
    "Jangan ubah makna, struktur, atau terminologi hukum. "
    "Jangan tambah atau hapus kalimat. "
    "Pertahankan nomor ayat dan format list. "
    "Output HANYA JSON array string, tanpa penjelasan, tanpa markdown."
)


def _ocr_noise_score(text: str) -> float:
    """
    Hitung rasio kata yang terindikasi OCR rusak:
      1. Ada karakter noise (·, ;, \\, ~, dsb.) di tengah kata.
      2. Konsonan tanpa vokal > 3 huruf berturut-turut (khas Indonesia rusak OCR).
    """
    if not text:
        return 0.0
    noise_chars = set("·;\\~^`|{}[]<>") | {chr(i) for i in range(128, 256)}
    words       = re.findall(r"\S+", text)
    if not words:
        return 0.0
    noise_count = 0
    for word in words:
        inner = word.strip(".,;:!?()\"-'")
        if any(c in noise_chars for c in inner):
            noise_count += 1
            continue
        letters = re.sub(r"[^a-zA-Z]", "", inner)
        if len(letters) > 3 and not re.search(r"[aeiouAEIOU]", letters):
            noise_count += 1
    return noise_count / len(words)


def _chunk_needs_correction(text: str) -> bool:
    return _ocr_noise_score(text) >= OCR_BAD_THRESHOLD


class OcrCorrector:
    """
    Wrapper Qwen2.5-7B-Instruct via HuggingFace Transformers + bitsandbytes 4-bit.

    GTX 1080 (8GB VRAM):
      - 4-bit quantization  ≈ 4.3GB VRAM  ✅
      - 8-bit quantization  ≈ 7.5GB VRAM  ✅ (tight)
      - fp16                ≈ 14GB VRAM   ❌

    Model ~15GB didownload otomatis ke ~/.cache/huggingface pada pertama kali.
    """

    def __init__(
        self,
        model_name: str = OCR_CORRECTION_MODEL,
        load_in_4bit: bool = OCR_CORRECTION_4BIT,
    ):
        try:
            import torch
            from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        except ImportError:
            raise RuntimeError(
                "Dependency belum terinstall.\n"
                "Jalankan: pip install transformers accelerate bitsandbytes"
            )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA tidak terdeteksi.\n"
                "Pastikan driver NVIDIA dan PyTorch CUDA sudah terinstall:\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cu118"
            )

        gpu_name = torch.cuda.get_device_name(0)
        vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"🖥️  GPU: {gpu_name} ({vram_gb:.1f}GB VRAM)")
        print(f"🤖 Loading OCR corrector: {model_name} ({'4-bit' if load_in_4bit else 'fp16'})...")
        print(f"   (Download otomatis ~15GB jika belum ada di cache)")
        t0 = time.time()

        self._tokenizer_lm = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,   # hemat ~0.4GB lagi
                bnb_4bit_quant_type="nf4",        # nf4 lebih akurat dari fp4
                bnb_4bit_compute_dtype=torch.float16,
            )

        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",             # otomatis alokasi layer ke GPU/CPU
            trust_remote_code=True,
            torch_dtype=torch.float16 if not load_in_4bit else None,
        )
        self._model.eval()
        print(f"   ✅ Model loaded ({time.time() - t0:.1f}s)")

    def _build_input(self, texts: List[str]):
        import torch
        items    = [{"i": i, "t": t} for i, t in enumerate(texts)]
        user_msg = (
            "Koreksi OCR teks hukum berikut. "
            "Kembalikan JSON array string dengan urutan yang sama:\n\n"
            + json.dumps(items, ensure_ascii=False)
        )
        messages = [
            {"role": "system", "content": _OCR_CORRECTION_SYSTEM},
            {"role": "user",   "content": user_msg},
        ]
        text_input = self._tokenizer_lm.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self._tokenizer_lm(text_input, return_tensors="pt").to(self._model.device)

    def _infer(self, texts: List[str]) -> List[str]:
        import torch
        inputs = self._build_input(texts)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self._tokenizer_lm.eos_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        generated = output_ids[0][input_len:]
        raw       = self._tokenizer_lm.decode(generated, skip_special_tokens=True).strip()

        # Bersihkan markdown fence jika model nambah
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()

        # Isolasi array JSON
        arr_match = re.search(r"\[.*\]", raw, re.DOTALL)
        if arr_match:
            raw = arr_match.group(0)

        try:
            result = json.loads(raw)
            # Normalise jika model return list of objects {"i": 0, "t": "..."}
            if result and isinstance(result[0], dict):
                result = [item.get("t", item.get("text", "")) for item in result]
            if isinstance(result, list) and len(result) == len(texts):
                return [str(r) for r in result]
        except (json.JSONDecodeError, KeyError):
            pass

        print("    ⚠️  Parse JSON gagal, pakai teks asli batch ini.")
        return texts

    def correct_chunks(self, chunks: List[Dict]) -> List[Dict]:
        bad_indices = [i for i, c in enumerate(chunks) if _chunk_needs_correction(c["text"])]

        if not bad_indices:
            print("  ✅ Semua chunk OCR bersih, skip koreksi.")
            return chunks

        print(f"  🔧 {len(bad_indices)}/{len(chunks)} chunk perlu koreksi OCR...")
        result = [dict(c) for c in chunks]

        for batch_start in range(0, len(bad_indices), OCR_CORRECTION_BATCH):
            batch_idx = bad_indices[batch_start: batch_start + OCR_CORRECTION_BATCH]
            texts_in  = [chunks[i]["text"] for i in batch_idx]
            nums      = [i + 1 for i in batch_idx]
            print(f"    Batch chunk #{nums} ...", end=" ", flush=True)
            t0 = time.time()

            texts_out = self._infer(texts_in)

            print(f"({time.time() - t0:.1f}s)")
            for chunk_idx, fixed in zip(batch_idx, texts_out):
                result[chunk_idx]["text"]          = fixed
                result[chunk_idx]["ocr_corrected"] = True

        fixed = sum(1 for c in result if c.get("ocr_corrected"))
        print(f"  ✅ Koreksi selesai: {fixed} chunk diperbaiki.")
        return result


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE UTAMA
# ─────────────────────────────────────────────────────────────────────────────
def iter_pdf_files(pdf_directory: str) -> Iterable[str]:
    for root, _, files in os.walk(pdf_directory):
        for file in sorted(files):
            if file.lower().endswith(".pdf"):
                yield os.path.join(root, file)


def process_pdf(filepath: str, corrector: Optional[OcrCorrector] = None) -> List[Dict]:
    print(f"\n📄 Memproses: {os.path.basename(filepath)}")
    text = extract_and_clean_text(filepath)
    if not text:
        print("  ⚠️  Teks kosong, dilewati.")
        return []
    metadata = extract_metadata(filepath, text)
    chunks   = structural_chunking(text, metadata)
    print(
        f"  ✅ {metadata['regulation_type']} No {metadata['nomor']} "
        f"Th {metadata['publication_year']} | {len(chunks)} chunks"
    )
    if corrector is not None:
        chunks = corrector.correct_chunks(chunks)
    return chunks


def process_corpus(
    pdf_directory: str,
    corrector: Optional[OcrCorrector] = None,
    limit: Optional[int] = None,
) -> List[Dict]:
    all_chunks: List[Dict] = []
    for index, filepath in enumerate(iter_pdf_files(pdf_directory), start=1):
        if limit is not None and index > limit:
            break
        all_chunks.extend(process_pdf(filepath, corrector=corrector))
    return all_chunks


def save_chunks(chunks: List[Dict], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"💾 Disimpan ke: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess PDF regulasi Indonesia → JSON chunks untuk RAG."
    )
    parser.add_argument("--pdf-dir",        default=DEFAULT_PDF_DIR,     help="Folder berisi PDF.")
    parser.add_argument("--pdf",                                          help="Proses satu PDF saja.")
    parser.add_argument("--output",         default=DEFAULT_OUTPUT_PATH, help="Path output JSON.")
    parser.add_argument("--limit",          type=int,                    help="Batasi jumlah PDF (preview).")
    parser.add_argument("--ocr-correction", action="store_true",         help="Aktifkan OCR post-correction via Qwen2.5-7B.")
    parser.add_argument("--no-ocr-correction", dest="ocr_correction", action="store_false")
    parser.set_defaults(ocr_correction=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not TESSERACT_READY:
        print("⚠️  Tesseract tidak ditemukan. Halaman scan diproses tanpa OCR.")

    # Inisialisasi OCR corrector jika diminta
    corrector: Optional[OcrCorrector] = None
    if args.ocr_correction:
        print("\n🔧 Inisialisasi OCR corrector (Qwen2.5-7B)...")
        corrector = OcrCorrector()

    # Proses
    if args.pdf:
        corpus_chunks = process_pdf(args.pdf, corrector=corrector)
    else:
        corpus_chunks = process_corpus(args.pdf_dir, corrector=corrector, limit=args.limit)

    print(f"\n✅ Total chunk: {len(corpus_chunks)}")
    save_chunks(corpus_chunks, args.output)

    if corpus_chunks:
        print("\n📄 Contoh chunk pertama:")
        print(json.dumps(corpus_chunks[0], ensure_ascii=False, indent=2))