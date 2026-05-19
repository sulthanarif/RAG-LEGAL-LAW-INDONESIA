# Pipeline Production-Grade Legal RAG Indonesia
## Case-Based Employment Law Article Recommendation System using Retrieval-Augmented Generation

---

## Daftar Isi

1. [Arsitektur Keseluruhan](#1-arsitektur-keseluruhan)
2. [Phase 1 - OCR & Preprocessing](#2-phase-1--ocr--preprocessing)
3. [Phase 2 - Legal Structure Parser](#3-phase-2--legal-structure-parser)
4. [Phase 3 - Legal-Aware Chunking](#4-phase-3--legal-aware-chunking)
5. [Phase 4 - Skema Metadata Final](#5-phase-4--skema-metadata-final)
6. [Phase 5 - Embedding Pipeline](#6-phase-5--embedding-pipeline)
7. [Phase 6 - Vector Database (ChromaDB)](#7-phase-6--vector-database-chromadb)
8. [Phase 7 - Retrieval and Lex Posterior Ranking](#8-phase-7--retrieval-and-lex-posterior-ranking)
9. [Phase 8 - Post-Retrieval Deduplication and Neural Reranking](#9-phase-8--post-retrieval-deduplication-and-neural-reranking)
10. [Phase 9 - Context Assembler](#10-phase-9--context-assembler)
11. [Phase 10 - Answer Generation & Citation](#11-phase-10--answer-generation--citation)
12. [Phase 11 - Evaluasi Pipeline](#12-phase-11--evaluasi-pipeline)
13. [Roadmap Implementasi](#13-roadmap-implementasi)

---

## 1. Arsitektur Keseluruhan

```text
PDF Regulasi Ketenagakerjaan
        |
        v
Phase 1 - OCR/Text Extraction + Cleaning
        |
        v
Phase 2 - Legal Structure Parser
        |   BAB / Bagian / Paragraf / Pasal
        v
Phase 3 - Legal-Aware Chunking
        |   Pasal-level chunks + long-pasal split
        v
Phase 4 - Chroma-ready Metadata Finalization
        |   display_text / embedding_text / citation_text
        v
Phase 5 - Embedding Pipeline
        |   intfloat/multilingual-e5-base
        v
Phase 6 - ChromaDB Vector Store + BM25 Index
        |   data/chroma_db + data/bm25_index.pkl
        v
Phase 7 - Retrieval and Lex Posterior Ranking
        |   Dense cosine + BM25 + RRF + publication_year ranking
        v
Phase 8 - Deduplication and Neural Reranking
        |   Remove repeated article chunks before prompting
        v
Phase 9 - Context Assembler
        |   [R1] citation_text + display_text
        v
Phase 10 - Local LLM Answer Generation
        |   Qwen open-weight model + mandatory citations
        v
Phase 11 - Evaluation
            Precision@k, MRR, Faithfulness, Answer Relevancy, Error Analysis
```

---

## 2. Phase 1 — OCR & Preprocessing

### 2.1 Tujuan

Menghasilkan teks bersih, konsisten, dan terlacak dari PDF hukum Indonesia yang formatnya tidak seragam.

### 2.2 Urutan Operasi (WAJIB berurutan)

```
PDF Binary
    │
    ▼  (1) Coba text extraction native
PyMuPDF / pdfplumber
    │
    ├── Berhasil (confidence > 0.85) ──► Langsung ke Cleaning
    │
    └── Gagal / terlalu banyak karakter aneh
         │
         ▼  (2) Image rendering + pre-processing
    PDF → PNG per halaman (300 DPI)
         │
         ▼  (3) Image Enhancement (SEBELUM OCR)
    Deskew → Denoise → Threshold → Contrast
         │
         ▼  (4) OCR
    Tesseract (lang: ind+eng) / EasyOCR
         │
         ▼
    Teks mentah per halaman
         │
         ▼  (5) Cleaning Pipeline
    Legal Text Normalization
```

### 2.3 Legal Cleaning Rule Engine

Jangan hardcode regex. Buat rule engine yang maintainable:

```python
from dataclasses import dataclass
from typing import Callable
import re

@dataclass
class CleaningRule:
    name: str
    pattern: str
    replacement: str
    description: str
    enabled: bool = True

LEGAL_CLEANING_RULES: list[CleaningRule] = [
    CleaningRule(
        name="fix_ocr_bab",
        pattern=r'\bBA[B8]\b',
        replacement='BAB',
        description="Perbaiki OCR error: BA8 → BAB"
    ),
    CleaningRule(
        name="fix_ocr_pasal",
        pattern=r'\bPasa[lI1]\b',
        replacement='Pasal',
        description="Perbaiki OCR error: Pasal huruf I/1 → l"
    ),
    CleaningRule(
        name="fix_ocr_ayat",
        pattern=r'\(([0-9Oo]+)\)',
        replacement=lambda m: f"({m.group(1).replace('O','0').replace('o','0')})",
        description="Perbaiki OCR: (O) → (0), (o) → (0)"
    ),
    CleaningRule(
        name="normalize_whitespace",
        pattern=r'[ \t]+',
        replacement=' ',
        description="Normalisasi spasi berlebih"
    ),
    CleaningRule(
        name="normalize_newlines",
        pattern=r'\n{3,}',
        replacement='\n\n',
        description="Maksimal 2 baris kosong"
    ),
    CleaningRule(
        name="remove_header_footer",
        pattern=r'(?m)^-\s*\d+\s*-$',
        replacement='',
        description="Hapus nomor halaman format - 12 -"
    ),
    CleaningRule(
        name="fix_pasal_number",
        pattern=r'Pasa[l]\s+(\d+[A-Z]?)',
        replacement=r'Pasal \1',
        description="Normalisasi spasi setelah Pasal"
    ),
    CleaningRule(
        name="fix_ayat_bracket",
        pattern=r'\(\s*(\d+)\s*\)',
        replacement=r'(\1)',
        description="Normalisasi bracket ayat: ( 1 ) → (1)"
    ),
    CleaningRule(
        name="fix_huruf_list",
        pattern=r'(?m)^([a-z])\s*\.\s+',
        replacement=r'\1. ',
        description="Normalisasi list huruf: a.   → a. "
    ),
]

def apply_cleaning_rules(text: str, rules: list[CleaningRule] = LEGAL_CLEANING_RULES) -> dict:
    """Terapkan semua cleaning rules dan kembalikan teks bersih + log."""
    applied = []
    for rule in rules:
        if not rule.enabled:
            continue
        if callable(rule.replacement):
            new_text = re.sub(rule.pattern, rule.replacement, text)
        else:
            new_text = re.sub(rule.pattern, rule.replacement, text)
        if new_text != text:
            applied.append(rule.name)
            text = new_text
    return {"text": text, "rules_applied": applied}
```

### 2.4 Struktur Output Phase 1

Setiap halaman PDF harus menghasilkan struktur ini sebelum masuk parser:

```python
@dataclass
class PageResult:
    page_number: int        # 1-indexed
    text: str               # Teks bersih
    char_start: int         # Offset karakter absolut dari awal dokumen
    char_end: int           # Offset karakter absolut
    ocr_used: bool          # True kalau pakai OCR fallback
    ocr_confidence: float   # 0.0 - 1.0, -1 kalau tidak pakai OCR
    rules_applied: list     # Cleaning rules yang aktif di halaman ini
    source_file: str        # Nama file PDF asal
```

### 2.5 Normalisasi Nama Regulasi

Penting untuk konsistensi metadata lintas dokumen:

```python
REGULATION_TYPE_MAP = {
    # Undang-Undang
    r'undang.undang|(?<!\w)uu(?!\w)': 'Undang-Undang',
    # Peraturan Pemerintah
    r'peraturan pemerintah|(?<!\w)pp(?!\w)': 'Peraturan Pemerintah',
    # Peraturan Presiden
    r'peraturan presiden|perpres': 'Peraturan Presiden',
    # Peraturan Menteri
    r'peraturan menteri ketenagakerjaan|permenaker': 'Peraturan Menteri Ketenagakerjaan',
    r'peraturan menteri(?!\s+keten)': 'Peraturan Menteri',
    # Putusan
    r'putusan mahkamah konstitusi|putusan mk': 'Putusan MK',
    r'putusan mahkamah agung|putusan ma': 'Putusan MA',
    # Keputusan
    r'keputusan menteri|kepmen': 'Keputusan Menteri',
    r'keputusan presiden|keppres': 'Keputusan Presiden',
    # Instruksi
    r'instruksi presiden|inpres': 'Instruksi Presiden',
    # PKWT / Regulasi teknis
    r'peraturan pelaksana': 'Peraturan Pelaksana',
}

# Hierarki kekuatan hukum (semakin kecil = semakin tinggi)
REGULATION_HIERARCHY = {
    'Undang-Undang': 1,
    'Peraturan Pemerintah Pengganti Undang-Undang': 1,
    'Peraturan Pemerintah': 2,
    'Peraturan Presiden': 3,
    'Peraturan Menteri Ketenagakerjaan': 4,
    'Peraturan Menteri': 4,
    'Keputusan Presiden': 4,
    'Keputusan Menteri': 5,
    'Instruksi Presiden': 5,
    'Putusan MK': 1,    # Setara UU
    'Putusan MA': 3,
}
```

---

## 3. Phase 2 — Legal Structure Parser

### 3.1 Tujuan

Mengubah teks mentah menjadi **Legal Tree** bertingkat sebelum chunking. Parser harus stateful, bukan regex split sederhana.

### 3.2 Pola Regex per Jenis Elemen

```python
import re
from dataclasses import dataclass, field
from typing import Optional

# --- Pola Header ---
PATTERNS = {
    # BAB: BAB I, BAB II, BAB X, BAB XII
    'bab': re.compile(
        r'^BAB\s+(I{1,4}|IV|VI{0,3}|IX|XI{0,3}|XIV|XV|XVI{0,3}|XIX|XX{1,3}|[IVXLC]+|\d+)',
        re.MULTILINE
    ),
    # Bagian: Bagian Kesatu, Bagian Kedua, dst
    'bagian': re.compile(
        r'^Bagian\s+(Kesatu|Kedua|Ketiga|Keempat|Kelima|Keenam|Ketujuh|Kedelapan|Kesembilan|Kesepuluh|\w+)',
        re.MULTILINE
    ),
    # Paragraf
    'paragraf': re.compile(
        r'^Paragraf\s+(\d+|\w+)',
        re.MULTILINE
    ),
    # Pasal: Pasal 1, Pasal 5A, Pasal 156A
    'pasal': re.compile(
        r'^Pasal\s+(\d+[A-Z]?)',
        re.MULTILINE
    ),
    # Ayat: (1), (2)
    'ayat': re.compile(
        r'^\((\d+)\)',
        re.MULTILINE
    ),
    # List huruf: a., b.
    'huruf': re.compile(
        r'^([a-z])\.',
        re.MULTILINE
    ),
    # List angka: 1., 2.
    'angka': re.compile(
        r'^(\d+)\.',
        re.MULTILINE
    ),
    # Definisi di Pasal 1: angka 1., angka 2.
    'definisi': re.compile(
        r'^(\d+)\.\s+\w+\s+adalah\b',
        re.MULTILINE
    ),
    # Amandemen
    'amandemen_ubah': re.compile(
        r'Pasal\s+(\d+[A-Z]?)\s+diubah',
        re.IGNORECASE
    ),
    'amandemen_hapus': re.compile(
        r'Pasal\s+(\d+[A-Z]?)\s+dihapus',
        re.IGNORECASE
    ),
    'amandemen_sisip': re.compile(
        r'[Dd]i antara Pasal\s+(\d+[A-Z]?)\s+dan\s+Pasal\s+(\d+[A-Z]?)',
    ),
    # Penjelasan
    'penjelasan': re.compile(
        r'^PENJELASAN\s*$',
        re.MULTILINE
    ),
    # Lampiran
    'lampiran': re.compile(
        r'^LAMPIRAN\s*(I{1,4}|IV|VI{0,3}|IX|X{1,3}|\d+)?',
        re.MULTILINE
    ),
}
```

### 3.3 Stateful Parser

```python
@dataclass
class AyatNode:
    ayat_no: int
    text: str
    items: list[dict] = field(default_factory=list)  # huruf/angka nested
    page_start: int = 0
    page_end: int = 0
    char_start: int = 0
    char_end: int = 0

@dataclass
class PasalNode:
    pasal_id: str          # "Pasal 5", "Pasal 156A"
    ayat: list[AyatNode] = field(default_factory=list)
    text_intro: str = ""   # Teks sebelum ayat pertama (kalau ada)
    is_amendment: bool = False
    amendment_type: str = ""  # "ubah", "hapus", "sisip"
    amendment_of: str = ""    # Referensi ke chunk/pasal asal
    page_start: int = 0
    page_end: int = 0

@dataclass
class ParagrafNode:
    paragraf_id: str
    paragraf_title: str
    pasals: list[PasalNode] = field(default_factory=list)

@dataclass
class BagianNode:
    bagian_id: str
    bagian_title: str
    paragrafs: list[ParagrafNode] = field(default_factory=list)
    pasals: list[PasalNode] = field(default_factory=list)  # Pasal langsung tanpa paragraf

@dataclass
class BabNode:
    bab_id: str            # "BAB I"
    bab_title: str         # "KETENTUAN UMUM"
    bagians: list[BagianNode] = field(default_factory=list)
    pasals: list[PasalNode] = field(default_factory=list)  # Pasal langsung tanpa bagian

@dataclass
class RegulationTree:
    regulation_type: str
    nomor: str
    tahun: int
    tentang: str
    source_file: str
    active_status: str
    babs: list[BabNode] = field(default_factory=list)
    has_penjelasan: bool = False
    has_lampiran: bool = False
    penjelasan_text: str = ""


class LegalParser:
    """
    Stateful parser untuk regulasi hukum Indonesia.
    Mendukung: UU, PP, Perpres, Permen, Kepmen, Putusan MK/MA.
    """

    def __init__(self):
        self.reset_state()

    def reset_state(self):
        self.current_bab: Optional[BabNode] = None
        self.current_bagian: Optional[BagianNode] = None
        self.current_paragraf: Optional[ParagrafNode] = None
        self.current_pasal: Optional[PasalNode] = None
        self.current_ayat: Optional[AyatNode] = None
        self.current_section = "batang_tubuh"  # atau "penjelasan", "lampiran"
        self.buffer = []

    def parse(self, pages: list[PageResult], reg_meta: dict) -> RegulationTree:
        """
        Parse semua halaman menjadi RegulationTree.
        pages: output dari Phase 1
        reg_meta: metadata regulasi dari nama file / header
        """
        tree = RegulationTree(
            regulation_type=reg_meta['regulation_type'],
            nomor=reg_meta['nomor'],
            tahun=reg_meta['tahun'],
            tentang=reg_meta['tentang'],
            source_file=reg_meta['source_file'],
            active_status=reg_meta.get('active_status', 'Berlaku'),
        )

        self.reset_state()
        self.tree = tree

        for page in pages:
            lines = page.text.split('\n')
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                self._process_line(line, page)

        # Flush buffer terakhir
        self._flush_buffer()
        return tree

    def _process_line(self, line: str, page: PageResult):
        """Routing setiap baris ke handler yang tepat."""

        # Deteksi masuk ke section Penjelasan
        if PATTERNS['penjelasan'].match(line):
            self._flush_buffer()
            self.current_section = 'penjelasan'
            self.tree.has_penjelasan = True
            return

        # Deteksi masuk ke section Lampiran
        if PATTERNS['lampiran'].match(line):
            self._flush_buffer()
            self.current_section = 'lampiran'
            self.tree.has_lampiran = True
            return

        # Kalau sudah di section penjelasan/lampiran, jangan parse struktur
        if self.current_section in ('penjelasan', 'lampiran'):
            self.buffer.append(line)
            return

        # --- Deteksi elemen struktural ---
        if PATTERNS['bab'].match(line):
            self._flush_buffer()
            self._handle_bab(line, page)
        elif PATTERNS['bagian'].match(line):
            self._flush_buffer()
            self._handle_bagian(line, page)
        elif PATTERNS['paragraf'].match(line):
            self._flush_buffer()
            self._handle_paragraf(line, page)
        elif PATTERNS['pasal'].match(line):
            self._flush_buffer()
            self._handle_pasal(line, page)
        elif PATTERNS['ayat'].match(line):
            self._flush_buffer()
            self._handle_ayat(line, page)
        else:
            # Teks biasa → masuk buffer
            self.buffer.append(line)

    def _handle_bab(self, line: str, page: PageResult):
        match = PATTERNS['bab'].match(line)
        bab_id = f"BAB {match.group(1)}"
        # Judul BAB biasanya di baris berikutnya — akan diambil dari buffer
        new_bab = BabNode(bab_id=bab_id, bab_title="")
        self.tree.babs.append(new_bab)
        self.current_bab = new_bab
        self.current_bagian = None
        self.current_paragraf = None
        self.current_pasal = None
        self.current_ayat = None

    def _handle_pasal(self, line: str, page: PageResult):
        match = PATTERNS['pasal'].match(line)
        pasal_id = f"Pasal {match.group(1)}"

        # Cek amandemen
        is_amendment = False
        amendment_type = ""
        if PATTERNS['amandemen_hapus'].search(line):
            is_amendment = True
            amendment_type = "hapus"
        elif PATTERNS['amandemen_ubah'].search(line):
            is_amendment = True
            amendment_type = "ubah"

        new_pasal = PasalNode(
            pasal_id=pasal_id,
            page_start=page.page_number,
            is_amendment=is_amendment,
            amendment_type=amendment_type,
        )

        # Tambahkan ke parent yang tepat
        if self.current_paragraf:
            self.current_paragraf.pasals.append(new_pasal)
        elif self.current_bagian:
            self.current_bagian.pasals.append(new_pasal)
        elif self.current_bab:
            self.current_bab.pasals.append(new_pasal)

        self.current_pasal = new_pasal
        self.current_ayat = None

    def _handle_ayat(self, line: str, page: PageResult):
        match = PATTERNS['ayat'].match(line)
        ayat_no = int(match.group(1))
        ayat_text = line[match.end():].strip()

        new_ayat = AyatNode(
            ayat_no=ayat_no,
            text=ayat_text,
            page_start=page.page_number,
        )
        if self.current_pasal:
            self.current_pasal.ayat.append(new_ayat)
        self.current_ayat = new_ayat

    def _flush_buffer(self):
        """Flush teks buffer ke node aktif yang tepat."""
        if not self.buffer:
            return
        text = ' '.join(self.buffer).strip()
        self.buffer = []

        if not text:
            return

        if self.current_section == 'penjelasan':
            self.tree.penjelasan_text += text + '\n'
            return

        # Cek apakah ini judul BAB (setelah header BAB)
        if self.current_bab and not self.current_bab.bab_title:
            self.current_bab.bab_title = text
            return

        # Teks masuk ke ayat aktif
        if self.current_ayat:
            self.current_ayat.text += ' ' + text
            return

        # Teks intro pasal (sebelum ayat pertama)
        if self.current_pasal:
            self.current_pasal.text_intro += ' ' + text
            return
```

---

## 4. Phase 3 — Legal-Aware Chunking

### 4.1 Aturan Chunking per Kasus

```
┌─────────────────────────────────────────────────────────┐
│  DECISION TREE CHUNKING                                  │
│                                                          │
│  Node saat ini: AyatNode                                 │
│        │                                                 │
│        ├── len(text) < 50 karakter?                      │
│        │       └── MERGE dengan sibling ayat berikutnya  │
│        │                                                 │
│        ├── len(text) > 1500 karakter?                    │
│        │       └── SPLIT per kalimat / sub-poin          │
│        │                                                 │
│        ├── pasal_id == "Pasal 1" AND definisi?           │
│        │       └── 1 definisi = 1 chunk                  │
│        │                                                 │
│        ├── is_amendment == True?                         │
│        │       └── chunk_kind = "amendment"              │
│        │                                                 │
│        └── Default: 1 ayat = 1 chunk                     │
└─────────────────────────────────────────────────────────┘
```

### 4.2 Implementasi Chunker

```python
from typing import Optional
import re

MIN_CHUNK_CHARS = 60
MAX_CHUNK_CHARS = 1500

class LegalChunker:

    def chunk_regulation(self, tree: RegulationTree) -> list[dict]:
        """Konversi RegulationTree menjadi list of chunks siap embed."""
        chunks = []
        chunk_index = 0

        for bab in tree.babs:
            # Proses langsung pasal di bawah BAB (tanpa Bagian)
            for pasal in bab.pasals:
                new_chunks = self._chunk_pasal(pasal, bab, None, None, tree, chunk_index)
                chunks.extend(new_chunks)
                chunk_index += len(new_chunks)

            # Proses Bagian
            for bagian in bab.bagians:
                for paragraf in bagian.paragrafs:
                    for pasal in paragraf.pasals:
                        new_chunks = self._chunk_pasal(pasal, bab, bagian, paragraf, tree, chunk_index)
                        chunks.extend(new_chunks)
                        chunk_index += len(new_chunks)

                for pasal in bagian.pasals:
                    new_chunks = self._chunk_pasal(pasal, bab, bagian, None, tree, chunk_index)
                    chunks.extend(new_chunks)
                    chunk_index += len(new_chunks)

        return chunks

    def _chunk_pasal(
        self,
        pasal: PasalNode,
        bab: BabNode,
        bagian: Optional[BagianNode],
        paragraf: Optional[ParagrafNode],
        tree: RegulationTree,
        start_index: int
    ) -> list[dict]:
        chunks = []

        # Deteksi apakah ini Pasal Definisi (Pasal 1)
        is_definisi_pasal = pasal.pasal_id in ("Pasal 1",)

        # Kalau tidak ada ayat tapi ada text_intro — jadikan satu chunk
        if not pasal.ayat and pasal.text_intro.strip():
            chunk = self._build_chunk(
                text=pasal.text_intro.strip(),
                pasal=pasal,
                ayat=None,
                bab=bab,
                bagian=bagian,
                paragraf=paragraf,
                tree=tree,
                chunk_kind="pasal",
                chunk_index=start_index,
                definition_no=None,
            )
            chunks.append(chunk)
            return chunks

        # Proses per ayat
        i = 0
        def_counter = 1
        ayat_list = pasal.ayat

        while i < len(ayat_list):
            ayat = ayat_list[i]
            text = ayat.text.strip()

            # Rule: Definisi (Pasal 1) → 1 definisi = 1 chunk
            if is_definisi_pasal:
                def_matches = re.findall(r'\d+\.\s+\w[\w\s]+\s+adalah\b[^.]+\.', text)
                if def_matches:
                    for def_text in def_matches:
                        chunk = self._build_chunk(
                            text=def_text.strip(),
                            pasal=pasal,
                            ayat=ayat,
                            bab=bab,
                            bagian=bagian,
                            paragraf=paragraf,
                            tree=tree,
                            chunk_kind="definition",
                            chunk_index=start_index + len(chunks),
                            definition_no=def_counter,
                        )
                        chunks.append(chunk)
                        def_counter += 1
                    i += 1
                    continue

            # Rule: Ayat terlalu pendek → merge dengan berikutnya
            if len(text) < MIN_CHUNK_CHARS and i + 1 < len(ayat_list):
                merged_text = text + ' ' + ayat_list[i+1].text.strip()
                chunk = self._build_chunk(
                    text=merged_text,
                    pasal=pasal,
                    ayat=ayat,
                    bab=bab,
                    bagian=bagian,
                    paragraf=paragraf,
                    tree=tree,
                    chunk_kind="ayat_merged",
                    chunk_index=start_index + len(chunks),
                    definition_no=None,
                    ayat_no_end=ayat_list[i+1].ayat_no,
                )
                chunks.append(chunk)
                i += 2
                continue

            # Rule: Ayat terlalu panjang → split per kalimat
            if len(text) > MAX_CHUNK_CHARS:
                split_chunks = self._split_long_ayat(
                    text, pasal, ayat, bab, bagian, paragraf, tree,
                    start_index + len(chunks)
                )
                chunks.extend(split_chunks)
                i += 1
                continue

            # Default: 1 ayat = 1 chunk
            chunk_kind = "amendment" if pasal.is_amendment else "ayat"
            chunk = self._build_chunk(
                text=text,
                pasal=pasal,
                ayat=ayat,
                bab=bab,
                bagian=bagian,
                paragraf=paragraf,
                tree=tree,
                chunk_kind=chunk_kind,
                chunk_index=start_index + len(chunks),
                definition_no=None,
            )
            chunks.append(chunk)
            i += 1

        return chunks

    def _split_long_ayat(self, text, pasal, ayat, bab, bagian, paragraf, tree, start_index):
        """Split ayat panjang per kalimat, pertahankan konteks."""
        sentences = re.split(r'(?<=[.;:])\s+', text)
        buffer = ""
        sub_chunks = []
        sub_index = 0

        for sent in sentences:
            if len(buffer) + len(sent) > MAX_CHUNK_CHARS and buffer:
                chunk = self._build_chunk(
                    text=buffer.strip(),
                    pasal=pasal, ayat=ayat,
                    bab=bab, bagian=bagian, paragraf=paragraf,
                    tree=tree,
                    chunk_kind="ayat_split",
                    chunk_index=start_index + sub_index,
                    definition_no=None,
                    split_part=sub_index + 1,
                )
                sub_chunks.append(chunk)
                buffer = sent
                sub_index += 1
            else:
                buffer += (' ' if buffer else '') + sent

        if buffer.strip():
            chunk = self._build_chunk(
                text=buffer.strip(),
                pasal=pasal, ayat=ayat,
                bab=bab, bagian=bagian, paragraf=paragraf,
                tree=tree,
                chunk_kind="ayat_split",
                chunk_index=start_index + sub_index,
                definition_no=None,
                split_part=sub_index + 1,
            )
            sub_chunks.append(chunk)

        return sub_chunks

    def _build_chunk(
        self, text, pasal, ayat, bab, bagian, paragraf, tree,
        chunk_kind, chunk_index, definition_no,
        ayat_no_end=None, split_part=None
    ) -> dict:
        """Bangun objek chunk final dengan 3 representasi teks."""

        ayat_no = ayat.ayat_no if ayat else None

        # --- Citation Text (untuk referensi legal) ---
        citation_parts = [tree.regulation_type, f"No. {tree.nomor}", f"Tahun {tree.tahun}"]
        if ayat_no:
            citation_parts.append(f"{pasal.pasal_id} ayat ({ayat_no})")
        else:
            citation_parts.append(pasal.pasal_id)
        citation_text = ' '.join(citation_parts)

        # --- Display Text (untuk ditampilkan ke user) ---
        display_lines = [
            f"{pasal.pasal_id}" + (f" ayat ({ayat_no})" if ayat_no else ""),
            text
        ]
        display_text = '\n'.join(display_lines)

        # --- Embedding Text (konteks penuh untuk model embedding) ---
        # INI YANG PALING PENTING: tambahkan hierarki dan konteks
        embed_lines = [
            f"{tree.regulation_type} Nomor {tree.nomor} Tahun {tree.tahun}",
            f"Tentang {tree.tentang}",
            "",
        ]
        if bab.bab_id:
            embed_lines.append(f"{bab.bab_id}: {bab.bab_title}")
        if bagian:
            embed_lines.append(f"{bagian.bagian_id}: {bagian.bagian_title}")
        if paragraf:
            embed_lines.append(f"{paragraf.paragraf_id}: {paragraf.paragraf_title}")
        embed_lines.append("")
        embed_lines.append(pasal.pasal_id + (f" Ayat ({ayat_no})" if ayat_no else ""))
        embed_lines.append("")
        embed_lines.append(text)
        embedding_text = '\n'.join(embed_lines)

        return {
            # Teks utama
            "text": text,                         # Teks bersih (untuk LLM)
            "display_text": display_text,         # Untuk ditampilkan ke user
            "embedding_text": embedding_text,     # Untuk embed — dengan konteks hierarki
            "citation_text": citation_text,       # Untuk sitasi legal

            # Metadata regulasi
            "metadata": {
                # Identitas regulasi
                "regulation_type": tree.regulation_type,
                "nomor": tree.nomor,
                "tentang": tree.tentang,
                "year": tree.tahun,
                "publication_year": tree.tahun,   # Duplikat untuk backward compat.
                "active_status": tree.active_status,
                "source_file": tree.source_file,

                # Hierarki struktur
                "bab": bab.bab_id,
                "bab_title": bab.bab_title,
                "bagian": bagian.bagian_id if bagian else None,
                "bagian_title": bagian.bagian_title if bagian else None,
                "paragraf": paragraf.paragraf_id if paragraf else None,
                "paragraf_title": paragraf.paragraf_title if paragraf else None,

                # Identitas chunk
                "pasal_id": pasal.pasal_id,
                "ayat_no": ayat_no,
                "ayat_no_end": ayat_no_end,       # Kalau merged, end ayat
                "definition_no": definition_no,
                "split_part": split_part,          # Kalau di-split

                # Klasifikasi
                "chunk_kind": chunk_kind,
                # chunk_kind values:
                #   "definition"   → item definisi di Pasal 1
                #   "ayat"         → ayat normal
                #   "pasal"        → pasal tanpa ayat
                #   "ayat_merged"  → dua ayat pendek digabung
                #   "ayat_split"   → ayat panjang dipotong
                #   "amendment"    → pasal amandemen
                "chunk_index": chunk_index,

                # Amandemen
                "is_amendment": pasal.is_amendment,
                "amendment_type": pasal.amendment_type,   # "ubah", "hapus", "sisip"
                "amendment_of": pasal.amendment_of,       # chunk_id yang diubah

                # Traceability
                "page_start": pasal.page_start,
                "page_end": pasal.page_end,
                "char_start": ayat.char_start if ayat else 0,
                "char_end": ayat.char_end if ayat else 0,

                # Sitasi
                "citation_text": citation_text,

                # Hierarki kekuatan hukum (untuk Lex Posterior scoring)
                "regulation_hierarchy": REGULATION_HIERARCHY.get(tree.regulation_type, 99),
            }
        }
```

### 4.3 Contoh Output Chunk Final

Output chunk final yang dipakai oleh notebook dan ChromaDB adalah hasil `finalize_chunks_for_chroma.py`, bukan output mentah extractor.

```json
{
  "id": "060edad653077c90cff970227bb120f5ddf74aa6",
  "text": "1. Alat Pelindung Diri selanjutnya disingkat APD adalah ...",
  "display_text": "Peraturan Menteri Ketenagakerjaan No. PER.08/MEN/VII/2010 Tahun 2010 tentang ALAT PELINDUNG DIRI\nPasal 1\n1. Alat Pelindung Diri ...",
  "embedding_text": "Peraturan Menteri Ketenagakerjaan No. PER.08/MEN/VII/2010 Tahun 2010 tentang ALAT PELINDUNG DIRI\nPasal 1\n1. Alat Pelindung Diri ...",
  "citation_text": "Peraturan Menteri Ketenagakerjaan No. PER.08/MEN/VII/2010 Tahun 2010 tentang ALAT PELINDUNG DIRI, Pasal 1",
  "metadata": {
    "regulation_type": "Peraturan Menteri Ketenagakerjaan",
    "regulation_hierarchy": 4,
    "nomor": "PER.08/MEN/VII/2010",
    "tentang": "ALAT PELINDUNG DIRI",
    "year": 2010,
    "publication_year": 2010,
    "active_status": "Berlaku",
    "source_file": "PERMEN8VII2010.pdf",
    "source_path": "data\\pdf\\peraturan-menteri\\PERMEN8VII2010.pdf",
    "extractor": "pymupdf_pdfplumber",
    "chunk_level": "pasal",
    "pasal_id": "Pasal 1",
    "chunk_kind": "pasal",
    "chunk_index": 1,
    "token_count_estimate": 297,
    "bab": "BAB I",
    "bab_title": "Ketentuan Umum",
    "bagian": "",
    "bagian_title": "",
    "paragraf": "",
    "paragraf_title": ""
  }
}
```

---

## 5. Phase 4 - Skema Metadata Final

### 5.1 Tujuan

Phase 4 memastikan chunk sudah siap masuk ChromaDB dan aman dipakai untuk inference. File final harus bebas dari metadata `Unknown` untuk field inti, tidak membawa lampiran panjang yang nyangkut pada pasal terakhir, dan chunk terlalu panjang harus dipecah secara stabil.

Artefak resmi:

```text
data/processed_chunks_ringan_pasal_chroma_ready.json
```

### 5.2 Field Top-Level Wajib

| Field | Tipe | Keterangan |
|---|---|---|
| `id` | str | ID stabil chunk |
| `text` | str | Isi pasal/chunk yang sudah dibersihkan |
| `display_text` | str | Teks untuk prompt, debugging, dan tampilan user |
| `embedding_text` | str | Teks untuk embedding, berisi konteks regulasi + pasal |
| `citation_text` | str | Sitasi hukum siap tampil |
| `metadata` | dict | Metadata legal dan teknis |

### 5.3 Metadata Wajib untuk ChromaDB

| Field | Tipe | Keterangan |
|---|---|---|
| `regulation_type` | str | Jenis regulasi: UU, PP, Perpres, Permen, dst |
| `regulation_hierarchy` | int | Level hierarki untuk ranking |
| `nomor` | str | Nomor regulasi |
| `tentang` | str | Judul/topik regulasi |
| `year` | int | Tahun regulasi |
| `publication_year` | int | Tahun publikasi, dipakai lex posterior |
| `active_status` | str | Status berlaku |
| `source_file` | str | Nama file PDF sumber |
| `source_path` | str | Path file PDF sumber |
| `extractor` | str | Extractor yang dipakai |
| `chunk_level` | str | `pasal` untuk pipeline saat ini |
| `pasal_id` | str | ID pasal, misal `Pasal 1` |
| `chunk_kind` | str | `pasal` atau `pasal_split` |
| `chunk_index` | int | Urutan chunk dalam dokumen |
| `token_count_estimate` | int | Estimasi token |
| `bab` | str | BAB jika terdeteksi |
| `bab_title` | str | Judul BAB jika terdeteksi |
| `bagian` | str | Bagian jika terdeteksi |
| `bagian_title` | str | Judul bagian jika terdeteksi |
| `paragraf` | str | Paragraf jika terdeteksi |
| `paragraf_title` | str | Judul paragraf jika terdeteksi |

### 5.4 Validasi Sebelum Ingestion

```python
def validate_chroma_ready_chunk(chunk: dict) -> list[str]:
    errors = []
    required_top = ["id", "text", "display_text", "embedding_text", "citation_text", "metadata"]
    for field in required_top:
        if not chunk.get(field):
            errors.append(f"top-level field kosong: {field}")

    meta = chunk.get("metadata", {})
    required_meta = [
        "regulation_type", "nomor", "tentang", "year", "publication_year",
        "active_status", "source_file", "pasal_id", "chunk_kind", "chunk_index",
        "regulation_hierarchy",
    ]
    for field in required_meta:
        if meta.get(field) in (None, "", "Unknown"):
            errors.append(f"metadata field tidak valid: {field}")

    if int(meta.get("year") or 0) <= 0:
        errors.append("year tidak valid")
    if int(meta.get("publication_year") or 0) <= 0:
        errors.append("publication_year tidak valid")
    if meta.get("active_status") not in ("Berlaku", "Tidak Berlaku", "Sebagian Berlaku"):
        errors.append("active_status tidak valid")

    return errors
```

### 5.5 Finalizer Rule

`finalize_chunks_for_chroma.py` melakukan:

- repair metadata dari raw text dan nama file;
- rebuild `display_text`, `embedding_text`, dan `citation_text`;
- trim lampiran yang ikut ke pasal terakhir;
- quarantine sumber yang terlalu noise;
- split pasal panjang menjadi `pasal_split`;
- generate report kualitas.

---

## 6. Phase 5 - Embedding Pipeline

### 6.1 Model Sesuai Proposal

Proposal menggunakan embedding model open-source lokal agar data hukum dan query pengguna tetap on-premise.

| Komponen | Pilihan Production Prototype |
|---|---|
| Embedding model | `intfloat/multilingual-e5-base` |
| Vector dimension | 768 |
| Query prefix | `query: ...` |
| Passage prefix | `passage: ...` |
| Input chunk field | `embedding_text` |
| Display/context field | `display_text` |
| Citation field | `citation_text` |

Alasan pemilihan `multilingual-e5-base`:
- sesuai proposal sebagai open-source local embedding model;
- cukup ringan untuk laptop/GPU consumer;
- mendukung Bahasa Indonesia dengan kualitas multilingual yang stabil;
- cocok dengan ChromaDB cosine similarity.

### 6.2 Artefak Input Phase 5

Phase 5 tidak membaca output preprocessing lama. Input resmi adalah:

```text
data/processed_chunks_ringan_pasal_chroma_ready.json
```

Field minimal tiap chunk:

```json
{
  "id": "...",
  "text": "isi pasal mentah yang sudah dibersihkan",
  "display_text": "teks lengkap untuk prompt/context",
  "embedding_text": "teks yang dipakai untuk embedding",
  "citation_text": "format sitasi hukum siap tampil",
  "metadata": {
    "regulation_type": "Peraturan Pemerintah",
    "nomor": "35",
    "year": 2021,
    "publication_year": 2021,
    "active_status": "Berlaku",
    "pasal_id": "Pasal 156",
    "bab": "BAB ...",
    "source_file": "...pdf"
  }
}
```

### 6.3 Implementasi Embedding

```python
from sentence_transformers import SentenceTransformer
import torch

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE)

def embed_passages(chunks: list[dict], batch_size: int = 64) -> list[list[float]]:
    texts = ["passage: " + c["embedding_text"] for c in chunks]
    return embedding_model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()

def embed_query(query: str) -> list[float]:
    return embedding_model.encode(
        ["query: " + query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0].tolist()
```

---

## 7. Phase 6 - Vector Database (ChromaDB)

### 7.1 Alasan ChromaDB

Sesuai proposal, vector database lokal dapat memakai ChromaDB atau FAISS. Implementasi production prototype ini menetapkan **ChromaDB** sebagai vector database utama karena:

- mudah dijalankan lokal tanpa service tambahan;
- persistent di folder project;
- mendukung metadata per chunk;
- cukup untuk skala final project dan eksperimen retrieval;
- sederhana untuk deployment akademik on-premise.

### 7.2 Struktur Storage

```text
data/chroma_db/
  collection: hukum_ketenagakerjaan

data/bm25_index.pkl
  BM25 lexical index untuk exact keyword match
```

Chroma menyimpan:
- `ids`: `chunk.id`
- `documents`: `display_text`
- `embeddings`: vector dari `embedding_text`
- `metadatas`: metadata legal + `citation_text`

BM25 menyimpan indeks lexical dari gabungan `display_text` dan `embedding_text`.

### 7.3 Implementasi ChromaDB

```python
import chromadb
import json
from pathlib import Path

CHROMA_DB_DIR = Path("data/chroma_db")
COLLECTION_NAME = "hukum_ketenagakerjaan"
DATA_PATH = Path("data/processed_chunks_ringan_pasal_chroma_ready.json")

client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

# Rebuild collection agar tidak bercampur dengan schema lama.
try:
    client.delete_collection(COLLECTION_NAME)
except Exception:
    pass

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine", "schema": "ringan_pasal_chroma_ready_v1"},
)

def normalize_metadata(chunk: dict) -> dict:
    meta = dict(chunk.get("metadata", {}))
    meta["chunk_id"] = chunk.get("id", "")
    meta["citation_text"] = chunk.get("citation_text", "")

    safe = {}
    for key, value in meta.items():
        if value is None:
            safe[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe

chunks = json.loads(DATA_PATH.read_text(encoding="utf-8"))
embeddings = embed_passages(chunks)

collection.add(
    ids=[c["id"] for c in chunks],
    embeddings=embeddings,
    documents=[c["display_text"] for c in chunks],
    metadatas=[normalize_metadata(c) for c in chunks],
)
```

---

## 8. Phase 7 - Retrieval and Lex Posterior Ranking

### 8.1 Tujuan

Retrieval harus mencari pasal relevan dari deskripsi kasus pengguna, lalu memprioritaskan aturan terbaru jika ada konflik norma. Ini langsung mengikuti novelty proposal: **Hierarchy-Based Metadata Filtering Mechanism** dan prinsip **lex posterior derogat legi priori**.

### 8.2 Arsitektur Retrieval

```text
User Case Description
    |
    +--> Dense Retrieval: Chroma cosine similarity
    |
    +--> Lexical Retrieval: BM25 exact term matching
    |
    +--> RRF Fusion
    |
    +--> Metadata-aware ranking
         - active_status = Berlaku diprioritaskan
         - publication_year lebih baru diprioritaskan
         - hierarchy lebih tinggi diprioritaskan
         - duplicate pasal dibuang
```

### 8.3 Dense + BM25 Retrieval

```python
import pickle
import numpy as np
from rank_bm25 import BM25Okapi
import re

def tokenize_for_bm25(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())

def dense_search(query: str, fetch_k: int = 30) -> list[dict]:
    qvec = embed_query(query)
    res = collection.query(
        query_embeddings=[qvec],
        n_results=fetch_k,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for doc_id, doc, meta, dist in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        hits.append({
            "id": doc_id,
            "text": doc,
            "metadata": meta,
            "dense_distance": float(dist),
            "source": "dense",
        })
    return hits

def bm25_search(query: str, bm25: BM25Okapi, bm25_ids: list[str], fetch_k: int = 30) -> list[dict]:
    scores = bm25.get_scores(tokenize_for_bm25(query))
    order = np.argsort(scores)[::-1][:fetch_k]
    ids = [bm25_ids[i] for i in order if scores[i] > 0]

    got = collection.get(ids=ids, include=["documents", "metadatas"])
    lookup = {doc_id: (doc, meta) for doc_id, doc, meta in zip(
        got["ids"], got["documents"], got["metadatas"]
    )}

    hits = []
    for i in order:
        doc_id = bm25_ids[i]
        if scores[i] <= 0 or doc_id not in lookup:
            continue
        doc, meta = lookup[doc_id]
        hits.append({
            "id": doc_id,
            "text": doc,
            "metadata": meta,
            "bm25_score": float(scores[i]),
            "source": "bm25",
        })
    return hits
```

### 8.4 RRF Fusion dan Lex Posterior Scoring

```python
def rrf_fuse(result_sets: list[list[dict]], weights: list[float], rrf_k: int = 60) -> list[dict]:
    fused = {}
    for hits, weight in zip(result_sets, weights):
        for rank, hit in enumerate(hits, 1):
            item = fused.setdefault(hit["id"], {"score": 0.0, "hit": hit})
            item["score"] += weight / (rrf_k + rank)

    out = []
    for item in sorted(fused.values(), key=lambda x: x["score"], reverse=True):
        hit = item["hit"]
        hit["rrf_score"] = item["score"]
        out.append(hit)
    return out

def lex_posterior_score(hit: dict) -> float:
    meta = hit.get("metadata", {})
    score = hit.get("rrf_score", 0.0)

    year = int(meta.get("publication_year") or meta.get("year") or 0)
    hierarchy = int(meta.get("regulation_hierarchy") or 99)
    active = str(meta.get("active_status", "")).lower() == "berlaku"

    if active:
        score += 0.030
    score += min(max(year - 2000, 0), 40) * 0.001
    score += max(0, 6 - hierarchy) * 0.003

    return score

def dedupe_and_rank(hits: list[dict], k: int = 5) -> list[dict]:
    ranked = sorted(hits, key=lex_posterior_score, reverse=True)
    seen = set()
    out = []

    for hit in ranked:
        meta = hit.get("metadata", {})
        key = (meta.get("source_file", ""), meta.get("pasal_id", ""), meta.get("chunk_index", ""))
        if key in seen:
            continue
        seen.add(key)
        hit["final_score"] = lex_posterior_score(hit)
        out.append(hit)
        if len(out) >= k:
            break

    return out
```

### 8.5 Baseline untuk Evaluasi Novelty

Untuk membuktikan kontribusi lex posterior ranking, eksperimen wajib membandingkan dua mode:

| Mode | Deskripsi |
|---|---|
| Vanilla RAG | Chroma cosine similarity Top-k tanpa metadata ranking |
| Proposed RAG | Dense + BM25 + RRF + lex posterior metadata ranking |

Perbandingan ini sesuai Section 3.4 proposal: proposed system dibandingkan dengan minimal baseline tanpa lex posterior filtering rule.

---

## 9. Phase 8 - Post-Retrieval Deduplication and Neural Reranking

### 9.1 Deduplication Wajib

Karena long article bisa di-split dengan overlap, retrieval bisa mengembalikan potongan pasal yang sama berkali-kali. Sistem harus dedupe sebelum context injection.

```python
def dedupe_by_article(hits: list[dict], k: int = 5) -> list[dict]:
    seen = set()
    out = []
    for hit in hits:
        meta = hit.get("metadata", {})
        key = (
            meta.get("source_file", ""),
            meta.get("pasal_id", ""),
            meta.get("chunk_kind", ""),
            meta.get("chunk_index", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(hit)
        if len(out) >= k:
            break
    return out
```

### 9.2 Neural Reranker

Reranker neural aktif secara default setelah candidate retrieval dense + BM25 dan deduplication. Hasil evaluasi tetap sebaiknya membandingkan baseline dense-only, hybrid BM25, dan hybrid + reranker agar kontribusi setiap komponen terlihat jelas.

```python
USE_RERANKER = True

if USE_RERANKER:
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")
```

---

## 10. Phase 9 - Context Assembler

### 10.1 Tujuan

Context assembler mengubah daftar hasil retrieval menjadi konteks yang stabil untuk prompt LLM. Untuk prototype ini, context harus:

- memakai `display_text`, bukan raw `text` pendek;
- menyertakan ID referensi `[R1]`, `[R2]`, dst;
- menyertakan `citation_text` agar user bisa menelusuri dasar hukum;
- membatasi jumlah dokumen agar prompt tidak terlalu panjang;
- mempertahankan ranking lex posterior.

### 10.2 Implementasi

```python
def build_reference(meta: dict, idx: int) -> str:
    citation = meta.get("citation_text", "")
    if citation:
        return f"[R{idx}] {citation}"
    return f"[R{idx}] {meta.get('regulation_type', 'Aturan')} No. {meta.get('nomor', '')} Tahun {meta.get('year', '')}, {meta.get('pasal_id', '')}"

def assemble_context(docs: list[dict], max_docs: int = 5) -> str:
    parts = []
    for i, doc in enumerate(docs[:max_docs], 1):
        meta = doc["metadata"]
        parts.append(f"{build_reference(meta, i)}\n{doc['text']}")
    return "\n\n".join(parts)
```

---

## 11. Phase 10 - Answer Generation & Citation

### 11.1 Model Generator Sesuai Proposal

Generator menggunakan open-weight LLM lokal. Proposal menyebut Qwen3.5-27B-AWQ atau model Qwen quantized setara untuk inference consumer GPU.

| Komponen | Pilihan |
|---|---|
| LLM | Qwen open-weight instruct model |
| Quantization | 4-bit jika GPU tersedia |
| Prompting | Grounded QA dengan citation wajib |
| Citation format | `[R1]`, `[R2]`, dst |
| Hallucination control | jawab hanya dari context |

### 11.2 Prompt Generation

```python
SYSTEM_PROMPT = """Anda adalah asisten hukum ketenagakerjaan Indonesia.
Jawab hanya berdasarkan KONTEKS.
Jika konteks tidak cukup, katakan bahwa referensi tidak ditemukan.
Setiap klaim hukum wajib mencantumkan ID referensi seperti [R1] atau [R2].
Utamakan aturan dengan tahun terbaru jika ada konflik.
Gunakan bahasa Indonesia yang jelas dan teks polos."""

def build_prompt(query: str, docs: list[dict]) -> str:
    context = assemble_context(docs)
    return f"""{SYSTEM_PROMPT}

KONTEKS:
{context}

PERTANYAAN:
{query}

JAWABAN:"""
```

### 11.3 Citation Validation

```python
import re

def validate_reference_ids(answer: str, docs: list[dict]) -> dict:
    cited = sorted(set(int(x) for x in re.findall(r"\[R(\d+)\]", answer)))
    allowed = set(range(1, len(docs) + 1))
    invalid = [f"R{i}" for i in cited if i not in allowed]
    return {
        "cited": [f"R{i}" for i in cited],
        "invalid": invalid,
        "ok": not invalid,
    }
```

Validasi ini lebih stabil dibanding meminta model menulis nama peraturan secara bebas, karena nomor/tahun peraturan bisa typo ketika digenerate.

---

## 12. Phase 11 - Evaluasi Pipeline

### 12.1 Desain Eksperimen

Evaluasi mengikuti proposal:

1. Bandingkan **Vanilla RAG** vs **Proposed RAG**.
2. Gunakan skenario kasus ketenagakerjaan, terutama PHK, PKWT, outsourcing, dan upah.
3. Retrieval dinilai dengan Precision@k dan MRR.
4. Generation dinilai dengan Faithfulness dan Answer Relevancy, memakai RAGAS atau human evaluator.
5. Lakukan qualitative error analysis pada sampel terburuk.

### 12.2 Retrieval Metrics

```python
def precision_at_k(results: list[dict], expected: set[tuple], k: int = 5) -> float:
    top = results[:k]
    if not top:
        return 0.0
    hits = 0
    for r in top:
        meta = r["metadata"]
        key = (meta.get("source_file"), meta.get("pasal_id"))
        if key in expected:
            hits += 1
    return hits / k

def mrr_at_k(results: list[dict], expected: set[tuple], k: int = 5) -> float:
    for rank, r in enumerate(results[:k], 1):
        meta = r["metadata"]
        key = (meta.get("source_file"), meta.get("pasal_id"))
        if key in expected:
            return 1 / rank
    return 0.0
```

### 12.3 Generation Metrics

RAGAS metrics yang dipakai:

| Metric | Tujuan |
|---|---|
| Faithfulness | Mengukur apakah klaim jawaban didukung konteks |
| Answer Relevancy | Mengukur relevansi jawaban terhadap kasus |
| Context Precision | Mengukur presisi konteks retrieved |

Jika RAGAS tidak stabil karena keterbatasan local LLM, evaluasi dapat dilengkapi dengan penilaian manual expert/human evaluator sebagaimana proposal.

### 12.4 Qualitative Error Analysis

Untuk 10% sampel terburuk, catat sumber kegagalan:

- preprocessing/chunking error;
- pasal relevan tidak masuk top-k;
- metadata year/status salah;
- lex posterior ranking mengangkat aturan baru tapi tidak substantif;
- LLM gagal mengutip `[R#]` dengan benar;
- jawaban benar tetapi terlalu umum.

---

## 13. Roadmap Implementasi

```text
Phase 1-3: Extraction, cleaning, legal parser, pasal chunking
  - praproses_ringan_pasal.py
  - output: data/processed_chunks_ringan_pasal.json

Phase 4: Chroma-ready finalization
  - finalize_chunks_for_chroma.py
  - output: data/processed_chunks_ringan_pasal_chroma_ready.json

Phase 5-6: Embedding and ChromaDB ingestion
  - notebooks/01_Phase_4_5_6_Embedding_ChromaDB_BM25.ipynb
  - output: data/chroma_db + data/bm25_index.pkl

Phase 7-9: Retrieval, lex posterior ranking, context assembly
  - notebooks/02_Phase_7_8_9_HybridRetrieval_Reranker_Advanced.ipynb

Phase 10-11: Generation, citation validation, evaluation
  - notebooks/03_Phase_10_11_Generation_Eval.ipynb
```

---

## Catatan Kritis

1. **Jangan masukkan chunk mentah ke ChromaDB.** Gunakan `processed_chunks_ringan_pasal_chroma_ready.json`.
2. **Embedding wajib memakai `embedding_text`.** Field ini sudah membawa konteks regulasi dan pasal.
3. **Prompt wajib memakai `display_text`.** Field ini lebih cocok untuk dibaca LLM dan user.
4. **Referensi wajib dari `citation_text`.** Jangan meminta LLM mengarang nama peraturan dari memori.
5. **Lex posterior adalah heuristic prototype.** Untuk konflik norma yang rumit, hasil tetap perlu validasi hukum manusia.
6. **ChromaDB adalah vector database utama project ini.** Implementasi final proposal memakai ChromaDB secara konsisten.
7. **Baseline wajib ada.** Laporkan vanilla RAG vs proposed RAG agar novelty metadata ranking terukur.
