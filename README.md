# Legal RAG Indonesia - Employment Law

Prototype sistem rekomendasi pasal hukum ketenagakerjaan Indonesia berbasis Retrieval-Augmented Generation (RAG). Pipeline ini memakai structural chunking per pasal, OCR fallback untuk PDF bermasalah, embedding lokal, dan ChromaDB sebagai vector database.

## Tujuan

Project ini dibuat untuk final project:

**Case-Based Employment Law Article Recommendation System using Retrieval-Augmented Generation**

Fokus sistem:

- Menjawab kasus ketenagakerjaan berdasarkan rujukan pasal.
- Mengurangi hallucination dengan konteks hasil retrieval.
- Memakai metadata hukum seperti `publication_year`, `active_status`, dan struktur `BAB/Pasal`.
- Menyiapkan data agar mudah masuk ChromaDB.

## Struktur Penting

```text
.
├── data/
│   ├── pdf/                         # PDF mentah, di-ignore dari Git
│   ├── pdf_butuh_ocr/               # PDF yang perlu OCR, di-ignore dari Git
│   ├── raw_text_ringan/             # cache ekstraksi teks, di-ignore dari Git
│   ├── raw_text_paddle/             # cache OCR Paddle, di-ignore dari Git
│   ├── processed_chunks_ringan_pasal.json
│   └── processed_chunks_ringan_pasal_chroma_ready.json
├── notebooks/
├── praproses_ringan_pasal.py
├── paddle_ocr_extract.py
├── finalize_chunks_for_chroma.py
├── pipeline_legal_rag_indonesia.md
├── requirements.txt
└── requirements-paddle-gpu.txt
```

File besar dan artifact hasil generate sudah masuk `.gitignore`, jadi tidak ikut ke GitHub.

## Instalasi

Disarankan pakai environment baru.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Untuk PaddleOCR CPU, cukup install `requirements.txt`.

Untuk PaddleOCR GPU, install base dulu lalu override Paddle CPU ke GPU:

```powershell
python -m pip uninstall -y paddlepaddle
python -m pip install -r requirements-paddle-gpu.txt
```

Cek Paddle GPU:

```powershell
python -c "import paddle; print(paddle.device.is_compiled_with_cuda()); print(paddle.device.get_device())"
```

Output ideal:

```text
True
gpu:0
```

## Alur Preprocessing

Pipeline preprocessing terdiri dari 3 tahap:

1. Ekstraksi dan chunking ringan per pasal.
2. OCR fallback untuk PDF yang hasil ekstraksinya rusak.
3. Finalisasi JSON agar siap masuk ChromaDB.

### 1. Ekstraksi Ringan Per Pasal

Jalankan:

```powershell
python .\praproses_ringan_pasal.py
```

Output utama:

```text
data/processed_chunks_ringan_pasal.json
data/processed_chunks_ringan_pasal.failures.json
data/processing_report.csv
```

PDF yang kualitas ekstraksinya buruk akan masuk/tercatat di:

```text
data/pdf_butuh_ocr/
```

### 2. OCR Dengan PaddleOCR

Untuk scan semua file di `data/pdf_butuh_ocr` dan merge ke JSON utama:

```powershell
python .\paddle_ocr_extract.py --device gpu:0
```

Kalau tidak punya Paddle GPU:

```powershell
python .\paddle_ocr_extract.py --device cpu
```

Default `--device auto` akan memakai GPU jika Paddle GPU build tersedia.

Command aman untuk test dulu tanpa merge:

```powershell
python .\paddle_ocr_extract.py --limit 1 --max-pages 5 --no-merge --device gpu:0
```

OCR satu file saja:

```powershell
python .\paddle_ocr_extract.py --pdf .\data\pdf_butuh_ocr\5PP2021.pdf --device gpu:0
```

Output OCR:

```text
data/raw_text_paddle/
data/processed_chunks_paddle_ocr.json
data/paddle_ocr_report.json
```

Catatan: OCR full bisa lama karena beberapa PDF punya ratusan sampai ribuan halaman.

### 3. Finalisasi Chroma-Ready

Setelah preprocessing dan OCR selesai:

```powershell
python .\finalize_chunks_for_chroma.py
```

Output:

```text
data/processed_chunks_ringan_pasal_chroma_ready.json
data/processed_chunks_ringan_pasal_chroma_report.json
```

File `*_chroma_ready.json` adalah file final yang dipakai notebook ingestion dan retrieval.

## Format Chunk Final

Setiap chunk final berisi field utama:

```json
{
  "id": "...",
  "text": "...",
  "display_text": "...",
  "embedding_text": "...",
  "citation_text": "...",
  "metadata": {
    "regulation_type": "...",
    "nomor": "...",
    "year": 2021,
    "publication_year": 2021,
    "active_status": "Berlaku",
    "source_file": "...",
    "pasal_id": "Pasal ...",
    "bab": "...",
    "bab_title": "...",
    "quality_status": "ok"
  }
}
```

Pemakaian:

- `embedding_text`: teks untuk embedding.
- `display_text`: teks untuk ditampilkan ke user.
- `citation_text`: sitasi singkat untuk jawaban.
- `metadata`: filter dan ranking retrieval.

## ChromaDB dan Notebook

Gunakan file:

```text
data/processed_chunks_ringan_pasal_chroma_ready.json
```

Notebook utama:

```text
vector_pipeline.ipynb
rag_chat.ipynb
rag_inference_evaluation.ipynb
notebooks/01_Phase_4_5_6_Embedding_ChromaDB_BM25.ipynb
notebooks/02_Phase_7_8_9_HybridRetrieval_Reranker_Advanced.ipynb
notebooks/03_Phase_10_11_Generation_Eval.ipynb
```

Vector database yang dipakai adalah **ChromaDB**, bukan Qdrant.

## Urutan Full Pipeline

```powershell
# 1. Install dependencies
python -m pip install -r requirements.txt

# Optional GPU PaddleOCR
python -m pip uninstall -y paddlepaddle
python -m pip install -r requirements-paddle-gpu.txt

# 2. Extract dan chunk PDF
python .\praproses_ringan_pasal.py

# 3. OCR file yang butuh OCR
python .\paddle_ocr_extract.py --device gpu:0

# 4. Finalisasi untuk ChromaDB
python .\finalize_chunks_for_chroma.py
```

Setelah tahap 4 selesai, data siap dimasukkan ke ChromaDB.

## GitHub Notes

File/folder berikut sengaja tidak di-commit:

- PDF mentah.
- Raw OCR/extracted text.
- JSON chunks hasil generate.
- ChromaDB index.
- model/checkpoint besar.
- folder backup dan hasil eksperimen.

Artinya setelah clone repo, pengguna perlu menjalankan pipeline preprocessing ulang atau menyediakan artifact data sendiri.

