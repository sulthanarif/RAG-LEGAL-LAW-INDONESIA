# Legal RAG Indonesia - Employment Law

Prototype sistem rekomendasi pasal hukum ketenagakerjaan Indonesia berbasis Retrieval-Augmented Generation (RAG). Pipeline ini memakai structural chunking per pasal, OCR fallback, cleanup typo OCR, embedding lokal, ChromaDB, hybrid retrieval, dan inference berbasis konteks.

Project ini disusun untuk final project:

**Case-Based Employment Law Article Recommendation System using Retrieval-Augmented Generation**

## Fokus Sistem

- Menjawab kasus ketenagakerjaan berdasarkan rujukan pasal.
- Mengurangi hallucination dengan retrieval dan sitasi eksplisit.
- Memakai metadata hukum seperti `publication_year`, `active_status`, `BAB`, dan `Pasal`.
- Memakai **ChromaDB**, bukan Qdrant.
- Menyediakan pipeline dari PDF mentah sampai inference/evaluation notebook.

## Struktur Penting

```text
.
|-- data/
|   |-- pdf/                         # PDF mentah, di-ignore dari Git
|   |-- pdf_butuh_ocr/               # PDF yang perlu OCR, di-ignore dari Git
|   |-- raw_text_ringan/             # cache ekstraksi teks, di-ignore dari Git
|   |-- raw_text_paddle/             # cache OCR Paddle, di-ignore dari Git
|   |-- processed_chunks_ringan_pasal.json
|   `-- processed_chunks_ringan_pasal_chroma_ready.json
|-- notebooks/
|   |-- 01_Phase_4_5_6_Embedding_ChromaDB_BM25.ipynb
|   |-- 02_Phase_7_8_9_HybridRetrieval_Reranker_Advanced.ipynb
|   `-- 03_Phase_10_11_Generation_Eval.ipynb
|-- praproses_ringan_pasal.py
|-- paddle_ocr_extract.py
|-- finalize_chunks_for_chroma.py
|-- llm_typo_correct_chroma_ready.py
|-- pipeline_legal_rag_indonesia.md
|-- requirements.txt
`-- requirements-paddle-gpu.txt
```

File besar dan artifact hasil generate sudah masuk `.gitignore`, jadi tidak ikut ke GitHub.

## Compute Requirements

Minimal untuk preprocessing teks biasa:

- CPU laptop/server biasa cukup.
- RAM 8-16 GB cukup untuk ekstraksi ringan.
- Tidak perlu GPU untuk `praproses_ringan_pasal.py` dan `finalize_chunks_for_chroma.py`.

OCR Paddle:

- CPU bisa, tapi lambat untuk ratusan/ribuan halaman.
- GPU disarankan untuk `paddle_ocr_extract.py`.
- Contoh tested: RTX 3090 24GB.

Embedding Chroma:

- GPU mempercepat embedding `intfloat/multilingual-e5-base`.
- CPU bisa, tapi lebih lambat.
- ChromaDB disimpan lokal di `data/chroma_db`.

LLM typo correction:

- Opsional.
- Untuk `Qwen/Qwen2.5-7B-Instruct`, RTX 3090/4090 24GB disarankan.
- Script otomatis load 4-bit untuk model 7B.
- Pakai `--only-suspicious-text` agar hanya chunk yang mengandung pola typo OCR dikirim ke LLM.

Inference / generation:

- Retrieval dan prompt assembly bisa jalan CPU/GPU.
- Generator LLM lokal butuh GPU sesuai model yang dipakai.
- Untuk notebook inference, gunakan environment RAG/Torch yang bersih.

Streamlit/demo app:

- UI Streamlit/Flask ringan bisa jalan CPU.
- Retrieval Chroma + BM25 ringan.
- Kalau app memanggil LLM lokal, compute mengikuti model generator.
- Kalau app hanya retrieval/citation preview, GPU tidak wajib.

## Warnings

- Jangan campur **Paddle GPU cu118** dan **Torch CUDA 12.x** di environment yang sama. Ini bisa membuat `import torch` error seperti `undefined symbol: ncclCommShrink`.
- Disarankan pisahkan environment:
  - environment OCR Paddle GPU
  - environment RAG notebook / Torch
- Jangan commit PDF, Chroma index, raw OCR text, atau JSON chunk besar.
- Jangan menjalankan `git reset --hard` atau `git clean -fdx` di server rental kalau file hasil typo correction belum dibackup.
- `git pull` normal tidak menghapus file ignored seperti `data/processed_chunks_ringan_pasal_chroma_ready_typo_corrected.json`, tapi backup tetap disarankan.
- LLM typo correction bisa lama. Untuk 7B, proses semua chunk bisa berjam-jam. Gunakan `--only-suspicious-text`.
- Nilai `--max-new-tokens 100000` tidak realistis untuk model 7B. Default script sekarang `2048`.
- Hasil OCR/LLM correction tetap perlu spot-check. Jangan over-correct istilah legal seperti `Tk. I`, `Strata I`, atau `Baperjakat`.

## Instalasi

Environment RAG/notebook:

```powershell
python -m venv .venv-rag
.\.venv-rag\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux:

```bash
python -m venv .venv-rag
source .venv-rag/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Kalau `import sentence_transformers` gagal karena `torchcodec` atau FFmpeg:

```bash
python -m pip uninstall -y torchcodec
python -m pip install --force-reinstall "sentence-transformers==3.4.1"
```

Environment OCR GPU:

```bash
python -m venv .venv-ocr
source .venv-ocr/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip uninstall -y paddlepaddle paddlepaddle-gpu
python -m pip install --no-cache-dir --force-reinstall -r requirements-paddle-gpu.txt
```

Cek Paddle GPU:

```bash
python - <<'PY'
import paddle
print(paddle.__version__)
print(paddle.device.is_compiled_with_cuda())
print(paddle.device.get_device())
PY
```

Output ideal:

```text
True
gpu:0
```

## Full Pipeline

### 1. Ekstraksi Dan Chunking Ringan

```bash
python praproses_ringan_pasal.py
```

Output:

```text
data/processed_chunks_ringan_pasal.json
data/processed_chunks_ringan_pasal.failures.json
data/processing_report.csv
```

PDF yang kualitas ekstraksinya buruk akan masuk/tercatat di:

```text
data/pdf_butuh_ocr/
```

### 2. OCR Fallback Dengan PaddleOCR

Scan semua file di `data/pdf_butuh_ocr` dan merge ke JSON utama:

```bash
python paddle_ocr_extract.py --device gpu:0
```

Test aman tanpa merge:

```bash
python paddle_ocr_extract.py --limit 1 --max-pages 5 --no-merge --device gpu:0
```

OCR satu file:

```bash
python paddle_ocr_extract.py --pdf data/pdf_butuh_ocr/5PP2021.pdf --device gpu:0
```

Output:

```text
data/raw_text_paddle/
data/processed_chunks_paddle_ocr.json
data/paddle_ocr_report.json
```

### 3. Finalisasi Chroma-Ready

```bash
python finalize_chunks_for_chroma.py
```

Output:

```text
data/processed_chunks_ringan_pasal_chroma_ready.json
data/processed_chunks_ringan_pasal_chroma_report.json
```

Tahap ini:

- memperbaiki metadata nomor/tahun jika bisa,
- memotong chunk terlalu panjang,
- membersihkan typo OCR deterministik,
- membuat `display_text`, `embedding_text`, dan `citation_text`,
- memastikan chunk siap masuk ChromaDB.

Format `citation_text` dibuat ringkas:

```text
Jenis Peraturan No. X Tahun Y, Pasal Z
```

### 4. Opsional: Local LLM Typo Correction

Jalankan setelah finalisasi jika OCR masih banyak typo.

Test 10 chunk:

```bash
python llm_typo_correct_chroma_ready.py \
  --provider transformers \
  --model Qwen/Qwen2.5-7B-Instruct \
  --device cuda \
  --only-suspicious-text \
  --limit 10
```

Full run:

```bash
python llm_typo_correct_chroma_ready.py \
  --provider transformers \
  --model Qwen/Qwen2.5-7B-Instruct \
  --device cuda \
  --only-suspicious-text
```

Output:

```text
data/processed_chunks_ringan_pasal_chroma_ready_typo_corrected.json
data/llm_typo_correction_cache.jsonl
```

Notebook batch 1 akan merge file typo-corrected ke file canonical secara default jika file tersebut ada:

```text
data/processed_chunks_ringan_pasal_chroma_ready_typo_corrected.json
-> data/processed_chunks_ringan_pasal_chroma_ready.json
```

Backup base dibuat sekali:

```text
data/processed_chunks_ringan_pasal_chroma_ready.before_typo_corrected.json
```

### 5. Embedding Dan Ingestion ChromaDB

Notebook:

```text
notebooks/01_Phase_4_5_6_Embedding_ChromaDB_BM25.ipynb
```

Fungsi:

- load `processed_chunks_ringan_pasal_chroma_ready.json`,
- merge typo-corrected artifact jika tersedia,
- embed `embedding_text` dengan `intfloat/multilingual-e5-base`,
- rebuild collection Chroma,
- build BM25 index.

Output:

```text
data/chroma_db/
data/bm25_index.pkl
```

Path di notebook folder `notebooks/`:

```python
DATA_PATH = Path("../data/processed_chunks_ringan_pasal_chroma_ready.json")
CHROMA_DB_DIR = Path("../data/chroma_db")
BM25_PATH = Path("../data/bm25_index.pkl")
```

### 6. Hybrid Retrieval

Notebook:

```text
notebooks/02_Phase_7_8_9_HybridRetrieval_Reranker_Advanced.ipynb
```

Fungsi:

- dense retrieval dari ChromaDB,
- sparse retrieval dari BM25,
- fusion/reranking,
- lex posterior scoring berbasis metadata tahun dan hirarki,
- preview referensi ringkas.

### 7. Inference Dan Evaluation

Notebook:

```text
notebooks/03_Phase_10_11_Generation_Eval.ipynb
rag_inference_evaluation.ipynb
rag_chat.ipynb
```

Fungsi:

- retrieve context,
- assemble prompt dengan referensi `[R1]`, `[R2]`, dst.,
- generate jawaban berbasis konteks,
- tampilkan sitasi,
- evaluasi faithfulness/context metrics jika dataset evaluasi tersedia.

Untuk notebook root, path tetap:

```python
DATA_PATH = Path("data/processed_chunks_ringan_pasal_chroma_ready.json")
```

Untuk notebook di folder `notebooks/`, path memakai `../data/...`.

## Format Chunk Final

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

- `text`: isi pasal bersih.
- `display_text`: teks untuk ditampilkan dan disimpan sebagai Chroma document.
- `embedding_text`: teks untuk embedding.
- `citation_text`: sitasi singkat.
- `metadata`: filter/ranking retrieval.

## GitHub Notes

File/folder berikut sengaja tidak di-commit:

- PDF mentah.
- Raw OCR/extracted text.
- JSON chunks hasil generate.
- JSON typo-corrected hasil LLM.
- ChromaDB index.
- model/checkpoint besar.
- folder backup dan hasil eksperimen.

Setelah clone repo, pengguna perlu menjalankan pipeline preprocessing ulang atau menyediakan artifact data sendiri.

