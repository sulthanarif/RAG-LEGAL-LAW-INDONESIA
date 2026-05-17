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

Untuk PaddleOCR GPU, paling aman pakai environment terpisah khusus OCR. Jangan campur Paddle GPU `cu118` dengan environment notebook RAG yang memakai Torch CUDA 12.x, karena library `nvidia-*` bisa bentrok dan membuat `import torch` gagal.

Environment notebook/RAG:

```powershell
python -m venv .venv-rag
.\.venv-rag\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Environment OCR GPU:

```powershell
python -m venv .venv-ocr
.\.venv-ocr\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip uninstall -y paddlepaddle paddlepaddle-gpu
python -m pip install --no-cache-dir --force-reinstall -r requirements-paddle-gpu.txt
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

Kalau `pip install -r requirements-paddle-gpu.txt` bilang sudah installed tetapi script masih error `Paddle yang terinstall CPU build`, berarti package CPU masih kebaca oleh Python. Jalankan uninstall dua package di atas, lalu reinstall dengan `--force-reinstall`.

Kalau setelah install Paddle GPU notebook gagal `import torch` dengan error `undefined symbol: ncclCommShrink`, berarti Paddle GPU dan Torch CUDA bentrok dalam satu environment. Solusinya pisahkan environment OCR dan RAG, atau reinstall Torch setelah pekerjaan OCR selesai.

Kalau `import sentence_transformers` gagal karena `torchcodec` atau FFmpeg, hapus `torchcodec` dan pakai versi `sentence-transformers` yang dipin di requirements:

```powershell
python -m pip uninstall -y torchcodec
python -m pip install --force-reinstall "sentence-transformers==3.4.1"
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

Tahap finalisasi juga menjalankan cleanup deterministik untuk typo OCR umum sebelum data masuk Chroma, misalnya `RAT{MAT` -> `RAHMAT`, `KNRTU` -> `KARTU`, `KEHI1,ANGAN` -> `KEHILANGAN`, dan normalisasi `Pasal Ii` -> `Pasal II`. Header display juga dipendekkan agar preview retrieval tidak penuh judul `tentang ...` yang panjang.

### 4. Opsional: Local LLM Typo Correction

Kalau typo OCR masih banyak, jalankan koreksi berbasis model kecil setelah finalisasi. Script ini membaca file Chroma-ready dan menulis file baru, sehingga artifact asli tidak ketimpa. Default-nya memakai model lokal via `transformers`, bukan API.

Test 10 chunk dulu di GPU RTX 3090:

```powershell
python .\llm_typo_correct_chroma_ready.py --provider transformers --model Qwen/Qwen2.5-7B-Instruct --device cuda --limit 10 --max-new-tokens 2048
```

Untuk model 7B, script otomatis load 4-bit agar lebih ringan di GPU 24GB. Kalau ingin mematikan perilaku ini, tambahkan `--no-auto-4bit-7b`.

Full run lokal:

```powershell
python .\llm_typo_correct_chroma_ready.py --provider transformers --model Qwen/Qwen2.5-7B-Instruct --device cuda --max-new-tokens 2048
```

Kalau pakai model selain 7B dan VRAM mepet, pakai 4-bit eksplisit:

```powershell
python .\llm_typo_correct_chroma_ready.py --provider transformers --model Qwen/Qwen2.5-7B-Instruct --device cuda --load-in-4bit --max-new-tokens 2048
```

Output default:

```text
data/processed_chunks_ringan_pasal_chroma_ready_typo_corrected.json
data/llm_typo_correction_cache.jsonl
```

Setelah itu ingestion Chroma bisa diarahkan ke file `*_typo_corrected.json`.

Catatan: `--max-new-tokens` default script memang menerima nilai besar, tetapi untuk chunk final saat ini rata-rata sudah dipotong sampai sekitar 450 kata. Untuk koreksi typo, `2048` atau `4096` biasanya jauh lebih cepat dan cukup. Nilai `100000` hampir pasti lambat/OOM/ditolak karena Qwen 2.5 7B tidak punya output budget sebesar itu dalam praktik.

Alternatif lokal via Ollama:

```powershell
ollama pull qwen2.5:7b-instruct
python .\llm_typo_correct_chroma_ready.py --provider ollama --model qwen2.5:7b-instruct --max-new-tokens 2048
```

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
- `citation_text`: sitasi singkat untuk jawaban, formatnya cukup `Jenis Peraturan No. X Tahun Y, Pasal Z`.
- `metadata`: filter dan ranking retrieval.

## ChromaDB dan Notebook

Gunakan file:

```text
data/processed_chunks_ringan_pasal_chroma_ready.json
```

Untuk notebook yang berada di folder `notebooks/`, path yang dipakai adalah:

```text
../data/processed_chunks_ringan_pasal_chroma_ready.json
../data/chroma_db
../data/bm25_index.pkl
```

Untuk notebook yang berada di root project, path tetap memakai `data/...`.

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
python -m pip uninstall -y paddlepaddle paddlepaddle-gpu
python -m pip install --no-cache-dir --force-reinstall -r requirements-paddle-gpu.txt

# 2. Extract dan chunk PDF
python .\praproses_ringan_pasal.py

# 3. OCR file yang butuh OCR
python .\paddle_ocr_extract.py --device gpu:0

# 4. Finalisasi untuk ChromaDB
python .\finalize_chunks_for_chroma.py
```

Setelah tahap finalisasi selesai, data siap dimasukkan ke ChromaDB. Kalau memakai LLM typo correction, gunakan file `data/processed_chunks_ringan_pasal_chroma_ready_typo_corrected.json` sebagai input ingestion.

## GitHub Notes

File/folder berikut sengaja tidak di-commit:

- PDF mentah.
- Raw OCR/extracted text.
- JSON chunks hasil generate.
- ChromaDB index.
- model/checkpoint besar.
- folder backup dan hasil eksperimen.

Artinya setelah clone repo, pengguna perlu menjalankan pipeline preprocessing ulang atau menyediakan artifact data sendiri.
