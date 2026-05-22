# Paket Evaluasi RAG Hukum Ketenagakerjaan

Isi paket:

1. ground_truth_eval_20.csv
2. 04_inference_chat_eval_no_thinking.ipynb
3. 05_evaluasi_otomatis_plot.ipynb

Urutan pakai:

1. Salin semua file ke root project RAG kamu.
2. Pastikan notebook indexing sebelumnya sudah menghasilkan data berikut.
   data/processed_chunks_ringan_pasal_chroma_ready.json
   data/chroma_db
   data/bm25_index.pkl
3. Jalankan notebook 04 untuk menghasilkan eval_outputs/rag_inference_results.csv.
4. Jalankan notebook 05 untuk membuat tabel evaluasi dan plot.

Perintah push GitHub:

git add ground_truth_eval_20.csv 04_inference_chat_eval_no_thinking.ipynb 05_evaluasi_otomatis_plot.ipynb eval_outputs
git commit -m "add rag evaluation notebooks and ground truth"
git push origin main
