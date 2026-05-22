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
3. Jalankan notebook 04 kalau ingin chat/inference interaktif manual.
4. Jalankan notebook 05 untuk batch inference ground truth, evaluasi, tabel, failure review, dan plot otomatis.

Perintah push GitHub:

git add data/ground_truth_eval_20.csv notebooks/04_inference_chat_eval_no_thinking.ipynb notebooks/05_evaluasi_otomatis_plot.ipynb data/eval_outputs
git commit -m "add rag evaluation notebooks and ground truth"
git push origin main
