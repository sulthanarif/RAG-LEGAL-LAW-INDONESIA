# Laporan Evaluasi RAG

Jumlah soal: 20
Model: Qwen/Qwen3.5-9B

## Ringkasan

- Rata-rata overall score: 0.849
- Median overall score: 0.869
- Rata-rata LLM Judge score: 0.910
- Rata-rata semantic similarity: 0.700
- Keyword coverage: 0.743
- Retrieval citation coverage: 0.848
- Retrieval citation hit rate: 1.000
- Retrieval law hit rate: 1.000
- Retrieval article hit rate: 1.000
- Article hit@3: 0.750
- Article MRR: 0.642
- Precision@3: 0.717
- Answer citation hit rate: 1.000
- Rata-rata latency detik: 54.91

## Lima Skor Terendah

| id   | topic                                    |   overall_score |   llm_judge_score | keyword_missing                                                                                                                                             |   expected_article_rank | top1_reference                                                                                                  |
|:-----|:-----------------------------------------|----------------:|------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------:|:----------------------------------------------------------------------------------------------------------------|
| Q05  | PKWT                                     |        0.654932 |              0.7  | huruf latin; bahasa Indonesia; bahasa asing; penafsiran; bahasa Indonesia berlaku                                                                           |                       6 | Peraturan Pemerintah No. 35 Tahun 2021, Pasal 1 | PP Nomor 35 Tahun 2021.pdf | Pasal 1                          |
| Q16  | PHK dan perselisihan hubungan industrial |        0.715253 |              0.95 | mencegah PHK; setengah kali pesangon; satu kali UPMK; 30 hari kerja; mediasi; Pengadilan Hubungan Industrial                                                |                       3 | Undang-Undang No. 13 Tahun 2003, Pasal 1 | UU_Nomor_13_Tahun_2003.pdf | Pasal 1                                 |
| Q17  | PHK dan Jaminan Kehilangan Pekerjaan     |        0.718199 |              0.85 | kehilangan pekerjaan; 45 persen                                                                                                                             |                       4 | Peraturan Pemerintah No. 37 Tahun 2021, Pasal 1 | pp_Nomor_37_Tahun_2021.pdf | Pasal 1                          |
| Q20  | K3, narkotika, dan jaminan sosial        |        0.72913  |              0.85 | kebijakan tertulis; konsultasi pekerja; lapor Kepolisian; instansi ketenagakerjaan; alat pelindung diri; JKK; JKM; pemberi kerja wajib membayar hak pekerja |                       1 | Peraturan Menteri Ketenagakerjaan No. PER.11/MEN/VI/2005 Tahun 2005, Pasal 8 | PER-11-MEN-VI-2005.pdf | Pasal 8 |
| Q19  | Pekerja migran Indonesia                 |        0.731233 |              0.85 | hak dan kewajiban; mekanisme pelindungan; penyelesaian sengketa; sertifikat kompetensi; paspor; visa kerja; perjanjian penempatan; pekerjaan upah perintah  |                       1 | Peraturan Pemerintah No. 10 Tahun 2020, Pasal 3 | PP_10_2020.pdf | Pasal 3                                      |

## File Penting

1. ../data/eval_outputs/rag_inference_results.csv
2. ../data/eval_outputs/evaluation_table.csv
3. ../data/eval_outputs/evaluation_summary.csv
4. ../data/eval_outputs/evaluation_by_topic.csv
5. ../data/eval_outputs/failure_review.csv
6. ../data/eval_outputs/evaluation_report.xlsx
7. ../data/eval_outputs/plots
