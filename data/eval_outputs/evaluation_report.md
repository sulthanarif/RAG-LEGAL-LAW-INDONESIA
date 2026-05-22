# Laporan Evaluasi RAG

Jumlah soal: 20
Model: Qwen/Qwen3.5-9B

## Ringkasan

- Rata-rata overall score: 0.840
- Median overall score: 0.852
- Rata-rata semantic similarity: 0.752
- Keyword coverage: 0.766
- Retrieval citation coverage: 0.848
- Retrieval citation hit rate: 1.000
- Retrieval law hit rate: 1.000
- Retrieval article hit rate: 1.000
- Article hit@3: 0.750
- Article MRR: 0.642
- Precision@3: 0.717
- Answer citation hit rate: 1.000
- Rata-rata latency detik: 42.57

## Lima Skor Terendah

| id   | topic                                    |   overall_score | keyword_missing                                                                                                                     |   expected_article_rank | top1_reference                                                                         |
|:-----|:-----------------------------------------|----------------:|:------------------------------------------------------------------------------------------------------------------------------------|------------------------:|:---------------------------------------------------------------------------------------|
| Q17  | PHK dan Jaminan Kehilangan Pekerjaan     |        0.709614 | kehilangan pekerjaan; 45 persen                                                                                                     |                       4 | Peraturan Pemerintah No. 37 Tahun 2021, Pasal 1 | pp_Nomor_37_Tahun_2021.pdf | Pasal 1 |
| Q19  | Pekerja migran Indonesia                 |        0.722506 | hak dan kewajiban; mekanisme pelindungan; penyelesaian sengketa; sertifikat kompetensi; paspor; visa kerja; pekerjaan upah perintah |                       1 | Peraturan Pemerintah No. 10 Tahun 2020, Pasal 3 | PP_10_2020.pdf | Pasal 3             |
| Q05  | PKWT                                     |        0.727523 | huruf latin; penafsiran; bahasa Indonesia berlaku                                                                                   |                       6 | Peraturan Pemerintah No. 35 Tahun 2021, Pasal 1 | PP Nomor 35 Tahun 2021.pdf | Pasal 1 |
| Q16  | PHK dan perselisihan hubungan industrial |        0.731582 | mencegah PHK; setengah kali pesangon; satu kali UPMK; Pengadilan Hubungan Industrial                                                |                       3 | Undang-Undang No. 13 Tahun 2003, Pasal 1 | UU_Nomor_13_Tahun_2003.pdf | Pasal 1        |
| Q09  | Lembur                                   |        0.756736 | lembur; perintah kerja lembur                                                                                                       |                       7 | Undang-Undang No. 13 Tahun 2003, Pasal 78 | UU_Nomor_13_Tahun_2003.pdf | Pasal 78      |

## File Penting

1. ../data/eval_outputs/rag_inference_results.csv
2. ../data/eval_outputs/evaluation_table.csv
3. ../data/eval_outputs/evaluation_summary.csv
4. ../data/eval_outputs/evaluation_by_topic.csv
5. ../data/eval_outputs/failure_review.csv
6. ../data/eval_outputs/evaluation_report.xlsx
7. ../data/eval_outputs/plots
