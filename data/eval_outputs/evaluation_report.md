# Laporan Evaluasi RAG

Jumlah soal: 20
Model: Qwen/Qwen3.5-9B

## Ringkasan

- Rata-rata overall score: 0.847
- Rata-rata LLM Judge score: 0.845
- Rata-rata semantic similarity: 0.752
- Keyword coverage: 0.766
- Retrieval citation coverage: 0.848
- Retrieval article hit rate: 1.000
- Article MRR: 0.642
- Answer citation hit rate: 1.000
- Rata-rata latency detik: 41.96

## Lima Skor Terendah

| id   | topic                                    |   overall_score |   llm_judge_score | keyword_missing                                                                                                                     |   expected_article_rank | top1_reference                                                                         |
|:-----|:-----------------------------------------|----------------:|------------------:|:------------------------------------------------------------------------------------------------------------------------------------|------------------------:|:---------------------------------------------------------------------------------------|
| Q05  | PKWT                                     |        0.596114 |              0    | huruf latin; penafsiran; bahasa Indonesia berlaku                                                                                   |                       6 | Peraturan Pemerintah No. 35 Tahun 2021, Pasal 1 | PP Nomor 35 Tahun 2021.pdf | Pasal 1 |
| Q19  | Pekerja migran Indonesia                 |        0.718842 |              0.75 | hak dan kewajiban; mekanisme pelindungan; penyelesaian sengketa; sertifikat kompetensi; paspor; visa kerja; pekerjaan upah perintah |                       1 | Peraturan Pemerintah No. 10 Tahun 2020, Pasal 3 | PP_10_2020.pdf | Pasal 3             |
| Q09  | Lembur                                   |        0.73613  |              0.6  | lembur; perintah kerja lembur                                                                                                       |                       7 | Undang-Undang No. 13 Tahun 2003, Pasal 78 | UU_Nomor_13_Tahun_2003.pdf | Pasal 78      |
| Q17  | PHK dan Jaminan Kehilangan Pekerjaan     |        0.738259 |              0.85 | kehilangan pekerjaan; 45 persen                                                                                                     |                       4 | Peraturan Pemerintah No. 37 Tahun 2021, Pasal 1 | pp_Nomor_37_Tahun_2021.pdf | Pasal 1 |
| Q16  | PHK dan perselisihan hubungan industrial |        0.747727 |              0.85 | mencegah PHK; setengah kali pesangon; satu kali UPMK; Pengadilan Hubungan Industrial                                                |                       3 | Undang-Undang No. 13 Tahun 2003, Pasal 1 | UU_Nomor_13_Tahun_2003.pdf | Pasal 1        |

## File Penting

1. ../data/eval_outputs/rag_inference_results.csv
2. ../data/eval_outputs/evaluation_table.csv
3. ../data/eval_outputs/evaluation_summary.csv
4. ../data/eval_outputs/evaluation_by_topic.csv
5. ../data/eval_outputs/failure_review.csv
6. ../data/eval_outputs/evaluation_report.xlsx
7. ../data/eval_outputs/plots
