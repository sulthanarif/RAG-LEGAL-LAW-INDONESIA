# Combined RAG Evaluation Report

Report ini menggabungkan evaluasi otomatis internal, local LLM-as-a-judge, dan external LLM-as-a-judge dari GPT5.5 serta Gemini3.5.

## Dataset

- Jumlah pertanyaan: 20
- Model jawaban: Qwen/Qwen3.5-9B
- External judge files:
  - `external_llm_as_a_judge_GPT5.5.xlsx`
  - `external_llm_as_a_judge_Gemini3.5.xlsx`

## File Output Gabungan

- `evaluation_table_with_external_judges.csv`: tabel utama gabungan semua metrik dan external judge.
- `evaluation_summary_with_external_judges.csv`: ringkasan mean/median/min/max semua metrik penting.
- `evaluation_by_topic_with_external_judges.csv`: agregasi performa per topik.
- `external_judge_comparison.csv`: perbandingan local judge, GPT5.5, Gemini3.5, gap internal-eksternal, dan alasan judge.
- `evaluation_report_with_external_judges.xlsx`: workbook gabungan dengan sheet detail, summary, by topic, external judges, dan failure review.

## Ringkasan Skor

| Metrik | Nilai |
|---|---:|
| Internal overall mean | 0.849 |
| Local LLM judge mean | 0.910 |
| GPT5.5 judge mean | 0.665 |
| Gemini3.5 judge mean | 0.820 |
| External judge mean | 0.742 |
| Combined internal+external mean | 0.796 |
| Retrieval citation coverage | 0.848 |
| Keyword coverage | 0.743 |
| Mean external disagreement | 0.185 |

## Interpretasi Metrik

- `overall_score`: skor otomatis internal dari semantic similarity, keyword coverage, retrieval citation coverage, retrieval/article hit, dan metrik pendukung lain.
- `llm_judge_score`: skor local LLM-as-a-judge yang dihitung di notebook evaluasi.
- `external_judge_gpt55_score` dan `external_judge_gemini35_score`: skor external judge yang dinormalisasi dari 0-10 menjadi 0-1.
- `external_judge_mean_score`: rata-rata dua external judge.
- `external_judge_disagreement`: selisih skor tertinggi dan terendah antar external judge; makin tinggi berarti penilaian antar judge tidak stabil.
- `internal_external_gap`: `overall_score - external_judge_mean_score`; positif berarti skor otomatis internal lebih optimis dari external judge.
- `overall_score_with_external`: rata-rata 50% `overall_score` dan 50% `external_judge_mean_score`.

## Lima Skor Gabungan Terendah

| id   | topic                                    |   overall_score |   external_judge_mean_score |   overall_score_with_external | combined_quality_label   |
|:-----|:-----------------------------------------|----------------:|----------------------------:|------------------------------:|:-------------------------|
| Q05  | PKWT                                     |        0.654932 |                        0.4  |                      0.527466 | fair                     |
| Q20  | K3, narkotika, dan jaminan sosial        |        0.72913  |                        0.4  |                      0.564565 | fair                     |
| Q17  | PHK dan Jaminan Kehilangan Pekerjaan     |        0.718199 |                        0.45 |                      0.584099 | fair                     |
| Q01  | PKWT                                     |        0.868556 |                        0.35 |                      0.609278 | fair                     |
| Q16  | PHK dan perselisihan hubungan industrial |        0.715253 |                        0.55 |                      0.632627 | fair                     |

## Lima Skor Gabungan Tertinggi

| id   | topic       |   overall_score |   external_judge_mean_score |   overall_score_with_external | combined_quality_label   |
|:-----|:------------|----------------:|----------------------------:|------------------------------:|:-------------------------|
| Q08  | Waktu kerja |        0.974708 |                        0.95 |                      0.962354 | excellent                |
| Q12  | PHK         |        0.938027 |                        0.95 |                      0.944014 | excellent                |
| Q02  | PKWT        |        0.928664 |                        0.95 |                      0.939332 | excellent                |
| Q04  | PKWT harian |        0.9059   |                        0.95 |                      0.92795  | excellent                |
| Q03  | PKWT        |        0.943558 |                        0.9  |                      0.921779 | excellent                |

## Plot Evaluasi

- `plots/01_overall_score_per_question.png`: Skor akhir internal per pertanyaan. Dipakai untuk melihat soal mana yang paling lemah secara otomatis.
- `plots/02_overall_score_distribution.png`: Distribusi skor internal. Membantu membaca apakah performa terkonsentrasi di skor tinggi atau tersebar.
- `plots/03_average_metric_scores.png`: Rata-rata tiap metrik internal seperti semantic similarity, coverage keyword, citation coverage, dan hit retrieval.
- `plots/04_score_by_topic.png`: Rata-rata skor internal per topik hukum.
- `plots/05_latency_vs_score.png`: Hubungan latency dengan skor, untuk melihat tradeoff kualitas dan waktu inferensi.
- `plots/06_metric_heatmap.png`: Heatmap metrik per pertanyaan. Bagus untuk menemukan pola kegagalan spesifik.
- `plots/07_quality_label_count.png`: Jumlah pertanyaan di label poor/fair/good/excellent berdasarkan skor internal.
- `plots/08_expected_article_rank.png`: Rank pasal target dalam hasil retrieval. Rank rendah berarti retrieval menaruh rujukan penting di atas.
- `plots/09_semantic_vs_keyword.png`: Perbandingan kedekatan semantik jawaban dengan coverage keyword eksplisit.
- `plots/10_score_spread_by_topic.png`: Sebaran skor per topik, menunjukkan konsistensi performa lintas topik.
- `plots/11_answer_word_count.png`: Panjang jawaban per pertanyaan. Dipakai untuk melihat jawaban terlalu pendek/panjang.
- `plots/12_binary_hit_stack.png`: Stack hit retrieval/citation biner per pertanyaan.

## Catatan Penggunaan

Untuk laporan akhir, gunakan `overall_score_with_external` sebagai skor gabungan karena sudah menggabungkan sinyal otomatis/retrieval dengan penilaian external judge. Gunakan `external_judge_comparison.csv` untuk melihat alasan ketat dari masing-masing external judge, terutama pada pertanyaan dengan `external_judge_disagreement` tinggi atau `internal_external_gap` besar.
