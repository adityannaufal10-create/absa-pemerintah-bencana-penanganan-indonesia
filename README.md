# Aspect-Based Sentiment Analysis (ABSA) Opini Publik terhadap Efektivitas Penanganan Bencana oleh Pemerintah Indonesia

Repository ini berisi kode program, dataset, dan dokumentasi eksperimen pemodelan **Aspect-Based Sentiment Analysis (ABSA)** untuk mengkaji opini publik terhadap efektivitas penanganan bencana oleh Pemerintah Indonesia dari komentar media sosial **YouTube** dan **Instagram** dengan perbandingan **lima metode hyperparameter tuning**.

---

## 📌 Ringkasan Penelitian

* **Topik:** Evaluasi persepsi & opini publik terhadap penanganan bencana oleh pemerintah Indonesia.
* **Sumber Data:** Komentar publik dari YouTube Data API v3 dan Instagram (via Apify) pada periode Januari–September 2026.
* **Pendekatan:** Aspect-Based Sentiment Analysis (ABSA) multi-head 4-kelas (TIDAK MEMBAHAS, POSITIF, NEGATIF, NETRAL).
* **Hasil Terbaik:** Model **TF-IDF + Logistic Regression** dengan tuning **Hybrid Grid → Bayesian** mencapai **F1-score pasangan 0,521**.

---

## 🗂️ 6 Aspek Penanganan Bencana

Analisis membagi evaluasi ke dalam 6 aspek utama yang divalidasi dengan *topic modeling* LDA:

| Kode | Nama Aspek | Cakupan Utama |
| :---: | :--- | :--- |
| **A1** | Kecepatan Respons & Evakuasi | Tim SAR, pemadaman, respon cepat, kehadiran petugas di lapangan. |
| **A2** | Bantuan, Logistik & Pemulihan | Penyaluran dan kecukupan bantuan, hunian sementara (huntara), pemulihan pascabencana. |
| **A3** | Informasi, Komunikasi & Peringatan Dini | Kejujuran laporan pejabat, kejelasan informasi, hoaks, peringatan dini. |
| **A4** | Anggaran & Transparansi | Alokasi dana bencana, korupsi, pemotongan anggaran (misal: MBG/Kopdes vs BNPB). |
| **A5** | Koordinasi Antarlembaga & Kinerja Pejabat | Penilaian presiden, menteri, kepala BNPB, kepala daerah, koordinasi pusat-daerah. |
| **A6** | Pencegahan, Mitigasi & Penegakan Hukum | Pencegahan karhutla, alih fungsi lahan/sawit, penindakan pembakar dan korporasi. |

---

## 🛠️ Alur Kerja & Metode

### 1. Data & Anotasi
* **Corong Seleksi:** Dari 70.579 komentar mentah disaring menjadi 1.056 populasi (kriteria K1–K7).
* **Sampel Berlabel:** 282 komentar utama (241 YouTube, 41 Instagram) dinilai oleh 3 annotator (335 baris pasangan komentar-aspek / 1.692 sel).
* **Reliabilitas Anotator:** Mencapai Fleiss' $\kappa = 0,786$ (*substantive agreement*). Deteksi aspek mencapai $\kappa = 0,837 - 0,953$.

### 2. Model yang Dibandingkan
* **Konvensional:** TF-IDF + Logistic Regression, Linear SVM, Complement Naive Bayes.
* **Deep Learning:** IndoBERT (`indobenchmark/indobert-base-p1`) dan IndoBERTweet (`indolem/indobertweet-base-uncased`).

### 3. 5 Metode Hyperparameter Tuning
Eksperimen menggunakan *Nested 5-Fold Cross-Validation* dengan anggaran seimbang (30 evaluasi per metode):
1. **Grid Search**
2. **Random Search**
3. **Bayesian Optimization (TPE)**
4. **Genetic Algorithm**
5. **Hybrid Grid → Bayesian (Coarse-to-Fine)**

---

## 📊 Hasil Utama & Perbandingan Model

| No | Model | Tuning | Precision | Recall | F1-Score Pasangan | F1 Deteksi |
| :-: | :--- | :--- | :-: | :-: | :-: | :-: |
| 🥇 | **TF-IDF + Logistic Regression** | **Hybrid Grid → Bayesian** | **0,513** | **0,530** | **0,521 ± 0,073** | **0,582** |
| 🥈 | IndoBERTweet (multi-head) | - | 0,589 | 0,415 | 0,486 ± 0,040 | 0,540 |
| 🥉 | TF-IDF + Complement NB | Hybrid Grid → Bayesian | 0,432 | 0,557 | 0,484 ± 0,055 | 0,544 |
| 4 | TF-IDF + Linear SVM | Hybrid Grid → Bayesian | 0,530 | 0,426 | 0,471 ± 0,100 | 0,515 |
| 5 | IndoBERT (multi-head) | - | 0,514 | 0,350 | 0,415 ± 0,045 | 0,479 |

### Key Findings:
* **Model Sederhana + Tuning vs Transformer:** Pada dataset berukuran kecil (282 sampel), model TF-IDF + Logistic Regression yang dituning mengalahkan transformer (IndoBERT/IndoBERTweet) karena parameter efektifnya jauh lebih sedikit sehingga tidak mudah *under-fit*.
* **Hybrid Grid → Bayesian** terbukti memberikan performa terbaik secara konsisten di ketiga model konvensional.
* **Persentase Sentimen Negatif Publik:** Tingkat ketidakpuasan masyarakat paling tinggi berada pada aspek **Anggaran & Transparansi (92,7% negatif)**, **Informasi & Peringatan Dini (88,5% negatif)**, serta **Kinerja Pejabat (81,2% negatif)**.

---

## 👥 Penulis / Anggota Kelompok 10

Penelitian ini disusun untuk memenuhi Ujian Tengah Semester (UTS) Mata Kuliah Natural Language Processing (NLP):

* **Mikael Ardiyanta Widyadana Purniawan**
* **Raphael Angelo Adikara Purnama**
* **Aditya Naufal Jay Putra**
* **Najwa Yasyfa Dhiya Sugina**
* **Exsalia Eka Nevita**
* **Anindhito Gading Rasunajati**

**Program Studi S1 Teknologi Sains Data**  
Fakultas Teknologi Maju dan Multidisiplin  
**Universitas Airlangga**

---
