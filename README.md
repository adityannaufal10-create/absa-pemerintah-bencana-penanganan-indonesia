# Aspect-Based Sentiment Analysis (ABSA) Opini Publik terhadap Efektivitas Penanganan Bencana oleh Pemerintah Indonesia

Repository ini berisi kode program, dataset, dan dokumentasi eksperimen pemodelan **Aspect-Based Sentiment Analysis (ABSA)** untuk mengkaji opini publik terhadap efektivitas penanganan bencana oleh Pemerintah Indonesia dari komentar media sosial **YouTube** dan **Instagram** dengan perbandingan **lima metode hyperparameter tuning**[cite: 5].

---

## 📌 Ringkasan Penelitian

* **Topik:** Evaluasi persepsi & opini publik terhadap penanganan bencana oleh pemerintah Indonesia[cite: 5, 6].
* **Sumber Data:** Komentar publik dari YouTube Data API v3 dan Instagram (via Apify) pada periode Januari–September 2026[cite: 8].
* **Pendekatan:** Aspect-Based Sentiment Analysis (ABSA) multi-head 4-kelas (TIDAK MEMBAHAS, POSITIF, NEGATIF, NETRAL)[cite: 6, 11].
* **Hasil Terbaik:** Model **TF-IDF + Logistic Regression** dengan tuning **Hybrid Grid → Bayesian** mencapai **F1-score pasangan 0,521**[cite: 18].

---

## 🗂️ 6 Aspek Penanganan Bencana

Analisis membagi evaluasi ke dalam 6 aspek utama yang divalidasi dengan *topic modeling* LDA[cite: 9, 10]:

| Kode | Nama Aspek | Cakupan Utama |
| :---: | :--- | :--- |
| **A1** | Kecepatan Respons & Evakuasi | Tim SAR, pemadaman, respon cepat, kehadiran petugas di lapangan[cite: 9]. |
| **A2** | Bantuan, Logistik & Pemulihan | Penyaluran dan kecukupan bantuan, hunian sementara (huntara), pemulihan pascabencana[cite: 10]. |
| **A3** | Informasi, Komunikasi & Peringatan Dini | Kejujuran laporan pejabat, kejelasan informasi, hoaks, peringatan dini[cite: 10]. |
| **A4** | Anggaran & Transparansi | Alokasi dana bencana, korupsi, pemotongan anggaran (misal: MBG/Kopdes vs BNPB)[cite: 10]. |
| **A5** | Koordinasi Antarlembaga & Kinerja Pejabat | Penilaian presiden, menteri, kepala BNPB, kepala daerah, koordinasi pusat-daerah[cite: 10]. |
| **A6** | Pencegahan, Mitigasi & Penegakan Hukum | Pencegahan karhutla, alih fungsi lahan/sawit, penindakan pembakar dan korporasi[cite: 10]. |

---

## 🛠️ Alur Kerja & Metode

### 1. Data & Anotasi
* **Corong Seleksi:** Dari 70.579 komentar mentah disaring menjadi 1.056 populasi (kriteria K1–K7)[cite: 8].
* **Sampel Berlabel:** 282 komentar utama (241 YouTube, 41 Instagram) dinilai oleh 3 annotator (335 baris pasangan komentar-aspek / 1.692 sel)[cite: 8].
* **Reliabilitas Anotator:** Mencapai Fleiss' $\kappa = 0,786$ (*substantive agreement*)[cite: 14]. Deteksi aspek mencapai $\kappa = 0,837 - 0,953$[cite: 14, 15].

### 2. Model yang Dibandingkan
* **Konvensional:** TF-IDF + Logistic Regression, Linear SVM, Complement Naive Bayes[cite: 11].
* **Deep Learning:** IndoBERT (`indobenchmark/indobert-base-p1`) dan IndoBERTweet (`indolem/indobertweet-base-uncased`)[cite: 11].

### 3. 5 Metode Hyperparameter Tuning
Eksperimen menggunakan *Nested 5-Fold Cross-Validation* dengan anggaran seimbang (30 evaluasi per metode)[cite: 11, 12]:
1. **Grid Search**[cite: 12]
2. **Random Search**[cite: 12]
3. **Bayesian Optimization (TPE)**[cite: 12]
4. **Genetic Algorithm**[cite: 12]
5. **Hybrid Grid → Bayesian (Coarse-to-Fine)**[cite: 12]

---

## 📊 Hasil Utama & Perbandingan Model

| No | Model | Tuning | Precision | Recall | F1-Score Pasangan | F1 Deteksi |
| :-: | :--- | :--- | :-: | :-: | :-: | :-: |
| 🥇 | **TF-IDF + Logistic Regression** | **Hybrid Grid → Bayesian** | **0,513** | **0,530** | **0,521 ± 0,073** | **0,582** |
| 🥈 | IndoBERTweet (multi-head) | - | 0,589 | 0,415 | 0,486 ± 0,040 | 0,540 |
| 🥉 | TF-IDF + Complement NB | Hybrid Grid → Bayesian | 0,432 | 0,557 | 0,484 ± 0,055 | 0,544 |
| 4 | TF-IDF + Linear SVM | Hybrid Grid → Bayesian | 0,530 | 0,426 | 0,471 ± 0,100 | 0,515 |
| 5 | IndoBERT (multi-head) | - | 0,514 | 0,350 | 0,415 ± 0,045 | 0,479 |

*[cite: 18, 19]*

### Key Findings:
* **Model Sederhana + Tuning vs Transformer:** Pada dataset berukuran kecil (282 sampel), model TF-IDF + Logistic Regression yang dituning mengalahkan transformer (IndoBERT/IndoBERTweet) karena parameter efektifnya jauh lebih sedikit sehingga tidak mudah *under-fit*[cite: 25].
* **Hybrid Grid → Bayesian** terbukti memberikan performa terbaik secara konsisten di ketiga model konvensional[cite: 18, 25].
* **Persentase Sentimen Negatif Publik:** Tingkat ketidakpuasan masyarakat paling tinggi berada pada aspek **Anggaran & Transparansi (92,7% negatif)**, **Informasi & Peringatan Dini (88,5% negatif)**, serta **Kinerja Pejabat (81,2% negatif)**[cite: 14].

---

## 👥 Penulis / Anggota Kelompok 10

Penelitian ini disusun untuk memenuhi Ujian Tengah Semester (UTS) Mata Kuliah Natural Language Processing (NLP)[cite: 5]:

* **Mikael Ardiyanta Widyadana Purniawan**[cite: 5]
* **Raphael Angelo Adikara Purnama**[cite: 5]
* **Aditya Naufal Jay Putra**[cite: 5]
* **Najwa Yasyfa Dhiya Sugina**[cite: 5]
* **Exsalia Eka Nevita**[cite: 5]
* **Anindhito Gading Rasunajati**[cite: 5]

**Program Studi S1 Teknologi Sains Data**  
Fakultas Teknologi Maju dan Multidisiplin  
**Universitas Airlangga**[cite: 5]

---
