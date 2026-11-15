<div align="center">

<img src="image/logo_bank.png" alt="Bank Churn Logo" width="120"/>

# 🏦 Bank Customer Churn Prediction & Analytics Dashboard

**Sistem prediksi churn nasabah berbasis Machine Learning (Decision Tree) dengan antarmuka Streamlit.**

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Decision%20Tree-F7931E?logo=scikit-learn&logoColor=white)
![Trello](https://img.shields.io/badge/Project%20Management-Trello-0052CC?logo=trello&logoColor=white)

</div>

<img src="image/bank.jpeg" alt="Bank Illustration" width="100%"/>

---

## 📌 Ringkasan

Churn (nasabah berhenti menggunakan layanan bank) adalah masalah bisnis yang mahal, karena mempertahankan nasabah lama umumnya lebih murah daripada mencari nasabah baru. Proyek mini ini membangun:

1. **Model klasifikasi** untuk memprediksi apakah seorang nasabah berpotensi *churn* atau tidak.
2. **Aplikasi web Streamlit** agar model dapat dipakai langsung dengan mengisi data nasabah lewat form.
3. **Dashboard analitik (Google Looker Studio)** yang memberi wawasan tentang kondisi nasabah bank.

Seluruh proses pengerjaan dikelola dengan **Trello Kanban**.

## 🔗 Tautan Penting

| Item | Link |
|------|------|
| 📋 Trello Board | [Bank Customer Churner Prediction and Analytics Dashboard](https://trello.com/b/udEzHNug/bank-customer-churner-prediction-and-analytics-dashboard) |
| 🚀 Streamlit App | _https://bankchurnersprediction.streamlit.app/_ |

---

## 📂 Struktur Repository

```
bankchurnprediction/
├── image/
│   ├── bank.jpeg                                   # Ilustrasi header aplikasi
│   └── logo_bank.png                               # Logo / page icon aplikasi
├── Bank Churner Prediction.ipynb                   # Notebook: EDA → preprocessing → modelling → evaluasi
├── BankChurners.csv                                # Dataset mentah (6.399 nasabah)
├── final_data_predicted.csv                        # Data uji beserta hasil prediksi model
├── decision_tree_for_churn_classification.pkl      # Model Decision Tree terlatih
├── encoders.pkl                                    # Label Encoder (country & gender)
├── main.py                                         # Aplikasi Streamlit
├── requirements.txt                                # Dependensi aplikasi
└── README.md
```

---

## 📊 Dataset

- **Sumber:** Bank Customer Churn dataset ([Google Drive](https://drive.google.com/file/d/1QIfV-uJMy0BYKgIt9MpM2SJzLL5aCEmc/view?usp=sharing))
- **Ukuran:** 6.399 baris × 12 kolom
- **Kualitas data:** tidak ada *missing value*, tidak ada duplikasi `customer_id`, dan tidak ditemukan kesalahan format pada kolom kategorikal.
- **Distribusi target:** ±68,2% *tidak churn* dan ±31,8% *churn* (data tidak seimbang).

| Fitur | Deskripsi |
|-------|-----------|
| `customer_id` | ID unik nasabah (**tidak** dipakai untuk training) |
| `credit_score` | Skor kredit nasabah (350–850) |
| `country` | Negara asal nasabah: France, Spain, Germany |
| `gender` | Jenis kelamin: Male / Female |
| `age` | Usia nasabah (tahun) |
| `tenure` | Lama menjadi nasabah (tahun) |
| `balance` | Saldo rekening |
| `products_number` | Jumlah produk bank yang digunakan |
| `credit_card` | Memiliki kartu kredit (0 = tidak, 1 = ya) |
| `active_member` | Anggota aktif (0 = tidak, 1 = ya) |
| `estimated_salary` | Estimasi penghasilan nasabah |
| `churn` | **Target** (0 = tidak churn, 1 = churn) |

---

## 🔬 Alur Kerja Machine Learning

Seluruh langkah didokumentasikan di [`Bank Churner Prediction.ipynb`](<Bank Churner Prediction.ipynb>).

### 1. Exploratory Data Analysis (EDA)
- Histogram distribusi, boxplot outlier, dan heatmap korelasi untuk fitur numerik.
- Visualisasi kategorikal: distribusi country, gender, serta hubungannya dengan churn dan status aktif.

**Temuan utama:**
- Mayoritas fitur numerik **tidak berdistribusi normal**.
- **Outlier** ditemukan pada `credit_score` (skor sangat rendah, < 400) dan `age` (usia > 60). Outlier **tidak dihapus** karena tujuan utamanya adalah memahami pola dan anomali churn.
- Tidak ada korelasi antar fitur yang signifikan.
- France mendominasi jumlah nasabah (hampir sama dengan Spain + Germany), dan nasabah Male lebih banyak daripada Female. Meski begitu, nasabah **Female menyumbang churn lebih banyak**.

### 2. Pre-processing
- Membuang `customer_id`.
- **Train/test split 80:20** dengan `stratify=y` dan `random_state=42` (5.119 data latih, 1.280 data uji).
- **Label Encoding** untuk `country` (France=0, Germany=1, Spain=2) dan `gender` (Female=0, Male=1). Encoder disimpan ke `encoders.pkl`.
- **SMOTE** pada data latih untuk menyeimbangkan kelas churn (3.489 vs 3.489).
- Scaling tidak diperlukan karena Decision Tree tidak sensitif terhadap skala fitur.

### 3. Modelling
- Algoritma: **Decision Tree Classifier**.
- **Hyperparameter tuning** dengan `GridSearchCV` (cv=5) pada `max_depth`, `min_samples_split`, dan `min_samples_leaf`.
- Parameter terbaik: `max_depth=10`, `min_samples_leaf=4`, `min_samples_split=10`.
- Model akhir memakai `criterion='gini'` dan `class_weight={0: 1, 1: 2.14}`, lalu disimpan ke `decision_tree_for_churn_classification.pkl`.

### 4. Evaluasi (data uji, 1.280 nasabah)

**Akurasi: ≈ 75,9%**

| Kelas | Precision | Recall | F1-Score | Support |
|-------|:---------:|:------:|:--------:|:-------:|
| 0 – Tidak Churn | 0,86 | 0,77 | 0,81 | 873 |
| 1 – Churn | 0,60 | 0,73 | 0,66 | 407 |

**Confusion Matrix**

|  | Prediksi: Tidak Churn | Prediksi: Churn |
|---|:---:|:---:|
| **Aktual: Tidak Churn** | 676 (TN) | 197 (FP) |
| **Aktual: Churn** | 111 (FN) | 296 (TP) |

**Interpretasi:** model cukup baik menangkap nasabah yang benar-benar churn (recall 73%), tetapi precision kelas churn baru 60%, artinya sekitar 40% prediksi "churn" adalah *false positive*. Hal ini sejalan dengan ketidakseimbangan data yang masih memengaruhi model.

---

## 🖥️ Aplikasi Streamlit

Aplikasi `main.py` memuat model dan encoder, meminta input data nasabah lewat form, lalu menampilkan hasil prediksi:

- ✅ **0** → Nasabah tidak berpotensi churn
- ⚠️ **1** → Nasabah berpotensi churn

**Input yang diminta:** ID Nasabah, Credit Score, Negara, Jenis Kelamin, Usia, Tenure, Balance, Jumlah Produk, Kepemilikan Kartu Kredit, Status Anggota Aktif, dan Estimated Salary.

### Cara Menjalankan Secara Lokal

```bash
# 1. Clone repository
git clone <url-repository-kamu>
cd bankchurnprediction

# 2. (Opsional) Buat virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependensi
pip install -r requirements.txt

# 4. Jalankan aplikasi
streamlit run main.py
```

Aplikasi akan terbuka di `http://localhost:8501`.

> **Catatan:** file `.pkl` dibuat dengan scikit-learn versi tertentu. Jika muncul peringatan versi saat memuat model, gunakan versi scikit-learn yang sama dengan saat training.

### Menjalankan Notebook

Notebook ditulis di **Google Colab** (membaca data dari Google Drive). Untuk menjalankannya di lokal, ganti path `pd.read_csv(...)` ke `BankChurners.csv`, hapus sel `drive.mount`, dan install dependensi tambahan:

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn imbalanced-learn prettytable joblib
```

---

## 📋 Manajemen Proyek (Trello)

Proyek ini dikelola menggunakan **Trello Kanban** untuk memantau progres tiap tahap, mulai dari perencanaan, EDA, pemodelan, pembuatan aplikasi, dashboard, hingga dokumentasi akhir.

🔗 **[Buka Trello Board](https://trello.com/b/udEzHNug/bank-customer-churner-prediction-and-analytics-dashboard)**

---

## 🧰 Tools & Teknologi

- **Bahasa & Library:** Python, pandas, NumPy, matplotlib, seaborn, scikit-learn, imbalanced-learn, joblib
- **Aplikasi:** Streamlit
- **Manajemen Proyek:** Trello
- **Editor / Environment:** Visual Studio Code, Google Colab

---

## 🚧 Keterbatasan & Rencana Pengembangan

- Precision kelas churn masih 60%, sehingga perlu eksplorasi model lain (Random Forest, Gradient Boosting, XGBoost) dan metrik tambahan seperti ROC-AUC.
- SMOTE pada data latih sudah dipakai bersamaan dengan `class_weight`; kombinasi keduanya layak dievaluasi ulang.
- Validasi input di aplikasi bisa diperketat (misalnya rentang credit score 350–850 dan tenure 0–10).
- Menampilkan probabilitas churn (`predict_proba`) selain label 0/1 agar hasil lebih informatif bagi tim bisnis.

---

## 👤 Author

**_mayanggumelarr_**

<div align="center">

⭐ Jika proyek ini bermanfaat, jangan ragu memberi bintang pada repository ini!

</div>
