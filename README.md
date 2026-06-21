# 🎭 Aplikasi Klasifikasi Emosi Wajah (Facial Emotion Classification App)

> **📝 Note:** Project ini dibuat untuk memenuhi tugas Ujian Akhir Semester (UAS) Praktikum Pengenalan Pola.

Aplikasi web interaktif berbasis **Streamlit** untuk mengklasifikasikan emosi dari wajah manusia secara real-time. Model deep learning dideploy menggunakan **LiteRT (TensorFlow Lite)** agar proses inferensi berjalan cepat dan ringan, bahkan di lingkungan komputasi terbatas seperti Streamlit Community Cloud.

🔗 **Link Aplikasi:** [face-emotion-app.streamlit.app](https://facial-emotion-app-vurt32fsl4fnc4mzdpgmte.streamlit.app/)

---

## 📊 Dataset
Dataset yang digunakan adalah [Real-world Affective Faces Database (RAF-DB) Face Emotion Dataset](https://www.kaggle.com/datasets/nishchalchandel/raf-db-face-emotion-dataset) dari Kaggle. Dataset ini memuat citra wajah manusia terdiri dari **15.339 gambar** yang siap pakai dengan pembagian rasio 80/10/10 (Train, Validation, Test). Dataset ini terdiri dari **7 kelas emosi**:

* *Angry* (Marah)
* *Disgust* (Jijik)
* *Fear* (Takut)
* *Happy* (Bahagia)
* *Neutral* (Netral)
* *Sad* (Sedih)
* *Surprise* (Terkejut)

Seluruh gambar pada dataset ini melalui proses augmentasi (rotasi, shift, zoom, horizontal flip, brightness adjustment) untuk meningkatkan variasi data dan mencegah terjadinya *overfitting* selama pelatihan model.

---

## 🧠 Arsitektur Model & Performa
Model klasifikasi dibangun menggunakan arsitektur **Custom CNN (Convolutional Neural Network)** berbasis TensorFlow/Keras:

1.  **Input Layer:** Menerima gambar dengan resolusi `224x224` piksel (RGB).
2.  **Feature Extraction:**
    * **Block 1:** `Conv2D` (32 filter, kernel 3x3, ReLU) + `BatchNormalization` + `MaxPooling2D` (2x2)
    * **Block 2:** `Conv2D` (64 filter, kernel 3x3, ReLU) + `BatchNormalization` + `MaxPooling2D` (2x2)
    * **Block 3:** `Conv2D` (128 filter, kernel 3x3, ReLU) + `BatchNormalization` + `MaxPooling2D` (2x2)
    * **Block 4:** `Conv2D` (256 filter, kernel 3x3, ReLU) + `BatchNormalization` + `MaxPooling2D` (2x2)
3.  **Classification Head:**
    * `GlobalAveragePooling2D` untuk meringkas fitur dan mencegah overfitting (lebih baik dari Flatten).
    * `Dense` (512 unit, ReLU) dengan `L2 Regularizer` & `Dropout(0.4)`.
    * `Dense` (256 unit, ReLU) dengan `L2 Regularizer` & `Dropout(0.3)`.
    * `Dense` (7 unit, Softmax) sebagai output untuk 7 kelas klasifikasi emosi.

### Hasil Evaluasi Model
* **Test Accuracy:** **73.29%**
* Model dikonversi menjadi format `.tflite` untuk digunakan pada aplikasi web agar ukuran file lebih kecil dan meminimalkan konsumsi RAM saat dijalankan. Selain itu, model juga diekspor ke format TensorFlow.js dan SavedModel untuk fleksibilitas *deployment*.

---

## 📁 Struktur Direktori Project
```text
facial-emotion-classification/
│
├── hasil_model/
│   └── tflite/
│       └── face_emotion_model.tflite  # Model TensorFlow Lite hasil konversi
│
├── app.py                             # Kode utama aplikasi Streamlit
├── requirements.txt                   # Daftar dependensi Python
├── notebook.ipynb                     # Notebook pelatihan dan konversi model
├── best_face_emotion_model.keras      # Model Keras terbaik (.keras format)
└── README.md                          # Dokumentasi project (file ini)
```

---

## 🚀 Cara Penggunaan & Menjalankan Project

### 1. Training Model (Opsional)

Jika kamu ingin melatih ulang model, buka `notebook.ipynb` melalui Jupyter Notebook atau Google Colab.
**Penting:** Jika menggunakan Colab, pastikan kamu memasukkan kredensial API Kaggle kamu (`KAGGLE_API_TOKEN`) di fitur **Secrets** Colab agar dataset dapat diunduh otomatis.

### 2. Menjalankan Project Secara Lokal

**Prasyarat**
Pastikan Anda sudah menginstal Python (disarankan versi 3.10 ke atas) dan `pip`.

**Langkah-langkah**

1. **Clone repositori ini:**
   ```bash
   git clone https://github.com/falah-bit/facial-emotion-app.git
   cd facial-emotion-classification
   ```

2. **Instal dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan aplikasi Streamlit:**
   ```bash
   streamlit run app.py
   ```

4. Buka browser Anda dan akses aplikasi di `http://localhost:8501`.

---

## 🛠️ Stack Teknologi

* **Deep Learning Framework:** TensorFlow & Keras
* **Deployment Runtime:** LiteRT (oleh Google, suksesor TensorFlow Lite)
* **Web Framework:** Streamlit
* **Image Processing & Utils:** NumPy, Pillow (PIL), Matplotlib, Seaborn
* **Programming Language:** Python 3.10+