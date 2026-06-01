# 🧵 Sistem QC Cacat Kain Batik

Proyek ini dibuat untuk mencoba penerapan Deep Learning dalam proses Quality Control (QC) kain batik. Sistem akan mengenali beberapa jenis cacat pada kain berdasarkan gambar yang diberikan.

Dataset yang digunakan berisi berbagai jenis cacat kain serta gambar kain yang tidak memiliki cacat. Model dibangun menggunakan TensorFlow dan Keras.

## Yang Bisa Dilakukan

- Menghitung jumlah dataset tiap kelas
- Membagi dataset menjadi train, validation, dan test
- Melatih model klasifikasi gambar
- Menguji performa model
- Siap dijalankan di Google Colab atau lokal

---

## Struktur Folder

```text
sistem-qc-cacat-kain-batik/
│
├── dataset/
├── dataset_split/
│
├── count_dataset.py
├── prepare_dataset.py
├── train_model.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Kelas yang Digunakan

- Broken stitch
- Defect free
- Hole
- Horizontal
- Lines
- Needle mark
- Pinched fabric
- Stain
- Vertical

Total dataset yang digunakan sekitar 2900+ gambar.

---

## Cara Menjalankan

### Clone Repository

```bash
git clone https://github.com/username/sistem-qc-cacat-kain-batik.git

cd sistem-qc-cacat-kain-batik
```

### Install Library

```bash
pip install -r requirements.txt
```

### Cek Jumlah Dataset

```bash
python count_dataset.py
```

### Split Dataset

```bash
python prepare_dataset.py
```

### Training Model

```bash
python train_model.py
```

---

## Training di Google Colab

Upload project ke GitHub lalu jalankan:

```python
!git clone https://github.com/username/sistem-qc-cacat-kain-batik.git
%cd sistem-qc-cacat-kain-batik

!pip install -r requirements.txt
!python train_model.py
```

Jangan lupa aktifkan GPU pada menu Runtime → Change Runtime Type → GPU.

---

## Tools yang Dipakai

- Python
- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib

---

## Kenapa Proyek Ini Dibuat?

Awalnya proyek ini dibuat sebagai pembelajaran tentang Computer Vision dan Deep Learning. Ide utamanya adalah bagaimana proses pengecekan kualitas kain yang biasanya dilakukan secara manual bisa dibantu oleh model AI untuk mengenali jenis cacat secara otomatis.

Masih banyak yang bisa dikembangkan, misalnya:

- Transfer Learning
- Prediksi real-time
- Dashboard Streamlit
- Deploy ke web

---

