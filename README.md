# Aplikasi AI untuk Ekstraksi Data dari Kuesioner PDF

Aplikasi ini adalah sebuah prototipe yang dikembangkan menggunakan Streamlit dan PyTorch untuk mendemonstrasikan proses otomatisasi ekstraksi data dari kuesioner kesehatan yang telah dipindai dan disimpan dalam format PDF. Sistem ini menggunakan pendekatan AI dua tahap untuk mengenali karakter tulisan tangan di dalam kotak-kotak jawaban yang telah ditentukan.

## Alur Kerja AI

1.  **Deteksi Kotak Karakter**: Sebuah model **SSDLite dengan backbone MobileNetV3-Large** digunakan untuk mendeteksi lokasi setiap kotak karakter individual pada area pertanyaan yang telah di-crop sebelumnya dari halaman PDF.
2.  **Klasifikasi Karakter**: Gambar dari setiap kotak karakter yang terdeteksi kemudian dimasukkan ke dalam model klasifikasi **MobileNetV3-Small** yang telah dilatih untuk mengenali digit (0-9) dan beberapa huruf kapital spesifik (A-G, X, Y, Z).
3.  **Output**: Hasil pengenalan karakter dari setiap area pertanyaan digabungkan dan dapat diunduh dalam format file Excel.

## 🌟 Fitur Aplikasi

-   **Unggah File PDF**: Pengguna dapat mengunggah file PDF kuesioner secara langsung melalui antarmuka web.
-   **Pemilihan Anotasi Dinamis**: Opsi untuk memilih jenis kelamin ("pria" atau "perempuan") untuk menggunakan file anotasi area yang sesuai.
-   **Proses Otomatis**: Menjalankan pipeline deteksi dan klasifikasi secara otomatis dengan menekan satu tombol.
-   **Tampilan Hasil Interaktif**: Menampilkan hasil ekstraksi dengan paginasi, memungkinkan pengguna untuk meninjau gambar area pertanyaan dan teks yang dikenali.
-   **Unduh Laporan**: Menghasilkan dan menyediakan laporan dalam format `.xlsx` yang berisi semua data yang diekstraksi, termasuk gambar area pertanyaan untuk verifikasi.

## 🛠️ Teknologi yang Digunakan

-   **Framework Aplikasi Web**: Streamlit
-   **Bahasa Pemrograman**: Python
-   **Machine Learning & Deep Learning**: PyTorch, Torchvision
-   **Pemrosesan Gambar & PDF**: Pillow, pdf2image
-   **Manipulasi Data & Laporan**: NumPy, Openpyxl

## 🚀 Setup dan Instalasi Lokal

Untuk menjalankan aplikasi ini di komputer lokal, ikuti langkah-langkah berikut:

### 1. Prasyarat

-   [Python](https://www.python.org/downloads/) (versi 3.9+)
-   [Git](https://git-scm.com/downloads/)
-   **Poppler**: `pdf2image` memerlukan `poppler-utils`.
    -   **Untuk Windows**: Unduh Poppler [dari sini](https://github.com/oschwartz10612/poppler-windows/releases/), ekstrak, dan tambahkan folder `bin` ke PATH environment variable Anda.
    -   **Untuk macOS (via Homebrew)**: `brew install poppler`
    -   **Untuk Linux (Debian/Ubuntu)**: `sudo apt-get install poppler-utils`

### 2. Klone Repositori

Buka terminal Anda dan klone repositori ini:

```bash
git clone https://github.com/AL1isHere/Kuesioner_Extraction.git
cd Kuesioner_Extraction
```

### 3. Buat Environment Virtual (Sangat Direkomendasikan)

```bash
# Membuat environment virtual
python -m venv venv

# Mengaktifkan environment
# Windows
venv\Scripts\activate
# macOS & Linux
source venv/bin/activate
```

### 4. Instal Dependensi

Instal semua library Python yang diperlukan menggunakan file `requirements.txt`:

```bash
pip install -r requirements.txt
```

### 5. Jalankan Aplikasi

Setelah semua dependensi terinstal, jalankan aplikasi Streamlit dengan perintah:

```bash
streamlit run app.py
```

Aplikasi akan terbuka secara otomatis di browser web Anda.

## 📂 Struktur File Proyek

Struktur file di repositori ini diatur dengan pola Model-View-Controller (MVC) untuk keterbacaan dan pemeliharaan yang lebih baik:

```
├── app_data/                # Menyimpan model AI, anotasi, dan data statis lainnya.
├── config.py                # Menyimpan semua konfigurasi dan parameter aplikasi.
├── model.py                 # Berisi semua logika inti pemrosesan data dan AI.
├── controller.py            # Bertindak sebagai perantara antara UI dan logika model.
├── app.py                   # File utama untuk menampilkan UI (View) dan menjalankan aplikasi.
├── requirements.txt         # Daftar library Python yang dibutuhkan.
└── packages.txt             # Daftar dependensi sistem (untuk deployment di Streamlit Cloud).
```

## ☁️ Deployment

Aplikasi ini dapat di-deploy secara publik menggunakan platform seperti **Streamlit Community Cloud** atau **Hugging Face Spaces**.

-   Pastikan semua file (`app.py`, `controller.py`, `model.py`, `config.py`, `requirements.txt`, `packages.txt`, dan folder `app_data/` beserta isinya) telah diunggah ke repositori GitHub.
-   Hubungkan repositori GitHub Anda ke platform deployment pilihan Anda.
-   Pastikan platform menggunakan `requirements.txt` dan `packages.txt` untuk mengatur environment.
-   **Penting**: Untuk mengatasi error `size mismatch`, pastikan versi `torch` dan `torchvision` di `requirements.txt` sama persis dengan versi yang digunakan saat melatih model.

---
Dibuat dengan ❤️ untuk Skripsi.
