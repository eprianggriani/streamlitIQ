import streamlit as st
import numpy as np
import pickle
import pandas as pd
import sqlite3

# Mengubah layout 
st.set_page_config(page_title="Prediksi IQ & Outcome", layout="wide", page_icon="🧠")

# Load the train model dan scaler (hanya dilakukan sekali saat aplikasi dimulai)
@st.cache_resource
def load_models():
    classifier = pickle.load(open('Klasifikasi.sav', 'rb'))
    regresi_nilai = pickle.load(open('NilaiIQ.sav', 'rb'))
    scaler = pickle.load(open('scaler.sav', 'rb'))
    return classifier, regresi_nilai, scaler

classifier, regresi_nilai, scaler = load_models()

# Sidebar untuk informasi tambahan
st.sidebar.title("🔍 Tentang Aplikasi")
st.sidebar.info("Aplikasi ini memprediksi Nilai IQ dan Outcome berdasarkan skor mentah yang diinputkan pengguna. Prediksi dibuat menggunakan model Machine Learning yang terdiri dari **RandomForestClassifier** dan **LinearRegression**.")

# Fungsi untuk membuat dan membuka koneksi ke SQLite
def get_database_connection():
    conn = sqlite3.connect('data_prediksi_iq.db')
    return conn

# Buat tabel jika belum ada (hanya dieksekusi satu kali)
def create_table():
    conn = get_database_connection()
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS prediksi_iq (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nama TEXT,
        nilai_iq INTEGER,
        kategori TEXT
    )
    ''')
    conn.commit()
    conn.close()

create_table()

# Sidebar untuk riwayat dan hapus data
st.sidebar.markdown("---")  # Pembatas
st.sidebar.markdown("## 📑 Riwayat Data")
with st.sidebar.expander("Lihat Riwayat Prediksi IQ"):
    conn = get_database_connection()
    c = conn.cursor()
    # Ambil semua data dari database dan tampilkan dalam tabel
    c.execute('SELECT * FROM prediksi_iq')
    data = c.fetchall()
    conn.close()

    # Jika ada data, tampilkan dalam dataframe
    if data:
        df = pd.DataFrame(data, columns=["ID", "Nama", "Nilai IQ", "Kategori"])
        st.dataframe(df)

        # Download tombol untuk database sebagai CSV
        csv = df.to_csv(index=False)
        st.download_button(
            label="📄 Unduh Hasil sebagai CSV",
            data=csv,
            file_name="Hasil_Prediksi_IQ.csv",
            mime="text/csv"
        )
    else:
        st.info("Belum ada riwayat prediksi yang tersimpan.")

# Tombol hapus riwayat di sidebar
if st.sidebar.button("🗑️ Hapus Riwayat Data"):
    conn = get_database_connection()
    c = conn.cursor()
    # Menghapus semua data dalam tabel
    c.execute('DELETE FROM prediksi_iq')
    conn.commit()
    conn.close()
    st.sidebar.success("Riwayat data telah dihapus.")

# Judul Aplikasi
st.markdown("<h1 style='text-align: center; color: blue;'>🧠 Aplikasi Prediksi Nilai IQ dan Outcome</h1>", unsafe_allow_html=True)

# Input dari pengguna 
st.markdown("<h3 style='text-align: center;'>Masukkan Nama dan Skor Mentah Anda di bawah ini:</h3>", unsafe_allow_html=True)

# Input nama pengguna
nama = st.text_input("👤 Nama Anda:")

# Input data pengguna
input_data = st.number_input("⚖️ Skor Mentah (X):", min_value=0, max_value=100, step=1)

# Button untuk menghitung hasil
if st.button("🔍 Hitung Hasil"):
    if nama and input_data:
        # Proses input data
        input_data_as_numpy_array = np.array(input_data).reshape(1, -1)

        # Normalisasi data input
        std_data = scaler.transform(input_data_as_numpy_array)

        # Prediksi Nilai IQ
        prediksi_iq = regresi_nilai.predict(std_data)
        prediksi_iq = round(prediksi_iq[0])

        # Prediksi Outcome
        prediction = classifier.predict(std_data)

        # Divider sebelum menampilkan hasil
        st.markdown("---")
        
        # Menampilkan hasil prediksi 
        st.markdown("<h2 style='text-align: center; color: green;'>📊 Hasil Prediksi</h2>", unsafe_allow_html=True)
        st.success(f"**Hay {nama}**")
        st.success(f"**Nilai IQ Anda: {prediksi_iq}**")

        # Tampilkan outcome prediksi 
        if prediction[0] == 1:
            kategori = "Di bawah rata-rata"
            st.warning(f"Kategori Anda: **{kategori}**")
        elif prediction[0] == 2:
            kategori = "Rata-rata"
            st.info(f"Kategori Anda: **{kategori}**")
        else:
            kategori = "Di atas rata-rata"
            st.success(f"Kategori Anda: **{kategori}**")

        # Menyimpan data ke SQLite
        conn = get_database_connection()
        c = conn.cursor()
        c.execute('''
            INSERT INTO prediksi_iq (nama, nilai_iq, kategori) 
            VALUES (?, ?, ?)
        ''', (nama, prediksi_iq, kategori))
        conn.commit()
        conn.close()

        # Divider sebelum unduhan
        st.markdown("---")

    else:
        st.warning("Harap masukkan Nama dan Skor Mentah untuk melihat hasil prediksi.")

# Tampilan tambahan di bawah
st.markdown("<p style='text-align: center;'>Ingin mengulang prediksi? Masukkan skor baru dan klik tombol di atas!</p>", unsafe_allow_html=True)
