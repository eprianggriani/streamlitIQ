import streamlit as st
import numpy as np
import pickle
import pandas as pd
import sqlite3

# Mengubah layout 
st.set_page_config(page_title="Prediksi IQ & Outcome", layout="wide", page_icon="🧠")

# Load the train model dan scaler
classifier = pickle.load(open('Klasifikasi.sav', 'rb'))
regresi_nilai = pickle.load(open('NilaiIQ.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

# Sidebar untuk informasi tambahan
st.sidebar.title("🔍 Tentang Aplikasi")
st.sidebar.info("Aplikasi ini memprediksi Nilai IQ dan Outcome berdasarkan skor mentah yang diinputkan pengguna. Prediksi dibuat menggunakan model Machine Learning yang terdiri dari **RandomForestClassifier** dan **LinearRegression**.")

# Buat koneksi ke SQLite
conn = sqlite3.connect('data_prediksi_iq.db')
c = conn.cursor()

# Buat tabel jika belum ada
c.execute('''
CREATE TABLE IF NOT EXISTS prediksi_iq (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nama TEXT,
    nilai_iq INTEGER,
    kategori TEXT
)
''')

# Sidebar untuk riwayat dan hapus data
st.sidebar.markdown("---")  # Pembatas
st.sidebar.markdown("## 📑 Riwayat Data")
with st.sidebar.expander("Lihat Riwayat Prediksi IQ"):
    # Ambil semua data dari database dan tampilkan dalam tabel
    c.execute('SELECT * FROM prediksi_iq')
    data = c.fetchall()

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
    # Menghapus semua data dalam tabel
    c.execute('DELETE FROM prediksi_iq')
    conn.commit()
    st.rerun()  # Jalankan ulang aplikasi untuk menampilkan pembaruan

# Inisialisasi Session State
if "prediksi" not in st.session_state:
    st.session_state["prediksi"] = None
if "kategori" not in st.session_state:
    st.session_state["kategori"] = None
if "nama" not in st.session_state:
    st.session_state["nama"] = ""

# Judul Aplikasi
st.markdown("<h1 style='text-align: center; color: blue;'>🧠 Aplikasi Prediksi Nilai IQ dan Outcome</h1>", unsafe_allow_html=True)

# Input dari pengguna 
st.markdown("<h3 style='text-align: center;'>Masukkan Nama dan Skor Mentah Anda di bawah ini:</h3>", unsafe_allow_html=True)

# Input nama pengguna
nama = st.text_input("👤 Nama Anda:", value=st.session_state["nama"])

# Input data pengguna
input_data = st.number_input("⚖️ Skor Mentah (X):", min_value=0, max_value=100, step=1)

# Button untuk menghitung hasil
if st.button("🔍 Hitung Hasil"):
    if nama and input_data:
        # Simpan nama ke session state
        st.session_state["nama"] = nama

        # Proses input data
        input_data_as_numpy_array = np.array(input_data).reshape(1, -1)

        # Normalisasi data input
        std_data = scaler.transform(input_data_as_numpy_array)

        # Prediksi Nilai IQ
        prediksi_iq = regresi_nilai.predict(std_data)
        prediksi_iq = round(prediksi_iq[0])
        st.session_state["prediksi"] = prediksi_iq

        # Prediksi Outcome
        prediction = classifier.predict(std_data)
        if prediction[0] == 1:
            kategori = "Di bawah rata-rata"
        elif prediction[0] == 2:
            kategori = "Rata-rata"
        else:
            kategori = "Di atas rata-rata"
        st.session_state["kategori"] = kategori

        # Menyimpan data ke SQLite
        c.execute('''
            INSERT INTO prediksi_iq (nama, nilai_iq, kategori) 
            VALUES (?, ?, ?)
        ''', (nama, prediksi_iq, kategori))
        conn.commit()
        st.rerun()  # Refresh untuk memperbarui tabel dan UI
    else:
        st.warning("Harap masukkan Nama dan Skor Mentah untuk melihat hasil prediksi.")

# Tampilkan hasil prediksi jika ada di session state
if st.session_state["prediksi"] is not None and st.session_state["kategori"] is not None:
    st.markdown("---")
    st.markdown("<h2 style='text-align: center; color: green;'>📊 Hasil Prediksi</h2>", unsafe_allow_html=True)
    st.success(f"**Hay {st.session_state['nama']}**")
    st.success(f"**Nilai IQ Anda: {st.session_state['prediksi']}**")
    if st.session_state["kategori"] == "Di bawah rata-rata":
        st.warning(f"Kategori Anda: **{st.session_state['kategori']}**")
    elif st.session_state["kategori"] == "Rata-rata":
        st.info(f"Kategori Anda: **{st.session_state['kategori']}**")
    else:
        st.success(f"Kategori Anda: **{st.session_state['kategori']}**")

# Tutup koneksi ke SQLite
conn.close()

# Tampilan tambahan di bawah
st.markdown("<p style='text-align: center;'>Ingin mengulang prediksi? Masukkan skor baru dan klik tombol di atas!</p>", unsafe_allow_html=True)
