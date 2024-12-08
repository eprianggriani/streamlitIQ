import streamlit as st
import numpy as np
import pickle
import pandas as pd
import mysql.connector

# Mengubah layout 
st.set_page_config(page_title="Prediksi IQ & Outcome", layout="wide", page_icon="🧠")

# Load the train model dan scaler
classifier = pickle.load(open('Klasifikasi.sav', 'rb'))
regresi_nilai = pickle.load(open('NilaiIQ.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

# Sidebar untuk informasi tambahan
st.sidebar.title("🔍 Tentang Aplikasi")
st.sidebar.info("Aplikasi ini memprediksi Nilai IQ dan Outcome berdasarkan skor mentah yang diinputkan pengguna. Prediksi dibuat menggunakan model Machine Learning yang terdiri dari **RandomForestClassifier** dan **LinearRegression**.")

# Inisialisasi Session State untuk menyimpan riwayat sesi
if "riwayat_prediksi" not in st.session_state:
    st.session_state["riwayat_prediksi"] = []

if "prediksi" not in st.session_state:
    st.session_state["prediksi"] = None
if "kategori" not in st.session_state:
    st.session_state["kategori"] = None
if "nama" not in st.session_state:
    st.session_state["nama"] = ""

# Fungsi untuk menghubungkan ke database MySQL
def connect_to_db():
    conn = mysql.connector.connect(
        host="localhost",  # Ganti dengan host MySQL kamu
        user="root",  # Ganti dengan username MySQL kamu
        database="prediksi_iq",  # Ganti dengan nama database MySQL kamu
    )
    return conn

# Membuat tabel jika belum ada
def create_table():
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prediksi_iq (
            id INT AUTO_INCREMENT PRIMARY KEY,
            nama VARCHAR(255),
            nilai_iq INT,
            kategori VARCHAR(50)
        )
    ''')
    conn.commit()
    cursor.close()
    conn.close()

# Menyimpan hasil prediksi ke MySQL
def save_prediction_to_db(nama, nilai_iq, kategori):
    conn = connect_to_db()
    cursor = conn.cursor()
    query = "INSERT INTO prediksi_iq (nama, nilai_iq, kategori_outcome) VALUES (%s, %s, %s)"
    values = (nama, nilai_iq, kategori)
    cursor.execute(query, values)
    conn.commit()
    cursor.close()
    conn.close()

# Panggil fungsi untuk membuat tabel
create_table()

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

        # Menyimpan riwayat ke session state
        st.session_state["riwayat_prediksi"].append({
            "Nama": nama,
            "Nilai IQ": prediksi_iq,
            "Kategori": kategori
        })

        # Simpan ke database MySQL
        save_prediction_to_db(nama, prediksi_iq, kategori)
        
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

# Menampilkan riwayat dari session state
if st.session_state["riwayat_prediksi"]:
    st.sidebar.markdown("## 📑 Riwayat Prediksi Sesi Ini")
    df_riwayat = pd.DataFrame(st.session_state["riwayat_prediksi"])
    st.sidebar.dataframe(df_riwayat)

    # Tombol download CSV
    csv = df_riwayat.to_csv(index=False)
    st.sidebar.download_button(
        label="📄 Unduh Riwayat sebagai CSV",
        data=csv,
        file_name="Riwayat_Prediksi_IQ_Sesi.csv",
        mime="text/csv"
    )

# Tampilan tambahan di bawah
st.markdown("<p style='text-align: center;'>Ingin mengulang prediksi? Masukkan skor baru dan klik tombol di atas!</p>", unsafe_allow_html=True)
