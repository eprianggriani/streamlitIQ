import streamlit as st
import numpy as np
import pickle
import pandas as pd
from io import BytesIO
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

        # Kategori berdasarkan prediksi
        kategori = "Di bawah rata-rata" if prediction[0] == 1 else "Rata-rata" if prediction[0] == 2 else "Di atas rata-rata"

        # Divider sebelum menampilkan hasil
        st.markdown("---")
        
        # Menampilkan hasil prediksi 
        st.markdown("<h2 style='text-align: center; color: green;'>📊 Hasil Prediksi</h2>", unsafe_allow_html=True)
        st.success(f"**Hay {nama}**")
        st.success(f"**Nilai IQ Anda: {prediksi_iq}**")
        st.info(f"Kategori Anda: **{kategori}**")

        # Footer 
        st.markdown("---")

        # *** Menyimpan ke MySQL ***
                # *** Menyimpan ke MySQL ***
        try:
            conn = mysql.connector.connect(
                host="localhost",  # Ganti dengan host database Anda
                user="root",       # Ganti dengan username MySQL Anda
                password="",       # Ganti dengan password MySQL Anda
                database="prediksi_iq"  # Nama database
            )
            cursor = conn.cursor()

            # Menyimpan data baru ke tabel
            cursor.execute("INSERT INTO prediksi_iq (nama, nilai_iq, kategori_outcome) VALUES (%s, %s, %s)",
                           (nama, prediksi_iq, kategori))
            conn.commit()

            # Menampilkan pesan sukses
            st.success("Hasil prediksi berhasil disimpan!")

            # Mengambil semua data dari database
            cursor.execute("SELECT * FROM prediksi_iq")
            rows = cursor.fetchall()

            # Konversi data ke DataFrame untuk Excel
            df = pd.DataFrame(rows, columns=["ID", "Nama", "Nilai IQ", "Kategori Outcome"])

        except mysql.connector.Error as err:
            st.error(f"Error: {err}")
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()

        # Menyimpan semua data sebagai file Excel untuk diunduh
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Data Prediksi")
        excel_buffer.seek(0)

        # Menampilkan tombol unduh Excel
        st.download_button(
            label="📄 Unduh Data sebagai Excel",
            data=excel_buffer,
            file_name="Data_Prediksi_IQ.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    else:
        st.warning("Harap masukkan Nama dan Skor Mentah untuk melihat hasil prediksi.")

# Tampilan tambahan di bawah
st.markdown("<p style='text-align: center;'>Ingin mengulang prediksi? Masukkan skor baru dan klik tombol di atas!</p>", unsafe_allow_html=True)
