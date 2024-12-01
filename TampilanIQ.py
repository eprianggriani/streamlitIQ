import streamlit as st
import numpy as np
import pickle
from io import BytesIO
from xhtml2pdf import pisa

# Fungsi untuk membuat PDF
def generate_pdf(content):
    pdf_buffer = BytesIO()
    pisa_status = pisa.CreatePDF(
        src=BytesIO(content.encode("utf-8")),
        dest=pdf_buffer
    )
    pdf_buffer.seek(0)
    if pisa_status.err:
        raise ValueError("Gagal membuat PDF.")
    return pdf_buffer

# Mengubah layout menjadi lebih lebar
st.set_page_config(page_title="Prediksi IQ & Outcome", layout="wide", page_icon="🧠")

# Load the trained models and scaler
classifier = pickle.load(open('Klasifikasi.sav', 'rb'))
regresi_nilai = pickle.load(open('NilaiIQ.sav', 'rb'))
scaler = pickle.load(open('scaler.sav', 'rb'))

# Sidebar untuk informasi tambahan
st.sidebar.title("🔍 Tentang Aplikasi")
st.sidebar.info("Aplikasi ini memprediksi Nilai IQ dan Outcome berdasarkan skor mentah yang diinputkan pengguna. Prediksi dibuat menggunakan model Machine Learning yang terdiri dari **RandomForestClassifier** dan **LinearRegression**.")

# Judul Aplikasi
st.markdown("<h1 style='text-align: center; color: blue;'>🧠 Aplikasi Prediksi Nilai IQ dan Outcome</h1>", unsafe_allow_html=True)

# Input dari pengguna dengan section yang jelas
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
        
        # Menampilkan hasil prediksi dengan style yang berbeda
        st.markdown("<h2 style='text-align: center; color: green;'>📊 Hasil Prediksi</h2>", unsafe_allow_html=True)
        st.success(f"**Hay {nama}**")
        st.success(f"**Nilai IQ Anda: {prediksi_iq}**")

        # Tampilkan outcome prediksi tanpa emotikon
        if prediction[0] == 1:
            st.warning("Kategori Anda: **Di bawah rata-rata**")
        elif prediction[0] == 2:
            st.info("Kategori Anda: **Rata-rata**")
        else:
            st.success("Kategori Anda: **Di atas rata-rata**")

        # Footer dengan gaya lebih halus
        st.markdown("---")

        # Menampilkan hasil PDF dengan format desimal pada kolom tertentu
        html_content = f"""
        <h1>Prediksi Nilai IQ dan Outcome</h1>
        <p><strong>Nama:</strong> {nama}</p>
        <p><strong>Nilai IQ:</strong> {prediksi_iq}</p>
        <h2>Hasil Diagnosis:</h2>
        <p>{'Di bawah rata-rata' if prediction[0] == 1 else 'Rata-rata' if prediction[0] == 2 else 'Di atas rata-rata'}</p>
        """
        try:
            # Generate the PDF file
            pdf_file = generate_pdf(html_content)

            # Provide a button to download the PDF
            st.download_button(
                label="📄 Unduh PDF",
                data=pdf_file,
                file_name="Hasil_Prediksi_IQ.pdf",
                mime="application/pdf"
            )
        except ValueError as e:
            st.error(f"Gagal membuat PDF: {e}")

    else:
        st.warning("Harap masukkan Nama dan Skor Mentah untuk melihat hasil prediksi.")

# Tampilan tambahan di bawah
st.markdown("<p style='text-align: center;'>Ingin mengulang prediksi? Masukkan skor baru dan klik tombol di atas!</p>", unsafe_allow_html=True)
