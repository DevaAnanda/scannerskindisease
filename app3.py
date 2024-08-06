import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

# Daftar label penyakit kulit
label_models = [
    "Atopic_Dermatitis", "Basal_Cell_Carcinoma", "Benign_Keratosis",
    "Chickenpox", "Eczema", "Melanocytic_Nevi", "Melanoma",
    "Tinea_Versicolor"
]

# Informasi tentang setiap penyakit kulit
label_information = {
    "Atopic_Dermatitis": "Dermatitis atopik adalah kondisi yang menyebabkan kulit kering, gatal, dan meradang. Kondisi ini umum terjadi pada anak-anak, tetapi bisa dialami pada segala usia. Dermatitis atopik bersifat jangka panjang (kronis) dan cenderung kambuh dari waktu ke waktu. Meskipun dapat mengganggu, kondisi ini tidak menular.",
    "Basal_Cell_Carcinoma": "Basal Cell Carcinoma adalah jenis kanker kulit yang paling sering berkembang pada area kulit yang terpapar sinar matahari, seperti wajah. Pada kulit berwarna coklat dan hitam, karsinoma sel basal sering terlihat seperti benjolan berwarna coklat atau hitam mengilap dengan tepi yang menggulung.", 
    "Benign_Keratosis": "Benign keratosis seboroik adalah pertumbuhan kulit umum yang tidak bersifat kanker (jinak). Orang cenderung mendapatkan lebih banyak seiring bertambahnya usia. Keratosis seboroik biasanya berwarna coklat, hitam, atau coklat muda. Pertumbuhan (lesi) ini terlihat seperti lilin atau bersisik dan sedikit terangkat.",
    "Chickenpox": "Chickenpox adalah penyakit yang sangat menular yang disebabkan oleh virus varicella-zoster (VZV), sejenis virus herpes. Penyakit ini sering kali ringan, ditandai dengan ruam gatal di wajah, kulit kepala, dan tubuh dengan bintik-bintik merah muda dan lepuh kecil berisi cairan yang mengering dan menjadi keropeng empat hingga lima hari kemudian.",
    "Eczema": "Eczema adalah penyakit kronis (jangka panjang) yang menyebabkan peradangan, kemerahan, dan iritasi pada kulit. Kondisi ini umum terjadi dan biasanya dimulai pada masa kanak-kanak; namun, siapa pun dapat mengalami penyakit ini pada usia berapa pun.",
    "Melanocytic_Nevi": "Melanocytic Nevi adalah istilah medis untuk tahi lalat. Nevus dapat muncul di mana saja pada tubuh. Mereka bersifat jinak (tidak bersifat kanker) dan biasanya tidak memerlukan pengobatan. Sebagian kecil dari nevus melanositik dapat berkembang menjadi melanoma di dalamnya.",
    "Melanoma": "Melanoma adalah jenis kanker kulit yang dimulai di sel melanosit. Melanosit adalah sel yang menghasilkan pigmen yang memberi warna pada kulit. Pigmen ini disebut melanin. Ilustrasi ini menunjukkan sel-sel melanoma yang meluas dari permukaan kulit ke lapisan kulit yang lebih dalam.",
    "Tinea_Versicolor": "Tinea versicolor adalah infeksi jamur umum pada kulit yang menyebabkan bercak-bercak kecil berubah warna. Bercak-bercak ini mungkin lebih terang atau lebih gelap dari warna kulit Anda. Tinea versicolor disebabkan oleh jenis jamur yang secara alami ada pada kulit Anda."
}

# Memuat model yang sudah dilatih
model = tf.keras.models.load_model("./my_model_cnn_percobaan6.h5")

st.title("CNN Model for Skin Disease Classification")

# untuk menyesuaikan threshold
confidence_threshold = 0.7

uploaded_file = st.file_uploader("Pilih file gambar")
output = None

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    images = cv2.imdecode(file_bytes, -1)
    image_resized = cv2.resize(images, (224, 224))
    images = np.expand_dims(image_resized, axis=0)
    images = tf.keras.applications.mobilenet_v2.preprocess_input(images)  # Normalisasi
    pred = model.predict(images)
    max_confidence = np.max(pred)
    if max_confidence < confidence_threshold:
        output = "Bukan Kulit"
    else:
        output = label_models[np.argmax(pred)]

show = st.button("Tampilkan Gambar")
if show:
    if uploaded_file is None:
        st.write("Masukkan gambar terlebih dahulu")
    else:
        st.image(uploaded_file, width=200)

if uploaded_file is not None:
    st.markdown(f"<h1 style='color: white;'>{output}</h1>", unsafe_allow_html=True)
    if output != "Bukan Kulit":
        st.markdown(f"<h3 style='color: white;'>{label_information[output]}</h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='color: white;'>Gambar yang diunggah bukan gambar kulit atau tidak dapat dikenali oleh model.</h3>", unsafe_allow_html=True)
