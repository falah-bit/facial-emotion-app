import os
import numpy as np
import streamlit as st
from PIL import Image

# 1. Setup Halaman
st.set_page_config(
    page_title="Klasifikasi Emosi Wajah",
    page_icon="😃",
    layout="centered"
)

st.title("😃 Aplikasi Klasifikasi Emosi Wajah")
st.write("Unggah foto wajah untuk memprediksi emosinya.")

# 2. Path Model (Sudah disesuaikan dengan struktur folder terakhir)
MODEL_PATH = "hasil_model/tflite/face_emotion_model.tflite"
LABEL_PATH = "hasil_model/tflite/label.txt" 

# 3. Memuat nama kelas/label
@st.cache_resource
def load_labels():
    if os.path.exists(LABEL_PATH):
        with open(LABEL_PATH, "r") as f:
            return [line.strip() for line in f.readlines()]
    else:
        # Default labels emosi. 
        # PENTING: Pastikan urutan ini SAMA PERSIS dengan urutan label saat melatih model!
        return [
            "Marah (Angry)", 
            "Jijik (Disgust)", 
            "Takut (Fear)", 
            "Senang (Happy)", 
            "Netral (Neutral)", 
            "Sedih (Sad)", 
            "Terkejut (Surprise)"
        ]

labels = load_labels()

# 4. Memuat Interpreter TFLite
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model tidak ditemukan di `{MODEL_PATH}`. Pastikan folder dan file modelnya ada.")
        return None
    
    # Deteksi backend interpreter
    try:
        import ai_edge_litert.interpreter as tflite
    except ImportError:
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            try:
                import tensorflow.lite as tflite
            except ImportError:
                st.error("Gagal memuat interpreter TFLite. Silakan install `tensorflow` melalui terminal.")
                return None

    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

# 5. Preprocessing Gambar
def preprocess_image(image, target_size=(224, 224)):
    # CATATAN: Ubah "RGB" menjadi "L" dan ukuran (224, 224) menjadi (48, 48) 
    # JIKA modelmu dilatih menggunakan gambar grayscale (hitam putih) berukuran 48x48.
    img = image.convert("RGB") 
    img = img.resize(target_size)
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, 0).astype(np.float32)
    return arr

# 6. Widget Upload Gambar
uploaded_files = st.file_uploader("Pilih gambar wajah...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    for idx, uploaded_file in enumerate(uploaded_files):
        with st.container(border=True):
            st.write(f"### 📷 Gambar #{idx+1}: `{uploaded_file.name}`")
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar yang diunggah", use_container_width=True)
            
            if interpreter is not None:
                with st.spinner("Menganalisis ekspresi wajah..."):
                    # Preprocess
                    input_data = preprocess_image(image)
                    
                    # Dapatkan tensor input & output
                    input_details = interpreter.get_input_details()
                    output_details = interpreter.get_output_details()
                    
                    # Jalankan inference
                    interpreter.set_tensor(input_details[0]['index'], input_data)
                    interpreter.invoke()
                    
                    # Ambil hasil prediksi
                    output_data = interpreter.get_tensor(output_details[0]['index'])[0]
                    pred_idx = np.argmax(output_data)
                    confidence = output_data[pred_idx]
                    
                    # Tampilkan Hasil
                    st.success(f"Prediksi Emosi: **{labels[pred_idx]}**")
                    st.write(f"Tingkat Keyakinan (Confidence): **{confidence * 100:.2f}%**")
                    
                    # Progres bar
                    st.progress(float(confidence))
            else:
                st.warning("Model tidak siap.")