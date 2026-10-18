import os
import tensorflow as tf
import numpy as np

# Menentukan lokasi model secara absolut
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "ml", "models", "nmt_model.h5")

nmt_model = None

def load_model():
    """Memuat model NMT dari disk ke memori."""
    global nmt_model
    if os.path.exists(MODEL_PATH):
        try:
            nmt_model = tf.keras.models.load_model(MODEL_PATH)
            print(f"[nmt_service] Model NMT berhasil dimuat dari: {MODEL_PATH}")
            return True
        except Exception as e:
            print(f"[nmt_service] Info: model belum dapat dimuat ({e}), menggunakan mode hybrid.")
            return False
    return False

# Inisialisasi awal saat modul diimpor
load_model()

def reload_nmt_model():
    """Memuat ulang model NMT setelah proses training/fine-tuning selesai."""
    return load_model()

def translate_nmt(teks_awal_hybrid: str, arah_terjemahan: str) -> str:
    """
    Memproses teks dari terjemahan awal (hybrid) agar susunannya lebih alami 
    menggunakan model Neural Machine Translation (NMT).
    """
    # Nanti di sini Anda dapat memasukkan logika:
    # 1. Mengubah teks_awal_hybrid menjadi sequence angka menggunakan Tokenizer Keras.
    # 2. Melakukan prediksi: hasil_prediksi = nmt_model.predict(sequence_angka)
    # 3. Mengubah hasil_prediksi kembali menjadi teks bahasa target.
    
    # Mengembalikan hasil terjemahan bersih tanpa teks tambahan
    teks_akhir = teks_awal_hybrid
    
    return teks_akhir