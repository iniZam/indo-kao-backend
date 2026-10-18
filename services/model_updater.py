import os
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from services import db_service, hybrid_service, nmt_service

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "kamus_kao_indo.csv")
MODEL_PATH = os.path.join(BASE_DIR, "ml", "models", "nmt_model.h5")

# Global state untuk memantau status pelatihan model
TRAINING_STATUS: Dict[str, Any] = {
    "is_training": False,
    "progress": "idle",
    "last_trained": None,
    "error": None,
    "total_samples": 0,
    "epoch_info": ""
}


def sinkronkan_terverifikasi_ke_kamus(target_id: Optional[int] = None) -> Dict[str, Any]:
    """
    Menyinkronkan data yang sudah berstatus 'terverifikasi' ke dalam file kamus_kao_indo.csv
    dan memuat ulang kamus pada memori hybrid_service.
    Jika target_id ditentukan, hanya menyinkronkan item dengan ID tersebut.
    """
    if target_id is not None:
        item = db_service.ambil_terjemahan_by_id(target_id)
        if not item or item.get("status") != "terverifikasi":
            return {"status": "error", "message": "Item belum terverifikasi atau tidak ditemukan.", "synced_count": 0}
        daftar_item = [item]
    else:
        daftar_item = db_service.ambil_terverifikasi_belum_sinkron()

    if not daftar_item:
        return {"status": "success", "message": "Tidak ada data baru untuk disinkronkan.", "synced_count": 0}

    # Baca file kamus saat ini
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception:
        df = pd.DataFrame(columns=["INDONESIA", "PAGU", "MODOLE", "TOWILIKO"])

    id_sukses = []

    for item in daftar_item:
        teks_sumber = str(item["teks_sumber"]).strip()
        # Gunakan teks koreksi penutur lokal jika ada, jika tidak pakai teks sistem
        teks_final = str(item["teks_koreksi"] or item["teks_terjemahan"]).strip()
        arah = str(item["arah_terjemahan"]).lower()
        dialek = str(item["dialek"] or "PAGU").upper()

        if not teks_sumber or not teks_final:
            continue

        is_ke_indonesia = (
            "ke_indonesia" in arah
            or "ke_indo" in arah
            or "ke_id" in arah
            or arah.startswith("kao_")
            or arah in ["kao_ke_indo", "daerah_ke_id"]
        )

        if is_ke_indonesia:
            teks_indo = teks_final
            teks_daerah = teks_sumber
        else:
            teks_indo = teks_sumber
            teks_daerah = teks_final

        # Cari apakah teks Bahasa Indonesia sudah ada di kamus
        kunci_norm = hybrid_service.normalisasi_kunci(teks_indo)
        idx_match = None

        for idx, row in df.iterrows():
            if hybrid_service.normalisasi_kunci(str(row["INDONESIA"])) == kunci_norm:
                idx_match = idx
                break

        if idx_match is not None:
            # Perbarui kolom dialek spesifik
            if dialek in df.columns:
                df.at[idx_match, dialek] = teks_daerah
            else:
                df.at[idx_match, "PAGU"] = teks_daerah
        else:
            # Buat baris baru
            baris_baru = {
                "INDONESIA": teks_indo,
                "PAGU": teks_daerah if dialek == "PAGU" else "",
                "MODOLE": teks_daerah if dialek == "MODOLE" else "",
                "TOWILIKO": teks_daerah if dialek == "TOWILIKO" else ""
            }
            # Jika bukan ketiga dialek itu, default ke PAGU
            if dialek not in ["PAGU", "MODOLE", "TOWILIKO"]:
                baris_baru["PAGU"] = teks_daerah

            df = pd.concat([df, pd.DataFrame([baris_baru])], ignore_index=True)

        id_sukses.append(item["id"])

    # Simpan kembali ke file CSV
    df.to_csv(CSV_PATH, index=False)

    # Tandai di database bahwa data sudah disinkronkan
    if id_sukses:
        db_service.tandai_disinkronkan(id_sukses)

    # Muat ulang memory hybrid_service secara dinamis agar langsung aktif
    hybrid_service.load_kamus()

    return {
        "status": "success",
        "message": f"{len(id_sukses)} data terverifikasi berhasil dimasukkan ke kamus dan memori diperbarui.",
        "synced_count": len(id_sukses),
        "total_entri_kamus": len(df)
    }


def _worker_pelatihan_nmt(epochs: int = 30, batch_size: int = 64):
    """Fungsi pekerja yang berjalan di background thread untuk melatih model NMT."""
    global TRAINING_STATUS
    TRAINING_STATUS["is_training"] = True
    TRAINING_STATUS["progress"] = "Membaca dataset kamus..."
    TRAINING_STATUS["error"] = None

    try:
        # Ambil ID yang belum dilatih
        item_belum_dilatih = db_service.ambil_belum_dilatih()
        id_yang_dilatih = [item["id"] for item in item_belum_dilatih]

        import tensorflow as tf
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences

        if not os.path.exists(CSV_PATH):
            raise FileNotFoundError(f"File dataset {CSV_PATH} tidak ditemukan.")

        df = pd.read_csv(CSV_PATH)
        sumber_teks = []
        target_teks = []

        for _, row in df.iterrows():
            indo = str(row["INDONESIA"]).strip()
            if not indo:
                continue

            # Masukkan pasangan dari tiap dialek yang tersedia
            pagu = str(row["PAGU"]).strip()
            modole = str(row["MODOLE"]).strip()
            towiliko = str(row["TOWILIKO"]).strip()

            if pagu:
                sumber_teks.append(pagu)
                target_teks.append(indo)
            if modole:
                sumber_teks.append(modole)
                target_teks.append(indo)
            if towiliko:
                sumber_teks.append(towiliko)
                target_teks.append(indo)

        if not sumber_teks:
            raise ValueError("Tidak ada data pelatihan yang valid di kamus.")

        TRAINING_STATUS["total_samples"] = len(sumber_teks)
        TRAINING_STATUS["progress"] = f"Menyiapkan tokenizer untuk {len(sumber_teks)} sampel kalimat..."

        target_teks_tagged = ["<start> " + teks + " <end>" for teks in target_teks]

        # Normalisasi
        sumber_teks = [t.lower() for t in sumber_teks]
        target_teks_tagged = [t.lower() for t in target_teks_tagged]

        tokenizer_sumber = Tokenizer(filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
        tokenizer_sumber.fit_on_texts(sumber_teks)
        tensor_sumber = tokenizer_sumber.texts_to_sequences(sumber_teks)
        tensor_sumber = pad_sequences(tensor_sumber, padding="post")

        tokenizer_target = Tokenizer(filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n', lower=True)
        tokenizer_target.fit_on_texts(target_teks_tagged)
        tensor_target = tokenizer_target.texts_to_sequences(target_teks_tagged)
        tensor_target = pad_sequences(tensor_target, padding="post")

        vocab_size_sumber = len(tokenizer_sumber.word_index) + 1
        vocab_size_target = len(tokenizer_target.word_index) + 1
        embedding_dim = 256
        units = 512

        # Bangun Arsitektur Encoder-Decoder LSTM
        encoder_inputs = Input(shape=(None,))
        encoder_embedding = Embedding(vocab_size_sumber, embedding_dim)(encoder_inputs)
        encoder_lstm = LSTM(units, return_state=True)
        _, state_h, state_c = encoder_lstm(encoder_embedding)
        encoder_states = [state_h, state_c]

        decoder_inputs = Input(shape=(None,))
        decoder_embedding = Embedding(vocab_size_target, embedding_dim)(decoder_inputs)
        decoder_lstm = LSTM(units, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
        decoder_dense = Dense(vocab_size_target, activation="softmax")
        decoder_outputs = decoder_dense(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        target_data = np.zeros_like(tensor_target)
        target_data[:, :-1] = tensor_target[:, 1:]

        TRAINING_STATUS["progress"] = f"Melatih model NMT ({epochs} epoch)..."

        # Callback custom untuk update progress status
        class ProgressCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                acc = logs.get("accuracy", 0)
                loss = logs.get("loss", 0)
                TRAINING_STATUS["epoch_info"] = f"Epoch {epoch + 1}/{epochs} - loss: {loss:.4f} - acc: {acc:.4f}"

        model.fit(
            [tensor_sumber, tensor_target],
            target_data,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[ProgressCallback()],
            verbose=0
        )

        # Simpan model
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        model.save(MODEL_PATH)

        # Muat ulang model pada nmt_service
        nmt_service.reload_nmt_model()

        # Tandai data yang baru dilatih ke database
        if id_yang_dilatih:
            db_service.tandai_sudah_dilatih(id_yang_dilatih)

        TRAINING_STATUS["progress"] = "Selesai"
        TRAINING_STATUS["last_trained"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[model_updater] Pelatihan NMT selesai. Model berhasil disimpan di {MODEL_PATH}")

    except Exception as e:
        TRAINING_STATUS["error"] = str(e)
        TRAINING_STATUS["progress"] = f"Error: {e}"
        print(f"[model_updater] Terjadi error pada pelatihan NMT: {e}")

    finally:
        TRAINING_STATUS["is_training"] = False


def mulai_pelatihan_nmt_async(epochs: int = 30, batch_size: int = 64) -> Dict[str, Any]:
    """Menjalankan pelatihan model NMT secara non-blocking di background thread."""
    global TRAINING_STATUS
    if TRAINING_STATUS["is_training"]:
        return {
            "status": "busy",
            "message": "Pelatihan model NMT sedang berjalan. Harap tunggu hingga selesai.",
            "data": TRAINING_STATUS
        }

    thread = threading.Thread(
        target=_worker_pelatihan_nmt,
        args=(epochs, batch_size),
        daemon=True
    )
    thread.start()

    return {
        "status": "started",
        "message": f"Pelatihan model NMT ({epochs} epoch) berhasil dimulai di background.",
        "data": TRAINING_STATUS
    }


def ambil_status_pelatihan() -> Dict[str, Any]:
    """Mengembalikan status live dari pelatihan model."""
    return TRAINING_STATUS
