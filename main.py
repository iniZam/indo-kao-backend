from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import traceback

# 1. Mematikan pesan optimasi oneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 2. Menyembunyikan log INFO dan WARNING dari TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Import modul-modul dari folder services
from services import (
    stt_service,
    prep_service,
    hybrid_service,
    nmt_service,
    tts_service,
    db_service,
    model_updater
)

app = FastAPI(title="API Penerjemah Bahasa Kao - Indonesia")

# Konfigurasi CORS agar Android bebas mengakses API ini
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Memisahkan folder input dan output agar direktori rapi
os.makedirs("static/audio_input", exist_ok=True)
os.makedirs("static/audio_output", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# =============================================================================
# 1. ENDPOINT PENERJEMAHAN SUARA (AUDIO)
# =============================================================================
@app.post("/api/v1/translate/audio")
async def translate_audio(
    audio_file: UploadFile = File(...), 
    arah_terjemahan: str = Form(...) # Contoh: "kao_ke_indo" atau "indonesia_ke_pagu"
):
    """
    Endpoint ini menerima file audio dari Android dan arah terjemahannya,
    lalu memprosesnya melewati seluruh pipeline kecerdasan buatan.
    Hasil terjemahan otomatis dicatat ke antrean verifikasi penutur lokal / admin.
    """
    temp_audio_path = f"static/audio_input/{audio_file.filename}"
    with open(temp_audio_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)

    try:
        print(f"\n[translate_audio] Menerima audio: {audio_file.filename} (arah: {arah_terjemahan})")
        # 1. Speech Recognition: Mengubah ucapan menjadi teks
        teks_sumber = stt_service.recognize_speech(temp_audio_path, arah_terjemahan)
        
        if not teks_sumber:
            print(f"[translate_audio] Peringatan: Audio hening atau ucapan tidak terdeteksi oleh STT. Mengembalikan status 400.")
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Suara tidak berhasil dikenali. Silakan coba rekam kembali dengan lebih jelas."}
            )

        print(f"[translate_audio] Ucapan berhasil dikenali: '{teks_sumber}'")

        # 2. Preprocessing Teks: Spacing, Normalisasi, Tokenisasi, Filtering
        teks_bersih = prep_service.preprocess_text(teks_sumber)

        # -- KOREKSI FUZZY: Perbaiki typo/kesalahan STT berdasarkan kemiripan dengan kamus --
        teks_bersih_dikoreksi = hybrid_service.koreksi_fuzzy(teks_bersih, arah_terjemahan, cutoff=0.8)
        
        teks_bersih_str = " ".join(teks_bersih) if isinstance(teks_bersih, list) else str(teks_bersih)
        
        # Perbarui teks sumber agar Android menampilkan teks yang sudah dikoreksi
        if teks_bersih_dikoreksi.lower() != teks_bersih_str.lower():
            print(f"[translate_audio] Teks dikoreksi: '{teks_bersih_str}' -> '{teks_bersih_dikoreksi}'")
            teks_bersih = teks_bersih_dikoreksi
            teks_sumber = teks_bersih_dikoreksi  # Override teks asli dengan versi terkoreksi

        # 3. Hybrid Translation: Prioritas Exact Match & Phrase Matching
        teks_awal = hybrid_service.translate_hybrid(teks_bersih, arah_terjemahan)

        # 4. Neural Machine Translation (NMT)
        teks_akhir = nmt_service.translate_nmt(teks_awal, arah_terjemahan)

        # 5. Text-to-Speech (TTS): Disimpan ke folder output
        output_audio_filename = f"translated_{audio_file.filename}.mp3"
        output_audio_path = f"static/audio_output/{output_audio_filename}"
        tts_service.generate_audio(teks_akhir, arah_terjemahan, output_audio_path)

        # 6. Simpan hasil terjemahan ke database untuk disaring & diverifikasi
        try:
            if not hybrid_service.is_exact_match(teks_bersih, arah_terjemahan):
                db_service.simpan_terjemahan(
                    teks_sumber=teks_sumber,
                    teks_terjemahan=teks_akhir,
                    arah_terjemahan=arah_terjemahan,
                    sumber_input="audio"
                )
            else:
                print(f"[translate_audio] '{teks_sumber}' adalah exact match 100%. Melewati antrean verifikasi.")
        except Exception as e_db:
            print(f"[translate_audio] Info simpan DB: {e_db}")

        # 7. Mengembalikan Respons JSON ke Aplikasi Android
        return JSONResponse(content={
            "status": "success",
            "teks_asli": teks_sumber,
            "teks_sumber_dikenali": teks_sumber,
            "teks_terjemahan": teks_akhir,
            "teks_terjemahan_akhir": teks_akhir,
            "url_audio_hasil": f"/static/audio_output/{output_audio_filename}"
        })

    except Exception as e:
        print("[ERROR /api/v1/translate/audio]:", e)
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


# =============================================================================
# 2. ENDPOINT PENERJEMAHAN TEKS
# =============================================================================
@app.post("/api/translate")
async def translate_text(request: Request):
    """
    Endpoint khusus untuk inputan berupa teks ketikan (bukan suara).
    Mendukung request baik dalam bentuk JSON (@Body Retrofit) maupun Form-Data (@Field Retrofit).
    Hasil terjemahan otomatis dicatat ke antrean verifikasi penutur lokal / admin.
    """
    try:
        teks_sumber = ""
        arah_terjemahan = "indo_ke_kao"

        content_type = request.headers.get("content-type", "").lower()
        if "application/json" in content_type:
            try:
                data = await request.json()
                teks_sumber = data.get("teks") or data.get("teks_sumber") or data.get("text") or ""
                arah_terjemahan = data.get("bahasa_tujuan") or data.get("arah_terjemahan") or "indo_ke_kao"
            except Exception:
                pass

        if not teks_sumber:
            # Coba membaca dari form-data jika JSON kosong / content-type form-encoded
            try:
                form = await request.form()
                teks_sumber = form.get("teks_sumber") or form.get("teks") or form.get("text") or ""
                arah_terjemahan = form.get("arah_terjemahan") or form.get("bahasa_tujuan") or "indo_ke_kao"
            except Exception:
                pass

        teks_sumber = str(teks_sumber).strip()
        if not teks_sumber:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Teks sumber tidak boleh kosong."}
            )

        # 1. Preprocessing teks sumber ketikan
        teks_bersih = prep_service.preprocess_text(teks_sumber)

        # -- KOREKSI FUZZY: Perbaiki typo berdasarkan kemiripan dengan kamus --
        teks_bersih_dikoreksi = hybrid_service.koreksi_fuzzy(teks_bersih, arah_terjemahan, cutoff=0.8)
        
        teks_bersih_str = " ".join(teks_bersih) if isinstance(teks_bersih, list) else str(teks_bersih)
        
        if teks_bersih_dikoreksi.lower() != teks_bersih_str.lower():
            print(f"[translate_text] Teks dikoreksi: '{teks_bersih_str}' -> '{teks_bersih_dikoreksi}'")
            teks_bersih = teks_bersih_dikoreksi
            teks_sumber = teks_bersih_dikoreksi

        # 2. Hybrid Translation dengan Prioritas Exact Match & Phrase Matching
        teks_awal = hybrid_service.translate_hybrid(teks_bersih, arah_terjemahan)

        # 3. Neural Machine Translation (NMT)
        teks_akhir = nmt_service.translate_nmt(teks_awal, arah_terjemahan)

        # 4. Simpan hasil terjemahan ke database untuk disaring & diverifikasi
        try:
            if not hybrid_service.is_exact_match(teks_bersih, arah_terjemahan):
                db_service.simpan_terjemahan(
                    teks_sumber=teks_sumber,
                    teks_terjemahan=teks_akhir,
                    arah_terjemahan=arah_terjemahan,
                    sumber_input="teks"
                )
            else:
                print(f"[translate_text] '{teks_sumber}' adalah exact match 100%. Melewati antrean verifikasi.")
        except Exception as e_db:
            print(f"[translate_text] Info simpan DB: {e_db}")

        # 5. Kembalikan response lengkap ke Android
        return JSONResponse(content={
            "status": "success",
            "teks_asli": teks_sumber,
            "teks_sumber_dikenali": teks_sumber,
            "teks_terjemahan": teks_akhir,
            "teks_terjemahan_akhir": teks_akhir
        })

    except Exception as e:
        print("[ERROR POST /api/translate]:", e)
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})


# =============================================================================
# 3. ENDPOINT VERIFIKASI HASIL TERJEMAHAN (UNTUK PENUTUR LOKAL / ADMIN)
# =============================================================================
@app.get("/api/v1/verifikasi/list")
def list_verifikasi(
    status: str = "menunggu",
    dialek: str = "semua",
    q: str = "",
    limit: int = 100,
    offset: int = 0
):
    """
    Mengambil daftar riwayat hasil terjemahan user untuk disaring & diverifikasi.
    Dapat difilter berdasarkan status (menunggu, terverifikasi, ditolak, semua),
    dialek (PAGU, MODOLE, TOWILIKO, semua), dan kata kunci pencarian (q).
    """
    items = db_service.ambil_daftar_terjemahan(
        status=status,
        dialek=dialek,
        pencarian=q,
        limit=limit,
        offset=offset
    )
    stats = db_service.ambil_statistik()
    return JSONResponse(content={
        "status": "success",
        "total": len(items),
        "data": items,
        "stats": stats
    })


@app.get("/api/v1/verifikasi/stats")
def stats_verifikasi():
    """Mengambil ringkasan statistik status verifikasi terjemahan."""
    stats = db_service.ambil_statistik()
    return JSONResponse(content={"status": "success", "data": stats})


@app.post("/api/v1/verifikasi/{item_id}/proses")
async def proses_verifikasi(item_id: int, request: Request):
    """
    Memproses verifikasi satu item terjemahan:
    - Mengubah status: 'terverifikasi' (disetujui) atau 'ditolak'.
    - Menyimpan perbaikan 'teks_koreksi' dari penutur lokal jika terjemahan mesin perlu disempurnakan.
    - Mencatat nama penutur lokal / admin.
    - Jika status 'terverifikasi', secara default langsung disinkronkan ke kamus & model.
    """
    try:
        body = {}
        content_type = request.headers.get("content-type", "").lower()
        if "application/json" in content_type:
            body = await request.json()
        else:
            form = await request.form()
            body = dict(form)
    except Exception:
        body = {}

    status_baru = body.get("status", "terverifikasi")
    teks_koreksi = body.get("teks_koreksi")
    penutur_lokal = body.get("penutur_lokal") or body.get("admin")
    catatan = body.get("catatan")
    sinkron_sekarang = body.get("sinkron_sekarang", True)
    if isinstance(sinkron_sekarang, str):
        sinkron_sekarang = sinkron_sekarang.lower() in ["true", "1", "yes"]

    sukses = db_service.perbarui_verifikasi(
        item_id=item_id,
        status_baru=status_baru,
        teks_koreksi=teks_koreksi,
        penutur_lokal=penutur_lokal,
        catatan=catatan
    )

    if not sukses:
        return JSONResponse(status_code=404, content={"status": "error", "message": f"Item ID {item_id} tidak ditemukan."})

    sync_result = None
    auto_train_triggered = False
    if status_baru == "terverifikasi" and sinkron_sekarang:
        sync_result = model_updater.sinkronkan_terverifikasi_ke_kamus(item_id)
        
        # Cek apakah data baru yang belum dilatih mencapai 10
        belum_dilatih_count = db_service.hitung_belum_dilatih()
        if belum_dilatih_count >= 10:
            status_pelatihan = model_updater.ambil_status_pelatihan()
            if not status_pelatihan.get("is_training"):
                model_updater.mulai_pelatihan_nmt_async(epochs=30, batch_size=64)
                auto_train_triggered = True

    item = db_service.ambil_terjemahan_by_id(item_id)
    return JSONResponse(content={
        "status": "success",
        "message": f"Item ID {item_id} berhasil diproses dengan status '{status_baru}'.",
        "item": item,
        "sync_result": sync_result,
        "auto_train_triggered": auto_train_triggered
    })


@app.delete("/api/v1/verifikasi/{item_id}")
def hapus_verifikasi(item_id: int):
    """Menghapus item riwayat terjemahan dari database."""
    sukses = db_service.hapus_terjemahan(item_id)
    if not sukses:
        return JSONResponse(status_code=404, content={"status": "error", "message": f"Item ID {item_id} tidak ditemukan."})
    return JSONResponse(content={"status": "success", "message": f"Item ID {item_id} berhasil dihapus."})


# =============================================================================
# 4. ENDPOINT PEMBARUAN KAMUS & PELATIHAN ULANG MODEL NMT
# =============================================================================
@app.post("/api/v1/model/sinkronkan")
def sinkronkan_ke_kamus():
    """
    Menyinkronkan semua data yang berstatus 'terverifikasi' ke dalam kamus_kao_indo.csv
    dan memuat ulang kamus di memori secara real-time.
    """
    hasil = model_updater.sinkronkan_terverifikasi_ke_kamus()
    return JSONResponse(content=hasil)


@app.post("/api/v1/model/retrain")
async def retrain_model_nmt(request: Request):
    """
    Memulai proses pelatihan ulang (fine-tuning / retraining) model NMT di background
    menggunakan seluruh dataset kamus termasuk data baru yang telah diverifikasi.
    """
    epochs = 30
    batch_size = 64
    try:
        data = await request.json()
        epochs = int(data.get("epochs", epochs))
        batch_size = int(data.get("batch_size", batch_size))
    except Exception:
        pass

    # Pastikan data terverifikasi terbaru disinkronkan terlebih dahulu
    model_updater.sinkronkan_terverifikasi_ke_kamus()
    hasil = model_updater.mulai_pelatihan_nmt_async(epochs=epochs, batch_size=batch_size)
    return JSONResponse(content=hasil)


@app.get("/api/v1/model/status")
def status_pelatihan_model():
    """Mengecek status live pelatihan ulang model NMT."""
    status = model_updater.ambil_status_pelatihan()
    return JSONResponse(content={"status": "success", "data": status})


# =============================================================================
# 5. HALAMAN WEB VERIFIKASI (UNTUK PENUTUR LOKAL / ADMIN)
# =============================================================================
@app.get("/verifikasi", response_class=HTMLResponse)
def halaman_verifikasi():
    """Menampilkan halaman HTML verifikasi terjemahan."""
    html_file = os.path.join(BASE_DIR, "static", "verifikasi.html")
    if os.path.exists(html_file):
        with open(html_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>Halaman verifikasi belum tersedia di static/verifikasi.html.</h1>")


@app.get("/")
def read_root():
    return {
        "message": "Web-Server API Penerjemah Bahasa Kao - Indonesia Aktif!",
        "portal_verifikasi": "/verifikasi",
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)