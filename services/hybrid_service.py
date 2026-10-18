import os
import re
from typing import Union, List
import pandas as pd

# Menentukan lokasi file CSV secara absolut agar aman dijalankan dari folder mana saja
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, 'data', 'kamus_kao_indo.csv')

KAMUS_DF = None

# Struktur data hash map untuk pencarian O(1) cepat
kamus_id_ke_pagu = {}
kamus_id_ke_modole = {}
kamus_id_ke_towiliko = {}

kamus_pagu_ke_id = {}
kamus_modole_ke_id = {}
kamus_towiliko_ke_id = {}
kamus_all_kao_ke_id = {}


def normalisasi_kunci(teks: str) -> str:
    """
    Membersihkan teks untuk pencocokan kunci kamus:
    - Lowercase
    - Menghapus tanda baca umum (titik, koma, tanda seru, tanya, tanda hubung, kurung, kutip dua)
    - Menghapus apostrof / glottal stop agar ejaan dengan/tanpa petik tetap cocok
    - Menghapus spasi berlebih
    """
    if not isinstance(teks, str):
        return ""
    t = teks.lower().strip()
    t = re.sub(r'[\.,!?\"\(\)\-–—:;/]+', ' ', t)
    t = re.sub(r"['’`]", '', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t


def load_kamus():
    """
    Memuat atau memperbarui kamus dari file CSV ke memory (hash maps).
    Fungsi ini dapat dipanggil ulang saat data baru ditambahkan/diverifikasi.
    """
    global KAMUS_DF, kamus_id_ke_pagu, kamus_id_ke_modole, kamus_id_ke_towiliko
    global kamus_pagu_ke_id, kamus_modole_ke_id, kamus_towiliko_ke_id, kamus_all_kao_ke_id

    try:
        KAMUS_DF = pd.read_csv(CSV_PATH)
    except Exception:
        # Fallback ke path relatif jika path absolut gagal
        KAMUS_DF = pd.read_csv('data/kamus_kao_indo.csv')

    # Bersihkan dictionary lama
    kamus_id_ke_pagu.clear()
    kamus_id_ke_modole.clear()
    kamus_id_ke_towiliko.clear()

    kamus_pagu_ke_id.clear()
    kamus_modole_ke_id.clear()
    kamus_towiliko_ke_id.clear()
    kamus_all_kao_ke_id.clear()

    for _, row in KAMUS_DF.iterrows():
        indo_raw = str(row['INDONESIA']).strip()
        pagu_raw = str(row['PAGU']).strip()
        modole_raw = str(row['MODOLE']).strip()
        towiliko_raw = str(row['TOWILIKO']).strip()

        k_indo = normalisasi_kunci(indo_raw)
        k_pagu = normalisasi_kunci(pagu_raw)
        k_modole = normalisasi_kunci(modole_raw)
        k_towiliko = normalisasi_kunci(towiliko_raw)

        # Fallback jika ada sel dialek yang kosong di CSV
        val_pagu = pagu_raw or modole_raw or towiliko_raw or indo_raw
        val_modole = modole_raw or pagu_raw or towiliko_raw or indo_raw
        val_towiliko = towiliko_raw or pagu_raw or modole_raw or indo_raw

        if k_indo:
            kamus_id_ke_pagu[k_indo] = val_pagu
            kamus_id_ke_modole[k_indo] = val_modole
            kamus_id_ke_towiliko[k_indo] = val_towiliko

        if k_pagu:
            kamus_pagu_ke_id[k_pagu] = indo_raw
            kamus_all_kao_ke_id[k_pagu] = indo_raw
        if k_modole:
            kamus_modole_ke_id[k_modole] = indo_raw
            kamus_all_kao_ke_id[k_modole] = indo_raw
        if k_towiliko:
            kamus_towiliko_ke_id[k_towiliko] = indo_raw
            kamus_all_kao_ke_id[k_towiliko] = indo_raw

    print(f"[hybrid_service] Kamus berhasil dimuat ulang ({len(KAMUS_DF)} entri).")


import difflib

# Muat kamus pertama kali saat modul di-import
load_kamus()


def koreksi_fuzzy(teks_bersih: Union[str, list], arah_terjemahan: str = "indo_ke_kao", cutoff: float = 0.8) -> str:
    """
    Mengoreksi teks input (typo/kesalahan dengar) berdasarkan kamus dataset
    menggunakan algoritma string similarity (difflib).
    Mencocokkan seluruh kalimat atau kata per kata.
    """
    if not teks_bersih:
        return ""
        
    if isinstance(teks_bersih, list):
        teks_str = " ".join([str(t) for t in teks_bersih if str(t).strip()])
    else:
        teks_str = str(teks_bersih).strip()
        
    arah = str(arah_terjemahan).strip().lower()
    is_ke_indonesia = (
        "ke_indonesia" in arah
        or "ke_indo" in arah
        or "ke_id" in arah
        or arah.startswith("kao_")
        or arah in ["kao_ke_indo", "daerah_ke_id"]
    )

    if is_ke_indonesia:
        if "modole" in arah:
            kamus_target = {**kamus_all_kao_ke_id, **kamus_modole_ke_id}
        elif "towiliko" in arah:
            kamus_target = {**kamus_all_kao_ke_id, **kamus_towiliko_ke_id}
        elif "pagu" in arah:
            kamus_target = {**kamus_all_kao_ke_id, **kamus_pagu_ke_id}
        else:
            kamus_target = kamus_all_kao_ke_id
    else:
        if "modole" in arah:
            kamus_target = kamus_id_ke_modole
        elif "towiliko" in arah:
            kamus_target = kamus_id_ke_towiliko
        else:
            kamus_target = kamus_id_ke_pagu

    kalimat_norm = normalisasi_kunci(teks_str)
    if not kalimat_norm:
        return teks_str

    kamus_keys = list(kamus_target.keys())
    
    # 1. Coba cocokkan seluruh kalimat terlebih dahulu
    matches_kalimat = difflib.get_close_matches(kalimat_norm, kamus_keys, n=1, cutoff=cutoff)
    if matches_kalimat:
        return matches_kalimat[0]
        
    # 2. Coba cocokkan kata per kata jika kalimat utuh tidak memenuhi cutoff
    kata_list = kalimat_norm.split()
    hasil = []
    for kata in kata_list:
        match_kata = difflib.get_close_matches(kata, kamus_keys, n=1, cutoff=cutoff)
        if match_kata:
            hasil.append(match_kata[0])
        else:
            hasil.append(kata)
            
    return " ".join(hasil)


def is_exact_match(teks_bersih: Union[str, list], arah_terjemahan: str = "indo_ke_kao") -> bool:
    """
    Mengecek apakah keseluruhan kalimat sudah ada 100% di dalam dataset/kamus.
    """
    if not teks_bersih:
        return False

    if isinstance(teks_bersih, list):
        teks_str = " ".join([str(t) for t in teks_bersih if str(t).strip()])
    else:
        teks_str = str(teks_bersih).strip()

    arah = str(arah_terjemahan).strip().lower()
    is_ke_indonesia = (
        "ke_indonesia" in arah
        or "ke_indo" in arah
        or "ke_id" in arah
        or arah.startswith("kao_")
        or arah in ["kao_ke_indo", "daerah_ke_id"]
    )

    if is_ke_indonesia:
        if "modole" in arah:
            kamus_target = {**kamus_all_kao_ke_id, **kamus_modole_ke_id}
        elif "towiliko" in arah:
            kamus_target = {**kamus_all_kao_ke_id, **kamus_towiliko_ke_id}
        elif "pagu" in arah:
            kamus_target = {**kamus_all_kao_ke_id, **kamus_pagu_ke_id}
        else:
            kamus_target = kamus_all_kao_ke_id
    else:
        if "modole" in arah:
            kamus_target = kamus_id_ke_modole
        elif "towiliko" in arah:
            kamus_target = kamus_id_ke_towiliko
        else:
            kamus_target = kamus_id_ke_pagu

    kalimat_norm = normalisasi_kunci(teks_str)
    return kalimat_norm in kamus_target


def translate_hybrid(teks_bersih: Union[str, list], arah_terjemahan: str = "indo_ke_kao") -> str:
    """
    Menerapkan Hybrid Translation dengan Prioritas Exact Match:
    1. PRIORITAS UTAMA: Cocok 100% Seluruh Kalimat
       Jika satu kalimat penuh cocok persis di kamus, langsung gunakan hasil tersebut
       tanpa memecah kalimat menjadi kata per kata.
    2. PRIORITAS KEDUA: Cocok Frase / Kumpulan Kata Terpanjang (Greedy Longest Match First)
       Sebelum dipecah per kata tunggal, cari kelompok kata (n-gram) yang cocok 100%
       (misal: 'air putih', 'ayam goreng', 'selamat pagi apa kabar').
    3. PRIORITAS KETIGA: Penerjemahan Kata Tunggal
       Sisa kata yang tidak masuk dalam kumpulan kata diterjemahkan per kata.
    4. Kata yang tidak ditemukan di kamus tetap dipertahankan.
    """
    if not teks_bersih:
        return ""

    if isinstance(teks_bersih, list):
        teks_str = " ".join([str(t) for t in teks_bersih if str(t).strip()])
    else:
        teks_str = str(teks_bersih).strip()

    if not teks_str:
        return ""

    arah = str(arah_terjemahan).strip().lower()

    # Cek apakah arah terjemahan ke Bahasa Indonesia
    is_ke_indonesia = (
        "ke_indonesia" in arah
        or "ke_indo" in arah
        or "ke_id" in arah
        or arah.startswith("kao_")
        or arah in ["kao_ke_indo", "daerah_ke_id"]
    )

    if is_ke_indonesia:
        if "modole" in arah:
            kamus_target = {**kamus_all_kao_ke_id, **kamus_modole_ke_id}
        elif "towiliko" in arah:
            kamus_target = {**kamus_all_kao_ke_id, **kamus_towiliko_ke_id}
        elif "pagu" in arah:
            kamus_target = {**kamus_all_kao_ke_id, **kamus_pagu_ke_id}
        else:
            kamus_target = kamus_all_kao_ke_id
    else:
        # Arah Indonesia ke Bahasa Daerah (Pagu, Modole, Towiliko)
        if "modole" in arah:
            kamus_target = kamus_id_ke_modole
        elif "towiliko" in arah:
            kamus_target = kamus_id_ke_towiliko
        else:
            # Default ke dialek Pagu jika tidak dispesifikasikan
            kamus_target = kamus_id_ke_pagu

    kalimat_norm = normalisasi_kunci(teks_str)
    if not kalimat_norm:
        return teks_str

    # =========================================================================
    # PRIORITAS 1: Cek apakah seluruh kalimat cocok 100% di kamus
    # =========================================================================
    if kalimat_norm in kamus_target:
        return kamus_target[kalimat_norm]

    # =========================================================================
    # PRIORITAS 2: Cek kumpulan kata / frase terpanjang (Greedy N-gram)
    # sebelum dipecah menjadi kata tunggal
    # =========================================================================
    kata_list = kalimat_norm.split()
    n = len(kata_list)
    hasil_terjemahan = []
    i = 0
    max_ngram = 16  # Sesuai panjang frase maksimum di kamus

    while i < n:
        cocok = False
        batas_k = min(max_ngram, n - i)

        # Cek dari panjang frase terbesar (batas_k) hingga 2 kata
        for k in range(batas_k, 1, -1):
            sub_frase = " ".join(kata_list[i : i + k])
            if sub_frase in kamus_target:
                hasil_terjemahan.append(kamus_target[sub_frase])
                i += k
                cocok = True
                break

        # =====================================================================
        # PRIORITAS 3: Kata tunggal jika tidak ada kumpulan kata yang cocok
        # =====================================================================
        if not cocok:
            kata_tunggal = kata_list[i]
            if kata_tunggal in kamus_target:
                hasil_terjemahan.append(kamus_target[kata_tunggal])
            else:
                hasil_terjemahan.append(kata_tunggal)
            i += 1

    return " ".join(hasil_terjemahan)