import os
import sqlite3
from datetime import datetime
from typing import List, Dict, Any, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "terjemahan.db")


def get_connection() -> sqlite3.Connection:
    """Membuka koneksi ke SQLite database dengan row_factory dictionary."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Menginisialisasi tabel-tabel database jika belum ada."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS riwayat_terjemahan (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                teks_sumber TEXT NOT NULL,
                teks_terjemahan TEXT NOT NULL,
                arah_terjemahan TEXT NOT NULL,
                dialek TEXT DEFAULT 'PAGU',
                sumber_input TEXT DEFAULT 'teks', -- 'teks' atau 'audio'
                status TEXT DEFAULT 'menunggu',   -- 'menunggu', 'terverifikasi', 'ditolak'
                teks_koreksi TEXT,               -- Koreksi dari penutur lokal/admin jika ada
                penutur_lokal TEXT,              -- Nama penutur lokal/admin yang memverifikasi
                catatan TEXT,                    -- Catatan verifikasi tambahan
                disinkronkan_ke_model INTEGER DEFAULT 0, -- 1 jika sudah masuk kamus/dataset
                dilatih_ke_model INTEGER DEFAULT 0, -- 1 jika sudah dilatihkan ke nmt model
                waktu_dibuat TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                waktu_verifikasi TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON riwayat_terjemahan(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_waktu ON riwayat_terjemahan(waktu_dibuat DESC)")
        
        # Tambahkan kolom dilatih_ke_model jika belum ada (untuk kompatibilitas db lama)
        try:
            cursor.execute("ALTER TABLE riwayat_terjemahan ADD COLUMN dilatih_ke_model INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass # Kolom sudah ada
        
        conn.commit()


def tentukan_dialek(arah_terjemahan: str) -> str:
    """Mendeteksi dialek target atau sumber dari string arah_terjemahan."""
    arah = str(arah_terjemahan).lower()
    if "modole" in arah:
        return "MODOLE"
    elif "towiliko" in arah:
        return "TOWILIKO"
    return "PAGU"


def simpan_terjemahan(
    teks_sumber: str,
    teks_terjemahan: str,
    arah_terjemahan: str,
    sumber_input: str = "teks"
) -> int:
    """
    Menyimpan hasil terjemahan baru dari user (teks/audio)
    dengan status awal 'menunggu' (belum diverifikasi).
    """
    init_db()
    dialek = tentukan_dialek(arah_terjemahan)
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO riwayat_terjemahan (
                teks_sumber, teks_terjemahan, arah_terjemahan, dialek, sumber_input, status
            ) VALUES (?, ?, ?, ?, ?, 'menunggu')
        """, (
            str(teks_sumber).strip(),
            str(teks_terjemahan).strip(),
            str(arah_terjemahan).strip(),
            dialek,
            str(sumber_input).strip()
        ))
        conn.commit()
        return cursor.lastrowid


def ambil_daftar_terjemahan(
    status: Optional[str] = None,
    dialek: Optional[str] = None,
    pencarian: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
) -> List[Dict[str, Any]]:
    """
    Mengambil daftar riwayat terjemahan dengan opsi filter status, dialek, dan pencarian teks.
    """
    init_db()
    query = "SELECT * FROM riwayat_terjemahan WHERE 1=1"
    params: List[Any] = []

    if status and status.lower() not in ["semua", "all", ""]:
        query += " AND status = ?"
        params.append(status.lower())

    if dialek and dialek.lower() not in ["semua", "all", ""]:
        query += " AND dialek = ?"
        params.append(dialek.upper())

    if pencarian and pencarian.strip():
        query += " AND (teks_sumber LIKE ? OR teks_terjemahan LIKE ? OR teks_koreksi LIKE ?)"
        keyword = f"%{pencarian.strip()}%"
        params.extend([keyword, keyword, keyword])

    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def ambil_terjemahan_by_id(item_id: int) -> Optional[Dict[str, Any]]:
    """Mengambil satu riwayat terjemahan berdasarkan ID."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM riwayat_terjemahan WHERE id = ?", (item_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def ambil_statistik() -> Dict[str, int]:
    """Menghitung statistik jumlah data berdasarkan status verifikasi."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN status = 'menunggu' THEN 1 ELSE 0 END) as menunggu,
                SUM(CASE WHEN status = 'terverifikasi' THEN 1 ELSE 0 END) as terverifikasi,
                SUM(CASE WHEN status = 'ditolak' THEN 1 ELSE 0 END) as ditolak,
                SUM(CASE WHEN status = 'terverifikasi' AND disinkronkan_ke_model = 0 THEN 1 ELSE 0 END) as belum_sinkron
            FROM riwayat_terjemahan
        """)
        row = cursor.fetchone()
        return {
            "total": row["total"] or 0,
            "menunggu": row["menunggu"] or 0,
            "terverifikasi": row["terverifikasi"] or 0,
            "ditolak": row["ditolak"] or 0,
            "belum_sinkron": row["belum_sinkron"] or 0
        }


def perbarui_verifikasi(
    item_id: int,
    status_baru: str,
    teks_koreksi: Optional[str] = None,
    penutur_lokal: Optional[str] = None,
    catatan: Optional[str] = None
) -> bool:
    """
    Memverifikasi atau menolak hasil terjemahan oleh penutur lokal / admin.
    """
    init_db()
    waktu_sekarang = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE riwayat_terjemahan
            SET status = ?,
                teks_koreksi = COALESCE(?, teks_koreksi),
                penutur_lokal = COALESCE(?, penutur_lokal),
                catatan = COALESCE(?, catatan),
                waktu_verifikasi = ?
            WHERE id = ?
        """, (
            status_baru.lower(),
            teks_koreksi.strip() if teks_koreksi is not None else None,
            penutur_lokal.strip() if penutur_lokal is not None else None,
            catatan.strip() if catatan is not None else None,
            waktu_sekarang,
            item_id
        ))
        conn.commit()
        return cursor.rowcount > 0


def tandai_disinkronkan(item_ids: List[int]) -> bool:
    """Menandai daftar ID bahwa data telah dimasukkan ke dataset/kamus model."""
    init_db()
    if not item_ids:
        return True
    with get_connection() as conn:
        cursor = conn.cursor()
        placeholders = ",".join(["?"] * len(item_ids))
        cursor.execute(f"""
            UPDATE riwayat_terjemahan
            SET disinkronkan_ke_model = 1
            WHERE id IN ({placeholders})
        """, item_ids)
        conn.commit()
        return True


def ambil_terverifikasi_belum_sinkron() -> List[Dict[str, Any]]:
    """Mengambil semua terjemahan terverifikasi yang belum dimasukkan ke dataset kamus."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM riwayat_terjemahan
            WHERE status = 'terverifikasi' AND disinkronkan_ke_model = 0
            ORDER BY id ASC
        """)
        return [dict(row) for row in cursor.fetchall()]


def hitung_belum_dilatih() -> int:
    """Menghitung jumlah data terverifikasi yang sudah disinkronkan ke CSV namun belum dilatih ke model NMT."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        # Harus terverifikasi dan belum dilatih
        cursor.execute("""
            SELECT COUNT(*) as count FROM riwayat_terjemahan
            WHERE status = 'terverifikasi' AND dilatih_ke_model = 0
        """)
        row = cursor.fetchone()
        return row["count"] if row else 0


def ambil_belum_dilatih() -> List[Dict[str, Any]]:
    """Mengambil semua terjemahan terverifikasi yang belum dilatih."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM riwayat_terjemahan
            WHERE status = 'terverifikasi' AND dilatih_ke_model = 0
            ORDER BY id ASC
        """)
        return [dict(row) for row in cursor.fetchall()]


def tandai_sudah_dilatih(item_ids: List[int]) -> bool:
    """Menandai daftar ID bahwa data telah dilatihkan ke model."""
    init_db()
    if not item_ids:
        return True
    with get_connection() as conn:
        cursor = conn.cursor()
        placeholders = ",".join(["?"] * len(item_ids))
        cursor.execute(f"""
            UPDATE riwayat_terjemahan
            SET dilatih_ke_model = 1
            WHERE id IN ({placeholders})
        """, item_ids)
        conn.commit()
        return True


def hapus_terjemahan(item_id: int) -> bool:
    """Menghapus item riwayat terjemahan berdasarkan ID."""
    init_db()
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM riwayat_terjemahan WHERE id = ?", (item_id,))
        conn.commit()
        return cursor.rowcount > 0


# Pastikan tabel otomatis dibuat saat modul pertama kali diimpor
init_db()

