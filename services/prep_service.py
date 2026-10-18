import re
import nltk
from nltk.tokenize import word_tokenize

# Otomatis unduh resource NLTK jika belum tersedia
for res in ['punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'tokenizers/{res}')
    except LookupError:
        try:
            nltk.download(res, quiet=True)
        except Exception as e:
            print(f"[prep_service] Info: gagal mengunduh NLTK {res}: {e}")

def preprocess_text(teks_sumber: str) -> list:
    """
    Mempersiapkan teks mentah hasil pengenalan suara sebelum diterjemahkan.
    """
    if not teks_sumber:
        return []

    # 1. Spacing: Memperbaiki spasi antar kata agar terhindar dari kata yang menempel.
    # Contoh: memisahkan huruf yang menempel dengan tanda baca.
    teks_spacing = re.sub(r'(?<=[a-zA-Z])(?=[.,!?])|(?<=[.,!?])(?=[a-zA-Z])', ' ', str(teks_sumber))

    # 2. Normalisasi: Pengubahan huruf besar menjadi huruf kecil dan penyeragaman.
    teks_normal = teks_spacing.lower().strip()

    # 3. Tokenisasi: Memecah kalimat menjadi unit kata (token).
    try:
        tokens = word_tokenize(teks_normal)
    except Exception as e:
        # Fallback jika NLTK tokenizer bermasalah
        tokens = re.findall(r'\b\w+\b', teks_normal)

    # 4. Filtering: Penghapusan kata tidak penting (noise/stop words) seperti kata pengisi dan tanda baca terisolasi.
    stop_words = {"aduh", "eh", "em", "hmm"} 
    tokens_filtered = [
        word for word in tokens 
        if word not in stop_words and not re.match(r'^[.,!?\'"()\-–—:;]+$', word)
    ]

    return tokens_filtered