from gtts import gTTS

def generate_audio(teks_akhir: str, arah_terjemahan: str, output_path: str):
    """
    Mengubah teks hasil terjemahan NMT menjadi suara (Text-to-Speech).
    """
    # gTTS mendukung bahasa Indonesia ('id'). 
    # Untuk output bahasa daerah, pendekatan serupa bisa digunakan selama di-support,
    # atau disimulasikan menggunakan logat 'id'.
    lang_code = 'id' 
    
    # Membuat objek gTTS dan menyimpannya sebagai file audio (misal: .mp3)
    tts = gTTS(text=teks_akhir, lang=lang_code, slow=False)
    tts.save(output_path)