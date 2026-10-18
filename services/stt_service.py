import speech_recognition as sr

def recognize_speech(audio_path: str, arah_terjemahan: str) -> str:
    """
    Mengubah sinyal suara yang direkam pengguna menjadi teks (Speech-to-Text).
    """
    recognizer = sr.Recognizer()
    
    try:
        with sr.AudioFile(audio_path) as source:
            # Membaca file audio yang dikirim dari Android
            audio_data = recognizer.record(source)

        # Menentukan bahasa pengenalan berdasarkan pilihan pengguna
        lang_code = "id-ID" if arah_terjemahan == "indo_ke_kao" else "id-ID" 
        
        teks_hasil = recognizer.recognize_google(audio_data, language=lang_code)
        return teks_hasil
        
    except sr.UnknownValueError:
        # Jika suara tidak dikenali / hening
        return "" 
    except sr.RequestError as e:
        print(f"[stt_service] Google Speech API RequestError: {e}")
        return ""
    except Exception as e:
        print(f"[stt_service] Error saat memproses file audio ({audio_path}): {e}")
        return ""