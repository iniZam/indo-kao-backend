import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# 1. MEMUAT DATASET
# Memuat data maksimal 5.600 data bahasa Kao dan bahasa Indonesia sesuai batasan masalah
df = pd.read_csv('data/kamus_kao_indo.csv')

# Membuat list kosong untuk menampung data input dan target
sumber_teks = []
target_teks = []

# Menggabungkan 3 dialek Bahasa Kao (Pagu, Modole, Towiliko) ke dalam satu dataset pelatihan
# agar NMT mengenali semua variasi bahasa Kao lalu memetakannya ke Bahasa Indonesia
for index, row in df.iterrows():
    indo = str(row['INDONESIA'])
    
    # 1. Menambahkan pasangan dialek PAGU ke INDONESIA[cite: 1]
    sumber_teks.append(str(row['PAGU']))
    target_teks.append(indo)
    
    # 2. Menambahkan pasangan dialek MODOLE ke INDONESIA[cite: 1]
    sumber_teks.append(str(row['MODOLE']))
    target_teks.append(indo)
    
    # 3. Menambahkan pasangan dialek TOWILIKO ke INDONESIA[cite: 1]
    sumber_teks.append(str(row['TOWILIKO']))
    target_teks.append(indo)

# Menambahkan token <start> dan <end> pada bahasa target untuk proses Decoder NMT
target_teks = ['<start> ' + teks + ' <end>' for teks in target_teks]

# 2. PREPROCESSING TEKS (NORMALISASI & TOKENISASI)
# Proses mengubah huruf besar menjadi kecil dan memecah kalimat menjadi token
def normalisasi_teks(teks_list):
    return [teks.lower() for teks in teks_list] # Normalisasi huruf kecil

sumber_teks = normalisasi_teks(sumber_teks)
target_teks = normalisasi_teks(target_teks)

# Tokenisasi bahasa sumber
tokenizer_sumber = Tokenizer(filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
tokenizer_sumber.fit_on_texts(sumber_teks)
tensor_sumber = tokenizer_sumber.texts_to_sequences(sumber_teks) # Konversi ke bentuk numerik
tensor_sumber = pad_sequences(tensor_sumber, padding='post')

# Tokenisasi bahasa target
tokenizer_target = Tokenizer(filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n', lower=True)
tokenizer_target.fit_on_texts(target_teks)
tensor_target = tokenizer_target.texts_to_sequences(target_teks) # Konversi ke bentuk numerik[cite: 1]
tensor_target = pad_sequences(tensor_target, padding='post')

# 3. MEMBANGUN ARSITEKTUR MODEL NMT (RNN / LSTM)
# Menggunakan model saraf untuk memprediksi urutan kalimat agar hasil terjemahan sangat baik[cite: 1]
vocab_size_sumber = len(tokenizer_sumber.word_index) + 1
vocab_size_target = len(tokenizer_target.word_index) + 1
embedding_dim = 256
units = 512

# Bagian Encoder
encoder_inputs = Input(shape=(None,))
encoder_embedding = Embedding(vocab_size_sumber, embedding_dim)(encoder_inputs)
encoder_lstm = LSTM(units, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Bagian Decoder
decoder_inputs = Input(shape=(None,))
decoder_embedding = Embedding(vocab_size_target, embedding_dim)(decoder_inputs)
decoder_lstm = LSTM(units, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(vocab_size_target, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# 4. KOMPILASI DAN PELATIHAN MODEL
# TensorFlow dan Keras digunakan untuk membangun dan mengelola model ini[cite: 1]
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Menyiapkan data target yang digeser 1 langkah ke depan (Teacher Forcing)
target_data = np.zeros_like(tensor_target)
target_data[:, :-1] = tensor_target[:, 1:]

print(f"Memulai pelatihan NMT dengan {len(sumber_teks)} baris data...")
# Proses pelatihan dengan validation split untuk melihat akurasi
history = model.fit(
    [tensor_sumber, tensor_target], 
    target_data, 
    batch_size=64, 
    epochs=100, # Bisa disesuaikan
    validation_split=0.2 
)

# 5. MENYIMPAN MODEL
# Menyimpan model ke format .h5 untuk diimplementasikan ke Web-Server Backend[cite: 1]
model.save('ml/models/nmt_model.h5')
print("Pelatihan selesai. Model berhasil disimpan sebagai nmt_model.h5.")