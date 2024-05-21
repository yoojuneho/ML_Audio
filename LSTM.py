# LSTM 기법을 이용한 잡음 제거
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, TimeDistributed
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# 잡음 추가 함수
def add_noise(y, noise_factor=0.1):
    noise = np.random.randn(len(y))
    y_noisy = y + noise_factor * noise
    y_noisy = np.clip(y_noisy, -1., 1.)
    return y_noisy

# 데이터 준비
def load_and_preprocess_data(filename, sr=16000):
    y, sr = librosa.load(filename, sr=sr)
    y = y / np.max(np.abs(y))  # 정규화
    y_noisy = add_noise(y)     # 잡음 추가
    return y, y_noisy

# 데이터 로드
filename = 'C:/ML_Project/raw16k/test_wav/fmcc_test_0001.wav'
clean_data, noisy_data = load_and_preprocess_data(filename)

# LSTM 모델 정의
timesteps = 200  # 각 샘플의 시간 단계 수
input_dim = 1  # 각 입력의 특성 수

noisy_data = noisy_data[:len(noisy_data) - len(noisy_data) % timesteps]
clean_data = clean_data[:len(clean_data) - len(clean_data) % timesteps]

noisy_data = noisy_data.reshape((-1, timesteps, input_dim))
clean_data = clean_data.reshape((-1, timesteps, input_dim))

input_layer = Input(shape=(timesteps, input_dim))
encoded = LSTM(128, return_sequences=True)(input_layer)
encoded = LSTM(64, return_sequences=True)(encoded)
decoded = LSTM(128, return_sequences=True)(encoded)
decoded = TimeDistributed(Dense(input_dim))(decoded)

# 모델 컴파일
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# 모델 학습
autoencoder.fit(noisy_data, clean_data, epochs=50, batch_size=16, shuffle=True)

# 잡음 제거
denoised_data = autoencoder.predict(noisy_data)
denoised_data = denoised_data.flatten()

# 결과 저장
output_wav_file = 'C:/ML_Project/raw16k/fmcc_test_0001_denoised_lstm.wav'
sf.write(output_wav_file, denoised_data, 16000)

# 결과 시각화
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
librosa.display.waveshow(clean_data.flatten(), sr=16000)
plt.title("Original Clean Audio")
plt.subplot(3, 1, 2)
librosa.display.waveshow(noisy_data.flatten(), sr=16000)
plt.title("Noisy Audio")
plt.subplot(3, 1, 3)
librosa.display.waveshow(denoised_data, sr=16000)
plt.title("Denoised Audio (LSTM)")
plt.tight_layout()
plt.show()

print(f'Noise-reduced audio saved to {output_wav_file}')
