# 위너 기법을 이용한 잡음 제거
import numpy as np
import librosa
import soundfile as sf
from scipy.signal import wiener
import matplotlib.pyplot as plt

# PCM 파일 로드 함수
def load_pcm_file(filename, sr=16000, dtype=np.int16):
    with open(filename, 'rb') as f:
        pcm_data = np.frombuffer(f.read(), dtype=dtype)
    return pcm_data.astype(np.float32), sr

# 위너 필터를 통한 노이즈 제거 함수
def wiener_filter(y):
    y_filtered = wiener(y)
    return y_filtered

# PCM 파일 경로
pcm_file = 'C:/ML_Project/raw16k/test/fmcc_test_0001.raw'
output_wav_file = 'C:/ML_Project/raw16k/fmcc_test_0001_denoised_wiener.wav'

# PCM 파일 로드 (16kHz, 16bit, mono)
y, sr = load_pcm_file(pcm_file)

# PCM 데이터의 범위를 [-1, 1]로 정규화
y = y / np.max(np.abs(y))

# 노이즈 제거
y_denoised = wiener_filter(y)

# 노이즈 제거된 신호를 wav 파일로 저장
sf.write(output_wav_file, y_denoised, sr, format='WAV', subtype='PCM_16')

# 결과 시각화
plt.figure(figsize=(12, 6))
plt.subplot(3, 1, 1)
librosa.display.waveshow(y, sr=sr)
plt.title("Original Audio")
plt.subplot(3, 1, 2)
librosa.display.waveshow(y_denoised, sr=sr)
plt.title("Denoised Audio (Wiener Filter)")
plt.subplot(3, 1, 3)
plt.specgram(y_denoised, Fs=sr)
plt.title("Denoised Audio Spectrogram (Wiener Filter)")
plt.show()

print(f'Noise-reduced audio saved to {output_wav_file}')
