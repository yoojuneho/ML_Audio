import numpy as np
import pywt
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

# 웨이브렛 변환을 이용한 잡음 제거 함수
def wavelet_denoising(y, wavelet='db8', level=2, mode='soft'):
    coeffs = pywt.wavedec(y, wavelet, level=level, mode='per')
    sigma = np.median(np.abs(coeffs[-level])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(y)))
    denoised_coeffs = coeffs[:]
    denoised_coeffs[1:] = [pywt.threshold(i, value=uthresh, mode=mode) for i in denoised_coeffs[1:]]
    y_denoised = pywt.waverec(denoised_coeffs, wavelet, mode='per')
    return y_denoised

# PCM 파일 로드 함수
def load_pcm_file(filename, sr=16000, dtype=np.int16):
    with open(filename, 'rb') as f:
        pcm_data = np.frombuffer(f.read(), dtype=dtype)
    return pcm_data.astype(np.float32), sr

# PCM 파일 경로
pcm_file = 'C:/ML_Project/raw16k/test_wav/fmcc_test_0001.wav'
output_wav_file = 'C:/ML_Project/raw16k/fmcc_test_0001_denoised_wavelet.wav'

# PCM 파일 로드 (16kHz, 16bit, mono)
y, sr = load_pcm_file(pcm_file)

# PCM 데이터의 범위를 [-1, 1]로 정규화
y = y / np.max(np.abs(y))

# 노이즈 제거
y_denoised = wavelet_denoising(y, wavelet='db8', level=2, mode='soft')

# 노이즈 제거된 신호를 wav 파일로 저장
sf.write(output_wav_file, y_denoised, sr, format='WAV', subtype='PCM_16')

# 결과 시각화
plt.figure(figsize=(12, 8))
plt.subplot(3, 1, 1)
librosa.display.waveshow(y, sr=sr)
plt.title("Original Audio")
plt.subplot(3, 1, 2)
librosa.display.waveshow(y_denoised, sr=sr)
plt.title("Denoised Audio (Wavelet)")
plt.tight_layout()
plt.show()

print(f'Noise-reduced audio saved to {output_wav_file}')
