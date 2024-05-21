import numpy as np
import librosa
import soundfile as sf

# PCM 파일 로드
def load_pcm_file(filename, sr=16000, dtype=np.int16):
    with open(filename, 'rb') as f:
        pcm_data = np.frombuffer(f.read(), dtype=dtype)
    return pcm_data.astype(np.float32), sr

# PCM 파일 경로
pcm_file = 'C:/ML_Project/raw16k/test/fmcc_test_0001.raw'
wav_file = 'C:/ML_Project/raw16k/test/fmcc_test_0001.wav'

# PCM 파일 로드 (16kHz, 16bit, mono)
y, sr = load_pcm_file(pcm_file)

# PCM 데이터의 범위를 [-1, 1]로 정규화
y = y / np.max(np.abs(y))

# raw 데이터를 wav 파일로 저장
sf.write(wav_file, y, sr, format='WAV', subtype='PCM_16')

# 변환 후 wav 파일 로드 및 확인
y_loaded, sr_loaded = librosa.load(wav_file, sr=sr)

# 프리-엠퍼시스 적용
pre_emphasis = 0.97
y_preemphasized = np.append(y_loaded[0], y_loaded[1:] - pre_emphasis * y_loaded[:-1])

# MFCC 계산
mfccs = librosa.feature.mfcc(y=y_preemphasized, sr=sr_loaded, n_mfcc=13)

# 기본 주파수 (F0) 계산
f0, voiced_flag, voiced_probs = librosa.pyin(y_loaded, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), frame_length=2048, hop_length=512)

# 유효한 F0 값 필터링
valid_f0 = f0[np.isfinite(f0)]

# 유효한 F0 값이 있을 때 평균 계산
if len(valid_f0) > 0:
    mean_f0 = np.mean(valid_f0)
else:
    mean_f0 = float('nan')

print(f'MFCC Shape: {mfccs.shape}')
print(f'Mean F0: {mean_f0:.2f} Hz')
