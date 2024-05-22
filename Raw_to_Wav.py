import os
import numpy as np
import soundfile as sf

# PCM 파일 로드 함수
def load_pcm_file(filename, sr=16000, dtype=np.int16):
    with open(filename, 'rb') as f:
        pcm_data = np.frombuffer(f.read(), dtype=dtype)
    return pcm_data.astype(np.float32), sr

# 폴더 경로
input_folder = 'C:/ML_Project/raw16k/train/MLSH0'
output_folder = 'C:/ML_Project/raw16k/train/MLSH0_test'

# 출력 폴더가 없으면 생성
os.makedirs(output_folder, exist_ok=True)

# 입력 폴더 내의 모든 파일을 처리
for filename in os.listdir(input_folder):
    if filename.endswith('.raw'):
        raw_file_path = os.path.join(input_folder, filename)
        wav_file_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.wav')

        # PCM 파일 로드 (16kHz, 16bit, mono)
        y, sr = load_pcm_file(raw_file_path)

        # PCM 데이터의 범위를 [-1, 1]로 정규화
        y = y / np.max(np.abs(y))

        # raw 데이터를 wav 파일로 저장
        sf.write(wav_file_path, y, sr, format='WAV', subtype='PCM_16')

        print(f'Converted {raw_file_path} to {wav_file_path}')
