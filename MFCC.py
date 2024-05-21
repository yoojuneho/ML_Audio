import librosa
import numpy as np

def extract_mfcc_features(file_path):
    # 음성 파일을 로드합니다.
    y, sr = librosa.load(file_path, sr=None)

    # 12개의 MFCC를 추출합니다.
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=12)

    # 에너지 특징을 계산합니다.
    energy = np.sum(librosa.feature.rms(y=y, frame_length=2048, hop_length=512) ** 2, axis=0)
    
    # 델타와 더블 델타 특징을 계산합니다.
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    # 델타 에너지와 더블 델타 에너지를 계산합니다.
    delta_energy = np.gradient(energy)
    delta2_energy = np.gradient(delta_energy)

    # 모든 특징을 결합하여 39차원 벡터로 만듭니다.
    features = np.vstack([mfcc, energy, delta_mfcc, delta2_mfcc, delta_energy, delta2_energy])

    return features.T  # 벡터를 전치하여 반환합니다.

# 예제 파일 경로를 지정합니다.
file_path = 'C:/ML_Project/raw16k/fmcc_test_0001_spec.wav'
features = extract_mfcc_features(file_path)

# 결과를 출력합니다.
print("MFCC Features Shape:", features.shape)
print(features)
