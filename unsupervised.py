import os
import numpy as np
import librosa
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.stats import mode

# MFCC 특징을 추출하는 함수
def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    energy = np.sum(librosa.feature.rms(y=y, frame_length=2048, hop_length=512) ** 2, axis=0)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    delta_energy = np.gradient(energy)
    delta2_energy = np.gradient(delta_energy)
    features = np.vstack([mfcc, energy, delta_mfcc, delta2_mfcc, delta_energy, delta2_energy])
    return features.T

# 파일 경로 설정
data_folder = 'C:/ML_Project/raw16k/train/MLSH0_test'
file_paths = [os.path.join(data_folder, fname) for fname in os.listdir(data_folder) if fname.endswith('.wav')]

# 특징 추출
X = []
file_indices = []

for i, file_path in enumerate(file_paths):
    mfcc_features = extract_mfcc(file_path)
    X.append(mfcc_features)
    file_indices.extend([i] * mfcc_features.shape[0])

X = np.vstack(X)

# 특징 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# GMM 모델 학습 (클러스터 수는 2로 설정)
gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
gmm.fit(X_scaled)

# 클러스터 할당 예측
labels = gmm.predict(X_scaled)

# 파일 단위로 클러스터 라벨의 최빈값을 성별로 간주
file_labels = []
for i in range(len(file_paths)):
    indices = np.array(file_indices) == i
    if np.any(indices):
        file_label = mode(labels[indices])
        if isinstance(file_label.mode, np.ndarray):
            file_labels.append('male' if file_label.mode[0] == 0 else 'feml')
        else:
            file_labels.append('male' if file_label == 0 else 'feml')
    else:
        file_labels.append('unknown')

# 결과를 텍스트 파일로 저장
output_file = 'C:/ML_Project/raw16k/train_predict/MLSH0_predict.txt'
with open(output_file, 'w') as f:
    for gender in file_labels:
        f.write(f'{gender}\n')

print(f'Predicted genders have been saved to {output_file}')
