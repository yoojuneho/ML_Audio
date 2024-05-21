# LDA 차원 축소 기법을 이용한 GMM 
import os
import numpy as np
import librosa
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# MFCC 특징을 추출하는 함수
def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)  # 더 많은 MFCC 계수 사용
    energy = np.sum(librosa.feature.rms(y=y, frame_length=2048, hop_length=512) ** 2, axis=0)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    delta_energy = np.gradient(energy)
    delta2_energy = np.gradient(delta_energy)
    features = np.vstack([mfcc, energy, delta_mfcc, delta2_mfcc, delta_energy, delta2_energy])
    return features.T

# 파일 경로 및 라벨 파일 경로 설정
data_folder = 'C:/ML_Project/raw16k/test_wav'
label_file = 'C:/ML_Project/fmcc_test_ref.txt'

# 라벨 파일 읽기
file_paths = []
labels = []
with open(label_file, 'r') as f:
    for line in f:
        parts = line.strip().split()
        file_paths.append(os.path.join(data_folder, parts[0] + '.wav'))
        labels.append(0 if parts[1] == 'male' else 1)  # male은 0, female은 1

# 특징 추출 및 레이블 생성
X = []
y = []

for file_path, label in zip(file_paths, labels):
    mfcc_features = extract_mfcc(file_path)
    X.append(mfcc_features)
    y.extend([label] * mfcc_features.shape[0])

X = np.vstack(X)
y = np.array(y)

# 특징 스케일링
scaler = StandardScaler()
X = scaler.fit_transform(X)

# LDA 적용
lda = LinearDiscriminantAnalysis(n_components=1)  # LDA는 클래스 - 1 차원으로 축소
X_lda = lda.fit_transform(X, y)

# 학습 및 테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_lda, y, test_size=0.2, random_state=42)

# GMM 모델 학습
gmm_male = GaussianMixture(n_components=32, covariance_type='full', random_state=42)
gmm_female = GaussianMixture(n_components=32, covariance_type='full', random_state=42)

# 남성, 여성 데이터로 각각의 모델 학습
gmm_male.fit(X_train[y_train == 0])
gmm_female.fit(X_train[y_train == 1])

# 예측
log_likelihood_male = gmm_male.score_samples(X_test)
log_likelihood_female = gmm_female.score_samples(X_test)

# 성별 분류
y_pred = np.where(log_likelihood_male > log_likelihood_female, 0, 1)

# 정확도 평가
accuracy = accuracy_score(y_test, y_pred)
print(f'LDA Accuracy: {accuracy * 100:.2f}%')
