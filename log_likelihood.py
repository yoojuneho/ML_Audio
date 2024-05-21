import os
import numpy as np
import librosa
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# MFCC 특징을 추출하는 함수
def extract_mfcc(file_path, n_mfcc=20):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    energy = np.sum(librosa.feature.rms(y=y, frame_length=2048, hop_length=512) ** 2, axis=0)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    delta_energy = np.gradient(energy)
    delta2_energy = np.gradient(delta_energy)
    features = np.vstack([mfcc, energy, delta_mfcc, delta2_mfcc, delta_energy, delta2_energy])
    return features.T

# 데이터 로드 및 처리 함수
def load_data(data_folder, label_file, n_mfcc=20):
    file_paths = []
    labels = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            file_paths.append(os.path.join(data_folder, parts[0] + '.wav'))
            labels.append(0 if parts[1] == 'male' else 1)

    X = []
    y = []
    for file_path, label in zip(file_paths, labels):
        mfcc_features = extract_mfcc(file_path, n_mfcc=n_mfcc)
        X.append(mfcc_features)
        y.extend([label] * mfcc_features.shape[0])

    X = np.vstack(X)
    y = np.array(y)
    return X, y

# GMM 학습 및 평가 함수
def train_and_evaluate(X, y, test_size=0.2, n_components=64):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    gmm_male = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
    gmm_female = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)

    gmm_male.fit(X_train[y_train == 0])
    gmm_female.fit(X_train[y_train == 1])

    log_likelihood_male = gmm_male.score_samples(X_test)
    log_likelihood_female = gmm_female.score_samples(X_test)

    # 로그 우도 합 계산
    log_likelihood_sum_male = np.sum(log_likelihood_male)
    log_likelihood_sum_female = np.sum(log_likelihood_female)

    # 성별 분류
    y_pred = np.where(log_likelihood_male > log_likelihood_female, 0, 1)

    # 정확도 평가
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
    return accuracy

# 메인 실행 코드
if __name__ == "__main__":
    data_folder = 'C:/ML_Project/raw16k/test_wav'
    label_file = 'C:/ML_Project/fmcc_test_ref.txt'

    n_mfcc = 20
    test_size = 0.2
    n_components = 64

    X, y = load_data(data_folder, label_file, n_mfcc=n_mfcc)
    accuracy = train_and_evaluate(X, y, test_size=test_size, n_components=n_components)
