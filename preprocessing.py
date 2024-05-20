import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Bidirectional, Input

def read_raw_audio(file_path, sample_rate=16000):
    with open(file_path, 'rb') as f:
        audio = np.frombuffer(f.read(), dtype=np.int16)
    return audio.astype(np.float32) / 32768.0  # 16-bit PCM -> float32

def save_audio(file_path, audio, sample_rate=16000):
    sf.write(file_path, audio, sample_rate)

def add_noise(audio, noise_factor=0.5):
    noise = np.random.randn(len(audio))
    augmented_data = audio + noise_factor * noise
    augmented_data = augmented_data.astype(type(audio[0]))
    return augmented_data

def extract_features(signal, sr=16000, n_mfcc=13):
    mfcc_features = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    return mfcc_features.T

def build_denoise_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(TimeDistributed(Dense(1)))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def denoise_audio(model, noisy_signal):
    noisy_features = extract_features(noisy_signal)
    noisy_features = np.expand_dims(noisy_features, axis=0)  # 배치 차원 추가
    denoised_features = model.predict(noisy_features)
    denoised_features = np.squeeze(denoised_features, axis=0)  # 배치 차원 제거
    # 새로운 변환 방식: 여기서는 오디오 신호 복원이 필요
    # 예를 들어, Griffin-Lim 알고리즘을 사용할 수 있습니다.
    # 복원이 실제로 필요하지 않다면 그대로 denoised_features를 사용합니다.
    denoised_audio = librosa.feature.inverse.mfcc_to_audio(denoised_features)
    return denoised_audio

def plot_waveforms(original, noisy, denoised, sr=16000):
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.title('Original Audio')
    plt.plot(original)
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')

    plt.subplot(3, 1, 2)
    plt.title('Noisy Audio')
    plt.plot(noisy)
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')

    plt.subplot(3, 1, 3)
    plt.title('Denoised Audio')
    plt.plot(denoised)
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')

    plt.tight_layout()
    plt.show()

# 경로 설정
root_dir = r'C:\Users\yooju\OneDrive\바탕 화면\기계학습 프로젝트 폴더\raw16k\train'
output_dir = r'C:\Users\yooju\OneDrive\바탕 화면\기계학습 프로젝트 폴더\denoised'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 모델 로드 또는 학습 (여기서는 학습된 모델을 사용한다고 가정)
input_shape = (None, 13)  # 예시 입력 형태
model = build_denoise_model(input_shape)
# model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)  # 학습 단계

# 모든 서브 폴더와 파일 처리
for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.endswith('.raw'):
            file_path = os.path.join(subdir, file)
            clean_audio = read_raw_audio(file_path)
            noisy_audio = add_noise(clean_audio)

            # 노이즈 제거
            denoised_audio = denoise_audio(model, noisy_audio)

            # 결과 저장
            output_subdir = os.path.join(output_dir, os.path.relpath(subdir, root_dir))
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)

            output_file_path = os.path.join(output_subdir, file.replace('.raw', '.wav'))
            save_audio(output_file_path, denoised_audio)

            # 첫 번째 파일만 예제로 시각화
            plot_waveforms(clean_audio, noisy_audio, denoised_audio)
            break  # 예제를 위해 첫 번째 파일만 시각화 후 종료

print("노이즈 제거 및 저장 완료")
