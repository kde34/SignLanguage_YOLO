import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

# --- 1. 기본 세팅 (수집한 폴더명과 반드시 일치해야 함) ---
DATA_PATH = os.path.join('SignLang_Data') 

# 우리 조의 진짜 수어 리스트 (6개 클래스)
actions = np.array(['now_asl', 'hi_asl', 'name_asl', 'now_ksl', 'hi_ksl', 'name_ksl'])

no_sequences = 30     # 단어당 30개 영상 세트
sequence_length = 30  # 영상 하나당 30프레임

# 문자를 숫자로 변환 (now_asl: 0, hi_asl: 1, ..., name_ksl: 5)
label_map = {label:num for num, label in enumerate(actions)}

# --- 2. 수집된 .npy 데이터 로드 ---
print("📚 9조의 수어 데이터를 불러오는 중입니다...")
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            # 파일 경로 확인 후 데이터 로드
            file_path = os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num))
            res = np.load(file_path)
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences) # 특징점 데이터 (문제지)
y = to_categorical(labels).astype(int) # 정답 레이블 (정답지)

# 학습 데이터와 테스트 데이터 분리 (5%는 검증용)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# --- 3. LSTM 신경망 모델 설계 ---
model = Sequential()
# input_shape=(프레임수, 특징점수) -> 30프레임 동안 126개 좌표 변화 분석
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 126)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax')) # 6개 단어 중 확률 출력

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# --- 4. 모델 학습 시작 ---
print(f"\n🚀 총 {len(actions)}개 단어 학습을 시작합니다!")
model.fit(X_train, y_train, epochs=200) # 200번 반복 학습

# --- 5. 학습 결과 저장 ---
model.save('sign_lang_model.keras')
print("\n🎉 학습 완료! 'sign_lang_model.keras' 파일이 생성되었습니다.")