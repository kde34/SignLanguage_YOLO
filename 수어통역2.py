import cv2
import numpy as np
import os
import mediapipe as mp
from ultralytics import YOLO
from tensorflow.keras.models import load_model

# 1. 모델 및 매핑 설정
actions = np.array(['now_asl', 'hi_asl', 'name_asl', 'now_ksl', 'hi_ksl', 'name_ksl'])
model = load_model('sign_lang_model.keras')

translation_map = {
    'now_asl': 'Jigeum', 'hi_asl': 'Annyeong', 'name_asl': 'Ireum',
    'now_ksl': 'Now', 'hi_ksl': 'Hello', 'name_ksl': 'Name'
}

yolo_model = YOLO('yolov8n.pt') 
mp_hands = mp.solutions.hands

def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.multi_hand_landmarks[0].landmark]).flatten() if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0 else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.multi_hand_landmarks[1].landmark]).flatten() if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 1 else np.zeros(21*3)
    return np.concatenate([lh, rh])

# --- 실시간 처리 변수 ---
sequence = []
current_action = "" # [수정] 누적 리스트 대신 현재 단어만 담는 변수
threshold = 0.8     # [수정] 신뢰도 0.8로 원복

cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results_mp = hands.process(rgb_frame)

        if results_mp.multi_hand_landmarks:
            keypoints = extract_keypoints(results_mp)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                action_idx = np.argmax(res)
                confidence = res[action_idx]

                # 80% 이상 확신할 때만 단어 교체
                if confidence > threshold:
                    current_action = translation_map[actions[action_idx]] # [핵심] 그냥 덮어씌우기
        
        else:
            sequence = []
            # 손이 화면에서 사라지면 글자도 지우고 싶으면 아래 주석을 해제해!
            # current_action = "" 

        # 3. UI 출력: 현재 인식된 단어 하나만 표시
        cv2.rectangle(frame, (0,0), (640, 50), (245, 117, 16), -1)
        
        # 단어가 있을 때만 중앙에 가깝게 출력
        cv2.putText(frame, current_action, (250, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

        cv2.imshow('9조 실시간 통역 시스템 (Single Output)', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()