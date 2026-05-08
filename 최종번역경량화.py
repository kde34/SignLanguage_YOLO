import cv2
import numpy as np
import os
from ultralytics import YOLO
import mediapipe as mp
from tensorflow.keras.models import load_model

import requests
import datetime
import uuid
import hmac
import hashlib
import threading

# --- [다이어트 설정 1] 프레임 스킵 변수 ---
FRAME_SKIP = 3  # 3프레임마다 1번만 분석 (나머지는 화면만 출력)
YOLO_SKIP = 10  # YOLO(사람 찾기)는 10프레임마다 1번만 수행
frame_count = 0
main_speaker_box = None # 마지막으로 찾은 사람 위치 기억

def send_solapi_sms(to_number, translated_word):
    """솔라피 문자 전송 함수 (기존 로직 유지)"""
    API_KEY = "NCSUA3MEFMSQTU1C"
    API_SECRET = "S0RHL4WUCCNDBFBFGDXROJUNXKZIWRAI"
    FROM_NUMBER = "01054554649" 
    
    date = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    salt = str(uuid.uuid1().hex)
    data = date + salt
    signature = hmac.new(API_SECRET.encode(), data.encode(), hashlib.sha256).hexdigest()
    auth_header = f'HMAC-SHA256 apiKey={API_KEY}, date={date}, salt={salt}, signature={signature}'
    
    text_message = f"[9조 실시간 수어 번역]\n방금 인식된 수어는 '{translated_word}' 입니다!"
    
    try:
        response = requests.post(
            'https://api.solapi.com/messages/v4/send',
            headers={'Authorization': auth_header, 'Content-Type': 'application/json'},
            json={'message': {'to': to_number, 'from': FROM_NUMBER, 'text': text_message}}
        )
    except Exception as e:
        print(f"⚠️ SMS 오류: {e}")

# --- 모델 및 설정 로드 (기존 동일) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = load_model(os.path.join(BASE_DIR, 'sign_lang_model.keras'))
yolo_model = YOLO('yolov8n.pt') 

actions = np.array(['now_asl', 'hi_asl', 'name_asl', 'now_ksl', 'hi_ksl', 'name_ksl'])
translation_map = {'now_asl': 'Jigeum', 'hi_asl': 'Annyeong', 'name_asl': 'Ireum',
                   'now_ksl': 'Now', 'hi_ksl': 'Hello', 'name_ksl': 'Name'}

video_map = {
    'now_ksl': os.path.join(BASE_DIR, 'asl_videos', 'asl_now.mp4'),
    'hi_ksl': os.path.join(BASE_DIR, 'asl_videos', 'asl_hi.mp4'),
    'name_ksl': os.path.join(BASE_DIR, 'asl_videos', 'asl_name.mp4'),
    'now_asl': os.path.join(BASE_DIR, 'ksl_videos', 'ksl_now.mp4'),
    'hi_asl': os.path.join(BASE_DIR, 'ksl_videos', 'ksl_hi.mp4'),
    'name_asl': os.path.join(BASE_DIR, 'ksl_videos', 'ksl_name.mp4')
}

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(hand_results):
    if hand_results.multi_hand_landmarks:
        res = []
        for hand_landmarks in hand_results.multi_hand_landmarks:
            lst = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
            res.append(lst)
        flat_data = np.concatenate(res)
        if len(flat_data) < 126: 
            flat_data = np.concatenate([flat_data, np.zeros(126 - len(flat_data))])
        return flat_data[:126]
    return np.zeros(126)

def play_video(video_path):
    if not os.path.exists(video_path): return
    cap_v = cv2.VideoCapture(video_path)
    while cap_v.isOpened():
        ret, v_frame = cap_v.read()
        if not ret: break
        cv2.imshow('Cross-Translation Video', v_frame)
        if cv2.waitKey(25) & 0xFF == ord('q'): break
    cap_v.release()
    cv2.destroyWindow('Cross-Translation Video')

sequence = []
current_action = "Waiting..."
last_played_action = ""
threshold = 0.8

# --- [다이어트 설정 2] 카메라 해상도 낮추기 ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # 640 -> 320
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240) # 480 -> 240

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        frame_count += 1
        frame = cv2.flip(frame, 1)

        # [다이어트 핵심] 분석 주기 조절 (3프레임당 1번만 분석)
        if frame_count % FRAME_SKIP == 0:
            
            # [다이어트] 사람 찾기(YOLO)는 10프레임에 한 번만 수행
            if frame_count % YOLO_SKIP == 0:
                results = yolo_model.predict(frame, conf=0.5, classes=[0], verbose=False)
                max_area = 0
                temp_box = None
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        area = (x2 - x1) * (y2 - y1)
                        if area > max_area:
                            max_area = area
                            temp_box = (x1, y1, x2, y2)
                if temp_box: main_speaker_box = temp_box

            # 수어 분석 로직 (기존 동일)
            if main_speaker_box:
                x1, y1, x2, y2 = main_speaker_box
                speaker_crop = frame[y1:y2, x1:x2]
                
                if speaker_crop.size != 0:
                    # 크롭 이미지 크기를 더 작게 조정해서 연산량 감소
                    image_rgb = cv2.cvtColor(cv2.resize(speaker_crop, (160, 120)), cv2.COLOR_BGR2RGB)
                    results_mp = hands.process(image_rgb)

                    if results_mp.multi_hand_landmarks:
                        keypoints = extract_keypoints(results_mp)
                        sequence.append(keypoints)
                        sequence = sequence[-30:]

                        if len(sequence) == 30:
                            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                            action_idx = np.argmax(res)
                            if res[action_idx] > threshold:
                                detected_raw = actions[action_idx]
                                current_action = translation_map[detected_raw]
                                
                                if detected_raw in video_map and detected_raw != last_played_action:
                                    # [비동기] 문자 및 영상 실행
                                    threading.Thread(target=send_solapi_sms, args=("01054554649", current_action), daemon=True).start()
                                    threading.Thread(target=play_video, args=(video_map[detected_raw],), daemon=True).start()
                                    last_played_action = detected_raw
                                    sequence = [] 
                    else:
                        sequence = []
                        last_played_action = ""

        # 화면 출력 UI (UI는 매 프레임 업데이트하여 끊김 없어 보이게 함)
        cv2.rectangle(frame, (0,0), (320, 30), (245, 117, 16), -1)
        cv2.putText(frame, current_action, (100, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('9조 Final System (RPi4 Optimized)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()