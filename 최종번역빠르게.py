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
import threading  # [추가] 속도 개선을 위한 스레딩 모듈

def send_solapi_sms(to_number, translated_word):
    """솔라피를 이용해 번역된 수어를 문자로 전송하는 함수 (기존과 동일)"""
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
            json={
                'message': {
                    'to': to_number,
                    'from': FROM_NUMBER,
                    'text': text_message
                }
            }
        )
        if response.status_code == 200:
            print(f"📱 문자 전송 성공! ➔ 내용: {translated_word}")
        else:
            print(f"❌ 문자 전송 실패 ➔ 에러코드: {response.text}")
    except Exception as e:
        print(f"⚠️ 문자 전송 중 오류 발생: {e}")

# --- 나머지 설정 (기존과 동일) ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'sign_lang_model.keras')
model = load_model(model_path)
yolo_model = YOLO('yolov8n.pt') 

actions = np.array(['now_asl', 'hi_asl', 'name_asl', 'now_ksl', 'hi_ksl', 'name_ksl'])
translation_map = {
    'now_asl': 'Jigeum', 'hi_asl': 'Annyeong', 'name_asl': 'Ireum',
    'now_ksl': 'Now', 'hi_ksl': 'Hello', 'name_ksl': 'Name'
}

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
    else:
        return np.zeros(126)

def play_video(video_path):
    """별도 창에서 영상 재생 (기존과 동일하지만 이제 비동기로 실행됨)"""
    if not os.path.exists(video_path):
        return

    cap_v = cv2.VideoCapture(video_path)
    while cap_v.isOpened():
        ret, v_frame = cap_v.read()
        if not ret: break
        cv2.imshow('Cross-Translation Video', v_frame)
        if cv2.waitKey(25) & 0xFF == ord('q'): break
    cap_v.release()
    cv2.destroyWindow('Cross-Translation Video')

# --- 실시간 처리 (스레딩 적용) ---
sequence = []
current_action = "Waiting..."
last_played_action = ""
threshold = 0.8

cap = cv2.VideoCapture(0)
with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        frame = cv2.flip(frame, 1)

        results = yolo_model.predict(frame, conf=0.5, classes=[0], verbose=False)
        max_area = 0
        main_speaker_box = None

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                if area > max_area:
                    max_area = area
                    main_speaker_box = (x1, y1, x2, y2)

        if main_speaker_box:
            x1, y1, x2, y2 = main_speaker_box
            speaker_crop = frame[y1:y2, x1:x2]
            
            if speaker_crop.size != 0:
                speaker_crop_resized = cv2.resize(speaker_crop, (640, 480))
                image_rgb = cv2.cvtColor(speaker_crop_resized, cv2.COLOR_BGR2RGB)
                results_mp = hands.process(image_rgb)

                if results_mp.multi_hand_landmarks:
                    for hand_landmarks in results_mp.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            speaker_crop_resized, 
                            hand_landmarks, 
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                        )

                    frame[y1:y2, x1:x2] = cv2.resize(speaker_crop_resized, (x2 - x1, y2 - y1))

                    keypoints = extract_keypoints(results_mp)
                    sequence.append(keypoints)
                    sequence = sequence[-30:]

                    if len(sequence) == 30:
                        res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                        action_idx = np.argmax(res)
                        confidence = res[action_idx]

                        if confidence > threshold:
                            detected_raw = actions[action_idx]
                            current_action = translation_map[detected_raw]
                            
                            # [수정 부분] 비동기 처리 시작
                            if detected_raw in video_map and detected_raw != last_played_action:
                                # 문자 전송을 보조 쓰레드에서 실행 (병목 방지)
                                t_sms = threading.Thread(target=send_solapi_sms, args=("01054554649", current_action))
                                t_sms.daemon = True
                                t_sms.start()
                                
                                # 영상 재생을 보조 쓰레드에서 실행 (메인 화면 멈춤 방지)
                                t_video = threading.Thread(target=play_video, args=(video_map[detected_raw],))
                                t_video.daemon = True
                                t_video.start()
                                
                                last_played_action = detected_raw
                                sequence = [] 
                else:
                    sequence = []
                    last_played_action = ""
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        else:
            sequence = []
            last_played_action = ""

        cv2.rectangle(frame, (0,0), (640, 50), (245, 117, 16), -1)
        cv2.putText(frame, current_action, (250, 38), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

        cv2.imshow('9조 Final Multilingual System', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()