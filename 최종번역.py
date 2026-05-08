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

def send_solapi_sms(to_number, translated_word):
    """솔라피를 이용해 번역된 수어를 문자로 전송하는 함수"""
    
    # 솔라피 정보 입력창
    API_KEY = "NCSUA3MEFMSQTU1C"
    API_SECRET = "S0RHL4WUCCNDBFBFGDXROJUNXKZIWRAI"
    FROM_NUMBER = "01054554649" # 예: 01012345678 (하이픈 빼고 숫자만)
    
    # 솔라피 공식 암호화(HMAC) 인증 과정
    date = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    salt = str(uuid.uuid1().hex)
    data = date + salt
    signature = hmac.new(API_SECRET.encode(), data.encode(), hashlib.sha256).hexdigest()
    auth_header = f'HMAC-SHA256 apiKey={API_KEY}, date={date}, salt={salt}, signature={signature}'
    
    # 스마트폰으로 날아갈 문자 내용 구성
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

# [핵심] 현재 실행 중인 파일의 위치를 자동으로 추적!
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. 모델 로드 
model_path = os.path.join(BASE_DIR, 'sign_lang_model.keras')
model = load_model(model_path)
yolo_model = YOLO('yolov8n.pt') 

# 2. 클래스 정의
actions = np.array(['now_asl', 'hi_asl', 'name_asl', 'now_ksl', 'hi_ksl', 'name_ksl'])

# 3. 텍스트 매핑
translation_map = {
    'now_asl': 'Jigeum', 'hi_asl': 'Annyeong', 'name_asl': 'Ireum',
    'now_ksl': 'Now', 'hi_ksl': 'Hello', 'name_ksl': 'Name'
}

# [핵심] 절대 경로를 사용한 영상 교차 매핑
video_map = {
    'now_ksl': os.path.join(BASE_DIR, 'asl_videos', 'asl_now.mp4'),
    'hi_ksl': os.path.join(BASE_DIR, 'asl_videos', 'asl_hi.mp4'),
    'name_ksl': os.path.join(BASE_DIR, 'asl_videos', 'asl_name.mp4'),
    'now_asl': os.path.join(BASE_DIR, 'ksl_videos', 'ksl_now.mp4'),
    'hi_asl': os.path.join(BASE_DIR, 'ksl_videos', 'ksl_hi.mp4'),
    'name_asl': os.path.join(BASE_DIR, 'ksl_videos', 'ksl_name.mp4')
}

# --- Mediapipe 세팅 ---
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
    """지정된 경로의 영상을 별도 창에서 재생"""
    if not os.path.exists(video_path):
        print(f"⚠️ 경고: '{video_path}' 파일을 찾을 수 없습니다!")
        return

    cap_v = cv2.VideoCapture(video_path)
    while cap_v.isOpened():
        ret, v_frame = cap_v.read()
        if not ret: break
        
        cv2.imshow('Cross-Translation Video', v_frame)
        if cv2.waitKey(25) & 0xFF == ord('q'): break
    
    cap_v.release()
    cv2.destroyWindow('Cross-Translation Video')

# --- 실시간 처리 변수 ---
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

        # YOLO로 사람 인식
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
                    # [핵심 시각화] 크롭된 화면에 관절과 뼈대를 그립니다
                    for hand_landmarks in results_mp.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            speaker_crop_resized, 
                            hand_landmarks, 
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3), # 관절 점: 빨간색
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)  # 이어지는 선: 초록색
                        )

                    # [핵심 시각화] 뼈대가 그려진 크롭 화면을 원래 메인 화면 크기로 늘려서 덮어쓰기
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
                            
                            # 교차 영상 재생
                            if detected_raw in video_map and detected_raw != last_played_action:
                                send_solapi_sms("01054554649", current_action)
                                play_video(video_map[detected_raw])
                                last_played_action = detected_raw
                                sequence = [] 
                else:
                    sequence = []
                    last_played_action = ""
            
            # 얼굴/상체 테두리에 그려지는 YOLO 인식 박스 (파란색으로 변경)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        else:
            sequence = []
            last_played_action = ""

        # 상단 UI 바 출력
        cv2.rectangle(frame, (0,0), (640, 50), (245, 117, 16), -1)
        cv2.putText(frame, current_action, (250, 38), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3, cv2.LINE_AA)

        cv2.imshow('9조 Final Multilingual System', frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()