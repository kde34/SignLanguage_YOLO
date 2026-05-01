import cv2
import numpy as np
import os
from ultralytics import YOLO
import mediapipe as mp

# --- 1. 저장할 폴더 및 파라미터 세팅 ---
DATA_PATH = os.path.join('SignLang_Data') # 저장될 최상위 폴더명
actions = np.array(['hello', 'thanks', 'love'])
no_sequences = 30     # 단어당 촬영할 영상(폴더) 개수
sequence_length = 30  # 영상 하나당 추출할 프레임 수 (약 1~1.5초)

# 폴더 자동 생성 (기존에 있으면 무시)
for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# --- 2. 좌표(x, y, z) 추출 및 Numpy 배열화 함수 ---
def extract_keypoints(hand_results):
    # 손이 화면에 인식되었다면
    if hand_results.multi_hand_landmarks:
        res = []
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # 21개 랜드마크의 x, y, z 좌표를 뽑아서 1차원 리스트로 쫙 폅니다.
            lst = np.array([[res.x, res.y, res.z] for res in hand_landmarks.landmark]).flatten()
            res.append(lst)
        
        flat_data = np.concatenate(res)
        # 손이 1개만 찍히든 2개가 찍히든 크기를 고정! (21개 * 3좌표 * 손2개 = 126개)
        # 빈칸이 생기면 모두 0으로 채워줍니다. (인공지능이 헷갈리지 않게 규격 통일)
        if len(flat_data) < 126: 
            flat_data = np.concatenate([flat_data, np.zeros(126 - len(flat_data))])
        return flat_data[:126]
    else:
        # 손이 안 보이면 126칸 모두 0으로 채움
        return np.zeros(126)

# --- 3. 모델 로드 및 웹캠 켜기 ---
model = YOLO('yolov8n.pt')
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

print("데이터 수집 모듈이 시작되었습니다!")
print("카메라 창을 한 번 클릭하고 영어 's'를 누르면 수집이 시작됩니다.")

# 수집 시작 대기 로직 (s 키를 누를 때까지 대기)
while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2.putText(frame, "Press 's' to START DATA COLLECTION", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow("Data Collection", frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

# --- 4. 본격적인 데이터 자동 수집 루프 ---
quit_flag = False # 🚩 강제 종료를 위한 깃발 세팅

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    
    for action in actions: # 'hello' -> 'thanks' -> 'love' 순서로 진행
        if quit_flag: break # 깃발이 올라가면 첫 번째 반복문 탈출
            
        for sequence in range(no_sequences): # 단어당 30번씩 촬영
            if quit_flag: break # 깃발이 올라가면 두 번째 반복문 탈출
                
            for frame_num in range(sequence_length): # 한 번 촬영할 때 30프레임 연속 찰칵!
                
                success, frame = cap.read()
                if not success: break
                
                frame = cv2.flip(frame, 1)
                
                # YOLO 화자 탐지 로직
                results = model.predict(frame, conf=0.5, classes=[0], verbose=False)
                max_area = 0
                main_speaker_box = None

                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        area = (x2 - x1) * (y2 - y1)
                        if area > max_area:
                            max_area = area
                            main_speaker_box = (x1, y1, x2, y2)

                # 메인 화자가 잡혔을 때만 뼈대 추출 및 저장
                if main_speaker_box:
                    x1, y1, x2, y2 = main_speaker_box
                    speaker_crop = frame[y1:y2, x1:x2]
                    
                    if speaker_crop.size != 0:
                        speaker_crop = cv2.resize(speaker_crop, (640, 480))
                        image_rgb = cv2.cvtColor(speaker_crop, cv2.COLOR_BGR2RGB)
                        hand_results = hands.process(image_rgb)
                        
                        if hand_results.multi_hand_landmarks:
                            for hand_landmarks in hand_results.multi_hand_landmarks:
                                mp_drawing.draw_landmarks(speaker_crop, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        
                        # ✨ [핵심] 여기서 좌표를 추출해서 Numpy 배열(.npy)로 저장합니다! ✨
                        keypoints = extract_keypoints(hand_results)
                        npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                        np.save(npy_path, keypoints)
                        
                        # --- 사용자 안내 UI (화면에 글씨 띄우기) ---
                        if frame_num == 0: 
                            # 새 촬영이 시작되기 전에 준비할 시간(2초)을 줍니다.
                            cv2.putText(speaker_crop, f"STARTING COLLECTION: '{action}' Video Num {sequence}", (15,20), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                            cv2.imshow("Data Collection", speaker_crop)
                            cv2.waitKey(2000) 
                        else: 
                            # 녹화가 진행되는 동안 빨간색 글씨 띄우기
                            cv2.putText(speaker_crop, f"Recording '{action}' Video Num {sequence}", (15,20), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                            cv2.imshow("Data Collection", speaker_crop)
                else:
                    cv2.imshow("Data Collection", frame)

                # 🚨 [수정된 부분] 중간에 끄고 싶을 때 'q' 입력
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    quit_flag = True # 🚩 "나 진짜 끝낼 거야!" 하고 깃발 번쩍 들기
                    break # 가장 안쪽 반복문 탈출

cap.release()
cv2.destroyAllWindows()
print("프로그램이 정상 종료되었습니다!")