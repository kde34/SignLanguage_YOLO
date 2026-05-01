import cv2
import numpy as np
import os
import mediapipe as mp
from ultralytics import YOLO

# 1. 9조 전용 설정 (6개 클래스, 30세트, 30프레임)
DATA_PATH = os.path.join('SignLang_Data') 
actions = np.array(['now_asl', 'hi_asl', 'name_asl', 'now_ksl', 'hi_ksl', 'name_ksl']) 
no_sequences = 30 
sequence_length = 30 

# 폴더 자동 생성
for action in actions: 
    for sequence in range(no_sequences):
        try: os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except: pass

# 2. 모델 및 도구 로드
yolo_model = YOLO('yolov8n.pt') # 김다은 학생 담당
mp_hands = mp.solutions.hands # 김하진 학생 담당
mp_drawing = mp.solutions.drawing_utils

def extract_keypoints(results):
    """MediaPipe 결과를 126개 좌표로 변환"""
    lh = np.array([[res.x, res.y, res.z] for res in results.multi_hand_landmarks[0].landmark]).flatten() if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0 else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.multi_hand_landmarks[1].landmark]).flatten() if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 1 else np.zeros(21*3)
    return np.concatenate([lh, rh])

# 3. 실시간 수집 루프 시작
cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    
    for action in actions:
        sequence = 0
        while sequence < no_sequences:
            frame_num = 0
            temp_keypoints = [] # 한 세트(30프레임)를 임시 저장할 리스트
            
            print(f"준비: {action} - {sequence}번 세트 (스페이스바를 누르면 시작)")

            while frame_num < sequence_length:
                success, frame = cap.read()
                frame = cv2.flip(frame, 1)
                
                # STEP 1: YOLO 화자 탐지 (김다은)
                yolo_results = yolo_model.predict(frame, conf=0.5, classes=[0], verbose=False)
                
                # STEP 2: MediaPipe 특징점 추출 (김하진)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results_mp = hands.process(rgb_frame)

                # 화면 안내 UI
                cv2.putText(frame, f'ACTION: {action} (#{sequence})', (15,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                if frame_num == 0:
                    cv2.putText(frame, 'Ready? Press SPACE to Record', (100,240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                else:
                    cv2.putText(frame, f'RECORDING: {frame_num}/30', (150,240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                cv2.imshow('9조 Sign Language Collector', frame)
                key = cv2.waitKey(10) & 0xFF

                # 촬영 시작 제어 (스페이스바 클릭 시)
                if key == ord(' ') or frame_num > 0:
                    keypoints = extract_keypoints(results_mp)
                    temp_keypoints.append(keypoints)
                    frame_num += 1
                
                if key == ord('q'): # 프로그램 강제 종료
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

            # STEP 3: 촬영 데이터 확인 및 저장/재촬영 결정
            # 촬영 완료 후 정지 화면에서 물어봄
            cv2.putText(frame, 'DONE! [SPACE]: SAVE / [r]: RETRY', (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.imshow('9조 Sign Language Collector', frame)
            
            while True:
                confirm_key = cv2.waitKey(0) & 0xFF
                if confirm_key == ord(' '): # 저장
                    for idx, kp in enumerate(temp_keypoints):
                        np.save(os.path.join(DATA_PATH, action, str(sequence), str(idx)), kp)
                    sequence += 1
                    print(f"성공: {action} {sequence-1}번 저장 완료!")
                    break
                elif confirm_key == ord('r'): # 재촬영
                    print(f"재촬영: {action} {sequence}번을 다시 찍습니다.")
                    break

    cap.release()
    cv2.destroyAllWindows()