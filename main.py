import cv2
from ultralytics import YOLO
import mediapipe as mp

# 1. YOLO 모델 로드 (가장 가볍고 빠른 n 버전 사용)
model = YOLO('yolov8n.pt')

# 2. MediaPipe 손 인식 도구 불러오기 (YOLO 다음에 추가!)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 3. 웹캠 활성화
cap = cv2.VideoCapture(0)
print("YOLO + MediaPipe 통합 모듈이 시작되었습니다!")
print("영상 창을 클릭하고 'q'를 누르면 안전하게 종료됩니다.")

# 손 인식 모델 설정 (최대 2개의 손)
with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("카메라를 읽을 수 없습니다.")
            break

        # 거울처럼 보기 편하게 원본 영상 좌우 반전
        frame = cv2.flip(frame, 1)

        # 4. 화자(사람) 탐지
        results = model.predict(frame, conf=0.5, classes=[0], verbose=False)

        max_area = 0
        main_speaker_box = None

        # 5. 감지된 사람 중 가장 '가까이 있는(면적이 큰)' 사람 찾기
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                
                if area > max_area:
                    max_area = area
                    main_speaker_box = (x1, y1, x2, y2)

        # 6. 메인 화자가 있을 때만 독립 분리(Crop) 및 뼈대 추출 수행
        if main_speaker_box:
            x1, y1, x2, y2 = main_speaker_box
            
            # 배경을 제외하고 화자만 독립적으로 분리(Crop)
            speaker_crop = frame[y1:y2, x1:x2]
            
            if speaker_crop.size != 0:
                # 규격화 (640x480)
                speaker_crop = cv2.resize(speaker_crop, (640, 480))
                
                # --- 여기서부터 MediaPipe 적용 ---
                # Crop된 화면을 BGR에서 RGB로 변환 (MediaPipe용)
                image_rgb = cv2.cvtColor(speaker_crop, cv2.COLOR_BGR2RGB)
                
                # 손가락 뼈대 추출!
                hand_results = hands.process(image_rgb)
                
                # 뼈대가 인식되었다면 Crop된 화면 위에 뼈대 그리기
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            speaker_crop, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # -------------------------------
                
                # 원본 영상에 YOLO 박스 그리기 (확인용)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Main Speaker", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # 두 개의 창 띄우기 (Crop된 화면에 뼈대가 나옵니다!)
                cv2.imshow("1. Original Stream (YOLO Tracking)", frame)
                cv2.imshow("2. Cropped Speaker + MediaPipe Hands", speaker_crop)
                
        else:
            # 사람이 감지되지 않을 때는 원본만 출력
            cv2.imshow("1. Original Stream (YOLO Tracking)", frame)

        # 7. 종료 조건: 영상 창 클릭 후 'q' 입력
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 8. 자원 해제
cap.release()
cv2.destroyAllWindows()
print("프로그램이 정상 종료되었습니다.")
