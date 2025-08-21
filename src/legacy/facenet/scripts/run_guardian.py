import socket
import cv2
import numpy as np
from recognize_utils import process_recognition
from fall_detection_utils import load_models, detect_fall
import warnings
warnings.filterwarnings('ignore')

# UDP 설정
UDP_IP = "0.0.0.0"
UDP_PORT = 5000
BUFFER_SIZE = 65536

# 모델 경로
YOLO_MODEL_PATH = "./yolov8n-pose.pt"
XGB_MODEL_PATH = "./00_Models/model_weights.xgb"
FALL_THRESHOLD = 0.5

def receive_frame(sock):
    try:
        data, _ = sock.recvfrom(BUFFER_SIZE)
        np_data = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        return frame
    except socket.timeout:
        print("[WARNING] 프레임 수신 시간 초과")
        return None
    except Exception as e:
        print(f"[ERROR] 프레임 수신 실패: {e}")
        return None

def processor():
    print("[START] Guardian 시스템 작동 중...")

    # UDP 소켓 생성
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    sock.settimeout(0.5)

    model_yolo, model_xgb = load_models(YOLO_MODEL_PATH, XGB_MODEL_PATH)
    last_fall_state = [False]

    while True:
        frame = receive_frame(sock)
        if frame is None:
            continue

        # 얼굴 인식
        frame, name = process_recognition(frame)

        # 쓰러짐 감지
        try:
            results = model_yolo.predict(frame, verbose=False)[0]
            if results.keypoints is not None and len(results.keypoints.data) > 0:
                for kp in results.keypoints.data:
                    keypoints = kp[:, :2].cpu().numpy()
                    frame, is_fall = detect_fall(frame, model_yolo, model_xgb, last_fall_state)
                    label = "Fall" if is_fall else "Normal"
                    color = (0, 0, 255) if is_fall else (0, 255, 0)
                    cv2.putText(frame, label, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        except Exception as e:
            print(f"[ERROR] 쓰러짐 감지 중 예외 발생: {e}")

        cv2.imshow("Guardian View", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    sock.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    processor()







