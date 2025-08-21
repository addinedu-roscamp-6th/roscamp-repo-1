import cv2
import os
import numpy as np
import face_recognition
from datetime import datetime

# 설정
FACE_DB_DIR = 'face_db'
THRESHOLD = 0.6
CAMERA_URL = 'http://192.168.4.1:5000/camera'
FACE_SCALE = 2  # 캡처된 얼굴 확대 비율
LOG_INTERVAL = 1.0  # 초 단위 로그 기록 간격
ALERT_INTERVAL = 10.0  # 외부인 알림 간격

# 벡터 불러오기
face_db = {}
for file in os.listdir(FACE_DB_DIR):
    if file.endswith('.npy'):
        name = file[:-4]
        data = np.load(os.path.join(FACE_DB_DIR, file))
        if data.ndim == 1:
            data = data[np.newaxis, :]
        face_db[name] = data

# 비어있을 경우 오류 방지용
names = list(face_db.keys())
vectors = np.vstack(list(face_db.values())) if face_db else np.array([])

# 로그 및 외부인 캡처
last_logged = {}
last_alert = {}

def save_log(name, score):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"access_log_{date_str}.csv")

    if name not in last_logged or (now - last_logged[name]).total_seconds() >= LOG_INTERVAL:
        last_logged[name] = now
        with open(path, 'a') as f:
            f.write(f"{time_str},{name},{score:.4f}\n")

def handle_unknown(face_img):
    now = datetime.now()
    name = now.strftime("%Y%m%d_%H%M%S")
    alert_dir = "alerts"
    os.makedirs(alert_dir, exist_ok=True)

    # 알림 제한 시간
    if 'unknown' in last_alert and (now - last_alert['unknown']).total_seconds() < ALERT_INTERVAL:
        return
    last_alert['unknown'] = now

    resized = cv2.resize(face_img, (face_img.shape[1]*FACE_SCALE, face_img.shape[0]*FACE_SCALE))
    save_path = os.path.join(alert_dir, f"unknown_{name}.jpg")
    cv2.imwrite(save_path, resized)
    print(f"[⚠️] 외부인 저장됨 → {save_path}")

def main():
    print("📷 카메라 연결 중...")
    cap = cv2.VideoCapture(CAMERA_URL)
    if not cap.isOpened():
        print("❌ 카메라 열기 실패")
        return

    print("🧠 실시간 얼굴 인식 시작")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, locations)

        for (top, right, bottom, left), enc in zip(locations, encodings):
            if vectors.shape[0] == 0:
                name = "Unknown"
                score = 0
            else:
                sims = face_recognition.face_distance(vectors, enc)
                best_idx = np.argmin(sims)
                score = 1 - sims[best_idx]  # 유사도 (1 - 거리)
                name = names[best_idx] if score >= THRESHOLD else "Unknown"

            face_img = frame[top:bottom, left:right]
            label = f"{name} ({score:.2f})"
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if name == "Unknown":
                handle_unknown(face_img)
            else:
                save_log(name, score)

        cv2.imshow("MTCNN + Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
