import cv2
import numpy as np
import os
import time
import threading
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from ultralytics import YOLO
import insightface
from playsound import playsound

# ---------------- 설정 ----------------
DB_DIR = "face_db"
LOG_DIR = "logs"
ALERT_DIR = "alerts"
THRESHOLD = 0.6
LOG_INTERVAL = 3
ALERT_INTERVAL = 10
FACE_SCALE = 3
CAMERA_URL = "http://192.168.4.1:5000/camera"

# ---------------- 모델 초기화 ----------------
print("✅ YOLOv8 로드 중...")
yolo_model = YOLO("yolov8n-face.pt")  # YOLOv8 얼굴 전용 모델

print("✅ InsightFace 로드 중...")
face_model = insightface.app.FaceAnalysis(name='buffalo_s')
face_model.prepare(ctx_id=0, det_size=(640, 640))

# ---------------- 얼굴 임베딩 로드 ----------------
face_embeddings = {}
for file in os.listdir(DB_DIR):
    if file.endswith(".npy"):
        name = file[:-4]
        face_embeddings[name] = np.load(os.path.join(DB_DIR, file))

if not face_embeddings:
    print("❌ 등록된 얼굴이 없습니다.")
    exit()

names = []
vectors = []
for name, emb in face_embeddings.items():
    if emb.ndim == 1:
        emb = np.expand_dims(emb, axis=0)
    vectors.append(emb)
    names.extend([name] * emb.shape[0])
vectors = np.vstack(vectors)

# ---------------- 로그 저장 ----------------
last_logged_time = {}
last_alert_time = 0

def save_log(name, similarity):
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    log_filename = os.path.join(LOG_DIR, f"{date_str}.csv")
    os.makedirs(LOG_DIR, exist_ok=True)

    if name in last_logged_time:
        if (now - last_logged_time[name]).total_seconds() < LOG_INTERVAL:
            return
    last_logged_time[name] = now

    with open(log_filename, 'a') as f:
        f.write(f"{now.strftime('%Y-%m-%d %H:%M:%S')},{name},{round(similarity,4)}\n")

def handle_unknown(face_crop):
    global last_alert_time
    now = time.time()
    if now - last_alert_time < ALERT_INTERVAL:
        return
    last_alert_time = now

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(ALERT_DIR, exist_ok=True)
    resized = cv2.resize(face_crop, (face_crop.shape[1]*FACE_SCALE, face_crop.shape[0]*FACE_SCALE))
    filename = os.path.join(ALERT_DIR, f"unknown_{timestamp}.jpg")
    cv2.imwrite(filename, resized)
    print(f"🚨 Unknown 감지됨 → 저장: {filename}")

    cv2.imshow(f"Unknown [{timestamp}]", resized)
    threading.Thread(target=close_window_after_delay, args=(f"Unknown [{timestamp}]", 10), daemon=True).start()

    try:
        playsound("alert_sound.mp3")
    except:
        pass

def close_window_after_delay(window_name, delay):
    time.sleep(delay)
    cv2.destroyWindow(window_name)

# ---------------- 실행 ----------------
def main():
    print("📷 카메라 연결 중...")
    cap = cv2.VideoCapture(CAMERA_URL)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    if not cap.isOpened():
        print("❌ 카메라 연결 실패!")
        return

    print("🚀 얼굴 인식 시작!")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = yolo_model(frame, verbose=False)[0]
        for box in results.boxes.xyxy.cpu().numpy():
            x1, y1, x2, y2 = map(int, box)
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            # 임베딩 추출
            faces = face_model.get(face_crop)
            if not faces:
                continue
            emb = faces[0].embedding

            similarities = cosine_similarity([emb], vectors)[0]
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]
            best_match = names[best_idx] if best_score >= THRESHOLD else "Unknown"

            # 표시
            label = f"{best_match} ({best_score:.2f})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # 로그 & 알림
            if best_match == "Unknown":
                handle_unknown(face_crop)
            else:
                save_log(best_match, best_score)

        cv2.imshow("YOLO + InsightFace", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
