# face_client.py
import socket
import cv2
import numpy as np
import requests
import threading
import queue
import time
from insightface.app import FaceAnalysis
from clean_face_db import load_face_db, find_best_match

UDP_PORT = 5006
BUFFER_SIZE = 65535
FRAME_SKIP = 5
ALERT_THRESHOLD = 30

FLASK_URL = "http://192.168.0.110:5000"

def send_post(endpoint, payload=None):
    try:
        r = requests.post(f"{FLASK_URL}/{endpoint}", json=payload, timeout=1)
        print(f"[POST] {endpoint} → {r.status_code}")
    except Exception as e:
        print(f"[ERROR] {endpoint} 전송 실패: {e}")

def udp_receiver(frame_queue, stop_event):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", UDP_PORT))
    sock.settimeout(1)

    print(f"[UDP] {UDP_PORT} 포트에서 수신 대기 중...")
    buffer = b""

    while not stop_event.is_set():
        try:
            data, _ = sock.recvfrom(BUFFER_SIZE)
            buffer += data
            if len(data) < BUFFER_SIZE:  # 마지막 조각
                np_arr = np.frombuffer(buffer, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if frame is not None:
                    if not frame_queue.full():
                        frame_queue.put(frame)
                buffer = b""
        except socket.timeout:
            continue

def face_recognizer(frame_queue, stop_event):
    face_db = load_face_db()
    if not face_db:
        print("[ERROR] 얼굴 DB 없음")
        return

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0)

    missing_count = 0
    last_state = None

    while not stop_event.is_set():
        if frame_queue.empty():
            time.sleep(0.01)
            continue

        frame = frame_queue.get()
        faces = app.get(frame)

        if faces:
            face = faces[0]
            x1, y1, x2, y2 = map(int, face.bbox)
            emb = face.embedding
            name, score = find_best_match(emb, face_db)
            label = f"{name} ({score:.2f})"

            if last_state != name:
                send_post("log", {"name": name})
                last_state = name
            missing_count = 0

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        else:
            missing_count += 1
            if missing_count >= ALERT_THRESHOLD and last_state != "HELP":
                send_post("alert_help")
                last_state = "HELP"

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            stop_event.set()
            break

    send_post("resume_patrol")

def main():
    frame_queue = queue.Queue(maxsize=2)
    stop_event = threading.Event()

    threading.Thread(target=udp_receiver, args=(frame_queue, stop_event), daemon=True).start()
    face_recognizer(frame_queue, stop_event)

if __name__ == "__main__":
    main()



