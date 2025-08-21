


# import cv2
# import numpy as np
# import requests
# import time
# import mysql.connector
# from insightface.app import FaceAnalysis
# from sklearn.metrics.pairwise import cosine_similarity

# # ========== 설정 ==========
# STREAM_URL = "http://192.168.0.97:5000/camera"
# SIMILARITY_THRESHOLD = 0.4
# DUPLICATE_INTERVAL = 60  # 최근 1분 내 중복 방지 (초)

# # ========== 중복 인식 방지용 캐시 ==========
# last_recognized = {}

# # ========== 얼굴 임베딩 로드 ==========
# def load_embeddings_from_db():
#     conn = mysql.connector.connect(
#         host="localhost",
#         user="root",
#         password="Guard1an@123",
#         database="guardian"
#     )
#     cursor = conn.cursor()
#     cursor.execute("SELECT name, embedding FROM faces")
#     results = cursor.fetchall()
#     cursor.close()
#     conn.close()

#     embeddings = []
#     for name, emb_blob in results:
#         emb = np.frombuffer(emb_blob, dtype=np.float32)
#         embeddings.append((name, emb))
#     return embeddings

# # ========== 인식 로그 저장 ==========
# def log_recognition_to_db(name):
#     conn = mysql.connector.connect(
#         host="localhost",
#         user="root",
#         password="Guard1an@123",
#         database="guardian"
#     )
#     cursor = conn.cursor()
#     cursor.execute("INSERT INTO face_logs (name, recognized_at) VALUES (%s, NOW())", (name,))
#     conn.commit()
#     cursor.close()
#     conn.close()

# # ========== MJPEG 프레임 수신 ==========
# def receive_mjpeg_stream(url):
#     stream = requests.get(url, stream=True)
#     bytes_data = b''
#     for chunk in stream.iter_content(chunk_size=1024):
#         bytes_data += chunk
#         start = bytes_data.find(b'\xff\xd8')
#         end = bytes_data.find(b'\xff\xd9')
#         if start != -1 and end != -1 and end > start:
#             jpg = bytes_data[start:end+2]
#             bytes_data = bytes_data[end+2:]
#             frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
#             if frame is not None:
#                 # ✅ MJPEG는 기본 거꾸로 들어와서 회전
#                 frame = cv2.rotate(frame, cv2.ROTATE_180)
#                 return frame
#     return None

# # ========== 메인 ==========
# def main():
#     print("[INFO] 얼굴 인식 시작 (ESC 눌러 종료)")
#     app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
#     app.prepare(ctx_id=0)

#     known_faces = load_embeddings_from_db()

#     try:
#         while True:
#             frame = receive_mjpeg_stream(STREAM_URL)
#             if frame is None:
#                 continue

#             frame = cv2.rotate(frame, cv2.ROTATE_180)  # 회전 2번 (최종 정방향)

#             faces = app.get(frame)

#             for face in faces:
#                 x1, y1, x2, y2 = map(int, face.bbox)
#                 current_time = time.time()

#                 # 유사도 비교
#                 best_name = "Unknown"
#                 best_score = -1

#                 for name, db_emb in known_faces:
#                     sim = cosine_similarity([face.embedding], [db_emb])[0][0]
#                     if sim > best_score:
#                         best_score = sim
#                         best_name = name

#                 if best_score < SIMILARITY_THRESHOLD:
#                     best_name = "Unknown"

#                 # 중복 인식 방지: 1분 내 동일인물 제외
#                 if best_name != "Unknown":
#                     last_time = last_recognized.get(best_name, 0)
#                     if current_time - last_time >= DUPLICATE_INTERVAL:
#                         log_recognition_to_db(best_name)
#                         last_recognized[best_name] = current_time
#                         print(f"[LOG] {best_name} 인식됨 (DB 저장)")
#                     else:
#                         print(f"[SKIP] {best_name} 최근에 인식됨, 기록 생략")

#                 # 박스 & 이름 표시
#                 label = best_name if best_name != "Unknown" else "Unknown"
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
#                 cv2.putText(frame, label, (x1, y1 - 10),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

#             cv2.imshow("Face Recognition (Pinky)", frame)
#             if cv2.waitKey(1) & 0xFF == 27:
#                 print("[INFO] ESC 입력됨. 종료합니다.")
#                 break

#     except Exception as e:
#         print(f"[ERROR] 오류 발생: {e}")
#     finally:
#         cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()




#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import pickle
import numpy as np
import requests
import threading
import time
import mysql.connector
from insightface.app import FaceAnalysis

# ===== 설정 =====
STREAM_URL = "http://192.168.0.97:5000/camera"   # 핑키 Flask MJPEG
MAX_CAPTURE = 5
FACE_DIR = "face_db"
WINDOW_NAME = "Register Face (HTTP, Ultra-Low-Latency)"
FRAME_ANALYZE_STRIDE = 2
BUFFER_MAX_BYTES = 256 * 1024  # 읽기 버퍼 상한

DB_CONF = dict(
    host="127.0.0.1",
    user="root",
    password="Guard1an@123",
    database="guardian",
)

os.makedirs(FACE_DIR, exist_ok=True)

# ===== InsightFace (경량 모드) =====
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"],
    allowed_modules=['detection', 'recognition']
)
app.prepare(ctx_id=0, det_size=(320, 320))

# ===== 최신 프레임만 유지하는 리더 스레드 =====
class MJPEGLatestReader:
    def __init__(self, url, timeout=10):
        self.url = url
        self.timeout = timeout
        self.buf = bytearray()
        self.lock = threading.Lock()
        self.latest_jpg = None
        self.latest_seq = 0
        self.stop_flag = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_flag = True
        self.thread.join(timeout=1.0)

    def _run(self):
        try:
            with requests.get(self.url, stream=True, timeout=self.timeout) as r:
                r.raise_for_status()
                it = r.iter_content(chunk_size=1024)
                for chunk in it:
                    if self.stop_flag:
                        break
                    if not chunk:
                        continue
                    self.buf.extend(chunk)

                    # 버퍼 상한 초과 시, 마지막 SOI부터만 보관 → 과거 데이터 즉시 폐기
                    if len(self.buf) > BUFFER_MAX_BYTES:
                        last_soi = self.buf.rfind(b'\xff\xd8')
                        if last_soi != -1:
                            self.buf = bytearray(self.buf[last_soi:])
                        else:
                            self.buf = bytearray()

                    # 항상 "가장 마지막 완성 JPEG"를 추출
                    last_eoi = self.buf.rfind(b'\xff\xd9')
                    if last_eoi == -1:
                        continue
                    # EOI 이전의 가장 가까운 SOI 찾기
                    last_soi_before_eoi = self.buf.rfind(b'\xff\xd8', 0, last_eoi)
                    if last_soi_before_eoi == -1:
                        # SOI 없으면 앞부분 버림
                        self.buf = bytearray(self.buf[last_eoi+2:])
                        continue

                    jpg = bytes(self.buf[last_soi_before_eoi:last_eoi+2])
                    # latest_jpg 업데이트 (큐=1)
                    with self.lock:
                        self.latest_jpg = jpg
                        self.latest_seq += 1

                    # 버퍼는 마지막 EOI 이후로 트림 (밀림 방지)
                    self.buf = bytearray(self.buf[last_eoi+2:])
        except Exception as e:
            print(f"[ERROR] MJPEG reader: {e}")

    def get_latest(self):
        with self.lock:
            return (self.latest_seq, self.latest_jpg)

# ===== 저장 유틸 =====
def save_face(name: str, embeddings: list[np.ndarray]):
    mean_emb = np.mean(embeddings, axis=0, dtype=np.float32)
    np.save(os.path.join(FACE_DIR, f"{name}.npy"), mean_emb)
    print(f"[INFO] {name} 평균 임베딩 저장 완료 (로컬)")

    # face_db.pkl 업데이트(옵션)
    try:
        path = "face_db.pkl"
        if os.path.exists(path):
            with open(path, "rb") as f:
                face_db = pickle.load(f)
        else:
            face_db = {}
        face_db[name] = mean_emb
        with open(path, "wb") as f:
            pickle.dump(face_db, f)
        print("[INFO] face_db.pkl 저장 완료")
    except Exception as e:
        print(f"[WARN] face_db.pkl 업데이트 실패: {e}")

    # DB 저장
    try:
        conn = mysql.connector.connect(**DB_CONF)
        cur = conn.cursor()
        cur.execute("DELETE FROM faces WHERE name=%s", (name,))
        cur.execute("INSERT INTO faces (name, embedding) VALUES (%s, %s)",
                    (name, mean_emb.tobytes()))
        conn.commit()
        cur.close(); conn.close()
        print(f"[INFO] MySQL에 {name} 저장 완료 (bytes={mean_emb.nbytes})")
    except Exception as e:
        print(f"[ERROR] DB 저장 실패: {e}")

# ===== 메인 =====
def main():
    name = input("등록할 이름을 입력하세요: ").strip()
    if not name:
        print("[ERROR] 이름이 비었습니다. 종료.")
        return

    reader = MJPEGLatestReader(STREAM_URL, timeout=10)
    reader.start()
    print(f"[INFO] 저지연 모드로 최신 프레임만 사용: {STREAM_URL}")
    print(f"[TIP] 's' 캡처 / 'q' 종료 — {MAX_CAPTURE}장 모으면 끝")

    embeddings = []
    last_seq_used = -1
    shown_once = False
    frame_i = 0

    try:
        while True:
            seq, jpg = reader.get_latest()
            if jpg is None or seq == last_seq_used:
                # 새 프레임이 아직 없으면 잠깐 쉼
                if not shown_once:
                    # 첫 프레임 뜰 때까지 상태 표시만
                    cv2.imshow(WINDOW_NAME, np.zeros((240,320,3), dtype=np.uint8))
                    cv2.waitKey(1)
                    shown_once = True
                time.sleep(0.005)
                continue

            last_seq_used = seq
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            frame_i += 1
            do_analyze = (frame_i % FRAME_ANALYZE_STRIDE == 0)

            faces = app.get(frame) if do_analyze else []
            for face in faces:
                x1, y1, x2, y2 = map(int, face.bbox)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"{len(embeddings)}/{MAX_CAPTURE}",
                            (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                if faces:
                    embeddings.append(faces[0].embedding.astype(np.float32))
                    print(f"[INFO] {len(embeddings)}장 수집됨")
                    if len(embeddings) >= MAX_CAPTURE:
                        print("[INFO] 수집 완료. 평균 임베딩 계산…")
                        break
                else:
                    print("[WARN] 얼굴이 감지되지 않았습니다.")
            elif key == ord('q'):
                embeddings.clear()
                break

    finally:
        reader.stop()
        cv2.destroyAllWindows()

    if embeddings:
        save_face(name, embeddings)

if __name__ == "__main__":
    main()








