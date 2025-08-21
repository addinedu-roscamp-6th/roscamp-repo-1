


# import os
# import time
# import numpy as np
# import cv2
# from insightface.app import FaceAnalysis
# import mysql.connector
# import pickle  # 추가

# # ========== 설정 ==========
# MAX_CAPTURE = 5
# FACE_DIR = "face_db"
# os.makedirs(FACE_DIR, exist_ok=True)

# # ========== 얼굴 인식 모델 초기화 ==========
# app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
# app.prepare(ctx_id=0)

# # ========== 카메라 열기 ==========
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# if not cap.isOpened():
#     print("[ERROR] 카메라 열기 실패")
#     exit()

# # ========== 이름 입력 ==========
# name = input("등록할 이름을 입력하세요: ")

# print("[INFO] 's' 키로 얼굴 스캔, 'q' 키로 종료")
# print("[INFO] 얼굴을 다양한 각도에서 보여주고 's' 키를 5번 눌러주세요.")

# embeddings = []

# # ========== 루프 ==========
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("[ERROR] 카메라 프레임 읽기 실패")
#         break

#     faces = app.get(frame)

#     for face in faces:
#         x1, y1, x2, y2 = map(int, face.bbox)
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
#         cv2.putText(frame, f"{len(embeddings)}/{MAX_CAPTURE} 스캔됨", (x1, y1 - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

#     cv2.imshow("Register Face", frame)

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('s'):
#         if faces:
#             emb = faces[0].embedding
#             embeddings.append(emb)
#             print(f"[INFO] {len(embeddings)}장 수집됨")
#         else:
#             print("[WARN] 얼굴이 감지되지 않았습니다.")

#         if len(embeddings) >= MAX_CAPTURE:
#             print("[INFO] 5장 수집 완료. 평균 임베딩 계산 중...")
#             break

#     elif key == ord('q'):
#         print("[INFO] 등록 취소.")
#         break

# # ========== 자원 정리 ==========
# cap.release()
# cv2.destroyAllWindows()

# # ========== 등록 처리 ==========
# if len(embeddings) >= 1:
#     mean_emb = np.mean(embeddings, axis=0)
#     np.save(f"{FACE_DIR}/{name}.npy", mean_emb)
#     print(f"[INFO] {name} 평균 임베딩 저장 완료 (로컬 파일)")

#     # ✅ face_db.pkl 저장 또는 업데이트
#     face_db_path = "face_db.pkl"
#     if os.path.exists(face_db_path):
#         with open(face_db_path, "rb") as f:
#             face_db = pickle.load(f)
#     else:
#         face_db = {}

#     face_db[name] = mean_emb

#     with open(face_db_path, "wb") as f:
#         pickle.dump(face_db, f)

#     print(f"[INFO] face_db.pkl 저장 완료")

#     # ✅ MySQL 저장
#     try:
#         conn = mysql.connector.connect(
#             host="localhost",
#             user="root",
#             password="Guard1an@123",
#             database="guardian"
#         )
#         cursor = conn.cursor()

#         cursor.execute("DELETE FROM faces WHERE name = %s", (name,))
#         conn.commit()

#         query = "INSERT INTO faces (name, embedding) VALUES (%s, %s)"
#         cursor.execute(query, (name, mean_emb.tobytes()))
#         conn.commit()
#         cursor.close()
#         conn.close()
#         print(f"[INFO] MySQL에 {name} 저장 완료 (중복 제거됨)")
#     except Exception as e:
#         print(f"[ERROR] DB 저장 실패: {e}")


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import pickle
import requests
import mysql.connector
from insightface.app import FaceAnalysis

# ========== 설정 ==========
STREAM_URL = "http://192.168.0.97:5000/camera"   # 핑키 Flask MJPEG 스트림
MAX_CAPTURE = 5
FACE_DIR = "face_db"
os.makedirs(FACE_DIR, exist_ok=True)

# ========== 얼굴 인식 모델 초기화 ==========
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

# ========== 이름 입력 ==========
name = input("등록할 이름을 입력하세요: ")
print("[INFO] 's' 키로 얼굴 스캔, 'q' 키로 종료")
print(f"[INFO] 얼굴을 다양한 각도에서 보여주고 's' 키를 {MAX_CAPTURE}번 눌러주세요.")

embeddings = []
buf = bytearray()

print(f"[INFO] MJPEG 스트림 연결: {STREAM_URL}")

# ========== 프레임 수신 루프 (HTTP MJPEG) ==========
try:
    with requests.get(STREAM_URL, stream=True, timeout=10) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=2048):
            if not chunk:
                continue
            buf.extend(chunk)

            # JPEG SOI(FFD8) ~ EOI(FFD9) 구간 파싱
            s = buf.find(b'\xff\xd8')
            e = buf.find(b'\xff\xd9')
            if s != -1 and e != -1 and e > s:
                jpg = bytes(buf[s:e+2])
                del buf[:e+2]

                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                # 얼굴 인식
                faces = app.get(frame)

                # 얼굴 박스/진행도 오버레이
                for face in faces:
                    x1, y1, x2, y2 = map(int, face.bbox)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                    cv2.putText(frame, f"{len(embeddings)}/{MAX_CAPTURE} 스캔됨",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                cv2.imshow("Register Face (HTTP)", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('s'):
                    if faces:
                        emb = faces[0].embedding.astype(np.float32)  # 안전하게 float32 고정
                        embeddings.append(emb)
                        print(f"[INFO] {len(embeddings)}장 수집됨")
                        if len(embeddings) >= MAX_CAPTURE:
                            print("[INFO] 수집 완료. 평균 임베딩 계산 중...")
                            break
                    else:
                        print("[WARN] 얼굴이 감지되지 않았습니다.")
                elif key == ord('q'):
                    print("[INFO] 등록 취소.")
                    embeddings.clear()
                    break

except requests.RequestException as e:
    print(f"[ERROR] 스트림 연결 실패: {e}")

finally:
    cv2.destroyAllWindows()

# ========== 등록 처리 ==========
if len(embeddings) >= 1:
    # np.mean 기본은 float64 → 이후 DB/인식과 맞추려면 float32로 저장
    mean_emb = np.mean(embeddings, axis=0, dtype=np.float32)
    np.save(f"{FACE_DIR}/{name}.npy", mean_emb)
    print(f"[INFO] {name} 평균 임베딩 저장 완료 (로컬 파일)")

    # ✅ face_db.pkl 저장 또는 업데이트
    face_db_path = "face_db.pkl"
    try:
        if os.path.exists(face_db_path):
            with open(face_db_path, "rb") as f:
                face_db = pickle.load(f)
        else:
            face_db = {}
        face_db[name] = mean_emb
        with open(face_db_path, "wb") as f:
            pickle.dump(face_db, f)
        print(f"[INFO] face_db.pkl 저장 완료")
    except Exception as e:
        print(f"[WARN] face_db.pkl 업데이트 실패: {e}")

    # ✅ MySQL 저장 (root 계정 사용)
    try:
        conn = mysql.connector.connect(
            host="127.0.0.1",
            user="root",
            password="Guard1an@123",
            database="guardian"
        )
        cursor = conn.cursor()
        cursor.execute("DELETE FROM faces WHERE name = %s", (name,))
        conn.commit()

        query = "INSERT INTO faces (name, embedding) VALUES (%s, %s)"
        cursor.execute(query, (name, mean_emb.tobytes()))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"[INFO] MySQL에 {name} 저장 완료 (중복 제거됨)")
        print(f"[INFO] bytes={mean_emb.nbytes}  (예: 2048이면 512 float32)")
    except Exception as e:
        print(f"[ERROR] DB 저장 실패: {e}")
else:
    print("[INFO] 수집된 임베딩이 없어 저장을 생략합니다.")








