import cv2
import pickle
import numpy as np
from insightface.app import FaceAnalysis

# 얼굴 인식 모델 초기화
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0)

# 얼굴 DB 로드
with open("face_db.pkl", "rb") as f:
    face_db = pickle.load(f)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def process_recognition(frame):
    """
    얼굴 인식을 수행하고, 가장 유사한 이름을 반환
    """
    try:
        faces = app.get(frame)
    except Exception as e:
        print(f"[ERROR] 얼굴 인식 실패: {e}")
        return frame, "unknown"

    name = "unknown"

    if faces:
        face = faces[0]  # 첫 번째 얼굴만 사용
        emb = face["embedding"]
        max_sim = 0.0

        for db_name, db_emb in face_db.items():
            sim = cosine_similarity(emb, db_emb)
            if sim > max_sim:
                max_sim = sim
                name = db_name if sim > 0.4 else "unknown"

        # 시각화용 박스와 이름 표시
        box = face["bbox"].astype(int)
        cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), 2)
        cv2.putText(frame, name, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return frame, name

def recognize_faces(frame):
    return process_recognition(frame)

