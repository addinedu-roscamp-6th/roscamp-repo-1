import cv2
import numpy as np
import os
import insightface

app = insightface.app.FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

os.makedirs("face_db", exist_ok=True)

name = input("등록할 이름 (영문): ").strip()
img_path = input("이미지 파일 경로 (예: /home/user/face.jpg): ").strip()

img = cv2.imread(img_path)
if img is None:
    print("❌ 이미지 파일을 불러올 수 없습니다. 경로를 다시 확인해주세요.")
    exit()

faces = app.get(img)

if not faces:
    print("❌ 얼굴이 감지되지 않았습니다. 더 밝거나 정면 사진을 사용해보세요.")
    exit()

emb = faces[0].embedding
path = f"face_db/{name}.npy"

if os.path.exists(path):
    existing = np.load(path)
    new_array = np.vstack([existing, emb])
else:
    new_array = np.array([emb])

np.save(path, new_array)
print(f"✅ '{name}' 등록 완료! 총 벡터 수: {new_array.shape[0]}")
