import os
import numpy as np

def load_face_db(path="face_db"):
    face_db = {}
    if not os.path.exists(path):
        os.makedirs(path)
    for file in os.listdir(path):
        if file.endswith(".npy"):
            emb = np.load(os.path.join(path, file))
            if emb.shape == (512,):
                name = os.path.splitext(file)[0]
                face_db[name] = emb
    return face_db

def find_best_match(embedding, face_db):
    from sklearn.metrics.pairwise import cosine_similarity

    names = list(face_db.keys())
    db_embs = np.array(list(face_db.values()))
    if db_embs.ndim == 1:
        db_embs = db_embs.reshape(1, -1)
    embedding = embedding.reshape(1, -1)

    sims = cosine_similarity(embedding, db_embs)[0]
    best_idx = np.argmax(sims)
    best_name = names[best_idx]
    best_score = sims[best_idx]
    return best_name, best_score

