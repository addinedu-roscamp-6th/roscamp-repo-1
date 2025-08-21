import numpy as np
import pandas as pd
import cv2

def load_models(yolo_model_path, xgb_model_path, YOLO, xgb):
    model_yolo = YOLO(yolo_model_path)
    model_xgb = xgb.Booster()
    model_xgb.load_model(xgb_model_path)
    return model_yolo, model_xgb

def detect_fall(frame, model_yolo, model_xgb, last_fall_state):
    if frame is None:
        print("[WARNING] frame is None")
        return frame, False

    results = model_yolo(frame)[0]
    keypoints = results.keypoints

    if keypoints is None or len(keypoints.xy) == 0:
        return frame, False

    coords = keypoints.xy[0].cpu().numpy()
    if coords.shape[0] != 17:
        return frame, False

    df = extract_features_from_keypoints(coords)

    try:
        pred = model_xgb.predict(df.values)[0]
    except Exception as e:
        print(f"[ERROR] XGB 예외 발생: {e}")
        return frame, False

    is_fall = pred == 1

    if is_fall and not last_fall_state[0]:
        print("[ALERT] 쓰러짐 감지")

    last_fall_state[0] = is_fall
    return frame, is_fall

def extract_features_from_keypoints(keypoints):
    """
    keypoints: (17, 2) 형태의 NumPy 배열 (x, y 좌표)
    return: DataFrame 형태의 features
    """
    feature_dict = {}
    for i in range(17):
        feature_dict[f'x{i}'] = keypoints[i][0]
        feature_dict[f'y{i}'] = keypoints[i][1]

    df = pd.DataFrame([feature_dict])
    df = df[[f'x{i}' for i in range(17)] + [f'y{i}' for i in range(17)]]  # 순서 정렬
    return df


