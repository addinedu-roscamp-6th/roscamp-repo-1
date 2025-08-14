import socket
import json
import base64
import time
import cv2
import numpy as np
import os
import sys
from datetime import datetime
from pymycobot.mycobot280 import MyCobot280

HOST = "0.0.0.0"
PORT = 5000
ROBOT_TYPE = "JetCobot"
ROBOT_ID = "JET-01"
SAVE_DIR = "./jetcobot_capture_img2"

name_to_id = {
    "FileA": 1,
    "FileB": 2,
    "FileC": 3,
    "FileD": 4,
    "CoffeeA": 5,
    "CoffeeB": 6,
    "CoffeeC": 7,
    "CoffeeD": 8,
    "SnackA": 9,
    "SnackB": 10,
    "SnackC": 11,
    "SnackD": 12,
}


def resolve_marker_id(val: str):
    if val in name_to_id:
        return name_to_id[val], val
    if val.isdigit():
        return int(val), None
    return None, None


def save_and_encode(frame, prefix: str):
    os.makedirs(SAVE_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{ts}_{prefix}.jpg"
    path = os.path.join(SAVE_DIR, fname)

    cv2.imwrite(path, frame)
    ok, buf = cv2.imencode(".jpg", frame)
    b64 = base64.b64encode(buf.tobytes()).decode() if ok else None
    return b64, path, ts


def pick_and_place_and_photos(target_id: int, item_name_hint: str | None = None):
    mc = MyCobot280("/dev/ttyJETCOBOT", 1_000_000)
    mc.thread_lock = True

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    time.sleep(2)

    arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    arucoParams = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)

    data = np.load("camera_calibration.npz")
    K = data["K"]
    dist = data["dist"]

    marker_length = 0.02
    objp = np.array(
        [
            [-marker_length / 2, marker_length / 2, 0],
            [marker_length / 2, marker_length / 2, 0],
            [marker_length / 2, -marker_length / 2, 0],
            [-marker_length / 2, -marker_length / 2, 0],
        ],
        dtype=np.float32,
    )

    mc.send_angles([0, 117, -150, -4, 1.58, -45.61], 50)
    time.sleep(1)

    mc.send_angles([0, 117, -150, -37.96, 1.58, -45.61], 50)
    time.sleep(2)

    mc.send_angles([0, 117, -150, -4, 1.58, -45.61], 50)
    time.sleep(1)

    mc.send_angles([0, 125, -50, -75, 5, -45], 50, 0)
    time.sleep(1)

    mc.set_gripper_value(100, 50)
    time.sleep(1)

    # before photo
    ret, f1 = cap.read()
    for _ in range(5):
        ret, f1 = cap.read()
        time.sleep(0.05)
    before_b64, before_path, before_ts = (None, None, None)
    if ret:
        before_b64, before_path, before_ts = save_and_encode(f1, "before")

    # detect target marker (timeout 15s)
    found = False
    start_time = time.time()
    target_corners = None
    frame = None

    while time.time() - start_time < 15:
        ok, frame = cap.read()
        if not ok:
            continue
        corners, ids, _ = detector.detectMarkers(frame)
        if ids is None:
            continue
        ids = ids.flatten()
        for i, mid in enumerate(ids):
            if mid == target_id:
                target_corners = corners[i]
                found = True
                break
        if found:
            break

    if not found:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        return {"status": "error", "message": "marker not found"}

    imgp = target_corners[0].astype(np.float32)
    success, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist)
    if not success:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        return {"status": "error", "message": "solvePnP failed"}

    x, y, z = (tvec.reshape(-1) * 1000).tolist()

    center_x = int(np.mean(target_corners[0][:, 0]))
    center_y = int(np.mean(target_corners[0][:, 1]))
    H, W = frame.shape[:2]
    x_centered = center_x - (W // 2)
    y_centered = (H // 2) - center_y

    if x_centered > 0 and y_centered > 0:
        quadrant = "++"
    elif x_centered < 0 and y_centered > 0:
        quadrant = "-+"
    elif x_centered < 0 and y_centered < 0:
        quadrant = "--"
    else:
        quadrant = "+-"

    corrections = {
        "++": {"pos": (-75, 0, 50), "rot": (-85, -47, -95)},
        "-+": {"pos": (-70, -5, 50), "rot": None},
        "--": {"pos": (-70, -5, 60), "rot": None},
        "+-": {"pos": (-80, 0, 50), "rot": (-80, -47, -95)},
    }

    dx, dy, dz = corrections[quadrant]["pos"]
    rot_correction = corrections[quadrant]["rot"]

    x_robot, y_robot, z_robot = z + dx, -x + dy, -y + dz

    current_coords = mc.get_coords()
    target_coords = current_coords.copy()
    target_coords[0] += float(x_robot)
    target_coords[1] += float(y_robot)
    target_coords[2] += float(z_robot)

    if rot_correction is not None:
        target_coords[3], target_coords[4], target_coords[5] = rot_correction
    else:
        target_coords[3], target_coords[4], target_coords[5] = (
            current_coords[3],
            current_coords[4],
            current_coords[5],
        )

    mc.send_angles([0, 40, -140, 100, 0, -45], 50)
    time.sleep(3)

    mc.send_coords(target_coords, 50, 0)
    time.sleep(3)

    mc.set_gripper_value(0, 50)
    picked_at = datetime.now().isoformat(timespec="seconds")
    time.sleep(1)

    re_coords = mc.get_coords()
    re_coords[2] += 25
    mc.send_coords(re_coords, 50, 0)
    time.sleep(2)

    mc.send_angles([0, 40, -140, 100, 0, -45], 50)
    time.sleep(3)

    mc.send_angles([-95.53, -27.33, -24.08, -39.28, 10.45, -62.84], 50)
    time.sleep(3)

    mc.send_coords([-82.5, -267.5, 110, -173.41, 0, -135.64], 50, 0)
    time.sleep(3)

    mc.set_gripper_value(100, 50)
    time.sleep(1)

    mc.send_angles([-95.53, -22.5, -22.5, -40, 2.5, -45.61], 50)
    time.sleep(3)

    mc.send_angles([0, 125, -50, -75, 5, -45], 50, 0)
    time.sleep(3)

    # after photo
    ret2, f2 = cap.read()
    for _ in range(5):
        ret2, f2 = cap.read()
        time.sleep(0.05)
    after_b64, after_path, after_ts = (None, None, None)
    if ret2:
        after_b64, after_path, after_ts = save_and_encode(f2, "after")

    time.sleep(1)
    cap.release()
    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass

    mc.send_angles([0, 117, -150, -4, 1.58, -45.61], 50)
    time.sleep(3)

    mc.send_angles([0, 117, -150, -37.96, 1.58, -45.61], 50)
    time.sleep(3)

    return {
        "status": "ok",
        "marker_id": int(target_id),
        "item_name": item_name_hint,
        "picked_at": picked_at,
        "robot_type": ROBOT_TYPE,
        "robot_id": ROBOT_ID,
        "photo_before_b64": before_b64,
        "photo_before_ts": before_ts,
        "photo_before_path": before_path,
        "photo_after_b64": after_b64,
        "photo_after_ts": after_ts,
        "photo_after_path": after_path,
    }


def serve():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, PORT))
    s.listen(1)

    conn, addr = s.accept()
    f = conn.makefile("rwb", buffering=0)

    conn.sendall(b"ACK\n")

    for line in f:
        msg = line.decode().strip()
        if not msg:
            break

        if not msg.lower().startswith("start,"):
            resp = {"status": "error", "message": "invalid command"}
            conn.sendall(("RESULT " + json.dumps(resp, ensure_ascii=False) + "\n").encode())
            continue

        val = msg.split(",", 1)[1].strip()
        marker_id, item_name = resolve_marker_id(val)
        if marker_id is None:
            resp = {"status": "error", "message": "unknown id or name"}
            conn.sendall(("RESULT " + json.dumps(resp, ensure_ascii=False) + "\n").encode())
            continue

        result = pick_and_place_and_photos(marker_id, item_name)
        conn.sendall(("RESULT " + json.dumps(result, ensure_ascii=False) + "\n").encode())


if __name__ == "__main__":
    serve()
