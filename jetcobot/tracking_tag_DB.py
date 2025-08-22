import socket
import json
import base64
import time
import cv2
import numpy as np
import os
from datetime import datetime
from pymycobot.mycobot280 import MyCobot280
import mysql.connector
from mysql.connector import Error

HOST = "0.0.0.0"
PORT = 5000
ROBOT_TYPE = "jetcobot"
ROBOT_ID = "JET-01"
SAVE_DIR = "./jetcobot_capture_img2"

DB_CFG = {
    "host": "192.168.0.218",
    "port": 3306,
    "user": "guardian",
    "password": "guardian@123",
    "database": "guardian",
}

def to_sql_ts(ts: str | None) -> str | None:
    return ts.replace('T', ' ') if ts else None

name_to_id = {
    "FileA": 1, "FileB": 2, "FileC": 3, "FileD": 4,
    "CoffeeA": 5, "CoffeeB": 6, "CoffeeC": 7, "CoffeeD": 8,
    "SnackA": 9, "SnackB": 10, "SnackC": 11, "SnackD": 12,
}

RUNNING = True
mc = None
cap = None
detector = None
K = None
dist = None
objp = None

def get_db_conn():
    return mysql.connector.connect(**DB_CFG)

def log_pickup_to_db(payload: dict):
    sql = """
    INSERT INTO jetcobot_pickup_logs
    (robot_type, robot_id, request_type, requester, request_time,
     pickup_time, pickup_location_json, pickup_status, object_type,
     jetcobot_camera_snapshot_url, release_request_time, release_time, release_status)
    VALUES
    (%(robot_type)s, %(robot_id)s, %(request_type)s, %(requester)s, %(request_time)s,
     %(pickup_time)s, %(pickup_location_json)s, %(pickup_status)s, %(object_type)s,
     %(jetcobot_camera_snapshot_url)s, %(release_request_time)s, %(release_time)s, %(release_status)s)
    """
    conn = cur = None
    try:
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute(sql, payload)
        conn.commit()
        print(f"[DB][OK] inserted id={cur.lastrowid}")
        return cur.lastrowid
    except Error as e:
        print(f"[DB][ERROR] {e}\n[DB][PAYLOAD] {payload}")
        return None
    finally:
        try:
            if cur: cur.close()
            if conn and conn.is_connected(): conn.close()
        except:
            pass

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

def ensure_inited():
    global mc, cap, detector, K, dist, objp
    if mc is None:
        mc = MyCobot280("/dev/ttyJETCOBOT", 1_000_000)
        mc.thread_lock = True
    if detector is None:
        arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        arucoParams = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(arucoDict, arucoParams)
    if K is None or dist is None or objp is None:
        data = np.load("camera_calibration.npz")
        K = data["K"]
        dist = data["dist"]
        marker_length = 0.02
        objp = np.array(
            [
                [-marker_length / 2,  marker_length / 2, 0],
                [ marker_length / 2,  marker_length / 2, 0],
                [ marker_length / 2, -marker_length / 2, 0],
                [-marker_length / 2, -marker_length / 2, 0],
            ],
            dtype=np.float32,
        )
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        time.sleep(2)

def pick_and_place_and_photos(target_id: int, item_name_hint: str | None = None):
    ensure_inited()
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
    ret, f1 = cap.read()

    for _ in range(5):
        ret, f1 = cap.read(); time.sleep(0.05)
    before_b64, before_path, before_ts = (None, None, None)

    if ret:
        before_b64, before_path, before_ts = save_and_encode(f1, "before")
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
        return {"status": "error", "message": "marker not found"}
    imgp = target_corners[0].astype(np.float32)
    success, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist)

    if not success:
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
        "++": {"pos": (-70, 0, 50), "rot": (-85, -47, -95)},
        "-+": {"pos": (-70, -5, 50), "rot": None},
        "--": {"pos": (-70, -5, 60), "rot": None},
        "+-": {"pos": (-70, 0, 50), "rot": (-80, -47, -95)},
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
        target_coords[3], target_coords[4], target_coords[5] = (current_coords[3], current_coords[4], current_coords[5])

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
    mc.send_angles([-95.53, -27.33, -24.08, -30, 5, -45], 50)
    time.sleep(3)
    mc.send_coords([-82.5, -267.5, 130, -173.41, 0, -135.64], 50, 0)
    time.sleep(3)
    mc.set_gripper_value(100, 50)
    released_at = datetime.now().isoformat(timespec="seconds")
    time.sleep(1)
    mc.send_angles([-95.53, -22.5, -22.5, -40, 2.5, -45.61], 50)
    time.sleep(3)
    mc.send_angles([0, 125, -50, -75, 5, -45], 50, 0)
    time.sleep(3)

    ret2, f2 = cap.read()
    for _ in range(5):
        ret2, f2 = cap.read(); time.sleep(0.05)
    after_b64, after_path, after_ts = (None, None, None)
    if ret2:
        after_b64, after_path, after_ts = save_and_encode(f2, "after")
    time.sleep(1)

    mc.send_angles([0, 117, -150, -4, 1.58, -45.61], 50)
    time.sleep(3)
    mc.send_angles([0, 117, -150, -37.96, 1.58, -45.61], 50)
    time.sleep(3)
    
    return {
        "status": "ok",
        "marker_id": int(target_id),
        "item_name": item_name_hint,
        "picked_at": picked_at,
        "released_at": released_at,
        "robot_type": ROBOT_TYPE,
        "robot_id": ROBOT_ID,
        "photo_before_b64": before_b64,
        "photo_before_ts": before_ts,
        "photo_before_path": before_path,
        "photo_after_b64": after_b64,
        "photo_after_ts": after_ts,
        "photo_after_path": after_path,
        "pickup_location": {
            "x_mm": float(x_robot), "y_mm": float(y_robot), "z_mm": float(z_robot),
            "quadrant": quadrant, "image_center": [int(W/2), int(H/2)],
            "marker_center": [int(center_x), int(center_y)],
        },
    }

def handle_client(conn):
    f = conn.makefile("rwb", buffering=0)
    conn.sendall(b"ACK\n")
    while True:
        try:
            line = f.readline()
        except ConnectionResetError:
            break
        if not line:
            break
        msg = line.decode().strip()
        if not msg:
            continue
        if msg.lower() == "shutdown":
            resp = {"status": "ok", "message": "server shutting down"}
            conn.sendall(("RESULT " + json.dumps(resp, ensure_ascii=False) + "\n").encode())
            return "shutdown"
        if not msg.lower().startswith("start,"):
            resp = {"status": "error", "message": "invalid command"}
            conn.sendall(("RESULT " + json.dumps(resp, ensure_ascii=False) + "\n").encode())
            continue

        request_time_iso = datetime.now().isoformat(timespec="seconds")
        requester = None
        request_type = "object"
        val = msg.split(",", 1)[1].strip()
        marker_id, item_name = resolve_marker_id(val)
        if marker_id is None:
            resp = {"status": "error", "message": "unknown id or name"}
            conn.sendall(("RESULT " + json.dumps(resp, ensure_ascii=False) + "\n").encode())
            continue

        try:
            result = pick_and_place_and_photos(marker_id, item_name)
        except Exception as e:
            result = {"status": "error", "message": f"exception: {type(e).__name__}: {e}"}

        pickup_ok  = 1 if result.get("status") == "ok" else 0
        release_ok = 1 if (result.get("status") == "ok" and result.get("released_at")) else 0
        location_json = json.dumps(result.get("pickup_location"), ensure_ascii=False) if result.get("pickup_location") else None

        db_row = {
            "robot_type": ROBOT_TYPE,
            "robot_id": ROBOT_ID,
            "request_type": request_type,
            "requester": requester,
            "request_time": to_sql_ts(request_time_iso),
            "pickup_time": to_sql_ts(result.get("picked_at")),
            "pickup_location_json": location_json,
            "pickup_status": pickup_ok,
            "object_type": result.get("item_name") or f"marker_{marker_id}",
            "jetcobot_camera_snapshot_url": result.get("photo_after_path") or result.get("photo_before_path"),
            "release_request_time": to_sql_ts(result.get("picked_at")),
            "release_time": to_sql_ts(result.get("released_at")),
            "release_status": release_ok,
        }
        row_id = log_pickup_to_db(db_row)
        print(f"[DB] row_id={row_id}")

        conn.sendall(("RESULT " + json.dumps(result, ensure_ascii=False) + "\n").encode())
        conn.sendall("적재 완료\n".encode())
    return None

def serve():
    global RUNNING, cap
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(5)
    try:
        while RUNNING:
            conn, addr = s.accept()
            try:
                signal = handle_client(conn)
                if signal == "shutdown":
                    RUNNING = False
            finally:
                try:
                    conn.close()
                except:
                    pass
    finally:
        try:
            if cap is not None and cap.isOpened():
                cap.release()
        except:
            pass

if __name__ == "__main__":
    serve()
