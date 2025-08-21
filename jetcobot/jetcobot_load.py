import socket
import mysql.connector
import re
import time
import json
import base64
import os


# DB 연결 정보
db_config = {
    'host': '192.168.0.218',
    'user': 'guardian',
    'password': 'guardian@123',
    'database': 'guardian'
}


# 로봇 제어 PC IP/포트
HOST = "192.168.0.4"  # 여기에 실제 로봇 서버 IP 넣기
PORT = 5000


SAVE_DIR = "./jetcobot_capture_img2"


def save_b64_to_dir(b64_str, ts, prefix):
    if not b64_str or not ts:
        return None
    os.makedirs(SAVE_DIR, exist_ok=True)
    fname = f"{ts}_{prefix}.jpg"
    path = os.path.join(SAVE_DIR, fname)
    data = base64.b64decode(b64_str.encode())
    with open(path, "wb") as f:
        f.write(data)
    return path


def get_pending_request():
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, request_detail 
        FROM user_request 
        WHERE delivery_status = '처리중' 
        ORDER BY created_at ASC 
        LIMIT 1
    """)
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    if row is None:
        return None, None
    return row[0], row[1]


def update_request_status(request_id, status='완료'):
    conn = mysql.connector.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE user_request SET delivery_status = %s WHERE id = %s
    """, (status, request_id))
    conn.commit()
    cursor.close()
    conn.close()


def extract_number_from_code(code):
    match = re.search(r'\d+', code)
    return int(match.group()) if match else None


def send_robot_command(marker_num):
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect((HOST, PORT))
        msg = f"start,{marker_num}\n"
        print(f"[SEND] {msg.strip()}")
        client_socket.sendall(msg.encode())


        f = client_socket.makefile('rwb', buffering=0)


        ack = f.readline().decode().strip()
        print(f"[RECV] {ack}")


        line = f.readline().decode().strip()
        if not line.startswith("RESULT "):
            print("invalid response")
            client_socket.close()
            return False


        res = json.loads(line[len("RESULT "):])
        if res.get("status") != "ok":
            print(f"error from robot server: {res.get('message')}")
            client_socket.close()
            return False


        p1 = save_b64_to_dir(res.get("photo_before_b64"), res.get("photo_before_ts"), "before")
        p2 = save_b64_to_dir(res.get("photo_after_b64"),  res.get("photo_after_ts"),  "after")
        print("marker_id:", res.get("marker_id"))
        print("item_name:", res.get("item_name"))
        print("picked_at:", res.get("picked_at"))
        print("robot_type:", res.get("robot_type"))
        print("robot_id:", res.get("robot_id"))
        print("server_saved_before:", res.get("photo_before_path"))
        print("server_saved_after:",  res.get("photo_after_path"))
        print("client_saved_before:", p1)
        print("client_saved_after:",  p2)


        done_msg = f.readline().decode().strip()
        if done_msg == "적재 완료":
            print("[INFO] 로봇이 작업을 완료했습니다.")
            client_socket.close()
            return True


        client_socket.close()
    except Exception as e:
        print(f"[ERROR] 로봇 명령 중 오류 발생: {e}")
        try:
            client_socket.close()
        except:
            pass
    return False


def main_loop():
    print("로봇 제어 대기 시작...")
    request_id, item_codes = get_pending_request()
    print(f"현재 조회된 요청: {request_id}, {item_codes}")  # 디버그용 출력
    if request_id is not None and item_codes:
        first_code = item_codes.split(',')[0].strip()
        marker_num = extract_number_from_code(first_code)
        if marker_num is not None:
            print(f"처리 요청 {request_id}: 마커 {marker_num} 동작 시작")
            success = send_robot_command(marker_num)
            if success:
               # update_request_status(request_id, '완료')
                print(f"처리 요청 {request_id}: 상태를 완료로 변경")
            else:
                print(f"처리 요청 {request_id}: 명령 전송 실패")
        else:
            print(f"아이템 코드에서 숫자를 추출하지 못했습니다: '{first_code}'")
    else:
        print("처리할 요청이 없습니다.")


if __name__ == "__main__":
    main_loop()
