import time
import random
import threading
import subprocess
import mysql.connector


def get_connection():
    return mysql.connector.connect(
        host="192.168.0.218",
        port=3306,
        user="guardian",
        password="guardian@123",
        database="guardian",
    )


def update_pinky_status(status):
    try:
        connection = get_connection()
        cursor = connection.cursor()
        sql = "INSERT INTO pinky_status (status) VALUES (%s)"
        cursor.execute(sql, (status,))
        connection.commit()
    finally:
        cursor.close()
        connection.close()


stop_event = threading.Event()
pause_count = 0
patrol_thread = None


def patrol():
    while not stop_event.is_set():
        print("순찰중")
        update_pinky_status("순찰중")
        time.sleep(1)


def start_patrol():
    global patrol_thread
    patrol_thread = threading.Thread(target=patrol)
    patrol_thread.start()


def run_sequence():
    global pause_count
    stop_event.set()
    patrol_thread.join()

    print("이동중")
    update_pinky_status("이동중")
    move_duration = random.uniform(3, 7)
    time.sleep(move_duration)
    print("이동완료")
    update_pinky_status("일시정지")
    pause_count += 1

    print("jetcobot_load.py 실행 중...")
    subprocess.run(["python", "jetcobot_load.py"])
    print("jetcobot_load.py 실행 완료")

    update_pinky_status("배송중")

    time.sleep(5)
    print("배송 완료")
    update_pinky_status("일시정지")
    pause_count += 1

    # 배송 완료 시 user_request 테이블에서 처리중인 요청 하나를 완료로 상태 변경
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE user_request
            SET delivery_status='완료'
            WHERE delivery_status='처리중'
            ORDER BY created_at ASC
            LIMIT 1
        """)
        conn.commit()
    except Exception as e:
        print(f"user_request 업데이트 중 오류: {e}")
    finally:
        cursor.close()
        conn.close()

    if pause_count >= 2:
        print("3초 후 순찰재개")
        time.sleep(3)
        update_pinky_status("순찰중")
        pause_count = 0
        stop_event.clear()
        start_patrol()
    else:
        print("일시정지 중. 다음 입력 대기 중.")


def exists_pending_request():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM user_request WHERE delivery_status = '처리중'")
    count = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    return count > 0


def monitor_pending_request(poll_interval=3):
    start_patrol()
    while True:
        time.sleep(poll_interval)
        if exists_pending_request():
            print("새로운 처리중 요청 감지! 작업 실행")
            run_sequence()
        else:
            print("처리중 요청 없음, 계속 순찰 중...")


try:
    monitor_pending_request()
except KeyboardInterrupt:
    stop_event.set()
    if patrol_thread and patrol_thread.is_alive():
        patrol_thread.join()
    print("프로그램 종료")
