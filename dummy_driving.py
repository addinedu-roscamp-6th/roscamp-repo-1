import time
import random
import threading
import subprocess

def patrol():
    while not stop_event.is_set():
        print("순찰중")
        time.sleep(1)

stop_event = threading.Event()

def run_sequence():
    stop_event.set()  # 순찰중 멈춤
    patrol_thread.join()

    print("이동중")
    move_duration = random.uniform(3, 7)
    time.sleep(move_duration)
    print("이동완료")

    subprocess.run(["python", "dummy_load.py"])

    print("다시 이동중")
    time.sleep(5)
    print("이동 완료")

    stop_event.clear()
    start_patrol()

def start_patrol():
    global patrol_thread
    patrol_thread = threading.Thread(target=patrol)
    patrol_thread.start()

stop_event.clear()
start_patrol()

print("엔터를 누르면 이동 시퀀스 시작 (종료하려면 Ctrl+C)")

try:
    while True:
        input()
        run_sequence()
except KeyboardInterrupt:
    stop_event.set()
    patrol_thread.join()
    print("프로그램 종료")
