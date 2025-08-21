import requests

def log_face(name):
    url = "http://localhost:5001/log"  # Flask API 주소
    data = {"name": name}
    try:
        r = requests.post(url, json=data)
        print(f"[LOG] {name} 기록됨: {r.status_code}")
    except Exception as e:
        print(f"[ERROR] 전송 실패: {e}")
