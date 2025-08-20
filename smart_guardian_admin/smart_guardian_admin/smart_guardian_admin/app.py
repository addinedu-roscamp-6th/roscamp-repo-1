from flask import Flask, render_template
from datetime import datetime, timedelta
import random

app = Flask(__name__)

# 팀원별 패턴
team_patterns = {
    "이동연":  {"role": "직원",   "status": "출입 승인", "method": "얼굴 인식"},
    "백기광":  {"role": "직원",   "status": "출입 승인", "method": "얼굴 인식"},
    "박지안":  {"role": "방문객", "status": "출입 거부", "method": "미등록 인원"},
    "위다인":  {"role": "직원",   "status": "출입 승인", "method": "얼굴 인식"},
    "김성민":  {"role": "방문객", "status": "출입 거부", "method": "미등록 인원"},
    "박진우":  {"role": "직원",   "status": "출입 승인", "method": "얼굴 인식"}
}

# 출입 기록 더미 데이터 생성
def generate_access_logs(count=30):
    names = list(team_patterns.keys())
    logs = []
    now = datetime.now()
    for i in range(count):
        name = random.choice(names)
        pattern = team_patterns[name]
        logs.append({
            "time": (now - timedelta(minutes=i*3)).strftime("%Y-%m-%d %H:%M:%S"),
            "name": name,
            "role": pattern["role"],
            "status": pattern["status"],
            "method": pattern["method"]
        })
    return logs

# 시스템 로그 더미 데이터 생성
def generate_system_logs(count=30):
    levels = ["INFO", "WARNING", "ERROR"]
    messages = {
        "이동연": ["얼굴 인식 성공", "정상 출입 처리됨"],
        "백기광": ["얼굴 인식 성공", "정상 출입 처리됨"],
        "박지안": ["미등록 인원 감지", "출입 거부됨"],
        "위다인": ["얼굴 인식 성공", "정상 출입 처리됨"],
        "김성민": ["미등록 인원 감지", "출입 거부됨"],
        "박진우": ["얼굴 인식 성공", "정상 출입 처리됨"]
    }

    logs = []
    now = datetime.now()
    for i in range(count):
        name = random.choice(list(team_patterns.keys()))
        logs.append({
            "time": (now - timedelta(minutes=i*2)).strftime("%Y-%m-%d %H:%M:%S"),
            "level": random.choice(levels),
            "message": f"{name}: {random.choice(messages[name])}"
        })
    return logs

access_logs = generate_access_logs()
system_logs = generate_system_logs()

@app.route("/access")
def access_list():
    return render_template("access.html", access_logs=access_logs)

@app.route("/logs")
def system_log_list():
    return render_template("logs.html", system_logs=system_logs)

if __name__ == "__main__":
    app.run(debug=True)
