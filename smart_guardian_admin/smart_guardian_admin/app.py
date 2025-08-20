from flask import Flask, render_template, request, redirect, url_for
import math
from datetime import datetime, timedelta
import os

app = Flask(__name__)

# ==== Alerts (forced inject next to app = Flask(...)) ====
from flask import request, render_template
ALERTS_DUMMY = [
    {"id":1,"ts":"2025-08-15 12:00:00","type":"미등록 인원","severity":"중간","source":"LobbyCam","message":"영역 경계 통과 감지"},
    {"id":2,"ts":"2025-08-15 11:52:00","type":"쓰러짐","severity":"높음","source":"WarehouseCam","message":"자세 이상 탐지"}
]
@app.get("/alerts")
def alerts():
    data = ALERTS_DUMMY[:]
    q = request.args.get("q","").strip()
    if q:
        ql = q.lower()
        data = [a for a in data if ql in a["message"].lower() or ql in a["source"].lower() or ql in a["type"].lower()]
    return render_template("alerts.html", alerts=data, total=len(data),
                           total_pages=1, page=1, q=q, type_f="all", sev="all", sort="ts_desc",
                           type_options=["미등록 인원","쓰러짐","정상","침입 의심"],
                           sev_options=["높음","중간","낮음","정보"])

# 진단용 라우트
@app.get("/__ping")
def __ping(): return "ok", 200


# ========= 더미 데이터 =========
# ---- Alerts 더미 ----
ALERTS_DUMMY = [
  {
    "id": 1,
    "ts": "2025-08-15 14:25:56",
    "type": "미등록 인원",
    "severity": "낮음",
    "source": "Pinky-01",
    "message": "출입문 장시간 열린 상태"
  },
  {
    "id": 2,
    "ts": "2025-08-15 14:18:56",
    "type": "쓰러짐",
    "severity": "낮음",
    "source": "Jetcobot-01",
    "message": "움직임 패턴 이상"
  },
  {
    "id": 3,
    "ts": "2025-08-15 14:11:56",
    "type": "미등록 인원",
    "severity": "높음",
    "source": "WarehouseCam",
    "message": "움직임 패턴 이상"
  },
  {
    "id": 4,
    "ts": "2025-08-15 14:04:56",
    "type": "쓰러짐",
    "severity": "높음",
    "source": "Pinky-01",
    "message": "출입문 장시간 열린 상태"
  },
  {
    "id": 5,
    "ts": "2025-08-15 13:57:56",
    "type": "쓰러짐",
    "severity": "중간",
    "source": "LobbyCam",
    "message": "영역 경계 통과 감지"
  },
  {
    "id": 6,
    "ts": "2025-08-15 13:50:56",
    "type": "미등록 인원",
    "severity": "높음",
    "source": "WarehouseCam",
    "message": "영역 경계 통과 감지"
  },
  {
    "id": 7,
    "ts": "2025-08-15 13:43:56",
    "type": "침입 의심",
    "severity": "중간",
    "source": "GateCam",
    "message": "움직임 패턴 이상"
  },
  {
    "id": 8,
    "ts": "2025-08-15 13:36:56",
    "type": "쓰러짐",
    "severity": "중간",
    "source": "GateCam",
    "message": "움직임 패턴 이상"
  },
  {
    "id": 9,
    "ts": "2025-08-15 13:29:56",
    "type": "미등록 인원",
    "severity": "정보",
    "source": "Jetcobot-01",
    "message": "순찰 정상 완료"
  },
  {
    "id": 10,
    "ts": "2025-08-15 13:22:56",
    "type": "미등록 인원",
    "severity": "높음",
    "source": "GateCam",
    "message": "움직임 패턴 이상"
  },
  {
    "id": 11,
    "ts": "2025-08-15 13:15:56",
    "type": "미등록 인원",
    "severity": "낮음",
    "source": "LobbyCam",
    "message": "영역 경계 통과 감지"
  },
  {
    "id": 12,
    "ts": "2025-08-15 13:08:56",
    "type": "침입 의심",
    "severity": "낮음",
    "source": "Jetcobot-01",
    "message": "영역 경계 통과 감지"
  },
  {
    "id": 13,
    "ts": "2025-08-15 13:01:56",
    "type": "침입 의심",
    "severity": "중간",
    "source": "LobbyCam",
    "message": "영역 경계 통과 감지"
  },
  {
    "id": 14,
    "ts": "2025-08-15 12:54:56",
    "type": "미등록 인원",
    "severity": "중간",
    "source": "Pinky-01",
    "message": "순찰 정상 완료"
  },
  {
    "id": 15,
    "ts": "2025-08-15 12:47:56",
    "type": "미등록 인원",
    "severity": "낮음",
    "source": "Jetcobot-01",
    "message": "영역 경계 통과 감지"
  },
  {
    "id": 16,
    "ts": "2025-08-15 12:40:56",
    "type": "미등록 인원",
    "severity": "높음",
    "source": "Jetcobot-01",
    "message": "영역 경계 통과 감지"
  },
  {
    "id": 17,
    "ts": "2025-08-15 12:33:56",
    "type": "미등록 인원",
    "severity": "낮음",
    "source": "Pinky-01",
    "message": "출입문 장시간 열린 상태"
  },
  {
    "id": 18,
    "ts": "2025-08-15 12:26:56",
    "type": "미등록 인원",
    "severity": "높음",
    "source": "LobbyCam",
    "message": "출입문 장시간 열린 상태"
  },
  {
    "id": 19,
    "ts": "2025-08-15 12:19:56",
    "type": "미등록 인원",
    "severity": "중간",
    "source": "LobbyCam",
    "message": "움직임 패턴 이상"
  },
  {
    "id": 20,
    "ts": "2025-08-15 12:12:56",
    "type": "정상",
    "severity": "낮음",
    "source": "WarehouseCam",
    "message": "출입문 장시간 열린 상태"
  },
  {
    "id": 21,
    "ts": "2025-08-15 12:05:56",
    "type": "정상",
    "severity": "낮음",
    "source": "GateCam",
    "message": "움직임 패턴 이상"
  },
  {
    "id": 22,
    "ts": "2025-08-15 11:58:56",
    "type": "침입 의심",
    "severity": "높음",
    "source": "LobbyCam",
    "message": "영역 경계 통과 감지"
  },
  {
    "id": 23,
    "ts": "2025-08-15 11:51:56",
    "type": "미등록 인원",
    "severity": "낮음",
    "source": "Jetcobot-01",
    "message": "영역 경계 통과 감지"
  },
  {
    "id": 24,
    "ts": "2025-08-15 11:44:56",
    "type": "쓰러짐",
    "severity": "낮음",
    "source": "GateCam",
    "message": "영역 경계 통과 감지"
  },
  {
    "id": 25,
    "ts": "2025-08-15 11:37:56",
    "type": "정상",
    "severity": "중간",
    "source": "Jetcobot-01",
    "message": "순찰 정상 완료"
  },
  {
    "id": 26,
    "ts": "2025-08-15 11:30:56",
    "type": "미등록 인원",
    "severity": "낮음",
    "source": "WarehouseCam",
    "message": "영역 경계 통과 감지"
  },
  {
    "id": 27,
    "ts": "2025-08-15 11:23:56",
    "type": "미등록 인원",
    "severity": "중간",
    "source": "WarehouseCam",
    "message": "자세 이상 탐지"
  },
  {
    "id": 28,
    "ts": "2025-08-15 11:16:56",
    "type": "쓰러짐",
    "severity": "높음",
    "source": "Pinky-01",
    "message": "순찰 정상 완료"
  },
  {
    "id": 29,
    "ts": "2025-08-15 11:09:56",
    "type": "쓰러짐",
    "severity": "높음",
    "source": "GateCam",
    "message": "자세 이상 탐지"
  },
  {
    "id": 30,
    "ts": "2025-08-15 11:02:56",
    "type": "쓰러짐",
    "severity": "낮음",
    "source": "GateCam",
    "message": "움직임 패턴 이상"
  },
  {
    "id": 31,
    "ts": "2025-08-15 10:55:56",
    "type": "미등록 인원",
    "severity": "정보",
    "source": "GateCam",
    "message": "영역 경계 통과 감지"
  },
  {
    "id": 32,
    "ts": "2025-08-15 10:48:56",
    "type": "침입 의심",
    "severity": "높음",
    "source": "WarehouseCam",
    "message": "영역 경계 통과 감지"
  },
  {
    "id": 33,
    "ts": "2025-08-15 10:41:56",
    "type": "미등록 인원",
    "severity": "중간",
    "source": "Jetcobot-01",
    "message": "출입문 장시간 열린 상태"
  },
  {
    "id": 34,
    "ts": "2025-08-15 10:34:56",
    "type": "쓰러짐",
    "severity": "중간",
    "source": "LobbyCam",
    "message": "자세 이상 탐지"
  },
  {
    "id": 35,
    "ts": "2025-08-15 10:27:56",
    "type": "침입 의심",
    "severity": "낮음",
    "source": "WarehouseCam",
    "message": "영역 경계 통과 감지"
  },
  {
    "id": 36,
    "ts": "2025-08-15 10:20:56",
    "type": "미등록 인원",
    "severity": "정보",
    "source": "Jetcobot-01",
    "message": "출입문 장시간 열린 상태"
  },
  {
    "id": 37,
    "ts": "2025-08-15 10:13:56",
    "type": "쓰러짐",
    "severity": "정보",
    "source": "LobbyCam",
    "message": "순찰 정상 완료"
  },
  {
    "id": 38,
    "ts": "2025-08-15 10:06:56",
    "type": "미등록 인원",
    "severity": "중간",
    "source": "Jetcobot-01",
    "message": "움직임 패턴 이상"
  },
  {
    "id": 39,
    "ts": "2025-08-15 09:59:56",
    "type": "쓰러짐",
    "severity": "정보",
    "source": "LobbyCam",
    "message": "순찰 정상 완료"
  },
  {
    "id": 40,
    "ts": "2025-08-15 09:52:56",
    "type": "정상",
    "severity": "중간",
    "source": "Jetcobot-01",
    "message": "영역 경계 통과 감지"
  },
  {
    "id": 41,
    "ts": "2025-08-15 09:45:56",
    "type": "침입 의심",
    "severity": "낮음",
    "source": "GateCam",
    "message": "순찰 정상 완료"
  },
  {
    "id": 42,
    "ts": "2025-08-15 09:38:56",
    "type": "미등록 인원",
    "severity": "낮음",
    "source": "WarehouseCam",
    "message": "자세 이상 탐지"
  },
  {
    "id": 43,
    "ts": "2025-08-15 09:31:56",
    "type": "쓰러짐",
    "severity": "높음",
    "source": "Jetcobot-01",
    "message": "출입문 장시간 열린 상태"
  },
  {
    "id": 44,
    "ts": "2025-08-15 09:24:56",
    "type": "정상",
    "severity": "낮음",
    "source": "Jetcobot-01",
    "message": "출입문 장시간 열린 상태"
  },
  {
    "id": 45,
    "ts": "2025-08-15 09:17:56",
    "type": "미등록 인원",
    "severity": "중간",
    "source": "LobbyCam",
    "message": "영역 경계 통과 감지"
  },
  {
    "id": 46,
    "ts": "2025-08-15 09:10:56",
    "type": "쓰러짐",
    "severity": "정보",
    "source": "Jetcobot-01",
    "message": "출입문 장시간 열린 상태"
  },
  {
    "id": 47,
    "ts": "2025-08-15 09:03:56",
    "type": "침입 의심",
    "severity": "낮음",
    "source": "WarehouseCam",
    "message": "영역 경계 통과 감지"
  },
  {
    "id": 48,
    "ts": "2025-08-15 08:56:56",
    "type": "정상",
    "severity": "높음",
    "source": "GateCam",
    "message": "영역 경계 통과 감지"
  },
  {
    "id": 49,
    "ts": "2025-08-15 08:49:56",
    "type": "미등록 인원",
    "severity": "낮음",
    "source": "WarehouseCam",
    "message": "영역 경계 통과 감지"
  },
  {
    "id": 50,
    "ts": "2025-08-15 08:42:56",
    "type": "미등록 인원",
    "severity": "높음",
    "source": "Pinky-01",
    "message": "자세 이상 탐지"
  },
  {
    "id": 51,
    "ts": "2025-08-15 08:35:56",
    "type": "미등록 인원",
    "severity": "정보",
    "source": "WarehouseCam",
    "message": "움직임 패턴 이상"
  },
  {
    "id": 52,
    "ts": "2025-08-15 08:28:56",
    "type": "쓰러짐",
    "severity": "낮음",
    "source": "GateCam",
    "message": "움직임 패턴 이상"
  },
  {
    "id": 53,
    "ts": "2025-08-15 08:21:56",
    "type": "침입 의심",
    "severity": "중간",
    "source": "Pinky-01",
    "message": "영역 경계 통과 감지"
  },
  {
    "id": 54,
    "ts": "2025-08-15 08:14:56",
    "type": "쓰러짐",
    "severity": "낮음",
    "source": "WarehouseCam",
    "message": "순찰 정상 완료"
  },
  {
    "id": 55,
    "ts": "2025-08-15 08:07:56",
    "type": "정상",
    "severity": "정보",
    "source": "LobbyCam",
    "message": "영역 경계 통과 감지"
  },
  {
    "id": 56,
    "ts": "2025-08-15 08:00:56",
    "type": "정상",
    "severity": "낮음",
    "source": "Pinky-01",
    "message": "영역 경계 통과 감지"
  },
  {
    "id": 57,
    "ts": "2025-08-15 07:53:56",
    "type": "정상",
    "severity": "정보",
    "source": "WarehouseCam",
    "message": "영역 경계 통과 감지"
  },
  {
    "id": 58,
    "ts": "2025-08-15 07:46:56",
    "type": "침입 의심",
    "severity": "높음",
    "source": "LobbyCam",
    "message": "움직임 패턴 이상"
  },
  {
    "id": 59,
    "ts": "2025-08-15 07:39:56",
    "type": "미등록 인원",
    "severity": "중간",
    "source": "Pinky-01",
    "message": "자세 이상 탐지"
  },
  {
    "id": 60,
    "ts": "2025-08-15 07:32:56",
    "type": "침입 의심",
    "severity": "높음",
    "source": "GateCam",
    "message": "영역 경계 통과 감지"
  },
  {
    "id": 61,
    "ts": "2025-08-15 07:25:56",
    "type": "침입 의심",
    "severity": "높음",
    "source": "GateCam",
    "message": "영역 경계 통과 감지"
  },
  {
    "id": 62,
    "ts": "2025-08-15 07:18:56",
    "type": "쓰러짐",
    "severity": "높음",
    "source": "LobbyCam",
    "message": "자세 이상 탐지"
  },
  {
    "id": 63,
    "ts": "2025-08-15 07:11:56",
    "type": "미등록 인원",
    "severity": "높음",
    "source": "WarehouseCam",
    "message": "움직임 패턴 이상"
  },
  {
    "id": 64,
    "ts": "2025-08-15 07:04:56",
    "type": "침입 의심",
    "severity": "정보",
    "source": "WarehouseCam",
    "message": "움직임 패턴 이상"
  },
  {
    "id": 65,
    "ts": "2025-08-15 06:57:56",
    "type": "미등록 인원",
    "severity": "높음",
    "source": "GateCam",
    "message": "움직임 패턴 이상"
  },
  {
    "id": 66,
    "ts": "2025-08-15 06:50:56",
    "type": "쓰러짐",
    "severity": "높음",
    "source": "GateCam",
    "message": "출입문 장시간 열린 상태"
  },
  {
    "id": 67,
    "ts": "2025-08-15 06:43:56",
    "type": "미등록 인원",
    "severity": "정보",
    "source": "LobbyCam",
    "message": "영역 경계 통과 감지"
  },
  {
    "id": 68,
    "ts": "2025-08-15 06:36:56",
    "type": "정상",
    "severity": "정보",
    "source": "WarehouseCam",
    "message": "영역 경계 통과 감지"
  },
  {
    "id": 69,
    "ts": "2025-08-15 06:29:56",
    "type": "미등록 인원",
    "severity": "높음",
    "source": "WarehouseCam",
    "message": "순찰 정상 완료"
  },
  {
    "id": 70,
    "ts": "2025-08-15 06:22:56",
    "type": "정상",
    "severity": "중간",
    "source": "GateCam",
    "message": "순찰 정상 완료"
  },
  {
    "id": 71,
    "ts": "2025-08-15 06:15:56",
    "type": "침입 의심",
    "severity": "낮음",
    "source": "GateCam",
    "message": "자세 이상 탐지"
  },
  {
    "id": 72,
    "ts": "2025-08-15 06:08:56",
    "type": "침입 의심",
    "severity": "낮음",
    "source": "LobbyCam",
    "message": "영역 경계 통과 감지"
  },
  {
    "id": 73,
    "ts": "2025-08-15 06:01:56",
    "type": "정상",
    "severity": "중간",
    "source": "LobbyCam",
    "message": "영역 경계 통과 감지"
  },
  {
    "id": 74,
    "ts": "2025-08-15 05:54:56",
    "type": "침입 의심",
    "severity": "중간",
    "source": "Jetcobot-01",
    "message": "순찰 정상 완료"
  },
  {
    "id": 75,
    "ts": "2025-08-15 05:47:56",
    "type": "정상",
    "severity": "중간",
    "source": "Jetcobot-01",
    "message": "출입문 장시간 열린 상태"
  },
  {
    "id": 76,
    "ts": "2025-08-15 05:40:56",
    "type": "미등록 인원",
    "severity": "낮음",
    "source": "Pinky-01",
    "message": "움직임 패턴 이상"
  },
  {
    "id": 77,
    "ts": "2025-08-15 05:33:56",
    "type": "미등록 인원",
    "severity": "정보",
    "source": "LobbyCam",
    "message": "자세 이상 탐지"
  },
  {
    "id": 78,
    "ts": "2025-08-15 05:26:56",
    "type": "정상",
    "severity": "중간",
    "source": "WarehouseCam",
    "message": "자세 이상 탐지"
  },
  {
    "id": 79,
    "ts": "2025-08-15 05:19:56",
    "type": "쓰러짐",
    "severity": "높음",
    "source": "Pinky-01",
    "message": "움직임 패턴 이상"
  },
  {
    "id": 80,
    "ts": "2025-08-15 05:12:56",
    "type": "정상",
    "severity": "낮음",
    "source": "Jetcobot-01",
    "message": "자세 이상 탐지"
  },
  {
    "id": 81,
    "ts": "2025-08-15 05:05:56",
    "type": "침입 의심",
    "severity": "낮음",
    "source": "Jetcobot-01",
    "message": "출입문 장시간 열린 상태"
  },
  {
    "id": 82,
    "ts": "2025-08-15 04:58:56",
    "type": "침입 의심",
    "severity": "낮음",
    "source": "GateCam",
    "message": "출입문 장시간 열린 상태"
  },
  {
    "id": 83,
    "ts": "2025-08-15 04:51:56",
    "type": "미등록 인원",
    "severity": "낮음",
    "source": "GateCam",
    "message": "순찰 정상 완료"
  },
  {
    "id": 84,
    "ts": "2025-08-15 04:44:56",
    "type": "미등록 인원",
    "severity": "낮음",
    "source": "WarehouseCam",
    "message": "순찰 정상 완료"
  },
  {
    "id": 85,
    "ts": "2025-08-15 04:37:56",
    "type": "미등록 인원",
    "severity": "낮음",
    "source": "WarehouseCam",
    "message": "영역 경계 통과 감지"
  },
  {
    "id": 86,
    "ts": "2025-08-15 04:30:56",
    "type": "쓰러짐",
    "severity": "낮음",
    "source": "LobbyCam",
    "message": "영역 경계 통과 감지"
  },
  {
    "id": 87,
    "ts": "2025-08-15 04:23:56",
    "type": "침입 의심",
    "severity": "낮음",
    "source": "LobbyCam",
    "message": "영역 경계 통과 감지"
  },
  {
    "id": 88,
    "ts": "2025-08-15 04:16:56",
    "type": "정상",
    "severity": "낮음",
    "source": "WarehouseCam",
    "message": "움직임 패턴 이상"
  },
  {
    "id": 89,
    "ts": "2025-08-15 04:09:56",
    "type": "정상",
    "severity": "중간",
    "source": "Pinky-01",
    "message": "순찰 정상 완료"
  },
  {
    "id": 90,
    "ts": "2025-08-15 04:02:56",
    "type": "침입 의심",
    "severity": "정보",
    "source": "Jetcobot-01",
    "message": "움직임 패턴 이상"
  },
  {
    "id": 91,
    "ts": "2025-08-15 03:55:56",
    "type": "미등록 인원",
    "severity": "중간",
    "source": "LobbyCam",
    "message": "영역 경계 통과 감지"
  },
  {
    "id": 92,
    "ts": "2025-08-15 03:48:56",
    "type": "미등록 인원",
    "severity": "정보",
    "source": "WarehouseCam",
    "message": "출입문 장시간 열린 상태"
  },
  {
    "id": 93,
    "ts": "2025-08-15 03:41:56",
    "type": "침입 의심",
    "severity": "중간",
    "source": "LobbyCam",
    "message": "출입문 장시간 열린 상태"
  },
  {
    "id": 94,
    "ts": "2025-08-15 03:34:56",
    "type": "미등록 인원",
    "severity": "높음",
    "source": "LobbyCam",
    "message": "출입문 장시간 열린 상태"
  },
  {
    "id": 95,
    "ts": "2025-08-15 03:27:56",
    "type": "침입 의심",
    "severity": "높음",
    "source": "LobbyCam",
    "message": "영역 경계 통과 감지"
  },
  {
    "id": 96,
    "ts": "2025-08-15 03:20:56",
    "type": "쓰러짐",
    "severity": "높음",
    "source": "Pinky-01",
    "message": "출입문 장시간 열린 상태"
  },
  {
    "id": 97,
    "ts": "2025-08-15 03:13:56",
    "type": "미등록 인원",
    "severity": "높음",
    "source": "LobbyCam",
    "message": "순찰 정상 완료"
  },
  {
    "id": 98,
    "ts": "2025-08-15 03:06:56",
    "type": "쓰러짐",
    "severity": "중간",
    "source": "GateCam",
    "message": "출입문 장시간 열린 상태"
  },
  {
    "id": 99,
    "ts": "2025-08-15 02:59:56",
    "type": "쓰러짐",
    "severity": "중간",
    "source": "WarehouseCam",
    "message": "영역 경계 통과 감지"
  },
  {
    "id": 100,
    "ts": "2025-08-15 02:52:56",
    "type": "쓰러짐",
    "severity": "낮음",
    "source": "GateCam",
    "message": "움직임 패턴 이상"
  },
  {
    "id": 101,
    "ts": "2025-08-15 02:45:56",
    "type": "정상",
    "severity": "중간",
    "source": "Jetcobot-01",
    "message": "순찰 정상 완료"
  },
  {
    "id": 102,
    "ts": "2025-08-15 02:38:56",
    "type": "침입 의심",
    "severity": "정보",
    "source": "GateCam",
    "message": "순찰 정상 완료"
  },
  {
    "id": 103,
    "ts": "2025-08-15 02:31:56",
    "type": "침입 의심",
    "severity": "정보",
    "source": "Pinky-01",
    "message": "출입문 장시간 열린 상태"
  },
  {
    "id": 104,
    "ts": "2025-08-15 02:24:56",
    "type": "미등록 인원",
    "severity": "중간",
    "source": "Jetcobot-01",
    "message": "자세 이상 탐지"
  },
  {
    "id": 105,
    "ts": "2025-08-15 02:17:56",
    "type": "미등록 인원",
    "severity": "낮음",
    "source": "Jetcobot-01",
    "message": "움직임 패턴 이상"
  },
  {
    "id": 106,
    "ts": "2025-08-15 02:10:56",
    "type": "침입 의심",
    "severity": "낮음",
    "source": "GateCam",
    "message": "영역 경계 통과 감지"
  },
  {
    "id": 107,
    "ts": "2025-08-15 02:03:56",
    "type": "침입 의심",
    "severity": "높음",
    "source": "LobbyCam",
    "message": "영역 경계 통과 감지"
  },
  {
    "id": 108,
    "ts": "2025-08-15 01:56:56",
    "type": "쓰러짐",
    "severity": "높음",
    "source": "Jetcobot-01",
    "message": "영역 경계 통과 감지"
  },
  {
    "id": 109,
    "ts": "2025-08-15 01:49:56",
    "type": "침입 의심",
    "severity": "정보",
    "source": "GateCam",
    "message": "출입문 장시간 열린 상태"
  },
  {
    "id": 110,
    "ts": "2025-08-15 01:42:56",
    "type": "침입 의심",
    "severity": "중간",
    "source": "LobbyCam",
    "message": "영역 경계 통과 감지"
  },
  {
    "id": 111,
    "ts": "2025-08-15 01:35:56",
    "type": "미등록 인원",
    "severity": "높음",
    "source": "GateCam",
    "message": "영역 경계 통과 감지"
  },
  {
    "id": 112,
    "ts": "2025-08-15 01:28:56",
    "type": "쓰러짐",
    "severity": "중간",
    "source": "LobbyCam",
    "message": "순찰 정상 완료"
  },
  {
    "id": 113,
    "ts": "2025-08-15 01:21:56",
    "type": "미등록 인원",
    "severity": "높음",
    "source": "Jetcobot-01",
    "message": "영역 경계 통과 감지"
  },
  {
    "id": 114,
    "ts": "2025-08-15 01:14:56",
    "type": "미등록 인원",
    "severity": "낮음",
    "source": "Jetcobot-01",
    "message": "영역 경계 통과 감지"
  },
  {
    "id": 115,
    "ts": "2025-08-15 01:07:56",
    "type": "미등록 인원",
    "severity": "높음",
    "source": "GateCam",
    "message": "출입문 장시간 열린 상태"
  },
  {
    "id": 116,
    "ts": "2025-08-15 01:00:56",
    "type": "쓰러짐",
    "severity": "정보",
    "source": "WarehouseCam",
    "message": "영역 경계 통과 감지"
  },
  {
    "id": 117,
    "ts": "2025-08-15 00:53:56",
    "type": "미등록 인원",
    "severity": "높음",
    "source": "LobbyCam",
    "message": "움직임 패턴 이상"
  },
  {
    "id": 118,
    "ts": "2025-08-15 00:46:56",
    "type": "쓰러짐",
    "severity": "높음",
    "source": "Pinky-01",
    "message": "움직임 패턴 이상"
  },
  {
    "id": 119,
    "ts": "2025-08-15 00:39:56",
    "type": "정상",
    "severity": "중간",
    "source": "Jetcobot-01",
    "message": "움직임 패턴 이상"
  },
  {
    "id": 120,
    "ts": "2025-08-15 00:32:56",
    "type": "쓰러짐",
    "severity": "낮음",
    "source": "Jetcobot-01",
    "message": "움직임 패턴 이상"
  }
]
FACES = [
    {"id": i, "name": f"직원{i:03d}",
     "created_at": (datetime.now()-timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S")}
    for i in range(1, 153)
]

SECURITY = {
    "registered_faces": len(FACES),
    "unknown_today": 5,
    "falls_today": 1,
    }
RECENT_SECURITY_EVENTS = [
    {"ts": (datetime.now()-timedelta(minutes=3)).strftime("%Y-%m-%d %H:%M"),
     "type": "미등록 인원", "severity": "중간", "message": "로비 카메라에서 미등록 인원 감지"},
    {"ts": (datetime.now()-timedelta(minutes=15)).strftime("%Y-%m-%d %H:%M"),
     "type": "쓰러짐", "severity": "높음", "message": "창고 구역에서 쓰러짐 의심 자세"},
    {"ts": (datetime.now()-timedelta(hours=1)).strftime("%Y-%m-%d %H:%M"),
     "type": "미등록 인원", "severity": "높음", "message": "출입문 근처에서 날카로운 물체 감지"},
    {"ts": (datetime.now()-timedelta(hours=2)).strftime("%Y-%m-%d %H:%M"),
     "type": "미등록 인원", "severity": "낮음", "message": "주차장 구역 경계 통과"},
    {"ts": (datetime.now()-timedelta(hours=3)).strftime("%Y-%m-%d %H:%M"),
     "type": "정상", "severity": "정보", "message": "정기 순찰 완료"},
]

DEVICES = [
    {"name": "Pinky-01", "battery": 84, "status": "online", "location": "Lobby"},
    {"name": "Jetcobot-01", "battery": 67, "status": "online", "location": "Warehouse"},
    {"name": "GlobalCam-01", "battery": None, "status": "online", "location": "Gate"},
    {"name": "Pinky-02", "battery": 25, "status": "charging", "location": "Dock"},
    {"name": "AI-Decision-Server", "battery": None, "status": "online", "location": "ServerRoom"},
]

HEALTH = {"uptime": "99.95%", "cpu": "23%", "ram": "58%", "disk": "71%", "network": "정상"}

SCHEDULES = [
    {"time": (datetime.now()+timedelta(minutes=30)).strftime("%H:%M"),
     "title": "로비 순찰", "route": "Route-A", "owner": "관리자A"},
    {"time": (datetime.now()+timedelta(hours=1, minutes=10)).strftime("%H:%M"),
     "title": "창고 안전점검", "route": "Route-B", "owner": "관리자B"},
    {"time": (datetime.now()+timedelta(hours=2)).strftime("%H:%M"),
     "title": "야간 출입기록 점검", "route": "Route-C", "owner": "관리자C"},
]

def paginate(items, page, page_size):
    total = len(items)
    total_pages = max(1, math.ceil(total / page_size))
    start = (page - 1) * page_size
    end = start + page_size
    return items[start:end], total, total_pages

@app.get("/")
def dashboard():
    summary_cards = [
        {"label": "등록된 얼굴", "value": SECURITY["registered_faces"], "icon": "people"},
        {"label": "미등록 인원(오늘)", "value": SECURITY["unknown_today"], "icon": "person-exclamation"},
        {"label": "쓰러짐(오늘)", "value": SECURITY["falls_today"], "icon": "activity"},
                {"label": "Online 장비", "value": sum(1 for d in DEVICES if d["status"]=="online"), "icon": "router"},
        {"label": "Uptime", "value": HEALTH["uptime"], "icon": "clock-history"},
    ]
    return render_template(
        "dashboard.html",
        summary_cards=summary_cards,
        recent_events=RECENT_SECURITY_EVENTS,
        devices=DEVICES,
        health=HEALTH,
        schedules=SCHEDULES
    )

@app.get("/faces")
def faces_list():
    from flask import request
    q = request.args.get("q", "").strip()
    sort = request.args.get("sort", "created_desc")
    page = max(int(request.args.get("page", 1)), 1)
    page_size = 20

    data = FACES[:]
    if q:
        data = [x for x in data if q.lower() in x["name"].lower()]

    if sort == "name_asc":
        data.sort(key=lambda x: x["name"])
    elif sort == "name_desc":
        data.sort(key=lambda x: x["name"], reverse=True)
    elif sort == "created_asc":
        data.sort(key=lambda x: x["created_at"])
    else:
        data.sort(key=lambda x: x["created_at"], reverse=True)

    page_items, total_count, total_pages = paginate(data, page, page_size)
    return render_template("faces.html",
        faces=page_items, total_count=total_count,
        total_pages=total_pages, page=page, page_size=page_size)

@app.post("/faces/<int:face_id>/delete")
def faces_delete(face_id):
    global FACES
    FACES = [x for x in FACES if x["id"] != face_id]
    return redirect(url_for("faces_list"))

@app.get("/incidents")
def incidents(): return render_template("blank.html", title="Incidents")
@app.get("/system")
def system():    return render_template("blank.html", title="System")
@app.get("/stats")
def stats():     return render_template("blank.html", title="Stats")



@app.get("/devices")
def devices():  return render_template("blank.html", title="Devices")

@app.get("/cameras")
def cameras():  return render_template("blank.html", title="Cameras")

@app.get("/patrols")
def patrols():  return render_template("blank.html", title="Patrols")

@app.get("/maps")
def maps():     return render_template("blank.html", title="Maps")

@app.get("/users")
def users():    return render_template("blank.html", title="Users & Roles")

@app.get("/tasks")
def tasks():    return render_template("blank.html", title="Tasks")

@app.get("/settings")
def settings(): return render_template("blank.html", title="Settings")

@app.get("/profile")
def profile():  return render_template("blank.html", title="Profile")





import ext_routes

import ext_routes
ext_routes.register(app)

if __name__ == "__main__":
    import os
    app.run(host="0.0.0.0", port=int(os.getenv("APP_PORT","8000")), debug=False, use_reloader=False)
