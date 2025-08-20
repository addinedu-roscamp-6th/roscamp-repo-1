from datetime import datetime, timedelta
import random, csv, io
from flask import render_template, request, Response

# ---------------- 공용 더미 데이터 ----------------
TEAM_PATTERNS = {
    "이동연": {"role":"직원","status":"출입 승인","method":"얼굴 인식"},
    "백기광": {"role":"직원","status":"출입 승인","method":"얼굴 인식"},
    "박지안": {"role":"방문객","status":"출입 거부","method":"미등록 인원"},
    "위다인": {"role":"직원","status":"출입 승인","method":"얼굴 인식"},
    "김성민": {"role":"방문객","status":"출입 거부","method":"미등록 인원"},
    "박진우": {"role":"직원","status":"출입 승인","method":"얼굴 인식"},
}
NAMES = list(TEAM_PATTERNS.keys())

def _parse_date(s):
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None

def _paginate(items, page, per_page):
    total = len(items)
    start = (page-1) * per_page
    end = start + per_page
    return items[start:end], total

def _querystring(params, **overrides):
    q = dict(params)
    q.update(overrides)
    return "&".join(f"{k}={v}" for k, v in q.items() if v not in (None, ""))

# ---------------- 출입 기록 ----------------
def _gen_access(n=120):
    now = datetime.now()
    rows = []
    for i in range(n):
        name = random.choice(NAMES)
        pat = TEAM_PATTERNS[name]
        rows.append({
            "ts": (now - timedelta(minutes=i*3)).strftime("%Y-%m-%d %H:%M:%S"),
            "date": (now - timedelta(minutes=i*3)).strftime("%Y-%m-%d"),
            "name": name,
            "role": pat["role"],
            "result": pat["status"],
            "method": pat["method"],
            "door": random.choice(["LobbyGate","OfficeDoor","WarehouseDoor","GateCam"]),
        })
    return rows

# ---------------- 시스템 로그 ----------------
def _gen_logs(n=200):
    now = datetime.now()
    levels = ["INFO","WARN","ERROR"]
    msgs = {
        "이동연": ["얼굴 인식 성공", "정상 출입 처리됨"],
        "백기광": ["얼굴 인식 성공", "정상 출입 처리됨"],
        "박지안": ["미등록 인원 감지", "출입 거부됨"],
        "위다인": ["얼굴 인식 성공", "정상 출입 처리됨"],
        "김성민": ["미등록 인원 감지", "출입 거부됨"],
        "박진우": ["얼굴 인식 성공", "정상 출입 처리됨"],
    }
    rows = []
    for i in range(n):
        who = random.choice(NAMES)
        rows.append({
            "ts": (now - timedelta(minutes=i*2)).strftime("%Y-%m-%d %H:%M:%S"),
            "date": (now - timedelta(minutes=i*2)).strftime("%Y-%m-%d"),
            "level": random.choice(levels),
            "source": who,
            "message": f"{who}: {random.choice(msgs[who])}",
        })
    return rows

# ---------------- 알림/경보 (우리 기능만: 미등록 인원, 쓰러짐) ----------------
ALERT_TYPES = ["미등록 인원", "쓰러짐"]
SEVERITIES = ["낮음", "중간", "높음"]

def _gen_alerts(n=150):
    now = datetime.now()
    msgs = {
        "미등록 인원": [
            "미등록 인원 감지", "출입 장치 앞 미등록 인원 탐지", "등록되지 않은 인원 확인",
        ],
        "쓰러짐": [
            "바닥에 쓰러짐 패턴 감지", "인체 이상 자세 탐지", "움직임 정지 및 쓰러짐 의심",
        ],
    }
    rows = []
    for i in range(n):
        k = random.choice(ALERT_TYPES)
        rows.append({
            "ts": (now - timedelta(minutes=i*4)).strftime("%Y-%m-%d %H:%M:%S"),
            "date": (now - timedelta(minutes=i*4)).strftime("%Y-%m-%d"),
            "kind": k,
            "severity": random.choice(SEVERITIES),
            "message": random.choice(msgs[k]),
        })
    return rows

# 전역 더미
ACCESS_D = _gen_access()
SYSLOG_D = _gen_logs()
ALERTS_D = _gen_alerts()

# ---------------- 등록 진입점 ----------------
def register(app):
    # --- Devices routes ---
    register_devices_routes(app)

    # --- 보안 이벤트 라우트 등록 ---
    register_events_routes(app)

    # ====== ACCESS ======
    @app.get("/access")
    def access_view():
        name = request.args.get("name","").strip()
        result = request.args.get("result","").strip()
        dfrom = _parse_date(request.args.get("from",""))
        dto   = _parse_date(request.args.get("to",""))
        page = int(request.args.get("page", 1) or 1)
        per_page = int(request.args.get("per_page", 20) or 20)

        data = sorted(ACCESS_D, key=lambda x: x["ts"], reverse=True)
        if name:   data = [r for r in data if name in r["name"]]
        if result: data = [r for r in data if r["result"] == result]
        if dfrom:  data = [r for r in data if r["date"] >= dfrom.strftime("%Y-%m-%d")]
        if dto:    data = [r for r in data if r["date"] <= dto.strftime("%Y-%m-%d")]

        page_items, total = _paginate(data, page, per_page)
        pages = (total + per_page - 1)//per_page
        return render_template("access.html",
                               logs=page_items,
                               name=name, result=result,
                               dfrom=request.args.get("from",""),
                               dto=request.args.get("to",""),
                               page=page, per_page=per_page,
                               total=total, pages=pages,
                               qs=lambda **kw: _querystring(request.args.to_dict(flat=True), **kw))

    @app.get("/access.csv")
    def access_csv():
        name = request.args.get("name","").strip()
        result = request.args.get("result","").strip()
        dfrom = _parse_date(request.args.get("from",""))
        dto   = _parse_date(request.args.get("to",""))
        data = sorted(ACCESS_D, key=lambda x: x["ts"], reverse=True)
        if name:   data = [r for r in data if name in r["name"]]
        if result: data = [r for r in data if r["result"] == result]
        if dfrom:  data = [r for r in data if r["date"] >= dfrom.strftime("%Y-%m-%d")]
        if dto:    data = [r for r in data if r["date"] <= dto.strftime("%Y-%m-%d")]

        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["시간","이름","구분","결과","인증 방식","문/게이트"])
        for r in data:
            w.writerow([r["ts"], r["name"], r["role"], r["result"], r["method"], r["door"]])
        return Response(buf.getvalue().encode("utf-8-sig"),
                        mimetype="text/csv",
                        headers={"Content-Disposition":"attachment; filename=access.csv"})

    # ====== LOGS ======
    @app.get("/logs")
    def logs_view():
        name = request.args.get("name","").strip()
        level = request.args.get("level","").strip()
        dfrom = _parse_date(request.args.get("from",""))
        dto   = _parse_date(request.args.get("to",""))
        page = int(request.args.get("page", 1) or 1)
        per_page = int(request.args.get("per_page", 20) or 20)

        data = sorted(SYSLOG_D, key=lambda x: x["ts"], reverse=True)
        if name:  data = [r for r in data if name in r["source"]]
        if level: data = [r for r in data if r["level"] == level]
        if dfrom: data = [r for r in data if r["date"] >= dfrom.strftime("%Y-%m-%d")]
        if dto:   data = [r for r in data if r["date"] <= dto.strftime("%Y-%m-%d")]

        page_items, total = _paginate(data, page, per_page)
        pages = (total + per_page - 1)//per_page
        return render_template("logs.html",
                               logs=page_items,
                               name=name, level=level,
                               dfrom=request.args.get("from",""),
                               dto=request.args.get("to",""),
                               page=page, per_page=per_page,
                               total=total, pages=pages,
                               qs=lambda **kw: _querystring(request.args.to_dict(flat=True), **kw))

    @app.get("/logs.csv")
    def logs_csv():
        name = request.args.get("name","").strip()
        level = request.args.get("level","").strip()
        dfrom = _parse_date(request.args.get("from",""))
        dto   = _parse_date(request.args.get("to",""))
        data = sorted(SYSLOG_D, key=lambda x: x["ts"], reverse=True)
        if name:  data = [r for r in data if name in r["source"]]
        if level: data = [r for r in data if r["level"] == level]
        if dfrom: data = [r for r in data if r["date"] >= dfrom.strftime("%Y-%m-%d")]
        if dto:   data = [r for r in data if r["date"] <= dto.strftime("%Y-%m-%d")]

        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["시간","레벨","소스","메시지"])
        for r in data:
            w.writerow([r["ts"], r["level"], r["source"], r["message"]])
        return Response(buf.getvalue().encode("utf-8-sig"),
                        mimetype="text/csv",
                        headers={"Content-Disposition":"attachment; filename=logs.csv"})

    # ====== ALERTS (알림/경보) ======
    def _alerts_data_filtered():
        kind   = request.args.get("kind","").strip()       # 미등록 인원, 쓰러짐
        sev    = request.args.get("sev","").strip()        # 낮음/중간/높음
        q      = request.args.get("q","").strip()          # 메시지 키워드
        dfrom  = _parse_date(request.args.get("from",""))
        dto    = _parse_date(request.args.get("to",""))

        data = sorted(ALERTS_D, key=lambda x: x["ts"], reverse=True)
        if kind:  data = [r for r in data if r["kind"] == kind]
        if sev:   data = [r for r in data if r["severity"] == sev]
        if q:     data = [r for r in data if q in r["message"]]
        if dfrom: data = [r for r in data if r["date"] >= dfrom.strftime("%Y-%m-%d")]
        if dto:   data = [r for r in data if r["date"] <= dto.strftime("%Y-%m-%d")]
        return data

    def alerts_view():
        page = int(request.args.get("page", 1) or 1)
        per_page = int(request.args.get("per_page", 20) or 20)

        data = _alerts_data_filtered()
        page_items, total = _paginate(data, page, per_page)
        pages = (total + per_page - 1)//per_page

        return render_template("alerts.html",
                               rows=page_items,
                               kind=request.args.get("kind",""),
                               sev=request.args.get("sev",""),
                               q=request.args.get("q",""),
                               dfrom=request.args.get("from",""),
                               dto=request.args.get("to",""),
                               page=page, per_page=per_page,
                               total=total, pages=pages,
                               qs=lambda **kw: _querystring(request.args.to_dict(flat=True), **kw))

    def alerts_csv():
        data = _alerts_data_filtered()
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["시간","유형","심각도","메시지"])
        for r in data:
            w.writerow([r["ts"], r["kind"], r["severity"], r["message"]])
        return Response(buf.getvalue().encode("utf-8-sig"),
                        mimetype="text/csv",
                        headers={"Content-Disposition":"attachment; filename=alerts.csv"})

    # 기존 /alerts 뷰가 있어도 덮어쓰기
    endp = None
    for rule in app.url_map.iter_rules():
        if rule.rule == "/alerts":
            endp = rule.endpoint
            break
    if endp:
        app.view_functions[endp] = alerts_view
    else:
        app.add_url_rule("/alerts", "alerts", alerts_view, methods=["GET"])

    app.add_url_rule("/alerts.csv", "alerts_csv", alerts_csv, methods=["GET"])

# ---------------- 보안 이벤트 (alerts에서 파생: 미등록 인원/쓰러짐만) ----------------
EV_STATUSES = ["열림","진행중","해결"]
EV_PRIOS    = ["낮음","중간","높음"]
EV_LOCS     = ["Pinky-01","Lobby","Warehouse","Gate"]

def _gen_events(n=100):
    # ALERTS_D가 존재한다고 가정 (위에서 생성)
    # 알림 일부를 사건(이벤트)로 승격 + 메타(상태/우선순위/담당/위치) 부여
    import random
    from datetime import datetime
    rows = []
    base = ALERTS_D[:]
    base.sort(key=lambda x: x["ts"], reverse=True)
    for i, a in enumerate(base[:n]):
        rows.append({
            "id": i+1,
            "ts": a["ts"],
            "date": a["date"],
            "kind": a["kind"],                      # 미등록 인원 / 쓰러짐
            "prio": random.choice(EV_PRIOS),        # 낮/중/높
            "status": random.choice(EV_STATUSES),   # 열림/진행중/해결
            "loc": random.choice(EV_LOCS),
            "assignee": random.choice(NAMES),       # 담당자
            "message": a["message"],
        })
    return rows

EVENTS_D = _gen_events()

def register_events_routes(app):
    from flask import render_template, request, Response
    import csv, io

    def _parse_date(s):
        try:
            from datetime import datetime
            return datetime.strptime(s, "%Y-%m-%d").date()
        except Exception:
            return None

    def _paginate(items, page, per_page):
        total = len(items)
        start = (page-1)*per_page
        end   = start+per_page
        return items[start:end], total

    def _querystring(params, **overrides):
        q = dict(params); q.update(overrides)
        return "&".join(f"{k}={v}" for k, v in q.items() if v not in (None, ""))

    def _events_filtered():
        kind  = request.args.get("kind","").strip()
        prio  = request.args.get("prio","").strip()
        stat  = request.args.get("status","").strip()
        asg   = request.args.get("assignee","").strip()
        loc   = request.args.get("loc","").strip()
        q     = request.args.get("q","").strip()
        dfrom = _parse_date(request.args.get("from",""))
        dto   = _parse_date(request.args.get("to",""))

        data = sorted(EVENTS_D, key=lambda x: x["ts"], reverse=True)
        if kind:  data = [r for r in data if r["kind"] == kind]
        if prio:  data = [r for r in data if r["prio"] == prio]
        if stat:  data = [r for r in data if r["status"] == stat]
        if asg:   data = [r for r in data if asg in r["assignee"]]
        if loc:   data = [r for r in data if r["loc"] == loc]
        if q:     data = [r for r in data if q in r["message"]]
        if dfrom: data = [r for r in data if r["date"] >= dfrom.strftime("%Y-%m-%d")]
        if dto:   data = [r for r in data if r["date"] <= dto.strftime("%Y-%m-%d")]
        return data

    def events_view():
        page     = int(request.args.get("page", 1) or 1)
        per_page = int(request.args.get("per_page", 20) or 20)
        data = _events_filtered()
        page_items, total = _paginate(data, page, per_page)
        pages = (total + per_page - 1)//per_page
        return render_template("events.html",
            rows=page_items, total=total, page=page, pages=pages, per_page=per_page,
            kind=request.args.get("kind",""), prio=request.args.get("prio",""),
            status=request.args.get("status",""), assignee=request.args.get("assignee",""),
            loc=request.args.get("loc",""), q=request.args.get("q",""),
            dfrom=request.args.get("from",""), dto=request.args.get("to",""),
            kinds=ALERT_TYPES, prios=EV_PRIOS, stats=EV_STATUSES, locs=EV_LOCS, assignees=NAMES,
            qs=lambda **kw: _querystring(request.args.to_dict(flat=True), **kw)
        )

    def events_csv():
        data = _events_filtered()
        buf = io.StringIO(); w = csv.writer(buf)
        w.writerow(["ID","시간","유형","우선순위","상태","위치","담당","메시지"])
        for r in data:
            w.writerow([r["id"], r["ts"], r["kind"], r["prio"], r["status"], r["loc"], r["assignee"], r["message"]])
        return Response(buf.getvalue().encode("utf-8-sig"),
                        mimetype="text/csv",
                        headers={"Content-Disposition":"attachment; filename=events.csv"})

    # /events 등록
    app.add_url_rule("/events", "events", events_view, methods=["GET"])
    app.add_url_rule("/events.csv", "events_csv", events_csv, methods=["GET"])

    # 기존 /incidents가 있으면 같은 뷰로 연결(덮어쓰기)
    ep = None
    for rule in app.url_map.iter_rules():
        if rule.rule == "/incidents":
            ep = rule.endpoint
            break
    if ep:
        app.view_functions[ep] = events_view

# ---------------- Devices (장비) ----------------
DEV_TYPES = ["camera","robot","sensor","gateway"]
DEV_STAT  = ["online","offline","degraded"]
DEV_LOCS  = ["Pinky-01","Lobby","Warehouse","Gate","Office","Parking"]
DEV_FW    = ["1.0.3","1.1.0","1.1.2","2.0.0-rc1"]

def _gen_devices(n=40):
    import random, hashlib, time
    from datetime import datetime, timedelta
    rows=[]
    now = datetime.now()
    for i in range(n):
        dtype = random.choice(DEV_TYPES)
        name = f"{dtype[:1].upper()}{dtype[1:]}-{100+i}"
        rows.append({
            "id": 100+i,
            "name": name,
            "type": dtype,
            "status": random.choice(DEV_STAT),
            "loc": random.choice(DEV_LOCS),
            "fw": random.choice(DEV_FW),
            "last_hb": (now - timedelta(minutes=random.randint(0, 300))).strftime("%Y-%m-%d %H:%M:%S"),
            "ip": f"192.168.{random.randint(0,9)}.{random.randint(10,250)}",
            "serial": f"SN-{100+i:06d}",
        })
    return rows

DEVICES_D = _gen_devices()

def register_devices_routes(app):
    # 중복 등록 방지
    existing_endpoints = {r.endpoint for r in app.url_map.iter_rules()}
    if {'devices_list','devices_csv','devices'} & existing_endpoints:
        return
    from flask import render_template, request, Response
    import io, csv
    from datetime import datetime

    def _parse_date(s):
        try:
            return datetime.strptime(s, "%Y-%m-%d").date()
        except Exception:
            return None

    def _paginate(items, page, per_page):
        total=len(items); start=(page-1)*per_page; end=start+per_page
        return items[start:end], total

    def _qs(params, **overrides):
        q=dict(params); q.update(overrides)
        return "&".join(f"{k}={v}" for k,v in q.items() if v not in (None,""))

    def _filtered():
        t  = request.args.get("type","").strip()      # camera/robot/sensor/gateway
        st = request.args.get("status","").strip()    # online/offline/degraded
        loc= request.args.get("loc","").strip()
        q  = request.args.get("q","").strip()         # 이름/IP/시리얼 키워드
        data = sorted(DEVICES_D, key=lambda x: x["name"])
        if t:   data = [r for r in data if r["type"]==t]
        if st:  data = [r for r in data if r["status"]==st]
        if loc: data = [r for r in data if r["loc"]==loc]
        if q:
            ql=q.lower()
            data=[r for r in data if ql in r["name"].lower() or ql in r["ip"].lower() or ql in r["serial"].lower()]
        return data

    def devices_view():
        page = int(request.args.get("page",1) or 1)
        per_page = int(request.args.get("per_page",20) or 20)
        data = _filtered()
        page_items, total = _paginate(data, page, per_page)
        pages = (total + per_page - 1)//per_page
        return render_template("devices.html",
            rows=page_items, total=total, page=page, pages=pages, per_page=per_page,
            type=request.args.get("type",""), status=request.args.get("status",""),
            loc=request.args.get("loc",""), q=request.args.get("q",""),
            types=DEV_TYPES, statuses=DEV_STAT, locs=DEV_LOCS,
            qs=lambda **kw: _qs(request.args.to_dict(flat=True), **kw)
        )

    def devices_csv():
        data = _filtered()
        buf = io.StringIO(); w=csv.writer(buf)
        w.writerow(["ID","이름","종류","상태","위치","FW","마지막 하트비트","IP","Serial"])
        for r in data:
            w.writerow([r["id"], r["name"], r["type"], r["status"], r["loc"], r["fw"], r["last_hb"], r["ip"], r["serial"]])
        return Response(buf.getvalue().encode("utf-8-sig"),
                        mimetype="text/csv",
                        headers={"Content-Disposition":"attachment; filename=devices.csv"})

    app.add_url_rule("/devices", endpoint="devices_list", view_func=devices_view, methods=["GET"])
    app.add_url_rule("/devices.csv", "devices_csv", devices_csv, methods=["GET"])
