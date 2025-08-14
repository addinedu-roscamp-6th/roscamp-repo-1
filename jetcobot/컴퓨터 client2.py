import socket, sys, json, base64, os

HOST = "192.168.5.1"
PORT = 5000
SAVE_DIR = "./jetcobot_capture_img2"

if len(sys.argv) < 2:
    print("usage: python3 client.py <name_or_id>")
    sys.exit(1)

target = sys.argv[1]

def save_b64_to_dir(b64_str, ts, prefix):
    if not b64_str or not ts: return None
    os.makedirs(SAVE_DIR, exist_ok=True)
    fname = f"{ts}_{prefix}.jpg"
    path = os.path.join(SAVE_DIR, fname)
    data = base64.b64decode(b64_str.encode())
    with open(path, "wb") as f:
        f.write(data)
    return path

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((HOST, PORT))
f = sock.makefile('rwb', buffering=0)

sock.sendall((f"start,{target}\n").encode())

_ = f.readline().decode().strip()

line = f.readline().decode().strip()
if not line.startswith("RESULT "):
    print("invalid response")
    sys.exit(1)

res = json.loads(line[len("RESULT "):])

if res.get("status") == "ok":
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
else:
    print("error:", res.get("message"))

sock.close()
