# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import os
# import math
# import subprocess
# from math import atan2
# from collections import deque

# import pickle
# import numpy as np
# import cv2
# import cv2.aruco as aruco

# import rclpy
# from rclpy.node import Node
# from rclpy.duration import Duration
# from rcl_interfaces.msg import SetParametersResult

# from geometry_msgs.msg import PointStamped, Twist
# from std_msgs.msg import Int32

# # ====== [필요시 설치] pip install mysql-connector-python ======
# import mysql.connector


# # ========================= 사용자 설정 =========================
# # -- 캘리브레이션 경로(환경에 맞게)
# CALIB_PATH = '/home/a/pinky_ws/src/aruco_navigator/aruco_navigator/camera_info.pkl'   # 예: '/home/a/pinky_ws/src/aruco_navigator/aruco_navigator/camera_info.pkl'
# CAM_INDEX = 2
# ARUCO_ROBOT_ID = 8
# MARKER_LENGTH = 0.10  # m

# # Homography 기준 4점 (undistort + resize(640x480) 기준 픽셀 좌표)
# img_pts = np.array([[206,347],[576,335],[577,111],[188,123]], dtype=np.float32)
# # 해당 픽셀 4점에 대응하는 실제 세계 좌표(단위: m)
# world_pts = np.array([[0.00,0.00],[1.60,0.00],[1.60,1.00],[0.00,1.00]], dtype=np.float32)

# # 순찰 웨이포인트 (테두리 루프)
# PATROL_WP = [
#     # 좌측 세로 ↑ (x=0.30)
#     (0.30, 0.05),
#     (0.30, 0.41),  # ★ 픽업 앵커 후보
#     (0.30, 0.68),  # ★ 픽업 앵커 후보
#     (0.30, 0.95),

#     # 상단 가로 → (y=0.95)
#     (0.56, 0.95),
#     (0.82, 0.95),
#     (1.08, 0.95),
#     (1.18, 0.95),  # ★ 드랍 앵커
#     (1.34, 0.95),
#     (1.60, 0.95),

#     # 우측 세로 ↓ (x=1.60)
#     (1.60, 0.65),
#     (1.60, 0.35),
#     (1.60, 0.05),

#     # 하단 가로 ← (y=0.05)
#     (1.34, 0.05),
#     (1.08, 0.05),
#     (0.82, 0.05),
#     (0.56, 0.05),
#     (0.30, 0.05),
# ]

# # ── 픽업 지점(앵커=루프 위, inside=실제 위치) ──
# P1_ANCHOR, P1_IN = (0.30, 0.41), (0.08, 0.41)
# P2_ANCHOR, P2_IN = (0.30, 0.68), (0.08, 0.70)
# P3_ANCHOR, P3_IN = (0.30, 0.95), (0.06, 0.95)

# # 드랍(고정)
# DROPOFF_ANCHOR = (1.18, 0.95)  # 루프 상
# DROPOFF_IN     = (1.18, 0.65)  # 루프 밖(실제 드랍)

# # 실제 지점에서의 대기 시간(초)
# WAIT_AT = {
#     DROPOFF_IN: 3.0,
#     P1_IN: 39.0,
#     P2_IN: 39.0,
#     P3_IN: 39.0,
# }

# # ── 픽업 스크립트 설정 ──
# # 픽업 위치에 도착했을 때 실행할 외부 스크립트 경로
# # ** 이 경로를 실제 jecobot_load.py가 있는 경로로 수정하세요. **
# PICKUP_SCRIPT_PATH = '/home/a/pinky_ws/src/aruco_navigator/aruco_navigator/jecobot_load.py'


# # 스케일(필요 시 m변환)
# SCALE_TO_M = 1.0

# # 제어/토픽
# USE_CMD_VEL = True
# CMD_VEL_TOPIC = '/cmd_vel'
# CTRL_RATE_HZ = 20  # (현재는 로거용)

# # ------------------ 제어 기본값 (ROS 파라미터로 덮어쓰기 가능) ------------------
# KP_LIN_DEFAULT = 0.24
# MAX_LIN_DEFAULT = 0.35
# MIN_LIN_DEFAULT = 0.06
# SLOWDOWN_RADIUS_DEFAULT = 0.55
# KP_ANG_DEFAULT = 0.90
# KD_ANG_DEFAULT = 0.30
# D_LPF_ALPHA_DEFAULT = 0.20
# MAX_ANG_DEFAULT = 1.20
# TURN_IN_PLACE_DEFAULT = 1.00
# ANG_DEADBAND_DEFAULT = 0.04
# GOAL_RADIUS_DEFAULT = 0.08
# POSE_STALE_SEC = 0.3
# LIN_ACC = 0.5
# ANG_ACC = 2.0

# # ------------------ DB 설정 ------------------
# DB_CONFIG = {
#     'host': '192.168.0.218',
#     'port': 3306,
#     'user': 'guardian',
#     'password': 'guardian@123',
#     'database': 'guardian',
# }
# # 폴링 주기(초)
# REQUEST_POLL_SEC = 3.0


# class ArucoNav(Node):
#     def __init__(self):
#         super().__init__('aruco_nav')

#         # 퍼블리셔
#         self.pose_pub   = self.create_publisher(PointStamped, '/robot_pose', 10)
#         self.front_pub  = self.create_publisher(PointStamped, '/robot_front', 10)
#         self.goal_pub   = self.create_publisher(PointStamped, '/nav_goal', 10)
#         self.target_pub = self.create_publisher(Int32, '/target_marker', 10)
#         self.cmd_pub    = self.create_publisher(Twist, CMD_VEL_TOPIC, 10) if USE_CMD_VEL else None

#         # 내부 상태
#         self.patrol_waypoints = PATROL_WP[:]
#         self.waypoints = self.patrol_waypoints[:]
#         self.goal_idx = 0
#         self.last_goal_sent = None
#         self.last_pose = None         # (x, y, yaw)
#         self.last_pose_time = None
#         self.seen = False
#         self.prev_cmd_lin = 0.0
#         self.prev_cmd_ang = 0.0
#         self.prev_time = self.get_clock().now()
#         # PD 상태
#         self.prev_ang_err = 0.0
#         self.d_ang_filt = 0.0
#         # 모드
#         self.mode = 'PATROL'  # 'PATROL' or 'MISSION'
#         # 대기 상태
#         self.waiting = False
#         self.wait_end_time = None
#         # 미션 대기열 (앵커/인사이드/라벨) + 해당 DB 요청 id 동기 큐
#         self.pending = deque()
#         self.pending_req_ids = deque()
#         # 현재 처리 중인 DB 요청 id(수동 키 트리거면 None)
#         self.current_request_id = None
#         # DB 중복 방지용(이미 큐에 올린 id 기억)
#         self.seen_request_ids = set()

#         # 카메라 파라미터 로드
#         with open(CALIB_PATH, 'rb') as f:
#             calib = pickle.load(f)
#         self.mtx  = calib['camera_matrix']
#         self.dist = calib['dist_coeff']

#         # ArUco 설정
#         self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
#         self.params = aruco.DetectorParameters()
#         self.marker_length = MARKER_LENGTH

#         # 고정 Homography
#         self.cached_H, _ = cv2.findHomography(img_pts, world_pts)

#         # 카메라 열기
#         cv2.namedWindow("win", cv2.WINDOW_NORMAL)
#         cv2.resizeWindow("win", 640, 480)
#         self.cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
#         if not self.cap.isOpened():
#             self.get_logger().error("카메라 열기 실패")
#             rclpy.shutdown()
#             return

#         # 첫 프레임 준비 + undistort 맵
#         ok, test = self.cap.read()
#         if not ok:
#             self.get_logger().error("첫 프레임 캡쳐 실패")
#             rclpy.shutdown()
#             return
#         raw_h, raw_w = test.shape[:2]
#         self.new_mtx, _ = cv2.getOptimalNewCameraMatrix(
#             self.mtx, self.dist, (raw_w, raw_h), 1, (raw_w, raw_h)
#         )
#         self.map1, self.map2 = cv2.initUndistortRectifyMap(
#             self.mtx, self.dist, None, self.new_mtx, (raw_w, raw_h), cv2.CV_16SC2
#         )
#         sx, sy = 640.0 / raw_w, 480.0 / raw_h
#         self.mtx_640 = self.new_mtx.copy()
#         self.mtx_640[0, 0] *= sx; self.mtx_640[1, 1] *= sy
#         self.mtx_640[0, 2] *= sx; self.mtx_640[1, 2] *= sy
#         self.dist_640 = np.zeros_like(self.dist)

#         # ROS 파라미터 선언 & 로드
#         self.declare_parameters('', [
#             ('kp_lin', KP_LIN_DEFAULT),
#             ('max_lin', MAX_LIN_DEFAULT),
#             ('min_lin', MIN_LIN_DEFAULT),
#             ('slowdown_radius', SLOWDOWN_RADIUS_DEFAULT),
#             ('kp_ang', KP_ANG_DEFAULT),
#             ('kd_ang', KD_ANG_DEFAULT),
#             ('d_lpf_alpha', D_LPF_ALPHA_DEFAULT),
#             ('max_ang', MAX_ANG_DEFAULT),
#             ('turn_in_place', TURN_IN_PLACE_DEFAULT),
#             ('ang_deadband', ANG_DEADBAND_DEFAULT),
#             ('goal_radius', GOAL_RADIUS_DEFAULT),
#         ])
#         self._load_params()
#         self.add_on_set_parameters_callback(self._on_params_changed)

#         # 첫 목표 발행
#         if self.waypoints:
#             self.publish_goal(*self.waypoints[0])

#         # DB 폴링 타이머(스레드 대신 안전하게 ROS 타이머 사용)
#         self.request_poll_timer = self.create_timer(REQUEST_POLL_SEC, self.poll_db_once)

#     # --------------------- 파라미터 헬퍼 ---------------------
#     def _load_params(self):
#         self.kp_lin          = float(self.get_parameter('kp_lin').value)
#         self.max_lin         = float(self.get_parameter('max_lin').value)
#         self.min_lin         = float(self.get_parameter('min_lin').value)
#         self.slowdown_radius = float(self.get_parameter('slowdown_radius').value)
#         self.kp_ang          = float(self.get_parameter('kp_ang').value)
#         self.kd_ang          = float(self.get_parameter('kd_ang').value)
#         self.d_lpf_alpha     = float(self.get_parameter('d_lpf_alpha').value)
#         self.max_ang         = float(self.get_parameter('max_ang').value)
#         self.turn_in_place   = float(self.get_parameter('turn_in_place').value)
#         self.ang_deadband    = float(self.get_parameter('ang_deadband').value)
#         self.goal_radius     = float(self.get_parameter('goal_radius').value)

#     def _on_params_changed(self, params):
#         for p in params:
#             if p.name == 'kp_lin':          self.kp_lin = float(p.value)
#             elif p.name == 'max_lin':         self.max_lin = float(p.value)
#             elif p.name == 'min_lin':         self.min_lin = float(p.value)
#             elif p.name == 'slowdown_radius': self.slowdown_radius = float(p.value)
#             elif p.name == 'kp_ang':          self.kp_ang = float(p.value)
#             elif p.name == 'kd_ang':          self.kd_ang = float(p.value)
#             elif p.name == 'd_lpf_alpha':     self.d_lpf_alpha = float(p.value)
#             elif p.name == 'max_ang':         self.max_ang = float(p.value)
#             elif p.name == 'turn_in_place':   self.turn_in_place = float(p.value)
#             elif p.name == 'ang_deadband':    self.ang_deadband = float(p.value)
#             elif p.name == 'goal_radius':     self.goal_radius = float(p.value)
#         return SetParametersResult(successful=True)

#     # --------------------- DB 유틸 ---------------------
#     def get_connection(self):
#         return mysql.connector.connect(**DB_CONFIG)

#     def poll_db_once(self):
#         """user_request 테이블에서 '처리중' 요청을 조회해 대기열에 추가/즉시 실행."""
#         try:
#             conn = self.get_connection()
#             cur = conn.cursor(dictionary=True)
#             cur.execute("""
#                 SELECT id, request_type
#                 FROM user_request
#                 WHERE delivery_status = '처리중'
#                 ORDER BY created_at ASC
#                 LIMIT 10
#             """)
#             rows = cur.fetchall()
#             cur.close()
#             conn.close()
#         except Exception as e:
#             self.get_logger().error(f"DB 폴링 실패: {e}")
#             return

#         # 신규만 처리(중복 방지)
#         for row in rows:
#             rid = row.get('id')
#             rtype = (row.get('request_type') or '').strip()
#             if rid in self.seen_request_ids or rid == self.current_request_id:
#                 continue

#             pick_anchor, pick_inside, label = self.request_type_to_pickup(rtype)
#             if pick_anchor is None:
#                 self.get_logger().warn(f"알 수 없는 요청 타입: id={rid}, type='{rtype}' → 무시")
#                 self.seen_request_ids.add(rid)
#                 continue

#             # 즉시 시작 가능하면 바로 시작, 아니면 대기열로
#             if self.mode == 'PATROL' and self.current_request_id is None:
#                 self.current_request_id = rid
#                 self.seen_request_ids.add(rid)
#                 self._start_mission_for(pick_anchor, pick_inside, label)
#                 self.get_logger().info(f"웹요청 즉시 시작: id={rid} type='{rtype}' → {label}")
#             else:
#                 self.pending.append((pick_anchor, pick_inside, label))
#                 self.pending_req_ids.append(rid)
#                 self.seen_request_ids.add(rid)
#                 self.get_logger().info(f"웹요청 대기열 추가: id={rid} type='{rtype}' → {label} (대기 {len(self.pending)}건)")

#     def update_request_status(self, request_id, status):
#         if request_id is None:
#             return
#         try:
#             conn = self.get_connection()
#             cur = conn.cursor()
#             cur.execute(
#                 "UPDATE user_request SET delivery_status=%s WHERE id=%s",
#                 (status, request_id)
#             )
#             conn.commit()
#             cur.close()
#             conn.close()
#             self.get_logger().info(f"DB 상태 업데이트: id={request_id} → '{status}'")
#         except Exception as e:
#             self.get_logger().error(f"DB 상태 업데이트 실패(id={request_id}, status={status}): {e}")

#     def request_type_to_pickup(self, request_type: str):
#         """웹 요청 타입을 픽업 위치로 매핑."""
#         t = request_type.lower()
#         if t.startswith('a') or t == 'coffee':
#             return P1_ANCHOR, P1_IN, 'P1'
#         elif t.startswith('b') or t == 'snack':
#             return P2_ANCHOR, P2_IN, 'P2'
#         elif t.startswith('c') or t == 'file':
#             return P3_ANCHOR, P3_IN, 'P3'
#         else:
#             return None, None, None

#     # --------------------- 유틸 ---------------------
#     def pixel_to_world(self, H, px, py):
#         pt = np.array([px, py, 1.0], dtype=np.float32)
#         w = H @ pt
#         w /= max(1e-9, w[2])
#         return float(w[0]), float(w[1])

#     def publish_point(self, topic_pub, x, y, frame='map'):
#         msg = PointStamped()
#         msg.header.frame_id = frame
#         msg.header.stamp = self.get_clock().now().to_msg()
#         msg.point.x = x * SCALE_TO_M
#         msg.point.y = y * SCALE_TO_M
#         msg.point.z = 0.0
#         topic_pub.publish(msg)

#     def publish_goal(self, gx, gy):
#         if self.last_goal_sent == (gx, gy):
#             return
#         self.publish_point(self.goal_pub, gx, gy)
#         self.last_goal_sent = (gx, gy)
#         if 0 <= self.goal_idx < len(self.waypoints):
#             cx, cy = self.waypoints[self.goal_idx]
#             self.get_logger().info(f"현재 목표: {self.goal_idx+1}/{len(self.waypoints)} ({cx:.3f}, {cy:.3f})")

#     def _set_waypoints(self, wps, reset_idx=True):
#         self.waypoints = [(float(x), float(y)) for (x, y) in wps]
#         if reset_idx:
#             self.goal_idx = 0
#         self.last_goal_sent = None
#         if self.waypoints:
#             self.publish_goal(*self.waypoints[self.goal_idx])

#     # ----- 루프 경로 헬퍼들 -----
#     def _closest_wp_index(self, pt):
#         x, y = pt
#         best_i, best_d = 0, 1e9
#         for i, (px, py) in enumerate(self.patrol_waypoints):
#             d = math.hypot(px - x, py - y)
#             if d < best_d:
#                 best_i, best_d = i, d
#         return best_i

#     def _edge_len(self, a, b):
#         ax, ay = a; bx, by = b
#         return math.hypot(ax - bx, ay - by)

#     def _path_len(self, pts):
#         if len(pts) < 2: return 0.0
#         return sum(self._edge_len(pts[i], pts[i+1]) for i in range(len(pts)-1))

#     def _forward_indices(self, s, e):
#         n = len(self.patrol_waypoints)
#         out = [s]
#         while out[-1] != e:
#             out.append((out[-1] + 1) % n)
#         return out

#     def _backward_indices(self, s, e):
#         n = len(self.patrol_waypoints)
#         out = [s]
#         while out[-1] != e:
#             out.append((out[-1] - 1 + n) % n)
#         return out

#     def _indices_to_points(self, idxs):
#         return [self.patrol_waypoints[i] for i in idxs]

#     def _shortest_path_points(self, s, e):
#         f_idx = self._forward_indices(s, e)
#         b_idx = self._backward_indices(s, e)
#         f_pts = self._indices_to_points(f_idx)
#         b_pts = self._indices_to_points(b_idx)
#         return f_pts if self._path_len(f_pts) <= self._path_len(b_pts) else b_pts

#     def _current_loop_index(self):
#         if self.last_pose is not None:
#             x, y, _ = self.last_pose
#             return self._closest_wp_index((x, y))
#         if self.last_goal_sent is not None:
#             return self._closest_wp_index(self.last_goal_sent)
#         return self._closest_wp_index(DROPOFF_ANCHOR)

#     # ── 미션 경로 생성 (주어진 픽업 앵커/인사이드) ──
#     def _start_mission_for(self, pick_anchor, pick_inside, label="P"):
#         # 미션 중이면 대기열로
#         if self.mode != 'PATROL':
#             self.pending.append((pick_anchor, pick_inside, label))
#             # DB 요청이면 pending_req_ids에 이미 push되어 있음(웹 트리거) / 수동 키면 None일 수 있음
#             self.get_logger().info(f"📬 미션 대기열 추가: {label} (대기 {len(self.pending)}건)")
#             return

#         # 출발점: 현재 루프 인덱스
#         cur_i  = self._current_loop_index()
#         pick_i = self._closest_wp_index(pick_anchor)
#         drop_i = self._closest_wp_index(DROPOFF_ANCHOR)

#         to_pick_pts      = self._shortest_path_points(cur_i,  pick_i)
#         pick_to_drop_pts = self._shortest_path_points(pick_i, drop_i)

#         def drop_first_if_same(seq, pt):
#             return seq[1:] if (len(seq) > 0 and seq[0] == pt) else seq

#         mission_path = []
#         mission_path += to_pick_pts
#         mission_path += [pick_inside, pick_anchor]                      # 픽업 외출/복귀
#         mission_path += drop_first_if_same(pick_to_drop_pts, pick_anchor) # 앵커 중복 제거
#         mission_path += [DROPOFF_IN, DROPOFF_ANCHOR]                      # 드랍 외출/복귀

#         self.mode = 'MISSION'
#         self._set_waypoints(mission_path, reset_idx=True)
#         self.get_logger().info(f"🚚 미션 시작: {label} → 루프 최단 경로 + 대기 포함")

#     def _run_pickup_script(self):
#         """픽업 지점에서 외부 스크립트(jecobot_load.py)를 실행합니다."""
#         self.get_logger().info("📦 픽업 동작 실행: jecobot_load.py를 호출합니다.")
        
#         try:
#             # os.path.dirname은 스크립트 경로의 디렉터리를 가져옴
#             script_dir = os.path.dirname(os.path.abspath(PICKUP_SCRIPT_PATH))
#             result = subprocess.run(
#                 ["python3", os.path.basename(PICKUP_SCRIPT_PATH)],
#                 cwd=script_dir,
#                 check=True,
#                 text=True,
#                 capture_output=True,
#                 timeout=30  # 스크립트 실행 최대 시간(초) 설정
#             )
#             self.get_logger().info(f"✅ 픽업 동작 완료: 스크립트가 성공적으로 종료되었습니다.")
#             self.get_logger().info(f"스크립트 출력:\n{result.stdout}")
#         except FileNotFoundError:
#             self.get_logger().error("❌ jecobot_load.py 파일을 찾을 수 없습니다. 경로를 확인하세요.")
#         except subprocess.TimeoutExpired:
#             self.get_logger().error(f"❌ 픽업 동작 실패: 스크립트 실행 시간이 초과되었습니다.")
#         except subprocess.CalledProcessError as e:
#             self.get_logger().error(f"❌ 픽업 동작 실패: 스크립트 실행 중 오류 발생. 반환 코드: {e.returncode}")
#             self.get_logger().error(f"스크립트 오류 출력:\n{e.stderr}")
#         except Exception as e:
#             self.get_logger().error(f"❌ 예외 발생: {e}")

#     # --------------------- 메인 루프 ---------------------
#     def process_frame(self):
#         ret, frame = self.cap.read()
#         if not ret:
#             return

#         und = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)
#         img = cv2.resize(und, (640, 480))

#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.params)
#         if ids is not None and len(ids) > 0:
#             aruco.drawDetectedMarkers(img, corners, ids)

#         if ids is not None and ARUCO_ROBOT_ID in ids.flatten():
#             rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
#                 corners, self.marker_length, self.mtx_640, self.dist_640
#             )
#             idx_robot = list(ids.flatten()).index(ARUCO_ROBOT_ID)

#             c_robot = corners[idx_robot][0]
#             px_c = float(c_robot[:, 0].mean())
#             py_c = float(c_robot[:, 1].mean())
#             rx, ry = self.pixel_to_world(self.cached_H, px_c, py_c)
#             self.publish_point(self.pose_pub, rx, ry)

#             # 전방점(마커 +Y)
#             front_obj = np.array([[0.0, self.marker_length, 0.0]], dtype=np.float32)
#             imgpts, _ = cv2.projectPoints(front_obj, rvecs[idx_robot], tvecs[idx_robot],
#                                           self.mtx_640, self.dist_640)
#             fx, fy = imgpts[0].ravel().astype(int)
#             wx_f, wy_f = self.pixel_to_world(self.cached_H, fx, fy)
#             self.publish_point(self.front_pub, wx_f, wy_f)

#             yaw = atan2(wy_f - ry, wx_f - rx)
#             self.last_pose = (rx, ry, yaw)
#             self.last_pose_time = self.get_clock().now()
#             self.seen = True

#             self.check_and_advance_goal(rx, ry)
#         else:
#             self.seen = False
#             if USE_CMD_VEL and self.cmd_pub is not None:
#                 self.cmd_pub.publish(Twist())

#         cv2.imshow("win", img)
#         self.handle_keys()  # 수동 디버깅용(원치 않으면 함수 내용 비워도 됨)
#         if USE_CMD_VEL and self.cmd_pub is not None:
#             self.control_loop()

#     def check_and_advance_goal(self, x, y):
#         if self.goal_idx >= len(self.waypoints):
#             return

#         # 대기 상태인 경우 기존 로직 유지
#         if self.waiting:
#             now = self.get_clock().now()
#             if now >= self.wait_end_time:
#                 # 대기 종료
#                 gx, gy = self.waypoints[self.goal_idx]
#                 just_waited_here = (gx, gy)
#                 self.waiting = False
#                 self.wait_end_time = None
#                 self.get_logger().info("⏯ 대기 종료, 다음 목표로 진행")
#                 self.goal_idx += 1

#                 # 드랍 실제 지점에서 대기였다면 → 해당 요청 완료 처리
#                 if just_waited_here == DROPOFF_IN and self.current_request_id is not None:
#                     self.update_request_status(self.current_request_id, '완료')
#                     self.current_request_id = None

#                 if self.goal_idx >= len(self.waypoints):
#                     if self.mode == 'MISSION':
#                         self.get_logger().info("✅ 미션 완료.")
#                         if self.pending:
#                             anchor, inside, label = self.pending.popleft()
#                             next_req_id = self.pending_req_ids.popleft() if self.pending_req_ids else None
#                             self.mode = 'PATROL'
#                             self.current_request_id = next_req_id
#                             self._start_mission_for(anchor, inside, label)
#                             return
#                         self.mode = 'PATROL'
#                         drop_i = self._closest_wp_index(DROPOFF_ANCHOR)
#                         next_i = (drop_i + 1) % len(self.patrol_waypoints)
#                         self._set_waypoints(self.patrol_waypoints, reset_idx=False)
#                         self.goal_idx = next_i
#                         nx, ny = self.waypoints[self.goal_idx]
#                         self.publish_goal(nx, ny)
#                     else:
#                         self.goal_idx = 0
#                         nx, ny = self.waypoints[self.goal_idx]
#                         self.publish_goal(nx, ny)
#                     return
#                 nx, ny = self.waypoints[self.goal_idx]
#                 self.publish_goal(nx, ny)
#             else:
#                 gx, gy = self.waypoints[self.goal_idx]
#                 self.publish_goal(gx, gy)
#             return

#         # 대기 상태가 아닐 때: 도착 판정
#         gx, gy = self.waypoints[self.goal_idx]
#         d = math.hypot(gx - x, gy - y)

#         if d <= self.goal_radius:
#             self.get_logger().info(f" 목표 {self.goal_idx+1} 도착 (d={d:.3f})")

#             # 픽업 위치에 도착하면 외부 스크립트 실행
#             if (gx, gy) in [P1_IN, P2_IN, P3_IN]:
#                 self.get_logger().info("📦 픽업 지점 도착! 로봇 정지 후 외부 스크립트 실행을 대기합니다.")
#                 # cmd_vel을 0으로 발행하여 로봇을 정지시킵니다.
#                 if self.cmd_pub:
#                     self.cmd_pub.publish(Twist())
#                 self._run_pickup_script()
#                 self.get_logger().info("✅ 외부 스크립트 완료, 다음 목표로 진행합니다.")
                
#             # 도착지점이 대기 대상이면 대기 시작
#             wait_sec = float(WAIT_AT.get((gx, gy), 0.0))
#             if wait_sec > 0.0:
#                 self.get_logger().info(f"⏸ 대기 시작 @ ({gx:.2f}, {gy:.2f}) for {wait_sec:.1f}s")
#                 self.wait_end_time = self.get_clock().now() + Duration(seconds=wait_sec)
#                 self.waiting = True
#                 self.publish_goal(gx, gy)
#                 return

#             # 다음 목표로
#             self.goal_idx += 1
#             if self.goal_idx >= len(self.waypoints):
#                 if self.mode == 'MISSION':
#                     self.get_logger().info("✅ 미션 완료.")
#                     if self.pending:
#                         anchor, inside, label = self.pending.popleft()
#                         next_req_id = self.pending_req_ids.popleft() if self.pending_req_ids else None
#                         self.mode = 'PATROL'
#                         self.current_request_id = next_req_id
#                         self._start_mission_for(anchor, inside, label)
#                         return
#                     self.mode = 'PATROL'
#                     drop_i = self._closest_wp_index(DROPOFF_ANCHOR)
#                     next_i = (drop_i + 1) % len(self.patrol_waypoints)
#                     self._set_waypoints(self.patrol_waypoints, reset_idx=False)
#                     self.goal_idx = next_i
#                     nx, ny = self.waypoints[self.goal_idx]
#                     self.publish_goal(nx, ny)
#                 else:
#                     self.goal_idx = 0
#                     nx, ny = self.waypoints[self.goal_idx]
#                     self.publish_goal(nx, ny)
#                 return
#             nx, ny = self.waypoints[self.goal_idx]
#             self.publish_goal(nx, ny)
#         else:
#             self.publish_goal(gx, gy)

#     # --------------------- 제어(P + PD) ---------------------
#     def control_loop(self):
#         if not USE_CMD_VEL or self.cmd_pub is None:
#             return
#         if self.last_pose is None or self.goal_idx >= len(self.waypoints):
#             return
#         if self.waiting:
#             self.cmd_pub.publish(Twist())
#             return
#         now = self.get_clock().now()
#         if (not self.seen) or (self.last_pose_time is None) or \
#            ((now - self.last_pose_time).nanoseconds * 1e-9 > POSE_STALE_SEC):
#             self.cmd_pub.publish(Twist())
#             self.prev_cmd_lin = 0.0
#             self.prev_cmd_ang = 0.0
#             self.prev_time = now
#             return
#         dt = max(1e-3, (now - self.prev_time).nanoseconds * 1e-9)
#         self.prev_time = now
#         x, y, yaw = self.last_pose
#         gx, gy = self.waypoints[self.goal_idx]
#         dx, dy = gx - x, gy - y
#         dist = math.hypot(dx, dy)
#         heading = atan2(dy, dx)
#         ang_err = (heading - yaw + math.pi) % (2 * math.pi) - math.pi
#         if abs(ang_err) < self.ang_deadband:
#             ang_err = 0.0
#         lin = self.kp_lin * dist
#         if dist < self.slowdown_radius:
#             lin *= dist / max(self.slowdown_radius, 1e-6)
#         d_raw = (ang_err - self.prev_ang_err) / dt if dt > 1e-4 else 0.0
#         alpha = max(0.0, min(1.0, self.d_lpf_alpha))
#         self.d_ang_filt = (1.0 - alpha) * self.d_ang_filt + alpha * d_raw
#         ang = self.kp_ang * ang_err + self.kd_ang * self.d_ang_filt
#         if abs(ang_err) > self.turn_in_place:
#             lin = 0.0
#         if dist >= 0.05 and abs(ang_err) <= self.turn_in_place and 0.0 < abs(lin) < self.min_lin:
#             lin = math.copysign(self.min_lin, lin)
#         lin = max(-self.max_lin, min(self.max_lin, lin))
#         ang = max(-self.max_ang, min(self.max_ang, ang))
#         def rate_limit(curr, target, limit):
#             delta = target - curr
#             if  delta >  limit: return curr + limit
#             elif delta < -limit: return curr - limit
#             return target
#         lin = rate_limit(self.prev_cmd_lin, lin, LIN_ACC * dt)
#         ang = rate_limit(self.prev_cmd_ang, ang, ANG_ACC * dt)
#         self.prev_cmd_lin = lin
#         self.prev_cmd_ang = ang
#         cmd = Twist()
#         cmd.linear.x  = lin
#         cmd.angular.z = ang
#         self.cmd_pub.publish(cmd)
#         self.get_logger().info(
#             f"ctrl dist={dist:.3f} ang_err={ang_err:.2f} d={self.d_ang_filt:.2f} lin={lin:.3f} ang={ang:.3f}"
#         )
#         self.prev_ang_err = ang_err

#     # --------------------- 입력(옵션) ---------------------
#     def handle_keys(self):
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#             if USE_CMD_VEL and self.cmd_pub is not None:
#                 self.cmd_pub.publish(Twist())
#             rclpy.shutdown()
#         elif key == ord('1'):
#             self._start_mission_for(P1_ANCHOR, P1_IN, "P1")
#         elif key == ord('2'):
#             self._start_mission_for(P2_ANCHOR, P2_IN, "P2")
#         elif key == ord('3'):
#             self._start_mission_for(P3_ANCHOR, P3_IN, "P3")

#     # --------------------- 종료 ---------------------
#     def on_shutdown(self):
#         try:
#             if self.cap is not None:
#                 self.cap.release()
#         except Exception:
#             pass
#         cv2.destroyAllWindows()


# def main(args=None):
#     rclpy.init(args=args)
#     node = ArucoNav()
#     try:
#         while rclpy.ok():
#             rclpy.spin_once(node, timeout_sec=0.0)
#             node.process_frame()
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.on_shutdown()
#         try:
#             node.destroy_node()
#         except Exception:
#             pass
#         rclpy.shutdown()


# if __name__ == '__main__':
#     main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import subprocess
from math import atan2
from collections import deque

import pickle
import numpy as np
import cv2
import cv2.aruco as aruco

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rcl_interfaces.msg import SetParametersResult

from geometry_msgs.msg import PointStamped, Twist
from std_msgs.msg import Int32

# ====== [필요시 설치] pip install mysql-connector-python ======
import mysql.connector


# ========================= 사용자 설정 =========================
# -- 캘리브레이션 경로(환경에 맞게)
CALIB_PATH = '/home/a/pinky_ws/src/aruco_navigator/aruco_navigator/camera_info.pkl'   # 예: '/home/a/pinky_ws/src/aruco_navigator/aruco_navigator/camera_info.pkl'
CAM_INDEX = 2
ARUCO_ROBOT_ID = 8
MARKER_LENGTH = 0.10  # m

# Homography 기준 4점 (undistort + resize(640x480) 기준 픽셀 좌표)
img_pts = np.array([[206,347],[576,335],[577,111],[188,123]], dtype=np.float32)
# 해당 픽셀 4점에 대응하는 실제 세계 좌표(단위: m)
world_pts = np.array([[0.00,0.00],[1.60,0.00],[1.60,1.00],[0.00,1.00]], dtype=np.float32)

# 순찰 웨이포인트 (테두리 루프)
PATROL_WP = [
    # 좌측 세로 ↑ (x=0.30)
    (0.30, 0.05),
    (0.30, 0.41),  # ★ 픽업 앵커 후보
    (0.30, 0.68),  # ★ 픽업 앵커 후보
    (0.30, 0.95),

    # 상단 가로 → (y=0.95)
    (0.56, 0.95),
    (0.82, 0.95),
    (1.08, 0.95),
    (1.18, 0.95),  # ★ 드랍 앵커
    (1.34, 0.95),
    (1.60, 0.95),

    # 우측 세로 ↓ (x=1.60)
    (1.60, 0.65),
    (1.60, 0.35),
    (1.60, 0.05),

    # 하단 가로 ← (y=0.05)
    (1.34, 0.05),
    (1.08, 0.05),
    (0.82, 0.05),
    (0.56, 0.05),
    (0.30, 0.05),
]

# ── 픽업 지점(앵커=루프 위, inside=실제 위치) ──
P1_ANCHOR, P1_IN = (0.30, 0.41), (0.08, 0.41)
P2_ANCHOR, P2_IN = (0.30, 0.68), (0.08, 0.70)
P3_ANCHOR, P3_IN = (0.30, 0.95), (0.06, 0.95)

# 드랍(고정)
DROPOFF_ANCHOR = (1.18, 0.95)  # 루프 상
DROPOFF_IN     = (1.18, 0.65)  # 루프 밖(실제 드랍)

# 실제 지점에서의 대기 시간(초)
WAIT_AT = {
    DROPOFF_IN: 3.0,
    P1_IN: 39.0,
    P2_IN: 39.0,
    P3_IN: 39.0,
}

# ── 픽업 스크립트 설정 ──
# 픽업 위치에 도착했을 때 실행할 외부 스크립트 경로
# ** 이 경로를 실제 jecobot_load.py가 있는 경로로 수정하세요. **
PICKUP_SCRIPT_PATH = '/home/a/pinky_ws/src/aruco_navigator/aruco_navigator/jecobot_load.py'


# 스케일(필요 시 m변환)
SCALE_TO_M = 1.0

# 제어/토픽
USE_CMD_VEL = True
CMD_VEL_TOPIC = '/cmd_vel'
CTRL_RATE_HZ = 20  # (현재는 로거용)

# ------------------ 제어 기본값 (ROS 파라미터로 덮어쓰기 가능) ------------------
KP_LIN_DEFAULT = 0.24
MAX_LIN_DEFAULT = 0.35
MIN_LIN_DEFAULT = 0.06
SLOWDOWN_RADIUS_DEFAULT = 0.55
KP_ANG_DEFAULT = 0.90
KD_ANG_DEFAULT = 0.30
D_LPF_ALPHA_DEFAULT = 0.20
MAX_ANG_DEFAULT = 1.20
TURN_IN_PLACE_DEFAULT = 1.00
ANG_DEADBAND_DEFAULT = 0.04
GOAL_RADIUS_DEFAULT = 0.08
POSE_STALE_SEC = 0.3
LIN_ACC = 0.5
ANG_ACC = 2.0

# ------------------ DB 설정 ------------------
DB_CONFIG = {
    'host': '192.168.0.218',
    'port': 3306,
    'user': 'guardian',
    'password': 'guardian@123',
    'database': 'guardian',
}
# 폴링 주기(초)
REQUEST_POLL_SEC = 3.0

# [추가] ----------------- 시각화 색상 -----------------
COLOR_PATROL_PATH = (255, 255, 255) # White
COLOR_MISSION_PATH = (255, 0, 0)    # Blue
COLOR_GOAL = (0, 255, 255)          # Yellow
COLOR_ROBOT = (0, 165, 255)         # Orange
COLOR_ROBOT_FRONT = (0, 255, 0)     # Green
COLOR_PICKUP = (0, 0, 255)          # Red
COLOR_DROPOFF = (255, 0, 255)       # Magenta
COLOR_TEXT = (255, 255, 255)        # White


class ArucoNav(Node):
    def __init__(self):
        super().__init__('aruco_nav')

        # 퍼블리셔
        self.pose_pub   = self.create_publisher(PointStamped, '/robot_pose', 10)
        self.front_pub  = self.create_publisher(PointStamped, '/robot_front', 10)
        self.goal_pub   = self.create_publisher(PointStamped, '/nav_goal', 10)
        self.target_pub = self.create_publisher(Int32, '/target_marker', 10)
        self.cmd_pub    = self.create_publisher(Twist, CMD_VEL_TOPIC, 10) if USE_CMD_VEL else None

        # 내부 상태
        self.patrol_waypoints = PATROL_WP[:]
        self.waypoints = self.patrol_waypoints[:]
        self.goal_idx = 0
        self.last_goal_sent = None
        self.last_pose = None         # (x, y, yaw)
        self.last_pose_time = None
        self.seen = False
        self.prev_cmd_lin = 0.0
        self.prev_cmd_ang = 0.0
        self.prev_time = self.get_clock().now()
        # PD 상태
        self.prev_ang_err = 0.0
        self.d_ang_filt = 0.0
        # 모드
        self.mode = 'PATROL'  # 'PATROL' or 'MISSION'
        # 대기 상태
        self.waiting = False
        self.wait_end_time = None
        # 미션 대기열 (앵커/인사이드/라벨) + 해당 DB 요청 id 동기 큐
        self.pending = deque()
        self.pending_req_ids = deque()
        # 현재 처리 중인 DB 요청 id(수동 키 트리거면 None)
        self.current_request_id = None
        # DB 중복 방지용(이미 큐에 올린 id 기억)
        self.seen_request_ids = set()

        # 카메라 파라미터 로드
        with open(CALIB_PATH, 'rb') as f:
            calib = pickle.load(f)
        self.mtx  = calib['camera_matrix']
        self.dist = calib['dist_coeff']

        # ArUco 설정
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
        self.params = aruco.DetectorParameters()
        self.marker_length = MARKER_LENGTH

        # 고정 Homography
        self.cached_H, _ = cv2.findHomography(img_pts, world_pts)
        self.cached_H_inv = np.linalg.inv(self.cached_H) # [추가] 역 Homography 계산

        # 카메라 열기
        cv2.namedWindow("win", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("win", 640, 480)
        self.cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.get_logger().error("카메라 열기 실패")
            rclpy.shutdown()
            return

        # 첫 프레임 준비 + undistort 맵
        ok, test = self.cap.read()
        if not ok:
            self.get_logger().error("첫 프레임 캡쳐 실패")
            rclpy.shutdown()
            return
        raw_h, raw_w = test.shape[:2]
        self.new_mtx, _ = cv2.getOptimalNewCameraMatrix(
            self.mtx, self.dist, (raw_w, raw_h), 1, (raw_w, raw_h)
        )
        self.map1, self.map2 = cv2.initUndistortRectifyMap(
            self.mtx, self.dist, None, self.new_mtx, (raw_w, raw_h), cv2.CV_16SC2
        )
        sx, sy = 640.0 / raw_w, 480.0 / raw_h
        self.mtx_640 = self.new_mtx.copy()
        self.mtx_640[0, 0] *= sx; self.mtx_640[1, 1] *= sy
        self.mtx_640[0, 2] *= sx; self.mtx_640[1, 2] *= sy
        self.dist_640 = np.zeros_like(self.dist)

        # ROS 파라미터 선언 & 로드
        self.declare_parameters('', [
            ('kp_lin', KP_LIN_DEFAULT),
            ('max_lin', MAX_LIN_DEFAULT),
            ('min_lin', MIN_LIN_DEFAULT),
            ('slowdown_radius', SLOWDOWN_RADIUS_DEFAULT),
            ('kp_ang', KP_ANG_DEFAULT),
            ('kd_ang', KD_ANG_DEFAULT),
            ('d_lpf_alpha', D_LPF_ALPHA_DEFAULT),
            ('max_ang', MAX_ANG_DEFAULT),
            ('turn_in_place', TURN_IN_PLACE_DEFAULT),
            ('ang_deadband', ANG_DEADBAND_DEFAULT),
            ('goal_radius', GOAL_RADIUS_DEFAULT),
        ])
        self._load_params()
        self.add_on_set_parameters_callback(self._on_params_changed)

        # 첫 목표 발행
        if self.waypoints:
            self.publish_goal(*self.waypoints[0])

        # DB 폴링 타이머(스레드 대신 안전하게 ROS 타이머 사용)
        self.request_poll_timer = self.create_timer(REQUEST_POLL_SEC, self.poll_db_once)

    # --------------------- 파라미터 헬퍼 ---------------------
    def _load_params(self):
        self.kp_lin          = float(self.get_parameter('kp_lin').value)
        self.max_lin         = float(self.get_parameter('max_lin').value)
        self.min_lin         = float(self.get_parameter('min_lin').value)
        self.slowdown_radius = float(self.get_parameter('slowdown_radius').value)
        self.kp_ang          = float(self.get_parameter('kp_ang').value)
        self.kd_ang          = float(self.get_parameter('kd_ang').value)
        self.d_lpf_alpha     = float(self.get_parameter('d_lpf_alpha').value)
        self.max_ang         = float(self.get_parameter('max_ang').value)
        self.turn_in_place   = float(self.get_parameter('turn_in_place').value)
        self.ang_deadband    = float(self.get_parameter('ang_deadband').value)
        self.goal_radius     = float(self.get_parameter('goal_radius').value)

    def _on_params_changed(self, params):
        for p in params:
            if p.name == 'kp_lin':          self.kp_lin = float(p.value)
            elif p.name == 'max_lin':         self.max_lin = float(p.value)
            elif p.name == 'min_lin':         self.min_lin = float(p.value)
            elif p.name == 'slowdown_radius': self.slowdown_radius = float(p.value)
            elif p.name == 'kp_ang':          self.kp_ang = float(p.value)
            elif p.name == 'kd_ang':          self.kd_ang = float(p.value)
            elif p.name == 'd_lpf_alpha':     self.d_lpf_alpha = float(p.value)
            elif p.name == 'max_ang':         self.max_ang = float(p.value)
            elif p.name == 'turn_in_place':   self.turn_in_place = float(p.value)
            elif p.name == 'ang_deadband':    self.ang_deadband = float(p.value)
            elif p.name == 'goal_radius':     self.goal_radius = float(p.value)
        return SetParametersResult(successful=True)

    # --------------------- DB 유틸 ---------------------
    def get_connection(self):
        return mysql.connector.connect(**DB_CONFIG)

    def poll_db_once(self):
        """user_request 테이블에서 '처리중' 요청을 조회해 대기열에 추가/즉시 실행."""
        try:
            conn = self.get_connection()
            cur = conn.cursor(dictionary=True)
            cur.execute("""
                SELECT id, request_type
                FROM user_request
                WHERE delivery_status = '처리중'
                ORDER BY created_at ASC
                LIMIT 10
            """)
            rows = cur.fetchall()
            cur.close()
            conn.close()
        except Exception as e:
            self.get_logger().error(f"DB 폴링 실패: {e}")
            return

        # 신규만 처리(중복 방지)
        for row in rows:
            rid = row.get('id')
            rtype = (row.get('request_type') or '').strip()
            if rid in self.seen_request_ids or rid == self.current_request_id:
                continue

            pick_anchor, pick_inside, label = self.request_type_to_pickup(rtype)
            if pick_anchor is None:
                self.get_logger().warn(f"알 수 없는 요청 타입: id={rid}, type='{rtype}' → 무시")
                self.seen_request_ids.add(rid)
                continue

            # 즉시 시작 가능하면 바로 시작, 아니면 대기열로
            if self.mode == 'PATROL' and self.current_request_id is None:
                self.current_request_id = rid
                self.seen_request_ids.add(rid)
                self._start_mission_for(pick_anchor, pick_inside, label)
                self.get_logger().info(f"웹요청 즉시 시작: id={rid} type='{rtype}' → {label}")
            else:
                self.pending.append((pick_anchor, pick_inside, label))
                self.pending_req_ids.append(rid)
                self.seen_request_ids.add(rid)
                self.get_logger().info(f"웹요청 대기열 추가: id={rid} type='{rtype}' → {label} (대기 {len(self.pending)}건)")

    def update_request_status(self, request_id, status):
        if request_id is None:
            return
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            cur.execute(
                "UPDATE user_request SET delivery_status=%s WHERE id=%s",
                (status, request_id)
            )
            conn.commit()
            cur.close()
            conn.close()
            self.get_logger().info(f"DB 상태 업데이트: id={request_id} → '{status}'")
        except Exception as e:
            self.get_logger().error(f"DB 상태 업데이트 실패(id={request_id}, status={status}): {e}")

    def request_type_to_pickup(self, request_type: str):
        """웹 요청 타입을 픽업 위치로 매핑."""
        t = request_type.lower()
        if t.startswith('a') or t == 'coffee':
            return P1_ANCHOR, P1_IN, 'P1'
        elif t.startswith('b') or t == 'snack':
            return P2_ANCHOR, P2_IN, 'P2'
        elif t.startswith('c') or t == 'file':
            return P3_ANCHOR, P3_IN, 'P3'
        else:
            return None, None, None

    # --------------------- 유틸 ---------------------
    def pixel_to_world(self, H, px, py):
        pt = np.array([px, py, 1.0], dtype=np.float32)
        w = H @ pt
        w /= max(1e-9, w[2])
        return float(w[0]), float(w[1])

    # [추가] 월드 좌표를 픽셀 좌표로 변환하는 함수
    def _world_to_pixel(self, x, y):
        pt = np.array([x, y, 1.0], dtype=np.float32)
        p = self.cached_H_inv @ pt
        p /= p[2]
        return int(p[0]), int(p[1])

    def publish_point(self, topic_pub, x, y, frame='map'):
        msg = PointStamped()
        msg.header.frame_id = frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x = x * SCALE_TO_M
        msg.point.y = y * SCALE_TO_M
        msg.point.z = 0.0
        topic_pub.publish(msg)

    def publish_goal(self, gx, gy):
        if self.last_goal_sent == (gx, gy):
            return
        self.publish_point(self.goal_pub, gx, gy)
        self.last_goal_sent = (gx, gy)
        if 0 <= self.goal_idx < len(self.waypoints):
            cx, cy = self.waypoints[self.goal_idx]
            self.get_logger().info(f"현재 목표: {self.goal_idx+1}/{len(self.waypoints)} ({cx:.3f}, {cy:.3f})")

    def _set_waypoints(self, wps, reset_idx=True):
        self.waypoints = [(float(x), float(y)) for (x, y) in wps]
        if reset_idx:
            self.goal_idx = 0
        self.last_goal_sent = None
        if self.waypoints:
            self.publish_goal(*self.waypoints[self.goal_idx])

    # ----- 루프 경로 헬퍼들 -----
    def _closest_wp_index(self, pt):
        x, y = pt
        best_i, best_d = 0, 1e9
        for i, (px, py) in enumerate(self.patrol_waypoints):
            d = math.hypot(px - x, py - y)
            if d < best_d:
                best_i, best_d = i, d
        return best_i

    def _edge_len(self, a, b):
        ax, ay = a; bx, by = b
        return math.hypot(ax - bx, ay - by)

    def _path_len(self, pts):
        if len(pts) < 2: return 0.0
        return sum(self._edge_len(pts[i], pts[i+1]) for i in range(len(pts)-1))

    def _forward_indices(self, s, e):
        n = len(self.patrol_waypoints)
        out = [s]
        while out[-1] != e:
            out.append((out[-1] + 1) % n)
        return out

    def _backward_indices(self, s, e):
        n = len(self.patrol_waypoints)
        out = [s]
        while out[-1] != e:
            out.append((out[-1] - 1 + n) % n)
        return out

    def _indices_to_points(self, idxs):
        return [self.patrol_waypoints[i] for i in idxs]

    def _shortest_path_points(self, s, e):
        f_idx = self._forward_indices(s, e)
        b_idx = self._backward_indices(s, e)
        f_pts = self._indices_to_points(f_idx)
        b_pts = self._indices_to_points(b_idx)
        return f_pts if self._path_len(f_pts) <= self._path_len(b_pts) else b_pts

    def _current_loop_index(self):
        if self.last_pose is not None:
            x, y, _ = self.last_pose
            return self._closest_wp_index((x, y))
        if self.last_goal_sent is not None:
            return self._closest_wp_index(self.last_goal_sent)
        return self._closest_wp_index(DROPOFF_ANCHOR)

    # ── 미션 경로 생성 (주어진 픽업 앵커/인사이드) ──
    def _start_mission_for(self, pick_anchor, pick_inside, label="P"):
        # 미션 중이면 대기열로
        if self.mode != 'PATROL':
            self.pending.append((pick_anchor, pick_inside, label))
            # DB 요청이면 pending_req_ids에 이미 push되어 있음(웹 트리거) / 수동 키면 None일 수 있음
            self.get_logger().info(f"📬 미션 대기열 추가: {label} (대기 {len(self.pending)}건)")
            return

        # 출발점: 현재 루프 인덱스
        cur_i  = self._current_loop_index()
        pick_i = self._closest_wp_index(pick_anchor)
        drop_i = self._closest_wp_index(DROPOFF_ANCHOR)

        to_pick_pts      = self._shortest_path_points(cur_i,  pick_i)
        pick_to_drop_pts = self._shortest_path_points(pick_i, drop_i)

        def drop_first_if_same(seq, pt):
            return seq[1:] if (len(seq) > 0 and seq[0] == pt) else seq

        mission_path = []
        mission_path += to_pick_pts
        mission_path += [pick_inside, pick_anchor]                      # 픽업 외출/복귀
        mission_path += drop_first_if_same(pick_to_drop_pts, pick_anchor) # 앵커 중복 제거
        mission_path += [DROPOFF_IN, DROPOFF_ANCHOR]                      # 드랍 외출/복귀

        self.mode = 'MISSION'
        self._set_waypoints(mission_path, reset_idx=True)
        self.get_logger().info(f"🚚 미션 시작: {label} → 루프 최단 경로 + 대기 포함")

    def _run_pickup_script(self):
        """픽업 지점에서 외부 스크립트(jecobot_load.py)를 실행합니다."""
        self.get_logger().info("📦 픽업 동작 실행: jecobot_load.py를 호출합니다.")
        
        try:
            # os.path.dirname은 스크립트 경로의 디렉터리를 가져옴
            script_dir = os.path.dirname(os.path.abspath(PICKUP_SCRIPT_PATH))
            result = subprocess.run(
                ["python3", os.path.basename(PICKUP_SCRIPT_PATH)],
                cwd=script_dir,
                check=True,
                text=True,
                capture_output=True,
                timeout=30  # 스크립트 실행 최대 시간(초) 설정
            )
            self.get_logger().info(f"✅ 픽업 동작 완료: 스크립트가 성공적으로 종료되었습니다.")
            self.get_logger().info(f"스크립트 출력:\n{result.stdout}")
        except FileNotFoundError:
            self.get_logger().error("❌ jecobot_load.py 파일을 찾을 수 없습니다. 경로를 확인하세요.")
        except subprocess.TimeoutExpired:
            self.get_logger().error(f"❌ 픽업 동작 실패: 스크립트 실행 시간이 초과되었습니다.")
        except subprocess.CalledProcessError as e:
            self.get_logger().error(f"❌ 픽업 동작 실패: 스크립트 실행 중 오류 발생. 반환 코드: {e.returncode}")
            self.get_logger().error(f"스크립트 오류 출력:\n{e.stderr}")
        except Exception as e:
            self.get_logger().error(f"❌ 예외 발생: {e}")

    # [추가] 경로 시각화 헬퍼 함수
    def _draw_path(self, img, waypoints, color, thickness=2, dotted=False, goal_idx=-1):
        if len(waypoints) < 2:
            return
        for i in range(len(waypoints) - 1):
            start_pt = self._world_to_pixel(*waypoints[i])
            end_pt   = self._world_to_pixel(*waypoints[i+1])
            if dotted:
                dash_len = 10
                dist = math.hypot(end_pt[0] - start_pt[0], end_pt[1] - start_pt[1])
                if dist > dash_len:
                    num_dashes = int(dist / dash_len)
                    for j in range(num_dashes):
                        p1 = (int(start_pt[0] + (end_pt[0] - start_pt[0]) * j / num_dashes),
                              int(start_pt[1] + (end_pt[1] - start_pt[1]) * j / num_dashes))
                        p2 = (int(start_pt[0] + (end_pt[0] - start_pt[0]) * (j + 0.5) / num_dashes),
                              int(start_pt[1] + (end_pt[1] - start_pt[1]) * (j + 0.5) / num_dashes))
                        cv2.line(img, p1, p2, color, thickness)
            else:
                cv2.line(img, start_pt, end_pt, color, thickness)
        if goal_idx != -1 and goal_idx < len(waypoints):
            gx, gy = waypoints[goal_idx]
            gpx, gpy = self._world_to_pixel(gx, gy)
            # 목표점 시각화 (큰 원)
            cv2.circle(img, (gpx, gpy), 10, COLOR_GOAL, -1)


    # --------------------- 메인 루프 ---------------------
    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        und = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)
        img = cv2.resize(und, (640, 480))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # [추가] 순찰 경로(점선) 시각화 (항상 표시)
        self._draw_path(img, self.patrol_waypoints, COLOR_PATROL_PATH, thickness=1, dotted=True)
        # [추가] 현재 미션/순찰 경로(실선) 시각화
        self._draw_path(img, self.waypoints, COLOR_MISSION_PATH, goal_idx=self.goal_idx)

        # [추가] 픽업/드랍 지점 시각화
        pickups = [
            {"name":"P1", "inside":P1_IN},
            {"name":"P2", "inside":P2_IN},
            {"name":"P3", "inside":P3_IN},
        ]
        for pickup in pickups:
            px_in, py_in = self._world_to_pixel(*pickup['inside'])
            cv2.circle(img, (px_in, py_in), 10, COLOR_PICKUP, -1)
            cv2.putText(img, pickup['name'], (px_in + 15, py_in + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)

        dx_in, dy_in = self._world_to_pixel(*DROPOFF_IN)
        cv2.circle(img, (dx_in, dy_in), 10, COLOR_DROPOFF, -1)
        cv2.putText(img, "DROPOFF", (dx_in + 15, dy_in + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)


        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.params)
        if ids is not None and len(ids) > 0:
            aruco.drawDetectedMarkers(img, corners, ids)

        if ids is not None and ARUCO_ROBOT_ID in ids.flatten():
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, self.mtx_640, self.dist_640
            )
            idx_robot = list(ids.flatten()).index(ARUCO_ROBOT_ID)

            c_robot = corners[idx_robot][0]
            px_c = float(c_robot[:, 0].mean())
            py_c = float(c_robot[:, 1].mean())
            rx, ry = self.pixel_to_world(self.cached_H, px_c, py_c)
            self.publish_point(self.pose_pub, rx, ry)

            # 전방점(마커 +Y)
            front_obj = np.array([[0.0, self.marker_length, 0.0]], dtype=np.float32)
            imgpts, _ = cv2.projectPoints(front_obj, rvecs[idx_robot], tvecs[idx_robot],
                                          self.mtx_640, self.dist_640)
            fx, fy = imgpts[0].ravel().astype(int)
            wx_f, wy_f = self.pixel_to_world(self.cached_H, fx, fy)
            self.publish_point(self.front_pub, wx_f, wy_f)

            yaw = atan2(wy_f - ry, wx_f - rx)
            self.last_pose = (rx, ry, yaw)
            self.last_pose_time = self.get_clock().now()
            self.seen = True

            self.check_and_advance_goal(rx, ry)

            # [추가] 로봇 시각화
            # 로봇 위치
            rob_px, rob_py = self._world_to_pixel(rx, ry)
            cv2.circle(img, (rob_px, rob_py), 8, COLOR_ROBOT, -1) # 주황색 원
            # 로봇 전방 방향
            front_px, front_py = self._world_to_pixel(wx_f, wy_f)
            cv2.line(img, (rob_px, rob_py), (front_px, front_py), COLOR_ROBOT_FRONT, 2)
            # 로봇 포즈 텍스트
            pos_text = f"Pos: ({rx:.2f}, {ry:.2f})"
            yaw_text = f"Yaw: {math.degrees(yaw):.1f} deg"
            cv2.putText(img, pos_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)
            cv2.putText(img, yaw_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)
        else:
            self.seen = False
            if USE_CMD_VEL and self.cmd_pub is not None:
                self.cmd_pub.publish(Twist())

        # [추가] 상태 텍스트
        status_text = f"Mode: {self.mode}"
        if self.waiting:
            remaining = self.wait_end_time.nanoseconds * 1e-9 - self.get_clock().now().nanoseconds * 1e-9
            status_text += f" (WAITING: {remaining:.1f}s)"
        cv2.putText(img, status_text, (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


        cv2.imshow("win", img)
        self.handle_keys()  # 수동 디버깅용(원치 않으면 함수 내용 비워도 됨)
        if USE_CMD_VEL and self.cmd_pub is not None:
            self.control_loop()

    def check_and_advance_goal(self, x, y):
        if self.goal_idx >= len(self.waypoints):
            return

        # 대기 상태인 경우 기존 로직 유지
        if self.waiting:
            now = self.get_clock().now()
            if now >= self.wait_end_time:
                # 대기 종료
                gx, gy = self.waypoints[self.goal_idx]
                just_waited_here = (gx, gy)
                self.waiting = False
                self.wait_end_time = None
                self.get_logger().info("⏯ 대기 종료, 다음 목표로 진행")
                self.goal_idx += 1

                # 드랍 실제 지점에서 대기였다면 → 해당 요청 완료 처리
                if just_waited_here == DROPOFF_IN and self.current_request_id is not None:
                    self.update_request_status(self.current_request_id, '완료')
                    self.current_request_id = None

                if self.goal_idx >= len(self.waypoints):
                    if self.mode == 'MISSION':
                        self.get_logger().info("✅ 미션 완료.")
                        if self.pending:
                            anchor, inside, label = self.pending.popleft()
                            next_req_id = self.pending_req_ids.popleft() if self.pending_req_ids else None
                            self.mode = 'PATROL'
                            self.current_request_id = next_req_id
                            self._start_mission_for(anchor, inside, label)
                            return
                        self.mode = 'PATROL'
                        drop_i = self._closest_wp_index(DROPOFF_ANCHOR)
                        next_i = (drop_i + 1) % len(self.patrol_waypoints)
                        self._set_waypoints(self.patrol_waypoints, reset_idx=False)
                        self.goal_idx = next_i
                        nx, ny = self.waypoints[self.goal_idx]
                        self.publish_goal(nx, ny)
                    else:
                        self.goal_idx = 0
                        nx, ny = self.waypoints[self.goal_idx]
                        self.publish_goal(nx, ny)
                    return
                nx, ny = self.waypoints[self.goal_idx]
                self.publish_goal(nx, ny)
            else:
                gx, gy = self.waypoints[self.goal_idx]
                self.publish_goal(gx, gy)
            return

        # 대기 상태가 아닐 때: 도착 판정
        gx, gy = self.waypoints[self.goal_idx]
        d = math.hypot(gx - x, gy - y)

        if d <= self.goal_radius:
            self.get_logger().info(f" 목표 {self.goal_idx+1} 도착 (d={d:.3f})")

            # 픽업 위치에 도착하면 외부 스크립트 실행
            if (gx, gy) in [P1_IN, P2_IN, P3_IN]:
                self.get_logger().info("📦 픽업 지점 도착! 로봇 정지 후 외부 스크립트 실행을 대기합니다.")
                # cmd_vel을 0으로 발행하여 로봇을 정지시킵니다.
                if self.cmd_pub:
                    self.cmd_pub.publish(Twist())
                self._run_pickup_script()
                self.get_logger().info("✅ 외부 스크립트 완료, 다음 목표로 진행합니다.")
                
            # 도착지점이 대기 대상이면 대기 시작
            wait_sec = float(WAIT_AT.get((gx, gy), 0.0))
            if wait_sec > 0.0:
                self.get_logger().info(f"⏸ 대기 시작 @ ({gx:.2f}, {gy:.2f}) for {wait_sec:.1f}s")
                self.wait_end_time = self.get_clock().now() + Duration(seconds=wait_sec)
                self.waiting = True
                self.publish_goal(gx, gy)
                return

            # 다음 목표로
            self.goal_idx += 1
            if self.goal_idx >= len(self.waypoints):
                if self.mode == 'MISSION':
                    self.get_logger().info("✅ 미션 완료.")
                    if self.pending:
                        anchor, inside, label = self.pending.popleft()
                        next_req_id = self.pending_req_ids.popleft() if self.pending_req_ids else None
                        self.mode = 'PATROL'
                        self.current_request_id = next_req_id
                        self._start_mission_for(anchor, inside, label)
                        return
                    self.mode = 'PATROL'
                    drop_i = self._closest_wp_index(DROPOFF_ANCHOR)
                    next_i = (drop_i + 1) % len(self.patrol_waypoints)
                    self._set_waypoints(self.patrol_waypoints, reset_idx=False)
                    self.goal_idx = next_i
                    nx, ny = self.waypoints[self.goal_idx]
                    self.publish_goal(nx, ny)
                else:
                    self.goal_idx = 0
                    nx, ny = self.waypoints[self.goal_idx]
                    self.publish_goal(nx, ny)
                return
            nx, ny = self.waypoints[self.goal_idx]
            self.publish_goal(nx, ny)
        else:
            self.publish_goal(gx, gy)

    # --------------------- 제어(P + PD) ---------------------
    def control_loop(self):
        if not USE_CMD_VEL or self.cmd_pub is None:
            return
        if self.last_pose is None or self.goal_idx >= len(self.waypoints):
            return
        if self.waiting:
            self.cmd_pub.publish(Twist())
            return
        now = self.get_clock().now()
        if (not self.seen) or (self.last_pose_time is None) or \
           ((now - self.last_pose_time).nanoseconds * 1e-9 > POSE_STALE_SEC):
            self.cmd_pub.publish(Twist())
            self.prev_cmd_lin = 0.0
            self.prev_cmd_ang = 0.0
            self.prev_time = now
            return
        dt = max(1e-3, (now - self.prev_time).nanoseconds * 1e-9)
        self.prev_time = now
        x, y, yaw = self.last_pose
        gx, gy = self.waypoints[self.goal_idx]
        dx, dy = gx - x, gy - y
        dist = math.hypot(dx, dy)
        heading = atan2(dy, dx)
        ang_err = (heading - yaw + math.pi) % (2 * math.pi) - math.pi
        if abs(ang_err) < self.ang_deadband:
            ang_err = 0.0
        lin = self.kp_lin * dist
        if dist < self.slowdown_radius:
            lin *= dist / max(self.slowdown_radius, 1e-6)
        d_raw = (ang_err - self.prev_ang_err) / dt if dt > 1e-4 else 0.0
        alpha = max(0.0, min(1.0, self.d_lpf_alpha))
        self.d_ang_filt = (1.0 - alpha) * self.d_ang_filt + alpha * d_raw
        ang = self.kp_ang * ang_err + self.kd_ang * self.d_ang_filt
        if abs(ang_err) > self.turn_in_place:
            lin = 0.0
        if dist >= 0.05 and abs(ang_err) <= self.turn_in_place and 0.0 < abs(lin) < self.min_lin:
            lin = math.copysign(self.min_lin, lin)
        lin = max(-self.max_lin, min(self.max_lin, lin))
        ang = max(-self.max_ang, min(self.max_ang, ang))
        def rate_limit(curr, target, limit):
            delta = target - curr
            if  delta >  limit: return curr + limit
            elif delta < -limit: return curr - limit
            return target
        lin = rate_limit(self.prev_cmd_lin, lin, LIN_ACC * dt)
        ang = rate_limit(self.prev_cmd_ang, ang, ANG_ACC * dt)
        self.prev_cmd_lin = lin
        self.prev_cmd_ang = ang
        cmd = Twist()
        cmd.linear.x  = lin
        cmd.angular.z = ang
        self.cmd_pub.publish(cmd)
        self.get_logger().info(
            f"ctrl dist={dist:.3f} ang_err={ang_err:.2f} d={self.d_ang_filt:.2f} lin={lin:.3f} ang={ang:.3f}"
        )
        self.prev_ang_err = ang_err

    # --------------------- 입력(옵션) ---------------------
    def handle_keys(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            if USE_CMD_VEL and self.cmd_pub is not None:
                self.cmd_pub.publish(Twist())
            rclpy.shutdown()
        elif key == ord('1'):
            self._start_mission_for(P1_ANCHOR, P1_IN, "P1")
        elif key == ord('2'):
            self._start_mission_for(P2_ANCHOR, P2_IN, "P2")
        elif key == ord('3'):
            self._start_mission_for(P3_ANCHOR, P3_IN, "P3")

    # --------------------- 종료 ---------------------
    def on_shutdown(self):
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = ArucoNav()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            node.process_frame()
    except KeyboardInterrupt:
        pass
    finally:
        node.on_shutdown()
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()
