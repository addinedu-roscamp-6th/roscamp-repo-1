#!/usr/bin/env python3  최종 2
import os
import pickle
import math
from math import atan2
from collections import deque
import numpy as np
import cv2
import cv2.aruco as aruco
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import PointStamped, Twist
from std_msgs.msg import Int32
# ========================= 사용자 설정 =========================
CALIB_PATH = '/home/ww/dev_ws/cam/camera_info.pkl'
CAM_INDEX = 2                  # VideoCapture 인덱스
ARUCO_ROBOT_ID = 8             # 로봇에 붙인 태그 ID
MARKER_LENGTH = 0.10           # m (로봇 마커 한 변 길이)
# Homography 기준 4점 (undistort + resize(640x480) 기준 픽셀 좌표)
img_pts = np.array([[206,347],[576,335],[577,111],[188,123]], dtype=np.float32)
# 해당 픽셀 4점에 대응하는 실제 세계 좌표(단위: m)
world_pts = np.array([[0.00,0.00],[1.60,0.00],[1.60,1.00],[0.00,1.00]], dtype=np.float32)
# 순찰 웨이포인트 (테두리 따라가도록 촘촘히; 드랍 앵커 (1.18,0.95) 포함)
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
# ── ★ 픽업 지점 3개 (anchor=루프 위, inside=실제 위치) ─────────────────────
P1_ANCHOR  = (0.30, 0.41); P1_IN  = (0.08, 0.41)
P2_ANCHOR  = (0.30, 0.68); P2_IN  = (0.08, 0.70)
P3_ANCHOR  = (0.30, 0.95); P3_IN  = (0.06, 0.95)
PICKUPS = [
    {"name":"P1", "anchor":P1_ANCHOR, "inside":P1_IN, "wait":3.0},
    {"name":"P2", "anchor":P2_ANCHOR, "inside":P2_IN, "wait":3.0},
    {"name":"P3", "anchor":P3_ANCHOR, "inside":P3_IN, "wait":3.0},
]
# 드랍(고정)
DROPOFF_ANCHOR = (1.18, 0.95)   # 경로 위
DROPOFF_IN     = (1.18, 0.65)   # 경로 밖
DROPOFF_WAIT_S = 3.0
# === 픽업/드랍 실제 지점에서 대기 시간(초) ===
WAIT_AT = {
    P1_IN: 3.0,
    P2_IN: 3.0,
    P3_IN: 3.0,
    DROPOFF_IN: DROPOFF_WAIT_S,
}
SCALE_TO_M = 1.0
USE_CMD_VEL = True
CMD_VEL_TOPIC = '/cmd_vel'
CTRL_RATE_HZ = 20
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
# ================================================================================
class ArucoNav(Node):
    def __init__(self):
        super().__init__('aruco_nav')
        # 퍼블리셔
        self.pose_pub  = self.create_publisher(PointStamped, '/robot_pose', 10)
        self.front_pub = self.create_publisher(PointStamped, '/robot_front', 10)
        self.goal_pub  = self.create_publisher(PointStamped, '/nav_goal', 10)
        self.target_pub = self.create_publisher(Int32, '/target_marker', 10)
        self.cmd_pub = self.create_publisher(Twist, CMD_VEL_TOPIC, 10) if USE_CMD_VEL else None
        # 내부 상태
        self.patrol_waypoints = PATROL_WP[:]
        self.waypoints = self.patrol_waypoints[:]
        self.goal_idx = 0
        self.last_goal_sent = None
        self.last_pose = None            # (x, y, yaw)
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
        # :우편물이_담긴_우편함: 픽업 대기열
        self.pending = deque()
        # 카메라 파라미터 로드
        with open(CALIB_PATH, 'rb') as f:
            calib = pickle.load(f)
        self.mtx  = calib['camera_matrix']
        self.dist = calib['dist_coeff']
        # ArUco 설정
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
        self.params = aruco.DetectorParameters()
        self.marker_length = MARKER_LENGTH  # meters
        # 고정 Homography
        self.cached_H, _ = cv2.findHomography(img_pts, world_pts)
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
            if p.name == 'kp_lin':            self.kp_lin = float(p.value)
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

    # --------------------- 유틸 ---------------------
    def pixel_to_world(self, H, px, py):
        pt = np.array([px, py, 1.0], dtype=np.float32)
        w = H @ pt
        w /= w[2]
        return float(w[0]), float(w[1])
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
        self.get_logger().info(
            f"현재 목표: {self.goal_idx+1}번 좌표 ({self.waypoints[self.goal_idx][0]:.3f}, {self.waypoints[self.goal_idx][1]:.3f})"
        )
    def _set_waypoints(self, wps, reset_idx=True):
        self.waypoints = [(float(x), float(y)) for (x, y) in wps]
        if reset_idx:
            self.goal_idx = 0
        self.last_goal_sent = None
        if self.waypoints:
            self.publish_goal(*self.waypoints[self.goal_idx])
    # ★ 경로 위 인덱스/거리 헬퍼들
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
    # :흰색_확인_표시: 현재 루프 인덱스 추정 (질러가기 방지 핵심)
    def _current_loop_index(self):
        if self.last_pose is not None:                 # 1) 실제 포즈가 있으면 그 기준
            x, y, _ = self.last_pose
            return self._closest_wp_index((x, y))
        if self.last_goal_sent is not None:            # 2) 마지막 목표 기준
            return self._closest_wp_index(self.last_goal_sent)
        return self._closest_wp_index(DROPOFF_ANCHOR)  # 3) 폴백
    # ── 미션 경로 생성 (주어진 픽업 앵커/인사이드) ──
    def _start_mission_for(self, pick_anchor, pick_inside, label="P"):
        # 미션 중이면 큐에 쌓고 리턴
        if self.mode != 'PATROL':
            self.pending.append((pick_anchor, pick_inside, label))
            self.get_logger().info(f":우편물이_담긴_우편함: 미션 대기열 추가: {label} (대기 {len(self.pending)}건)")
            return
        # ★ 출발점: 글로벌 루프상 '현재 위치' 인덱스
        cur_i  = self._current_loop_index()
        pick_i = self._closest_wp_index(pick_anchor)
        drop_i = self._closest_wp_index(DROPOFF_ANCHOR)
        to_pick_pts      = self._shortest_path_points(cur_i,  pick_i)
        pick_to_drop_pts = self._shortest_path_points(pick_i, drop_i)
        def drop_first_if_same(seq, pt):
            return seq[1:] if (len(seq) > 0 and seq[0] == pt) else seq
        mission_path = []
        mission_path += to_pick_pts
        mission_path += [pick_inside, pick_anchor]                        # 픽업 외출/복귀
        mission_path += drop_first_if_same(pick_to_drop_pts, pick_anchor) # 앵커 중복 제거
        mission_path += [DROPOFF_IN, DROPOFF_ANCHOR]                      # 드랍 외출/복귀
        self.mode = 'MISSION'
        self._set_waypoints(mission_path, reset_idx=True)
        self.get_logger().info(f":트럭: 미션 시작: {label} → 루프 최단 경로 + 3초 정지 포함")
    # --------------------- 메인 루프 ---------------------
    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        und = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)
        img = cv2.resize(und, (640, 480))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
        else:
            self.seen = False
            if USE_CMD_VEL and self.cmd_pub is not None:
                self.cmd_pub.publish(Twist())
        cv2.imshow("win", img)
        if USE_CMD_VEL and self.cmd_pub is not None:
            self.control_loop()
    def check_and_advance_goal(self, x, y):
        if self.goal_idx >= len(self.waypoints):
            return
        
    # 대기 중이면 시간 끝나면 다음 목표로 advance
        if self.waiting:
            now = self.get_clock().now()
            if now >= self.wait_end_time:
                self.waiting = False
                self.wait_end_time = None
                self.get_logger().info(":재생_일시_중지_기호: 대기 종료, 다음 목표로 진행")
                self.goal_idx += 1
                if self.goal_idx >= len(self.waypoints):
                    if self.mode == 'MISSION':
                        self.get_logger().info(":흰색_확인_표시: 미션 완료.")
                        # :앞쪽_화살표: 대기열이 있으면 즉시 다음 미션
                        if self.pending:
                            anchor, inside, label = self.pending.popleft()
                            self.mode = 'PATROL'  # 다음 미션 시작 조건
                            self._start_mission_for(anchor, inside, label)
                            return
                        # 없으면 순찰 복귀
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
        gx, gy = self.waypoints[self.goal_idx]
        d = math.hypot(gx - x, gy - y)
        if d <= self.goal_radius:
            # 도착지점이 대기 대상이면 대기 시작(advance 안함)
            wait_sec = WAIT_AT.get((gx, gy), 0.0)
            if wait_sec > 0.0:
                self.get_logger().info(f":일시_중지: 대기 시작 @ ({gx:.2f}, {gy:.2f}) for {wait_sec:.1f}s")
                self.wait_end_time = self.get_clock().now() + Duration(seconds=float(wait_sec))
                self.waiting = True
                self.publish_goal(gx, gy)
                return
            # 다음 목표로
            self.get_logger().info(f" 목표 {self.goal_idx+1} 도착 (d={d:.3f})")
            self.goal_idx += 1
            if self.goal_idx >= len(self.waypoints):
                if self.mode == 'MISSION':
                    self.get_logger().info(":흰색_확인_표시: 미션 완료.")
                    # :앞쪽_화살표: 대기열이 있으면 즉시 다음 미션
                    if self.pending:
                        anchor, inside, label = self.pending.popleft()
                        self.mode = 'PATROL'
                        self._start_mission_for(anchor, inside, label)
                        return
                    # 없으면 순찰 복귀
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
        # 대기 중이면 정지 유지
        if self.waiting:
            self.cmd_pub.publish(Twist())
            return
        now = self.get_clock().now()
        # 최근 포즈가 오래되면 정지
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
        # 선속: P + 근접 감속
        lin = self.kp_lin * dist
        if dist < self.slowdown_radius:
            lin *= dist / max(self.slowdown_radius, 1e-6)
        # 각속: PD (D 저역통과)
        d_raw = (ang_err - self.prev_ang_err) / dt if dt > 1e-4 else 0.0
        alpha = max(0.0, min(1.0, self.d_lpf_alpha))
        self.d_ang_filt = (1.0 - alpha) * self.d_ang_filt + alpha * d_raw
        ang = self.kp_ang * ang_err + self.kd_ang * self.d_ang_filt
        # 큰 각오차면 회전 먼저
        if abs(ang_err) > self.turn_in_place:
            lin = 0.0
        # 최소 선속(정마찰 극복)
        if dist >= 0.05 and abs(ang_err) <= self.turn_in_place and 0.0 < abs(lin) < self.min_lin:
            lin = math.copysign(self.min_lin, lin)
        # 속도 제한
        lin = max(-self.max_lin, min(self.max_lin, lin))
        ang = max(-self.max_ang, min(self.max_ang, ang))
        # 가감속 제한
        def rate_limit(curr, target, limit):
            delta = target - curr
            if   delta >  limit: return curr + limit
            elif delta < -limit: return curr - limit
            return target
        lin = rate_limit(self.prev_cmd_lin, lin, LIN_ACC * dt)
        ang = rate_limit(self.prev_cmd_ang, ang, ANG_ACC * dt)
        self.prev_cmd_lin = lin
        self.prev_cmd_ang = ang
        # 퍼블리시
        cmd = Twist()
        cmd.linear.x  = lin
        cmd.angular.z = ang
        self.cmd_pub.publish(cmd)
        self.get_logger().info(
            f"ctrl dist={dist:.3f} ang_err={ang_err:.2f} d={self.d_ang_filt:.2f} lin={lin:.3f} ang={ang:.3f}"
        )
        self.prev_ang_err = ang_err
    # --------------------- 입력 ---------------------
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
        elif key == ord('k'):
            # 현재 위치 기준 루프 위 최단거리로 가장 가까운 픽업 앵커 자동 선택
            if self.mode != 'PATROL':
                self.get_logger().info("이미 미션 진행 중이라 새 요청은 대기열에 추가됩니다.")
                return
            # ★ 자동선택도 현재 루프 인덱스를 사용
            cur_i = self._current_loop_index()
            candidates = [
                ("P1", P1_ANCHOR, P1_IN),
                ("P2", P2_ANCHOR, P2_IN),
                ("P3", P3_ANCHOR, P3_IN),
            ]
            best = None
            best_len = 1e9
            for name, anchor, inside in candidates:
                ai = self._closest_wp_index(anchor)
                pts = self._shortest_path_points(cur_i, ai)
                L = self._path_len(pts)
                if L < best_len:
                    best_len = L
                    best = (name, anchor, inside)
            if best is not None:
                name, anchor, inside = best
                self._start_mission_for(anchor, inside, name)
    # --------------------- 종료 ---------------------
    def on_shutdown(self):
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
def main():
    rclpy.init()
    node = ArucoNav()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.0)
            node.process_frame()
            node.handle_keys()
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