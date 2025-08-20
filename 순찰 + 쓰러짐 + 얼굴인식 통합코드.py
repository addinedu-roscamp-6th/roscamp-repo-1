#!/usr/bin/env python3
import os
import pickle
import math
from math import hypot, atan2
import numpy as np
import cv2
import cv2.aruco as aruco
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import PointStamped, Twist
from std_msgs.msg import Int32, String
# 추가된 라이브러리 import
import mysql.connector
import datetime

# ========================= 사용자 설정 (추가) =========================
DB_INFO = {
    'host': '192.168.0.218',
    'port': 3306,
    'user': 'guardian',
    'password': 'guardian@123',
    'database': 'guardian'
}
# ====================================================================

# ========================= 사용자 설정 =========================
CALIB_PATH = '/home/a/pinky_ws/src/aruco_navigator/aruco_navigator/camera_info.pkl'
CAM_INDEX = 0
ARUCO_ROBOT_ID = 8
MARKER_LENGTH = 0.10
# Homography 기준 4점 (undistort + resize(640x480) 기준 픽셀 좌표)
img_pts = np.array([
    [206, 347],
    [576, 335],
    [577, 111],
    [188, 123],
], dtype=np.float32)
# 해당 픽셀 4점에 대응하는 실제 세계 좌표(단위: m)
world_pts = np.array([
    [0.00, 0.00],
    [1.60, 0.00],
    [1.60, 1.00],
    [0.00, 1.00],
], dtype=np.float32)
# 웨이포인트 (m)
WAYPOINTS_M = [
    (0.30, 0.05), #1
    (0.33, 0.95), #2
    (1.60, 0.95), #3
    (1.60, 0.05), #4
    (0.30, 0.05), #1
    (0.33, 0.95), #2
    (1.60, 0.95), #3
    (1.60, 0.05), #4
    (0.30, 0.05), #1
]

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
        
        # 쓰러짐 감지 서브스크라이버
        self.is_fall_detected = False
        self.fall_logged = False
        self.fall_sub = self.create_subscription(
            String, 
            '/fall_status', 
            self.fall_callback,
            10
        )
        
        # <<<<<<<<<<<<<<<< 추가된 부분 >>>>>>>>>>>>>>>>>>
        # 얼굴 인식 토픽 서브스크라이버
        self.face_rec_sub = self.create_subscription(
            String,
            '/recognized_person',
            self.face_recognition_callback,
            10
        )
        # <<<<<<<<<<<<<<<< 추가된 부분 끝 >>>>>>>>>>>>>>>>>>
        
        # DB 연결 (시작 시 한 번만)
        self.db_conn = self.connect_to_db()
        

        # 내부 상태
        self.goal_idx = 0
        self.waypoints = [(float(x), float(y)) for (x, y) in WAYPOINTS_M]
        self.last_goal_sent = None
        self.last_pose = None
        self.last_pose_time = None
        self.seen = False
        self.prev_cmd_lin = 0.0
        self.prev_cmd_ang = 0.0
        self.prev_time = self.get_clock().now()
        # PD 상태
        self.prev_ang_err = 0.0
        self.d_ang_filt = 0.0
        
        # 카메라 파라미터 로드
        with open(CALIB_PATH, 'rb') as f:
            calib = pickle.load(f)
        self.mtx  = calib['camera_matrix']
        self.dist = calib['dist_coeff']
        # ArUco 설정
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
        self.params = aruco.DetectorParameters()
        # 길이/사이즈
        self.marker_length = MARKER_LENGTH
        # 고정 Homography 계산 (undistort + 640x480 기준 픽셀 좌표 → 월드(m))
        self.cached_H, _ = cv2.findHomography(img_pts, world_pts)
        # 카메라 열기
        cv2.namedWindow("win", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("win", 640, 480)
        self.cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            self.get_logger().error("카메라 열기 실패")
            rclpy.shutdown()
            return
        # 첫 프레임으로 크기 파악 + undistort 맵/내부파라미터 준비
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
        self.mtx_640[0, 0] *= sx
        self.mtx_640[1, 1] *= sy
        self.mtx_640[0, 2] *= sx
        self.mtx_640[1, 2] *= sy
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
        # 시작 시 첫 목표 1회 발행
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
        
    # --------------------- DB 연결 헬퍼 (추가) ---------------------
    def connect_to_db(self):
        """데이터베이스에 연결하고 연결 객체를 반환합니다."""
        try:
            conn = mysql.connector.connect(**DB_INFO)
            self.get_logger().info("데이터베이스 연결 성공 ")
            return conn
        except mysql.connector.Error as err:
            self.get_logger().error(f"DB 연결 오류: {err} ")
            return None

    # --------------------------------------------------------
    # ---------- 유틸 ----------
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
        
    def face_recognition_callback(self, msg):
        """
        얼굴 인식 토픽 메시지를 받아 DB에 저장합니다.
        메시지 형식: "이름,시간"
        """
        try:
            name, timestamp_str = msg.data.split(',', 1)
            
            # (수정) "Unknown"이나 "None"인 경우에도 저장하도록 조건문 삭제
            
            # 메시지에서 받은 시간 문자열을 datetime 객체로 변환
            recognition_time = datetime.datetime.fromisoformat(timestamp_str)
            
            # DB에 데이터 삽입
            self.save_face_recognition_data(name, recognition_time)

        except ValueError as e:
            self.get_logger().error(f"얼굴 인식 메시지 형식 오류: {e} ")
            
    def save_face_recognition_data(self, name: str, recognized_at: datetime.datetime):
        """
        얼굴 인식 데이터를 DB의 face_logs 테이블에 저장합니다.
        """
        if not self.db_conn or not self.db_conn.is_connected():
            self.get_logger().error("DB 연결이 없어 얼굴 인식 데이터를 저장할 수 없습니다. ")
            return
            
        cursor = self.db_conn.cursor()
        sql = "INSERT INTO face_logs (name, recognized_at) VALUES (%s, %s)"
        data = (name, recognized_at)

        try:
            cursor.execute(sql, data)
            self.db_conn.commit()
            self.get_logger().info(f"얼굴 인식 데이터 DB 저장 성공: {name} ✅")
        except mysql.connector.Error as err:
            self.get_logger().error(f"DB 데이터 삽입 오류: {err} ")
        finally:
            cursor.close()
            
    def save_face_recognition_data(self, name: str, recognized_at: datetime.datetime):
        """
        얼굴 인식 데이터를 DB의 face_logs 테이블에 저장합니다.
        """
        if not self.db_conn or not self.db_conn.is_connected():
            self.get_logger().error("DB 연결이 없어 얼굴 인식 데이터를 저장할 수 없습니다. ")
            return
            
        cursor = self.db_conn.cursor()
        # image_445d7a.png에 보이는 것처럼, DB 테이블 컬럼명에 맞게 쿼리 수정
        sql = "INSERT INTO face_logs (name, recognized_at) VALUES (%s, %s)"
        data = (name, recognized_at)

        try:
            cursor.execute(sql, data)
            self.db_conn.commit()
            self.get_logger().info(f"얼굴 인식 데이터 DB 저장 성공: {name} ")
        except mysql.connector.Error as err:
            self.get_logger().error(f"DB 데이터 삽입 오류: {err} ")
        finally:
            cursor.close()
    # <<<<<<<<<<<<<<<< 추가된 부분 끝 >>>>>>>>>>>>>>>>>>
        
    # 쓰러짐 감지 콜백 함수
    def fall_callback(self, msg):
        """쓰러짐 감지 토픽 메시지를 처리하고 DB에 기록합니다."""
        try:
            data_parts = msg.data.split(',')
            if data_parts[0] == 'FallDetected':
                if not self.is_fall_detected:
                    self.get_logger().warn("쓰러짐 감지: 로봇을 정지합니다. ")
                    self.is_fall_detected = True
                
                # 쓰러짐이 감지되었고, 아직 DB에 기록되지 않았을 경우에만 저장
                if not self.fall_logged and len(data_parts) >= 3:
                    # 메시지에서 사진 경로와 시간을 추출하여 save_fall_data로 전달
                    image_path = data_parts[1]
                    timestamp_str = data_parts[2]
                    self.save_fall_data(image_path, timestamp_str)
                    
            elif 'Normal Driving' in msg.data:
                if self.is_fall_detected:
                    self.get_logger().info("쓰러짐 상태 해제: 정상 주행을 재개합니다. ")
                    self.is_fall_detected = False
                    self.fall_logged = False # 다음 감지를 위해 플래그 초기화
        except IndexError:
            self.get_logger().error(f"잘못된 메시지 형식입니다: {msg.data}")

    # 쓰러짐 데이터 저장 메서드
    def save_fall_data(self, fall_image_path: str, timestamp_str: str):
        """
        쓰러짐 데이터를 DB에 저장합니다.
        """
        if not self.db_conn or not self.db_conn.is_connected():
            self.get_logger().error("DB 연결이 없어 데이터를 저장할 수 없습니다. ")
            return
    
        # 1. 현재 로봇 위치 가져오기
        if self.last_pose:
            robot_x, robot_y, _ = self.last_pose
            location = f"({robot_x:.3f}, {robot_y:.3f})"
        else:
            location = "Unknown"
    
        # 2. 메시지에서 받은 시간 문자열을 datetime 객체로 변환
        try:
            fall_time = datetime.datetime.fromisoformat(timestamp_str)
        except ValueError as e:
            self.get_logger().error(f"시간 형식 변환 오류: {e}")
            fall_time = datetime.datetime.now() # 변환 실패 시 현재 시간 사용
    
        # 3. DB에 데이터 삽입
        cursor = self.db_conn.cursor()
        sql = "INSERT INTO fall_down_logs (falldown_detected_time, falldown_image_path, falldown_location) VALUES (%s, %s, %s)"
        data = (fall_time, fall_image_path, location)
    
        try:
            cursor.execute(sql, data)
            self.db_conn.commit()
            self.get_logger().info("쓰러짐 데이터 DB 저장 성공 ")
            self.fall_logged = True
        except mysql.connector.Error as err:
            self.get_logger().error(f"DB 데이터 삽입 오류: {err} ")
        finally:
            cursor.close()
            
    # ---------- 메인 루프 ----------
    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # ArucoNav 클래스에서 프레임에 접근할 수 있도록 클래스 변수로 저장
        self.last_frame = frame
        
        # Undistort & Resize (사전 계산한 remap 사용)
        und = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)
        img = cv2.resize(und, (640, 480))
        # 마커 감지
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.params)
        if ids is not None and len(ids) > 0:
            aruco.drawDetectedMarkers(img, corners, ids)
        # 로봇 마커 처리
        if ids is not None and ARUCO_ROBOT_ID in ids.flatten():
            # 포즈 추정 (보정+리사이즈된 내부 파라미터 사용)
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, self.mtx_640, self.dist_640
            )
            idx_robot = list(ids.flatten()).index(ARUCO_ROBOT_ID)
            # 로봇 중심 픽셀 → 월드 좌표
            c_robot = corners[idx_robot][0]
            px_c = float(c_robot[:, 0].mean())
            py_c = float(c_robot[:, 1].mean())
            robot_x, robot_y = self.pixel_to_world(self.cached_H, px_c, py_c)
            self.publish_point(self.pose_pub, robot_x, robot_y)
            # 전방점(마커 좌표계 +Y로 marker_length)
            front_obj = np.array([[0.0, self.marker_length, 0.0]], dtype=np.float32)
            imgpts, _ = cv2.projectPoints(front_obj, rvecs[idx_robot], tvecs[idx_robot],
                                          self.mtx_640, self.dist_640)
            fx, fy = imgpts[0].ravel().astype(int)
            front_x, front_y = self.pixel_to_world(self.cached_H, fx, fy)
            self.publish_point(self.front_pub, front_x, front_y)
            # 시각화
            cv2.arrowedLine(img, (int(px_c), int(py_c)), (fx, fy), (0, 255, 0), 2, tipLength=0.2)
            # yaw 추정 (로봇→front)
            yaw = atan2(front_y - robot_y, front_x - robot_x)
            self.last_pose = (robot_x, robot_y, yaw)
            self.last_pose_time = self.get_clock().now()
            self.seen = True
            # 목표 진행 로직
            self.check_and_advance_goal(robot_x, robot_y)
        else:
            # 로봇 안 보이면 속도 0, 플래그 false
            self.seen = False
            if USE_CMD_VEL and self.cmd_pub is not None:
                self.cmd_pub.publish(Twist())
        # Show
        cv2.imshow("win", img)
        if USE_CMD_VEL and self.cmd_pub is not None:
            self.control_loop()
    def check_and_advance_goal(self, robot_x, robot_y):
        """목표 도달 체크 후 다음 목표로"""
        if self.goal_idx >= len(self.waypoints):
            return
        gx, gy = self.waypoints[self.goal_idx]
        d = hypot(gx - robot_x, gy - robot_y)
        if d <= self.goal_radius:
            self.get_logger().info(f" 목표 {self.goal_idx+1} 도착 (d={d:.3f})")
            self.goal_idx += 1
            if self.goal_idx >= len(self.waypoints):
                self.get_logger().info("모든 목표 도달. 노드 종료합니다.")
                if USE_CMD_VEL and self.cmd_pub is not None:
                    self.cmd_pub.publish(Twist())
                rclpy.shutdown()
                return
            nx, ny = self.waypoints[self.goal_idx]
            self.publish_goal(nx, ny)
        else:
            # 아직 도착 전: 현재 목표(최초 포함) 발행 보장
            self.publish_goal(gx, gy)
    # ---------- P(선속) + PD(각속) ----------
    def control_loop(self):
        if not USE_CMD_VEL or self.cmd_pub is None:
            return
        
        # --- 추가된 부분: 쓰러짐 감지 시 정지 ---
        if self.is_fall_detected:
            self.cmd_pub.publish(Twist())
            self.prev_cmd_lin = 0.0
            self.prev_cmd_ang = 0.0
            self.prev_time = self.get_clock().now()
            return
        # --- 추가된 부분 끝 ---
        
        if self.last_pose is None or self.goal_idx >= len(self.waypoints):
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
        # 목표 방향/거리
        dx = gx - x
        dy = gy - y
        dist = math.hypot(dx, dy)
        heading = atan2(dy, dx)
        # 각도 오차 [-pi, pi]
        ang_err = (heading - yaw + math.pi) % (2 * math.pi) - math.pi
        # 아주 작은 각오차는 무시(직진 중 미세 흔들림 억제)
        if abs(ang_err) < self.ang_deadband:
            ang_err = 0.0
        # --------- 선속(전진) : P + 근접 감속 ----------
        lin = self.kp_lin * dist
        if dist < self.slowdown_radius:
            lin *= dist / max(self.slowdown_radius, 1e-6)
        # --------- 각속(회전) : PD ----------
        # D: 오차 미분 + 저역통과
        if dt > 1e-4:
            d_raw = (ang_err - self.prev_ang_err) / dt
        else:
            d_raw = 0.0
        alpha = max(0.0, min(1.0, self.d_lpf_alpha))
        self.d_ang_filt = (1.0 - alpha) * self.d_ang_filt + alpha * d_raw
        # PD 합성
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
        # ---- 가감속 제한(부드럽게) ----
        def rate_limit(curr, target, limit):
            delta = target - curr
            if delta > limit:  return curr + limit
            if delta < -limit: return curr - limit
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
        # 디버그 로그 + PD 상태 업데이트
        self.get_logger().info(
            f"ctrl dist={dist:.3f} ang_err={ang_err:.2f} d={self.d_ang_filt:.2f} lin={lin:.3f} ang={ang:.3f}"
        )
        self.prev_ang_err = ang_err
    # ---------- 입력 ----------
    def handle_keys(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            if USE_CMD_VEL and self.cmd_pub is not None:
                self.cmd_pub.publish(Twist())
            rclpy.shutdown()
    # ---------- 종료 ----------
    def on_shutdown(self):
        """노드 종료 시 DB 연결을 닫습니다."""
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        cv2.destroyAllWindows()
        
        # DB 연결 해제
        if self.db_conn and self.db_conn.is_connected():
            self.db_conn.close()
            self.get_logger().info("데이터베이스 연결 해제")

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
