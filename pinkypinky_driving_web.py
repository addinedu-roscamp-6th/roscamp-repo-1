import threading
import time
import math
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
import mysql.connector

# ========================= 사용자 설정 =========================
CALIB_PATH = '/home/dev_ws/cam/camera_info.pkl'
CAM_INDEX = 2                  # VideoCapture 인덱스
ARUCO_ROBOT_ID = 8             # 로봇에 붙인 태그 ID
MARKER_LENGTH = 0.10           # m (로봇 마커 한 변 길이)
# Homography 기준 4점
img_pts = np.array([[206,347],[576,335],[577,111],[188,123]], dtype=np.float32)
world_pts = np.array([[0.00,0.00],[1.60,0.00],[1.60,1.00],[0.00,1.00]], dtype=np.float32)
# 순찰 웨이포인트
PATROL_WP = [
    (0.30, 0.05), (0.30, 0.41), (0.30, 0.68), (0.30, 0.95),
    (0.56, 0.95), (0.82, 0.95), (1.08, 0.95), (1.18, 0.95), (1.34, 0.95), (1.60, 0.95),
    (1.60, 0.65), (1.60, 0.35), (1.60, 0.05),
    (1.34, 0.05), (1.08, 0.05), (0.82, 0.05), (0.56, 0.05), (0.30, 0.05),
]
# 픽업 지점과 대기 시간
P1_ANCHOR, P1_IN = (0.30, 0.41), (0.08, 0.41)
P2_ANCHOR, P2_IN = (0.30, 0.68), (0.08, 0.70)
P3_ANCHOR, P3_IN = (0.30, 0.95), (0.06, 0.95)
PICKUPS = [
    {"name":"P1", "anchor":P1_ANCHOR, "inside":P1_IN, "wait":3.0},
    {"name":"P2", "anchor":P2_ANCHOR, "inside":P2_IN, "wait":3.0},
    {"name":"P3", "anchor":P3_ANCHOR, "inside":P3_IN, "wait":3.0},
]
DROPOFF_ANCHOR = (1.18, 0.95)
DROPOFF_IN = (1.18, 0.65)
DROPOFF_WAIT_S = 3.0
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
# 제어 기본값
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

# DB 설정
DB_CONFIG = {
    'host': '192.168.0.218',
    'port': 3306,
    'user': 'guardian',
    'password': 'guardian@123',
    'database': 'guardian',
}

class ArucoNav(Node):
    def __init__(self):
        super().__init__('aruco_nav')

        # 퍼블리셔
        self.pose_pub  = self.create_publisher(PointStamped, '/robot_pose', 10)
        self.front_pub = self.create_publisher(PointStamped, '/robot_front', 10)
        self.goal_pub  = self.create_publisher(PointStamped, '/nav_goal', 10)
        self.target_pub = self.create_publisher(Int32, '/target_marker', 10)
        self.cmd_pub = self.create_publisher(Twist, CMD_VEL_TOPIC, 10) if USE_CMD_VEL else None

        # 상태 초기화
        self.patrol_waypoints = PATROL_WP[:]
        self.waypoints = self.patrol_waypoints[:]
        self.goal_idx = 0
        self.last_goal_sent = None
        self.last_pose = None
        self.last_pose_time = None
        self.seen = False
        self.prev_cmd_lin = 0.0
        self.prev_cmd_ang = 0.0
        self.prev_time = self.get_clock().now()
        self.prev_ang_err = 0.0
        self.d_ang_filt = 0.0

        self.mode = 'PATROL'  # PATROL or MISSION
        self.waiting = False
        self.wait_end_time = None
        self.pending = deque()

        # DB 관리용
        self.current_request_id = None
        self.request_check_period_sec = 3.0

        # 카메라 파라미터 로드
        with open(CALIB_PATH, 'rb') as f:
            calib = pickle.load(f)
        self.mtx  = calib['camera_matrix']
        self.dist = calib['dist_coeff']

        # ArUco 설정
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
        self.params = aruco.DetectorParameters()
        self.marker_length = MARKER_LENGTH

        # Homography
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
        
        # 파라미터 선언 및 로드
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

        # 초기 목표 설정
        if self.waypoints:
            self.publish_goal(*self.waypoints[0])

        # DB 요청 모니터링 쓰레드 시작
        self.db_thread = threading.Thread(target=self.monitor_requests_loop, daemon=True)
        self.db_thread.start()

    def _load_params(self):
        self.kp_lin = float(self.get_parameter('kp_lin').value)
        self.max_lin = float(self.get_parameter('max_lin').value)
        self.min_lin = float(self.get_parameter('min_lin').value)
        self.slowdown_radius = float(self.get_parameter('slowdown_radius').value)
        self.kp_ang = float(self.get_parameter('kp_ang').value)
        self.kd_ang = float(self.get_parameter('kd_ang').value)
        self.d_lpf_alpha = float(self.get_parameter('d_lpf_alpha').value)
        self.max_ang = float(self.get_parameter('max_ang').value)
        self.turn_in_place = float(self.get_parameter('turn_in_place').value)
        self.ang_deadband = float(self.get_parameter('ang_deadband').value)
        self.goal_radius = float(self.get_parameter('goal_radius').value)

    def _on_params_changed(self, params):
        for p in params:
            if p.name == 'kp_lin': self.kp_lin = float(p.value)
            elif p.name == 'max_lin': self.max_lin = float(p.value)
            elif p.name == 'min_lin': self.min_lin = float(p.value)
            elif p.name == 'slowdown_radius': self.slowdown_radius = float(p.value)
            elif p.name == 'kp_ang': self.kp_ang = float(p.value)
            elif p.name == 'kd_ang': self.kd_ang = float(p.value)
            elif p.name == 'd_lpf_alpha': self.d_lpf_alpha = float(p.value)
            elif p.name == 'max_ang': self.max_ang = float(p.value)
            elif p.name == 'turn_in_place': self.turn_in_place = float(p.value)
            elif p.name == 'ang_deadband': self.ang_deadband = float(p.value)
            elif p.name == 'goal_radius': self.goal_radius = float(p.value)
        return SetParametersResult(successful=True)

    def get_connection(self):
        return mysql.connector.connect(**DB_CONFIG)

    def monitor_requests_loop(self):
        while rclpy.ok():
            if self.mode == 'PATROL' and self.current_request_id is None:
                req = self.fetch_next_processing_request()
                if req:
                    self.get_logger().info(f"새 요청 감지 : id={req['id']} type={req['request_type']}")
                    self.current_request_id = req['id']
                    pick_anchor, pick_inside, label = self.request_type_to_pickup(req['request_type'])
                    if pick_anchor:
                        self._start_mission_for(pick_anchor, pick_inside, label)
            time.sleep(self.request_check_period_sec)

    def fetch_next_processing_request(self):
        try:
            conn = self.get_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("""
                SELECT id, request_type FROM user_request
                WHERE delivery_status='처리중'
                ORDER BY created_at ASC
                LIMIT 1
            """)
            row = cursor.fetchone()
            cursor.close()
            conn.close()
            return row
        except Exception as e:
            self.get_logger().error(f"DB 요청 조회 실패: {e}")
            return None

    def update_request_status(self, request_id, status):
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE user_request SET delivery_status=%s WHERE id=%s",
                (status, request_id)
            )
            conn.commit()
            cursor.close()
            conn.close()
        except Exception as e:
            self.get_logger().error(f"DB 요청 상태 업데이트 실패: {e}")

    def request_type_to_pickup(self, request_type):
        # 타입에 따라 픽업 포인트 매핑 (예시)
        request_type = request_type.lower()
        if request_type.startswith('a') or request_type == 'coffee':
            return P1_ANCHOR, P1_IN, 'P1'
        elif request_type.startswith('b') or request_type == 'snack':
            return P2_ANCHOR, P2_IN, 'P2'
        elif request_type.startswith('c') or request_type == 'file':
            return P3_ANCHOR, P3_IN, 'P3'
        else:
            return None, None, None

    def mission_complete(self):
        self.get_logger().info("미션 완료: DB 상태 업데이트 및 드랍 미션 실행")
        if self.current_request_id:
            self.update_request_status(self.current_request_id, '완료')
            self.current_request_id = None
        self._start_mission_for(DROPOFF_ANCHOR, DROPOFF_IN, "Dropoff")

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

    # 대기 중인 경우 시간 끝나면 다음 목표로 진행
    if self.waiting:
        now = self.get_clock().now()
        if now >= self.wait_end_time:
            self.waiting = False
            self.wait_end_time = None
            self.get_logger().info(":재생_일시_중지_기호: 대기 종료, 다음 목표로 진행")

            # 현재 목표가 드랍 위치라면 미션 완전 종료 처리
            if self.mode == 'MISSION' and self.goal_idx == len(self.waypoints) - 1:
                self.get_logger().info("드랍 위치 대기 완료 - 미션 종료 및 DB 상태 업데이트")
                if self.current_request_id:
                    self.update_request_status(self.current_request_id, '완료')
                    self.current_request_id = None
                # 순찰 모드로 복귀
                self.mode = 'PATROL'
                self._set_waypoints(self.patrol_waypoints, reset_idx=True)
                self.goal_idx = 0
                nx, ny = self.waypoints[self.goal_idx]
                self.publish_goal(nx, ny)
                return

            # 그 외에는 다음 웨이포인트로 진행
            self.goal_idx += 1
            if self.goal_idx >= len(self.waypoints):
                if self.mode == 'MISSION':
                    self.get_logger().info("미션 완료 (드랍 위치 미포함) - 드랍 위치로 이동")
                    # 미션 중간 완료 시 (픽업 후) 드랍 미션 준비
                    # 다음 목표를 드랍 위치로 설정
                    self._set_waypoints([DROPOFF_IN, DROPOFF_ANCHOR], reset_idx=True)
                    self.goal_idx = 0
                    nx, ny = self.waypoints[0]
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
        wait_sec = WAIT_AT.get((gx, gy), 0.0)
        if wait_sec > 0.0:
            self.get_logger().info(f":일시_중지: 대기 시작 @ ({gx:.2f}, {gy:.2f}) for {wait_sec:.1f}s")
            self.wait_end_time = self.get_clock().now() + Duration(seconds=float(wait_sec))
            self.waiting = True
            self.publish_goal(gx, gy)
            return

        self.get_logger().info(f" 목표 {self.goal_idx+1} 도착 (d={d:.3f})")
        self.goal_idx += 1
        if self.goal_idx >= len(self.waypoints):
            if self.mode == 'MISSION':
                self.get_logger().info(":흰색_확인_표시: 미션 완료.")
                # 픽업 위치 도착 후 대기 후 드랍으로 이동하도록
                # 현재는 미션 완료 처리 대신 드랍 위치로 진행 미션 설정
                self._set_waypoints([DROPOFF_IN, DROPOFF_ANCHOR], reset_idx=True)
                self.goal_idx = 0
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
        if delta > limit:
            return curr + limit
        elif delta < -limit:
            return curr - limit
        return target

    lin = rate_limit(self.prev_cmd_lin, lin, LIN_ACC * dt)
    ang = rate_limit(self.prev_cmd_ang, ang, ANG_ACC * dt)

    self.prev_cmd_lin = lin
    self.prev_cmd_ang = ang

    cmd = Twist()
    cmd.linear.x = lin
    cmd.angular.z = ang
    self.cmd_pub.publish(cmd)

    self.get_logger().info(
        f"ctrl dist={dist:.3f} ang_err={ang_err:.2f} d={self.d_ang_filt:.2f} lin={lin:.3f} ang={ang:.3f}"
    )

    self.prev_ang_err = ang_err




def main():
    rclpy.init()
    node = ArucoNav()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
            node.process_frame()
            
    except KeyboardInterrupt:
        pass
    finally:
        node.on_shutdown()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
