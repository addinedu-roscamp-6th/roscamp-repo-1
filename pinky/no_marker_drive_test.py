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
from std_msgs.msg import Int32
import heapq

# ========================= 사용자 설정 =========================
CALIB_PATH = '/home/a/pinky_ws/src/aruco_navigator/aruco_navigator/camera_info.pkl'
CAM_INDEX = 2                  # VideoCapture 인덱스
ARUCO_ROBOT_ID = 8             # 로봇에 붙인 태그 ID
MARKER_LENGTH = 0.10           # m (로봇 마커 한 변 길이)
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

# 웨이포인트 (m) - 이제 A* 알고리즘의 최종 목표 지점으로 사용됩니다.
FINAL_GOALS_M = [
     (0.30, 0.05), #1
     (0.30, 0.25),
     (0.30, 0.50),
     (0.30, 0.67),
    # (0.30, 0.97), #2
    # (1.60, 0.93), #3
    # (1.60, 0.05), #4
    (0.03,0.67),
]

SCALE_TO_M = 1.0               # 퍼블리시 스케일(이미 m이므로 1.0)
USE_CMD_VEL = True
CMD_VEL_TOPIC = '/cmd_vel'
CTRL_RATE_HZ = 20              # 제어 주기(Hz)
# ------------------ 제어 기본값 (ROS 파라미터로 덮어쓰기 가능) ------------------
KP_LIN_DEFAULT = 0.15          # 선속 P 게인
KP_ANG_DEFAULT = 0.80          # 각속 P 게인 (PD의 P)
KD_ANG_DEFAULT = 0.40          # 각속 D 게인 (PD의 D)
D_LPF_ALPHA_DEFAULT = 0.30     # D 저역통과필터(0.05~0.3 권장)
MAX_LIN_DEFAULT = 0.35         # m/s
MAX_ANG_DEFAULT = 1.0          # rad/s
GOAL_RADIUS_DEFAULT = 0.08     # m (도착 판정 반경)
SLOWDOWN_RADIUS_DEFAULT = 0.40 # m (감속 시작 거리)
TURN_IN_PLACE_DEFAULT = 1.0    # rad (이보다 크면 제자리 회전)
MIN_LIN_DEFAULT = 0.06         # m/s (정마찰 극복용 최소 선속)
POSE_STALE_SEC = 0.3           # s (포즈 오래되면 정지)
LIN_ACC = 0.5                  # m/s^2 (선속 가감속 제한)
ANG_ACC = 2.0                  # rad/s^2 (각속 가감속 제한)

# ========================= A* 알고리즘 헬퍼 클래스 =========================
class NodeAStar:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0 # 출발점부터의 비용
        self.h = 0 # 목표점까지의 휴리스틱 비용
        self.f = 0 # 총 비용 (g + h)
    def __eq__(self, other):
        return self.position == other.position
    def __lt__(self, other):
        return self.f < other.f

class AStarPlanner:
    def __init__(self, grid_map, resolution=0.05):
        self.grid_map = grid_map
        self.height, self.width = grid_map.shape
        self.resolution = resolution
        self.obstacles = np.argwhere(grid_map == 1) # 장애물 위치 캐싱
        self.movements = [
            (0, -1), (0, 1), (-1, 0), (1, 0), # 4방향
            (-1, -1), (-1, 1), (1, -1), (1, 1) # 8방향
        ]

    def plan_path(self, start_pos, end_pos):
        start_node = NodeAStar(None, start_pos)
        end_node = NodeAStar(None, end_pos)
        open_list = []
        closed_list = set()
        heapq.heappush(open_list, start_node)

        while len(open_list) > 0:
            current_node = heapq.heappop(open_list)
            closed_list.add(current_node.position)

            if current_node.position == end_node.position:
                path = []
                current = current_node
                while current is not None:
                    path.append(current.position)
                    current = current.parent
                return path[::-1] # 경로를 거꾸로 반환

            for new_pos in self.movements:
                node_position = (current_node.position[0] + new_pos[0], current_node.position[1] + new_pos[1])
                if node_position in closed_list:
                    continue

                if not (0 <= node_position[0] < self.height and 0 <= node_position[1] < self.width):
                    continue

                # 장애물 체크
                if self.grid_map[node_position[0], node_position[1]] == 1:
                    continue

                new_node = NodeAStar(current_node, node_position)
                new_node.g = current_node.g + self.calc_distance(current_node.position, node_position)
                new_node.h = self.calc_distance(new_node.position, end_node.position)
                new_node.f = new_node.g + new_node.h

                if any(node for node in open_list if new_node == node and new_node.g > node.g):
                    continue
                
                heapq.heappush(open_list, new_node)
        return None # 경로를 찾지 못함

    def calc_distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# ================================================================================

class ArucoNav(Node):
    # A* 알고리즘을 위한 맵 정의 (미터 단위: 1.6m x 1.0m, 해상도 0.05m)
    # 맵 크기: (1.0 / 0.05) x (1.6 / 0.05) = 20 x 32
    # 0: 이동 가능, 1: 장애물
    GRID_RESOLUTION = 0.05
    GRID_MAP = np.zeros((int(1.0/GRID_RESOLUTION), int(1.6/GRID_RESOLUTION)), dtype=np.int8)
    # 예시 장애물 추가 (맵의 (y, x) 좌표를 기준으로)
    GRID_MAP[3:17, 10] = 1  # 세로 장애물 (y: 0.15m~0.85m, x: 0.5m)
    # GRID_MAP[5, 10:22] = 1 # 가로 장애물 (y: 0.25m, x: 0.5m~1.05m)
    # GRID_MAP[14, 10:22] = 1 # 가로 장애물 (y: 0.7m, x: 0.5m~1.05m)
    
    def __init__(self):
        super().__init__('aruco_nav')
        # 퍼블리셔
        self.pose_pub  = self.create_publisher(PointStamped, '/robot_pose', 10)
        self.front_pub = self.create_publisher(PointStamped, '/robot_front', 10)
        self.goal_pub  = self.create_publisher(PointStamped, '/nav_goal', 10)
        self.target_pub = self.create_publisher(Int32, '/target_marker', 10)
        self.cmd_pub = self.create_publisher(Twist, CMD_VEL_TOPIC, 10) if USE_CMD_VEL else None
        # A* 경로 계획기
        self.final_goals = FINAL_GOALS_M
        self.a_star_planner = AStarPlanner(self.GRID_MAP, self.GRID_RESOLUTION)
        self.final_goal_idx = 0
        
        # 내부 상태
        self.goal_idx = 0
        self.waypoints = [] # A*로 생성된 경로를 저장할 리스트
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
        # 카메라 파라미터 로드
        with open(CALIB_PATH, 'rb') as f:
            calib = pickle.load(f)
        self.mtx  = calib['camera_matrix']
        self.dist = calib['dist_coeff']
        # ArUco 설정
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
        self.params = aruco.DetectorParameters()
        # 길이/사이즈
        self.marker_length = MARKER_LENGTH  # m
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
        self.mtx_640[0, 0] *= sx  # fx
        self.mtx_640[1, 1] *= sy  # fy
        self.mtx_640[0, 2] *= sx  # cx
        self.mtx_640[1, 2] *= sy  # cy
        self.dist_640 = np.zeros_like(self.dist)  # undistort 이후 왜곡 0 가정
        # ROS 파라미터 선언 & 로드
        self.declare_parameters('', [
            ('kp_lin', KP_LIN_DEFAULT),
            ('kp_ang', KP_ANG_DEFAULT),
            ('kd_ang', KD_ANG_DEFAULT),
            ('d_lpf_alpha', D_LPF_ALPHA_DEFAULT),
            ('max_lin', MAX_LIN_DEFAULT),
            ('max_ang', MAX_ANG_DEFAULT),
            ('goal_radius', GOAL_RADIUS_DEFAULT),
            ('slowdown_radius', SLOWDOWN_RADIUS_DEFAULT),
            ('turn_in_place', TURN_IN_PLACE_DEFAULT),
            ('min_lin', MIN_LIN_DEFAULT)
        ])
        self._load_params()
        self.add_on_set_parameters_callback(self._on_params_changed)
    # --------------------- 파라미터 헬퍼 ---------------------
    def _load_params(self):
        self.kp_lin          = float(self.get_parameter('kp_lin').value)
        self.kp_ang          = float(self.get_parameter('kp_ang').value)
        self.kd_ang          = float(self.get_parameter('kd_ang').value)
        self.d_lpf_alpha     = float(self.get_parameter('d_lpf_alpha').value)
        self.max_lin         = float(self.get_parameter('max_lin').value)
        self.max_ang         = float(self.get_parameter('max_ang').value)
        self.goal_radius     = float(self.get_parameter('goal_radius').value)
        self.slowdown_radius = float(self.get_parameter('slowdown_radius').value)
        self.turn_in_place   = float(self.get_parameter('turn_in_place').value)
        self.min_lin         = float(self.get_parameter('min_lin').value)
    def _on_params_changed(self, params):
        for p in params:
            if p.name == 'kp_lin':            self.kp_lin = float(p.value)
            elif p.name == 'kp_ang':          self.kp_ang = float(p.value)
            elif p.name == 'kd_ang':          self.kd_ang = float(p.value)
            elif p.name == 'd_lpf_alpha':     self.d_lpf_alpha = float(p.value)
            elif p.name == 'max_lin':         self.max_lin = float(p.value)
            elif p.name == 'max_ang':         self.max_ang = float(p.value)
            elif p.name == 'goal_radius':     self.goal_radius = float(p.value)
            elif p.name == 'slowdown_radius': self.slowdown_radius = float(p.value)
            elif p.name == 'turn_in_place':   self.turn_in_place = float(p.value)
            elif p.name == 'min_lin':         self.min_lin = float(p.value)
        return SetParametersResult(successful=True)
    # --------------------------------------------------------
    # ---------- 유틸 ----------
    def pixel_to_world(self, H, px, py):
        pt = np.array([px, py, 1.0], dtype=np.float32)
        w = H @ pt
        w /= w[2]
        return float(w[0]), float(w[1])
    
    def world_to_grid(self, x, y):
        """월드 좌표(m)를 격자 맵 좌표로 변환"""
        gx = int(x / self.GRID_RESOLUTION)
        gy = int(y / self.GRID_RESOLUTION)
        # 경계 체크
        gy = max(0, min(gy, self.GRID_MAP.shape[0]-1))
        gx = max(0, min(gx, self.GRID_MAP.shape[1]-1))
        return (gy, gx) # (y, x) 형태로 반환
    
    def grid_to_world(self, gy, gx):
        """격자 맵 좌표를 월드 좌표(m)로 변환"""
        x = (gx + 0.5) * self.GRID_RESOLUTION
        y = (gy + 0.5) * self.GRID_RESOLUTION
        return (x, y)

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
            f"현재 경로 목표: {self.goal_idx+1}/{len(self.waypoints)}번 좌표 ({self.waypoints[self.goal_idx][0]:.3f}, {self.waypoints[self.goal_idx][1]:.3f})"
        )

    # ---------- 메인 루프 ----------
    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
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
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, self.mtx_640, self.dist_640
            )
            idx_robot = list(ids.flatten()).index(ARUCO_ROBOT_ID)
            c_robot = corners[idx_robot][0]
            px_c = float(c_robot[:, 0].mean())
            py_c = float(c_robot[:, 1].mean())
            robot_x, robot_y = self.pixel_to_world(self.cached_H, px_c, py_c)
            self.publish_point(self.pose_pub, robot_x, robot_y)
            front_obj = np.array([[0.0, self.marker_length, 0.0]], dtype=np.float32)
            imgpts, _ = cv2.projectPoints(front_obj, rvecs[idx_robot], tvecs[idx_robot],
                                          self.mtx_640, self.dist_640)
            fx, fy = imgpts[0].ravel().astype(int)
            front_x, front_y = self.pixel_to_world(self.cached_H, fx, fy)
            self.publish_point(self.front_pub, front_x, front_y)
            cv2.arrowedLine(img, (int(px_c), int(py_c)), (fx, fy), (0, 255, 0), 2, tipLength=0.2)
            yaw = atan2(front_y - robot_y, front_x - robot_x)
            self.last_pose = (robot_x, robot_y, yaw)
            self.last_pose_time = self.get_clock().now()
            self.seen = True
            self.get_logger().info(f"pose x={robot_x:.3f}, y={robot_y:.3f}, yaw={yaw:.2f}")

            # 목표 진행 로직 (A* 경로 생성 포함)
            self.check_and_advance_goal(robot_x, robot_y)
            self.draw_path(img)
        else:
            self.seen = False
            if USE_CMD_VEL and self.cmd_pub is not None:
                self.cmd_pub.publish(Twist())
        # Show
        cv2.imshow("win", img)
        if USE_CMD_VEL and self.cmd_pub is not None:
            self.control_loop()

    def draw_path(self, img):
        if not self.waypoints:
            return
        
        # 맵 영역을 이미지에 그리기 (디버깅용)
        h, w = self.GRID_MAP.shape
        # Homography 역변환을 사용하여 월드 좌표를 픽셀 좌표로 변환
        H_inv = np.linalg.inv(self.cached_H)
        
        # 장애물 그리기
        for y in range(h):
            for x in range(w):
                if self.GRID_MAP[y, x] == 1:
                    wx, wy = self.grid_to_world(y, x)
                    px_x, px_y = self.world_to_px(H_inv, wx, wy)
                    cv2.rectangle(img, (int(px_x), int(px_y)), 
                                  (int(px_x + self.GRID_RESOLUTION*640/1.6), int(px_y + self.GRID_RESOLUTION*480/1.0)), 
                                  (0, 0, 255), -1)

        # 경로 그리기
        path_pixels = []
        for x, y in self.waypoints:
            px, py = self.world_to_px(H_inv, x, y)
            path_pixels.append((int(px), int(py)))
        
        for i in range(1, len(path_pixels)):
            cv2.line(img, path_pixels[i-1], path_pixels[i], (255, 0, 0), 2)
            cv2.circle(img, path_pixels[i-1], 3, (255, 255, 0), -1)

    def world_to_px(self, H_inv, x, y):
        # Homography 역변환을 사용하여 월드 좌표를 픽셀 좌표로 변환
        pt = np.array([x, y, 1.0], dtype=np.float32)
        px_pt = H_inv @ pt
        px_pt /= px_pt[2]
        return px_pt[0], px_pt[1]

    def check_and_advance_goal(self, robot_x, robot_y):
        """A* 알고리즘을 사용한 경로 생성 및 목표 도달 체크"""
        # 다음 목표가 없으면 (모든 최종 목표를 순회했거나 처음 시작)
        if not self.waypoints:
            # 모든 최종 목표를 순회했으면 다시 처음으로 돌아감
            if self.final_goal_idx >= len(self.final_goals):
                self.get_logger().info("모든 최종 목표를 순회했습니다. 경로를 재시작합니다.")
                self.final_goal_idx = 0 
                # 로봇이 현재 위치에 머물러있을 경우, 첫 번째 목표가 현재 위치라면 다음 목표로 건너뜀
                start_x, start_y = self.final_goals[self.final_goal_idx]
                if hypot(robot_x - start_x, robot_y - start_y) <= self.goal_radius:
                    self.final_goal_idx += 1
                    if self.final_goal_idx >= len(self.final_goals):
                        # 한 바퀴를 돌았는데 첫 목표와 현재 위치가 가까운 경우, 정지
                        self.get_logger().info("시작 위치가 첫 목표와 같아 순환을 중단합니다.")
                        return

            # 새로운 최종 목표를 위한 경로 생성
            start_grid = self.world_to_grid(robot_x, robot_y)
            end_x, end_y = self.final_goals[self.final_goal_idx]
            end_grid = self.world_to_grid(end_x, end_y)
            
            self.get_logger().info(f"A* 경로 계획 시작: 시작점 {start_grid}, 목표점 {end_grid}")
            path = self.a_star_planner.plan_path(start_grid, end_grid)
            
            if path:
                self.waypoints = [self.grid_to_world(gy, gx) for gy, gx in path]
                self.goal_idx = 0
                self.publish_goal(*self.waypoints[self.goal_idx])
                self.get_logger().info("A* 경로 계획 완료. 경로 추종 시작.")
            else:
                self.get_logger().error("A* 경로를 찾을 수 없습니다. 로봇을 멈춥니다.")
                if USE_CMD_VEL and self.cmd_pub is not None:
                    self.cmd_pub.publish(Twist())
                return
        
        # 현재 웨이포인트 추종
        if self.goal_idx >= len(self.waypoints):
            return

        gx, gy = self.waypoints[self.goal_idx]
        d = hypot(gx - robot_x, gy - robot_y)
        
        if d <= self.goal_radius:
            self.get_logger().info(f" 경로점 {self.goal_idx+1} 도착 (d={d:.3f})")
            self.goal_idx += 1
            
            # 현재 최종 목표의 모든 웨이포인트를 완료했다면
            if self.goal_idx >= len(self.waypoints):
                self.get_logger().info(f"최종 목표 {self.final_goal_idx+1} 도착.")
                self.final_goal_idx += 1
                self.waypoints = [] # 다음 최종 목표를 위해 경로 리셋
                return
            
            nx, ny = self.waypoints[self.goal_idx]
            self.publish_goal(nx, ny)
        else:
            self.publish_goal(gx, gy)
            
    # ---------- 간단한 PD 각속 + P 선속 제어기 (/cmd_vel) ----------
    def control_loop(self):
        if not USE_CMD_VEL or self.cmd_pub is None:
            return
        if self.last_pose is None or self.goal_idx >= len(self.waypoints):
            # A* 경로가 없거나 모두 도착했을 때
            self.cmd_pub.publish(Twist())
            self.prev_cmd_lin = 0.0
            self.prev_cmd_ang = 0.0
            self.prev_time = self.get_clock().now()
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
            f"ctrl dist={dist:.3f} yaw={yaw:.2f} ang_err={ang_err:.2f} d={self.d_ang_filt:.2f} lin={lin:.3f} ang={ang:.3f}"
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
