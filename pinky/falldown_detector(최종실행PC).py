# #!/usr/bin/env python3
# import cv2
# import numpy as np
# from ultralytics import YOLO
# from collections import deque
# import requests
# import datetime
# import rclpy
# from rclpy.node import Node
# from std_msgs.msg import String
# from geometry_msgs.msg import Twist
# import time

# class FallDetectionTestNode(Node):
#     POSE_CONNECTIONS = [
#         (0,1),(1,3),(0,2),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
#         (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)
#     ]

#     def __init__(self):
#         super().__init__('fall_detection_test_node')
#         self.publisher_ = self.create_publisher(String, '/fall_status', 10)
#         self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

#         self.model = YOLO("yolo11n-pose.pt")
#         self.url = 'http://192.168.0.154:5000/camera'
#         self.stream = requests.get(self.url, stream=True)
#         self.bytes_data = b''

#         self.prev_fall_state = False
#         self.fall_history = {}
#         self.last_state_change_time = None
#         self.STATE_HOLD_MAX = 0.5  # 사람 잠깐 사라져도 0.5초 유지

#         cv2.namedWindow("Fall Detection", flags=cv2.WINDOW_AUTOSIZE)
#         self.paused = False

#     def handle_keyboard_input(self):
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q') or key == 27:
#             return False
#         elif key == ord(' '):
#             self.paused = not self.paused
#         return True

#     def detect_fall(self, keypoints, person_id, angle_threshold=45, y_drop_threshold=50):
#         shoulder = (keypoints[5][:2] + keypoints[6][:2]) / 2
#         hip = (keypoints[11][:2] + keypoints[12][:2]) / 2

#         delta_y = abs(shoulder[1] - hip[1])
#         delta_x = abs(shoulder[0] - hip[0])
#         angle = np.degrees(np.arctan2(delta_y, delta_x))

#         if person_id not in self.fall_history:
#             self.fall_history[person_id] = deque(maxlen=5)
#         self.fall_history[person_id].append(shoulder[1])

#         y_change = self.fall_history[person_id][-1] - self.fall_history[person_id][0]
#         return angle < angle_threshold or y_change > y_drop_threshold

#     def draw_skeleton(self, frame, keypoints, color):
#         for p1_idx, p2_idx in self.POSE_CONNECTIONS:
#             if p1_idx < len(keypoints) and p2_idx < len(keypoints):
#                 p1, p2 = keypoints[p1_idx], keypoints[p2_idx]
#                 if p1[2] > 0.5 and p2[2] > 0.5:
#                     x1, y1, x2, y2 = map(int, [p1[0], p1[1], p2[0], p2[1]])
#                     cv2.line(frame, (x1, y1), (x2, y2), color, 2)
#                     cv2.circle(frame, (x1, y1), 4, color, -1)
#                     cv2.circle(frame, (x2, y2), 4, color, -1)

#     def publish_status(self, message: str):
#         msg = String()
#         msg.data = message
#         self.publisher_.publish(msg)
#         self.get_logger().info(f"Published: {message}")

#     def publish_cmd(self, stop: bool):
#         cmd = Twist()
#         cmd.linear.x = 0.0 if stop else 0.2
#         self.cmd_pub.publish(cmd)

#     def run(self):
#         self.get_logger().info("직진 테스트용 쓰러짐 감지 시작. ESC 또는 'q' 누르면 종료.")
#         try:
#             for chunk in self.stream.iter_content(chunk_size=1024):
#                 self.bytes_data += chunk
#                 start = self.bytes_data.find(b'\xff\xd8')
#                 end = self.bytes_data.find(b'\xff\xd9')
#                 if start != -1 and end != -1 and end > start:
#                     jpg = self.bytes_data[start:end+2]
#                     self.bytes_data = self.bytes_data[end+2:]
#                     frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
#                     if frame is None:
#                         continue

#                     if self.paused:
#                         if not self.handle_keyboard_input():
#                             break
#                         continue

#                     results = self.model(frame, verbose=False)[0]
#                     fall_detected_ids = set()

#                     if results.keypoints is not None:
#                         boxes = results.boxes.xyxy.cpu().numpy()
#                         keypoints = results.keypoints.data.cpu().numpy()

#                         for i in range(len(boxes)):
#                             if self.detect_fall(keypoints[i], i):
#                                 fall_detected_ids.add(i)

#                         for i, (box, keypoint) in enumerate(zip(boxes, keypoints)):
#                             color = (0,0,255) if i in fall_detected_ids else (0,255,0)
#                             x1, y1, x2, y2 = map(int, box)
#                             cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
#                             self.draw_skeleton(frame, keypoint, color)
#                             if i in fall_detected_ids:
#                                 cv2.putText(frame, "Fall Detected!", (x1, y1-10),
#                                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

#                         current_time = time.time()
#                         if fall_detected_ids:
#                             if not self.prev_fall_state:
#                                 self.prev_fall_state = True
#                                 self.publish_status(f"[FallDetected] time={datetime.datetime.now()}, location=PINKY")
#                                 self.publish_cmd(stop=True)
#                             self.last_state_change_time = current_time
#                         else:
#                             if self.prev_fall_state and (current_time - self.last_state_change_time) > self.STATE_HOLD_MAX:
#                                 self.prev_fall_state = False
#                                 self.publish_status("Normal Driving")
#                                 self.publish_cmd(stop=False)

#                     cv2.imshow("Fall Detection", frame)
#                     if not self.handle_keyboard_input():
#                         break
#         finally:
#             cv2.destroyAllWindows()
#             self.publish_cmd(stop=True)

# def main(args=None):
#     rclpy.init(args=args)
#     node = FallDetectionTestNode()
#     node.run()
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import requests
import datetime
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist
import time
import os

class FallDetectionTestNode(Node):
    POSE_CONNECTIONS = [
        (0,1),(1,3),(0,2),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
        (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)
    ]
    # 이미지 저장 경로 (로봇의 파일 시스템 경로)
    FALL_IMAGE_DIR = '/home/a/fall_images/'

    def __init__(self):
        super().__init__('fall_detection_test_node')
        self.publisher_ = self.create_publisher(String, '/fall_status', 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.model = YOLO("yolo11n-pose.pt")
        self.url = 'http://192.168.0.154:5000/camera'
        self.stream = requests.get(self.url, stream=True)
        self.bytes_data = b''

        self.prev_fall_state = False
        self.fall_history = {}
        self.last_state_change_time = None
        self.STATE_HOLD_MAX = 0.5

        cv2.namedWindow("Fall Detection", flags=cv2.WINDOW_AUTOSIZE)
        self.paused = False
        
        # 이미지 저장 디렉토리 확인 및 생성
        if not os.path.exists(self.FALL_IMAGE_DIR):
            os.makedirs(self.FALL_IMAGE_DIR)

    def handle_keyboard_input(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            return False
        elif key == ord(' '):
            self.paused = not self.paused
        return True

    def detect_fall(self, keypoints, person_id, angle_threshold=45, y_drop_threshold=50):
        shoulder = (keypoints[5][:2] + keypoints[6][:2]) / 2
        hip = (keypoints[11][:2] + keypoints[12][:2]) / 2

        delta_y = abs(shoulder[1] - hip[1])
        delta_x = abs(shoulder[0] - hip[0])
        angle = np.degrees(np.arctan2(delta_y, delta_x))

        if person_id not in self.fall_history:
            self.fall_history[person_id] = deque(maxlen=5)
        self.fall_history[person_id].append(shoulder[1])

        y_change = self.fall_history[person_id][-1] - self.fall_history[person_id][0]
        return angle < angle_threshold or y_change > y_drop_threshold

    def draw_skeleton(self, frame, keypoints, color):
        for p1_idx, p2_idx in self.POSE_CONNECTIONS:
            if p1_idx < len(keypoints) and p2_idx < len(keypoints):
                p1, p2 = keypoints[p1_idx], keypoints[p2_idx]
                if p1[2] > 0.5 and p2[2] > 0.5:
                    x1, y1, x2, y2 = map(int, [p1[0], p1[1], p2[0], p2[1]])
                    cv2.line(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(frame, (x1, y1), 4, color, -1)
                    cv2.circle(frame, (x2, y2), 4, color, -1)

    def publish_status(self, message: str, frame):
        msg = String()
        
        # 쓰러짐 감지 시 사진을 캡처하고 경로를 메시지에 추가
        if "FallDetected" in message:
            timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            image_filename = f"fall_{timestamp_str}.png"
            image_path = os.path.join(self.FALL_IMAGE_DIR, image_filename)
            
            # 이미지를 파일로 저장
            cv2.imwrite(image_path, frame)
            self.get_logger().info(f"Image saved to: {image_path}")
            
            # 메시지에 경로와 시간을 추가
            msg.data = f"FallDetected,{image_path},{datetime.datetime.now()}"
            self.get_logger().info(f"Published: {msg.data}")
        else:
            msg.data = message
            self.get_logger().info(f"Published: {message}")
            
        self.publisher_.publish(msg)

    def publish_cmd(self, stop: bool):
        cmd = Twist()
        cmd.linear.x = 0.0 if stop else 0.2
        self.cmd_pub.publish(cmd)

    def run(self):
        self.get_logger().info("직진 테스트용 쓰러짐 감지 시작. ESC 또는 'q' 누르면 종료.")
        try:
            for chunk in self.stream.iter_content(chunk_size=1024):
                self.bytes_data += chunk
                start = self.bytes_data.find(b'\xff\xd8')
                end = self.bytes_data.find(b'\xff\xd9')
                if start != -1 and end != -1 and end > start:
                    jpg = self.bytes_data[start:end+2]
                    self.bytes_data = self.bytes_data[end+2:]
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        continue

                    if self.paused:
                        if not self.handle_keyboard_input():
                            break
                        continue

                    results = self.model(frame, verbose=False)[0]
                    fall_detected_ids = set()

                    if results.keypoints is not None:
                        boxes = results.boxes.xyxy.cpu().numpy()
                        keypoints = results.keypoints.data.cpu().numpy()

                        for i in range(len(boxes)):
                            if self.detect_fall(keypoints[i], i):
                                fall_detected_ids.add(i)

                        for i, (box, keypoint) in enumerate(zip(boxes, keypoints)):
                            color = (0,0,255) if i in fall_detected_ids else (0,255,0)
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
                            self.draw_skeleton(frame, keypoint, color)
                            if i in fall_detected_ids:
                                cv2.putText(frame, "Fall Detected!", (x1, y1-10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

                        current_time = time.time()
                        if fall_detected_ids:
                            if not self.prev_fall_state:
                                self.prev_fall_state = True
                                self.publish_status("FallDetected", frame) # 👈 이 부분 수정
                                self.publish_cmd(stop=True)
                            self.last_state_change_time = current_time
                        else:
                            if self.prev_fall_state and (current_time - self.last_state_change_time) > self.STATE_HOLD_MAX:
                                self.prev_fall_state = False
                                self.publish_status("Normal Driving", frame=None)
                                self.publish_cmd(stop=False)

                    cv2.imshow("Fall Detection", frame)
                    if not self.handle_keyboard_input():
                        break
        finally:
            cv2.destroyAllWindows()
            self.publish_cmd(stop=True)

def main(args=None):
    rclpy.init(args=args)
    node = FallDetectionTestNode()
    node.run()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
