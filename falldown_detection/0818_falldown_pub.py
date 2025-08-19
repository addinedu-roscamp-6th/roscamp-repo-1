# # # # from ultralytics import YOLO
# # # # import cv2
# # # # import xgboost as xgb
# # # # import pandas as pd
# # # # import numpy as np
# # # # import requests
# # # # import os
# # # # import socket
# # # # import datetime

# # # # class FallDetection:
# # # #     def __init__(self):
# # # #         # 모델 경로 설정
# # # #         script_dir = os.path.dirname(os.path.abspath(__file__))
# # # #         os.chdir(script_dir)
# # # #         models_dir = os.path.abspath(os.path.join(script_dir, "../00_Models"))

# # # #         # 모델 파일 경로
# # # #         yolo_model_path = os.path.join(models_dir, "yolov8n-pose.pt")
# # # #         xgb_model_path = os.path.join(models_dir, "model_weights.xgb")

# # # #         # 모델 로드
# # # #         self.model_yolo = YOLO(yolo_model_path)
# # # #         self.model_xgb = xgb.XGBClassifier()
# # # #         self.model_xgb.load_model(xgb_model_path)

# # # #         # 핑키 카메라 스트림 URL
# # # #         self.url = 'http://192.168.0.154:5000/camera'
# # # #         self.stream = requests.get(self.url, stream=True)
# # # #         self.bytes_data = b''

# # # #         # 쓰러짐 감지 상태
# # # #         self.fall_detected_count = 0
# # # #         self.fall_detection_threshold = 5  # 연속 감지 임계값
# # # #         self.prev_fall_state = None  # 이전 상태 저장용

# # # #     def send_status_to_guardian_p(self, message):
# # # #         HOST = '192.168.0.131'  # Guardian_P 서버 IP
# # # #         PORT = 8888
# # # #         try:
# # # #             with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
# # # #                 sock.connect((HOST, PORT))
# # # #                 sock.sendall(message.encode('utf-8'))
# # # #                 print(f"Guardian_P에 메시지 전송 완료: {message}")
# # # #         except Exception as e:
# # # #             print("Guardian_P TCP 전송 실패:", e)

# # # #     def run(self):
# # # #         print("핑키 카메라 쓰러짐 감지 시작. ESC 누르면 종료합니다.")
# # # #         try:
# # # #             for chunk in self.stream.iter_content(chunk_size=1024):
# # # #                 self.bytes_data += chunk
# # # #                 start = self.bytes_data.find(b'\xff\xd8')
# # # #                 end = self.bytes_data.find(b'\xff\xd9')
# # # #                 if start != -1 and end != -1 and end > start:
# # # #                     jpg = self.bytes_data[start:end + 2]
# # # #                     self.bytes_data = self.bytes_data[end + 2:]
# # # #                     frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
# # # #                     if frame is None:
# # # #                         continue

# # # #                     results = self.model_yolo(frame, verbose=False)
# # # #                     annotated_frame = results[0].plot(boxes=False)
# # # #                     fall_detected = False

# # # #                     for r in results:
# # # #                         bound_box = r.boxes.xyxy
# # # #                         conf = r.boxes.conf.tolist()
# # # #                         keypoints = r.keypoints.xyn.tolist()

# # # #                         for idx, box in enumerate(bound_box):
# # # #                             if conf[idx] > 0.75:
# # # #                                 x1, y1, x2, y2 = map(int, box.tolist())
# # # #                                 data = {}
# # # #                                 for j in range(len(keypoints[idx])):
# # # #                                     data[f'x{j}'] = keypoints[idx][j][0]
# # # #                                     data[f'y{j}'] = keypoints[idx][j][1]

# # # #                                 df = pd.DataFrame([data])
# # # #                                 cut = self.model_xgb.predict(df)
# # # #                                 binary_pred = (cut > 0.5).astype(int)

# # # #                                 if binary_pred[0] == 0:  # 쓰러짐 감지
# # # #                                     fall_detected = True
# # # #                                     cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
# # # #                                     cv2.putText(annotated_frame, 'FallDown', (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)

# # # #                     if fall_detected:
# # # #                         self.fall_detected_count += 1
# # # #                     else:
# # # #                         self.fall_detected_count = 0

# # # #                     # 임계값 넘으면 쓰러짐 상태로 판단
# # # #                     current_state = fall_detected and (self.fall_detected_count >= self.fall_detection_threshold)

# # # #                     # 상태 변경 시에만 메시지 전송
# # # #                     if self.prev_fall_state != current_state:
# # # #                         if current_state:
# # # #                             msg = f"[FallDetected] time={datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, location=PINKY"
# # # #                         else:
# # # #                             msg = "Normal Driving"
# # # #                         self.send_status_to_guardian_p(msg)
# # # #                         self.prev_fall_state = current_state

# # # #                     cv2.imshow("FallDown Detection from Pinky", annotated_frame)
# # # #                     if cv2.waitKey(1) & 0xFF == 27:
# # # #                         print("종료합니다.")
# # # #                         break
# # # #         except Exception as e:
# # # #             print(f"오류 발생: {e}")
# # # #         finally:
# # # #             cv2.destroyAllWindows()

# # # # if __name__ == "__main__":
# # # #     fd = FallDetection()
# # # #     fd.run()

# # # from ultralytics import YOLO
# # # import cv2
# # # import xgboost as xgb
# # # import pandas as pd
# # # import numpy as np
# # # import requests
# # # import os
# # # import datetime
# # # import rclpy
# # # from rclpy.node import Node
# # # from std_msgs.msg import String

# # # class FallDetection(Node):
# # #     def __init__(self):
# # #         super().__init__('fall_detection_node')

# # #         # ROS 퍼블리셔 (Guardian_P로 TCP 대신 토픽으로 전송)
# # #         self.publisher_ = self.create_publisher(String, '/fall_status', 10)

# # #         # 모델 경로 설정
# # #         script_dir = os.path.dirname(os.path.abspath(__file__))
# # #         os.chdir(script_dir)
# # #         models_dir = os.path.abspath(os.path.join(script_dir, "../00_Models"))

# # #         # 모델 파일 경로
# # #         yolo_model_path = os.path.join(models_dir, "yolov8n-pose.pt")
# # #         xgb_model_path = os.path.join(models_dir, "model_weights.xgb")

# # #         # 모델 로드
# # #         self.model_yolo = YOLO(yolo_model_path)
# # #         self.model_xgb = xgb.XGBClassifier()
# # #         self.model_xgb.load_model(xgb_model_path)

# # #         # 핑키 카메라 스트림 URL
# # #         self.url = 'http://192.168.0.154:5000/camera'
# # #         self.stream = requests.get(self.url, stream=True)
# # #         self.bytes_data = b''

# # #         # 쓰러짐 감지 상태
# # #         self.fall_detected_count = 0
# # #         self.fall_detection_threshold = 5  # 연속 감지 임계값
# # #         self.prev_fall_state = None  # 이전 상태 저장용

# # #     def publish_status(self, message: str):
# # #         """ROS2 토픽으로 상태 발행"""
# # #         msg = String()
# # #         msg.data = message
# # #         self.publisher_.publish(msg)
# # #         self.get_logger().info(f"Published: {message}")

# # #     def run(self):
# # #         self.get_logger().info("핑키 카메라 쓰러짐 감지 시작. ESC 누르면 종료합니다.")
# # #         try:
# # #             for chunk in self.stream.iter_content(chunk_size=1024):
# # #                 self.bytes_data += chunk
# # #                 start = self.bytes_data.find(b'\xff\xd8')
# # #                 end = self.bytes_data.find(b'\xff\xd9')
# # #                 if start != -1 and end != -1 and end > start:
# # #                     jpg = self.bytes_data[start:end + 2]
# # #                     self.bytes_data = self.bytes_data[end + 2:]
# # #                     frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
# # #                     if frame is None:
# # #                         continue

# # #                     results = self.model_yolo(frame, verbose=False)
# # #                     annotated_frame = results[0].plot(boxes=False)
# # #                     fall_detected = False

# # #                     for r in results:
# # #                         bound_box = r.boxes.xyxy
# # #                         conf = r.boxes.conf.tolist()
# # #                         keypoints = r.keypoints.xyn.tolist()

# # #                         for idx, box in enumerate(bound_box):
# # #                             if conf[idx] > 0.75:
# # #                                 x1, y1, x2, y2 = map(int, box.tolist())
# # #                                 data = {}
# # #                                 for j in range(len(keypoints[idx])):
# # #                                     data[f'x{j}'] = keypoints[idx][j][0]
# # #                                     data[f'y{j}'] = keypoints[idx][j][1]

# # #                                 # feature 순서 정렬 (x0,y0,x1,y1,...)
# # #                                 cols = []
# # #                                 for j in range(len(keypoints[idx])):
# # #                                     cols.append(f'x{j}')
# # #                                     cols.append(f'y{j}')

# # #                                 df = pd.DataFrame([data])[cols]
# # #                                 cut = self.model_xgb.predict(df)
# # #                                 binary_pred = (cut > 0.5).astype(int)

# # #                                 if binary_pred[0] == 0:  # 쓰러짐 감지
# # #                                     fall_detected = True
# # #                                     cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
# # #                                     cv2.putText(annotated_frame, 'FallDown', (x1, y1 - 10),
# # #                                                 cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)

# # #                     if fall_detected:
# # #                         self.fall_detected_count += 1
# # #                     else:
# # #                         self.fall_detected_count = 0

# # #                     # 임계값 넘으면 쓰러짐 상태로 판단
# # #                     current_state = fall_detected and (self.fall_detected_count >= self.fall_detection_threshold)

# # #                     # 상태 변경 시에만 ROS 토픽 발행
# # #                     if self.prev_fall_state != current_state:
# # #                         if current_state:
# # #                             msg = f"[FallDetected] time={datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, location=PINKY"
# # #                         else:
# # #                             msg = "Normal Driving"
# # #                         self.publish_status(msg)
# # #                         self.prev_fall_state = current_state

# # #                     cv2.imshow("FallDown Detection from Pinky", annotated_frame)
# # #                     if cv2.waitKey(1) & 0xFF == 27:
# # #                         self.get_logger().info("종료합니다.")
# # #                         break
# # #         except Exception as e:
# # #             self.get_logger().error(f"오류 발생: {e}")
# # #         finally:
# # #             cv2.destroyAllWindows()


# # # def main(args=None):
# # #     rclpy.init(args=args)
# # #     node = FallDetection()
# # #     node.run()
# # #     node.destroy_node()
# # #     rclpy.shutdown()


# # # if __name__ == "__main__":
# # #     main()


# # import cv2
# # import time
# # import numpy as np
# # from ultralytics import YOLO
# # from collections import deque
# # import requests
# # import datetime
# # import rclpy
# # from rclpy.node import Node
# # from std_msgs.msg import String

# # class FallDetectionNode(Node):
# #     def __init__(self):
# #         super().__init__('fall_detection_node')

# #         # ROS 퍼블리셔 (/fall_status 토픽 발행)
# #         self.publisher_ = self.create_publisher(String, '/fall_status', 10)

# #         # YOLO Pose 모델
# #         self.model = YOLO("yolo11n-pose.pt")

# #         # 카메라 스트림 URL
# #         self.url = 'http://192.168.0.154:5000/camera'
# #         self.stream = requests.get(self.url, stream=True)
# #         self.bytes_data = b''

# #         # 쓰러짐 감지 기록 및 히스테리시스
# #         self.fall_history = {}
# #         self.fall_detected_count = 0
# #         self.normal_count = 0
# #         self.fall_detection_threshold = 5
# #         self.recovery_threshold = 15
# #         self.prev_fall_state = False

# #         # 윈도우
# #         cv2.namedWindow("Fall Detection", flags=cv2.WINDOW_AUTOSIZE)
# #         self.paused = False

# #     def handle_keyboard_input(self):
# #         key = cv2.waitKey(1) & 0xFF
# #         if key == ord('q') or key == 27:
# #             return False
# #         elif key == ord(' '):
# #             self.paused = not self.paused
# #         return True

# #     def detect_fall(self, keypoints, person_id, angle_threshold=45, y_drop_threshold=50):
# #         shoulder = (keypoints[5][:2] + keypoints[6][:2]) / 2
# #         hip = (keypoints[11][:2] + keypoints[12][:2]) / 2

# #         delta_y = abs(shoulder[1] - hip[1])
# #         delta_x = abs(shoulder[0] - hip[0])
# #         angle = np.degrees(np.arctan2(delta_y, delta_x))

# #         if person_id not in self.fall_history:
# #             self.fall_history[person_id] = deque(maxlen=5)
# #         self.fall_history[person_id].append(shoulder[1])

# #         y_change = self.fall_history[person_id][-1] - self.fall_history[person_id][0]

# #         if angle < angle_threshold or y_change > y_drop_threshold:
# #             return True
# #         return False

# #     POSE_CONNECTIONS = [
# #         (0,1),(1,3),(0,2),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
# #         (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)
# #     ]

# #     def draw_skeleton(self, frame, keypoints, color):
# #         for p1_idx, p2_idx in self.POSE_CONNECTIONS:
# #             if p1_idx < len(keypoints) and p2_idx < len(keypoints):
# #                 p1, p2 = keypoints[p1_idx], keypoints[p2_idx]
# #                 if p1[2] > 0.5 and p2[2] > 0.5:
# #                     x1, y1, x2, y2 = int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])
# #                     cv2.line(frame, (x1, y1), (x2, y2), color, 2)
# #                     cv2.circle(frame, (x1, y1), 4, color, -1)
# #                     cv2.circle(frame, (x2, y2), 4, color, -1)

# #     def publish_status(self, message: str):
# #         msg = String()
# #         msg.data = message
# #         self.publisher_.publish(msg)
# #         self.get_logger().info(f"Published: {message}")

# #     def run(self):
# #         self.get_logger().info("핑키 카메라 쓰러짐 감지 시작. ESC 누르면 종료합니다.")
# #         try:
# #             for chunk in self.stream.iter_content(chunk_size=1024):
# #                 self.bytes_data += chunk
# #                 start = self.bytes_data.find(b'\xff\xd8')
# #                 end = self.bytes_data.find(b'\xff\xd9')
# #                 if start != -1 and end != -1 and end > start:
# #                     jpg = self.bytes_data[start:end+2]
# #                     self.bytes_data = self.bytes_data[end+2:]
# #                     frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
# #                     if frame is None:
# #                         continue

# #                     if self.paused:
# #                         if not self.handle_keyboard_input():
# #                             break
# #                         continue

# #                     results = self.model(frame, verbose=False)[0]
# #                     fall_detected_ids = set()

# #                     if results.keypoints is not None:
# #                         boxes = results.boxes.xyxy.cpu().numpy()
# #                         keypoints = results.keypoints.data.cpu().numpy()

# #                         for i in range(len(boxes)):
# #                             if self.detect_fall(keypoints[i], i):
# #                                 fall_detected_ids.add(i)

# #                         for i, (box, keypoint) in enumerate(zip(boxes, keypoints)):
# #                             color = (0,0,255) if i in fall_detected_ids else (0,255,0)
# #                             x1, y1, x2, y2 = map(int, box)
# #                             cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
# #                             self.draw_skeleton(frame, keypoint, color)
# #                             if i in fall_detected_ids:
# #                                 cv2.putText(frame, "Fall Detected!", (x1, y1-10),
# #                                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

# #                         # 히스테리시스 적용
# #                         if fall_detected_ids:
# #                             self.fall_detected_count += 1
# #                             self.normal_count = 0
# #                         else:
# #                             self.normal_count += 1
# #                             self.fall_detected_count = 0

# #                         # 상태 전환
# #                         if not self.prev_fall_state and self.fall_detected_count >= self.fall_detection_threshold:
# #                             self.publish_status(f"[FallDetected] time={datetime.datetime.now()}, location=PINKY")
# #                             self.prev_fall_state = True
# #                         elif self.prev_fall_state and self.normal_count >= self.recovery_threshold:
# #                             self.publish_status("Normal Driving")
# #                             self.prev_fall_state = False

# #                     cv2.imshow("Fall Detection", frame)
# #                     if not self.handle_keyboard_input():
# #                         break
# #         except Exception as e:
# #             self.get_logger().error(f"오류 발생: {e}")
# #         finally:
# #             cv2.destroyAllWindows()

# # def main(args=None):
# #     rclpy.init(args=args)
# #     node = FallDetectionNode()
# #     node.run()
# #     node.destroy_node()
# #     rclpy.shutdown()

# # if __name__ == "__main__":
# #     main()

# import cv2
# import numpy as np
# from ultralytics import YOLO
# from collections import deque
# import requests
# import datetime
# import rclpy
# from rclpy.node import Node
# from std_msgs.msg import String

# class FallDetectionNode(Node):
#     def __init__(self):
#         super().__init__('fall_detection_node')

#         # ROS 퍼블리셔
#         self.publisher_ = self.create_publisher(String, '/fall_status', 10)

#         # YOLO Pose 모델
#         self.model = YOLO("yolo11n-pose.pt")

#         # 카메라 스트림 URL
#         self.url = 'http://192.168.0.154:5000/camera'
#         self.stream = requests.get(self.url, stream=True)
#         self.bytes_data = b''

#         # 쓰러짐 감지 기록
#         self.fall_history = {}
#         self.prev_fall_state = False  # False=Normal, True=FallDetected
#         self.fall_detected_count = 0
#         self.normal_count = 0
#         self.fall_detection_threshold = 5  # 연속 5프레임
#         self.recovery_threshold = 5        # 연속 5프레임 정상

#         # 윈도우
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

#         if angle < angle_threshold or y_change > y_drop_threshold:
#             return True
#         return False

#     POSE_CONNECTIONS = [
#         (0,1),(1,3),(0,2),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
#         (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)
#     ]

#     def draw_skeleton(self, frame, keypoints, color):
#         for p1_idx, p2_idx in self.POSE_CONNECTIONS:
#             if p1_idx < len(keypoints) and p2_idx < len(keypoints):
#                 p1, p2 = keypoints[p1_idx], keypoints[p2_idx]
#                 if p1[2] > 0.5 and p2[2] > 0.5:
#                     x1, y1, x2, y2 = int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])
#                     cv2.line(frame, (x1, y1), (x2, y2), color, 2)
#                     cv2.circle(frame, (x1, y1), 4, color, -1)
#                     cv2.circle(frame, (x2, y2), 4, color, -1)

#     def publish_status(self, message: str):
#         msg = String()
#         msg.data = message
#         self.publisher_.publish(msg)
#         self.get_logger().info(f"Published: {message}")

#     def run(self):
#         self.get_logger().info("핑키 카메라 쓰러짐 감지 시작. ESC 누르면 종료합니다.")
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

#                         # 프레임 기준 히스테리시스
#                         if fall_detected_ids:
#                             self.fall_detected_count += 1
#                             self.normal_count = 0
#                         else:
#                             self.normal_count += 1
#                             self.fall_detected_count = 0

#                         # 상태 전환
#                         if not self.prev_fall_state and self.fall_detected_count >= self.fall_detection_threshold:
#                             self.publish_status(f"[FallDetected] time={datetime.datetime.now()}, location=PINKY")
#                             self.prev_fall_state = True
#                         elif self.prev_fall_state and self.normal_count >= self.recovery_threshold:
#                             self.publish_status("Normal Driving")
#                             self.prev_fall_state = False

#                     cv2.imshow("Fall Detection", frame)
#                     if not self.handle_keyboard_input():
#                         break
#         except Exception as e:
#             self.get_logger().error(f"오류 발생: {e}")
#         finally:
#             cv2.destroyAllWindows()

# def main(args=None):
#     rclpy.init(args=args)
#     node = FallDetectionNode()
#     node.run()
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == "__main__":
#     main()

# import cv2
# import numpy as np
# from ultralytics import YOLO
# from collections import deque
# import requests
# import datetime
# import rclpy
# from rclpy.node import Node
# from std_msgs.msg import String
# import time

# class FallDetectionNode(Node):
#     def __init__(self):
#         super().__init__('fall_detection_node')

#         # ROS 퍼블리셔
#         self.publisher_ = self.create_publisher(String, '/fall_status', 10)

#         # YOLO Pose 모델
#         self.model = YOLO("yolo11n-pose.pt")

#         # 카메라 스트림 URL
#         self.url = 'http://192.168.0.154:5000/camera'
#         self.stream = requests.get(self.url, stream=True)
#         self.bytes_data = b''

#         # 쓰러짐 감지 기록
#         self.fall_history = {}
#         self.prev_fall_state = False  # False=Normal, True=FallDetected
#         self.fall_detected_count = 0
#         self.normal_count = 0
#         self.fall_detection_threshold = 5  # 연속 5프레임
#         self.recovery_threshold = 5        # 연속 5프레임 정상

#         # 상태 유지 시간 (초)
#         self.STATE_HOLD_TIME = 3.0
#         self.state_start_time = time.time()

#         # 윈도우
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

#     POSE_CONNECTIONS = [
#         (0,1),(1,3),(0,2),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
#         (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)
#     ]

#     def draw_skeleton(self, frame, keypoints, color):
#         for p1_idx, p2_idx in self.POSE_CONNECTIONS:
#             if p1_idx < len(keypoints) and p2_idx < len(keypoints):
#                 p1, p2 = keypoints[p1_idx], keypoints[p2_idx]
#                 if p1[2] > 0.5 and p2[2] > 0.5:
#                     x1, y1, x2, y2 = int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])
#                     cv2.line(frame, (x1, y1), (x2, y2), color, 2)
#                     cv2.circle(frame, (x1, y1), 4, color, -1)
#                     cv2.circle(frame, (x2, y2), 4, color, -1)

#     def publish_status(self, message: str):
#         msg = String()
#         msg.data = message
#         self.publisher_.publish(msg)
#         self.get_logger().info(f"Published: {message}")

#     def run(self):
#         self.get_logger().info("핑키 카메라 쓰러짐 감지 시작. ESC 누르면 종료합니다.")
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

#                         # 프레임 기반 히스테리시스
#                         if fall_detected_ids:
#                             self.fall_detected_count += 1
#                             self.normal_count = 0
#                         else:
#                             self.normal_count += 1
#                             self.fall_detected_count = 0

#                         # 상태 전환
#                         if not self.prev_fall_state:
#                             # Normal -> FallDetected
#                             if self.fall_detected_count >= self.fall_detection_threshold:
#                                 self.prev_fall_state = True
#                                 self.state_start_time = current_time
#                                 self.publish_status(f"[FallDetected] time={datetime.datetime.now()}, location=PINKY")
#                         else:
#                             # FallDetected -> Normal
#                             if (self.normal_count >= self.recovery_threshold and
#                                 (current_time - self.state_start_time) >= self.STATE_HOLD_TIME):
#                                 self.prev_fall_state = False
#                                 self.state_start_time = current_time
#                                 self.publish_status("Normal Driving")

#                     cv2.imshow("Fall Detection", frame)
#                     if not self.handle_keyboard_input():
#                         break
#         except Exception as e:
#             self.get_logger().error(f"오류 발생: {e}")
#         finally:
#             cv2.destroyAllWindows()

# def main(args=None):
#     rclpy.init(args=args)
#     node = FallDetectionNode()
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

class FallDetectionTestNode(Node):
    POSE_CONNECTIONS = [
        (0,1),(1,3),(0,2),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
        (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)
    ]

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
        self.STATE_HOLD_MAX = 0.5  # 사람 잠깐 사라져도 0.5초 유지

        cv2.namedWindow("Fall Detection", flags=cv2.WINDOW_AUTOSIZE)
        self.paused = False

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

    def publish_status(self, message: str):
        msg = String()
        msg.data = message
        self.publisher_.publish(msg)
        self.get_logger().info(f"Published: {message}")

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
                                self.publish_status(f"[FallDetected] time={datetime.datetime.now()}, location=PINKY")
                                self.publish_cmd(stop=True)
                            self.last_state_change_time = current_time
                        else:
                            if self.prev_fall_state and (current_time - self.last_state_change_time) > self.STATE_HOLD_MAX:
                                self.prev_fall_state = False
                                self.publish_status("Normal Driving")
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
