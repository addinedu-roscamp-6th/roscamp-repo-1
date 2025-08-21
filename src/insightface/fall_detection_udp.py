# import cv2
# import socket
# import numpy as np
# import pandas as pd
# import xgboost as xgb
# from ultralytics import YOLO
# import datetime
# import os

# class FallDetectionUDP:
#     def __init__(self):
#         # 모델 경로 설정
#         script_dir = os.path.dirname(os.path.abspath(__file__))
#         models_dir = os.path.abspath(os.path.join(script_dir, "../00_Models"))
#         yolo_model_path = os.path.join(models_dir, "yolov8n-pose.pt")
#         xgb_model_path = os.path.join(models_dir, "model_weights.xgb")

#         # 모델 로드
#         self.model_yolo = YOLO(yolo_model_path)
#         self.model_xgb = xgb.XGBClassifier()
#         self.model_xgb.load_model(xgb_model_path)

#         # UDP 수신 설정
#         self.udp_ip = "0.0.0.0"
#         self.udp_port = 5050
#         self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#         self.sock.bind((self.udp_ip, self.udp_port))

#         # 상태 변수
#         self.fall_detected_count = 0
#         self.fall_detection_threshold = 5
#         self.prev_fall_state = None

#     def send_status_to_guardian_p(self, message):
#         HOST = '192.168.0.131'  # Guardian_P 서버 IP
#         PORT = 8888
#         try:
#             with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
#                 sock.connect((HOST, PORT))
#                 sock.sendall(message.encode('utf-8'))
#                 print(f"[전송됨] Guardian_P: {message}")
#         except Exception as e:
#             print("[오류] Guardian_P 전송 실패:", e)

#     def run(self):
#         print("🔍 쓰러짐 감지 시작 (UDP 수신). ESC 누르면 종료됩니다.")
#         try:
#             while True:
#                 data, _ = self.sock.recvfrom(65536)
#                 frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
#                 if frame is None:
#                     continue

#                 results = self.model_yolo(frame, verbose=False)
#                 annotated_frame = results[0].plot(boxes=False)
#                 fall_detected = False

#                 for r in results:
#                     bound_box = r.boxes.xyxy
#                     conf = r.boxes.conf.tolist()
#                     keypoints = r.keypoints.xyn.tolist()
#                     for idx, box in enumerate(bound_box):
#                         if conf[idx] > 0.75:
#                             x1, y1, x2, y2 = map(int, box.tolist())
#                             data_dict = {}
#                             for j in range(len(keypoints[idx])):
#                                 data_dict[f'x{j}'] = keypoints[idx][j][0]
#                                 data_dict[f'y{j}'] = keypoints[idx][j][1]
#                             df = pd.DataFrame([data_dict])
#                             cut = self.model_xgb.predict(df)
#                             binary_pred = (cut > 0.5).astype(int)
#                             if binary_pred[0] == 0:  # 쓰러짐 감지
#                                 fall_detected = True
#                                 cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#                                 cv2.putText(annotated_frame, 'FallDown', (x1, y1 - 10),
#                                             cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)

#                 if fall_detected:
#                     self.fall_detected_count += 1
#                 else:
#                     self.fall_detected_count = 0

#                 # 상태 전이 감지
#                 current_state = fall_detected and (self.fall_detected_count >= self.fall_detection_threshold)
#                 if self.prev_fall_state != current_state:
#                     if current_state:
#                         msg = f"[FallDetected] time={datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, location=PINKY"
#                     else:
#                         msg = "Normal Driving"
#                     self.send_status_to_guardian_p(msg)
#                     self.prev_fall_state = current_state

#                 cv2.imshow("FallDown Detection (UDP)", annotated_frame)
#                 if cv2.waitKey(1) & 0xFF == 27:
#                     print("👋 종료합니다.")
#                     break

#         except Exception as e:
#             print("[오류 발생]", e)
#         finally:
#             cv2.destroyAllWindows()

# if __name__ == "__main__":
#     fd = FallDetectionUDP()
#     fd.run()



# from ultralytics import YOLO
# import cv2
# import xgboost as xgb
# import pandas as pd
# import numpy as np
# import requests
# import os
# import socket
# import datetime
# class FallDetection:
#     def __init__(self):
#         # 모델 경로 설정
#         script_dir = os.path.dirname(os.path.abspath(__file__))
#         os.chdir(script_dir)
#         models_dir = os.path.abspath(os.path.join(script_dir, "../00_Models"))
#         # 모델 파일 경로
#         yolo_model_path = os.path.join(models_dir, "yolov8n-pose.pt")
#         xgb_model_path = os.path.join(models_dir, "model_weights.xgb")
#         # 모델 로드
#         self.model_yolo = YOLO(yolo_model_path)
#         self.model_xgb = xgb.XGBClassifier()
#         self.model_xgb.load_model(xgb_model_path)
#         # 핑키 카메라 스트림 URL
#         self.url = 'http://192.168.0.97:5000/camera'
#         self.stream = requests.get(self.url, stream=True)
#         self.bytes_data = b''
#         # 쓰러짐 감지 상태
#         self.fall_detected_count = 0
#         self.fall_detection_threshold = 5  # 연속 감지 임계값
#         self.prev_fall_state = None  # 이전 상태 저장용
#     def send_status_to_guardian_p(self, message):
#         HOST = '192.168.0.131'  # Guardian_P 서버 IP
#         PORT = 8888
#         try:
#             with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
#                 sock.connect((HOST, PORT))
#                 sock.sendall(message.encode('utf-8'))
#                 print(f"Guardian_P에 메시지 전송 완료: {message}")
#         except Exception as e:
#             print("Guardian_P TCP 전송 실패:", e)
#     def run(self):
#         print("핑키 카메라 쓰러짐 감지 시작. ESC 누르면 종료합니다.")
#         try:
#             for chunk in self.stream.iter_content(chunk_size=1024):
#                 self.bytes_data += chunk
#                 start = self.bytes_data.find(b'\xff\xd8')
#                 end = self.bytes_data.find(b'\xff\xd9')
#                 if start != -1 and end != -1 and end > start:
#                     jpg = self.bytes_data[start:end + 2]
#                     self.bytes_data = self.bytes_data[end + 2:]
#                     frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
#                     if frame is None:
#                         continue
#                     results = self.model_yolo(frame, verbose=False)
#                     annotated_frame = results[0].plot(boxes=False)
#                     fall_detected = False
#                     for r in results:
#                         bound_box = r.boxes.xyxy
#                         conf = r.boxes.conf.tolist()
#                         keypoints = r.keypoints.xyn.tolist()
#                         for idx, box in enumerate(bound_box):
#                             if conf[idx] > 0.75:
#                                 x1, y1, x2, y2 = map(int, box.tolist())
#                                 data = {}
#                                 for j in range(len(keypoints[idx])):
#                                     data[f'x{j}'] = keypoints[idx][j][0]
#                                     data[f'y{j}'] = keypoints[idx][j][1]
#                                 df = pd.DataFrame([data])
#                                 cut = self.model_xgb.predict(df)
#                                 binary_pred = (cut > 0.5).astype(int)
#                                 if binary_pred[0] == 0:  # 쓰러짐 감지
#                                     fall_detected = True
#                                     cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#                                     cv2.putText(annotated_frame, 'FallDown', (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)
#                     if fall_detected:
#                         self.fall_detected_count += 1
#                     else:
#                         self.fall_detected_count = 0
#                     # 임계값 넘으면 쓰러짐 상태로 판단
#                     current_state = fall_detected and (self.fall_detected_count >= self.fall_detection_threshold)
#                     # 상태 변경 시에만 메시지 전송
#                     if self.prev_fall_state != current_state:
#                         if current_state:
#                             msg = f"[FallDetected] time={datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, location=PINKY"
#                         else:
#                             msg = "Normal Driving"
#                         self.send_status_to_guardian_p(msg)
#                         self.prev_fall_state = current_state
#                     cv2.imshow("FallDown Detection from Pinky", annotated_frame)
#                     if cv2.waitKey(1) & 0xFF == 27:
#                         print("종료합니다.")
#                         break
#         except Exception as e:
#             print(f"오류 발생: {e}")
#         finally:
#             cv2.destroyAllWindows()
# if __name__ == "__main__":
#     fd = FallDetection()
#     fd.run()



# fall_detection_udp_test.py

# from ultralytics import YOLO
# import cv2
# import xgboost as xgb
# import pandas as pd
# import numpy as np
# import requests
# import os
# import socket
# import datetime
# import traceback

# class FallDetection:
#     def __init__(self):
#         # 모델 경로 설정
#         script_dir = os.path.dirname(os.path.abspath(__file__))
#         os.chdir(script_dir)
#         models_dir = os.path.abspath(os.path.join(script_dir, "../00_Models"))

#         # 모델 파일 경로
#         yolo_model_path = os.path.join(models_dir, "yolov8n-pose.pt")
#         xgb_model_path = os.path.join(models_dir, "model_weights.xgb")

#         # 모델 로드
#         self.model_yolo = YOLO(yolo_model_path)
#         self.model_xgb = xgb.XGBClassifier()
#         self.model_xgb.load_model(xgb_model_path)

#         # 핑키 카메라 스트림 URL
#         self.url = 'http://192.168.0.97:5000/camera'
#         print(f"[INFO] 핑키 스트림 연결 시도 중: {self.url}")
#         self.stream = requests.get(self.url, stream=True)
#         self.bytes_data = b''

#         # 쓰러짐 감지 상태
#         self.fall_detected_count = 0
#         self.fall_detection_threshold = 5
#         self.prev_fall_state = None

#     def send_status_to_guardian_p(self, message):
#         HOST = '192.168.0.131'
#         PORT = 8888
#         try:
#             with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
#                 sock.connect((HOST, PORT))
#                 sock.sendall(message.encode('utf-8'))
#                 print(f"[전송 완료] Guardian_P에 메시지 전송: {message}")
#         except Exception as e:
#             print("[전송 실패] Guardian_P TCP:", e)

#     def run(self):
#         print("📹 핑키 카메라 쓰러짐 감지 시작. ESC 누르면 종료합니다.")
#         try:
#             for chunk in self.stream.iter_content(chunk_size=1024):
#                 self.bytes_data += chunk
#                 start = self.bytes_data.find(b'\xff\xd8')
#                 end = self.bytes_data.find(b'\xff\xd9')

#                 if start != -1 and end != -1 and end > start:
#                     jpg = self.bytes_data[start:end + 2]
#                     self.bytes_data = self.bytes_data[end + 2:]
#                     frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

#                     if frame is None:
#                         print("[경고] 프레임 디코딩 실패 (frame is None)")
#                         continue

#                     print(f"[DEBUG] 프레임 수신 완료 - shape: {frame.shape}")
#                     results = self.model_yolo(frame, verbose=False)
#                     annotated_frame = results[0].plot(boxes=False)

#                     fall_detected = False
#                     for r in results:
#                         bound_box = r.boxes.xyxy
#                         conf = r.boxes.conf.tolist()
#                         keypoints = r.keypoints.xyn.tolist()

#                         for idx, box in enumerate(bound_box):
#                             if conf[idx] > 0.75:
#                                 x1, y1, x2, y2 = map(int, box.tolist())
#                                 data = {}
#                                 for j in range(len(keypoints[idx])):
#                                     data[f'x{j}'] = keypoints[idx][j][0]
#                                     data[f'y{j}'] = keypoints[idx][j][1]
#                                 df = pd.DataFrame([data])
#                                 cut = self.model_xgb.predict(df)
#                                 binary_pred = (cut > 0.5).astype(int)
#                                 if binary_pred[0] == 0:
#                                     fall_detected = True
#                                     cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#                                     cv2.putText(annotated_frame, 'FallDown', (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)

#                     if fall_detected:
#                         self.fall_detected_count += 1
#                     else:
#                         self.fall_detected_count = 0

#                     current_state = fall_detected and (self.fall_detected_count >= self.fall_detection_threshold)
#                     if self.prev_fall_state != current_state:
#                         if current_state:
#                             msg = f"[FallDetected] time={datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, location=PINKY"
#                         else:
#                             msg = "Normal Driving"
#                         self.send_status_to_guardian_p(msg)
#                         self.prev_fall_state = current_state

#                     cv2.imshow("FallDown Detection from Pinky", annotated_frame)
#                     if cv2.waitKey(1) & 0xFF == 27:
#                         print("ESC 입력 - 프로그램 종료")
#                         break

#         except Exception as e:
#             print("[오류 발생]")
#             traceback.print_exc()

#         finally:
#             try:
#                 cv2.destroyAllWindows()
#             except Exception as e:
#                 print("cv2.destroyAllWindows() 실패:", e)
#                 print("💡 해결 방법: Ubuntu 환경에서는 'sudo apt install libgtk2.0-dev pkg-config' 후 OpenCV 재설치 필요")

# if __name__ == "__main__":
#     fd = FallDetection()
#     fd.run()



from ultralytics import YOLO
import cv2
import xgboost as xgb
import pandas as pd
import numpy as np
import requests
import os
import socket
import datetime
from insightface.app import FaceAnalysis
from clean_face_db import load_face_db, find_best_match

class FallDetectionWithFace:
    def __init__(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        models_dir = os.path.abspath(os.path.join(script_dir, "../00_Models"))

        self.model_yolo = YOLO(os.path.join(models_dir, "yolov8n-pose.pt"))
        self.model_xgb = xgb.XGBClassifier()
        self.model_xgb.load_model(os.path.join(models_dir, "model_weights.xgb"))

        self.face_db = load_face_db()
        self.face_app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.face_app.prepare(ctx_id=0)

        self.url = 'http://192.168.0.97:5000/camera'
        self.stream = requests.get(self.url, stream=True)
        self.bytes_data = b''

        self.fall_detected_count = 0
        self.fall_detection_threshold = 5
        self.prev_fall_state = None

    def send_status_to_guardian_p(self, message):
        HOST = '192.168.0.131'
        PORT = 8888
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((HOST, PORT))
                sock.sendall(message.encode('utf-8'))
                print(f"Guardian_P에 메시지 전송 완료: {message}")
        except Exception as e:
            print("Guardian_P TCP 전송 실패:", e)

    def recognize_faces(self, frame):
        results = self.face_app.get(frame)
        for face in results[:2]:  # 최대 2명
            x1, y1, x2, y2 = map(int, face.bbox)
            name, score = find_best_match(face.embedding, self.face_db)
            label = f"{name} ({score:.2f})"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def run(self):
        print("👀 핑키 쓰러짐 + 얼굴 인식 시작 (ESC로 종료)")
        try:
            for chunk in self.stream.iter_content(chunk_size=1024):
                self.bytes_data += chunk
                start = self.bytes_data.find(b'\xff\xd8')
                end = self.bytes_data.find(b'\xff\xd9')
                if start != -1 and end != -1 and end > start:
                    jpg = self.bytes_data[start:end + 2]
                    self.bytes_data = self.bytes_data[end + 2:]
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        continue

                    # YOLO + XGBoost 포즈 추정
                    results = self.model_yolo(frame, verbose=False)
                    annotated_frame = results[0].plot(boxes=False)
                    fall_detected = False

                    for r in results:
                        bound_box = r.boxes.xyxy
                        conf = r.boxes.conf.tolist()
                        keypoints = r.keypoints.xyn.tolist()
                        for idx, box in enumerate(bound_box):
                            if conf[idx] > 0.75:
                                x1, y1, x2, y2 = map(int, box.tolist())
                                data = {f'x{j}': keypoints[idx][j][0] for j in range(len(keypoints[idx]))}
                                data.update({f'y{j}': keypoints[idx][j][1] for j in range(len(keypoints[idx]))})
                                df = pd.DataFrame([data])
                                cut = self.model_xgb.predict(df)
                                binary_pred = (cut > 0.5).astype(int)
                                if binary_pred[0] == 0:
                                    fall_detected = True
                                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                                    cv2.putText(annotated_frame, 'FallDown', (x1, y1 - 10),
                                                cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 0, 255), 2)

                    if fall_detected:
                        self.fall_detected_count += 1
                    else:
                        self.fall_detected_count = 0

                    current_state = fall_detected and (self.fall_detected_count >= self.fall_detection_threshold)
                    if self.prev_fall_state != current_state:
                        msg = f"[FallDetected] time={datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, location=PINKY" if current_state else "Normal Driving"
                        self.send_status_to_guardian_p(msg)
                        self.prev_fall_state = current_state

                    # 얼굴 인식 호출
                    self.recognize_faces(annotated_frame)

                    # 화면 출력
                    cv2.imshow("Fall + Face Detection", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
        except Exception as e:
            print(f"[ERROR] 오류 발생: {e}")
        finally:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    FallDetectionWithFace().run()
