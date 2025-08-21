import os
import cv2
import pickle
import numpy as np
import requests
import threading
import time
import mysql.connector
import datetime
from insightface.app import FaceAnalysis

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# ==================== ROS 2 설정 ====================
NODE_NAME = "face_recognition_node"
RECOGNITION_TOPIC = "/recognized_person"
RECOGNITION_RATE_HZ = 5.0  # 5 Hz로 얼굴 확인

# ==================== 기존 스크립트 설정 ====================
STREAM_URL = "http://192.168.0.154:5000/camera"
DB_CONF = {
    'host': '192.168.0.218',
    'port': 3306,
    'user': 'guardian',
    'password': 'guardian@123',
    'database': 'guardian'
}
FACE_DIR = "faces"

# ==================== InsightFace 모델 ====================
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"],
    allowed_modules=['detection', 'recognition']
)
app.prepare(ctx_id=0, det_size=(320, 320))

# ==================== 데이터 로드 ====================
def load_face_database_from_db():
    face_db = {}
    try:
        conn = mysql.connector.connect(**DB_CONF)
        cur = conn.cursor()
        cur.execute("SELECT name, embedding FROM faces")
        rows = cur.fetchall()
        for name, embedding_bytes in rows:
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            face_db[name] = embedding
        cur.close()
        conn.close()
        print(f"[INFO] DB에서 {len(face_db)}명 임베딩 로드 완료. ")
        return face_db
    except Exception as e:
        print(f"[ERROR] DB에서 얼굴 데이터 로드 실패: {e} ")
        return {}

# ==================== MJPEG 리더 클래스 ====================
class MJPEGLatestReader:
    def __init__(self, url, timeout=10):
        self.url = url
        self.timeout = timeout
        self.buf = bytearray()
        self.lock = threading.Lock()
        self.latest_jpg = None
        self.latest_seq = 0
        self.stop_flag = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_flag = True
        self.thread.join(timeout=1.0)

    def _run(self):
        try:
            with requests.get(self.url, stream=True, timeout=self.timeout) as r:
                r.raise_for_status()
                it = r.iter_content(chunk_size=1024)
                for chunk in it:
                    if self.stop_flag:
                        break
                    if not chunk:
                        continue
                    self.buf.extend(chunk)

                    if len(self.buf) > 262144:
                        last_soi = self.buf.rfind(b'\xff\xd8')
                        if last_soi != -1:
                            self.buf = bytearray(self.buf[last_soi:])
                        else:
                            self.buf = bytearray()

                    last_eoi = self.buf.rfind(b'\xff\xd9')
                    if last_eoi == -1:
                        continue
                    last_soi_before_eoi = self.buf.rfind(b'\xff\xd8', 0, last_eoi)
                    if last_soi_before_eoi == -1:
                        self.buf = bytearray(self.buf[last_eoi+2:])
                        continue
                    jpg = bytes(self.buf[last_soi_before_eoi:last_eoi+2])
                    with self.lock:
                        self.latest_jpg = jpg
                        self.latest_seq += 1
                    self.buf = bytearray(self.buf[last_eoi+2:])
        except Exception as e:
            print(f"[ERROR] MJPEG reader: {e} ")

    def get_latest(self):
        with self.lock:
            return (self.latest_seq, self.latest_jpg)

# ==================== ROS 2 노드 클래스 ====================
class FaceRecognitionNode(Node):
    def __init__(self):
        super().__init__(NODE_NAME)
        self.get_logger().info(f"{NODE_NAME} 시작. 얼굴 인식 대기 중... ")

        # DB 연결
        self.db_conn = self.connect_to_db()

        self.publisher_ = self.create_publisher(String, RECOGNITION_TOPIC, 10)
        self.reader = MJPEGLatestReader(STREAM_URL)
        self.reader.start()

        self.known_faces = load_face_database_from_db()
        self.threshold = 0.5 # 인식 임계값 (작을수록 엄격)
        self.last_recognized_name = None

        self.last_seq_used = -1
        self.timer_ = self.create_timer(1.0 / RECOGNITION_RATE_HZ, self.timer_callback)

    def connect_to_db(self):
        """데이터베이스에 연결하고 연결 객체를 반환합니다."""
        try:
            conn = mysql.connector.connect(**DB_CONF)
            self.get_logger().info("데이터베이스 연결 성공 ")
            return conn
        except mysql.connector.Error as err:
            self.get_logger().error(f"DB 연결 오류: {err} ")
            return None

    def log_recognized_face(self, name: str):
        """인식된 얼굴을 DB에 기록합니다."""
        if not self.db_conn or not self.db_conn.is_connected():
            self.get_logger().error("DB 연결이 없어 데이터를 저장할 수 없습니다. ")
            return

        cursor = self.db_conn.cursor()
        sql = "INSERT INTO face_logs (name, recognized_time) VALUES (%s, %s)"
        data = (name, datetime.datetime.now())

        try:
            cursor.execute(sql, data)
            self.db_conn.commit()
            self.get_logger().info(f"인식 기록 DB 저장 성공: {name} ")
        except mysql.connector.Error as err:
            self.get_logger().error(f"DB 데이터 삽입 오류: {err} ")
        finally:
            cursor.close()

    def timer_callback(self):
        seq, jpg = self.reader.get_latest()
        if jpg is None or seq == self.last_seq_used:
            return

        self.last_seq_used = seq
        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            return

        faces = app.get(frame)
        recognized_name = "None"
        
        if faces:
            for face in faces:
                embedding = face.embedding.astype(np.float32)
                
                best_match_name = "Unknown"
                min_distance = -1.0
                
                for name, known_emb in self.known_faces.items():
                    distance = np.dot(embedding, known_emb)
                    if distance > min_distance:
                        min_distance = distance
                        best_match_name = name
                
                if min_distance > self.threshold:
                    recognized_name = best_match_name
                else:
                    recognized_name = "Unknown"

        # 인식된 이름이 이전과 다를 때만 토픽을 발행
        if recognized_name != self.last_recognized_name:
            
            # 인식 시점의 정확한 시간을 캡처
            recognition_time = datetime.datetime.now()
            
            # 메시지 생성 및 발행
            msg = String()
            msg.data = f"{recognized_name},{recognition_time}"
            self.publisher_.publish(msg)
            self.get_logger().info(f"인식 결과 발행: {msg.data}")
            
            # 이 부분이 DB에 로그를 기록하던 코드입니다.
            # 이젠 DB에 저장하지 않으므로, 이 코드를 삭제하거나 주석 처리합니다.
            # if recognized_name != "None":
            #     self.log_recognized_face(recognized_name)
            
            # 마지막으로 인식된 이름을 업데이트
            self.last_recognized_name = recognized_name

        # 시각화 (디버깅용)
        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, recognized_name, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        
        cv2.imshow("Recognition Window", frame)
        cv2.waitKey(1)

    def destroy_node(self):
        self.reader.stop()
        cv2.destroyAllWindows()
        if self.db_conn and self.db_conn.is_connected():
            self.db_conn.close()
            self.get_logger().info("데이터베이스 연결 해제")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = FaceRecognitionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()