from flask import Flask, Response
import threading
import time
import cv2
from pinkylib import Camera
import rclpy
from rclpy.node import Node

app = Flask(__name__)

frame_lock = threading.Lock()
latest_frame = None

class PinkyCameraNode(Node):
    def __init__(self):
        super().__init__('pinky_camera_node')
        self.cam = Camera()
        self.cam.start(width=320, height=240)

        # 프레임 갱신 타이머 & Flask 서버 스레드
        self.timer = self.create_timer(0.033, self.update_frame_callback)
        self.flask_thread = threading.Thread(target=self.start_flask, daemon=True)
        self.flask_thread.start()

        self.get_logger().info("📸 Pinky Camera Node started (Streaming only)")

    def update_frame_callback(self):
        global latest_frame
        frame = self.cam.get_frame()
        with frame_lock:
            latest_frame = frame

    def start_flask(self):
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

    def destroy_node(self):
        self.cam.close()
        super().destroy_node()

@app.route('/camera')
def camera_stream():
    def gen_frames():
        global latest_frame
        while True:
            with frame_lock:
                frame = latest_frame
            if frame is None:
                continue
            _, jpeg = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            time.sleep(0.01)
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def main(args=None):
    rclpy.init(args=args)
    node = PinkyCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
