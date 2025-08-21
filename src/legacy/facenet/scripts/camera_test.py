# camera_test.py
import cv2

for i in range(2):  # 보통 0~1 정도만 있으면 충분해
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✅ Camera index {i} is working.")
        ret, frame = cap.read()
        if ret:
            cv2.imshow(f"Camera {i}", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        cap.release()
    else:
        print(f"❌ Camera index {i} is NOT working.")
