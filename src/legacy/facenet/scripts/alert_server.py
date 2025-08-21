from flask import Flask, request
import os
from datetime import datetime
import threading
import tkinter as tk
from PIL import Image, ImageTk
from playsound import playsound

# ========== 설정 ==========
SAVE_DIR = "received"
ALERT_SOUND = "alert.wav"
POPUP_DURATION = 30  # 초

os.makedirs(SAVE_DIR, exist_ok=True)
app = Flask(__name__)


def show_popup(image_path):
    def popup():
        window = tk.Tk()
        window.title("⚠️ 외부인 감지")
        window.geometry("400x400")

        # 이미지 표시
        img = Image.open(image_path)
        img = img.resize((360, 360))
        photo = ImageTk.PhotoImage(img)
        label = tk.Label(window, image=photo)
        label.image = photo
        label.pack(pady=10)

        # 30초 후 닫기
        def close():
            window.destroy()

        window.after(POPUP_DURATION * 1000, close)
        window.mainloop()

    # 알림음 + 팝업 병렬 실행
    threading.Thread(target=playsound, args=(ALERT_SOUND,), daemon=True).start()
    threading.Thread(target=popup, daemon=True).start()


@app.route("/receive_unknown", methods=["POST"])
def receive_unknown():
    image = request.files.get("image")
    device = request.form.get("device", "unknown_device")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not image:
        return "No image provided", 400

    filename = f"{device}_{timestamp}.jpg"
    path = os.path.join(SAVE_DIR, filename)
    image.save(path)

    print(f"[RECEIVED] Unknown 얼굴 저장 완료: {path}")

    show_popup(path)

    return "Received", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
