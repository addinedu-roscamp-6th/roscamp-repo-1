# import cv2
# from ultralytics import YOLO
# import pandas as pd
# import os

# #=================================================================
# # 초기값 설정
# #=================================================================

# # 실행 경로 설정 
# script_dir = os.path.dirname(os.path.abspath(__file__))
# os.chdir(script_dir)

# # 모델 경로 설정
# Models_dir = os.path.abspath(os.path.join(script_dir, "../00_Models"))
# model_file = os.path.join(Models_dir, "yolov8n-pose.pt")

# # 저장 경로 설정
# save_dir = './pose_img/person/'
# keypoint_path = f'{save_dir}keypoints.csv'

# # 디렉토리 없으면 생성
# os.makedirs(save_dir, exist_ok=True)

# # YOLOv8 포즈 추정 모델 불러오기
# model = YOLO(model_file)

# # 카메라 또는 비디오 캡처 객체 생성 (0은 기본 웹캠)
# cap = cv2.VideoCapture("http://192.168.0.154:5000/camera")

# # 비디오의 총 프레임 수와 초당 프레임 수 가져오기
# frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
# fps = cap.get(cv2.CAP_PROP_FPS)

# # 비디오의 총 시간 계산 (초 단위)
# seconds = round(frames / fps) if fps != 0 else 0

# # 처리할 프레임 수 설정 (필요에 따라 변경 가능)
# frame_total = 500  
# i = 0
# a = 0

# # 모든 데이터를 저장할 리스트
# all_data = []

# # 비디오가 열려 있는 동안 프레임 처리
# while cap.isOpened():
#     # 처리할 특정 시간(밀리초 단위)에 해당하는 프레임으로 이동
#     cap.set(cv2.CAP_PROP_POS_MSEC, (i * ((seconds / frame_total) * 1000)))

#     # 현재 프레임 읽기
#     flag, frame = cap.read()

#     # 더 이상 프레임이 없으면 종료
#     if not flag or i >= 600:
#         break

#     # YOLOv8 모델로 프레임에서 객체 감지 수행
#     results = model(frame, verbose=False)

#     # ✅ 시각화 추가
#     annotated_frame = results[0].plot()
#     cv2.imshow('YOLO Pose Detection', annotated_frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
    
#     # 감지된 결과에 대해 반복 처리
#     for r in results:
#         bound_box = r.boxes.xyxy  # 경계 상자 좌표
#         conf = r.boxes.conf.tolist()  # 신뢰도 점수
#         keypoints = r.keypoints.xyn.tolist()  # 정규화된 키포인트 좌표

#         for index, box in enumerate(bound_box):
#             # 신뢰도 0.75 이상인 경우만 처리
#             if conf[index] > 0.4:
#                 x1, y1, x2, y2 = map(int, box.tolist())
#                 pict = frame[y1:y2, x1:x2]
#                 output_path = f'{save_dir}person_{a}.jpg'

#                 # 키포인트 저장용 딕셔너리 생성
#                 data = {'image_name': f'person_{a}.jpg'}
#                 for j, (x, y) in enumerate(keypoints[index]):
#                     data[f'x{j}'] = x
#                     data[f'y{j}'] = y

#                 # 이미지와 키포인트 저장
#                 all_data.append(data)
#                 cv2.imwrite(output_path, pict)
#                 a += 1

#     i += 1

# # 결과 출력
# print(f'프레임 처리 수: {i}, 저장된 이미지 수: {a}')

# # 리소스 해제
# cap.release()
# cv2.destroyAllWindows()

# # 모든 데이터를 DataFrame으로 변환 후 저장
# df = pd.DataFrame(all_data)
# df.to_csv(keypoint_path, index=False)

import cv2
from ultralytics import YOLO
import pandas as pd
import os

# 초기값 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
Models_dir = os.path.abspath(os.path.join(script_dir, "../00_Models"))
model_file = os.path.join(Models_dir, "yolov8n-pose.pt")
save_dir = './pose_img/person/'
keypoint_path = f'{save_dir}keypoints.csv'
os.makedirs(save_dir, exist_ok=True)

model = YOLO(model_file)
cap = cv2.VideoCapture("http://192.168.0.154:5000/camera")

a = 0
all_data = []

try:
    while cap.isOpened():
        flag, frame = cap.read()
        if not flag:
            break

        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()
        cv2.imshow('YOLO Pose Detection', annotated_frame)

        key = cv2.waitKey(1) & 0xFF

        # q 누르면 종료
        if key == ord('q'):
            break

        # 엔터 누르면 저장
        if key == 13:
            for r in results:
                bound_box = r.boxes.xyxy
                conf = r.boxes.conf.tolist()
                keypoints = r.keypoints.xyn.tolist()

                for index, box in enumerate(bound_box):
                    if conf[index] > 0.2:
                        x1, y1, x2, y2 = map(int, box.tolist())
                        pict = frame[y1:y2, x1:x2]
                        output_path = f'{save_dir}person_{a}.jpg'

                        data = {'image_name': f'person_{a}.jpg'}
                        for j, (x, y) in enumerate(keypoints[index]):
                            data[f'x{j}'] = x
                            data[f'y{j}'] = y

                        all_data.append(data)
                        cv2.imwrite(output_path, pict)
                        print(f'저장됨 → {output_path}')
                        a += 1

except KeyboardInterrupt:
    print("\n강제 종료됨")

finally:
    # 항상 실행: CSV 저장 + 자원 해제
    cap.release()
    cv2.destroyAllWindows()
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(keypoint_path, index=False)
        print(f'키포인트 CSV 저장 완료: {keypoint_path}')
    print(f'총 저장된 이미지 수: {a}')
