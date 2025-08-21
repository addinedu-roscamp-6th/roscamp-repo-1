# 순찰 + 얼굴인식만 실행

## 터미널 1에서

activate_yolo 입력 (가상환경 접속하는거)

Jazzy 입력 (ROS 환경설정하는거)

cd pinky_8_ws

source ./install/local_setup.bash

ros2 run face_detector face_detect

## 터미널 2에서

activate_yolo 입력 (가상환경 접속하는거)

Jazzy 입력 (ROS 환경설정하는거)

cd pinky_8_ws

source ./install/local_setup.bash

ros2 run face_detector patrol_face_DB

## 터미널 3에서

ssh pinky@192.168.0.154

cd pinky_violet

source ./install/local_setup.bash

ros2 launch pinky_bringup bringup.launch.xml motor_ratio:=0.9

(또는 ros2 launch pinky_bringup bringup.launch.xml)

## 터미널 4에서

ssh pinky@192.168.0.154

cd pinky_violet

source ./install/local_setup.bash

ros2 run pinky_camera camera_node
