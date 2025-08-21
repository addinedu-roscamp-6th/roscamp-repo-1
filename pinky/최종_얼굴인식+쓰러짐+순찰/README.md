# 얼굴인식 + 쓰러짐 + 순찰 (이렇게 돌리게 되면 테스트할 때는 카메라 딜레이가 있어서 쓰러짐 감지나 얼굴인식을 잘 하지 못했어요 영상 찍을때 따로따로 하는 코드로 찍어도 돼요)


## 터미널 1에서

1. activate_yolo 입력 (가상환경 접속하는거)

2. Jazzy 입력 (ROS 환경설정하는거)

3. cd pinky_8_ws

4. source ./install/local_setup.bash

5. ros2 run face_detector face_detect


## 터미널 2에서

activate_yolo 입력 (가상환경 접속하는거)

Jazzy 입력 (ROS 환경설정하는거)

cd pinky_8_ws

source ./install/local_setup.bash

ros2 run falldown_detector falldown_detect


## 터미널 3에서

1. activate_yolo 입력 (가상환경 접속하는거)

2. Jazzy 입력 (ROS 환경설정하는거)

3. cd pinky_8_ws

4. source ./install/local_setup.bash

5. ros2 run pinky_guardian face_falldown_patrol


## 터미널 4에서

ssh pinky@192.168.0.154

cd pinky_violet

source ./install/local_setup.bash

ros2 launch pinky_bringup bringup.launch.xml motor_ratio:=0.9

(또는 ros2 launch pinky_bringup bringup.launch.xml)


## 터미널 5에서

ssh pinky@192.168.0.154

cd pinky_violet

source ./install/local_setup.bash

ros2 run pinky_camera camera_node
