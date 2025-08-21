# 순찰 + 쓰러짐만 실행

터미널 1에서

1. activate_yolo 입력 (가상환경 접속하는거)

2. Jazzy 입력 (ROS 환경설정하는거)

3. cd pinky_8_ws

4. source ./install/local_setup.bash

5. ros2 run falldown_detector falldown_detect
   

터미널 2에서 

1. activate_yolo 입력 (가상환경 접속하는거)

2. Jazzy 입력 (ROS 환경설정하는거)

3. cd pinky_8_ws

4. source ./install/local_setup.bash

5. ros2 run falldown_detector patrol_falldown_DB

터미널 3에서 

1. ssh pinky@192.168.0.154

2. cd pinky_violet

3. source ./install/local_setup.bash

4. ros2 launch pinky_bringup bringup.launch.xml motor_ratio:=0.9
5. 
(또는 ros2 launch pinky_bringup bringup.launch.xml)

터미널 4에서

1. ssh pinky@192.168.0.154

2. cd pinky_violet

3. source ./install/local_setup.bash

4. ros2 run pinky_camera camera_node




