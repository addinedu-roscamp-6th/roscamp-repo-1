# 쓰러짐 + 순찰 / 얼굴인식 + 순찰
 
 mkdir pinky_8_ws

 cd pinky_8_ws






pinky_8_ws 이동 후 올린 src 폴더 압축 풀기

pinky_8_ws 안에서 빌드 colcon build

pinky_8_ws 안에서 빌드 후 source ./install/local_setup.bash



# (쓰러짐 사람 감지 + 순찰)

## 터미널 1 PC
pinky_8_ws 안에서

source ./install/local_setup.bash

ros2 run falldown_detector patrol_falldown_DB



## 터미널 2 PC
pinky_8_ws 안에서

source ./install/local_setup.bash

ros2 run falldown_detector falldown_detect



## 터미널 3 Pinky

cd pinky_violet

source ./install/local_setup.bash

ros2 launch pinky_bringup bringup.launch.xml 



## 터미널 4 Pinky

cd pinky_violet

source ./install/local_setup.bash

ros2 run pinky_camera camera_node





# (얼굴 인식 + 순찰)

## 터미널 1 

source ./install/local_setup.bash

ros2 run face_detector patrol_face_DB



## 터미널 2

source ./install/local_setup.bash

ros2 run face_detector face_detect



## 터미널 3 Pinky

cd pinky_violet

source ./install/local_setup.bash

ros2 launch pinky_bringup bringup.launch.xml 



## 터미널 4 Pinky

cd pinky_violet

source ./install/local_setup.bash

ros2 run pinky_camera camera_node





# 파일 트리 구조 

src

│   ├── face_detector

│   │   ├── face_detector

│   │   ├── package.xml

│   │   ├── resource

│   │   ├── setup.cfg

│   │   ├── setup.py

│   │   └── test

│   └── falldown_detector

│       ├── falldown_detector

│       ├── package.xml

│       ├── resource

│       ├── setup.cfg

│       ├── setup.py

│       └── test




