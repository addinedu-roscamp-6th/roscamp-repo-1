# - 순찰만 돌리는 코드

## 터미널 1 PC

순찰만 하는 코드.py => pc에서 실행

## 터미널 2 Pinky

cd pinky_violet

source ./install/local_setup.bash

ros2 launch pinky_bringup bringup.launch.xml 



# 쓰러짐 + 순찰 / 얼굴인식 + 순찰
 
 mkdir pinky_8_ws

 cd pinky_8_ws






pinky_8_ws 이동 후 올린 src 폴더 압축 풀기

pinky_8_ws 안에서 빌드 colcon build

pinky_8_ws 안에서 빌드 후 source ./install/local_setup.bash



# - (쓰러짐 사람 감지 + 순찰)

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





# - (얼굴 인식 + 순찰)

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

├── src

│   ├── face_detector

│   │   ├── face_detector

│   │   │   ├── face_detect.py

│   │   │   ├── __init__.py

│   │   │   └── patrol_face_DB.py

│   │   ├── package.xml

│   │   ├── resource

│   │   │   └── face_detector

│   │   ├── setup.cfg

│   │   ├── setup.py

│   │   └── test

│   │       ├── test_copyright.py

│   │       ├── test_flake8.py

│   │       └── test_pep257.py

│   └── falldown_detector

│       ├── falldown_detector

│       │   ├── falldown_detect.py

│       │   ├── __init__.py

│       │   └── patrol_falldown_DB.py

│       ├── package.xml

│       ├── resource

│       │   └── falldown_detector

│       ├── setup.cfg

│       ├── setup.py

│       └── test

│           ├── test_copyright.py

│           ├── test_flake8.py

│           └── test_pep257.py




# - 순찰 + 쓰러짐 + 얼굴인식 합친 코드를 돌리고 싶으면

## 터미널 1 PC

falldown_detect.py

## 터미널 2 PC

face_detect.py

## 터미널 3 PC

순찰 + 얼굴인식 + 쓰러짐.py

## 터미널 4 Pinky

cd pinky_violet

source ./install/local_setup.bash

ros2 launch pinky_bringup bringup.launch.xml 

## 터미널 5 Pinky

cd pinky_violet

source ./install/local_setup.bash

ros2 run pinky_camera camera_node

# - 키보드 1,2,3 눌러서 요청을 주면 그 좌표로 이동 후 요청이 끝나면 순찰함

## 터미널 1 PC

키보드 1,2,3 픽업드랍 + 순찰(최종실행PC).py

## 터미널 2 Pinky

cd pinky_violet

source ./install/local_setup.bash

ros2 launch pinky_bringup bringup.launch.xml 
