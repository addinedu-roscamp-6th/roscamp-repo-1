# pinky + jecobot + pickup_web 코드


터미널 1 켜서 

1. activate_yolo 입력 (가상환경 접속하는거)
   
2. Jazzy 입력 (ROS 환경설정하는거)
   
3. cd pinky_ws
   
4. source ./install/local_setup.bash
   
5. ros2 run aruco_navigator pinky_pickup_web
    
   
터미널 2 켜서 

1. ssh pinky@192.168.0.154
 
2. cd pinky_violet
 
3. source ./install/local_setup.bash
 
4. ros2 launch pinky_bringup bringup.launch.xml motor_ratio:=0.9

 
