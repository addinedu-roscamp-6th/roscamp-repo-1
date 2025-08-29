# 🤖 자율 순찰 및 비서 기능의 통합로봇 시스템 Guardian_P

## ✨ 프로젝트 개요

최근 순찰 / 경비 분야 인력 부족 및 이탈 상황이 많이 발생하고 있습니다. 따라서 순찰 경비 로봇의 사례가 늘고 있습니다. 
하지만 이는 단순한 경비 영역에 대한 정보 수집 및 상황 공유 기능에만 집중되고 있으며 실제 물리적 상황 해결 능력이 전무합니다.
이를 스마트 오피스 시스템에 결합하면 각각의 로봇 개체의 자율성이 높아져 효율적인 로봇 스마트 오피스 시스템을 이룰 것이라고 생각합니다.

따라서 저희 프로젝트는 자율주행 기능과 로봇팔을 결합하여 실내공간에서 자율 순찰과 상황 대응을 수행하는 스마트 경비 로봇을 개발하고자 하였습니다.

### 프로젝트 목표

- 프로젝트 목표 : 스마트 오피스 환경에서 보안 및 물품 지원

- 시스템 : 다개체 로봇 기반 지능형 경비 . 지원 서버

### 프로젝트 주요 기능

#### 자율주행/상황 대응
- 순찰 주행 : 정해진 경로를 따라 이동합니다.
- 장애물 감지 : 장애물을 감지하면 정지 후 사라지면 기존 경로를 주행합니다.
- 쓰러짐 감지 : 쓰러진 사람을 감지하면 DB 저장 및 정지 후 해결 완료 시 순찰을 재개합니다.
- 외부인 감지 : 외부인을 감지하면 DB 저장 및 관리자에게 알립니다.

#### 물품 적재 기능
- 물품 감지 및 적재 : 순찰 로봇이 픽업 장소에 도착 시 물품을 적재합니다.

#### 인터페이스 
- 관리자 GUI
- 사용자 GUI

## 💻 기술 스택

| 영역 | 기술 |
|-----|-----|
| **개발환경** | Ubuntu 24.04, VS Code, ROS2 Jazzy, Flask |
| **언어** | Python, SQL |
| **DB / GUI** |  MySQL, PyQt5 |
| **Network** | ROS2(DDS), TCP/IP, HTTP |
| **H/W** |  Jetcobot, pinky, Raspberry Pi, Cam |
| **Perception** | YOLOv8(Pose), OpenCV, InsightFace |
| **협업 및 일정관리** | Confluence, Jira, Git, Slack |

## ⚙️ 시스템 구성

### 하드웨어 아키텍처

<img width="1870" height="1080" alt="Image" src="https://github.com/user-attachments/assets/3c2e369d-4cb1-43fc-aac3-3a188753b416" />

### 소프트웨어 아키텍처

<img width="1806" height="1067" alt="Image" src="https://github.com/user-attachments/assets/5ead07cf-b405-44f6-808d-99e8940fc502" />


## 👥 역할분담

| 담당 업무 | 이름 |
|-----|-----|
| **YOLO (쓰러짐 감지)** | 김성민, 위다인 |
| **자율 주행 및 Path Planning** | 박지안, 서원우, 위다인 |
| **DB** |  김성민, 이동연 |
| **얼굴인식** | 이동연 |
| **로봇팔 제어** |  박진우, 백기광 |
| **Web 기반 연동 및 통합 제어** | 김성민, 위다인 |
| **GUI** | 김성민, 박진우, 백기광, 이동연 |

[1팀 최종 ppt 자료](<https://docs.google.com/presentation/d/1WiwNvU6zXkpr6sOqes_zbciIVpUQkfIj8enjXyJ0BXs/edit?usp=sharing>)
