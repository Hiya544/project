# project
CCTV를 활용한 실시간 객체 추적


1. 시스템 설계
CCTV를 활용한 실시간 객체 추적은 그림 1 과 같이 웹 애플리케이션. 관리서버.데이터베이스.웹캠으로 구성된다.
웹캠을 통해 수집한 영상들은 정보를 학습한 후 데이터베이스에 저장이 된다. 
사용자는 PC의 브라우저를 이용하여 학습된 정보를 바탕으로 인물을 지속적으로 추적할 수 있다. 상호 작용은 그림2와 같다

그림 1 시스템의 구성
![image01](https://github.com/Hiya544/project/assets/62420840/f622334e-4254-4213-a13a-bb241da5f9eb)
![화면 캡처 2023-10-03 180202](https://github.com/Hiya544/project/assets/62420840/cb933a98-1ce6-436e-8d38-c811f0fe9d1e)
그림2 시스템의 구성 요소간의 상호작용



![image02](https://github.com/Hiya544/project/assets/62420840/85fc2380-3d52-4369-a77c-db36f00fcb54)
2.추적 시스템의 기능 설계


데이터 수집 기능은 웹캠으로 받은 영상을 학습한후 객체 추적을 위해 사용할 데이터를 수집한다
1단계 에서는 객체 인식을 하기 전 영상의 화질을 최대한 끌어 올리기 위해 SRGAN을 활용하여 영상의 노이즈 와 화질올린다.
2단계 객체 인식 기능은 opencv , yolo의 객체 인식 알고리즘을 이용하여 영상속의 전체적인 객체를 인식한다
3단계 객체 추적 단계 에서는 CNN을 활용하여 객체 인식한 특정 인물을 지정하면 다른웹캠에서도 같은 대상을 지속적으로 추격하게끔 역할을 부여
