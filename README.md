# 1. 제목 및 동기

## 1.1 제목

실시간 객체 탐지 모델로 철도 선로 결함 탐지

## 1.2 동기

기존 철도 선로 탐지는 초음파를 이용한 탐상차와 휴대용 감지기였다.

### 1.2.1 기존방식 문제점

그러나 기존 방식들은 비효율적인 수동 탐지로 전문인력이 필요했고 소요 시간이 길었으므로 비용적인 문제가 있었고,

또한 현장 환경조건에 의존해야 했으므로 수시 점검이 불가하며, 노후화된 시스템 때문에 유지보수하기 어려웠다.

### 1.2.3 해결방안

이에 대한 해결 방안으로 선로를 자동 탐지하고, 최신화된 시스템으로 용이하게 유지보수 할 수 있는 딥러닝 기반 모델을 제시한다. 이는 전문 인력을 최소화하고 탐지 시간을 줄여 비용을 절약 가능하다.

---

# 2. 데이터셋 구성

## 2.1 데이터셋 소개

### 2.1.1 도시철도 데이터

도시철도 데이터는 정상 데이터가 4만9천여개, 비정상 데이터가 1만 1천여개로 약 6만개의 데이터가 학습 : 검증 : 시험 = 8:1:1 비율로 나뉘어진다.

최종 입력 데이터 경로에서 정상 및 비정상 데이터를 합쳤고, 각 이미지의 정상, 비정상 상태와 BBOX는 JSON파일에 표기 되어있다.

본 데이터를 이용한 모델 학습 방향은 입력 데이터인 철도 사진에 대한 정상 및 비정상 클래스를 탐지 하는것입니다. 모델은 인스턴스를 탐지하지 않는 대신 정상인지, 아니면 어떤 비정상을 가지고있는지 나타낸다.

## ii) 데이터셋 처리

본 데이터는 polygon과 bounding box가 무작위로 섞여있었으므로 객체 탐지를 하기위해 polygon 데이터를 최소한의 크기의 bouding box로 바꾸는 작업을 진행, 또한 json 라벨 데이터를 yolo형식으로 바꾸고 이미지를 resize하는 등의 yolo모델로 input하기 위한 전처리를 진행했다.

---

# 3.모델

## 3.1 모델 소개

YOLOv8은 2023년 1월 출시된 객체 감지, 이미지 분류 모델 학습 위한 통합 프레임워크로, 그 중에서도 학습에 가장 경량 모델인 Yolov8-n을 이용하였다. YOLOv8은 Darknet-53 , ResNet같은 백본 및 넥 아키텍쳐를 사용한다. 또한 다양한 사전 학습 모델을 가지고 있으며, 높은 Precision과 Recall 지수를 가지고 있다. 그러나 배경이 복잡하거나 객체가 겹칠 때 오탐지 가능성이 있고 빠른 추론속도를 가진 대신 정확성이 떨어진다는 단점을 가지고 있다.

YOLOv9은 2024년 2월에 일부 출시된 기술 한계 극복을 위해 PGI와 GELAN을 도입한 모델이다. 출시된 모델중 제일 작은 Yolov9-c를 학습에 적용하였다. YOLOv9은 YOLOv8에 비해 객체 탐지의 정확도가 개선되었고, 정보 병목 원리를 적용해 딥 네트워크에서의 정보 손실을 최소화하여 정보를 보존해 높은 성능을 유지한다. 또한 다양한 사전학습 모델을 갖고 있지만 현재까지는 v8-I에 버금가는 v9-C와 v9-E만 출시되었다.(2024년 6월기준) 그에 의해 큰 데이터셋과 모델을 훈련하기 위한 고성능의 자원을 요구하여 어느정도의 한계가 따른다는 단점이 있다.

## 3.2 학습 방법

배치 사이즈 16, 에폭 10,000, patience 10을 기준으로 진행하였다.

---

# 4. 결과 및 분석

## 4.1 결과

Yolov8n은 가장 경량화된 모델로, 3.2M 파라미터를 가지며, mAP50는 0.901다. Yolov8m 모델은 중간 사이즈로, 25.9M 파라미터를 가지며, mAP50은 0.922이다. Yolov8l은 43.7M 파라미터를 가지며, mAP50은 0.920이다. Yolov9c는 v9 시리즈 중 가장 작은 모델로, 25.3M 파라미터를 가지며, mAP50는 0.926이다. 테스트 결과, 모든 모델은 비슷한 성능을 보였다.

box plot은 데이터의 대략적 분포를 파악하는 시각화 기법이다. mAP50에서 8m, 8l, 9c 모델들은 유사한 성능을 보이며, YOLOv8n 모델은 조금 더 낮은 성능을 보인다. 이상치를 제외하면, 모든 모델은 AP_50과 AP_50_95에서 유사한 성능을 보인다.

## 4.2 고찰

---

비교 결과 큰 차이가 없었지만, 학습과 테스트 결과는 좋았다:

1. yolov8과 v9는 동일한 기본 yolo구조를 따르며, 객체 탐지와 클래스 예측을 동시에 한다.
2. yolov8과 v9의 최적화된 문제가 다르며, yolov8보다 v9는 대용량 데이터셋에 초점이 맞춰져 있다. 따라서 태스크의 목적에 따라 모델을 골라야한다.
3. 데이터 크기가 크거나 태스크가 복잡하지 않았기 때문에 네 모델의 성능이 다 좋았다.

적합한 태스크와 모델을 찾기 위해:

1. 클래스가 문제 해결과 관련이 있는지 연구해야 한다.
2. 정확도, 실시간성, 자원 제약에 대해 구체적으로 정해야 한다.
3. 모델을 튜닝하여 성능을 최적화해야 한다.
4. 실제 환경에서 성능을 테스트해야 한다.

결론적으로, ap 수치로 보면 v9가 더 낫다. 하지만 태스크의 고정적인 제약에 따라 다르게 판단할 수도 있다.
