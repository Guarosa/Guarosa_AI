# Project Guarosa 

본 프로젝트는 2020 하반기 멀티캠퍼스 융복합 프로젝트(IoT, Big Data, AI, Cloud)의 일환으로 진행되었습니다.

* 기획 의도: 하루 장시간 PC를 다루며 일하는 직장인의 각종 자세를 IoT 기기로 확인하여 당사자로 하여금 이를 교정하여 건강하게 업무를 지속하도록 보조하는 시스템을 구축
* 구성원: 2020 하반기 멀티캠퍼스 융복합 프로젝트 6조 ''과로사'' (총 6명 (AI 담당 총 2명))



본 프로젝트의 AI 모델은 다음과 같습니다.

* Eye Blink Detector
* Emotion Recognizer (by speech and facial expression)
* Posture Corrector



## Eye Blink Detector

* 양안의 깜빡임을 감지



## Emotion Recognizer

#### Speech-Emotion Recognizer

1. Data Input

   \- MIC를 통해 3초 가량의 voice를 녹음함(output.wav)

   \- 이를 모델에 투입하고 결과를 산출해 내는 과정이 무한루프를 통해 반복됨.

2. Prediction

   \- 감정의 category는 총 8개('angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise')로 구분됨.

   \- 음성 데이터를 모델에 투입하여 softmax 함수를 통해 최종 결과 값 산출해 냄.

   \- 즉, 전체 확률을 1로 두고 각각의 category에 대해 예측되는 확률을 출력함. 

   \- 이는 아래 # Prediction 섹션에서 변수 preds에 기록됨.

   \- preds의 index는 순서대로 'angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'와 상응함.

3. Moving Agerage

   \- 출력의 변동을 줄여 안정적으로 감정을 인식하고자 최근 5개의 음성데이터에 대해 이동 평균을 냄.

   \- 이는 아래 # Moving Average 섹션에서 변수 mean_stack에 기록됨.

4. Print Output

   \- 상위 3개의 감정결과와 이동평균 감정결과가 # Print Output 섹션을 통해 출력됨.

   

#### Facial Expression-Emotion Recognizer

1. Data Input

   \- 웹캠으로 얼굴을 촬영, Facial Detector 로 얼굴을 찾아 전송

   \- 입력받은 영상을 모델에 투입하고 결과를 산출해 내는 과정이 무한루프를 통해 반복됨.

2. Prediction

   \- 감정의 category는 총 8개('angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise')로 구분됨.

   \- 전이학습 (MobileNet, EffNet, DenseNet) 을 활용 후 앙상블하여 정확도를 높일 예정

   \- 학습 속도 개선을 위해 TFRecord로 데이터 변환 후 진행

   \- 정확도 상승 및 학습 속도 개선을 위해 얼굴 부위만 크롭하여 학습 진행 중

   \- [DataSet](https://zenodo.org/record/1188976#.X9OGGi-UFmA) : video 파일만 사용

3. Print Output

   \- 상위 3개의 감정결과가 Print Output 섹션을 통해 출력됨.

   

## Posture Corrector

* 양 어깨의 key points를 받아 (y 거리)/(x 거리)를 자세 구분 기준으로 삼음

     기준 < 0.20 : 정상(normal)

     0.70 > 기준 >= 0.20 : 잘못된 자세(warning)

     (그 외에는 이상치(abnormal))

     

* 기준치에 도달한 뒤 30 frame이 유지된 경우에만 해당 자세로 인정

  (경계 값에서 불필요하게 빈번한 신호발송 방지 위함)



## 설치 및 실행

* Webcam을 통해 영상 데이터를 받는 Eye Blink Detector와 Expression-Emotion Recognizer, Posture Corrector는  `video.py` 으로 함께 실행됩니다.

  ```bash
  cd video_sect
  python video.py
  ```

  이를 실행하기 위해서는 다음 패키지가 필요합니다. (최신 버전 요망)

  * Opencv

  * dlib

  * kera

  * numpy

  * imutils

  * MySQLdb

  * mediapipe

    

* MIC를 통해 음성 데이터를 받는 Speech-Emotion Recognizer는 `audio.py`  로 실행됩니다.

  ```bash
  cd audio_sect
  python audio.py
  ```

  이를 실행하기 위해서는 다음 패키지가 필요합니다. (최신 버전 요망)

  * keras

  * librosa

  * numpy

  * pandas

  * sciket-learn

  * keyboard

  * pyaudio

  * wave

  * MySQLdb

  * joblib

    

