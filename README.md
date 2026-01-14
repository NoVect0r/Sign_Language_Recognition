# 🤟 DeepSign: 실시간 수어(지문자/지숫자) 인식 및 단어 완성 시스템

> **MediaPipe와 LSTM을 활용한 4만 장 규모 자체 데이터셋 기반 수어 인터페이스**
> 
> **🏆 교내 경진대회 금상(1위) 수상작**

---

## 📺 Project Demonstration

<video src="./DeepSign%20final%20Test.mp4" controls width="100%"></video>

*영상이 보이지 않을 경우 [여기](./DeepSign%20final%20Test.mp4)를 클릭하여 확인하세요.*

---

## 🚀 핵심 기술 및 성과 (Key Achievements)

* **4만 장 규모의 자체 데이터셋 구축 및 정밀 라벨링**
    * 한글 수어 데이터셋 부족 문제를 해결하기 위해 MediaPipe를 활용하여 직접 4만 장의 지문자/지숫자 데이터를 수집하고 정밀 라벨링을 수행했습니다.

* **42차원 고도화 특징점(Feature Engineering) 설계**
    * 21개 관절 랜드마크 좌표를 손의 위치, 회전, 간격을 포함한 **42차원 상대적 위치 정보**로 변환하여 인식 정밀도를 향상시켰습니다.

* **LSTM 기반 문맥 판별 모델 도입**
    * 'ㅅ'과 'ㅠ'처럼 유사한 손모양 문자를 구분하기 위해 이전 입력 시퀀스를 바탕으로 다음 문자를 추론하는 **LSTM 기반 시계열 모델**을 도입했습니다.

* **실시간 단어 완성 파이프라인 구축**
    * 프레임 단위 데이터를 실시간으로 저장하고 딥러닝 모델과 연동하여, 개별 지문자 조합을 통해 완성된 단어를 출력하는 시스템을 구현했습니다. 

---

## 🏗️ 시스템 구조 (System Architecture)



1. **Data Acquisition**: MediaPipe를 활용한 21개 랜드마크 실시간 추정
2. **Preprocessing**: 42차원 특징점 변환 및 데이터 증강을 통한 환경 편차 최소화
3. **Classification**: LSTM 모델을 통해 현재 제스처 및 문맥을 고려한 문자 판별
4. **Integration**: 지문자 조합을 통한 실시간 단어 완성 및 GUI 출력

---

## 🛠️ 기술 스택 (Tech Stack)

* **Language**: Python 3.9+
* **Deep Learning**: TensorFlow, Keras (LSTM), MediaPipe
* **Data Science**: NumPy, Pandas, Scikit-learn, Matplotlib
* **Computer Vision**: OpenCV

---

## 📁 주요 파일 설명 (Main Files)

* **`SiotYu_distinction_LSTM_model_train.py`**: 'ㅅ', 'ㅠ' 구분 및 문맥 추론을 위한 LSTM 학습 코드
* **`get_featured_data_csv.py`**: 랜드마크를 42차원 특징점으로 가공하여 CSV로 저장하는 전처리 스크립트
* **`collect_datasets_IPWebcam.py`**: IP 웹캠 및 카메라를 활용한 대규모 데이터 수집 모듈
* **`original_separate_model_test.py`**: 학습된 모델의 실시간 성능 및 정확도 테스트 코드
