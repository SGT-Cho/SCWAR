

# SC ASR 프로젝트

## 소개
SC ASR 프로젝트는 **SC ASR 패키지**와 **GMM-HMM 모델**을 활용하여 음성인식 성능을 최적화하는 연구를 진행하였습니다. 본 연구는 대규모 데이터셋을 기반으로 음성 데이터를 분석하고, 간결한 코드와 표준 라이브러리만을 사용하여 모델 학습 및 평가를 수행하였습니다.

---

## 주요 특징
- **데이터 전처리 및 증강**: 
  - Google Speech Commands v0.02 데이터셋을 `Train:Validation:Test = 0.7:0.2:0.1` 비율로 분리.
  - FFT 기반 노이즈 제거, Time Stretching 및 Pitch Shifting 등의 증강 기법 적용.
  
- **특징 추출 (MFCC)**:
  - 13차원 특징 벡터를 추출하여 음향 모델 학습에 활용.

- **모델 학습 (GMM-HMM)**:
  - HMM 초기화를 위해 글로벌 평균 및 표준편차를 계산.
  - 단일 가우시안 기반 HMM 대신 Gaussian Mixture Model(HMM)을 활용하여 성능 개선.
  
---

## 실험 환경
- **운영 체제(OS)**: Windows 11 Pro
- **CPU**: AMD Ryzen 7 5800H with Radeon Graphics (3.20 GHz)
- **RAM**: 32GB
- **Python 버전**: 3.7.12
- **사용된 라이브러리**: 표준 Python 라이브러리 및 Numpy

---

## 모델 구조
- **GMM-HMM 기반**:
  - 각 상태(State)의 음향 분포를 모델링.
  - 발음 사전과 단어 시퀀스를 기반으로 단어 간 관계를 학습.
  
- **Lexicon 및 Language Model**:
  - 발음 사전과 단어 시퀀스의 관계를 학습하여 음성 인식 성능을 개선.

---

## 실험 결과
- **정확도**: 63.50%
- **모델 사이즈**:
  ```bash
  du -h 5.hmm
  848K    5.hmm
  ```
- **테스트 데이터 분석**:
  - 단일 가우시안 기반 HMM 대비 GMM-HMM 모델에서 10% 이상의 성능 향상.

---

## 실행 방법
### 1. 환경 설정
```bash
# Python 버전 설치
conda create -n sc_asr python=3.7.12
conda activate sc_asr

# 필수 라이브러리 설치
pip install numpy
```

### 2. 데이터 준비
- Google Speech Commands 데이터셋 다운로드:
  [Google Speech Commands 데이터셋](https://arxiv.org/abs/1804.03209).

### 3. 모델 실행 절차
1. **특징 추출**:
   ```bash
   python 01_compute_mfcc_kr_sc.py
   ```
2. **평균 및 표준편차 계산**:
   ```bash
   python 02_compute_mean_std_kr.py
   ```
3. **HMM 초기화**:
   ```bash
   python 02_init_hmm_kr.py
   ```
4. **GMM-HMM 학습**:
   ```bash
   python 03_train_gmmhmm_kr.py
   ```
5. **결과 분석**:
   ```bash
   python eval.py
   ```

---

## 향후 연구 방향
1. **GMM-HMM 구조 최적화**:
   - 상태(State) 수 및 믹스처(Mixture) 개수 최적화.
2. **데이터 증강**:
   - 노이즈 제거 및 데이터 증강 기법 추가 적용.
3. **하이브리드 접근**:
   - GMM-HMM과 딥러닝 기반 Acoustic Model 결합.

---

## 참고 문헌
1. Google Speech Commands 데이터셋 ([https://arxiv.org/abs/1804.03209](https://arxiv.org/abs/1804.03209))
2. GMM-HMM 관련 문서 ([https://wikidocs.net/223858](https://wikidocs.net/223858))
```
