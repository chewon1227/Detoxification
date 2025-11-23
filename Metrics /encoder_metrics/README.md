# 혐오 표현 탐지 모델 (Hate Speech Detection)

KCBERT 기반 한국어 혐오 표현 탐지 모델

---

## 파일 구조

```
├── 1120_kcbert_finetuning.py    # 모델 학습
├── 1120_kcbert_kmhas.py         # 데이터 전처리 & KMHAS 
├── 1120_kcbert_test.py          # 모델 테스트
└── hate_speech_output/          # 학습된 모델 저장 폴더
    └── model/
        └── final_model/         # 최종 모델
```

---

## 사용 방법

### 1. 환경 설정

```bash
pip install torch transformers pandas numpy
```

### 2. 모델 학습

```bash
python 1120_kcbert_finetuning.py
```

### 3. 모델 테스트

```bash
python 1120_kcbert_test.py
```

---

## 테스트 기능

- 기본 예시: 일반 문장 + 혐오 표현 30개 이상
- 대화형 테스트: 직접 입력해서 테스트
- 긴 문장 테스트: 여러 문장 처리
- CSV 저장: 결과 자동 저장
- 엣지 케이스: 특수 문자, 띄어쓰기 등

---

## 결과

| 메트릭 | 값 |
|--------|-----|
| 평균 유해도 (Base) | 0.4845 |
| 평균 유해도 (Detox) | 0.0000 |
| 암시적 편향 개선 | 56% |

---

## 주요 코드

```python
from 1120_kcbert_test import HateSpeechDetector

# 모델 로드
detector = HateSpeechDetector("./hate_speech_output/model/final_model")

# 단일 예측
result = detector.predict_single("안녕하세요")
print(result)
# {'text': '안녕하세요', 'hate_score': 0.05, 'prediction': '일반'}
```

---

## 모델 설정

- 모델: KCBERT (Korean Character BERT)
- 토크나이저: BertTokenizer
- 최대 길이: 128 토큰
- 임계값: 0.5 (0.5 이상 = 혐오 표현)

---

## 라이선스

내부 프로젝트