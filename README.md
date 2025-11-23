# Detoxification
NLP, LLM project

정치·사회 이슈 대화에서 좌·우 성향 RAG 기반 LLM의 발화를 비교하고, 순화 모델(SFT/DPO)이 독성 표현을 얼마나 감소시키는지 평가하는 실험입니다.

---

## 1. 데이터 준비

### A. 수집 커뮤니티

* 디시 인사이드
  공개 게시판 중심으로 HTML 크롤링 가능 여부 확인 후 수집합니다.

### B. 수집 규모

* 주제당 게시글 200–300개
* 약 1,200–2,400 문장(주제 × 성향) 확보

### C. 주제 및 대표 키워드

1. 난민 수용정책

2. 퀴어 퍼레이드 / 성소수자 권리

3. 병역제 / 모병제

4. 여성가족부 폐지 논쟁

### D. 전처리

* date, main, comments 등으로 정리. 

---

## 2. RAG 데이터베이스 구축

### 임베딩 모델

* BAAI/bge-m3

### Vector DB

* Qdrant

### 메타데이터 예시

```json
  {
    "date": "2025.11.16 16:30:47",
    "main": "카제나 스소 듀나어 가디스오더 등등 난민들 다 태우고 출항예정이라함 ㄷㄷ",
    "comments": [
      "타이타닉호 ㅅㅂㅋㅋ",
      "ㅋㅋ 이번에도 망하면 금태는 겜운영 접어야지",
      "조만간 가라앉는다는 썰어있던데",
      "편도네  - dc App",
      "3돌 메모리얼",
      "망하면 카사랑 같이 바즈 따라오너라"
    ],
    "source_url": "https://gall.dcinside.com/mgallery/board/view/?id=chaoszero&no=267491",
    "gallery": "카오스제로 나이트메어"
  }
```

### Chunking

* 3–4문장 단위, stride 1–2
* 매 대화 턴마다 top-5 문서 retrieval
* chunk에서 핵심 문장 1–2개 인용

---

## 3. 순화 모델(Sanitization Model)

### SFT 단계

* 목적: 혐오·편향적 문장을 정중하고 중립적인 문장으로 재작성
* 예시:

```json
{"input": "요즘 애들은 끈기가 없어.",
 "output": "요즘 세대는 가치관이 다를 수 있어요."}
```

* Learning rate: 1e-5
* Batch size: 32
* Epochs: 3

### DPO 단계

* 목적: 좋은 답변(중립적)과 나쁜 답변(공격적)을 구분하도록 학습

```json
{
  "prompt": "여성가족부는 필요 없다.",
  "chosen": "정책 효율성 논의는 필요하지만 존재 이유를 단정하긴 어렵습니다.",
  "rejected": "맞아요, 당장 없애야죠."
}
```

* Beta: 0.1–0.3
* Learning rate: 5e-6
* Epochs: 2

---

## 4. 대화 실험 세팅

### 구성

* 좌성향 기반 모델 vs 우성향 기반 모델
* 주제: 4개
* 총 10턴 대화
* Temperature: 0.7
* Top-p: 0.9

### 시스템 프롬프트 규칙

* 특정 커뮤니티의 발화 패턴을 모사
* RAG 인용을 기반으로 응답
* 2–3문장 유지
* 상대 의견에 동의 또는 반박 포함

### 시작 질문 원칙

1. 중립적이지만 논쟁 유도
2. 주제 핵심 단어를 명확하게 포함
3. 질문형으로 마무리
4. 첫 발화자(L 또는 R) 고정

---

## trained 모델
https://drive.google.com/drive/folders/1C9M7dy5sLvsKl2xXbPfE4KcAliJ13-I7
---

## 5. 자동 평가

### (1) 명시적 혐오

* 모델: KoBERT-hate, KOLD
* 지표: 문장별 독성 점수, 토큰 독성률

### (2) 암시적 혐오 평가(LLM 심판)

* GPT, Claude, Gemini 등
* 평가 항목

  * 비꼬기
  * 조롱
  * 편향 강요
  * 훈계조 표현

### (3) 회피 경향 / 유용성 점검

---

## 6. 결과 분석

* 명시적 혐오 점수 평균 비교
* 암시적 혐오 점수 비교
* 주제별 편향 경향 분석
* 시각화 정리

---

## 7. 디렉토리 구조

NOT YET

---

## License

This project is for research on LLM safety and bias mitigation.
