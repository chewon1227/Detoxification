# 🚀 빠른 시작 가이드

## 프로젝트 개요
편향된 RAG 데이터 환경에서 SFT+DPO 순화 모델의 혐오 표현 억제 효과 검증 연구

---

## ⚡ 5분 안에 시작하기

### 1. 환경 설정
```bash
# Conda 환경 생성
conda create -n rag_detox python=3.10 -y
conda activate rag_detox

# 패키지 설치
pip install -r requirements.txt
```

### 2. API 키 설정
```bash
# .env 파일 생성
cp .env.example .env

# .env 파일 편집
# CLOVA_API_KEY=nv-xxxxxxxxxx
```

### 3. 기본 테스트 실행
```bash
# 1단계: Chat API 테스트
python test_secure.py

# 2단계: 임베딩 API 테스트
python test_embedding.py

# 3단계: RAG 시스템 테스트
python test_rag.py

# 4단계: 2-Agent 대화 테스트
python test_dialogue.py
```

---

## 📁 파일 구조

```
project/
├── test_secure.py          # Chat API 기본 테스트
├── test_embedding.py       # 임베딩 API 테스트
├── test_rag.py            # RAG 시스템 프로토타입
├── test_dialogue.py       # 2-Agent 대화 시스템
├── project_roadmap.md     # 전체 프로젝트 로드맵
├── requirements.txt       # 필요 패키지
├── .env.example          # 환경 변수 템플릿
└── README.md             # 이 파일
```

---

## 🎯 Phase별 진행 가이드

### Phase 1: 기본 구현 (현재 단계)
✅ Chat Completions API 테스트  
✅ 임베딩 API 테스트  
✅ RAG 프로토타입 구현  
✅ 2-Agent 대화 시스템 구현

**다음 할 일:**
- [ ] 실제 API로 테스트 (현재는 시뮬레이션)
- [ ] 임베딩 URL 확인 및 수정
- [ ] Chroma DB 설치 및 테스트

### Phase 2: 데이터 수집
- [ ] 크롤링 대상 커뮤니티 선정
- [ ] 웹 크롤러 개발
- [ ] 데이터 정제 및 익명화
- [ ] SFT/DPO 데이터셋 구축

### Phase 3: 모델 튜닝
- [ ] CLOVA Studio 학습 API 문서 확인
- [ ] SFT 훈련 데이터 업로드
- [ ] SFT 학습 실행
- [ ] DPO 훈련 데이터 업로드
- [ ] DPO 학습 실행

### Phase 4: 본 실험
- [ ] 4개 에이전트 구축 (L_Base, R_Base, L_Detox, R_Detox)
- [ ] 실험 프로토콜 확정
- [ ] 대화 실험 실행 (여러 주제)
- [ ] 로그 저장 및 백업

### Phase 5: 평가 및 분석
- [ ] 한국어 혐오 표현 탐지 모델 준비
- [ ] Judge LLM 평가 실행
- [ ] 통계 분석
- [ ] 시각화 및 논문 작성

---

## 🔧 주요 코드 사용법

### Chat Completions (기본)
```python
from dotenv import load_dotenv
import requests
import os

load_dotenv()
API_KEY = os.getenv("CLOVA_API_KEY")

response = requests.post(
    "https://clovastudio.stream.ntruss.com/testapp/v1/chat-completions/HCX-003",
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    },
    json={
        "messages": [
            {"role": "user", "content": "안녕하세요"}
        ]
    }
)

print(response.json()["result"]["message"]["content"])
```

### RAG Agent 생성
```python
from test_rag import SimpleRAGAgent

# RAG 에이전트 생성
rag_L = SimpleRAGAgent("community_L")

# 문서 추가
documents = [
    "이번 사건은 정부 책임이 큽니다.",
    "안전 관리 시스템 개선이 필요합니다."
]
rag_L.add_documents(documents)

# 질문 답변
result = rag_L.generate_response("주요 원인은 무엇인가요?")
print(result["response"])
```

### 2-Agent 대화 실행
```python
from test_dialogue import DialogueAgent, DialogueExperiment

# 에이전트 생성
agent_L = DialogueAgent("Agent_L", "left", model_type="base")
agent_R = DialogueAgent("Agent_R", "right", model_type="base")

# 실험 실행
experiment = DialogueExperiment(
    agent_L, agent_R,
    topic="이태원 참사의 주요 원인은?"
)

log = experiment.run_dialogue(n_turns=5)
experiment.save_log()
```

---

## 💡 자주 묻는 질문

### Q1: 임베딩 API URL이 작동하지 않아요
**A:** 문서의 임베딩 API 엔드포인트를 확인하고 `test_embedding.py`의 `EMBEDDING_URL`을 수정하세요.

### Q2: Chroma DB 설치 오류가 나요
**A:** 
```bash
pip install --upgrade pip
pip install chromadb --no-cache-dir
```

### Q3: API 비용이 얼마나 나올까요?
**A:** 
- Chat API: 약 ₩100,000~200,000 (실험 규모에 따라)
- 임베딩 API: 약 ₩50,000
- **학습 API는 별도 문의 필요** (1544-5876)

### Q4: SFT/DPO 데이터셋은 어떻게 만드나요?
**A:** 
1. GPT-4/Claude로 초안 생성
2. 수동으로 검수 및 수정
3. CLOVA Studio 학습 API 형식에 맞게 변환

### Q5: 실험 결과는 언제 나오나요?
**A:** 
- 데이터 수집: 2-4주
- 모델 튜닝: 1-2주 (학습 대기 시간 포함)
- 실험 실행: 1주
- 분석: 2주
- **총 8-12주 예상**

---

## 🆘 도움이 필요하면

### CLOVA Studio 지원
- 고객센터: 1544-5876
- 포럼: https://www.ncloud.com/forum/7
- 문서: https://api.ncloud-docs.com/docs/ai-naver-clovastudio

### 프로젝트 관련
- `project_roadmap.md` 참조
- 각 파일의 주석 참조
- GitHub Issues (프로젝트 저장소가 있다면)

---

## ✅ 체크리스트

**환경 설정**
- [ ] Conda 환경 생성
- [ ] 패키지 설치
- [ ] API 키 발급 및 설정

**Phase 1 완료**
- [ ] Chat API 테스트 성공
- [ ] 임베딩 API 테스트 성공
- [ ] RAG 시스템 작동 확인
- [ ] 2-Agent 대화 테스트 성공

**다음 단계**
- [ ] `project_roadmap.md` 정독
- [ ] Phase 2 시작 준비
- [ ] 학습 API 비용 문의

---

**Good Luck! 🎓**
