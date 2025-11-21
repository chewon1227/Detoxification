# CLOVA Studio API 테스트 가이드

## 📋 준비사항

1. CLOVA Studio 테스트 API 키 발급
2. Python 3.9 이상

---

## 🛠️ 환경 설정

### 1. Conda 환경 생성 및 활성화

```bash
# 새 환경 생성
conda create -n clova python=3.10 -y

# 환경 활성화
conda activate clova

# 필요한 패키지 설치
pip install requests python-dotenv
```

### 2. API 키 설정

#### 방법 1: 코드에 직접 입력 (간단한 테스트용)

`test.py` 파일을 열어서 상단의 API_KEY 변수에 발급받은 키를 입력:

```python
API_KEY = "nv-xxxxxxxxxxxxxxxxxx"  # 여기에 실제 API 키 입력
```

#### 방법 2: 환경 변수 사용 (권장)

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일 편집하여 API 키 입력
# CLOVA_API_KEY=nv-xxxxxxxxxxxxxxxxxx
```

---

## 🚀 실행 방법

### 기본 테스트 (test.py)

```bash
python test.py
```

- 미리 정의된 2개의 테스트 메시지를 순차적으로 전송
- API 응답 및 토큰 사용량 확인

### 환경 변수 버전 (test_secure.py) - 권장

```bash
python test_secure.py
```

- `.env` 파일에서 API 키를 안전하게 로드
- 초기 테스트 후 대화형 모드 진입 가능
- 실시간으로 메시지를 입력하고 응답 확인

---

## 📝 API 엔드포인트 설명

문서에 따르면 다양한 API가 제공됩니다:

### Chat Completions (기본)
```
https://clovastudio.stream.ntruss.com/testapp/v1/chat-completions/HCX-003
```

### 다른 API들
- **Chat Completions v3 (이미지)**: 이미지 해석 + 대화
- **Chat Completions v3 (추론)**: 논리적 사고를 통한 추론
- **Function Calling**: 외부 함수/API 호출
- **Structured Outputs**: JSON Schema 형식 출력
- **임베딩**: 텍스트 벡터화
- **요약**: 긴 문장 요약
- 등등...

테스트 코드의 `API_URL`을 변경하여 다른 API도 테스트할 수 있습니다.

---

## 🔧 주요 파라미터 설명

```python
payload = {
    "messages": [...],        # 대화 내용
    "topP": 0.8,             # 누적 확률 (0~1, 높을수록 다양한 응답)
    "topK": 0,               # 상위 K개 토큰 선택 (0=비활성화)
    "maxTokens": 256,        # 최대 생성 토큰 수
    "temperature": 0.5,      # 창의성 (0~1, 높을수록 창의적)
    "repeatPenalty": 5.0,    # 반복 페널티 (높을수록 반복 억제)
    "includeAiFilters": True # AI 필터 포함 여부
}
```

---

## ⚠️ 주의사항

### API 키 보안
- **절대 GitHub 등에 API 키를 올리지 마세요!**
- `.env` 파일은 `.gitignore`에 추가
- 테스트 후 키가 노출되었다면 즉시 재발급

### API 제한
- 테스트 API는 개발/테스트 용도
- 실제 서비스 배포 시에는 서비스 API 키 신청 필요

### 구버전 URL
구버전 URL(`https://clovastudio.apigw.ntruss.com/`)은 지원 중단 예정이므로
새 URL(`https://clovastudio.stream.ntruss.com/`)을 사용하세요.

---

## 🐛 문제 해결

### 401 Unauthorized
- API 키가 올바른지 확인
- `Bearer` 접두사가 제대로 붙었는지 확인

### 타임아웃
- 네트워크 연결 확인
- `timeout` 값 증가 시도

### 응답 에러 (status.code != "20000")
- 응답 메시지에서 에러 원인 확인
- CLOVA Studio 문제 해결 문서 참조

---

## 📚 추가 리소스

- [CLOVA Studio API 문서](https://api.ncloud-docs.com/docs/ai-naver-clovastudio)
- [CLOVA Studio 포럼](https://www.ncloud.com/forum/7)
- 고객지원: 1544-5876

---

## ✅ 체크리스트

- [ ] Conda 환경 생성 및 활성화
- [ ] 필요한 패키지 설치
- [ ] API 키 발급 완료
- [ ] API 키 설정 (코드 또는 .env 파일)
- [ ] 테스트 실행 성공
- [ ] 응답 확인 완료

---

**Happy Coding! 🎉**
