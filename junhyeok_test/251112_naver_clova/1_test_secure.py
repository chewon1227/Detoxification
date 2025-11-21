import os
import requests
import json
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# API 설정 (환경 변수에서 가져오기)
API_KEY = os.getenv("CLOVA_API_KEY")
API_URL = os.getenv(
    "CLOVA_API_URL", 
    "https://clovastudio.stream.ntruss.com/testapp/v1/chat-completions/HCX-003"
)


def test_clova_chat(user_message):
    """
    CLOVA Studio Chat Completions API 테스트
    
    Args:
        user_message (str): 전송할 메시지
    """
    
    if not API_KEY:
        print("❌ 에러: API 키가 설정되지 않았습니다.")
        print("💡 .env 파일을 생성하고 CLOVA_API_KEY를 설정해주세요.")
        return None
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "당신은 친절한 AI 어시스턴트입니다."
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        "topP": 0.8,
        "topK": 0,
        "maxTokens": 256,
        "temperature": 0.5,
        "repeatPenalty": 5.0,
        "stopBefore": [],
        "includeAiFilters": True
    }
    
    try:
        print(f"📤 요청 메시지: {user_message}")
        print("⏳ 응답 대기 중...\n")
        
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"📊 상태 코드: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            if "status" in result:
                status = result.get("status", {})
                
                if status.get("code") == "20000":
                    ai_message = result.get("result", {}).get("message", {}).get("content", "")
                    print(f"\n🤖 AI 응답:\n{ai_message}\n")
                    
                    usage = result.get("result", {}).get("usage", {})
                    if usage:
                        print(f"📊 토큰 사용량: {usage.get('totalTokens', 0)} 토큰")
                    
                    return result
                else:
                    print(f"❌ API 에러: [{status.get('code')}] {status.get('message')}")
            else:
                print(f"\n📄 응답:\n{json.dumps(result, indent=2, ensure_ascii=False)}")
        else:
            print(f"❌ HTTP 에러: {response.status_code}")
            print(f"응답: {response.text}")
            
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
    
    return None


def interactive_mode():
    """대화형 모드"""
    print("\n💬 대화형 모드 (종료: 'quit' 또는 'exit')")
    print("-" * 60 + "\n")
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '종료']:
                print("\n👋 종료합니다.")
                break
            
            if not user_input:
                continue
            
            print()
            test_clova_chat(user_input)
            print("-" * 60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n👋 종료합니다.")
            break


def main():
    """메인 함수"""
    
    print("=" * 60)
    print("🚀 CLOVA Studio API 테스트")
    print("=" * 60 + "\n")
    
    if not API_KEY:
        print("⚠️  API 키가 설정되지 않았습니다!")
        print("\n설정 방법:")
        print("1. .env.example 파일을 .env로 복사")
        print("2. .env 파일에서 CLOVA_API_KEY에 발급받은 키 입력")
        print("3. 다시 실행\n")
        return
    
    # 간단한 테스트
    print("📝 간단한 테스트를 실행합니다...\n")
    test_clova_chat("안녕하세요! 자기소개 부탁드립니다.")
    
    print("\n" + "=" * 60)
    
    # 대화형 모드 진입 여부 확인
    try:
        response = input("\n💬 대화형 모드로 진입하시겠습니까? (y/n): ").strip().lower()
        if response in ['y', 'yes', 'ㅛ']:
            interactive_mode()
        else:
            print("\n✅ 테스트 완료!")
    except KeyboardInterrupt:
        print("\n\n✅ 테스트 완료!")


if __name__ == "__main__":
    main()
