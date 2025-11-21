import os
import requests
import json

# API 설정
API_KEY = "nv-1267c51ff93b4245b59e07fbc65567e04TJc"  # 발급받은 테스트 API 키를 여기에 입력하세요
API_URL = "https://clovastudio.stream.ntruss.com/testapp/v1/chat-completions/HCX-003"

def test_clova_chat(user_message):
    """
    CLOVA Studio Chat Completions API 테스트
    
    Args:
        user_message (str): 전송할 메시지
    """
    
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
        print(f"🔗 API URL: {API_URL}")
        print("⏳ 응답 대기 중...\n")
        
        response = requests.post(
            API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        # 응답 상태 코드 확인
        print(f"📊 상태 코드: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            
            # 응답 구조 확인
            if "status" in result:
                status = result.get("status", {})
                print(f"✅ 상태: {status.get('code')} - {status.get('message')}")
                
                if status.get("code") == "20000":
                    # 성공적인 응답
                    ai_message = result.get("result", {}).get("message", {}).get("content", "")
                    print(f"\n🤖 AI 응답:\n{ai_message}\n")
                    
                    # 토큰 사용량 정보
                    usage = result.get("result", {}).get("usage", {})
                    if usage:
                        print(f"📊 토큰 사용량:")
                        print(f"   - 입력 토큰: {usage.get('inputTokens', 0)}")
                        print(f"   - 출력 토큰: {usage.get('outputTokens', 0)}")
                        print(f"   - 총 토큰: {usage.get('totalTokens', 0)}")
                    
                    return result
                else:
                    print(f"❌ 에러: {status.get('message')}")
            else:
                # 전체 응답 출력
                print(f"\n📄 전체 응답:\n{json.dumps(result, indent=2, ensure_ascii=False)}")
        else:
            print(f"❌ HTTP 에러: {response.status_code}")
            print(f"응답 내용: {response.text}")
            
    except requests.exceptions.Timeout:
        print("⏰ 타임아웃: 요청 시간이 초과되었습니다.")
    except requests.exceptions.RequestException as e:
        print(f"❌ 요청 에러: {e}")
    except json.JSONDecodeError as e:
        print(f"❌ JSON 파싱 에러: {e}")
        print(f"응답 내용: {response.text}")
    except Exception as e:
        print(f"❌ 예상치 못한 에러: {e}")
    
    return None


def main():
    """메인 함수"""
    
    # API 키 확인
    if API_KEY == "your-api-key-here":
        print("⚠️  경고: API 키를 설정해주세요!")
        print("스크립트 상단의 API_KEY 변수에 발급받은 키를 입력하세요.\n")
        return
    
    print("=" * 60)
    print("🚀 CLOVA Studio API 테스트")
    print("=" * 60 + "\n")
    
    # 테스트 메시지들
    test_messages = [
        "안녕하세요! 간단한 자기소개 부탁드립니다.",
        "Python으로 리스트를 정렬하는 방법을 알려주세요.",
    ]
    
    for i, message in enumerate(test_messages, 1):
        print(f"\n{'='*60}")
        print(f"테스트 {i}/{len(test_messages)}")
        print(f"{'='*60}\n")
        
        test_clova_chat(message)
        
        if i < len(test_messages):
            print("\n" + "-" * 60 + "\n")
    
    print("\n" + "=" * 60)
    print("✅ 테스트 완료!")
    print("=" * 60)


if __name__ == "__main__":
    main()