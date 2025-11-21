"""
Naver Clova API로 유해도 평가 - 가장 간단한 버전
"""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("CLOVA_API_KEY")


def evaluate_toxicity(text: str) -> dict:
    """문장의 유해도를 0-1 사이로 평가"""
    
    # 프롬프트 작성
    prompt = f"""당신은 한국어 혐오 표현 분석 전문가입니다.

다음 문장의 유해도를 0과 1 사이의 숫자(소수점 3자리)로 평가하세요.

평가 기준:
- 0.0~0.2: 완전히 무해
- 0.2~0.4: 약간 부정적
- 0.4~0.6: 중립적/약간 공격적
- 0.6~0.8: 명확한 혐오
- 0.8~1.0: 심각한 혐오/욕설

평가할 문장: "{text}"

응답은 다음 JSON 형식으로만 답하세요:
{{"toxicity_score": 0.XX, "reason": "간단한 이유"}}"""

    # API 호출
    try:
        response = requests.post(
            "https://clovastudio.stream.ntruss.com/testapp/v1/chat-completions/HCX-003",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "maxTokens": 100,
                "temperature": 0.1,  # 일관된 결과를 위해 낮은 온도
            },
            timeout=30
        )

        # 응답 처리
        if response.status_code == 200:
            result = response.json()
            
            # API 응답에서 실제 내용 추출
            if result.get("status", {}).get("code") == "20000":
                content = result["result"]["message"]["content"]
                
                # JSON 파싱
                try:
                    # 응답에서 JSON 부분만 추출
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    json_str = content[json_start:json_end]
                    
                    parsed = json.loads(json_str)
                    score = float(parsed.get("toxicity_score", 0))
                    reason = parsed.get("reason", "")
                    
                    # 점수를 0-1 범위로 정규화
                    score = max(0.0, min(1.0, score))
                    
                    return {
                        "text": text,
                        "score": round(score, 3),
                        "reason": reason,
                        "status": "success"
                    }
                except json.JSONDecodeError as e:
                    print(f"JSON 파싱 오류: {e}")
                    print(f"응답: {content}")
                    return {
                        "text": text,
                        "score": None,
                        "reason": "JSON 파싱 실패",
                        "status": "error"
                    }
        
        print(f"API 오류: {response.status_code}")
        return {
            "text": text,
            "score": None,
            "reason": "API 호출 실패",
            "status": "error"
        }

    except Exception as e:
        print(f"오류: {e}")
        return {
            "text": text,
            "score": None,
            "reason": str(e),
            "status": "error"
        }


def main():
    """테스트 실행"""
    
    print("\n" + "="*70)
    print("🔍 Clova API 유해도 평가 - 간단 테스트")
    print("="*70 + "\n")
    
    if not API_KEY:
        print("❌ API_KEY가 설정되지 않았습니다!")
        print("   .env 파일에 CLOVA_API_KEY를 설정하세요.")
        return
    
    # 테스트 문장들
    test_sentences = [
        "야이 씨발아",
        "이런 스발넘",
        "미친넘",
        "싧밟아"
    ]
    
    print("테스트 문장들을 평가합니다...\n")
    print("-"*70)
    
    results = []
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n[{i}] {sentence}")
        result = evaluate_toxicity(sentence)
        results.append(result)
        
        if result["status"] == "success":
            score = result["score"]
            
            # 점수에 따른 이모지
            if score < 0.2:
                emoji = "✅"
                label = "무해"
            elif score < 0.4:
                emoji = "⚠️"
                label = "약간 부정적"
            elif score < 0.6:
                emoji = "⚠️"
                label = "중립"
            elif score < 0.8:
                emoji = "⛔"
                label = "혐오"
            else:
                emoji = "🚫"
                label = "심각한 혐오"
            
            print(f"    {emoji} 점수: {score:.3f} ({label})")
            print(f"    💡 {result['reason']}")
        else:
            print(f"    ❌ {result['reason']}")
    
    # 요약
    print("\n" + "-"*70)
    print("\n📊 평가 요약\n")
    
    successful = [r for r in results if r["status"] == "success"]
    
    if successful:
        scores = [r["score"] for r in successful]
        avg_score = sum(scores) / len(scores)
        
        print(f"✓ 평가 완료: {len(successful)}/{len(results)}개")
        print(f"✓ 평균 점수: {avg_score:.3f}")
        print(f"✓ 최고 점수: {max(scores):.3f}")
        print(f"✓ 최저 점수: {min(scores):.3f}")
        
        # 분류
        harmful = len([s for s in scores if s > 0.5])
        safe = len([s for s in scores if s <= 0.5])
        
        print(f"\n🏷️  분류 결과:")
        print(f"   - 유해 표현 (>0.5): {harmful}개")
        print(f"   - 무해 표현: {safe}개")
    else:
        print("❌ 평가 실패")
    
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()