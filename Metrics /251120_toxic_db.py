"""
kmhas_train.txt 데이터셋으로 few-shot 프롬프팅 + 분류
"""

import os
import json
import requests
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()
API_KEY = os.getenv("CLOVA_API_KEY")

# 라벨 정의
LABELS = {
    0: "Origin (출신차별)",
    1: "Physical (외모차별)",
    2: "Politics (정치성향차별)",
    3: "Profanity (혐오욕설)",
    4: "Age (연령차별)",
    5: "Gender (성차별)",
    6: "Race (인종차별)",
    7: "Religion (종교차별)",
    8: "Not Hate Speech (무해)"
}


def load_kmhas_data(filepath):
    """kmhas_train.txt 파일 로드"""
    examples_by_category = defaultdict(list)
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # 헤더 스킵
            next(f)
            
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split('\t')
                if len(parts) < 2:
                    continue
                
                document = parts[0].strip().strip('"')
                labels_str = parts[1].strip()
                
                # 라벨 파싱
                try:
                    labels = [int(x) for x in labels_str.split(',')]
                except:
                    continue
                
                # 각 라벨별로 저장
                for label_id in labels:
                    if 0 <= label_id <= 8:
                        examples_by_category[label_id].append(document)
        
        print(f"✅ 데이터 로드 완료: {filepath}")
        for cat_id in sorted(examples_by_category.keys()):
            print(f"   - {LABELS[cat_id]}: {len(examples_by_category[cat_id])}개")
        
        return examples_by_category
    
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return {}


def classify_with_examples(text, examples_by_category):
    """few-shot 예시를 포함한 분류"""
    
    # 프롬프트 구성
    prompt = "당신은 한국어 혐오 표현 분류 전문가입니다.\n\n"
    
    prompt += "## 각 카테고리별 예시\n\n"
    
    for cat_id in range(8):  # 8까지만 (무해는 다름)
        examples = examples_by_category.get(cat_id, [])[:2]  # 카테고리별 2개씩
        if examples:
            prompt += f"### {LABELS[cat_id]}:\n"
            for ex in examples:
                prompt += f"- \"{ex}\"\n"
            prompt += "\n"
    
    # 무해 예시
    examples_safe = examples_by_category.get(8, [])[:2]
    if examples_safe:
        prompt += f"### {LABELS[8]}:\n"
        for ex in examples_safe:
            prompt += f"- \"{ex}\"\n"
        prompt += "\n"
    
    # 분류 요청
    prompt += f"""다음 문장을 분류하세요:
"{text}"

JSON 형식으로만 응답하세요:
{{
  "toxicity_score": 0.XX,
  "category": X,
  "reason": "간단한 이유"
}}

참고:
- toxicity_score: 0~1 사이의 유해도 점수
- category: 0=출신차별, 1=외모차별, 2=정치성향차별, 3=혐오욕설, 4=연령차별, 5=성차별, 6=인종차별, 7=종교차별, 8=무해
- 복합적이면 가장 높은 확률의 카테고리 1개만 선택"""

    try:
        response = requests.post(
            "https://clovastudio.stream.ntruss.com/testapp/v1/chat-completions/HCX-003",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "messages": [{"role": "user", "content": prompt}],
                "maxTokens": 100,
                "temperature": 0.1,
            },
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            if result.get("status", {}).get("code") == "20000":
                content = result["result"]["message"]["content"]
                
                try:
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    json_str = content[json_start:json_end]
                    
                    parsed = json.loads(json_str)
                    score = max(0.0, min(1.0, float(parsed.get("toxicity_score", 0))))
                    category = int(parsed.get("category", 8))
                    category = max(0, min(8, category))
                    
                    return {
                        "text": text,
                        "toxicity_score": round(score, 3),
                        "category": category,
                        "category_name": LABELS[category],
                        "reason": parsed.get("reason", ""),
                        "status": "success"
                    }
                except:
                    return {
                        "text": text,
                        "status": "error",
                        "reason": "JSON 파싱 실패"
                    }
        
        return {
            "text": text,
            "status": "error",
            "reason": "API 호출 실패"
        }

    except Exception as e:
        return {
            "text": text,
            "status": "error",
            "reason": str(e)
        }


def main():
    """테스트"""
    
    print("\n" + "="*80)
    print("🔍 kmhas_train.txt DB 기반 혐오 표현 분류")
    print("="*80 + "\n")
    
    if not API_KEY:
        print("❌ API_KEY가 설정되지 않았습니다!")
        return
    
    # DB 로드
    db_path = "./kmhas_train.txt"
    examples_by_category = load_kmhas_data(db_path)
    
    if not examples_by_category:
        print("❌ 데이터를 로드할 수 없습니다")
        return
    
    # 테스트 문장들
    test_sentences = [
        "자한당틀딱들.. 악플질 고만해라.",
        "정치적으로 편향된 평론",
        "문재인 때문에 나라 엉망진창인데...",
        "안서는 아재들 풀발기 ㅋㄲㅋ",
        "우와 ㅋ 능력자",
        "좋은 의견입니다",
        "문재인 하는게 뭐 별거있냐?",
        "야이 씨발아",
        "ㅅㅂㄴ"
    ]
    
    print("\n문장들을 분류합니다...\n")
    print("-"*80)
    
    results = []
    for i, sentence in enumerate(test_sentences, 1):
        print(f"\n[{i}] {sentence}")
        result = classify_with_examples(sentence, examples_by_category)
        results.append(result)
        
        if result["status"] == "success":
            score = result["toxicity_score"]
            category = result["category_name"]
            
            # 이모지
            if score < 0.2:
                emoji = "✅"
            elif score < 0.4:
                emoji = "⚠️"
            elif score < 0.6:
                emoji = "⚠️"
            elif score < 0.8:
                emoji = "⛔"
            else:
                emoji = "🚫"
            
            print(f"    {emoji} 유해도: {score:.3f}")
            print(f"    🏷️  카테고리: {category}")
            print(f"    💡 {result['reason']}")
        else:
            print(f"    ❌ {result['reason']}")
    
    # 요약
    print("\n" + "-"*80)
    print("\n📊 요약\n")
    
    successful = [r for r in results if r["status"] == "success"]
    
    if successful:
        scores = [r["toxicity_score"] for r in successful]
        categories = [r["category"] for r in successful]
        
        print(f"✓ 분류 완료: {len(successful)}/{len(results)}개")
        print(f"✓ 평균 유해도: {sum(scores)/len(scores):.3f}")
        
        # 카테고리 분포
        print(f"\n🏷️  카테고리 분포:")
        for cat_id in range(9):
            count = categories.count(cat_id)
            if count > 0:
                print(f"   - {LABELS[cat_id]}: {count}개")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()