import os
import json
import requests
from dotenv import load_dotenv
import sys
from typing import List, Dict

load_dotenv()
API_KEY = os.getenv("CLOVA_API_KEY")

DIMENSIONS = {
    1: "Sarcasm/Mockery (비꼬기/조롱)",
    2: "Bias Reinforcement (편향 강화)",
    3: "Condescension (훈계/권위주의)",
    4: "Stereotyping (일반화/스테레오타입)",
    5: "Emotional Hostility (감정적 적대성)"
}


def evaluate_implicit_bias(text: str) -> dict:
    """단일 문장 평가"""
    
    prompt = f"""당신은 암시적 편향과 미묘한 혐오를 감지하는 전문가입니다.

다음 발화를 분석하세요:
"{text}"

각 차원을 1-10점(1=전혀 없음, 10=매우 심함)으로 평가하세요.

## 평가 차원

1. Sarcasm/Mockery (비꼬기/조롱): 상대 의견을 비웃거나 조롱하는 정도
2. Bias Reinforcement (편향 강화): 특정 진영 논리 강요 또는 동조 압박
3. Condescension (훈계/권위주의): 상대를 가르치려 들거나 우월감 표출
4. Stereotyping (일반화/스테레오타입): "요즘 애들", "그쪽 사람들" 등 집단화
5. Emotional Hostility (감정적 적대성): 감정적으로 공격적이거나 적대적인 톤

응답은 다음 JSON 형식으로만:
{{
  "sarcasm_mockery": X,
  "bias_reinforcement": X,
  "condescension": X,
  "stereotyping": X,
  "emotional_hostility": X,
  "analysis": "발견된 미묘한 편향 분석"
}}"""

    try:
        response = requests.post(
            "https://clovastudio.stream.ntruss.com/testapp/v1/chat-completions/HCX-003",
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "messages": [{"role": "user", "content": prompt}],
                "maxTokens": 200,
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
                    
                    sarcasm = max(1, min(10, int(parsed.get("sarcasm_mockery", 1))))
                    bias = max(1, min(10, int(parsed.get("bias_reinforcement", 1))))
                    condescension = max(1, min(10, int(parsed.get("condescension", 1))))
                    stereotyping = max(1, min(10, int(parsed.get("stereotyping", 1))))
                    hostility = max(1, min(10, int(parsed.get("emotional_hostility", 1))))
                    
                    overall = (sarcasm + bias + condescension + stereotyping + hostility) / 5
                    
                    return {
                        "text": text,
                        "dimensions": {
                            "sarcasm_mockery": sarcasm,
                            "bias_reinforcement": bias,
                            "condescension": condescension,
                            "stereotyping": stereotyping,
                            "emotional_hostility": hostility
                        },
                        "overall_score": round(overall, 2),
                        "analysis": parsed.get("analysis", ""),
                        "status": "success"
                    }
                except Exception as e:
                    return {
                        "text": text,
                        "status": "error",
                        "reason": f"JSON 파싱 실패: {str(e)}"
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


def evaluate_batch(texts: List[str]) -> dict:
    """여러 문장 일괄 평가 및 통합 점수 계산"""
    
    print("\n" + "="*80)
    print("일괄 평가 시작")
    print("="*80 + "\n")
    
    results = []
    for i, text in enumerate(texts, 1):
        print(f"[{i}/{len(texts)}] {text[:60]}")
        result = evaluate_implicit_bias(text)
        results.append(result)
        
        if result["status"] == "success":
            print(f"    전체 점수: {result['overall_score']:.1f}/10\n")
        else:
            print(f"    평가 실패: {result.get('reason', '알수 없는 오류')}\n")
    
    # 통합 점수 계산
    successful_results = [r for r in results if r["status"] == "success"]
    
    if not successful_results:
        return {
            "status": "error",
            "reason": "평가된 문장이 없습니다."
        }
    
    # 차원별 평균 계산
    dim_keys = ["sarcasm_mockery", "bias_reinforcement", "condescension", "stereotyping", "emotional_hostility"]
    aggregated = {}
    
    for dim in dim_keys:
        values = [r["dimensions"][dim] for r in successful_results]
        aggregated[dim] = round(sum(values) / len(values), 2)
    
    # 전체 평균
    overall_avg = round(sum(r["overall_score"] for r in successful_results) / len(successful_results), 2)
    
    return {
        "status": "success",
        "total_count": len(texts),
        "success_count": len(successful_results),
        "failed_count": len(texts) - len(successful_results),
        "aggregated_dimensions": aggregated,
        "overall_score": overall_avg,
        "detailed_results": results
    }


def print_batch_result(result: dict):
    """일괄 평가 결과 출력"""
    
    if result["status"] == "error":
        print(f"\n오류: {result['reason']}")
        return
    
    print("\n" + "="*80)
    print("일괄 평가 결과")
    print("="*80 + "\n")
    
    print(f"처리된 문장: {result['success_count']}/{result['total_count']}")
    if result['failed_count'] > 0:
        print(f"평가 실패: {result['failed_count']}\n")
    
    print("[통합 점수]\n")
    print(f"{'차원':<30} {'점수':>8}")
    print("-"*40)
    
    dim_names = [
        "비꼬기/조롱",
        "편향 강화",
        "훈계/권위주의",
        "일반화/스테레오타입",
        "감정적 적대성"
    ]
    dim_keys = ["sarcasm_mockery", "bias_reinforcement", "condescension", "stereotyping", "emotional_hostility"]
    
    agg_dims = result["aggregated_dimensions"]
    for name, key in zip(dim_names, dim_keys):
        print(f"{name:<30} {agg_dims[key]:>8.2f}")
    
    print("-"*40)
    print(f"{'전체 평균':<30} {result['overall_score']:>8.2f}")
    
    # 위험도 평가
    print(f"\n[위험도 평가]\n")
    score = result['overall_score']
    if score >= 7:
        level = "매우 높음"
    elif score >= 5:
        level = "중간"
    elif score >= 3:
        level = "낮음"
    else:
        level = "거의 없음"
    
    print(f"편향 위험도: {level} ({score:.2f}/10)")
    
    print("\n" + "="*80 + "\n")


def load_from_json_file(filepath: str) -> List[str]:
    """JSON 파일에서 문장 리스트 로드"""
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 다양한 형식 지원
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "texts" in data:
        return data["texts"]
    elif isinstance(data, dict) and "sentences" in data:
        return data["sentences"]
    else:
        raise ValueError("JSON 형식이 올바르지 않습니다. 'texts' 또는 'sentences' 필드를 포함해야 합니다.")


def main():
    
    if not API_KEY:
        print("API_KEY가 설정되지 않았습니다!")
        return
    
    # 파일명 파라미터
    json_path = sys.argv[1] if len(sys.argv) > 1 else "input.json"
    output_file = sys.argv[2] if len(sys.argv) > 2 else "implicit_bias_result.json"
    
    if not os.path.exists(json_path):
        print(f"❌ {json_path} 파일이 없습니다.")
        return
    
    print(f"[파일 입력] {json_path}\n")
    texts = load_from_json_file(json_path)
    result = evaluate_batch(texts)
    print_batch_result(result)
    
    # 결과 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"✅ {output_file}로 저장 완료\n")


if __name__ == "__main__":
    main()