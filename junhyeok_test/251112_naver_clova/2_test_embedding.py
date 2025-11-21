import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("CLOVA_API_KEY")
EMBEDDING_URL = "https://clovastudio.stream.ntruss.com/testapp/v1/api-tools/embedding/v2"

def test_embedding(texts):
    """
    CLOVA Studio 임베딩 v2 API 테스트
    
    Args:
        texts (list): 임베딩할 텍스트 리스트
    
    Returns:
        dict: API 응답 (임베딩 벡터 포함)
    """
    
    if not API_KEY:
        print("❌ API 키가 설정되지 않았습니다.")
        return None
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "text": texts[0] if isinstance(texts, list) else texts
    }
    
    try:
        print(f"📤 임베딩 요청: {len(texts)} 개 텍스트")
        print(f"   텍스트 미리보기: {texts[0][:50]}...")
        
        response = requests.post(
            EMBEDDING_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"📊 상태 코드: {response.status_code}\n")
        
        if response.status_code == 200:
            result = response.json()
            
            if "status" in result and result["status"]["code"] == "20000":
                embedding = result.get("result", {}).get("embedding", [])
                
                print(f"✅ 임베딩 성공!")
                print(f"   벡터 차원: {len(embedding)}")
                print(f"   벡터 미리보기 (처음 5개): {embedding[:5]}")
                
                # 토큰 사용량
                usage = result.get("result", {}).get("usage", {})
                if usage:
                    print(f"\n📊 토큰 사용량: {usage.get('totalTokens', 0)} 토큰")
                
                return result
            else:
                print(f"❌ API 에러: {result.get('status', {})}")
        else:
            print(f"❌ HTTP 에러: {response.status_code}")
            print(f"응답: {response.text}")
            
    except Exception as e:
        print(f"❌ 에러 발생: {e}")
    
    return None


def compare_embeddings(text1, text2):
    """
    두 텍스트의 임베딩 벡터 간 유사도 계산 (코사인 유사도)
    """
    import numpy as np
    
    print("\n" + "="*60)
    print("📊 임베딩 유사도 비교")
    print("="*60 + "\n")
    
    # 텍스트 1 임베딩
    result1 = test_embedding([text1])
    if not result1:
        return
    embedding1 = np.array(result1["result"]["embedding"])
    
    print("\n" + "-"*60 + "\n")
    
    # 텍스트 2 임베딩
    result2 = test_embedding([text2])
    if not result2:
        return
    embedding2 = np.array(result2["result"]["embedding"])
    
    # 코사인 유사도 계산
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    similarity = dot_product / (norm1 * norm2)
    
    print("\n" + "="*60)
    print(f"🎯 코사인 유사도: {similarity:.4f}")
    print("="*60)
    
    # 유사도 해석
    if similarity > 0.9:
        print("💡 해석: 매우 유사한 의미")
    elif similarity > 0.7:
        print("💡 해석: 유사한 의미")
    elif similarity > 0.5:
        print("💡 해석: 어느 정도 관련됨")
    else:
        print("💡 해석: 관련성 낮음")
    
    return similarity


def batch_embedding_demo():
    """
    배치 임베딩 데모 - RAG용 데이터 준비 시뮬레이션
    """
    print("\n" + "="*60)
    print("📦 배치 임베딩 데모 (RAG 데이터 준비)")
    print("="*60 + "\n")
    
    # 샘플 커뮤니티 데이터 (실제로는 크롤링한 데이터)
    community_posts = [
        "이번 사건의 주요 원인은 안전 관리 부실입니다.",
        "정부의 책임이 크다고 생각합니다.",
        "시민들의 안전 의식도 중요합니다.",
        "재발 방지를 위한 시스템 개선이 필요합니다.",
        "언론의 과도한 보도도 문제입니다."
    ]
    
    embeddings = []
    
    for i, post in enumerate(community_posts, 1):
        print(f"\n[{i}/{len(community_posts)}] 임베딩 중...")
        result = test_embedding([post])
        
        if result and "result" in result:
            embedding = result["result"]["embedding"]
            embeddings.append({
                "text": post,
                "embedding": embedding,
                "metadata": {"post_id": i}
            })
        
        print("-"*40)
    
    print(f"\n✅ 총 {len(embeddings)}개 임베딩 완료!")
    print("💾 실제 프로젝트에서는 이 데이터를 Vector DB에 저장합니다.")
    
    return embeddings


def main():
    """메인 함수"""
    
    if not API_KEY:
        print("⚠️ API 키를 설정해주세요!")
        return
    
    print("="*60)
    print("🧪 CLOVA Studio 임베딩 API 테스트")
    print("="*60)
    
    # 테스트 1: 단일 텍스트 임베딩
    print("\n[테스트 1] 단일 텍스트 임베딩")
    print("-"*60 + "\n")
    test_embedding(["안녕하세요. 이것은 테스트 문장입니다."])
    
    # 테스트 2: 유사도 비교 (유사한 문장)
    print("\n\n[테스트 2] 유사한 문장 비교")
    compare_embeddings(
        "이태원 참사의 원인은 안전 관리 부실입니다.",
        "이번 사건은 안전 관리가 제대로 되지 않아서 발생했습니다."
    )
    
    # 테스트 3: 유사도 비교 (다른 문장)
    print("\n\n[테스트 3] 다른 주제의 문장 비교")
    compare_embeddings(
        "이태원 참사의 원인은 안전 관리 부실입니다.",
        "오늘 날씨가 정말 좋네요."
    )
    
    # 테스트 4: 배치 임베딩
    print("\n\n[테스트 4] 배치 임베딩")
    batch_embedding_demo()
    
    print("\n" + "="*60)
    print("✅ 모든 테스트 완료!")
    print("="*60)
    print("\n💡 다음 단계: 이 임베딩 결과를 Vector DB (Chroma/FAISS)에 저장")


if __name__ == "__main__":
    main()
