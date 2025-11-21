"""
간단한 RAG (Retrieval-Augmented Generation) 프로토타입

이 코드는 Chroma DB를 사용하여 간단한 RAG 시스템을 구현합니다.
실제 프로젝트에서는 CLOVA Studio 임베딩 API를 사용해야 합니다.
"""

import os
import requests
from dotenv import load_dotenv

# Chroma DB를 사용하려면 먼저 설치: pip install chromadb
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    print("⚠️ Chroma DB가 설치되지 않았습니다.")
    print("설치 명령: pip install chromadb")
    CHROMA_AVAILABLE = False

load_dotenv()
API_KEY = os.getenv("CLOVA_API_KEY")


class SimpleRAGAgent:
    """간단한 RAG 에이전트"""
    
    def __init__(self, collection_name, use_clova_embedding=False):
        """
        Args:
            collection_name (str): 컬렉션 이름 (예: "community_L", "community_R")
            use_clova_embedding (bool): CLOVA 임베딩 사용 여부 (False면 기본 임베딩)
        """
        self.collection_name = collection_name
        self.use_clova_embedding = use_clova_embedding
        
        if not CHROMA_AVAILABLE:
            raise ImportError("Chroma DB를 먼저 설치하세요: pip install chromadb")
        
        # Chroma DB 클라이언트 초기화
        self.client = chromadb.Client(Settings(
            anonymized_telemetry=False,
            allow_reset=True
        ))
        
        # 컬렉션 생성 또는 가져오기
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"✅ 기존 컬렉션 '{collection_name}' 로드됨")
        except:
            self.collection = self.client.create_collection(name=collection_name)
            print(f"✅ 새 컬렉션 '{collection_name}' 생성됨")
    
    def add_documents(self, documents, metadatas=None):
        """
        문서를 Vector DB에 추가
        
        Args:
            documents (list): 문서 텍스트 리스트
            metadatas (list): 메타데이터 리스트 (선택)
        """
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        if metadatas is None:
            metadatas = [{"source": "community"} for _ in documents]
        
        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        print(f"📦 {len(documents)}개 문서 추가 완료")
    
    def search_similar(self, query, top_k=3):
        """
        유사한 문서 검색
        
        Args:
            query (str): 검색 쿼리
            top_k (int): 반환할 문서 수
        
        Returns:
            list: 유사 문서 리스트
        """
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        # 결과 포맷팅
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        
        similar_docs = []
        for doc, metadata, dist in zip(documents, metadatas, distances):
            similar_docs.append({
                "text": doc,
                "metadata": metadata,
                "distance": dist,
                "similarity": 1 - dist  # 거리를 유사도로 변환
            })
        
        return similar_docs
    
    def generate_response(self, query):
        """
        RAG 방식으로 응답 생성
        
        1. 쿼리와 유사한 문서 검색
        2. 검색된 문서를 컨텍스트로 LLM에 전달
        3. LLM이 컨텍스트 기반 답변 생성
        
        Args:
            query (str): 사용자 질문
        
        Returns:
            dict: 응답 및 참고 문서
        """
        # 1. 유사 문서 검색
        similar_docs = self.search_similar(query, top_k=3)
        
        print(f"\n🔍 검색된 참고 문서 ({len(similar_docs)}개):")
        for i, doc in enumerate(similar_docs, 1):
            print(f"  [{i}] (유사도: {doc['similarity']:.3f}) {doc['text'][:60]}...")
        
        # 2. 컨텍스트 구성
        context = "\n\n".join([f"[참고 {i+1}] {doc['text']}" 
                               for i, doc in enumerate(similar_docs)])
        
        # 3. LLM에 전달할 프롬프트 구성
        prompt = f"""다음 참고 자료를 바탕으로 질문에 답변해주세요.

참고 자료:
{context}

질문: {query}

답변:"""
        
        # 4. CLOVA Chat API 호출
        if not API_KEY:
            print("⚠️ API 키가 없어서 실제 LLM 호출은 건너뜁니다.")
            return {
                "query": query,
                "context_docs": similar_docs,
                "response": "[API 키 필요 - 실제 응답은 여기 생성됨]"
            }
        
        try:
            response = requests.post(
                "https://clovastudio.stream.ntruss.com/testapp/v1/chat-completions/HCX-003",
                headers={
                    "Authorization": f"Bearer {API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "messages": [
                        {"role": "system", "content": "당신은 제공된 참고 자료를 바탕으로 정확하게 답변하는 AI입니다."},
                        {"role": "user", "content": prompt}
                    ],
                    "maxTokens": 512,
                    "temperature": 0.3  # 낮은 온도로 일관된 답변
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("status", {}).get("code") == "20000":
                    ai_response = result["result"]["message"]["content"]
                    
                    return {
                        "query": query,
                        "context_docs": similar_docs,
                        "response": ai_response,
                        "usage": result["result"].get("usage", {})
                    }
            
            print(f"⚠️ API 호출 실패: {response.status_code}")
            
        except Exception as e:
            print(f"⚠️ API 호출 에러: {e}")
        
        return {
            "query": query,
            "context_docs": similar_docs,
            "response": "[API 호출 실패]"
        }


def demo_rag_system():
    """RAG 시스템 데모"""
    
    print("="*60)
    print("🤖 간단한 RAG 시스템 데모")
    print("="*60 + "\n")
    
    # 좌측 성향 커뮤니티 데이터 (시뮬레이션)
    community_L_docs = [
        "이번 사건은 정부의 안전 관리 부실이 주요 원인입니다. 사전 예방 조치가 미흡했습니다.",
        "책임자들의 문책이 필요합니다. 시스템적 개선이 이루어져야 합니다.",
        "시민들의 안전이 최우선이어야 합니다. 재발 방지 대책이 시급합니다.",
        "투명한 진상 조사가 필요합니다. 국민들에게 정확한 정보가 제공되어야 합니다.",
    ]
    
    # 우측 성향 커뮤니티 데이터 (시뮬레이션)
    community_R_docs = [
        "이번 사건은 현장 관리의 문제입니다. 개인의 안전 의식도 중요합니다.",
        "언론의 과도한 정치화가 문제 해결을 방해하고 있습니다.",
        "실무자들의 헌신에도 주목해야 합니다. 무조건적인 비난은 지양해야 합니다.",
        "객관적인 분석이 필요합니다. 감정적 대응보다 합리적 대책을 세워야 합니다.",
    ]
    
    # RAG 에이전트 생성
    print("1️⃣ RAG 에이전트 생성")
    print("-"*60 + "\n")
    
    rag_L = SimpleRAGAgent("community_L")
    rag_R = SimpleRAGAgent("community_R")
    
    # 문서 추가
    print("\n2️⃣ 커뮤니티 데이터 로딩")
    print("-"*60 + "\n")
    
    rag_L.add_documents(community_L_docs, 
                        [{"community": "L", "idx": i} for i in range(len(community_L_docs))])
    rag_R.add_documents(community_R_docs,
                        [{"community": "R", "idx": i} for i in range(len(community_R_docs))])
    
    # 테스트 쿼리
    test_query = "이번 사건의 주요 원인은 무엇인가요?"
    
    print(f"\n3️⃣ 테스트 쿼리: '{test_query}'")
    print("="*60 + "\n")
    
    # RAG_L 응답
    print("📍 RAG_L (좌측 성향 데이터) 응답:")
    print("-"*60)
    result_L = rag_L.generate_response(test_query)
    print(f"\n🤖 응답:\n{result_L['response']}\n")
    
    # RAG_R 응답
    print("\n📍 RAG_R (우측 성향 데이터) 응답:")
    print("-"*60)
    result_R = rag_R.generate_response(test_query)
    print(f"\n🤖 응답:\n{result_R['response']}\n")
    
    print("="*60)
    print("✅ RAG 시스템 데모 완료!")
    print("="*60)
    print("\n💡 실제 프로젝트에서는:")
    print("   1. 크롤링한 실제 커뮤니티 데이터 사용")
    print("   2. CLOVA 임베딩 API로 더 정확한 검색")
    print("   3. HCX_Detox 모델로 순화된 응답 생성")


if __name__ == "__main__":
    if not CHROMA_AVAILABLE:
        print("\n먼저 Chroma DB를 설치하세요:")
        print("pip install chromadb")
    else:
        demo_rag_system()
