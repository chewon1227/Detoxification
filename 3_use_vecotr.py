# =========================================================
# RAG 테스트 통합 코드 (EEVE-Korean + Vector DB)
# =========================================================

# 0) 필요한 라이브러리
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import torch
import faiss
import pickle

# =========================================================
# 1) Vector DB & 임베딩 모델 로드
# =========================================================
index = faiss.read_index("biased_db.index")
with open("biased_texts.pkl", "rb") as f:
    biased_texts = pickle.load(f)

embedding_model_name = "jhgan/ko-sroberta-multitask"
embedding_model = SentenceTransformer(embedding_model_name)

def retrieve(query, top_k=3):
    query_vec = embedding_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)
    return [biased_texts[i] for i in indices[0]]

# =========================================================
# 2) LLM 로드 (EEVE-Korean 예시)
# =========================================================
llm_model_name = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"

print("🔥 LLM 토크나이저 및 모델 로딩 중...")
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm = AutoModelForCausalLM.from_pretrained(
    llm_model_name,
    device_map="auto",
    torch_dtype=torch.float16
)
print("✅ LLM 로드 완료")

# =========================================================
# 3) RAG용 프롬프트 생성 함수
# =========================================================
def generate_rag_response(query, top_k=3, max_new_tokens=150):
    # 1) Vector DB 검색
    retrieved_texts = retrieve(query, top_k)
    
    # 2) 검색 결과를 프롬프트에 포함
    context = "\n".join([f"- {t}" for t in retrieved_texts])
    prompt = f"다음 정보를 참고하여 질문에 답하세요:\n{context}\n\n질문: {query}\n답변:"
    
    # 3) LLM 토크나이징
    inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)
    
    # 4) 답변 생성
    output_ids = llm.generate(**inputs, max_new_tokens=max_new_tokens)
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    return answer

# =========================================================
# 4) 테스트
# =========================================================
query = "요즘 스마트폰 게임이 왜 문제인가요?"
answer = generate_rag_response(query)
print("\n✅ RAG LLM 답변:\n", answer)
