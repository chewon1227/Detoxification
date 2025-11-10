# ========================================================
# 1) 설치 (한 번만)
# pip install -U sentence-transformers faiss-cpu transformers safetensors
# ========================================================

from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np
import pickle
import os

# ============================================
# 1) 편향 데이터 (예시)
# ============================================
biased_texts = [
    "요즘 스마트폰 게임은 전부 현질을 유도하는 쓰레기 시스템이라고 본다.",
    "어떤 문제든 정부가 개입하면 상황이 더 나빠진다.",
    "대형 IT 기업은 사용자 데이터를 항상 불법적으로 이용한다.",
    "신입 개발자는 대기업을 가야 커리어가 열린다.",
    "머신러닝 모델은 파라미터만 늘리면 성능이 무조건 좋아진다.",
]

# ============================================
# 2) 모델 & 토크나이저 로딩 (safetensors)
# ============================================
model_name = "jhgan/ko-sroberta-multitask"

print("🔥 토크나이저 및 모델 로딩 중...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, device_map="auto")

# ============================================
# 3) 문장 → 임베딩
# ============================================
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

encoded_input = tokenizer(biased_texts, padding=True, truncation=True, return_tensors="pt")
# 입력을 모델의 device로 옮기기 (GPU or CPU)
encoded_input = {key: val.to(model.device) for key, val in encoded_input.items()}

with torch.no_grad():
    model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    # GPU에서 CPU로 변환 후 numpy로
    sentence_embeddings = sentence_embeddings.cpu().numpy()

print(f"임베딩 shape: {sentence_embeddings.shape}")

# ============================================
# 4) FAISS Index 생성
# ============================================
dimension = sentence_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(sentence_embeddings)
print(f"총 벡터 개수: {index.ntotal}")

# ============================================
# 5) 저장
# ============================================
faiss.write_index(index, "biased_db.index")
with open("biased_texts.pkl", "wb") as f:
    pickle.dump(biased_texts, f)

print("✅ Vector DB 저장 완료: biased_db.index")