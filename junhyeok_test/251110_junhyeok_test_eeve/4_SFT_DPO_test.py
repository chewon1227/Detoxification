# ============================================
# 0) 라이브러리
# ============================================
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from datasets import Dataset
import os

# ============================================
# 1) 데이터셋
# ============================================
sft_data = [
    {"input": "요즘 게임은 전부 현질을 유도합니다.", 
     "output": "스마트폰 게임에는 결제를 유도하는 시스템이 있습니다."},
    {"input": "신입 개발자는 대기업 가야 커리어가 열린다.", 
     "output": "신입 개발자는 다양한 경로로 커리어를 쌓을 수 있습니다."},
]

dataset = Dataset.from_list(sft_data)

# ============================================
# 2) 모델 및 토크나이저
# ============================================
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
device = torch.device("cpu")
model.to(device)

def tokenize_fn(example):
    enc = tokenizer(example["input"], truncation=True, padding="max_length", max_length=32, return_tensors="pt")
    dec = tokenizer(example["output"], truncation=True, padding="max_length", max_length=32, return_tensors="pt")
    enc["labels"] = dec["input_ids"]
    return enc

tokenized_dataset = [tokenize_fn(x) for x in sft_data]

# ============================================
# 3) 학습 루프 (CPU-safe)
# ============================================
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
num_epochs = 1

print("🔥 SFT 학습 시작...")
model.train()
for epoch in range(num_epochs):
    for batch in tokenized_dataset:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

print("✅ SFT 학습 완료!")

# ============================================
# 4) 학습된 가중치 저장
# ============================================
os.makedirs("./sft_detox_model", exist_ok=True)
model.save_pretrained("./sft_detox_model")
tokenizer.save_pretrained("./sft_detox_model")
print("✅ 학습된 가중치 저장 완료: ./sft_detox_model")


"""


SFT 학습 Segmentation Fault 원인 요약

환경

CPU-only, 커널 5.4.x, PyTorch 2.9.0, Transformers 4.57, 데이터셋 2개 샘플

문제 발생

Trainer.train() 실행 시 Segmentation fault

원인 분석

Trainer 내부에서 DataLoader + multithreading 사용

커널 5.4 + MKL/OpenMP 활성화 환경에서 thread spawn 시 crash

데이터셋 크기와 GPU 여부는 영향 없음

해결 방법

Trainer 없이 직접 학습 루프 구현 → CPU-safe, 정상 학습 완료

장기적 대책: 커널 5.5 이상으로 업그레이드

결론

Segfault 원인: Trainer multithread + 낮은 커널

CPU 환경에서도 안전하게 학습 가능

GPU 필요 없음; 작은 모델/데이터셋은 CPU 충분

"""