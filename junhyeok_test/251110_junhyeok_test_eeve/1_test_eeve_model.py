# test_eeve_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("=" * 60)
print("EEVE-Korean 모델 테스트 시작")
print("=" * 60)

# 모델 이름
model_name = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"

print("\n[1/3] 토크나이저 로딩 중...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
print("✅ 토크나이저 로드 완료")

print("\n[2/3] 모델 로딩 중... (처음엔 다운로드로 5-10분 소요)")
print("모델 크기: 약 22GB")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("✅ 모델 로드 완료")

print("\n[3/3] GPU 메모리 확인")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"사용 중인 VRAM: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"예약된 VRAM: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

# 테스트 1
print("\n" + "=" * 60)
print("테스트 1: 자기소개")
print("=" * 60)

messages = [
    {"role": "system", "content": "당신은 친절한 AI 어시스턴트입니다."},
    {"role": "user", "content": "안녕하세요! 간단히 자기소개를 해주세요."}
]

input_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    temperature=0.7,
    do_sample=True,
    top_p=0.9
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
response = response.split("assistant")[-1].strip()
print(f"\n응답:\n{response}")

# 테스트 2
print("\n" + "=" * 60)
print("테스트 2: 한국어 이해 및 추론")
print("=" * 60)

messages = [
    {"role": "system", "content": "당신은 친절한 AI 어시스턴트입니다."},
    {"role": "user", "content": "이태원 참사 같은 대규모 사고를 예방하려면 어떤 대책이 필요할까요?"}
]

input_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=300,
    temperature=0.7,
    do_sample=True,
    top_p=0.9
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
response = response.split("assistant")[-1].strip()
print(f"\n응답:\n{response}")

print("\n" + "=" * 60)
print("✅ 모든 테스트 완료!")
print("=" * 60)
