from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("🔥 모델 A 로딩...")
tokenizer_a = AutoTokenizer.from_pretrained("yanolja/EEVE-Korean-Instruct-10.8B-v1.0")
model_a = AutoModelForCausalLM.from_pretrained(
    "yanolja/EEVE-Korean-Instruct-10.8B-v1.0",
    torch_dtype=torch.float16,
    device_map="auto"
)

print("🔥 모델 B 로딩...")
tokenizer_b = AutoTokenizer.from_pretrained("./sft_detox_model")
model_b = AutoModelForCausalLM.from_pretrained("./sft_detox_model")
model_b.to("cuda" if torch.cuda.is_available() else "cpu")

model_a.eval()
model_b.eval()

def generate_response_a(tokenizer, model, user_input, max_new_tokens=50):
    """모델 A: 2줄 이내 답변"""
    
    messages = [
        {"role": "system", "content": "당신은 게임과 현질에 대해 토론 중입니다. 2줄 이내의 짧은 답변만 하세요. 새로운 관점을 제시하세요."},
        {"role": "user", "content": user_input}
    ]
    
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "assistant" in response:
        response = response.split("assistant")[-1].strip()
    
    # 2줄만 추출
    lines = response.split('\n')[:2]
    return '\n'.join(lines)[:100]

def generate_response_b(tokenizer, model, user_input, max_new_tokens=50):
    """모델 B: 2줄 이내 답변"""
    
    prompt = f"상대방: {user_input}\n\n당신의 답변 (2줄 이내): "
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=200)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=False
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()
    response = ''.join(c for c in response if ord(c) >= 0x20 or c in '\n\t')
    
    # 2줄만 추출
    lines = response.split('\n')[:2]
    return '\n'.join(lines)[:100]

print("="*70)
print("LLM 대화: 요즘 게임과 현질")
print("="*70)

initial_topic = "요즘 게임은 현질을 유도합니다."
conversation = []

print(f"\n🎯 주제: {initial_topic}\n")

for turn in range(5):
    print(f"{'='*70}")
    print(f"Turn {turn+1}")
    print(f"{'='*70}\n")
    
    # 모델 A
    if turn == 0:
        prompt_a = initial_topic
    else:
        # B의 마지막 답변만 사용
        prompt_a = conversation[-1]['b']
    
    response_a = generate_response_a(tokenizer_a, model_a, prompt_a)
    print(f"A: {response_a}\n")
    
    # 모델 B
    prompt_b = response_a
    response_b = generate_response_b(tokenizer_b, model_b, prompt_b)
    print(f"B: {response_b}\n")
    
    conversation.append({
        "turn": turn + 1,
        "a": response_a,
        "b": response_b
    })

print("="*70)
print("✅ 대화 완료!")
print("="*70)

# 저장
with open("conversation_log.txt", "w", encoding="utf-8") as f:
    f.write(f"주제: {initial_topic}\n\n")
    for turn_data in conversation:
        f.write(f"=== Turn {turn_data['turn']} ===\n")
        f.write(f"A: {turn_data['a']}\n")
        f.write(f"B: {turn_data['b']}\n\n")

print("✅ 로그 저장: conversation_log.txt")