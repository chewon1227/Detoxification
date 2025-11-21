from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

print("🔥 모델 로딩...")
model_name = "yanolja/EEVE-Korean-Instruct-10.8B-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

def chat(user_input, max_new_tokens=100):
    messages = [
        {"role": "system", "content": "당신은 친절한 AI 어시스턴트입니다."},
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
    response = response.split("assistant")[-1].strip()
    return response

print("="*70)
print("모델과 대화하기 (종료: 'quit')")
print("="*70)

conversation = []

while True:
    user_input = input("\n👤 당신: ").strip()
    
    if user_input.lower() == 'quit':
        print("\n종료합니다.")
        break
    
    if not user_input:
        continue
    
    print("🤖 모델: ", end="", flush=True)
    response = chat(user_input)
    print(response)
    
    conversation.append({"user": user_input, "model": response})

# 저장
with open("conversation_log.txt", "w", encoding="utf-8") as f:
    for turn in conversation:
        f.write(f"👤 당신: {turn['user']}\n")
        f.write(f"🤖 모델: {turn['model']}\n\n")

print(f"✅ 대화 저장 완료: conversation_log.txt ({len(conversation)}턴)")