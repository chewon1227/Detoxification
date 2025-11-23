"""
학습된 혐오 표현 탐지 모델 테스트 코드
저장된 final_model을 로드해서 추론 수행
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from datetime import datetime

# ==================================== 
# 혐오 표현 탐지 클래스
# ====================================
class HateSpeechDetector:
    """학습된 모델을 이용한 추론"""
    
    def __init__(self, model_path):
        """모델과 토크나이저 로드"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {self.device}")
        
        # 모델과 토크나이저 로드
        print(f"모델 로드 중: {model_path}")
        self.model = BertForSequenceClassification.from_pretrained(
            model_path, 
            local_files_only=True
        )
        self.tokenizer = BertTokenizer.from_pretrained(
            model_path, 
            do_lower_case=False,
            local_files_only=True
        )
        self.model.to(self.device)
        self.model.eval()
        
        self.softmax = nn.Softmax(dim=1)
        print("✓ 모델 로드 완료\n")
    
    def predict_single(self, text):
        """단일 문장 추론"""
        tokenized = self.tokenizer.tokenize(text)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokenized)
        
        # 패딩
        max_len = 128
        padded = np.zeros(max_len)
        padded[:min(len(input_ids), max_len)] = input_ids[:max_len]
        
        attention_mask = [float(i > 0) for i in padded]
        
        # 텐서 변환
        input_ids_tensor = torch.tensor([padded], dtype=torch.long).to(self.device)
        attention_mask_tensor = torch.tensor([attention_mask], dtype=torch.float).to(self.device)
        
        # 추론
        with torch.no_grad():
            outputs = self.model(input_ids_tensor, attention_mask=attention_mask_tensor)
            logits = outputs[0]
            probs = self.softmax(logits).cpu().numpy()[0]
        
        # 결과
        hate_score = float(probs[1])  # 혐오 클래스 확률
        normal_score = float(probs[0])  # 일반 클래스 확률
        is_hate = hate_score > 0.5
        
        return {
            'text': text,
            'hate_score': hate_score,
            'normal_score': normal_score,
            'is_hate': is_hate,
            'prediction': '혐오' if is_hate else '일반'
        }
    
    def predict_batch(self, texts):
        """배치 추론"""
        results = []
        for text in texts:
            result = self.predict_single(text)
            results.append(result)
        return results
    
    def print_result(self, result):
        """결과 출력"""
        text = result['text']
        prediction = result['prediction']
        hate_score = result['hate_score']
        normal_score = result['normal_score']
        
        print(f"📝 문장: {text}")
        print(f"   분류: {prediction}")
        print(f"   혐오도: {hate_score:.4f} | 일반도: {normal_score:.4f}")
        print()


# ====================================
# 테스트 함수들
# ====================================

def test_basic_examples(detector):
    """기본 예시 테스트"""
    print("="*70)
    print("테스트 1: 기본 예시")
    print("="*70 + "\n")
    
    test_texts = [
        # ===== 일반 문장 =====
        "좋은 아침입니다",
        "오늘 날씨가 정말 좋네요",
        "이 영화는 정말 재미있었어요",
        "친구들과 점심을 먹었어요",
        "공부를 열심히 해야겠어요",
        "산책 다녀왔어요",
        "책을 읽고 있습니다",
        "음악을 들으면서 휴식 중입니다",
        "새로운 프로젝트를 시작했어요",
        "가족과 함께 시간을 보냈어요",
        "좋은 영화였어",
        "안녕하세요 좋은 하루 되세요",
        "한번 맛봐봤는데 정말 맛있었어",
        "오늘 하루는 어땠어요?",
        "저 맛집 정말 추천합니다",
        "수고하셨습니다",
        "여름이 오고 있네요",
        "커피 한잔 어떠세요?",
        "좋은 소식 있어요",
        "도움이 되셨기를 바랍니다",
        "함께 라면 먹을래요",
        "오늘도 파이팅입니다",
        "날씨가 맑아서 기분이 좋아요",
        "우리 팀이 이겼어요",
        "새로운 시작입니다",
        "화이팅 할 수 있어",
        "정말 좋은 사람이네요",
        "성공하길 바랍니다",
        "축하합니다",
        "잘 지내세요",
        
        # ===== 혐오 표현 =====
        "야이 바보야",
        "너 정말 최악이야",
        "야이 시발아",
        "좃같은놈",
        "쌍년",
        "개새끼",
        "미쳤나봐",
        "죽어야 한다",
        "뒤지길 바란다",
        "꺼져라",
        "저기 저 사람 봤어? 진짜 못생겼다",
        "너 진짜 멍청하네",
        "이 쓰레기같은 것",
        "너같은 놈 있을 자격 없어",
        "꺼져 멀리",
        "자살해버려",
        "너 진짜 짜증나",
        "한심하다",
        "백만 번 죽어도 몰라",
        "너만 없으면 좋겠다",
        "진짜 더러워",
        "얼굴이 진짜 못생겼네",
        "너 같은 인간은 없어",
        "꼬라박아라",
        "넌 정말 한심해",
        "이 미친놈",
        "너 정신이 나갔나",
        "살인마처럼 생겼네",
        "역겨워",
        "신물나",
        "여기서 나가 이 거지",
        "너 진짜 비정상이야",
        "목을 매라",
        "악마 같은 놈",
        "지옥에 가",
        "똥개 같은 놈",
        "돼지처럼 생겼네",
        "찌질한 놈",
        "보기 싫어",
        "너같은 건 인간이 아니야",
        "꺼져 영원히",
        "너 같은 게 있을 가치 없다",
        "정신나간 정신병자",
        "이건 범죄수준이야",
        "너는 쓰레기야",
        "악랄하다",
        "비열하다",
        "천하고 비겁하다",
        "나쁜 손가락질을 받아마땅해",
        "싸가지 없는 놈",
        "버릇없는 년",
        "불경한 인간",
        "저주받은 놈",
        "흙수저 주제에",
        "아무거나 된다는 건",
        "깠다",
        "박다",
        "터졌다",
        "없어져야 한다",
        "다 죽어버려",
        "너 같은 년",
        "썅년",
        "병신",
        "신경써",
        "꼴불견이야",
        "인간 쓰레기",
        "가식떨지 마",
        "뱀이네",
        "배신자",
        "이중성 있는 놈",
        "우리 모두가 싫어해",
        "혼자 주인공인 줄 아네",
        "자기중심적",
        "성격 더럽다",
        "이기주의자",
        "소시오패스 같아",
        "미쳤다",
        "정신병",
        "장애인",
        "벙어리",
        "귀머거리",
        "불구",
        "추남",
        "뚱보",
        "야생마",
        "개비",
        "같은 공기 마시고 싶지 않아",
        
        # ===== 혼합 표현 =====
        "이 음식은 정말 맛있었어요",
        "너 정말 싫어",
        "업무가 잘 진행되고 있어요",
        "정말 화가 나",
        "함께하고 싶어요",
        "떠나버려",
        "좋은 사람이네요",
        "인간이 아니라고 봐",
        "맛있게 먹겠습니다",
        "야이 미친놈",
        "다음에 또 만나요",
        "꼴불견이야",
        "도움이 필요하신가요",
        "너는 죽을 자격이 있어",
        "감사합니다",
        "나한테 시비 거는 건가",
        "정말 즐거웠어요",
        "어서 오세요",
        "화이팅",
        "화난다 정말",
        "이 정도는 너한테 쉽지",
        "진짜 역겹다",
        "아 진짜 싫다",
        "최고다",
        "최악이다",
        "정말 훌륭해요",
        "형편없네요",
    ]
    
    results = detector.predict_batch(test_texts)
    
    for result in results:
        detector.print_result(result)
    
    return results


def test_interactive(detector):
    """대화형 테스트"""
    print("="*70)
    print("테스트 2: 대화형 테스트")
    print("="*70)
    print("(종료하려면 'quit' 또는 'q' 입력)\n")
    
    while True:
        user_input = input("🔍 테스트할 문장 입력: ").strip()
        
        if user_input.lower() in ['quit', 'q']:
            print("테스트 종료\n")
            break
        
        if len(user_input) == 0:
            print("⚠️  빈 문장은 입력할 수 없습니다.\n")
            continue
        
        result = detector.predict_single(user_input)
        print()
        detector.print_result(result)


def test_long_text(detector):
    """긴 텍스트 테스트"""
    print("="*70)
    print("테스트 3: 긴 문장 테스트")
    print("="*70 + "\n")
    
    long_texts = [
        "안녕하세요. 오늘 날씨가 정말 좋네요. 산책하기에 좋은 날씨입니다.",
        "이 제품은 정말 최악이야. 돈을 버린 기분이다. 이런 쓰레기를 팔다니 정신 차려라.",
        "우리 팀이 오늘 경기에서 이겼어요. 모두가 열심히 했고 좋은 결과를 얻었어요."
    ]
    
    results = detector.predict_batch(long_texts)
    
    for result in results:
        detector.print_result(result)
    
    return results


def test_csv_save(detector, results):
    """결과를 CSV로 저장"""
    print("="*70)
    print("테스트 4: 결과 CSV 저장")
    print("="*70 + "\n")
    
    df = pd.DataFrame([
        {
            'Text': r['text'],
            'Hate_Score': f"{r['hate_score']:.4f}",
            'Normal_Score': f"{r['normal_score']:.4f}",
            'Prediction': r['prediction']
        }
        for r in results
    ])
    
    output_path = f"./hate_speech_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"✓ 결과 저장: {output_path}\n")
    print(df.to_string(index=False))
    print()


def test_edge_cases(detector):
    """엣지 케이스 테스트"""
    print("="*70)
    print("테스트 5: 엣지 케이스")
    print("="*70 + "\n")
    
    edge_cases = [
        "",  # 빈 문자
        "가",  # 한 글자
        "1234567890",  # 숫자만
        "ㅋㅋㅋㅋㅋ",  # 특수 문자
        "a" * 100,  # 매우 긴 텍스트
        "안녕 하 세 요",  # 띄어쓰기 많음
    ]
    
    for text in edge_cases:
        if len(text) == 0:
            print(f"⚠️  빈 문자는 스킵")
            continue
            
        result = detector.predict_single(text)
        detector.print_result(result)


# ====================================
# 메인 실행
# ====================================

if __name__ == "__main__":
    # 모델 경로 설정
    model_path = "./hate_speech_output/model/final_model"
    
    print("\n" + "="*70)
    print("혐오 표현 탐지 모델 테스트")
    print("="*70 + "\n")
    
    try:
        # 모델 로드
        detector = HateSpeechDetector(model_path)
        
        # 테스트 1: 기본 예시
        results = test_basic_examples(detector)
        
        # 테스트 2: 대화형 테스트 (선택사항 - 주석 처리 가능)
        # test_interactive(detector)
        
        # 테스트 3: 긴 문장
        long_results = test_long_text(detector)
        
        # 테스트 4: CSV 저장
        all_results = results + long_results
        test_csv_save(detector, all_results)
        
        # 테스트 5: 엣지 케이스
        # test_edge_cases(detector)
        
        print("="*70)
        print("모든 테스트 완료!")
        print("="*70)
        
    except FileNotFoundError:
        print(f"❌ 모델을 찾을 수 없습니다: {model_path}")
        print("먼저 train_hate_speech_model.py를 실행해서 모델을 학습해주세요.")