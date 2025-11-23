"""
NSMC (일반 문장) + K-MHaS (혐오표현) 이진분류 모델
레이블: 0 = 일반문장, 1 = 혐오표현

개선사항:
- 진행상황 실시간 표시
- 여러 output 파일 자동 생성
- 로그 파일 기록
- 메트릭 CSV 저장
- 모델 체크포인트
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random
import time
import datetime
import os
import csv
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras_preprocessing.sequence import pad_sequences

# ====================================
# 0. 출력 디렉토리 준비
# ====================================

OUTPUT_DIR = "./hate_speech_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/model", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/logs", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/results", exist_ok=True)

# 로그 파일
log_file = open(f"{OUTPUT_DIR}/logs/training_log.txt", 'w', encoding='utf-8')

def log_print(message):
    """콘솔과 파일에 동시에 출력"""
    print(message)
    log_file.write(message + "\n")
    log_file.flush()

# ====================================
# 1. 데이터 로드
# ====================================

def load_tsv_data(filepath):
    """TSV 파일 로드 (문장 \t 레이블)"""
    texts = []
    labels = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            
            text = parts[0].strip()
            label = int(parts[1].strip())
            
            if len(text) > 0:
                texts.append(text)
                labels.append(label)
    
    return texts, labels


def prepare_datasets_from_files(train_path, val_path, test_path):
    """
    이미 분할된 TSV 파일에서 데이터 로드
    레이블: 0 = 무해, 1 = 유해
    """
    log_print("\n[데이터 로드]")
    
    # Train 로드
    log_print(f"  Train 로드 중: {train_path}")
    train_texts, train_labels = load_tsv_data(train_path)
    train_hate = sum(1 for l in train_labels if l == 1)
    train_normal = sum(1 for l in train_labels if l == 0)
    log_print(f"    ✓ Train: {len(train_texts):,}개 (무해: {train_normal:,}, 유해: {train_hate:,})")
    
    # Val 로드
    log_print(f"  Val 로드 중: {val_path}")
    val_texts, val_labels = load_tsv_data(val_path)
    val_hate = sum(1 for l in val_labels if l == 1)
    val_normal = sum(1 for l in val_labels if l == 0)
    log_print(f"    ✓ Val: {len(val_texts):,}개 (무해: {val_normal:,}, 유해: {val_hate:,})")
    
    # Test 로드
    log_print(f"  Test 로드 중: {test_path}")
    test_texts, test_labels = load_tsv_data(test_path)
    test_hate = sum(1 for l in test_labels if l == 1)
    test_normal = sum(1 for l in test_labels if l == 0)
    log_print(f"    ✓ Test: {len(test_texts):,}개 (무해: {test_normal:,}, 유해: {test_hate:,})")
    
    log_print(f"\n[전체 데이터]")
    log_print(f"  총: {len(train_texts) + len(val_texts) + len(test_texts):,}개")
    
    return {
        'train': {'texts': train_texts, 'labels': train_labels},
        'val': {'texts': val_texts, 'labels': val_labels},
        'test': {'texts': test_texts, 'labels': test_labels}
    }


# ====================================
# 2. 토크나이징 및 텐서 변환
# ====================================

def data_to_tensor(sentences, labels, tokenizer, max_len=128):
    """텍스트를 BERT 입력 형태로 변환"""
    tokenized_texts = [tokenizer.tokenize(sent) for sent in sentences]
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    input_ids = pad_sequences(input_ids, maxlen=max_len, dtype="long", 
                             truncating="post", padding="post")
    
    attention_masks = []
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)
    
    tensor_inputs = torch.tensor(input_ids)
    tensor_labels = torch.tensor(labels, dtype=torch.long)
    tensor_masks = torch.tensor(attention_masks)
    
    return tensor_inputs, tensor_labels, tensor_masks


# ====================================
# 3. 평가 메트릭
# ====================================

def format_time(elapsed):
    """초를 hh:mm:ss 형식으로 변환"""
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def binary_metrics(predictions, labels):
    """이진 분류 메트릭 계산"""
    y_pred = np.argmax(predictions, axis=1)
    y_true = labels
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


# ====================================
# 4. 모델 학습
# ====================================

def train_binary_hate_speech_model(train_path, val_path, test_path,
                                  epochs=3, batch_size=32, learning_rate=2e-5):
    """이진분류 혐오 표현 탐지 모델 학습"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_labels = 2
    max_len = 128
    
    log_print("\n" + "=" * 70)
    log_print("이진분류 혐오 표현 탐지 모델 학습 시작")
    log_print("=" * 70)
    log_print(f"Device: {device}")
    log_print(f"Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {learning_rate}")
    
    # 1. 데이터 로드
    log_print("\n[1/5] 데이터 로드 중...")
    datasets = prepare_datasets_from_files(train_path, val_path, test_path)
    
    train_texts = datasets['train']['texts']
    train_labels = datasets['train']['labels']
    val_texts = datasets['val']['texts']
    val_labels = datasets['val']['labels']
    test_texts = datasets['test']['texts']
    test_labels = datasets['test']['labels']
    
    # 2. 토크나이징
    log_print("\n[2/5] 토크나이징 중...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', 
                                             do_lower_case=False)
    
    train_inputs, train_y, train_masks = data_to_tensor(train_texts, train_labels, 
                                                        tokenizer, max_len)
    val_inputs, val_y, val_masks = data_to_tensor(val_texts, val_labels, 
                                                  tokenizer, max_len)
    test_inputs, test_y, test_masks = data_to_tensor(test_texts, test_labels, 
                                                     tokenizer, max_len)
    log_print("  ✓ 토크나이징 완료")
    
    # 3. DataLoader 생성
    log_print("\n[3/5] DataLoader 생성 중...")
    train_data = TensorDataset(train_inputs, train_masks, train_y)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    
    val_data = TensorDataset(val_inputs, val_masks, val_y)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)
    
    test_data = TensorDataset(test_inputs, test_masks, test_y)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)
    
    log_print(f"  Train batches: {len(train_dataloader)}")
    log_print(f"  Val batches: {len(val_dataloader)}")
    log_print(f"  Test batches: {len(test_dataloader)}")
    
    # 4. 모델 설정
    log_print("\n[4/5] 모델 설정 중...")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-multilingual-cased",
        num_labels=num_labels
    )
    model.to(device)
    log_print("  ✓ 모델 로드 완료")
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
    
    # 메트릭 기록용
    metrics_history = {
        'epoch': [],
        'train_loss': [],
        'val_accuracy': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': []
    }
    
    # 5. 훈련 루프
    log_print("\n[5/5] 모델 훈련 중...")
    
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    model.zero_grad()
    training_start = time.time()
    
    for epoch_i in range(0, epochs):
        log_print(f"\n{'='*70}")
        log_print(f'Epoch {epoch_i + 1} / {epochs}')
        log_print(f"{'='*70}")
        
        # 훈련
        log_print('Training...')
        t0 = time.time()
        total_loss = 0
        model.train()
        
        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        
        for step, batch in progress_bar:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            
            outputs = model(b_input_ids,
                          token_type_ids=None,
                          attention_mask=b_input_mask,
                          labels=b_labels)
            
            loss = outputs[0]
            total_loss += loss.item()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            
            # 진행 상황 업데이트
            progress_bar.set_description(f"Loss: {loss.item():.4f}")
            
            if (step + 1) % 500 == 0:
                elapsed = format_time(time.time() - t0)
                log_print(f'  Batch {step+1:>5,}  of  {len(train_dataloader):>5,}.    Elapsed: {elapsed}')
        
        avg_train_loss = total_loss / len(train_dataloader)
        log_print(f"\nAverage training loss: {avg_train_loss:.4f}")
        log_print(f"Training epoch took: {format_time(time.time() - t0)}")
        
        # 검증
        log_print('\nRunning Validation...')
        t0 = time.time()
        model.eval()
        accum_logits, accum_labels = [], []
        
        progress_bar = tqdm(val_dataloader, total=len(val_dataloader))
        for batch in progress_bar:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            
            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.cpu().numpy()
            
            for b in logits:
                accum_logits.append(list(b))
            for b in label_ids:
                accum_labels.append(b)
        
        accum_logits = np.array(accum_logits)
        accum_labels = np.array(accum_labels)
        results = binary_metrics(accum_logits, accum_labels)
        
        log_print(f"Accuracy:  {results['accuracy']:.4f}")
        log_print(f"Precision: {results['precision']:.4f}")
        log_print(f"Recall:    {results['recall']:.4f}")
        log_print(f"F1 Score:  {results['f1']:.4f}")
        log_print(f"Validation took: {format_time(time.time() - t0)}")
        
        # 메트릭 기록
        metrics_history['epoch'].append(epoch_i + 1)
        metrics_history['train_loss'].append(avg_train_loss)
        metrics_history['val_accuracy'].append(results['accuracy'])
        metrics_history['val_precision'].append(results['precision'])
        metrics_history['val_recall'].append(results['recall'])
        metrics_history['val_f1'].append(results['f1'])
        
        # 모델 체크포인트 저장
        model.save_pretrained(f"{OUTPUT_DIR}/model/checkpoint_epoch_{epoch_i+1}")
    
    # 테스트
    log_print(f"\n{'='*70}")
    log_print("Running Test...")
    log_print(f"{'='*70}")
    
    t0 = time.time()
    model.eval()
    accum_logits, accum_labels = [], []
    
    progress_bar = tqdm(test_dataloader, total=len(test_dataloader))
    for batch in progress_bar:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        
        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
        
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.cpu().numpy()
        
        for b in logits:
            accum_logits.append(list(b))
        for b in label_ids:
            accum_labels.append(b)
    
    accum_logits = np.array(accum_logits)
    accum_labels = np.array(accum_labels)
    test_results = binary_metrics(accum_logits, accum_labels)
    
    log_print(f"\nTest Results:")
    log_print(f"Accuracy:  {test_results['accuracy']:.4f}")
    log_print(f"Precision: {test_results['precision']:.4f}")
    log_print(f"Recall:    {test_results['recall']:.4f}")
    log_print(f"F1 Score:  {test_results['f1']:.4f}")
    log_print(f"Test took: {format_time(time.time() - t0)}")
    
    total_time = format_time(time.time() - training_start)
    log_print(f"\nTotal training time: {total_time}")
    
    # 메트릭 CSV 저장
    metrics_df = pd.DataFrame(metrics_history)
    metrics_csv_path = f"{OUTPUT_DIR}/results/metrics.csv"
    metrics_df.to_csv(metrics_csv_path, index=False)
    log_print(f"\n✓ Metrics saved: {metrics_csv_path}")
    
    # 테스트 결과 저장
    test_results_path = f"{OUTPUT_DIR}/results/test_results.txt"
    with open(test_results_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("TEST RESULTS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Accuracy:  {test_results['accuracy']:.4f}\n")
        f.write(f"Precision: {test_results['precision']:.4f}\n")
        f.write(f"Recall:    {test_results['recall']:.4f}\n")
        f.write(f"F1 Score:  {test_results['f1']:.4f}\n")
    log_print(f"✓ Test results saved: {test_results_path}")
    
    log_print("\n" + "="*70)
    log_print("훈련 완료!")
    log_print("="*70)
    
    return model, tokenizer, device, test_results, accum_logits, accum_labels


# ====================================
# 5. 추론 함수
# ====================================

class HateSpeechDetector:
    """이진분류 혐오 표현 탐지기"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.softmax = nn.Softmax(dim=1)
    
    def predict_hate_score(self, text):
        """문장에 대한 혐오도 점수 반환"""
        tokenized = self.tokenizer.tokenize(text)
        input_ids = self.tokenizer.convert_tokens_to_ids(tokenized)
        
        max_len = 128
        padded = np.zeros(max_len)
        padded[:min(len(input_ids), max_len)] = input_ids[:max_len]
        attention_mask = [float(i > 0) for i in padded]
        
        input_ids_tensor = torch.tensor([padded], dtype=torch.long).to(self.device)
        attention_mask_tensor = torch.tensor([attention_mask], dtype=torch.float).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids_tensor, attention_mask=attention_mask_tensor)
        
        logits = outputs[0]
        probs = self.softmax(logits).cpu().numpy()[0]
        
        hate_score = float(probs[1])
        is_hate = hate_score > 0.5
        
        return hate_score, is_hate
    
    def predict_batch(self, texts):
        """배치로 혐오도 점수 계산"""
        results = []
        for text in texts:
            score, is_hate = self.predict_hate_score(text)
            results.append({
                'text': text,
                'hate_score': score,
                'is_hate': is_hate
            })
        return results


# ====================================
# 6. 메인 실행
# ====================================

if __name__ == "__main__":
    # 데이터셋 경로
    train_path = "./binary_hate_train.tsv"
    val_path = "./binary_hate_val.tsv"
    test_path = "./binary_hate_test.tsv"
    
    # 모델 학습
    model, tokenizer, device, test_results, test_logits, test_labels = train_binary_hate_speech_model(
        train_path,
        val_path,
        test_path,
        epochs=3,
        batch_size=32,
        learning_rate=2e-5
    )
    
    # 최종 모델 저장
    model.save_pretrained(f"{OUTPUT_DIR}/model/final_model")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/model/final_model")
    log_print(f"\n✓ Final model saved: {OUTPUT_DIR}/model/final_model")
    
    # 추론 예시
    log_print("\n" + "="*70)
    log_print("추론 예시")
    log_print("="*70)
    
    detector = HateSpeechDetector(model, tokenizer, device)
    
    test_texts = [
        "야이 바보야",
        "좋은 영화였어",
        "너 정말 최악이야",
        "안녕하세요 좋은 하루 되세요"
    ]
    
    results = detector.predict_batch(test_texts)
    
    # 결과 파일 저장
    inference_result_path = f"{OUTPUT_DIR}/results/inference_examples.csv"
    with open(inference_result_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Text', 'Hate_Score', 'Classification'])
        for result in results:
            classification = "혐오" if result['is_hate'] else "일반"
            writer.writerow([result['text'], f"{result['hate_score']:.4f}", classification])
            
            hate_label = "혐오" if result['is_hate'] else "일반"
            log_print(f"\n텍스트: {result['text']}")
            log_print(f"분류: {hate_label}")
            log_print(f"혐오도: {result['hate_score']:.4f}")
    
    log_print(f"\n✓ Inference results saved: {inference_result_path}")
    
    # 최종 요약
    log_print("\n" + "="*70)
    log_print("모든 파일 생성 완료!")
    log_print("="*70)
    log_print(f"Output directory: {OUTPUT_DIR}/")
    log_print(f"  - model/: 모델 저장")
    log_print(f"  - results/: 메트릭, 테스트 결과, 추론 예시")
    log_print(f"  - logs/: 훈련 로그")
    
    log_file.close()
    print(f"\n✓ 로그 파일: {OUTPUT_DIR}/logs/training_log.txt")