"""
K-MHaS 원본 데이터로 이진분류 학습
레이블: 0 = 무해, 1 = 유해

규칙:
- 8 (Not Hate Speech) → 무해 (0)
- 0-7 (나머지) → 유해 (1)
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
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras_preprocessing.sequence import pad_sequences

# ====================================
# 0. 출력 디렉토리 준비
# ====================================

OUTPUT_DIR = "./hate_speech_output_kmhas"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/model", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/logs", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/results", exist_ok=True)

log_file = open(f"{OUTPUT_DIR}/logs/training_log.txt", 'w', encoding='utf-8')

def log_print(message):
    """콘솔과 파일에 동시에 출력"""
    print(message)
    log_file.write(message + "\n")
    log_file.flush()

# ====================================
# 1. K-MHaS 데이터 로드 및 변환
# ====================================

def parse_kmhas_labels(label_str):
    """
    K-MHaS 레이블 파싱 및 이진분류 변환
    
    규칙:
    - 8 (Not Hate Speech) → 0 (무해)
    - 0-7 (나머지) → 1 (유해)
    
    Returns:
        label: 0 or 1
        label_counts: 각 카테고리 개수
    """
    try:
        label_indices = [int(x.strip()) for x in label_str.split(',')]
    except:
        return None, None
    
    # 각 카테고리별 카운트
    label_counts = {}
    for idx in label_indices:
        label_counts[idx] = label_counts.get(idx, 0) + 1
    
    # 가장 많은 레이블 선택
    max_label = max(label_counts.keys())
    
    # 이진분류 변환
    if max_label == 8:
        binary_label = 0  # 무해
    else:
        binary_label = 1  # 유해
    
    return binary_label, label_counts


def load_kmhas_file(filepath):
    """K-MHaS TSV 파일 로드"""
    texts = []
    labels = []
    label_dist = {}  # 레이블 분포
    
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        # 첫 줄이 헤더인지 확인
        first_line = lines[0].strip().split('\t')
        start_idx = 1 if first_line[0].lower() in ['document', 'text', 'sentence'] else 0
        
        for line in lines[start_idx:]:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            
            text = parts[0].strip()
            label_str = parts[1].strip()
            
            if len(text) == 0 or len(label_str) == 0:
                continue
            
            binary_label, label_counts = parse_kmhas_labels(label_str)
            
            if binary_label is None:
                continue
            
            texts.append(text)
            labels.append(binary_label)
            
            # 레이블 분포 기록
            if label_counts:
                for idx, count in label_counts.items():
                    label_dist[idx] = label_dist.get(idx, 0) + count
    
    return texts, labels, label_dist


def prepare_kmhas_datasets(train_path, val_path, test_path):
    """K-MHaS 3개 파일 로드 및 분석"""
    log_print("\n[K-MHaS 데이터 로드]")
    
    # Train 로드
    log_print(f"\n[Train] {train_path}")
    train_texts, train_labels, train_dist = load_kmhas_file(train_path)
    train_hate = sum(1 for l in train_labels if l == 1)
    train_normal = sum(1 for l in train_labels if l == 0)
    log_print(f"  ✓ {len(train_texts):,}개 (무해: {train_normal:,}, 유해: {train_hate:,})")
    log_print(f"  레이블 분포: {train_dist}")
    
    # Valid 로드
    log_print(f"\n[Valid] {val_path}")
    val_texts, val_labels, val_dist = load_kmhas_file(val_path)
    val_hate = sum(1 for l in val_labels if l == 1)
    val_normal = sum(1 for l in val_labels if l == 0)
    log_print(f"  ✓ {len(val_texts):,}개 (무해: {val_normal:,}, 유해: {val_hate:,})")
    log_print(f"  레이블 분포: {val_dist}")
    
    # Test 로드
    log_print(f"\n[Test] {test_path}")
    test_texts, test_labels, test_dist = load_kmhas_file(test_path)
    test_hate = sum(1 for l in test_labels if l == 1)
    test_normal = sum(1 for l in test_labels if l == 0)
    log_print(f"  ✓ {len(test_texts):,}개 (무해: {test_normal:,}, 유해: {test_hate:,})")
    log_print(f"  레이블 분포: {test_dist}")
    
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

def train_kmhas_hate_speech_model(train_path, val_path, test_path,
                                 epochs=3, batch_size=32, learning_rate=2e-5):
    """K-MHaS 데이터로 이진분류 모델 학습"""
    # GPU 2, 3 사용
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    num_labels = 2
    max_len = 128
    
    log_print("\n" + "=" * 70)
    log_print("K-MHaS 이진분류 혐오 표현 탐지 모델 학습")
    log_print("=" * 70)
    log_print(f"Device: {device}")
    log_print(f"GPU 사용: 2, 3번 (DataParallel)")
    log_print(f"Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {learning_rate}")
    
    # 1. 데이터 로드
    log_print("\n[1/5] 데이터 로드 중...")
    datasets = prepare_kmhas_datasets(train_path, val_path, test_path)
    
    train_texts = datasets['train']['texts']
    train_labels = datasets['train']['labels']
    val_texts = datasets['val']['texts']
    val_labels = datasets['val']['labels']
    test_texts = datasets['test']['texts']
    test_labels = datasets['test']['labels']
    
    # 2. 토크나이징
    log_print("\n[2/5] 토크나이징 중...")
    tokenizer = BertTokenizer.from_pretrained('beomi/kcbert-base', 
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
        "beomi/kcbert-base",
        num_labels=num_labels
    )
    
    # GPU 2, 3 동시 사용 (DataParallel)
    if torch.cuda.device_count() > 1:
        log_print(f"  사용 가능한 GPU: {torch.cuda.device_count()}개")
        model = nn.DataParallel(model, device_ids=[2, 3])
        log_print("  ✓ GPU 2, 3 DataParallel 설정")
    
    model.to(device)
    log_print("  ✓ KcBERT 모델 로드 완료")
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
    
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
        
        metrics_history['epoch'].append(epoch_i + 1)
        metrics_history['train_loss'].append(avg_train_loss)
        metrics_history['val_accuracy'].append(results['accuracy'])
        metrics_history['val_precision'].append(results['precision'])
        metrics_history['val_recall'].append(results['recall'])
        metrics_history['val_f1'].append(results['f1'])
        
        if hasattr(model, 'module'):
            model.module.save_pretrained(f"{OUTPUT_DIR}/model/checkpoint_epoch_{epoch_i+1}")
        else:
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
        f.write("K-MHaS TEST RESULTS\n")
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
# 5. 메인 실행
# ====================================

if __name__ == "__main__":
    # K-MHaS 데이터 경로
    train_path = "./K-MHaS/data/kmhas_train.txt"
    val_path = "./K-MHaS/data/kmhas_valid.txt"
    test_path = "./K-MHaS/data/kmhas_test.txt"
    
    # 모델 학습
    model, tokenizer, device, test_results, test_logits, test_labels = train_kmhas_hate_speech_model(
        train_path,
        val_path,
        test_path,
        epochs=3,
        batch_size=8,  # 32 → 16으로 감소
        learning_rate=2e-5
    )
    
    # 최종 모델 저장
    if hasattr(model, 'module'):
        model.module.save_pretrained(f"{OUTPUT_DIR}/model/final_model")
    else:
        model.save_pretrained(f"{OUTPUT_DIR}/model/final_model")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/model/final_model")
    log_print(f"\n✓ Final model saved: {OUTPUT_DIR}/model/final_model")
    
    # 최종 요약
    log_print("\n" + "="*70)
    log_print("모든 파일 생성 완료!")
    log_print("="*70)
    log_print(f"Output directory: {OUTPUT_DIR}/")
    log_print(f"  - model/: 모델 저장")
    log_print(f"  - results/: 메트릭, 테스트 결과")
    log_print(f"  - logs/: 훈련 로그")
    
    log_file.close()
    print(f"\n✓ 로그 파일: {OUTPUT_DIR}/logs/training_log.txt")