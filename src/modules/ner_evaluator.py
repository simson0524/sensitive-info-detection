# src/modules/ner_evaluator.py

import torch
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from typing import Dict, List, Any
import numpy as np

class Evaluator:
    """
    [NER Evaluator] 모델 평가 및 결과 파싱을 담당하는 클래스
    
    주요 기능:
    1. 토큰 레벨 성능 평가 (Precision, Recall, F1, Loss)
    2. Confusion Matrix 계산 (오탐/미탐 분석용)
    3. BIO 태그 파싱 -> 원본 문장 기준 Entity(단어) 추출
    4. DB 로깅용 JSON 데이터 생성 (메타정보 + 추론결과 + 토큰별 상세비교)
    """
    def __init__(self, model, device, tokenizer, id2label: Dict[int, str]):
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.id2label = id2label # {0: "O", 1: "B-PER", 2: "I-PER", ...}
        self.num_labels = len(id2label)
        self.o_label = "O" 

    def evaluate(self, dataloader, prefix: str = "Validation", mode: str = "valid") -> Dict:
        """
        검증 데이터셋을 순회하며 모델 성능을 측정하고 로그를 생성합니다.
        
        Args:
            mode (str): 'valid' (정답 비교 및 메트릭 계산) or 'test' (순수 추론)

        Returns:
            Dict: {
                'metrics': {loss, precision, recall, f1, confusion_matrix},
                'start_time': datetime,
                'end_time': datetime,
                'duration': float,
                'logs': List[Dict] (DB 저장용 로그 리스트)
            }
        """
        start_time = datetime.now()
        self.model.eval()
        total_loss = 0
        
        # 1. 전체 배치에 대한 예측/정답 모음 (Metric 계산용)
        preds_flat = []
        targets_flat = []
        
        # 2. Confusion Matrix 초기화 (Row: Pred, Col: GT)
        # N x N 행렬, 0으로 초기화
        conf_matrix = [[0] * self.num_labels for _ in range(self.num_labels)]
        
        # 3. DB 저장용 문장별 로그 리스트
        inference_logs = []

        with torch.no_grad():        
            for batch in tqdm(dataloader, desc=prefix, leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Labels는 Valid 모드일 때만 필요
                labels = None
                if mode == "valid":
                    labels = batch["labels"].to(self.device)

                # Forward Pass
                if mode == "valid":
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs['loss']
                    total_loss += loss.item()
                    logits = outputs['logits']
                else:
                    # Test 모드 (Loss 계산 안 함)
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = outputs['logits']
                    loss = 0.0 # Dummy

                # Prediction (가장 높은 확률의 클래스 선택)
                pred_ids = torch.argmax(logits, dim=-1) # shape: (batch_size, seq_len)

                # CPU로 데이터 이동
                batch_preds = pred_ids.cpu().tolist()
                current_input_ids = input_ids.cpu().tolist()
                
                # Valid 모드일 때만 라벨 처리
                batch_labels = labels.cpu().tolist() if labels is not None else None
                
                # --- [Logic 1] 배치 전체에 대한 Metric & Confusion Matrix 집계 ---
                if mode == "valid" and batch_labels:
                    for p_seq, l_seq in zip(batch_preds, batch_labels):
                        for p, l in zip(p_seq, l_seq):
                            if l != -100: # Special Token (Padding 등) 제외
                                preds_flat.append(p)
                                targets_flat.append(l)
                                
                                # Confusion Matrix 업데이트
                                if 0 <= p < self.num_labels and 0 <= l < self.num_labels:
                                    conf_matrix[p][l] += 1

                # --- [Logic 2] 개별 문장 단위 로깅 (DB용 JSON 생성) ---
                for i in range(len(batch_preds)):
                    # 메타 데이터 추출
                    sentence_id = batch['sentence_id'][i]
                    original_sentence = batch['sentence'][i]
                    file_name = batch['file_name'][i]
                    
                    # Tensor -> Item 변환
                    seq_val = batch['sentence_seq'][i]
                    sentence_seq = seq_val.item() if hasattr(seq_val, 'item') else seq_val

                    # 토큰 변환 & Offset Mapping 재계산 (문장 내 위치 추적용)
                    tokens = self.tokenizer.convert_ids_to_tokens(current_input_ids[i])
                    # (Dataset에서 offset_mapping을 제공하지 않는 경우 재계산 필요)
                    encoding = self.tokenizer(original_sentence, return_offsets_mapping=True, add_special_tokens=True)
                    offset_mapping = encoding['offset_mapping']
                    
                    # A. 스팬(Entity) 파싱: BIO 태그 -> 원본 단어 추출
                    pred_entities = self._parse_bio_to_entities(
                        tokens, batch_preds[i], offset_mapping, original_sentence
                    )
                    
                    # B. [NEW] 토큰 레벨 상세 비교 (Confusion 분석용)
                    # Valid 모드일 때만 생성
                    token_comparison = []
                    if mode == "valid" and batch_labels:
                        gt_ids = batch_labels[i]
                        pr_ids = batch_preds[i]
                        
                        for t_str, g_id, p_id in zip(tokens, gt_ids, pr_ids):
                            if g_id == -100: continue # 스페셜 토큰 생략
                            
                            token_comparison.append({
                                "token": t_str,
                                "gt": self.id2label.get(g_id, "UNK"),
                                "pred": self.id2label.get(p_id, "UNK"),
                                "is_correct": (g_id == p_id)
                            })

                    # C. 최종 JSON 생성 (DB의 sentence_inference_result 컬럼에 저장됨)
                    sentence_inference_result = {
                        # 1. 메타 정보
                        "sentence_id": sentence_id,
                        "source_file_name": file_name,
                        "sequence_in_file": sentence_seq,
                        "origin_sentence": original_sentence,
                        
                        # 2. 추론 결과 (Entity List)
                        "inference_results": pred_entities,
                        
                        # 3. 토큰별 상세 비교 (오답 분석용 - Valid Only)
                        "token_comparison": token_comparison,
                        
                        "entity_count": len(pred_entities)
                    }

                    # DB 로그 엔트리 구성
                    log_entry = {
                        "sentence_id": sentence_id, 
                        "sentence_inference_result": sentence_inference_result, 
                        "confidence_score": 1.0 # (추후 Logits 기반 확률 계산 가능)
                    }
                    inference_logs.append(log_entry)

        # 4. 최종 메트릭 계산
        metrics = {}
        if mode == "valid":
            metrics['loss'] = total_loss / len(dataloader)
            metrics['precision'] = precision_score(targets_flat, preds_flat, average="macro", zero_division=0)
            metrics['recall'] = recall_score(targets_flat, preds_flat, average="macro", zero_division=0)
            metrics['f1'] = f1_score(targets_flat, preds_flat, average="macro", zero_division=0)
            metrics['confusion_matrix'] = conf_matrix # 전체 데이터셋에 대한 혼동 행렬
        else:
            # Test 모드는 0점 처리
            metrics = {'loss': 0.0, 'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # [중요] duration도 metrics 안에 포함
        metrics['duration'] = duration

        return {
            'metrics': metrics,
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'logs': inference_logs
        }

    def _parse_bio_to_entities(self, tokens: List[str], tag_ids: List[int], offset_mapping: List[tuple], original_sentence: str) -> List[Dict]:
        """
        BIO 태그 시퀀스를 파싱하여 병합된 엔티티 리스트를 반환합니다.
        (B-PER, I-PER, I-PER -> "홍길동")
        """
        entities = []
        current_entity = None 

        for idx, (token, tag_id, (start, end)) in enumerate(zip(tokens, tag_ids, offset_mapping)):
            if start == 0 and end == 0: continue # Special Token
                
            tag_name = self.id2label.get(tag_id, "O")
            
            # Case 1: B- 태그 (새로운 개체명 시작)
            if tag_name.startswith("B-"):
                if current_entity:
                    entities.append(self._finalize_entity(current_entity, original_sentence))
                
                label = tag_name[2:]
                current_entity = {
                    "label": label,
                    "start_idx": start,
                    "end_idx": end
                }
                
            # Case 2: I- 태그 (이전 개체명과 연결)
            elif tag_name.startswith("I-"):
                label = tag_name[2:]
                # 이전과 라벨이 같으면 연결
                if current_entity and current_entity["label"] == label:
                    current_entity["end_idx"] = end
                else:
                    # 문맥이 끊겼거나 B 없이 I가 나온 경우 (보정 로직)
                    if current_entity:
                        entities.append(self._finalize_entity(current_entity, original_sentence))
                    
                    # 끊어진 I 태그를 B 태그처럼 취급하여 새로 시작
                    current_entity = {
                        "label": label,
                        "start_idx": start,
                        "end_idx": end
                    }
                    
            # Case 3: O 태그 (개체명 아님)
            else: 
                if current_entity:
                    entities.append(self._finalize_entity(current_entity, original_sentence))
                    current_entity = None
        
        # 마지막에 남아있는 엔티티 저장
        if current_entity:
            entities.append(self._finalize_entity(current_entity, original_sentence))
            
        return entities

    def _finalize_entity(self, entity_info: Dict, original_sentence: str) -> Dict:
        """
        엔티티 정보를 최종 포맷으로 변환합니다.
        토큰 조각을 합치는 대신, 원본 문장의 인덱스(start~end)를 슬라이싱하여 정확한 단어를 가져옵니다.
        """
        start = entity_info["start_idx"]
        end = entity_info["end_idx"]
        
        return {
            "word": original_sentence[start:end], # 원본 문장에서 추출 (가장 정확함)
            "start": start,
            "end": end,
            "label": entity_info["label"]
        }