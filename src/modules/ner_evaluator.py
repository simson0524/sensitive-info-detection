# src/modules/ner_evaluator.py

import torch
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from typing import Dict, List, Any
import numpy as np

class Evaluator:
    """
    모델 평가(Validation/Test) 및 스팬 파싱을 담당하는 클래스
    - BIO 태그를 파싱하여 원본 문장 기준의 Entity Span을 추출
    - 메타데이터, 추론 결과, Confusion Matrix 정보를 생성
    """
    def __init__(self, model, device, tokenizer, id2label: Dict[int, str]):
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.id2label = id2label # {0: "O", 1: "B-PER", 2: "I-PER", ...}
        self.num_labels = len(id2label)
        self.o_label = "O" 

    def evaluate(self, dataloader, prefix: str = "Validation") -> Dict:
        start_time = datetime.now()
        self.model.eval()
        total_loss = 0
        
        # 1. Metric 계산용 Flatten 리스트
        preds_flat = []
        targets_flat = []
        
        # 2. Confusion Matrix 초기화 (N x N)
        # 행(Row): Prediction, 열(Col): Ground Truth (또는 반대, sklearn 기준은 Row=GT, Col=Pred)
        # 여기서는 기존 코드 로직(metric[pred][gt])을 따름 -> Row: Pred, Col: GT
        conf_matrix = [[0] * self.num_labels for _ in range(self.num_labels)]
        
        # 3. DB 저장용 로그
        inference_logs = []

        with torch.no_grad():        
            for batch in tqdm(dataloader, desc=prefix, leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs['loss']
                total_loss += loss.item()

                # Prediction
                pred_ids = torch.argmax(outputs['logits'], dim=-1)

                # CPU 이동
                batch_preds = pred_ids.cpu().tolist()
                batch_labels = labels.cpu().tolist()
                current_input_ids = input_ids.cpu().tolist()
                
                # --- [Logic 1] 배치 전체에 대한 Metric & Confusion Matrix 집계 ---
                for p_seq, l_seq in zip(batch_preds, batch_labels):
                    for p, l in zip(p_seq, l_seq):
                        if l != -100: # Special Token 제외
                            preds_flat.append(p)
                            targets_flat.append(l)
                            # Confusion Matrix 업데이트 (Row: Pred, Col: GT)
                            conf_matrix[p][l] += 1

                # --- [Logic 2] 개별 문장 로깅 (DB용) ---
                for i in range(len(batch_preds)):
                    # 메타 데이터
                    sentence_id = batch['sentence_id'][i]
                    original_sentence = batch['sentence'][i]
                    file_name = batch['file_name'][i]
                    seq_val = batch['sentence_seq'][i]
                    sentence_seq = seq_val.item() if hasattr(seq_val, 'item') else seq_val

                    # 토큰 & Offset Mapping
                    tokens = self.tokenizer.convert_ids_to_tokens(current_input_ids[i])
                    encoding = self.tokenizer(original_sentence, return_offsets_mapping=True, add_special_tokens=True)
                    offset_mapping = encoding['offset_mapping']
                    
                    # A. 스팬(Entity) 파싱
                    pred_entities = self._parse_bio_to_entities(
                        tokens, batch_preds[i], offset_mapping, original_sentence
                    )
                    
                    # B. 토큰 레벨 상세 비교 (Confusion 분석용)
                    # 해당 문장에서 각 토큰별로 (토큰문자, 정답라벨, 예측라벨)을 저장
                    token_comparison = []
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

                    # C. 최종 JSON 생성
                    sentence_inference_result = {
                        "sentence_id": sentence_id,
                        "source_file_name": file_name,
                        "sequence_in_file": sentence_seq,
                        "origin_sentence": original_sentence,
                        
                        # [결과 1] 추출된 개체명 리스트 (스팬 단위)
                        "inference_results": pred_entities,
                        
                        # [결과 2] 토큰별 상세 비교 리스트 (오답 분석용)
                        "token_comparison": token_comparison,
                        
                        "entity_count": len(pred_entities)
                    }

                    log_entry = {
                        "sentence_id": sentence_id, 
                        "sentence_inference_result": sentence_inference_result, 
                        "confidence_score": 1.0 
                    }
                    inference_logs.append(log_entry)

        # 메트릭 계산
        precision = precision_score(targets_flat, preds_flat, average="macro", zero_division=0)
        recall = recall_score(targets_flat, preds_flat, average="macro", zero_division=0)
        f1 = f1_score(targets_flat, preds_flat, average="macro", zero_division=0)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        return {
            'metrics': {
                'loss': total_loss / len(dataloader),
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'confusion_matrix': conf_matrix
            },
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'logs': inference_logs
        }

    def _parse_bio_to_entities(self, tokens: List[str], tag_ids: List[int], offset_mapping: List[tuple], original_sentence: str) -> List[Dict]:
        """
        BIO 태그 시퀀스를 파싱하여 병합된 엔티티 리스트를 반환합니다.
        """
        entities = []
        current_entity = None 

        for idx, (token, tag_id, (start, end)) in enumerate(zip(tokens, tag_ids, offset_mapping)):
            if start == 0 and end == 0: continue 
                
            tag_name = self.id2label.get(tag_id, "O")
            
            if tag_name.startswith("B-"):
                if current_entity:
                    entities.append(self._finalize_entity(current_entity, original_sentence))
                
                label = tag_name[2:]
                current_entity = {
                    "label": label,
                    "start_idx": start,
                    "end_idx": end
                }
                
            elif tag_name.startswith("I-"):
                label = tag_name[2:]
                if current_entity and current_entity["label"] == label:
                    current_entity["end_idx"] = end
                else:
                    if current_entity:
                        entities.append(self._finalize_entity(current_entity, original_sentence))
                    current_entity = {
                        "label": label,
                        "start_idx": start,
                        "end_idx": end
                    }
                    
            else: # "O"
                if current_entity:
                    entities.append(self._finalize_entity(current_entity, original_sentence))
                    current_entity = None
        
        if current_entity:
            entities.append(self._finalize_entity(current_entity, original_sentence))
            
        return entities

    def _finalize_entity(self, entity_info: Dict, original_sentence: str) -> Dict:
        start = entity_info["start_idx"]
        end = entity_info["end_idx"]
        
        return {
            "word": original_sentence[start:end],
            "start": start,
            "end": end,
            "label": entity_info["label"]
        }