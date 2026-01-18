# src/modules/ner_evaluator.py

import torch
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Any
import numpy as np

class Evaluator:
    """
    [NER Evaluator] 개체명 단위(Entity-level) 성능 평가 및 분석 모듈
    
    특징:
    1. 토큰 단위가 아닌 '라벨(PER, ORG 등)' 단위의 Precision, Recall, F1 산출
    2. 중복 예측 시 최고 점수 엔티티만 선정 (Winner-takes-all)
    3. 시각화를 위한 관계 행렬(CM) 및 정확도 분포($0.5 + 0.5 \times Ratio$) 수집
    """
    def __init__(self, model, device, tokenizer, id2label: Dict[int, str]):
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.num_labels = len(id2label)
        self.o_label = "O"
        
        # BIO를 제거한 순수 의미 단위 라벨 추출 (예: PER, ORG, LOC)
        self.pure_labels = sorted(list(set([label.split("-")[-1] for label in id2label.values() if "-" in label])))
        # 시각화(Confusion Matrix) 및 메트릭 계산을 위한 전체 라벨 (O 포함)
        self.cm_labels = self.pure_labels + [self.o_label] if self.o_label not in self.pure_labels else self.pure_labels

    def evaluate(self, dataloader, prefix: str = "Validation", mode: str = "valid") -> Dict:
        start_time = datetime.now()
        self.model.eval()
        total_loss = 0
        inference_logs = []

        # [데이터 수집] 엔티티 기반 릴레이션 카운트 (TP, FP, FN 집계용)
        # label_relation_counts[GT_LABEL][PRED_LABEL]
        label_relation_counts = {label: {l: 0 for l in self.cm_labels} for label in self.cm_labels}
        
        # [데이터 수집] 라벨별 정확도 점수 분포 (Histogram용, O 제외)
        label_accuracy_distribution = {label: [] for label in self.pure_labels if label != self.o_label}

        with torch.no_grad():        
            for batch in tqdm(dataloader, desc=prefix, leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch["labels"].to(self.device) if mode == "valid" else None

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                logits = outputs['logits']
                if mode == "valid":
                    total_loss += outputs['loss'].item()

                pred_ids = torch.argmax(logits, dim=-1)
                batch_preds = pred_ids.cpu().tolist()
                batch_labels = labels.cpu().tolist() if labels is not None else None
                current_input_ids = input_ids.cpu().tolist()

                # 문장별 분석
                for i in range(len(batch_preds)):
                    original_sentence = batch['sentence'][i]
                    tokens = self.tokenizer.convert_ids_to_tokens(current_input_ids[i])
                    encoding = self.tokenizer(original_sentence, return_offsets_mapping=True, add_special_tokens=True)
                    offset_mapping = encoding['offset_mapping']

                    # 1. BIO -> 엔티티 리스트 변환 (GT, Pred)
                    pred_entities = self._parse_bio_to_entities(tokens, batch_preds[i], offset_mapping, original_sentence)
                    
                    if mode == "valid" and batch_labels:
                        gt_entities = self._parse_bio_to_entities(tokens, batch_labels[i], offset_mapping, original_sentence)
                        processed_pred_idx = set()

                        # 2. GT 엔티티를 기준으로 예측 결과 매칭
                        for gt_ent in gt_entities:
                            gt_label = gt_ent['label']
                            
                            # GT와 겹치는 예측 후보들 중 가장 점수가 높은 것 선택
                            candidate_preds = []
                            for p_idx, pr_ent in enumerate(pred_entities):
                                if self._is_overlapping(gt_ent, pr_ent):
                                    score = self._calculate_entity_score(gt_ent, batch_preds[i])
                                    candidate_preds.append({"p_idx": p_idx, "label": pr_ent['label'], "score": score})
                            
                            if candidate_preds:
                                # 최고 점수 예측 하나만 대표로 선정 (Winner-takes-all)
                                best_pred = max(candidate_preds, key=lambda x: x['score'])
                                label_relation_counts[gt_label][best_pred['label']] += 1
                                label_accuracy_distribution[gt_label].append(best_pred['score'])
                                processed_pred_idx.add(best_pred['p_idx'])
                            else:
                                # 미탐 (False Negative)
                                label_relation_counts[gt_label][self.o_label] += 1
                                label_accuracy_distribution[gt_label].append(0.0)

                        # 3. 오탐 (False Positive): GT와 겹치지 않는 나머지 예측들
                        for p_idx, pr_ent in enumerate(pred_entities):
                            if p_idx not in processed_pred_idx:
                                label_relation_counts[self.o_label][pr_ent['label']] += 1

                    # (참고) DB 로깅용 logs 생성 로직은 기존 코드와 동일하게 유지 가능
                    # --- [Logic 2] 개별 문장 단위 로깅 (DB용 JSON 생성) ---
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

        # 4. 엔티티 기반 메트릭 계산
        metrics = {}
        if mode == "valid":
            metrics['loss'] = total_loss / len(dataloader)
            
            # 의미 단위 라벨 기준 Precision, Recall, F1 계산
            perf_metrics = self._calculate_entity_metrics(label_relation_counts)
            metrics.update(perf_metrics)
            
            # 시각화용 데이터 포함
            metrics['label_relation_counts'] = label_relation_counts
            metrics['label_accuracy_distribution'] = label_accuracy_distribution
            metrics['confusion_matrix'] = {
                "labels": self.cm_labels,
                "values": [[label_relation_counts[g][p] for p in self.cm_labels] for g in self.cm_labels]
            }
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        return {
            'metrics': metrics,
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration, 
            'logs': inference_logs
        }

    def _calculate_entity_metrics(self, relation_counts: Dict) -> Dict:
        """
        label_relation_counts(엔티티 릴레이션)를 바탕으로 Macro Metrics 계산
        """
        precisions, recalls, f1s = [], [], []

        for label in self.pure_labels:
            if label == self.o_label: continue
            
            # TP: GT와 Pred가 동일한 경우
            tp = relation_counts[label][label]
            # FP: Pred가 해당 라벨인데 GT가 다른 라벨이거나 O인 경우
            fp = sum(relation_counts[other][label] for other in self.cm_labels if other != label)
            # FN: GT가 해당 라벨인데 Pred가 다른 라벨이거나 O인 경우
            fn = sum(relation_counts[label][other] for other in self.cm_labels if other != label)
            
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0
            
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)
            
        return {
            "precision": np.mean(precisions) if precisions else 0.0,
            "recall": np.mean(recalls) if recalls else 0.0,
            "f1": np.mean(f1s) if f1s else 0.0
        }

    def _calculate_entity_score(self, gt_ent, pred_ids) -> float:
        """가중치 점수 산출: B일치(0.5) + I일치비율(0.5)"""
        indices = gt_ent['token_indices']
        label = gt_ent['label']
        b_idx = indices[0]
        i_indices = indices[1:]
        
        score = 0.0
        if self.id2label.get(pred_ids[b_idx]) == f"B-{label}":
            score += 0.5
        
        if not i_indices:
            return score * 2.0 # 단일 토큰이면 B만 맞으면 1.0
            
        correct_i = sum(1 for idx in i_indices if self.id2label.get(pred_ids[idx]) == f"I-{label}")
        score += (correct_i / len(i_indices)) * 0.5
        return score

    def _is_overlapping(self, ent1, ent2):
        """원본 문장 인덱스 기준 겹침 여부 판단"""
        return not (ent1['end'] <= ent2['start'] or ent2['end'] <= ent1['start'])

    def _parse_bio_to_entities(self, tokens, tag_ids, offset_mapping, original_sentence):
        """BIO 시퀀스를 엔티티 리스트로 파싱 (token_indices 포함)"""
        entities, current_entity = [], None
        for idx, (token, tag_id, (start, end)) in enumerate(zip(tokens, tag_ids, offset_mapping)):
            if start == 0 and end == 0: continue

            tag_name = self.id2label.get(tag_id, "O")

            # Case 1: B- 태그 (새로운 개체명 시작)
            if tag_name.startswith("B-"):
                if current_entity: 
                    entities.append(current_entity)
                current_entity = {"label": tag_name[2:], "start": start, "end": end, "token_indices": [idx], "word": original_sentence[start:end]}
            
            # Case 2: I- 태그 (이전 개체명과 연결)
            elif tag_name.startswith("I-") and current_entity:
                label = tag_name[2:]
                if current_entity["label"] == label:
                    current_entity["end"] = end
                    current_entity["token_indices"].append(idx)
                else:
                    entities.append(current_entity)
                    current_entity = {"label": label, "start": start, "end": end, "token_indices": [idx], "word": original_sentence[start:end]}
            
            # Case 3: O 태그 (개체명 아님)
            else:
                if current_entity: entities.append(current_entity)
                current_entity = None
       
        # 마지막에 남아있는 엔티티 저장
        if current_entity: entities.append(current_entity)
        return entities