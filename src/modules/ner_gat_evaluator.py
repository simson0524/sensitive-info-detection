# src/modules/ner_gat_evaluator.py

import torch
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Any
import numpy as np

class NerGatEvaluator:
    """
    [GAT-NER Evaluator] 개체명 단위(Entity-level) 성능 평가 및 그래프 추론 모듈
    
    특징:
    1. 추론 시에도 z-score 기반의 edge_index/attr를 사용하여 GAT 레이어 연산 수행
    2. 토큰 단위가 아닌 '라벨(PER, ORG 등)' 단위의 Precision, Recall, F1 산출
    3. 시각화를 위한 관계 행렬(CM) 및 통계적 중요도에 따른 정확도 분포 수집
    """
    def __init__(self, model, device, tokenizer, id2label: Dict[int, str]):
        """
        Args:
            model: 학습된 RobertaNerGatModel
            device: GPU (cuda)
            tokenizer: RoBERTa 토크나이저
            id2label: ID를 BIO 라벨로 변환하는 딕셔너리
        """
        self.model = model
        self.device = device
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.num_labels = len(id2label)
        self.o_label = "O"
        
        # BIO를 제거한 의미 단위 라벨 추출 (예: PER, ORG)
        self.pure_labels = sorted(list(set([label.split("-")[-1] for label in id2label.values() if "-" in label])))
        # Confusion Matrix용 라벨 리스트
        self.cm_labels = self.pure_labels + [self.o_label] if self.o_label not in self.pure_labels else self.pure_labels

    def evaluate(self, dataloader, prefix: str = "Validation", mode: str = "valid") -> Dict:
        """
        그래프 데이터를 포함한 배치를 순회하며 성능 평가 및 추론 로그 생성
        """
        start_time = datetime.now()
        self.model.eval()  # 평가 모드 (Dropout 비활성화)
        total_loss = 0
        inference_logs = []

        # 엔티티 기반 TP, FP, FN 집계용 구조
        label_relation_counts = {label: {l: 0 for l in self.cm_labels} for label in self.cm_labels}
        label_accuracy_distribution = {label: [] for label in self.pure_labels if label != self.o_label}

        with torch.no_grad():        
            for batch in tqdm(dataloader, desc=prefix, leave=False):
                # [STEP 1] GAT 추론을 위한 그래프 데이터 추출
                # Dataset에서 미리 구성된 x(tokens), edge, z_scores를 사용
                input_ids = batch.x.to(self.device)
                attention_mask = batch.attention_mask.to(self.device)
                labels = batch.y.to(self.device) if mode == "valid" else None
                
                # GAT 전역 관계 연산에 필수적인 Symbolic 피처
                z_scores = batch.z_scores.to(self.device)
                edge_index = batch.edge_index.to(self.device)
                edge_attr = batch.edge_attr.to(self.device)

                # [STEP 2] Forward Pass (학습 때와 동일한 입력을 주어 동일한 그래프 연산 보장)
                outputs = self.model(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    z_scores=z_scores,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    labels=labels
                )
                
                logits = outputs['logits']
                if mode == "valid":
                    total_loss += outputs['loss'].item()

                # [STEP 3] 결과 후처리 및 엔티티 분석
                pred_ids = torch.argmax(logits, dim=-1)
                batch_preds = pred_ids.cpu().tolist()
                batch_labels = labels.cpu().tolist() if labels is not None else None
                current_input_ids = input_ids.cpu().tolist()

                # 문장별 루프 (Batch 내 개별 데이터 분석)
                for i in range(len(batch_preds)):
                    # Dataset에 보존된 메타 데이터 활용
                    original_sentence = batch.sentence[i]
                    sentence_id = batch.sentence_id[i]
                    file_name = batch.file_name[i]
                    sentence_seq = batch.sentence_seq[i].item() if hasattr(batch.sentence_seq[i], 'item') else batch.sentence_seq[i]

                    # 토큰 변환 및 오프셋 매핑
                    tokens = self.tokenizer.convert_ids_to_tokens(current_input_ids[i])
                    encoding = self.tokenizer(original_sentence, return_offsets_mapping=True, add_special_tokens=True)
                    offset_mapping = encoding['offset_mapping']

                    # BIO -> 엔티티 리스트 파싱
                    pred_entities = self._parse_bio_to_entities(tokens, batch_preds[i], offset_mapping, original_sentence)
                    token_comparison = []
                    
                    if mode == "valid" and batch_labels:
                        gt_entities = self._parse_bio_to_entities(tokens, batch_labels[i], offset_mapping, original_sentence)
                        processed_pred_idx = set()

                        # GT 엔티티를 기준으로 예측 결과 매칭 (Winner-takes-all 방식)
                        for gt_ent in gt_entities:
                            gt_label = gt_ent['label']
                            candidate_preds = []
                            
                            for p_idx, pr_ent in enumerate(pred_entities):
                                if self._is_overlapping(gt_ent, pr_ent):
                                    # z-score가 반영된 예측의 질적 점수 계산
                                    score = self._calculate_entity_score(gt_ent, batch_preds[i])
                                    candidate_preds.append({
                                        "p_idx": p_idx, "gt_label": gt_label, "gt_entity": gt_ent, 
                                        "pred_label": pr_ent['label'], "pred_entity": pr_ent, "score": score
                                    })
                            
                            if candidate_preds:
                                best_pred = max(candidate_preds, key=lambda x: x['score'])
                                label_relation_counts[gt_label][best_pred['pred_label']] += 1
                                label_accuracy_distribution[gt_label].append(best_pred['score'])
                                processed_pred_idx.add(best_pred['p_idx'])
                                token_comparison.extend(candidate_preds)
                            else:
                                # 미탐 (False Negative)
                                label_relation_counts[gt_label][self.o_label] += 1
                                label_accuracy_distribution[gt_label].append(0.0)
                                token_comparison.append({
                                    "gt_label": gt_label, "pred_label": "일반정보", "score": 0.0
                                })

                        # 오탐 (False Positive) 집계
                        for p_idx, pr_ent in enumerate(pred_entities):
                            if p_idx not in processed_pred_idx:
                                label_relation_counts[self.o_label][pr_ent['label']] += 1

                    # [STEP 4] 최종 JSON 로그 생성 (DB 적재용)
                    sentence_inference_result = {
                        "sentence_id": sentence_id,
                        "source_file_name": file_name,
                        "origin_sentence": original_sentence,
                        "inference_results": pred_entities,
                        "token_comparison": token_comparison,
                        "entity_count": len(pred_entities)
                    }

                    inference_logs.append({
                        "sentence_id": sentence_id, 
                        "sentence_inference_result": sentence_inference_result, 
                        "inferenced_counts": len(pred_entities)
                    })

        # [STEP 5] 성능 지표 계산
        metrics = {}
        if mode == "valid":
            metrics['loss'] = total_loss / len(dataloader)
            metrics.update(self._calculate_entity_metrics(label_relation_counts))
            metrics['confusion_matrix'] = {
                "labels": self.cm_labels,
                "values": [[label_relation_counts[g][p] for p in self.cm_labels] for g in self.cm_labels]
            }
        
        return {
            'metrics': metrics, 
            'duration': (datetime.now() - start_time).total_seconds(), 
            'logs': inference_logs
        }

    # --- 내부 헬퍼 함수 (생략 없이 기존 로직 유지) ---
    def _is_overlapping(self, ent1, ent2):
        return not (ent1['end'] <= ent2['start'] or ent2['end'] <= ent1['start'])

    def _calculate_entity_score(self, gt_ent, pred_ids):
        indices = gt_ent['token_indices']
        label = gt_ent['label']
        b_idx = indices[0]
        score = 0.5 if self.id2label.get(pred_ids[b_idx]) == f"B-{label}" else 0.0
        i_indices = indices[1:]
        if not i_indices: return score * 2.0
        correct_i = sum(1 for idx in i_indices if self.id2label.get(pred_ids[idx]) == f"I-{label}")
        return score + (correct_i / len(i_indices)) * 0.5

    def _parse_bio_to_entities(self, tokens, tag_ids, offset_mapping, original_sentence):
        entities, current_entity = [], None
        for idx, (token, tag_id, (start, end)) in enumerate(zip(tokens, tag_ids, offset_mapping)):
            if start == 0 and end == 0: continue
            tag_name = self.id2label.get(tag_id, "O")
            if tag_name.startswith("B-"):
                if current_entity: entities.append(current_entity)
                current_entity = {"label": tag_name[2:], "start": start, "end": end, "token_indices": [idx], "word": original_sentence[start:end]}
            elif tag_name.startswith("I-") and current_entity:
                if current_entity["label"] == tag_name[2:]:
                    current_entity["end"] = end
                    current_entity["token_indices"].append(idx)
                    current_entity['word'] = original_sentence[current_entity['start']:current_entity['end']]
            else:
                if current_entity: entities.append(current_entity)
                current_entity = None
        if current_entity: entities.append(current_entity)
        return entities

    def _calculate_entity_metrics(self, relation_counts):
        precisions, recalls, f1s = [], [], []
        for label in self.pure_labels:
            if label == self.o_label: continue
            tp = relation_counts[label][label]
            fp = sum(relation_counts[other][label] for other in self.cm_labels if other != label)
            fn = sum(relation_counts[label][other] for other in self.cm_labels if other != label)
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0
            precisions.append(p); recalls.append(r); f1s.append(f)
        return {"precision": np.mean(precisions), "recall": np.mean(recalls), "f1": np.mean(f1s)}