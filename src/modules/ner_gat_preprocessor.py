# src/modules/ner_gat_preprocessor.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Any, Tuple
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

# ==============================================================================
# 1. RobertaNerGatDataset (GAT 지원 데이터셋)
# ==============================================================================
class RobertaNerGatDataset(Dataset):
    """
    RoBERTa + GAT 모델을 위한 전처리 및 데이터셋 클래스
    - BIO 라벨링 및 Token Alignment
    - Word-level z-score를 Token-level로 매핑
    - Anchor Hub 전략 기반 동적 그래프(edge_index, edge_attr) 생성
    """
    def __init__(
        self, 
        samples: Dict, 
        annotations: Dict, 
        tokenizer, 
        ner_label2id: Dict, 
        z_score_map: Dict[str, float],
        max_length: int = 256, 
        data_category: str = "personal_data",
        hub_ratio: float = 0.15
    ):
        """
        Args:
            samples (Dict): {sentence_id: sentence_data}
            annotations (Dict): {sentence_id: annotation_list}
            tokenizer: HuggingFace Tokenizer
            ner_label2id (Dict): BIO 태그 -> ID 매핑
            z_score_map (Dict): 단어별 z-score 딕셔너리
            max_length (int): 최대 시퀀스 길이
            data_category (str): "personal_data" 또는 "confidential_data"
            hub_ratio (float): 문장 길이 대비 허브 노드 비율
        """
        self.samples = samples
        self.annotations = annotations
        self.tokenizer = tokenizer
        self.z_score_map = z_score_map
        self.max_length = max_length
        self.ner_label2id = ner_label2id
        self.data_category = data_category
        self.hub_ratio = hub_ratio
        self.instances = []
        
        self._create_instances()

    def _create_graph_structure(
        self, 
        z_scores: torch.Tensor, 
        word_ids: List[Optional[int]], 
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Anchor Hub 기반의 동적 그래프 생성 (내부 메서드)
        - 24GB VRAM 최적화를 위해 Sparse한 연결 구조 생성
        """
        valid_indices = torch.where(attention_mask == 1)[0]
        num_valid = len(valid_indices)
        
        # 1. 유동적 K 결정 (허브 개수)
        k = max(1, int(num_valid * self.hub_ratio))
        k = min(k, 10) # Over-smoothing 방지를 위한 상한선

        # 2. 허브 후보: 각 단어의 '첫 번째 토큰' 위치 추출
        head_indices = []
        last_word_id = -1
        for i in valid_indices:
            w_id = word_ids[i]
            if w_id is not None and w_id != last_word_id:
                head_indices.append(i.item())
                last_word_id = w_id
        
        # 3. z-score 기반 Top-K 앵커 허브 선정
        head_indices = torch.tensor(head_indices, dtype=torch.long)
        head_z = z_scores[head_indices]
        _, top_k_idx = torch.topk(head_z, min(k, len(head_indices)))
        hub_indices = head_indices[top_k_idx]

        edges = []
        edge_weights = []
        z_weights = torch.sigmoid(z_scores) # 가중치용 스케일링

        # 4. 로컬 간선 생성 (인접 문맥 유지)
        for i in range(len(valid_indices) - 1):
            u, v = valid_indices[i].item(), valid_indices[i+1].item()
            edges.extend([[u, v], [v, u]])
            edge_weights.extend([0.5, 0.5])

        # 5. 전역 앵커 허브 간선 생성 (정보 지름길)
        for h_idx in hub_indices:
            h_idx = h_idx.item()
            w = z_weights[h_idx].item()
            for t_idx in valid_indices:
                t_idx = t_idx.item()
                if h_idx != t_idx:
                    edges.extend([[t_idx, h_idx], [h_idx, t_idx]])
                    edge_weights.extend([w, w])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(-1)
        
        return edge_index, edge_attr

    def _create_instances(self):
        """
        데이터를 순회하며 BERT 입력값과 GAT용 그래프 데이터를 생성합니다.
        """
        for sent_id, sample_data in tqdm(self.samples.items(), desc="Create GAT-NER instances"):
            sentence = sample_data['sentence']
            
            # 1. 토큰화 및 기본 인코딩
            encoding = self.tokenizer(
                sentence,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_offsets_mapping=True
            )
            
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            offset_mapping = encoding["offset_mapping"]
            word_ids = encoding.word_ids()

            # 2. z-score 매핑 (Word-level -> Token-level)
            raw_words = sentence.split()
            z_scores = torch.zeros(self.max_length, dtype=torch.float)
            for i, w_id in enumerate(word_ids):
                if w_id is not None and w_id < len(raw_words):
                    z_scores[i] = self.z_score_map.get(raw_words[w_id], 0.0)

            # 3. BIO 라벨 할당 (기존 로직 유지)
            labels = [self.ner_label2id["O"]] * len(input_ids)
            sentence_annotations = self.annotations.get(sent_id, [])
            
            for ann in sentence_annotations:
                char_start, char_end, ann_label = ann['start'], ann['end'], ann['label']
                # (카테고리 필터링 로직 생략 - 필요 시 추가 가능)
                
                b_label_id = self.ner_label2id.get(f"B-{ann_label}", self.ner_label2id["O"])
                i_label_id = self.ner_label2id.get(f"I-{ann_label}", self.ner_label2id["O"])

                token_start, token_end = None, None
                for i, (os, oe) in enumerate(offset_mapping):
                    if os == 0 and oe == 0: continue
                    if token_start is None and os <= char_start < oe: token_start = i
                    if token_end is None and os < char_end <= oe: token_end = i

                if token_start is not None and token_end is not None:
                    labels[token_start] = b_label_id
                    for idx in range(token_start + 1, token_end + 1):
                        labels[idx] = i_label_id
            
            # 스페셜 토큰 마스킹 (-100)
            for i, (os, oe) in enumerate(offset_mapping):
                if os == 0 and oe == 0: labels[i] = -100

            # 4. GAT를 위한 그래프 구조 생성
            edge_index, edge_attr = self._create_graph_structure(
                z_scores, word_ids, torch.tensor(attention_mask)
            )

            # 5. 인스턴스 저장
            self.instances.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "z_scores": z_scores.tolist(),
                "labels": labels,
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "sentence_id": sent_id
            })

    def __getitem__(self, idx) -> Data:
        item = self.instances[idx]
        
        # PyG 전용 Data 객체로 반환 (DataLoader가 자동으로 인덱스 오프셋 처리)
        return Data(
            x=torch.tensor(item["input_ids"], dtype=torch.long),
            edge_index=item["edge_index"],
            edge_attr=item["edge_attr"],
            y=torch.tensor(item["labels"], dtype=torch.long),
            z_scores=torch.tensor(item["z_scores"], dtype=torch.float),
            attention_mask=torch.tensor(item["attention_mask"], dtype=torch.long)
        )

    def __len__(self):
        return len(self.instances)