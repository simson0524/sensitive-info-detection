# src/modules/ner_gat_preprocessor.py

import json
import torch
import os
from typing import List, Dict, Any, Tuple, Optional
from torch.utils.data import Dataset
from torch_geometric.data import Data
from transformers import PreTrainedTokenizerFast
from src.utils.common import normalize_label
from tqdm import tqdm

# ==============================================================================
# 1. RobertaNerGatDataset (데이터셋 생성 및 인스턴스화)
# ==============================================================================
class RobertaNerGatDataset(Dataset):
    """
    RoBERTa(Neural) + GAT(Symbolic) 하이브리드 모델 전용 데이터셋 클래스
    - 특징: z_score_map을 도메인별 2중 딕셔너리 구조로 처리함
    """
    def __init__(
        self, 
        samples: Dict, 
        annotations: Dict, 
        tokenizer: PreTrainedTokenizerFast, 
        ner_label2id: Dict, 
        ner_id2label: Dict,
        z_score_map: Dict[str, Dict[str, float]], # [수정] 도메인별 2중 맵 구조 반영
        max_length: int = 256, 
        data_category: str = "personal_data",
        hub_ratio: float = 0.15
    ):
        self.samples = samples
        self.annotations = annotations
        self.tokenizer = tokenizer
        self.ner_label2id = ner_label2id
        self.ner_id2label = ner_id2label
        self.z_score_map = z_score_map
        self.max_length = max_length
        self.data_category = data_category
        self.hub_ratio = hub_ratio
        self.instances = []
        
        if self.data_category not in ["personal_data", "confidential_data"]:
            raise ValueError(f"Invalid data_category: {self.data_category}")

        self._create_instances()

    def _create_graph_structure(
        self, 
        z_scores: torch.Tensor, 
        word_ids: List[Optional[int]], 
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Anchor Hub 전략 기반 동적 그래프 생성 (기존 로직 유지) """
        valid_indices = torch.where(attention_mask == 1)[0]
        num_valid = len(valid_indices)
        k = max(1, int(num_valid * self.hub_ratio))
        k = min(k, 10) 

        head_indices = []
        last_word_id = -1
        for i in valid_indices:
            w_id = word_ids[i]
            if w_id is not None and w_id != last_word_id:
                head_indices.append(i.item())
                last_word_id = w_id
        
        head_indices_ts = torch.tensor(head_indices, dtype=torch.long)
        head_z = z_scores[head_indices_ts]
        actual_k = min(k, len(head_indices_ts))
        _, top_k_idx = torch.topk(head_z, actual_k)
        hub_indices = head_indices_ts[top_k_idx]

        edges = []
        edge_weights = []
        z_weights = torch.sigmoid(z_scores) 

        for i in range(len(valid_indices) - 1):
            u, v = valid_indices[i].item(), valid_indices[i+1].item()
            edges.extend([[u, v], [v, u]])
            edge_weights.extend([0.5, 0.5])

        for h_idx in hub_indices:
            h_idx_val = h_idx.item()
            weight = z_weights[h_idx_val].item()
            for t_idx in valid_indices:
                t_idx_val = t_idx.item()
                if h_idx_val != t_idx_val:
                    edges.extend([[t_idx_val, h_idx_val], [h_idx_val, t_idx_val]])
                    edge_weights.extend([weight, weight])

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float).unsqueeze(-1)
        
        return edge_index, edge_attr

    def _create_instances(self):
        """ Raw 데이터를 순회하며 모델 입력용 인스턴스 생성 """
        for sent_id, sample_data in tqdm(self.samples.items(), desc="Create GAT-NER instances"):
            sentence = sample_data['sentence']
            
            # [도메인 파싱] 
            try:
                domain_id = str(int(sent_id.split('_')[1]))
            except:
                domain_id = "0"
            
            sentence_seq = sample_data.get('sequence', 0)
            filename = sample_data.get('filename', "")

            # 1. 토큰화 및 Offset Mapping
            encoding = self.tokenizer(
                sentence,
                max_length=self.max_length,
                truncation=True,
                return_offsets_mapping=True,
                padding='max_length'
            )
            
            input_ids = encoding['input_ids']
            decoded_input_ids = self.tokenizer.convert_ids_to_tokens(input_ids)
            attention_mask = torch.tensor(encoding['attention_mask'])
            offset_mapping = encoding["offset_mapping"]
            word_ids = encoding.word_ids()

            # 2. 토큰별 z-score 매핑 (도메인 특화 로직 반영)
            raw_words = sentence.split()
            z_scores = torch.zeros(self.max_length, dtype=torch.float)
            
            # [핵심] 현재 문장의 domain_id에 해당하는 z_score 맵을 먼저 가져옴
            current_domain_z_map = self.z_score_map.get(domain_id, {})
            
            for i, w_id in enumerate(word_ids):
                if w_id is not None and w_id < len(raw_words):
                    # 해당 도메인의 맵에서 단어의 z_score 조회
                    word_str = raw_words[w_id]
                    z_scores[i] = current_domain_z_map.get(word_str, 0.0)

            # 3. 라벨 할당 로직 (기존 스타일 유지)
            labels = [self.ner_label2id["O"]] * len(input_ids)
            sentence_annotations = self.annotations.get(sent_id, [])
            
            for ann in sentence_annotations:
                ann_label = ann['label']
                char_start, char_end = ann['start'], ann['end']
                norm_label = normalize_label(ann_label)

                if self.data_category == "personal_data" and norm_label == "기밀정보": continue
                if self.data_category == "confidential_data" and norm_label == "개인정보": continue
                if norm_label in ["일반정보", "준식별자"]: continue

                b_label_name, i_label_name = f"B-{ann_label}", f"I-{ann_label}"
                if b_label_name not in self.ner_label2id: continue
                
                b_label_id, i_label_id = self.ner_label2id[b_label_name], self.ner_label2id[i_label_name]

                token_start, token_end = None, None
                for i, (os, oe) in enumerate(offset_mapping):
                    if os == 0 and oe == 0: continue
                    if (token_start is None) and (os <= char_start < oe): token_start = i
                    if (token_end is None) and (os < char_end <= oe): token_end = i 

                if token_start is not None and token_end is not None:
                    labels[token_start] = b_label_id
                    for i in range(token_start + 1, token_end + 1):
                        if i < len(labels): labels[i] = i_label_id
            
            # 4. 스페셜 토큰 마스킹
            for i, (os, oe) in enumerate(offset_mapping):
                if os == 0 and oe == 0: labels[i] = -100
            
            # 5. GAT 그래프 구조 생성
            edge_index, edge_attr = self._create_graph_structure(z_scores, word_ids, attention_mask)

            # 6. 인스턴스 추가
            self.instances.append({
                "sentence": sentence,
                "sentence_id": sent_id,
                "domain_id": domain_id,
                "sentence_seq": sentence_seq,
                "input_ids": input_ids,
                "decoded_input_ids": decoded_input_ids,
                "attention_mask": attention_mask.tolist(),
                "labels": labels,
                "z_scores": z_scores.tolist(),
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "file_name": filename
            })

    def __getitem__(self, idx) -> Data:
        item = self.instances[idx]
        return Data(
            x=torch.tensor(item["input_ids"], dtype=torch.long),
            edge_index=item["edge_index"],
            edge_attr=item["edge_attr"],
            y=torch.tensor(item["labels"], dtype=torch.long),
            z_scores=torch.tensor(item["z_scores"], dtype=torch.float),
            attention_mask=torch.tensor(item["attention_mask"], dtype=torch.long),
            sentence=item["sentence"],
            sentence_id=item["sentence_id"],
            domain_id=item["domain_id"],
            sentence_seq=item["sentence_seq"],
            decoded_input_ids=item["decoded_input_ids"],
            file_name=item["file_name"]
        )
    
    def __len__(self):
        return len(self.instances)


# ==============================================================================
# 2. NerGatPreprocessor (데이터 로드 및 라벨 관리)
# ==============================================================================
class NerGatPreprocessor:
    """
    데이터 로드, BIO 라벨 맵 생성, Dataset 생성을 관리하는 클래스 (GAT 버전)
    """
    def __init__(self, tokenizer: PreTrainedTokenizerFast, max_len: int = 256, label2id: Dict[str, int] = None):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.source_label2id = label2id
        
        if label2id:
            self.ner_label2id, self.ner_id2label = self._build_bio_labels(label2id)
        else:
            self.ner_label2id = None
            self.ner_id2label = None

    def _build_bio_labels(self, source_label2id: Dict[str, int]) -> Tuple[Dict, Dict]:
        ner_l2i = {"O": 0}
        ner_i2l = {0: "O"}
        idx = 1
        for label_name in source_label2id.keys():
            if label_name in ["일반정보", "O"]: continue
            for prefix in ["B-", "I-"]:
                full_label = prefix + label_name
                if full_label not in ner_l2i:
                    ner_l2i[full_label] = idx
                    ner_i2l[idx] = full_label
                    idx += 1
        print(f"[NerGatPreprocessor] BIO Labels Created: {len(ner_l2i)} tags")
        return ner_l2i, ner_i2l

    def load_data(self, data_dir: List[str]) -> Tuple[Dict, Dict]:
        samples, annotations = {}, {}
        if isinstance(data_dir, str): data_dir = [data_dir]

        for d_path in data_dir:
            if not os.path.exists(d_path): continue
            for file_name in os.listdir(d_path):
                if not file_name.endswith(".json"): continue
                path = os.path.join(d_path, file_name)
                with open(path, "r", encoding='utf-8') as f:
                    try:
                        json_data = json.load(f)
                        json_list = json_data.get('data', [])
                        if isinstance(json_list, dict): json_list = [json_list]
                        
                        for item in json_list:
                            sent_id = item.get('id')
                            if not sent_id: continue
                            samples[sent_id] = {'sentence': item.get('sentence', ''), 'id': sent_id}
                            annotations[sent_id] = item.get('annotations', [])
                    except Exception as e:
                        print(f"[Error] Failed to load {file_name}: {e}")
        return samples, annotations

    def create_dataset(
        self, 
        samples: Dict, 
        annotations: Dict, 
        z_score_map: Dict[str, float],
        data_category: str = "personal_data"
    ) -> RobertaNerGatDataset:
        if self.ner_label2id is None:
            raise ValueError("Labels are not initialized.")
            
        return RobertaNerGatDataset(
            samples=samples,
            annotations=annotations,
            tokenizer=self.tokenizer,
            ner_label2id=self.ner_label2id,
            ner_id2label=self.ner_id2label,
            z_score_map=z_score_map,
            max_length=self.max_len,
            data_category=data_category
        )