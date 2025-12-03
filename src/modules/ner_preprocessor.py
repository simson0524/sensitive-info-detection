# src/modules/ner_preprocessor.py

import json
import torch
import os
from typing import List, Dict, Any, Tuple
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm

# ==============================================================================
# 1. NerDataset (데이터셋 생성 및 인스턴스화)
# ==============================================================================
class NerDataset(Dataset):
    """
    전처리된 데이터를 저장하고 PyTorch 모델에 공급하는 Dataset 클래스
    - 토큰화 (Tokenization)
    - BIO 라벨링 (Label Alignment)
    - 데이터 카테고리별 필터링
    """
    def __init__(
        self, 
        samples: Dict, 
        annotations: Dict, 
        tokenizer, 
        ner_label2id: Dict, 
        ner_id2label: Dict, 
        max_length: int = 256, 
        data_category: str = "personal_data"
    ):
        """
        Args:
            samples (Dict): {sentence_id: sentence_data} 형태의 샘플 데이터
            annotations (Dict): {sentence_id: annotation_list} 형태의 라벨 데이터
            tokenizer: HuggingFace Tokenizer
            ner_label2id (Dict): BIO 태그 -> ID 매핑
            ner_id2label (Dict): ID -> BIO 태그 매핑
            max_length (int): 시퀀스 최대 길이
            data_category (str): "personal_data" 또는 "confidential_data"
        """
        self.samples = samples
        self.annotations = annotations
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ner_label2id = ner_label2id
        self.ner_id2label = ner_id2label
        self.data_category = data_category
        self.instances = []
        
        if self.data_category not in ["personal_data", "confidential_data"]:
            raise ValueError(f"Invalid data_category: {self.data_category}")

        self._create_instances()

    def _create_instances(self):
        """
        Raw 데이터를 순회하며 모델 입력용 인스턴스(Token ID, Label ID 등)를 생성합니다.
        """
        # tqdm으로 진행 상황 표시
        for sent_id, sample_data in tqdm(self.samples.items(), desc="Create NER instances"):
            sentence = sample_data['sentence']
            # sentence_id는 key값으로 이미 넘어오므로 sample_data['id']와 동일함
            
            # domain_id 파싱 (예: doc_01_005 -> 1)
            # 실패 시 기본값 "0" 할당
            try:
                domain_id = str(int(sent_id.split('_')[1]))
            except:
                domain_id = "0"
            
            # Optional 필드들 (없으면 기본값)
            sentence_seq = sample_data.get('sequence', 0)
            filename = sample_data.get('filename', "")

            # 1. 토큰화 및 Offset Mapping 생성
            # return_offsets_mapping=True: 원본 문장의 char index와 토큰 index를 매핑하기 위함
            encoding = self.tokenizer(
                sentence,
                max_length=self.max_length,
                truncation=True,
                return_offsets_mapping=True,
                padding='max_length'
            )
            
            input_ids = encoding['input_ids']
            decoded_input_ids = self.tokenizer.convert_ids_to_tokens(input_ids)
            attention_mask = encoding['attention_mask']
            offset_mapping = encoding["offset_mapping"]

            # 2. 라벨 텐서 초기화 (모든 토큰을 "O"로 설정)
            labels = [self.ner_label2id["O"]] * len(input_ids)
            
            # 3. Annotations 기반으로 라벨 할당 (Token Alignment)
            # 해당 문장에 대한 정답 라벨 리스트 가져오기
            sentence_annotations = self.annotations.get(sent_id, [])
            
            for ann in sentence_annotations:
                ann_label = ann['label']
                char_start = ann['start']
                char_end = ann['end']

                # [필터링] data_category에 따라 학습할 라벨 선별
                if self.data_category == "personal_data":
                    if ann_label == "기밀정보": continue
                elif self.data_category == "confidential_data":
                    if ann_label in ["준식별자", "개인정보"]: continue
                
                if ann_label == "일반정보": continue

                # BIO 태그 이름 생성
                b_label_name = f"B-{ann_label}"
                i_label_name = f"I-{ann_label}"

                # 라벨 맵에 없는 태그는 스킵 (방어 코드)
                if b_label_name not in self.ner_label2id:
                    continue
                
                b_label_id = self.ner_label2id[b_label_name]
                i_label_id = self.ner_label_2_id[i_label_name]

                # 4. Character Index -> Token Index 변환 로직
                token_start = None
                token_end = None

                for i, (offset_start, offset_end) in enumerate(offset_mapping):
                    # 스페셜 토큰([CLS], [SEP] 등)은 offset이 (0,0)이므로 스킵
                    if offset_start == 0 and offset_end == 0: continue
                    
                    # 시작 토큰 찾기
                    if (token_start is None) and (offset_start <= char_start < offset_end):
                        token_start = i
                    
                    # 끝 토큰 찾기 (char_end는 포함되지 않음, < 범위 주의)
                    if (token_end is None) and (offset_start < char_end <= offset_end):
                        token_end = i 

                # 매핑 실패 시 스킵
                if token_start is None or token_end is None:
                    continue

                # 5. BIO 라벨 할당 (Start는 B, 나머지는 I)
                labels[token_start] = b_label_id
                for i in range(token_start + 1, token_end + 1):
                    if i < len(labels):
                        labels[i] = i_label_id
            
            # 6. 스페셜 토큰 위치에 -100 할당 (Loss 계산 제외)
            for i, (offset_start, offset_end) in enumerate(offset_mapping):
                if offset_start == 0 and offset_end == 0:
                    labels[i] = -100
            
            # 7. 최종 인스턴스 추가
            self.instances.append({
                "sentence": sentence,
                "sentence_id": sent_id,
                "domain_id": domain_id,
                "sentence_seq": sentence_seq,
                "input_ids": input_ids,
                "decoded_input_ids": decoded_input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "is_validated": False,
                "file_name": filename
            })

    def __getitem__(self, idx):
        item = self.instances[idx]
        
        # PyTorch 텐서로 변환하여 반환
        return {
            "idx": torch.tensor(idx),
            "sentence": item['sentence'],
            "sentence_id": item['sentence_id'],
            "domain_id": item['domain_id'],
            "sentence_seq": item['sentence_seq'],
            "input_ids": torch.tensor(item["input_ids"]),
            "decoded_input_ids": item["decoded_input_ids"],
            "attention_mask": torch.tensor(item["attention_mask"]),
            "labels": torch.tensor(item["labels"]),
            "is_validated": item["is_validated"],
            "file_name": item["file_name"]
        }
    
    def __len__(self):
        return len(self.instances)


# ==============================================================================
# 2. NerPreprocessor (데이터 로드 및 라벨 관리)
# ==============================================================================
class NerPreprocessor:
    """
    데이터 로드, BIO 라벨 맵 생성, Dataset 생성을 관리하는 클래스
    """
    def __init__(self, tokenizer: PreTrainedTokenizerFast, max_len: int = 256, label2id: Dict[str, int] = None):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.source_label2id = label2id
        
        # BIO 태깅용 라벨 맵 생성
        if label2id:
            self.ner_label2id, self.ner_id2label = self._build_bio_labels(label2id)
        else:
            self.ner_label2id = None
            self.ner_id2label = None

    def _build_bio_labels(self, source_label2id: Dict[str, int]) -> Tuple[Dict, Dict]:
        """
        기존 레이블을 NER(BIO)용 레이블로 변환 (O, B-PER, I-PER ...)
        """
        ner_l2i = {"O": 0}
        ner_i2l = {0: "O"}
        idx = 1
        
        for label_name in source_label2id.keys():
            if label_name in ["일반정보", "O"]:
                continue
            
            for prefix in ["B-", "I-"]:
                full_label = prefix + label_name
                if full_label not in ner_l2i:
                    ner_l2i[full_label] = idx
                    ner_i2l[idx] = full_label
                    idx += 1
        
        print(f"[NerPreprocessor] BIO Labels Created: {len(ner_l2i)} tags")
        return ner_l2i, ner_i2l

    def load_data(self, data_dir: str) -> Tuple[Dict, Dict]:
        """
        [수정됨] 변경된 JSON 포맷에 맞춰 데이터 로드
        - 각 파일은 리스트 형태일 수도 있고, JSON Lines(한 줄에 하나)일 수도 있음을 고려
        - 여기서는 "하나의 파일에 여러 JSON 객체가 리스트로 들어있는 구조"를 가정
        """
        samples = {}
        annotations = {}
        
        print(f"[NerPreprocessor] Loading JSON from {data_dir}...")
        if not os.path.exists(data_dir):
             print(f"[Warning] Directory not found: {data_dir}")
             return samples, annotations

        for file_name in os.listdir(data_dir):
            if not file_name.endswith(".json"):
                continue
                
            path = os.path.join(data_dir, file_name)
            with open(path, "r", encoding='utf-8') as f:
                try:
                    # 전체 파일을 읽어서 JSON 파싱 (리스트 형태 예상)
                    json_data = json.load(f)
                    
                    # 만약 파일 하나가 단일 객체라면 리스트로 감싸줌
                    if isinstance(json_data, dict):
                        json_data = [json_data]
                    
                    # 데이터 파싱 Loop
                    for item in json_data:
                        # 필수 필드 추출
                        sent_id = item.get('id')
                        if not sent_id: continue
                        
                        # Sample 데이터 저장
                        samples[sent_id] = {
                            'sentence': item.get('sentence', ''),
                            'id': sent_id,
                            'filename': item.get('filename', ''),
                            'sequence': item.get('sequence', 0)
                        }
                        
                        # Annotation 데이터 저장 (리스트)
                        annotations[sent_id] = item.get('annotations', [])
                        
                except Exception as e:
                    print(f"[Error] Failed to load {file_name}: {e}")
                    
        return samples, annotations

    def create_dataset(
        self, 
        samples: Dict, 
        annotations: Dict, 
        data_category: str = "personal_data"
    ) -> NerDataset:
        """
        NerDataset 객체 생성 (팩토리 메서드)
        """
        if self.ner_label2id is None:
            raise ValueError("Labels are not initialized. Please provide label2id.")
            
        return NerDataset(
            samples=samples,
            annotations=annotations,
            tokenizer=self.tokenizer,
            ner_label2id=self.ner_label2id,
            ner_id2label=self.ner_id2label, # ID -> Label 맵 전달
            max_length=self.max_len,
            data_category=data_category
        )