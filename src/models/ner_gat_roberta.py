# src/models/ner_gat_roberta.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
from torch_geometric.nn import GATConv

class RobertaNerGatModel(nn.Module):
    """
    RoBERTa + GAT 기반 개체명 인식 모델
    - BERT Hidden State와 Sigmoid 스케일링된 z-score를 결합
    - GAT를 통해 Anchor Hub(중요 노드) 중심의 전역적 문맥 재구성
    """

    def __init__(
        self, 
        encoder: nn.Module, 
        num_classes: int, 
        gat_hidden_dim: int = None,
        use_focal: bool = False, 
        focal_alpha: float = 1.0, 
        focal_gamma: float = 2.0, 
        dropout_rate: Optional[float] = None
    ):
        super().__init__()

        # 1. BERT Encoder 설정 (Base/Large 대응)
        self.encoder = encoder
        self.bert_hidden_size = encoder.config.hidden_size
        self.num_classes = num_classes
        
        # 2. GAT 헤드 설정: 인코더의 멀티헤드 개수와 동기화 (보통 12 또는 16)
        self.gat_heads = encoder.config.num_attention_heads
        
        # 3. GAT 입력 차원: BERT 출력(768 or 1024) + z-score(1) = 769 or 1025
        self.gat_input_dim = self.bert_hidden_size + 1
        
        # 헤드당 차원 결정 (정의되지 않은 경우 입력 차원 보존 수준으로 설정)
        if gat_hidden_dim is None:
            gat_hidden_dim = self.bert_hidden_size // self.gat_heads
        
        # 4. GAT 레이어: edge_dim=1을 추가하여 edge_attr(가중치) 수용 가능하게 설정
        self.gat_conv = GATConv(
            in_channels=self.gat_input_dim, 
            out_channels=gat_hidden_dim, 
            heads=self.gat_heads, 
            dropout=dropout_rate if dropout_rate else 0.1,
            concat=True,
            edge_dim=1 # Sigmoid(z-score) 가중치를 반영하기 위한 설정
        )

        # 5. 최종 Classifier: GAT 출력(gat_hidden_dim * heads) -> 클래스 분류
        self.classifier = nn.Linear(gat_hidden_dim * self.gat_heads, num_classes)
        
        # 6. Loss 및 Dropout 설정
        drop_prob = dropout_rate if dropout_rate is not None else encoder.config.hidden_dropout_prob
        self.dropout = nn.Dropout(drop_prob)
        self.use_focal = use_focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        z_scores: torch.Tensor,      # (batch, seq_len)
        edge_index: torch.Tensor,    # (2, Total_Edges) - PyG Batching 처리됨
        edge_attr: torch.Tensor,     # (Total_Edges, 1) - 에지별 가중치
        labels: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        
        # [STEP 1] RoBERTa Encoding
        encoder_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if token_type_ids is not None:
            encoder_inputs['token_type_ids'] = token_type_ids
        
        outputs = self.encoder(**encoder_inputs)
        sequence_output = outputs.last_hidden_state 

        # [STEP 2] Sigmoid Scaling & Feature Fusion
        # z-score를 0~1 사이로 가두어 통계적 중요도를 피처로 주입
        z_scores_scaled = torch.sigmoid(z_scores).unsqueeze(-1) 
        combined_features = torch.cat([sequence_output, z_scores_scaled], dim=-1) # (B, S, 769)

        # [STEP 3] Graph 데이터 평탄화
        # PyG 연산을 위해 (Batch * Seq) 형태의 단일 노드 리스트로 변환
        batch_size, seq_len, feat_dim = combined_features.size()
        x = combined_features.view(-1, feat_dim) 

        # [STEP 4] GAT Passage (Message Passing)
        # edge_index(주소록)와 edge_attr(가중치)를 이용해 이웃 정보를 가중합
        x = self.dropout(x)
        x = self.gat_conv(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x) 

        # [STEP 5] 분류 및 출력 복원
        logits = self.classifier(x)
        logits = logits.view(batch_size, seq_len, self.num_classes)

        result = {"logits": logits}

        # [STEP 6] Loss 계산 (Focal or CrossEntropy)
        if labels is not None:
            if self.use_focal:
                result["loss"] = self._compute_focal_loss(logits, labels)
            else:
                result["loss"] = F.cross_entropy(
                    logits.view(-1, self.num_classes), 
                    labels.view(-1), 
                    ignore_index=-100
                )

        return result

    def _compute_focal_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits_flat = logits.view(-1, self.num_classes)
        labels_flat = labels.view(-1)
        ce_loss = F.cross_entropy(logits_flat, labels_flat, reduction="none", ignore_index=-100)
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        
        active_mask = (labels_flat != -100)
        return focal_loss[active_mask].mean()