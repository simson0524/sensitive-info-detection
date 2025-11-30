# src/models/ner_roberta.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Union

class RobertaNerModel(nn.Module):
    """
    BERT/RoBERTa 기반의 개체명 인식(NER) 모델
    - Focal Loss 지원
    - HuggingFace Transformers 모델을 Encoder로 사용
    """

    def __init__(
        self, 
        encoder: nn.Module, 
        num_classes: int, 
        use_focal: bool = False, 
        focal_alpha: float = 1.0, 
        focal_gamma: float = 2.0, 
        loss_reduction: str = "mean",
        dropout_rate: Optional[float] = None
    ):
        """
        Args:
            encoder (nn.Module): Pretrained Model (e.g. RobertaModel)
            num_classes (int): 최종 분류할 레이블의 개수 (len(id2label))
                               ※ 주의: 이전 코드의 (num_labels * 2) - 1 계산은 외부에서 처리 후 넘겨주세요.
            use_focal (bool): Focal Loss 사용 여부
            focal_alpha (float): Focal Loss Alpha 값
            focal_gamma (float): Focal Loss Gamma 값
            loss_reduction (str): Loss Reduction 방식 ('mean', 'sum', 'none')
            dropout_rate (float, optional): Classifier Head의 Dropout 비율. None이면 Encoder 설정 따름.
        """
        super().__init__()

        # 1. Encoder 설정
        self.encoder = encoder
        self.hidden_size = encoder.config.hidden_size
        
        # 2. Classifier 설정
        self.num_classes = num_classes
        
        # Dropout: 입력이 없으면 Encoder의 config를 따름
        drop_prob = dropout_rate if dropout_rate is not None else encoder.config.hidden_dropout_prob
        self.dropout = nn.Dropout(drop_prob)
        
        # Linear Head
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)

        # 3. Loss 설정
        self.use_focal = use_focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.loss_reduction = loss_reduction

    def _compute_focal_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        NER용 Focal Loss 계산 (내부 메서드)
        """
        # 1. 차원 평탄화 (Flatten) -> (Batch * Seq, Num_Classes)
        logits_flat = logits.view(-1, self.num_classes)
        labels_flat = labels.view(-1)

        # 2. Cross Entropy Loss 계산 (reduction='none'으로 개별 Loss 구함)
        #    ignore_index=-100인 토큰(Padding, Special Token)은 0으로 처리됨
        ce_loss = F.cross_entropy(logits_flat, labels_flat, reduction="none", ignore_index=-100)

        # 3. Pt 계산 (확률)
        pt = torch.exp(-ce_loss)

        # 4. Focal Loss 수식 적용: alpha * (1-pt)^gamma * log(pt)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss

        # 5. Reduction (ignore_index 제외한 유효 토큰에 대해서만 처리)
        if self.loss_reduction == "none":
            return focal_loss
        
        # -100이 아닌(유효한) 토큰들만 마스킹하여 평균/합 계산
        active_mask = (labels_flat != -100)
        active_loss = focal_loss[active_mask]

        if self.loss_reduction == "mean":
            return active_loss.mean()
        elif self.loss_reduction == "sum":
            return active_loss.sum()
        
        return active_loss.mean() # Fallback

    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        labels: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward Pass
        
        Returns:
            dict: {'logits': Tensor, 'loss': Tensor(optional)}
        """
        
        # 1. Encoder 통과
        # RoBERTa는 token_type_ids가 필요 없지만, BERT는 필요할 수 있음 (kwargs 처리 가능)
        encoder_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
        if token_type_ids is not None:
            encoder_inputs['token_type_ids'] = token_type_ids

        outputs = self.encoder(**encoder_inputs)
        
        # last_hidden_state: (batch_size, seq_len, hidden_size)
        sequence_output = outputs.last_hidden_state

        # 2. Classifier Head
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output) # (batch_size, seq_len, num_classes)

        result = {"logits": logits}

        # 3. Loss Calculation (학습 시)
        if labels is not None:
            if self.use_focal:
                loss = self._compute_focal_loss(logits, labels)
            else:
                loss = F.cross_entropy(
                    logits.view(-1, self.num_classes), 
                    labels.view(-1), 
                    ignore_index=-100
                )
            result["loss"] = loss

        return result