# src/modules/ner_gat_trainer.py

import torch
from tqdm import tqdm
from datetime import datetime
from typing import Dict

class NerGatTrainer:
    """
    RoBERTa + GAT 하이브리드 모델의 학습(Training) 관리를 담당하는 클래스
    
    특징:
    1. Neural(문맥)과 Symbolic(통계) 결합 아키텍처 지원
    2. PyTorch Geometric(PyG)의 Batch 객체로부터 그래프 데이터(Edge, Attribute) 추출 및 모델 주입
    3. 전처리 단계에서 미리 계산된 토큰별 z-score를 활용하여 학습 효율 극대화
    """
    def __init__(self, model, optimizer, scheduler, device):
        """
        Args:
            model: RobertaNerGatModel 인스턴스
            optimizer: 학습 최적화 알고리즘 (AdamW 등)
            scheduler: Learning Rate Warmup/Decay 스케줄러
            device: GPU (cuda) 또는 CPU
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train_epoch(self, dataloader, epoch_idx: int) -> Dict:
        """
        1 Epoch 학습을 수행하고 통계 데이터를 반환합니다.
        
        Note: 
            전처리 단계(Dataset)에서 이미 DB의 z-score 매핑 및 그래프 생성이 완료되었으므로,
            본 루프에서는 추가적인 I/O 없이 DataLoader에서 데이터를 바로 꺼내어 사용합니다.
        """
        start_time = datetime.now()
        self.model.train()  # 모델을 학습 모드로 전환 (Dropout, BatchNorm 활성화)
        total_loss = 0
        
        # 학습 진행 상황 시각화 (Tqdm)
        desc = f"GAT Training (Epoch {epoch_idx})"
        
        # PyG DataLoader로부터 병합된 그래프 배치를 순회
        for batch in tqdm(dataloader, desc=desc, leave=False):
            
            # [STEP 1] 모델 입력 텐서 준비 및 GPU 전송
            # PyG 전용 Dataset 구조에 따라 필드별 데이터 추출
            input_ids = batch.x.to(self.device)              # 토큰 ID (B, S)
            attention_mask = batch.attention_mask.to(self.device) # 패딩 마스크 (B, S)
            labels = batch.y.to(self.device)                 # 정답 라벨 (B, S)
            
            # [STEP 2] GAT 전용 Symbolic 피처 추출
            # 전처리 시점에 주입된 토큰별 통계치와 그래프 연결 구조
            z_scores = batch.z_scores.to(self.device)        # 토큰별 z-score (B, S)
            edge_index = batch.edge_index.to(self.device)    # 그래프 간선 인덱스 (2, Total_Edges)
            edge_attr = batch.edge_attr.to(self.device)      # 간선 가중치 (Total_Edges, 1)

            # [STEP 3] Forward Pass: 하이브리드 인코딩 수행
            # 문맥 정보(Neural) + 통계 힌트(Z-score) + 전역 관계(GAT) 연산
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                z_scores=z_scores,
                edge_index=edge_index,
                edge_attr=edge_attr,
                labels=labels
            )
            
            # 모델 내부에서 계산된 Loss (Focal Loss 또는 CrossEntropy)
            loss = outputs['loss']

            # [STEP 4] Backward Pass & Parameter Optimization
            self.optimizer.zero_grad()  # 이전 그래디언트 초기화
            loss.backward()             # 역전파 수행
            
            # GAT 레이어의 안정적인 학습을 위한 Gradient Clipping (임계값 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()       # 가중치 업데이트
            
            # 스케줄러 단계 업데이트 (Warmup 등 반영)
            if self.scheduler:
                self.scheduler.step()
                
            total_loss += loss.item()
    
        # [STEP 5] 에폭 결과 요약
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        avg_loss = total_loss / len(dataloader)

        return {
            'loss': avg_loss,
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration
        }