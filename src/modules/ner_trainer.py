# src/modules/ner_trainer.py

import torch
from tqdm import tqdm
from datetime import datetime
from typing import Dict

class Trainer:
    """
    모델 학습(Training)을 담당하는 클래스
    """
    def __init__(self, model, optimizer, scheduler, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train_epoch(self, dataloader, epoch_idx: int) -> Dict:
        """
        1 Epoch 학습을 수행하고 결과를 반환합니다.
        """
        start_time = datetime.now()
        self.model.train()
        total_loss = 0
        
        # Tqdm 설명 문구
        desc = f"Training (Epoch {epoch_idx})"
        
        for batch in tqdm(dataloader, desc=desc, leave=False):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Forward Pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']

            # Backward Pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if self.scheduler:
                self.scheduler.step()
                
            total_loss += loss.item()
    
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        avg_loss = total_loss / len(dataloader)

        return {
            'loss': avg_loss,
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration
        }