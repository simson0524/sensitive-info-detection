# src/processes/process_1.py

import torch
import os
import logging
from datetime import datetime

# 1. Modules: 학습과 검증을 담당하는 모듈
from src.modules.ner_trainer import Trainer
from src.modules.ner_evaluator import Evaluator

# 2. Database: DB 연결 및 CRUD 유틸리티
from src.database.connection import db_manager
from src.database import crud

# 3. Utils: 시각화 및 파일 시스템 관련 도구
from src.utils.visualizer import plot_loss_graph, plot_confusion_matrix_trends 
from src.utils.common import ensure_dir, save_logs_to_csv

def run_process_1(config: dict, context: dict):
    """
    [Process 1] 모델 학습 및 검증 루프 (Execution Phase)
    
    Process 0에서 준비된 모델과 데이터셋을 받아 실제 학습(Train)과 검증(Valid)을 수행합니다.
    매 Epoch마다 결과 지표를 DB에 저장하고, 모델 가중치와 추론 결과를 파일(pt, csv)로 저장합니다.

    Args:
        config (dict): 설정 파일 내용 (experiment_config.yaml)
        context (dict): Process 0에서 생성된 객체들 (모델, 옵티마이저, 데이터로더 등)

    Returns:
        dict: 학습된 모델이 포함된 갱신된 Context
    """
    
    # ==============================================================================
    # [Step 1] Context Unpacking & Setup (준비 단계)
    # ==============================================================================
    experiment_code = context['experiment_code']
    device = context['device']
    model = context['model']
    optimizer = context['optimizer']
    scheduler = context['scheduler']
    train_loader = context['train_loader']
    valid_loader = context['valid_loader']
    preprocessor = context['preprocessor'] 

    train_conf = config['train']
    path_conf = config['path']

    # 로거 설정
    logger = logging.getLogger(experiment_code)
    logger.info(f"🚀 [Process 1] Start Training Loop for {experiment_code}")

    # ==============================================================================
    # [Step 2] Worker 모듈 초기화
    # ==============================================================================
    trainer = Trainer(model, optimizer, scheduler, device)
    
    evaluator = Evaluator(
        model, 
        device, 
        preprocessor.tokenizer, 
        preprocessor.ner_id2label
    )

    # ==============================================================================
    # [Step 3] 학습 루프 (Training Loop)
    # ==============================================================================
    best_f1 = 0.0
    min_valid_loss = float('inf')
    best_f1_epoch = -1
    min_loss_epoch = -1

    train_losses = []
    valid_losses = []

    cm_history = [] # Graph를 위한 confusion_matrix history
    
    # 저장 경로 준비
    ckpt_save_dir = os.path.join(path_conf['checkpoint_dir'], experiment_code)
    log_save_dir = os.path.join(path_conf['log_dir'], experiment_code) 
    ensure_dir(ckpt_save_dir)
    ensure_dir(log_save_dir)

    # DB 세션 시작
    with db_manager.get_db() as session:
        for epoch in range(1, train_conf['epochs'] + 1):
            logger.info(f"=== Epoch {epoch}/{train_conf['epochs']} ===")
            
            # -----------------------------------------------------------
            # 3-1. 학습 (Train Phase)
            # -----------------------------------------------------------
            train_result = trainer.train_epoch(train_loader, epoch)
            train_losses.append(train_result['loss'])
            
            # -----------------------------------------------------------
            # 3-2. 검증 (Validation Phase)
            # -----------------------------------------------------------
            valid_result = evaluator.evaluate(valid_loader, mode="valid")
            
            valid_metrics = valid_result['metrics']
            valid_logs = valid_result['logs']
            valid_losses.append(valid_metrics['loss'])

            if 'confusion_matrix' in valid_metrics:
                cm_history.append(valid_metrics['confusion_matrix'])
            
            # -----------------------------------------------------------
            # 3-3. 결과 통합 및 DB 저장 (Epoch 단위 요약)
            # -----------------------------------------------------------
            
            # (1) 모든 지표 통합 (JSONB용)
            epoch_all_metrics = {
                "train_loss": train_result['loss'],
                "train_time": train_result['duration'],
                "valid_loss": valid_metrics['loss'],
                "valid_precision": valid_metrics['precision'],
                "valid_recall": valid_metrics['recall'],
                "valid_f1": valid_metrics['f1'],
                "valid_time": valid_result['duration'],
                "confusion_matrix": valid_metrics['confusion_matrix']
            }
            
            # (2) DB 저장 (ExperimentProcessResult)
            crud.create_process_result(session, {
                "experiment_code": experiment_code,
                "process_code": "process_1",
                "process_epoch": epoch,
                "process_start_time": train_result['start_time'], 
                "process_end_time": valid_result['end_time'], 
                "process_duration": train_result['duration'] + valid_result['duration'],
                "process_results": epoch_all_metrics 
            })
            
            logger.info(f"Epoch {epoch} Result Saved. (Train Loss: {train_result['loss']:.4f} | Valid F1: {valid_metrics['f1']:.4f})")

            # -----------------------------------------------------------
            # 3-3-2. 문장 단위 추론 결과 저장 (DB + CSV) [UPDATED]
            # -----------------------------------------------------------
            # FK 정보 주입
            for log in valid_logs:
                log['experiment_code'] = experiment_code
                log['process_code'] = "process_1"
                log['process_epoch'] = epoch
            
            # (1) DB Bulk Insert
            crud.bulk_insert_inference_sentences(session, valid_logs)
            logger.info(f"Saved {len(valid_logs)} inference logs to DB.")

            # (2) [NEW] CSV 파일 추출 및 저장
            csv_file_name = f"{experiment_code}_process_1_{epoch}_inference_sentences.csv"
            csv_file_path = os.path.join(log_save_dir, csv_file_name)
            
            # 유틸리티 함수 호출 (JSON 필드는 문자열로 변환되어 저장됨)
            save_logs_to_csv(valid_logs, csv_file_path)

            # -----------------------------------------------------------
            # 3-4. 체크포인트 저장 (Model Checkpoint)
            # -----------------------------------------------------------
            # 파일명 통일: {code}_epoch_{epoch}.pt
            save_name = f"{experiment_code}_epoch_{epoch}.pt"
            save_path = os.path.join(ckpt_save_dir, save_name)
            torch.save(model.state_dict(), save_path)
            
            # Best 기록 갱신
            if valid_metrics['f1'] > best_f1:
                best_f1 = valid_metrics['f1']
                best_f1_epoch = epoch
                logger.info(f"✨ Current Best F1: {best_f1:.4f} (Epoch {epoch})")
            
            if valid_metrics['loss'] < min_valid_loss:
                min_valid_loss = valid_metrics['loss']
                min_loss_epoch = epoch
                logger.info(f"📉 Current Min Loss: {min_valid_loss:.4f} (Epoch {epoch})")

        # ==============================================================================
        # [Step 4] 실험 메타데이터 업데이트 (DB Update)
        # ==============================================================================
        exp_obj = crud.get_experiment(session, experiment_code)
        if exp_obj:
            current_config = exp_obj.experiment_config or {}
            
            # 최고 성능 모델 경로 구성 (파일명 형식 일치시킴)
            best_f1_path = os.path.join(ckpt_save_dir, f"{experiment_code}_epoch_{best_f1_epoch}.pt")
            min_loss_path = os.path.join(ckpt_save_dir, f"{experiment_code}_epoch_{min_loss_epoch}.pt")
            
            current_config['best_model_f1_path'] = best_f1_path
            current_config['best_model_loss_path'] = min_loss_path
            current_config['best_f1_score'] = best_f1
            current_config['min_valid_loss'] = min_valid_loss
            
            crud.update_experiment(session, experiment_code, {
                "experiment_config": current_config,
            })
            logger.info(f"✅ Experiment Meta Updated. (Best F1 Epoch: {best_f1_epoch})")

    # ==============================================================================
    # [Step 5] 마무리 및 시각화 (Finalize)
    # ==============================================================================
    # 학습 종료 후 Loss 그래프 저장 (log_save_dir 사용)
    plot_loss_graph(
        train_losses, 
        valid_losses, 
        log_save_dir, 
        experiment_code
    )

    # 2. [NEW] Label Distribution Graph (Confusion Matrix Trend)
    # preprocessor.ner_id2label을 넘겨줘서 ID를 라벨명(B-PER 등)으로 변환
    plot_confusion_matrix_trends(
        cm_history, 
        preprocessor.ner_id2label, 
        log_save_dir, 
        experiment_code
    )
    
    logger.info("[Process 1] Process Completed Successfully.")
    
    # 학습된 모델 객체를 포함하여 Context 반환
    return context