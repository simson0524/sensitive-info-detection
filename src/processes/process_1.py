# src/processes/process_1.py

import torch
import os
import logging

# 1. Modules
from src.modules.ner_trainer import Trainer
from src.modules.ner_evaluator import Evaluator

# 2. Database
from src.database.connection import db_manager
from src.database import crud

# 3. Utils
from src.utils.visualizer import plot_loss_graph
from src.utils.common import ensure_dir

def run_process_1(config: dict, context: dict):
    """
    [Process 1] 모델 학습 및 검증 루프 (Execution Phase)
    - Process 0에서 생성된 객체들을 사용하여 실제 학습 및 검증을 수행
    - DB experiment_process_results 테이블에 Epoch 단위로 통합 로그 저장
    """
    
    # --- [Step 1] Context Unpacking & Setup ---
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

    # --- [Step 2] Worker 모듈 초기화 ---
    trainer = Trainer(model, optimizer, scheduler, device)
    evaluator = Evaluator(
        model, 
        device, 
        preprocessor.tokenizer, 
        preprocessor.ner_id2label 
    )

    # --- [Step 3] 학습 루프 (Training Loop) ---
    best_f1 = 0.0
    min_valid_loss = float('inf')
    best_f1_epoch = -1
    min_loss_epoch = -1

    train_losses = []
    valid_losses = []
    
    # 체크포인트 저장 경로: outputs/checkpoints/{experiment_code}/
    ckpt_save_dir = os.path.join(path_conf['checkpoint_dir'], experiment_code)
    ensure_dir(ckpt_save_dir)

    # DB 세션 시작
    with db_manager.get_db() as session:
        for epoch in range(1, train_conf['epochs'] + 1):
            logger.info(f"=== Epoch {epoch}/{train_conf['epochs']} ===")
            
            # -----------------------------------------------------------
            # 3-1. 학습 (Train)
            # -----------------------------------------------------------
            train_result = trainer.train_epoch(train_loader, epoch)
            train_losses.append(train_result['loss'])
            
            # -----------------------------------------------------------
            # 3-2. 검증 (Validation)
            # -----------------------------------------------------------
            valid_result = evaluator.evaluate(valid_loader, mode="valid")
            valid_metrics = valid_result['metrics']
            valid_logs = valid_result['logs']
            valid_losses.append(valid_metrics['loss'])
            
            # -----------------------------------------------------------
            # 3-3. 결과 통합 및 DB 저장 (Epoch 단위)
            # -----------------------------------------------------------
            
            # (1) 모든 지표를 하나의 딕셔너리로 통합 (JSONB용)
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
            
            # (2) experiment_process_results 테이블에 저장 (1 Row per Epoch)
            # process_code는 'model_train' 또는 'process_1' 등으로 지정
            crud.create_process_result(session, {
                "experiment_code": experiment_code,
                "process_code": "process_1",
                "process_epoch": epoch,
                
                # 시간 정보는 전체(Train + Valid) 기준으로 기록
                "process_start_time": train_result['start_time'], 
                "process_end_time": valid_result['end_time'], 
                "process_duration": train_result['duration'] + valid_result['duration'],
                
                # 핵심: 여기에 모든 지표가 JSON으로 들어감
                "process_results": epoch_all_metrics 
            })
            
            logger.info(f"Epoch {epoch} Result Saved. (Train Loss: {train_result['loss']:.4f} | Valid F1: {valid_metrics['f1']:.4f})")

            # (3) 문장 단위 추론 결과 DB 저장 (Bulk Insert)
            # FK 정보 주입 (부모 테이블의 PK와 일치해야 함)
            for log in valid_logs:
                log['experiment_code'] = experiment_code
                log['process_code'] = "process_1"
                log['process_epoch'] = epoch
            
            crud.bulk_insert_inference_sentences(session, valid_logs)
            logger.info(f"Saved {len(valid_logs)} inference logs to DB.")

            # -----------------------------------------------------------
            # 3-4. 체크포인트 저장
            # -----------------------------------------------------------
            save_name = f"epoch_{epoch}.pt"
            save_path = os.path.join(ckpt_save_dir, save_name)
            torch.save(model.state_dict(), save_path)
            
            # Best 기록
            if valid_metrics['f1'] > best_f1:
                best_f1 = valid_metrics['f1']
                best_f1_epoch = epoch
                logger.info(f"✨ Current Best F1: {best_f1:.4f} (Epoch {epoch})")
            
            if valid_metrics['loss'] < min_valid_loss:
                min_valid_loss = valid_metrics['loss']
                min_loss_epoch = epoch
                logger.info(f"📉 Current Min Loss: {min_valid_loss:.4f} (Epoch {epoch})")

        # --- [Step 4] 메타데이터 업데이트 ---
        exp_obj = crud.get_experiment(session, experiment_code)
        if exp_obj:
            current_config = exp_obj.experiment_config or {}
            
            best_f1_path = os.path.join(ckpt_save_dir, f"epoch_{best_f1_epoch}.pt")
            min_loss_path = os.path.join(ckpt_save_dir, f"epoch_{min_loss_epoch}.pt")

            current_config['best_model_f1_path'] = best_f1_path
            current_config['best_model_loss_path'] = min_loss_path
            current_config['best_f1_score'] = best_f1
            current_config['min_valid_loss'] = min_valid_loss
            
            crud.update_experiment(session, experiment_code, {
                "experiment_config": current_config,
            })
            logger.info(f"✅ Experiment Meta Updated. (Best F1 Epoch: {best_f1_epoch})")

    # --- [Step 5] 마무리 및 시각화 ---
    save_dir = os.path.join(path_conf['log_dir'], experiment_code)
    
    plot_loss_graph(
        train_losses, 
        valid_losses, 
        save_dir, 
        experiment_code
    )
    
    logger.info("[Process 1] Process Completed Successfully.")
    
    return context