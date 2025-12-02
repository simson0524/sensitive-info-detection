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
from src.utils.visualizer import plot_loss_graph
from src.utils.common import ensure_dir

def run_process_1(config: dict, context: dict):
    """
    [Process 1] 모델 학습 및 검증 루프 (Execution Phase)
    
    Process 0에서 준비된 모델과 데이터셋을 받아 실제 학습(Train)과 검증(Valid)을 수행합니다.
    매 Epoch마다 결과 지표를 DB에 저장하고, 모델 가중치를 파일로 저장합니다.

    Args:
        config (dict): 설정 파일 내용 (experiment_config.yaml)
        context (dict): Process 0에서 생성된 객체들 (모델, 옵티마이저, 데이터로더 등)

    Returns:
        dict: 학습된 모델이 포함된 갱신된 Context
    """
    
    # ==============================================================================
    # [Step 1] Context Unpacking & Setup (준비 단계)
    # ==============================================================================
    # Process 0에서 넘어온 객체들을 사용하기 좋게 변수에 할당합니다.
    experiment_code = context['experiment_code']
    device = context['device']
    model = context['model']
    optimizer = context['optimizer']
    scheduler = context['scheduler']
    train_loader = context['train_loader']
    valid_loader = context['valid_loader']
    preprocessor = context['preprocessor'] # 토크나이저와 라벨맵을 포함하고 있음

    train_conf = config['train']
    path_conf = config['path']

    # 로거 설정: run_experiment.py에서 생성된 로거를 가져옵니다.
    logger = logging.getLogger(experiment_code)
    logger.info(f"🚀 [Process 1] Start Training Loop for {experiment_code}")

    # ==============================================================================
    # [Step 2] Worker 모듈 초기화
    # ==============================================================================
    # Trainer: 학습 데이터셋을 순회하며 역전파(Backprop)를 수행하는 객체
    trainer = Trainer(model, optimizer, scheduler, device)
    
    # Evaluator: 검증 데이터셋을 순회하며 메트릭 계산 및 오답 노트를 작성하는 객체
    evaluator = Evaluator(
        model, 
        device, 
        preprocessor.tokenizer, 
        preprocessor.ner_id2label # ID(0,1..)를 라벨(O, B-PER..)로 변환하기 위해 필요
    )

    # ==============================================================================
    # [Step 3] 학습 루프 (Training Loop)
    # ==============================================================================
    # 최고 성능 기록을 추적하기 위한 변수 초기화
    best_f1 = 0.0
    min_valid_loss = float('inf')
    best_f1_epoch = -1
    min_loss_epoch = -1

    # 그래프 그리기 위한 리스트
    train_losses = []
    valid_losses = []
    
    # 체크포인트 저장 경로 생성: outputs/checkpoints/{experiment_code}/
    ckpt_save_dir = os.path.join(path_conf['checkpoint_dir'], experiment_code)
    ensure_dir(ckpt_save_dir)

    # DB 세션 시작 (루프 전체를 하나의 세션 컨텍스트에서 처리)
    with db_manager.get_db() as session:
        for epoch in range(1, train_conf['epochs'] + 1):
            logger.info(f"=== Epoch {epoch}/{train_conf['epochs']} ===")
            
            # -----------------------------------------------------------
            # 3-1. 학습 (Train Phase)
            # -----------------------------------------------------------
            # Trainer가 1 Epoch 학습을 수행하고 Loss와 소요시간을 반환
            train_result = trainer.train_epoch(train_loader, epoch)
            train_losses.append(train_result['loss'])
            
            # -----------------------------------------------------------
            # 3-2. 검증 (Validation Phase)
            # -----------------------------------------------------------
            # Evaluator가 추론을 수행하고 메트릭과 상세 로그(오답 포함)를 반환
            # mode='valid': Ground Truth와 비교하여 정답 여부를 판단함
            valid_result = evaluator.evaluate(valid_loader, mode="valid")
            
            valid_metrics = valid_result['metrics'] # Loss, F1, Precision, Recall, Confusion Matrix
            valid_logs = valid_result['logs']       # 문장별 상세 추론 결과 (DB 저장용 List[Dict])
            valid_losses.append(valid_metrics['loss'])
            
            # -----------------------------------------------------------
            # 3-3. 결과 통합 및 DB 저장 (Epoch 단위 요약)
            # -----------------------------------------------------------
            
            # (1) 모든 지표를 하나의 딕셔너리로 통합 (JSONB 컬럼에 저장될 데이터)
            epoch_all_metrics = {
                "train_loss": train_result['loss'],
                "train_time": train_result['duration'],
                "valid_loss": valid_metrics['loss'],
                "valid_precision": valid_metrics['precision'],
                "valid_recall": valid_metrics['recall'],
                "valid_f1": valid_metrics['f1'],
                "valid_time": valid_result['duration'],
                "confusion_matrix": valid_metrics['confusion_matrix'] # List[List[int]] 형태
            }
            
            # (2) experiment_process_results 테이블에 저장 (1 Row per Epoch)
            # 이 테이블은 실험의 시계열적 변화(Loss 감소 등)를 기록합니다.
            crud.create_process_result(session, {
                "experiment_code": experiment_code,
                "process_code": "process_1", # 학습 프로세스 식별자
                "process_epoch": epoch,
                
                # 시간 정보: Train + Valid 전체 소요 시간
                "process_start_time": train_result['start_time'], 
                "process_end_time": valid_result['end_time'], 
                "process_duration": train_result['duration'] + valid_result['duration'],
                
                # 핵심: 모든 지표가 담긴 JSON
                "process_results": epoch_all_metrics 
            })
            
            logger.info(f"Epoch {epoch} Result Saved. (Train Loss: {train_result['loss']:.4f} | Valid F1: {valid_metrics['f1']:.4f})")

            # (3) 문장 단위 추론 결과 DB 저장 (Bulk Insert)
            # 오답 분석을 위해 모든 검증 문장의 결과를 저장합니다.
            # FK 정보(실험코드, 프로세스코드, 에폭)를 로그 딕셔너리에 주입
            for log in valid_logs:
                log['experiment_code'] = experiment_code
                log['process_code'] = "process_1"
                log['process_epoch'] = epoch
            
            # 대량 데이터 삽입 (속도 최적화)
            crud.bulk_insert_inference_sentences(session, valid_logs)
            logger.info(f"Saved {len(valid_logs)} inference logs to DB.")

            # -----------------------------------------------------------
            # 3-4. 체크포인트 저장 (Model Checkpoint)
            # -----------------------------------------------------------
            # 모든 Epoch의 모델을 저장합니다 (나중에 분석하거나 Resume 할 때 사용)
            save_name = f"epoch_{epoch}.pt"
            save_path = os.path.join(ckpt_save_dir, save_name)
            torch.save(model.state_dict(), save_path)
            
            # Best F1 Score 갱신 여부 확인
            if valid_metrics['f1'] > best_f1:
                best_f1 = valid_metrics['f1']
                best_f1_epoch = epoch
                logger.info(f"✨ Current Best F1: {best_f1:.4f} (Epoch {epoch})")
            
            # Min Loss 갱신 여부 확인 (Overfitting 감지용)
            if valid_metrics['loss'] < min_valid_loss:
                min_valid_loss = valid_metrics['loss']
                min_loss_epoch = epoch
                logger.info(f"📉 Current Min Loss: {min_valid_loss:.4f} (Epoch {epoch})")

        # ==============================================================================
        # [Step 4] 실험 메타데이터 업데이트 (DB Update)
        # ==============================================================================
        # 학습 종료 후, 가장 성능이 좋았던 모델의 경로를 Experiment 테이블에 기록합니다.
        # 추후 Inference 단계에서 이 경로를 참조하여 모델을 로드합니다.
        
        exp_obj = crud.get_experiment(session, experiment_code)
        if exp_obj:
            current_config = exp_obj.experiment_config or {}
            
            # 최고 성능 모델 경로 구성
            best_f1_path = os.path.join(ckpt_save_dir, f"epoch_{best_f1_epoch}.pt")
            min_loss_path = os.path.join(ckpt_save_dir, f"epoch_{min_loss_epoch}.pt")
            
            # Config JSON 업데이트
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
    # 학습 종료 후 Loss 그래프를 그려서 저장합니다.
    save_dir = os.path.join(path_conf['log_dir'], experiment_code)
    
    plot_loss_graph(
        train_losses, 
        valid_losses, 
        save_dir, 
        experiment_code
    )
    
    logger.info("[Process 1] Process Completed Successfully.")
    
    # 학습된 모델 객체를 포함하여 Context 반환
    return context