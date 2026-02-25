# src/processes/process_1.py

import torch
import os
import logging
from datetime import datetime

# 1. Modules: 모델 학습(Trainer)과 엔티티 기반 평가(Evaluator) 모듈
from src.modules.ner_trainer import Trainer
from src.modules.ner_evaluator import Evaluator

# 2. Database: DB 세션 관리 및 결과 적재 CRUD
from src.database.connection import db_manager
from src.database import crud

# 3. Utils: 시각화 도구 및 공통 유틸리티
from src.utils.visualizer import (
    plot_loss_graph, 
    plot_confusion_matrix_trends, 
    plot_label_relation_matrix,
    plot_label_accuracy_histograms
)
from src.utils.common import ensure_dir, save_logs_to_csv

def run_process_1(config: dict, context: dict):
    """
    [Process 1] 모델 학습 및 엔티티 레벨 다각도 검증 프로세스
    
    주요 업데이트 사항:
    - Evaluator에서 산출된 '엔티티 기반 Precision/Recall/F1' 기록
    - 의미 단위 라벨(Pure Label) 기준의 Confusion Matrix 추이 수집
    - 엔티티 정확도 가중치 점수($0.5 + 0.5 \times Ratio$) 분포 수집
    """
    
    # ==============================================================================
    # [Step 1] 객체 및 설정 준비
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
    model_type = train_conf.get('model_type', 'ner')

    logger = logging.getLogger(experiment_code)
    logger.info(f"🚀 [Process 1] Start Training Loop for {experiment_code}")

    # ==============================================================================
    # [Step 2] 실행 모듈 초기화
    # ==============================================================================
    if model_type == 'ner':
        trainer = Trainer(model, optimizer, scheduler, device)
        evaluator = Evaluator(
            model, 
            device, 
            preprocessor.tokenizer, 
            preprocessor.ner_id2label
        )
        logger.info("✅ NER Trainer & Evaluator initialized.")

    elif model_type == 'ner_gat':
        trainer = NerGatTrainer(model, optimizer, scheduler, device)
        evaluator = NerGatEvaluator(
            model, 
            device, 
            preprocessor.tokenizer, 
            preprocessor.ner_id2label
        )
        logger.info("✅ GAT-specific Trainer & Evaluator initialized.")

    # ==============================================================================
    # [Step 3] 에포크 반복 (Training & Evaluation Loop)
    # ==============================================================================
    best_f1 = 0.0
    min_valid_loss = float('inf')
    best_f1_epoch = -1
    min_loss_epoch = -1


    train_losses, valid_losses = [], []
    
    # 시각화 히스토리 (Trend 분석용)
    cm_history = []         # 에포크별 엔티티 기반 Confusion Matrix 데이터
    accuracy_history = []   # 에포크별 엔티티 정확도 점수($0.0 \sim 1.0$) 분포

    # 저장 경로 생성
    ckpt_save_dir = os.path.join(path_conf['checkpoint_dir'], experiment_code)
    log_save_dir = os.path.join(path_conf['log_dir'], experiment_code) 
    ensure_dir(ckpt_save_dir)
    ensure_dir(log_save_dir)

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
            # v_metrics['f1']은 이제 토큰 단위가 아닌 '엔티티 단위'의 F1임
            valid_result = evaluator.evaluate(valid_loader, mode="valid")
            v_metrics = valid_result['metrics']
            valid_losses.append(v_metrics['loss'])

            # 시각화 히스토리 누적
            if 'confusion_matrix' in v_metrics:
                # v_metrics['confusion_matrix'] 구조: {"labels": [...], "values": [[...]]}
                cm_history.append(v_metrics['confusion_matrix'])
            
            if 'label_accuracy_distribution' in v_metrics:
                accuracy_history.append(v_metrics['label_accuracy_distribution'])
            
            # -----------------------------------------------------------
            # 3-3. 결과 통합 및 DB 저장 (Epoch 단위 요약)
            # -----------------------------------------------------------
            epoch_summary = {
                "train_loss": train_result['loss'],
                "train_time": train_result['duration'],
                "valid_loss": v_metrics['loss'],
                "valid_f1": v_metrics['f1'],           # Entity-level F1
                "valid_precision": v_metrics['precision'],
                "valid_recall": v_metrics['recall'],
                "entity_accuracy_avg": {
                    label: (sum(scores)/len(scores) if scores else 0) 
                    for label, scores in v_metrics.get('label_accuracy_distribution', {}).items()
                },
                "confusion_matrix": v_metrics.get('confusion_matrix') # 의미 단위 CM
            }
            
            crud.create_process_result(session, {
                "experiment_code": experiment_code,
                "process_code": "process_1",
                "process_epoch": epoch,
                "process_start_time": train_result['start_time'], 
                "process_end_time": valid_result['end_time'], 
                "process_duration": train_result['duration'] + valid_result['duration'],
                "process_results": epoch_summary 
            })
            
            logger.info(f"Epoch {epoch} Result Saved. (Train Loss: {train_result['loss']:.4f} | Valid Loss: {v_metrics['loss']:.4f} | Valid F1: {v_metrics['f1']:.4f})")

            # -----------------------------------------------------------
            # 3-3-2. 문장 단위 추론 결과 저장 (DB + CSV) [UPDATED]
            # -----------------------------------------------------------
            # FK 정보 주입
            valid_logs = valid_result['logs']
            for log in valid_logs:
                log.update({"experiment_code": experiment_code, "process_code": "process_1", "process_epoch": epoch})
            
            # (1) DB Bulk Insert
            crud.bulk_insert_inference_sentences(session, valid_logs)
            logger.info(f"Saved {len(valid_logs)} inference logs to DB.")

            # (2) [NEW] CSV 파일 추출 및 저장( 유틸리티 함수 호출 (JSON 필드는 문자열로 변환되어 저장됨) )
            save_logs_to_csv(valid_logs, os.path.join(log_save_dir, f"{experiment_code}_process_1_{epoch}_inference_sentences.csv"))

            # -----------------------------------------------------------
            # 3-4. 체크포인트 저장 (Model Checkpoint)
            # -----------------------------------------------------------
            save_path = os.path.join(ckpt_save_dir, f"{experiment_code}_epoch_{epoch}.pt")
            torch.save(model.state_dict(), save_path)
            
            if v_metrics['f1'] > best_f1:
                best_f1 = v_metrics['f1']
                best_f1_epoch = epoch
                logger.info(f"✨ Current Best F1: {best_f1:.4f} (Epoch {epoch})")

            if v_metrics['loss'] < min_valid_loss:
                min_valid_loss = v_metrics['loss']
                min_loss_epoch = epoch
                logger.info(f"📉 Current Min Loss: {min_valid_loss:.4f} (Epoch {epoch})")

        # ==============================================================================
        # [Step 4] 실험 마스터 정보 갱신 (최종 성능 기록)
        # ==============================================================================
        crud.update_experiment(session, experiment_code, {
            "experiment_config": {
                "best_f1": best_f1,
                "best_epoch": min_loss_epoch,
                "best_model_path": os.path.join(ckpt_save_dir, f"{experiment_code}_epoch_{min_loss_epoch}.pt")
            }
        })

        context['best_epoch'] = min_loss_epoch
        
    # ==============================================================================
    # [Step 5] 시각화 리포트 생성 (학습 종료 후 일괄 생성)
    # ==============================================================================
    logger.info("📊 시각화 리포트 생성 시작 (Best Epoch 기준 상세 분석 및 전체 Trend)")

    # 1. Train/Valid Loss 곡선 (전체 Epoch)
    plot_loss_graph(train_losses, valid_losses, log_save_dir, experiment_code)

    # 2. 에포크별 Confusion Matrix Trend 분석
    # 각 GT 라벨별로 에포크가 진행됨에 따라 어떤 Pred 라벨들로 분류되었는지 추이를 그림
    # (예: GT 'PER'가 Epoch 1에는 'O'로 많이 가다가, 점차 'PER'로 수렴하는 과정 시각화)
    if cm_history:
        plot_confusion_matrix_trends(
            cm_history, 
            log_save_dir, 
            experiment_code
        )

    # 3. 최적 Epoch (Min Valid Loss) 기준 상세 시각화
    best_idx = min_loss_epoch-1

    if 0 <= best_idx < len(cm_history):
        logger.info(f"✨ 최적 Epoch({min_loss_epoch})의 상세 분석 그래프를 생성합니다.")
        
        # (A) 의미 단위 라벨 관계 히트맵 (Confusion Matrix)
        # 최적 시점에 어떤 라벨끼리 가장 많이 혼동되는지 확인
        plot_label_relation_matrix(
            cm_history[best_idx], 
            log_save_dir, 
            f"{experiment_code}_best_epoch_{min_loss_epoch}"
        )

        # (B) 라벨별 엔티티 정확도 분포 히스토그램
        # 최적 시점에 모델이 개체명의 시작(B)과 범위(I)를 얼마나 정확히 짚었는지 분포 확인
        plot_label_accuracy_histograms(
            accuracy_history[best_idx], 
            log_save_dir, 
            f"{experiment_code}_best_epoch_{min_loss_epoch}"
        )
    
    logger.info(f"✅ 시각화 리포트 저장 완료: {log_save_dir}")
    
    return context