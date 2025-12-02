# src/processes/process_4.py

import torch
import os
import logging
from datetime import datetime
from torch.utils.data import DataLoader

# Modules
from src.modules.ner_evaluator import Evaluator
from src.models.ner_roberta import RobertaNerModel

# Database
from src.database.connection import db_manager
from src.database import crud

# Utils
from src.utils.common import ensure_dir, save_logs_to_csv

def run_process_4(config: dict, context: dict):
    """
    [Process 4] 모델 보완 추론 및 Hybrid 검증 프로세스
    
    1. 규칙(사전/Regex)이 찾은 결과를 DB에서 로드하여 메모리에 매핑.
    2. 학습된 Best Model로 전체 검증 데이터에 대해 추론 수행.
    3. 모델 결과와 규칙 결과를 비교하여 유형 분류 및 통계 산출:
       - Double Check: 규칙도 찾고 모델도 찾음 (신뢰도 높음)
       - Model Complement: 규칙은 못 찾았는데 모델이 찾음 (모델의 기여도)
       - Rule Only: 규칙은 찾았는데 모델은 못 찾음 (모델의 한계)
    4. 분석 결과 및 로그 DB 저장 & CSV 추출.
    """
    
    # ==============================================================================
    # [Step 1] 설정 및 로거 초기화
    # ==============================================================================
    experiment_code = context['experiment_code']
    device = context['device']
    preprocessor = context['preprocessor']
    
    path_conf = config['path']
    train_conf = config['train']

    logger = logging.getLogger(experiment_code)
    logger.info(f"🚀 [Process 4] Start Hybrid Inference & Analysis")

    # ==============================================================================
    # [Step 2] 규칙 기반 탐지 결과 로드 (Process 2 & 3)
    # ==============================================================================
    logger.info("Loading Rule-based detection results from DB...")
    
    rule_hits = {}
    
    with db_manager.get_db() as session:
        for proc_code in ["process_2", "process_3"]:
            logs = crud.get_inference_sentences(session, experiment_code, proc_code, 1)
            for log in logs:
                sid = log['sentence_id']
                if sid not in rule_hits:
                    rule_hits[sid] = {}
                
                res = log.get('sentence_inference_result', {})
                results_list = res.get('inference_results', [])
                
                for r in results_list:
                    if r.get('match_result') in ['hit', 'prediction']:
                        word = r['word']
                        label = r['label']
                        rule_hits[sid][word] = label

    logger.info(f"Loaded rule hits for {len(rule_hits)} sentences.")

    # ==============================================================================
    # [Step 3] Best Model 로드
    # ==============================================================================
    logger.info("Loading Best Model from Checkpoint...")
    
    encoder = context['model'].encoder 
    num_labels = len(preprocessor.ner_label2id)
    
    best_model = RobertaNerModel(
        encoder=encoder,
        num_classes=num_labels,
        use_focal=False 
    ).to(device)
    
    ckpt_path = os.path.join(
        path_conf['checkpoint_dir'], experiment_code, f"{experiment_code}_best.pt"
    )
    
    if os.path.exists(ckpt_path):
        best_model.load_state_dict(torch.load(ckpt_path, map_location=device))
        logger.info(f"✅ Loaded weights from {ckpt_path}")
    else:
        logger.warning(f"⚠️ Checkpoint not found at {ckpt_path}. Using current model state.")
        best_model = context['model']

    # ==============================================================================
    # [Step 4] 추론 및 비교 분석 (Hybrid Logic)
    # ==============================================================================
    evaluator = Evaluator(
        best_model, 
        device, 
        preprocessor.tokenizer, 
        preprocessor.ner_id2label 
    )

    result = evaluator.evaluate(context['valid_loader'], mode="test")
    raw_logs = result['logs']

    stats = {
        "double_check": 0, "model_complement": 0, "rule_only": 0, "total_model_detected": 0
    }

    processed_logs = []

    for log in raw_logs:
        sid = log['sentence_id']
        model_results = log['sentence_inference_result']['inference_results']
        rule_findings = rule_hits.get(sid, {}).copy() 
        
        # 1. 모델 탐지 결과 순회
        for entity in model_results:
            word = entity['word']
            if word in rule_findings:
                entity['hybrid_status'] = "Double Check"
                stats['double_check'] += 1
                rule_findings.pop(word, None)
            else:
                entity['hybrid_status'] = "Model Complement"
                stats['model_complement'] += 1
            stats['total_model_detected'] += 1
            
        # 2. Rule Only 계산
        for r_word, r_label in rule_findings.items():
            stats['rule_only'] += 1
            model_results.append({
                "word": r_word,
                "label": r_label,
                "start": -1, 
                "end": -1,
                "hybrid_status": "Rule Only (Model Missed)"
            })
        
        log['sentence_inference_result']['inference_results'] = model_results
        log['sentence_inference_result']['entity_count'] = len(model_results)
        processed_logs.append(log)

    # 비율 계산
    total_detections = stats['double_check'] + stats['model_complement'] + stats['rule_only']
    if total_detections > 0:
        stats['ratio_double_check'] = round(stats['double_check'] / total_detections, 4)
        stats['ratio_complement'] = round(stats['model_complement'] / total_detections, 4)
        stats['ratio_rule_only'] = round(stats['rule_only'] / total_detections, 4)
    else:
        stats.update({'ratio_double_check': 0, 'ratio_complement': 0, 'ratio_rule_only': 0})

    logger.info(f"📊 Hybrid Analysis Result: {stats}")

    # ==============================================================================
    # [Step 5] DB 저장 및 CSV 추출
    # ==============================================================================
    
    # CSV 저장 경로 생성
    log_save_dir = os.path.join(path_conf['log_dir'], experiment_code)
    ensure_dir(log_save_dir)

    with db_manager.get_db() as session:
        # 5-1. 결과 요약 저장
        crud.create_process_result(session, {
            "experiment_code": experiment_code,
            "process_code": "process_4", 
            "process_epoch": 1,
            "process_start_time": datetime.now(), 
            "process_end_time": result.get('end_time', datetime.now()),
            "process_duration": result['metrics'].get('duration', 0.0),
            "process_results": {
                "hybrid_stats": stats,
                "base_metrics": result['metrics']
            }
        })

        # 5-2. 문장 로그 저장 (Bulk Insert)
        # FK 주입
        for log in processed_logs:
            log['experiment_code'] = experiment_code
            log['process_code'] = "process_4"
            log['process_epoch'] = 1
        
        crud.bulk_insert_inference_sentences(session, processed_logs)
        logger.info(f"Saved {len(processed_logs)} hybrid inference logs to DB.")
        
        # 5-3. [NEW] CSV 파일 추출
        csv_file_name = f"{experiment_code}_process_4_1_inference_sentences.csv"
        csv_file_path = os.path.join(log_save_dir, csv_file_name)
        
        save_logs_to_csv(processed_logs, csv_file_path)
        logger.info(f"Saved CSV log to {csv_file_path}")

    logger.info("[Process 4] Completed.")
    return context