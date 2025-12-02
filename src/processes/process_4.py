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

def run_process_4(config: dict, context: dict):
    """
    [Process 4] 모델 보완 추론 및 Hybrid 검증 프로세스
    
    1. 규칙(사전/Regex)이 찾은 결과를 DB에서 로드하여 메모리에 매핑.
    2. 학습된 Best Model로 전체 검증 데이터에 대해 추론 수행.
    3. 모델 결과와 규칙 결과를 비교하여 유형 분류 및 통계 산출:
       - Double Check: 규칙도 찾고 모델도 찾음 (신뢰도 높음)
       - Model Complement: 규칙은 못 찾았는데 모델이 찾음 (모델의 기여도)
       - Rule Only: 규칙은 찾았는데 모델은 못 찾음 (모델의 한계)
    4. 분석 결과 및 로그 DB 저장.
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
    
    # 구조: rule_hits[sentence_id] = { "단어": "라벨", ... }
    rule_hits = {}
    
    with db_manager.get_db() as session:
        # Process 2 (Dictionary) & Process 3 (Regex) 결과 모두 조회
        for proc_code in ["process_2", "process_3"]:
            # generator를 통해 대용량 로그 순회 (메모리 절약)
            logs = crud.get_inference_sentences(session, experiment_code, proc_code, 1)
            
            for log in logs:
                sid = log['sentence_id']
                if sid not in rule_hits:
                    rule_hits[sid] = {}
                
                # JSON 파싱
                res = log.get('sentence_inference_result', {})
                results_list = res.get('inference_results', [])
                
                for r in results_list:
                    # 'hit' (정탐) 또는 'prediction' (Test모드 탐지) 인 것만 수집
                    # 오탐(wrong)이나 미탐(mismatch)은 제외
                    if r.get('match_result') in ['hit', 'prediction']:
                        word = r['word']
                        label = r['label']
                        rule_hits[sid][word] = label

    logger.info(f"Loaded rule hits for {len(rule_hits)} sentences.")

    # ==============================================================================
    # [Step 3] Best Model 로드
    # ==============================================================================
    logger.info("Loading Best Model from Checkpoint...")
    
    # 모델 껍데기 생성 (기존 인코더 재사용하여 메모리 절약)
    encoder = context['model'].encoder 
    num_labels = len(preprocessor.ner_label2id)
    
    best_model = RobertaNerModel(
        encoder=encoder,
        num_classes=num_labels,
        use_focal=False # 추론엔 focal loss 불필요
    ).to(device)
    
    # 가중치 로드
    ckpt_path = os.path.join(
        path_conf['checkpoint_dir'], experiment_code, f"{experiment_code}_best.pt"
    )
    
    if os.path.exists(ckpt_path):
        best_model.load_state_dict(torch.load(ckpt_path, map_location=device))
        logger.info(f"✅ Loaded weights from {ckpt_path}")
    else:
        logger.warning(f"⚠️ Checkpoint not found at {ckpt_path}. Using current model state.")
        best_model = context['model'] # Fallback

    # ==============================================================================
    # [Step 4] 추론 및 비교 분석 (Hybrid Logic)
    # ==============================================================================
    evaluator = Evaluator(
        best_model, 
        device, 
        preprocessor.tokenizer, 
        preprocessor.ner_id2label 
    )

    # 전체 데이터셋에 대해 추론 (mode='test'로 하여 순수 예측값만 받음)
    # GT 비교는 여기서 별도로 수행하지 않고, Rule과의 비교에 집중합니다.
    result = evaluator.evaluate(context['valid_loader'], mode="test")
    raw_logs = result['logs'] # Evaluator가 만든 기본 로그 (List[Dict])

    # 통계 집계 변수 초기화
    stats = {
        "double_check": 0,      # 규칙 O, 모델 O
        "model_complement": 0,  # 규칙 X, 모델 O
        "rule_only": 0,         # 규칙 O, 모델 X
        "total_model_detected": 0
    }

    processed_logs = []

    for log in raw_logs:
        sid = log['sentence_id']
        
        # 모델이 찾은 결과 리스트 (Evaluator가 만든 구조)
        # inference_results: [{'word': '홍길동', 'label': '인물', ...}]
        model_results = log['sentence_inference_result']['inference_results']
        
        # 해당 문장의 규칙 탐지 결과 (Dict: {word: label})
        rule_findings = rule_hits.get(sid, {}).copy() # pop을 위해 복사본 사용
        
        # 1. 모델 탐지 결과 순회 (Double Check vs Complement 확인)
        for entity in model_results:
            word = entity['word']
            
            if word in rule_findings:
                # 규칙도 찾고 모델도 찾음 -> Double Check
                entity['hybrid_status'] = "Double Check"
                stats['double_check'] += 1
                # 확인된 규칙 결과는 제거 (나중에 Rule Only 계산 위함)
                rule_findings.pop(word, None)
            else:
                # 규칙은 못 찾았는데 모델이 찾음 -> Model Complement
                entity['hybrid_status'] = "Model Complement"
                stats['model_complement'] += 1
            
            stats['total_model_detected'] += 1
            
        # 2. Rule Only 계산 (모델 결과에는 없지만 규칙에는 남아있는 것)
        for r_word, r_label in rule_findings.items():
            stats['rule_only'] += 1
            
            # 로그에 추가 (선택 사항: 모델이 놓친 것도 기록하여 완벽한 하이브리드 결과 생성)
            model_results.append({
                "word": r_word,
                "label": r_label,
                "start": -1, # 위치 정보는 역추적 어려우므로 -1 또는 생략
                "end": -1,
                "hybrid_status": "Rule Only (Model Missed)"
            })
        
        # 업데이트된 로그(hybrid_status 포함) 저장
        log['sentence_inference_result']['inference_results'] = model_results
        log['sentence_inference_result']['entity_count'] = len(model_results)
        processed_logs.append(log)

    # -------------------------------------------------------------
    # 비율(Ratio) 계산
    # -------------------------------------------------------------
    total_detections = stats['double_check'] + stats['model_complement'] + stats['rule_only']
    if total_detections > 0:
        stats['ratio_double_check'] = round(stats['double_check'] / total_detections, 4)
        stats['ratio_complement'] = round(stats['model_complement'] / total_detections, 4)
        stats['ratio_rule_only'] = round(stats['rule_only'] / total_detections, 4)
    else:
        stats.update({'ratio_double_check': 0, 'ratio_complement': 0, 'ratio_rule_only': 0})

    logger.info(f"📊 Hybrid Analysis Result: {stats}")

    # ==============================================================================
    # [Step 5] DB 저장
    # ==============================================================================
    with db_manager.get_db() as session:
        # 5-1. 결과 요약 저장
        crud.create_process_result(session, {
            "experiment_code": experiment_code,
            "process_code": "process_4", 
            "process_epoch": 1,
            "process_start_time": datetime.now(), # 근사치
            "process_end_time": result.get('end_time', datetime.now()),
            "process_duration": result['metrics'].get('duration', 0.0),
            
            # 분석 통계 및 모델 기본 성능 지표 함께 저장
            "process_results": {
                "hybrid_stats": stats,
                "base_metrics": result['metrics'] # 모델 자체 성능 지표 (Loss 등)
            }
        })

        # 5-2. 문장 로그 저장 (Hybrid Status 포함)
        for log in processed_logs:
            log['experiment_code'] = experiment_code
            log['process_code'] = "process_4"
            log['process_epoch'] = 1
        
        crud.bulk_insert_inference_sentences(session, processed_logs)
        logger.info(f"Saved {len(processed_logs)} hybrid inference logs to DB.")

    logger.info("[Process 4] Completed.")
    return context