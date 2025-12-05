# src/processes/process_3.py

import logging
import os
from datetime import datetime
from tqdm import tqdm

# Modules
from src.modules.regex_matcher import RegexMatcher
from src.modules.result_aggregator import ResultAggregator

# Database
from src.database.connection import db_manager
from src.database import crud

# Utils
from src.utils.common import ensure_dir, save_logs_to_csv

def run_process_3(config: dict, context: dict):
    """
    [Process 3] 정규표현식(Regex) 매칭 검증 프로세스
    
    - 공통: RegexMatcher를 사용하여 문장 전체에서 PII 탐지 수행
    - Train 모드: BIO 태그를 파싱한 '정답 단어'와 정규식 탐지 결과를 1:1 비교 (정탐/오탐/미탐)
                  (라벨 Normalization 적용: 개인정보_1 -> 개인정보)
    - Test 모드: 정답 없이 정규식으로 탐지된 결과를 모두 저장
    - 결과: DB 저장 및 CSV 추출
    """
    
    # ==============================================================================
    # [Step 1] 설정 및 로거 초기화
    # ==============================================================================
    exp_conf = config['experiment']
    train_conf = config['train']
    path_conf = config['path']
    
    experiment_code = exp_conf['experiment_code']
    data_category = exp_conf.get('data_category', 'personal_data')
    run_mode = exp_conf.get('run_mode', 'train')
    
    logger = logging.getLogger(experiment_code)
    logger.info(f"🚀 [Process 3] Start Regex Matching Verification (Mode: {run_mode})")

    # ==============================================================================
    # [Step 2] 데이터 및 도구 준비
    # ==============================================================================
    valid_loader = context['valid_loader']
    preprocessor = context['preprocessor']
    tokenizer = preprocessor.tokenizer
    
    # BIO 라벨 맵 (ID <-> Name) {0: "O", 1: "B-개인정보_1", ...}
    ner_id2label = preprocessor.ner_id2label 

    # RegexMatcher 초기화 (내부적으로 Detectors 로드)
    matcher = RegexMatcher()

    # ==============================================================================
    # [Step 3] 매칭 및 검증 루프
    # ==============================================================================
    aggregator = ResultAggregator()
    start_time = datetime.now()
    process_epoch = 1

    logger.info("Starting regex detection loop...")
    
    log_save_dir = os.path.join(path_conf['log_dir'], experiment_code)
    ensure_dir(log_save_dir)
    
    for batch in tqdm(valid_loader, desc="Regex Matching"):
        batch_size = len(batch['sentence'])
        
        input_ids_batch = batch['input_ids'].cpu().tolist()
        labels_batch = batch['labels'].cpu().tolist()

        for i in range(batch_size):
            # 3-1. 메타 데이터 추출
            sentence_id = batch['sentence_id'][i]
            original_sentence = batch['sentence'][i]
            file_name = batch['file_name'][i]
            domain_id = batch['domain_id'][i]
            sentence_seq = batch['sentence_seq'][i]
            
            seq_val = sentence_seq.item() if hasattr(sentence_seq, 'item') else sentence_seq

            # 3-2. Regex 탐지 수행 (문장 전체 스캔)
            regex_results = matcher.detect(original_sentence)
            
            # 3-3. Regex 결과를 프로젝트 라벨로 매핑 및 필터링
            pred_spans = {}
            for res in regex_results:
                raw_label = res['label'] # "전화번호"
                type_info = matcher.DETECTOR_TYPE_MAP.get(raw_label, {})
                
                target_label = None
                if data_category == "personal_data" and type_info.get("category") == "개인":
                    target_label = "개인정보" 
                elif data_category == "confidential_data" and type_info.get("category") == "기밀":
                    target_label = "기밀정보"
                
                if target_label:
                    pred_spans[res['match']] = target_label

            # -----------------------------------------------------------
            # [Case A] Train Mode (GT와 비교하여 정밀 검증)
            # -----------------------------------------------------------
            if run_mode == 'train':
                current_input_ids = input_ids_batch[i]
                current_tags = labels_batch[i]
                tokens = tokenizer.convert_ids_to_tokens(current_input_ids)
                
                # GT 파싱 (e.g., {'홍길동': '개인정보_1'})
                gt_entities = _extract_entities_from_bio(tokens, current_tags, ner_id2label, tokenizer)
                
                # [수정] 라벨 Normalization (개인정보_1 -> 개인정보)
                normalized_gt_entities = {
                    word: _normalize_label(label) 
                    for word, label in gt_entities.items()
                }
                
                expected_label_name = "개인정보" if data_category == "personal_data" else "기밀정보"
                
                # 타겟 카테고리만 필터링
                target_gt_spans = {
                    word: label for word, label in normalized_gt_entities.items()
                    if label == expected_label_name
                }

                # 정탐/오탐/미탐 분류
                pred_words = set(pred_spans.keys())
                gt_words = set(target_gt_spans.keys())

                hits = pred_words & gt_words
                wrongs = pred_words - gt_words
                mismatches = gt_words - pred_words

                # 로그 기록
                for word in hits:
                    _add_log(aggregator, "hit", sentence_id, file_name, seq_val, 
                             original_sentence, word, domain_id, 
                             expected_label_name, expected_label_name, experiment_code, process_epoch)
                
                for word in wrongs:
                    _add_log(aggregator, "wrong", sentence_id, file_name, seq_val, 
                             original_sentence, word, domain_id, 
                             "O", expected_label_name, experiment_code, process_epoch)

                for word in mismatches:
                    _add_log(aggregator, "mismatch", sentence_id, file_name, seq_val,
                             original_sentence, word, domain_id,
                             expected_label_name, "O", experiment_code, process_epoch)

            # -----------------------------------------------------------
            # [Case B] Test Mode (단순 탐지 결과 저장)
            # -----------------------------------------------------------
            elif run_mode == 'test':
                for word, label in pred_spans.items():
                    _add_log(aggregator, "hit", sentence_id, file_name, seq_val,
                             original_sentence, word, domain_id,
                             "Unknown", label, experiment_code, process_epoch)

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # ==============================================================================
    # [Step 4] 결과 저장 (DB & CSV)
    # ==============================================================================
    with db_manager.get_db() as session:
        total_logs = 0
        all_logs_for_csv = []

        # 4-1. 프로세스 결과 요약 저장
        process_results = {
            "metrics": aggregator.get_metrics(),
            "detected_count": total_logs,
            "run_mode": run_mode
        }
        
        crud.create_process_result(session, {
            "experiment_code": experiment_code,
            "process_code": "process_3",
            "process_epoch": process_epoch,
            "process_start_time": start_time,
            "process_end_time": end_time,
            "process_duration": duration,
            "process_results": process_results
        })

        # 4-2. 문장 로그 저장
        for r_type in ["hit", "wrong", "mismatch"]:
            logs = aggregator.get_logs(r_type)
            if logs:
                crud.bulk_insert_inference_sentences(session, logs)
                all_logs_for_csv.extend(logs)
                total_logs += len(logs)
        
        logger.info(f"Saved {total_logs} inference logs to DB.")

        # [NEW] CSV 파일 추출
        if all_logs_for_csv:
            csv_file_name = f"{experiment_code}_process_3_{process_epoch}_inference_sentences.csv"
            csv_file_path = os.path.join(log_save_dir, csv_file_name)
            save_logs_to_csv(all_logs_for_csv, csv_file_path)
            logger.info(f"Saved CSV log to {csv_file_path}")
        
    logger.info("[Process 3] Completed Successfully.")
    return context


# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

def _normalize_label(label: str) -> str:
    """
    라벨 정규화: '개인정보_1' -> '개인정보'
    """
    if "_" in label and label.split("_")[-1].isdigit():
        return label.rsplit("_", 1)[0]
    return label

def _extract_entities_from_bio(tokens, tags, id2label, tokenizer):
    """
    BIO 태그 리스트를 파싱하여 {단어: 라벨} 딕셔너리로 반환
    """
    entities = {}
    current_tokens = []
    current_label = None
    
    for token, tag_id in zip(tokens, tags):
        if tag_id == -100: continue
        label_name = id2label.get(tag_id, "O")
        
        if label_name.startswith("B-"):
            if current_tokens:
                word = tokenizer.convert_tokens_to_string(current_tokens)
                entities[word] = current_label
            current_tokens = [token]
            current_label = label_name[2:]
            
        elif label_name.startswith("I-") and current_label == label_name[2:]:
            current_tokens.append(token)
            
        else:
            if current_tokens:
                word = tokenizer.convert_tokens_to_string(current_tokens)
                entities[word] = current_label
            current_tokens = []
            current_label = None
            
    if current_tokens:
        word = tokenizer.convert_tokens_to_string(current_tokens)
        entities[word] = current_label
        
    return entities

def _add_log(aggregator, match_type, sent_id, fname, seq, origin_sent, word, domain, gt, pred, exp_code, epoch=1):
    """로그 데이터 생성 및 집계기에 추가"""
    sentence_inference_result = {
        "sentence_id": sent_id,
        "source_file_name": fname,
        "sequence_in_file": seq,
        "origin_sentence": origin_sent,
        "domain_id": domain,
        "inference_results": [{
            "word": word,
            "label": pred,
            "match_result": match_type,
            "ground_truth": gt
        }],
        "entity_count": 1
    }
    
    log_entry = {
        "experiment_code": exp_code,
        "process_code": "process_3",
        "process_epoch": epoch,
        "sentence_id": sent_id,
        "sentence_inference_result": sentence_inference_result,
        "confidence_score": 1.0
    }
    
    aggregator.add_result(match_type, log_entry, 0)