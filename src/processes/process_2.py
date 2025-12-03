# src/processes/process_2.py

import logging
import os
from datetime import datetime
from tqdm import tqdm

# Modules: 사전 매칭 및 결과 집계를 위한 핵심 모듈
from src.modules.dictionary_matcher import DictionaryMatcher
from src.modules.result_aggregator import ResultAggregator

# Database: DB 연결 및 CRUD 유틸리티
from src.database.connection import db_manager
from src.database import crud

# Utils: 파일 저장 관련 유틸리티
from src.utils.common import ensure_dir, save_logs_to_csv

def run_process_2(config: dict, context: dict):
    """
    [Process 2] 사전(Dictionary) 매칭 검증 프로세스
    
    기 구축된 사전 데이터를 기반으로 PII를 탐지하고 성능을 검증합니다.
    
    주요 기능:
    1. DB에서 해당 도메인의 사전을 메모리에 로드 (Dictionary Matcher)
    2. 검증 데이터셋을 순회하며 사전 탐색 수행 (문장 전체 검색)
    3. [Train Mode] 정답(GT)과 비교하여 정탐/오탐/미탐 검증
       -> 오탐(Wrong) 발생 시, 해당 단어를 사전에서 즉시 무효화(Self-Cleaning)
    4. [Test Mode] 문장 내 포함된 사전 단어 단순 탐지 및 저장
    5. 결과 로그 DB 저장 및 CSV 파일 추출

    Args:
        config (dict): 설정 정보
        context (dict): 공유 객체 (DataLoader, Preprocessor 등)
    """
    
    # ==============================================================================
    # [Step 1] 설정 로드 및 로거 초기화
    # ==============================================================================
    exp_conf = config['experiment']
    dict_conf = config['dictionary_init']
    train_conf = config['train']
    path_conf = config['path'] # [NEW] CSV 저장용 경로
    
    experiment_code = exp_conf['experiment_code']
    data_category = exp_conf.get('data_category', 'personal_data') # 'personal_data' or 'confidential_data'
    run_mode = exp_conf.get('run_mode', 'train')
    
    # 로거 가져오기
    logger = logging.getLogger(experiment_code)
    logger.info(f"🚀 [Process 2] Start Dictionary Matching Verification (Mode: {run_mode})")

    # ==============================================================================
    # [Step 2] 데이터 및 매핑 정보 준비
    # ==============================================================================
    valid_loader = context['valid_loader']
    
    # GT 파싱을 위한 도구들 (Train 모드에서 필수)
    preprocessor = context['preprocessor']
    tokenizer = preprocessor.tokenizer
    ner_id2label = preprocessor.ner_id2label

    # 검증 대상 라벨 설정 (예: '개인정보')
    target_label_name = "개인정보" if data_category == "personal_data" else "기밀정보"
    
    # 통계용 라벨 ID (없으면 건너뜀)
    pred_label_id = train_conf['label_map'].get(target_label_name)
    if pred_label_id is None:
        logger.warning(f"⚠️ Target label '{target_label_name}' not found. Skipping Process 2.")
        return context

    # ==============================================================================
    # [Step 3] 사전 매처(Matcher) 초기화 및 데이터 로드
    # ==============================================================================
    with db_manager.get_db() as session:
        matcher = DictionaryMatcher(session)
        
        # 설정된 도메인 ID들에 해당하는 사전을 DB에서 로드하여 메모리에 캐싱
        # (Insertion > Deletion 인 유효 단어만 로드됨)
        matcher.load_dictionaries(dict_conf['domain_ids'], data_category)
        
        # 로드된 사전의 크기 등 통계 정보 가져오기
        dict_stats = matcher.get_stats()
        logger.info(f"📚 Dictionary Stats: {dict_stats}")

    # ==============================================================================
    # [Step 4] 매칭 및 검증 루프 (Validation Loop)
    # ==============================================================================
    aggregator = ResultAggregator() # 결과(정/오/미탐)를 수집하는 객체
    start_time = datetime.now()
    process_epoch = 1 # Rule-base 검증은 1회성 프로세스이므로 Epoch 1로 고정

    logger.info("Starting matching loop...")
    
    # 로그 저장 경로 생성 (CSV 저장용)
    log_save_dir = os.path.join(path_conf['log_dir'], experiment_code)
    ensure_dir(log_save_dir)
    
    # [중요] 오탐 시 사전 업데이트(Deletion Count 증가)를 위해 세션을 루프 밖에서 엽니다.
    with db_manager.get_db() as session:
        
        for batch in tqdm(valid_loader, desc="Dictionary Matching"):
            batch_size = len(batch['sentence'])
            
            # Tensor -> List 변환 (CPU로 이동)
            input_ids_batch = batch['input_ids'].cpu().tolist()
            labels_batch = batch['labels'].cpu().tolist()

            for i in range(batch_size):
                # -----------------------------------------------------------
                # 4-1. 메타 데이터 추출
                # -----------------------------------------------------------
                sentence_id = batch['sentence_id'][i]
                original_sentence = batch['sentence'][i]
                domain_id = batch['domain_id'][i]
                file_name = batch['file_name'][i]
                
                # Tensor -> Item 안전 변환
                seq_val = batch['sentence_seq'][i]
                sentence_seq = seq_val.item() if hasattr(seq_val, 'item') else seq_val

                # -----------------------------------------------------------
                # 4-2. 사전 탐색 (공통 로직)
                # -----------------------------------------------------------
                # 문장 전체를 스캔하여 사전에 있는 단어들을 모두 찾습니다.
                # (match_sentence는 List를 반환하므로 Set으로 변환하여 중복 제거)
                dict_matches = set(matcher.match_sentence(original_sentence, domain_id))

                # -----------------------------------------------------------
                # [Case A] Train Mode (BIO 태그 파싱 후 비교 + 오탐 제거)
                # -----------------------------------------------------------
                if run_mode == 'train':
                    # (1) 정답(GT) 추출: BIO 태그 -> 단어 리스트 변환
                    current_input_ids = input_ids_batch[i]
                    current_tags = labels_batch[i]
                    tokens = tokenizer.convert_ids_to_tokens(current_input_ids)
                    
                    gt_entities = _extract_entities_from_bio(tokens, current_tags, ner_id2label, tokenizer)
                    
                    # 현재 타겟 카테고리(예: 개인정보)에 해당하는 GT만 필터링
                    target_gt_words = {
                        word for word, label in gt_entities.items() 
                        if label == target_label_name
                    }

                    # (2) 정탐/오탐/미탐 분류 (집합 연산)
                    hits = target_gt_words & dict_matches       # 교집합 (둘 다 있음)
                    mismatches = target_gt_words - dict_matches # 차집합 (GT엔 있는데 사전엔 없음)
                    wrongs = dict_matches - target_gt_words     # 차집합 (사전엔 있는데 GT엔 없음)

                    # (3) 로그 기록 및 사전 업데이트
                    
                    # Hit (정탐)
                    for word in hits:
                        _add_log(aggregator, "hit", sentence_id, file_name, sentence_seq, original_sentence, 
                                 word, domain_id, target_label_name, target_label_name, experiment_code)
                    
                    # Mismatch (미탐)
                    for word in mismatches:
                        _add_log(aggregator, "mismatch", sentence_id, file_name, sentence_seq, original_sentence, 
                                 word, domain_id, target_label_name, "O", experiment_code)
                                 
                    # Wrong (오탐) -> 사전에서 무효화
                    for word in wrongs:
                        _add_log(aggregator, "wrong", sentence_id, file_name, sentence_seq, original_sentence, 
                                 word, domain_id, "O", target_label_name, experiment_code)
                        
                        # [핵심] 오탐 단어는 즉시 무효화 (Deletion Count = Insertion Count)
                        # 다음번 로드부터는 (Insertion > Deletion) 조건을 만족하지 않아 제외됨
                        crud.invalidate_dictionary_item(
                            session, word, data_category, domain_id
                        )

                # -----------------------------------------------------------
                # [Case B] Test Mode (단순 탐지 결과 저장)
                # -----------------------------------------------------------
                elif run_mode == 'test':
                    if dict_matches:
                        inference_results = []
                        for word in dict_matches:
                            inference_results.append({
                                "word": word,
                                "label": target_label_name,
                                "match_result": "prediction" # 정답을 모르므로 prediction으로 표기
                            })
                        
                        # JSON 구조 생성
                        sentence_inference_result = {
                            "sentence_id": sentence_id,
                            "source_file_name": file_name,
                            "sequence_in_file": sentence_seq,
                            "origin_sentence": original_sentence,
                            "domain_id": domain_id,
                            "inference_results": inference_results,
                            "entity_count": len(inference_results)
                        }
                        
                        log_entry = {
                            "experiment_code": experiment_code,
                            "process_code": "process_2",
                            "process_epoch": process_epoch,
                            "sentence_id": sentence_id,
                            "sentence_inference_result": sentence_inference_result,
                            "confidence_score": 1.0
                        }
                        # Test 모드에서는 모두 Hit으로 간주하거나 별도 처리
                        aggregator.add_result("hit", log_entry, pred_label_id)

        # (루프 종료 후 시간 기록)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # ==============================================================================
        # [Step 5] 최종 DB 저장 및 CSV 추출
        # ==============================================================================
        
        # 5-1. 문장 단위 상세 로그 저장 (Bulk Insert)
        total_logs = 0
        all_logs_for_csv = [] # CSV 저장을 위한 통합 리스트

        for r_type in ["hit", "wrong", "mismatch"]:
            logs = aggregator.get_logs(r_type)
            if logs:
                # DB에 대량 삽입
                crud.bulk_insert_inference_sentences(session, logs)
                # CSV용 리스트에 추가
                all_logs_for_csv.extend(logs) 
                total_logs += len(logs)
        
        logger.info(f"Saved {total_logs} inference logs to DB.")

        # 5-2. [NEW] CSV 파일 추출 및 저장
        if all_logs_for_csv:
            csv_file_name = f"{experiment_code}_process_2_{process_epoch}_inference_sentences.csv"
            csv_file_path = os.path.join(log_save_dir, csv_file_name)
            
            save_logs_to_csv(all_logs_for_csv, csv_file_path)
            logger.info(f"Saved CSV log to {csv_file_path}")

        # 5-3. 프로세스 요약 저장
        process_results = {
            "dictionary_stats": dict_stats,
            "metrics": aggregator.get_metrics(),
            "run_mode": run_mode
        }
        
        crud.create_process_result(session, {
            "experiment_code": experiment_code,
            "process_code": "process_2",
            "process_epoch": process_epoch,
            "process_start_time": start_time,
            "process_end_time": end_time,
            "process_duration": duration,
            "process_results": process_results
        })
        
    logger.info("[Process 2] Completed Successfully.")
    return context


# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

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
            
        else: # "O"
            if current_tokens:
                word = tokenizer.convert_tokens_to_string(current_tokens)
                entities[word] = current_label
            current_tokens = []
            current_label = None
            
    if current_tokens:
        word = tokenizer.convert_tokens_to_string(current_tokens)
        entities[word] = current_label
        
    return entities

def _add_log(aggregator, match_type, sent_id, fname, seq, origin_sent, word, domain, gt, pred, exp_code, epoch=1, is_hit=None):
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
        "process_code": "process_2",
        "process_epoch": epoch,
        "sentence_id": sent_id,
        "sentence_inference_result": sentence_inference_result,
        "confidence_score": 1.0
    }
    
    # 통계용 ID는 임의값(0) 사용
    aggregator.add_result(match_type, log_entry, 0)