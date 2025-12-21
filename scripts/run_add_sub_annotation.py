# scripts/run_add_sub_annotation.py

import sys
import os
import argparse
import json
import re
import numpy as np
from typing import List, Dict, Any
from transformers import AutoTokenizer

# [라이브러리 설명]
# Sudo 권한이 없는 환경에서도 Mecab을 사용할 수 있게 해주는 라이브러리입니다.
# 설치: pip install python-mecab-ko
from mecab import MeCab 

# -----------------------------------------------------------------------------
# [Path Setup]
# 이 스크립트는 'scripts/' 폴더에 위치하지만, 프로젝트 루트('src/')의 모듈을 불러와야 합니다.
# 따라서 현재 파일의 상위 상위 경로(프로젝트 루트)를 시스템 경로에 추가합니다.
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.dirname(current_dir)              
sys.path.append(project_root)

from src.utils.common import load_yaml
from src.utils.logger import setup_experiment_logger 

# 이 스크립트 전용 로거 설정 (로그 파일: outputs/logs/SUB_ANNOTATION/...)
logger = setup_experiment_logger("SUB_ANNOTATION")

def strip_existing_suffix(label: str) -> str:
    """
    [Safe Reset: 재실행 안전장치]
    라벨 뒤에 이미 붙어있는 접미사(_1, _2 등)를 제거합니다.
    
    Why?
    - 스크립트를 여러 번 실행하더라도 '개인정보_1_1' 처럼 중복되어 붙지 않게 하기 위함입니다.
    - 항상 원본 라벨('개인정보') 상태로 되돌린 후 다시 계산합니다.
    
    Args:
        label (str): 접미사가 있을 수 있는 라벨 (예: "개인정보_2")
    Returns:
        str: 접미사가 제거된 라벨 (예: "개인정보")
    """
    label = str(label).strip()
    # 정규식: 문자열 끝($)에 있는 '_숫자' 패턴을 찾아 빈 문자열로 치환
    return re.sub(r'_\d+$', '', label)

def get_tokens_from_word(word: str, tokenizer, mecab, cache: dict) -> List[str]:
    """
    [Key Matcher: 전처리 파이프라인]
    Annotation의 단어(예: '홍길동은')를 z_score.json의 Key(예: '홍길동')와 매칭하기 위해
    Z-Score 계산 때와 **100% 동일한 전처리 과정**을 수행합니다.
    
    Process:
    1. RoBERTa Tokenize: 문맥을 고려하지 않고 단순 서브워드 분리
    2. Merge '##': 서브워드를 결합하여 온전한 단어 복원
    3. MeCab Filter: 조사, 어미 등을 제거하고 핵심 품사(명사, 숫자 등)만 추출
    """
    # 1. 캐시 확인 (속도 최적화: 이미 분석한 단어는 바로 반환)
    if word in cache:
        return cache[word]

    # 2. RoBERTa Tokenize & Merge
    # 예: "홍길동은" -> ['홍', '##길', '##동', '##은'] -> ['홍길동은']
    raw_tokens = tokenizer.tokenize(str(word))
    special_tokens = set(tokenizer.all_special_tokens)
    merged_chunks = []
    
    for t in raw_tokens:
        if t in special_tokens: continue # [CLS], [SEP] 등 제외
        
        if t.startswith("##"):
            if merged_chunks: merged_chunks[-1] += t[2:] # 앞 단어에 붙임
            else: merged_chunks.append(t[2:])
        else:
            merged_chunks.append(t)
            
    # 3. MeCab Filter (핵심 단어만 남김)
    # 예: "홍길동은" -> ('홍길동', NNP), ('은', JX) -> "홍길동"만 추출
    target_tags = {'NNG', 'NNP', 'NNB', 'NR', 'SL', 'SN'}
    valid_tokens = []
    
    for chunk in merged_chunks:
        try:
            pos_results = mecab.pos(chunk)
            for w, tag in pos_results:
                if tag in target_tags:
                    valid_tokens.append(w)
        except:
            pass # 분석 실패 시 무시
            
    # 4. 결과 캐싱 및 반환
    cache[word] = valid_tokens
    return valid_tokens

def get_score_direct_lookup(word: str, tokenizer, mecab, score_data: Dict, scope: str, cache: dict) -> float:
    """
    [Score Lookup Strategy]
    단어를 전처리하여 Key를 얻고, z_score.json 데이터에서 점수를 직접 조회합니다.
    
    Why Direct Lookup?
    - Z-Score 계산 시 정답지에 있는 단어는 0점이라도 반드시 키(Key)로 등록해두었습니다.
    - 따라서 문서를 순회하며 검색할 필요 없이, Dictionary Key 조회(O(1))로 즉시 점수를 찾을 수 있습니다.
    """
    # 전처리된 토큰 획득 (예: "홍길동은" -> ["홍길동"])
    target_keys = get_tokens_from_word(word, tokenizer, mecab, cache)
    
    # 전처리 결과 유효한 토큰이 없으면(조사만 있거나 특수문자 등) 0점 처리
    if not target_keys:
        return 0.0

    found_scores = []
    
    # JSON 구조: { "doc_id": { "global": { "토큰명": 점수 }, ... } }
    # Global Z-Score는 문서 간 편차가 거의 없지만, 데이터 정합성을 위해 
    # 해당 토큰이 등장한 모든 점수의 평균을 사용합니다.
    
    for key in target_keys:
        key_scores = []
        # 전체 문서를 순회하며 해당 Key가 있는지 확인
        # (Tip: 실제로는 Global Score의 경우 아무 문서나 하나 잡아도 되지만, 안전하게 평균을 냄)
        for doc_content in score_data.values():
            scores_map = doc_content.get(scope, {})
            
            if key in scores_map:
                key_scores.append(scores_map[key])
        
        if key_scores:
            found_scores.append(np.mean(key_scores))
        else:
            # 키가 없다는 건, Z-Score 계산 당시 문서에도 없었고 정답지에도 없었다는 뜻 (이상 케이스)
            found_scores.append(0.0)
            
    if not found_scores:
        return 0.0
        
    return float(np.mean(found_scores))

def determine_suffix(score: float, low: float, high: float) -> str:
    """
    [Threshold Logic] 
    사용자가 설정한 기준 점수(low, high)에 따라 난이도/희귀도 등급을 매깁니다.
    """
    if score < low:
        return "_1" # 점수가 낮음 (흔한 단어 -> 쉬움)
    elif score <= high:
        return "_2" # 점수가 보통
    else:
        return "_3" # 점수가 높음 (희귀한 단어 -> 중요/어려움)

def process_json_file(file_path: str, z_score_data: Dict, tokenizer, mecab, settings: dict, token_cache: dict):
    """
    [File Processor]
    개별 JSON 파일 하나를 열어서, Annotations 내부의 라벨을 수정하고 저장합니다.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return

    # 설정값 추출
    scope = settings['scope']
    low_thr = settings['threshold_low']
    high_thr = settings['threshold_high']
    
    updated_count = 0
    
    # JSON 구조 순회: List[Dict] 형태
    for sentence_obj in data:
        if 'annotations' not in sentence_obj: continue
            
        for anno in sentence_obj['annotations']:
            label = anno.get('label', '')
            word = anno.get('word', '')
            
            # 무효 라벨(Non-labeled 등)은 건너뜀
            if not label or label.lower() in ['non-labeled', 'o', 'nan']:
                continue
            
            # 1. 초기화: 기존에 붙은 _1, _2 제거 (개인정보_1 -> 개인정보)
            base_label = strip_existing_suffix(label)
            
            # 2. 점수 조회: 단어를 전처리해서 Z-Score 찾아오기
            score = get_score_direct_lookup(word, tokenizer, mecab, z_score_data, scope, token_cache)
            
            # 3. 등급 결정: 점수에 따라 _1, _2, _3 접미사 선택
            suffix = determine_suffix(score, low_thr, high_thr)
            
            # 4. 라벨 업데이트
            new_label = f"{base_label}{suffix}"
            
            # 변경사항이 있을 때만 업데이트 카운트 증가
            if label != new_label:
                anno['label'] = new_label
                updated_count += 1

    # 변경된 JSON 저장
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            # indent=4로 저장하여 가독성 유지
            json.dump(data, f, ensure_ascii=False, indent=4)
    except Exception as e:
        logger.error(f"Failed to save {file_path}: {e}")

def process_domain(domain_path: str, settings: dict, tokenizer, mecab):
    """
    [Domain Processor]
    특정 도메인 폴더(예: 001_finance) 내의 데이터를 처리합니다.
    단, answer_sheet.csv(정답지)가 존재하는 도메인만 처리 대상으로 삼습니다.
    """
    domain_name = os.path.basename(domain_path)
    
    # [조건] answer_sheet.csv가 없으면 스킵 (Z-Score 계산 대상이 아니었을 확률 높음)
    csv_path = os.path.join(domain_path, 'answer_sheet.csv')
    if not os.path.exists(csv_path):
        return

    # 점수 파일 로드 (설정에 따라 z_score.json 또는 confidence_score.json)
    score_type = settings['score_type']
    score_file_path = os.path.join(domain_path, f"{score_type}.json")
    
    if not os.path.exists(score_file_path):
        logger.warning(f"[{domain_name}] Score file '{score_type}.json' missing. Skipping.")
        return

    try:
        with open(score_file_path, 'r', encoding='utf-8') as f:
            z_score_data = json.load(f)
    except Exception as e:
        logger.error(f"[{domain_name}] Failed to load score file: {e}")
        return

    logger.info(f"Processing Domain: {domain_name} (Thr: {settings['threshold_low']} ~ {settings['threshold_high']})")
    
    # 전처리 속도 향상을 위한 로컬 캐시 (도메인 단위로 리셋)
    token_cache = {} 

    # 해당 도메인 폴더 내의 학습 데이터(.json) 찾기 (메타 파일 제외)
    json_files = [
        f for f in os.listdir(domain_path) 
        if f.endswith('.json') and f not in ['z_score.json', 'confidence_score.json']
    ]
    
    for json_file in json_files:
        full_path = os.path.join(domain_path, json_file)
        process_json_file(full_path, z_score_data, tokenizer, mecab, settings, token_cache)

def main():
    # -------------------------------------------------------------------------
    # [USER CONFIGURATION] 사용자가 직접 수정하는 설정 구간
    # -------------------------------------------------------------------------
    SETTINGS = {
        "score_type": "z_score",   # 참조할 점수 파일 ('z_score' 등)
        "scope": "global",         # 점수 범위 ('global' 추천)
        
        # [점수 기준 (Thresholds)]
        # score < -0.2  --> _1 (Common/Easy)
        # -0.2 <= score <= 0.2 --> _2 (Normal)
        # 0.2 < score   --> _3 (Rare/Hard)
        "threshold_low": -0.2,
        "threshold_high": 0.2
    }
    # -------------------------------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    args = parser.parse_args()

    # 기본값 설정
    train_data_root = "data/train_data"
    model_name = "klue/roberta-base"

    # Config 파일 로드 (경로 및 모델명 확인용)
    if os.path.exists(args.config):
        try:
            config = load_yaml(args.config)
            if 'path' in config: train_data_root = config['path'].get('train_data_root', train_data_root)
            if 'train' in config: model_name = config['train'].get('model_name', model_name)
        except: pass

    # 절대 경로 변환
    if not os.path.isabs(train_data_root):
        train_data_root = os.path.join(project_root, train_data_root)

    # 리소스 초기화
    logger.info(f"Initializing Resources (Model: {model_name})...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        mecab = MeCab() # Sudo 없이 실행
    except Exception as e:
        logger.critical(f"Initialization failed: {e}")
        return

    # 프로세스 시작
    logger.info("=== Starting Sub-Annotation Update ===")
    
    if not os.path.exists(train_data_root):
        logger.error(f"Data root not found: {train_data_root}")
        return

    # 모든 도메인 폴더 순회
    for domain_dir in os.listdir(train_data_root):
        domain_path = os.path.join(train_data_root, domain_dir)
        # 디렉토리이고 숨김 파일이 아닌 경우 처리
        if os.path.isdir(domain_path) and not domain_dir.startswith('.'):
            process_domain(domain_path, SETTINGS, tokenizer, mecab)

    logger.info("=== All Done. Labels updated in JSON files. ===")

if __name__ == "__main__":
    main()