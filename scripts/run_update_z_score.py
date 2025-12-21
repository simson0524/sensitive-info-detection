# scripts/run_update_z_score.py

import sys
import os
import argparse
import json
import pandas as pd
from transformers import AutoTokenizer

# -----------------------------------------------------------------------------
# [Path Setup]
# 이 스크립트는 scripts/ 폴더 내에 위치하지만, 프로젝트 루트(src/)의 모듈을 참조해야 합니다.
# 따라서 현재 파일(__file__) 기준으로 상위 상위 디렉토리를 찾아 sys.path에 추가합니다.
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../sensitive-info-detector/scripts
project_root = os.path.dirname(current_dir)              # .../sensitive-info-detector
sys.path.append(project_root)

# [Module Import]
from src.modules.z_score_calculator import ZScoreCalculator
from src.utils.common import load_yaml
from src.utils.logger import setup_experiment_logger 
from src.utils.visualizer import plot_z_score_distribution

# [Global Logger Setup]
# 스크립트 실행 로그는 'Z_SCORE_UPDATE' 라는 실험 코드로 저장됩니다.
# 저장 위치: outputs/logs/Z_SCORE_UPDATE/Z_SCORE_UPDATE_experiment_log.txt
logger = setup_experiment_logger("Z_SCORE_UPDATE")

def normalize_label(raw_label: str) -> str:
    """
    [Helper Function: 라벨 정규화]
    Config나 데이터셋에 정의된 세부 라벨(예: 개인정보_1, 준식별자_2)을 
    시각화의 편의성을 위해 3대 대분류(개인정보, 준식별자, 기밀정보)로 통합합니다.
    
    Args:
        raw_label (str): 원본 라벨 (예: "개인정보_1")
    Returns:
        str: 통합된 라벨 (예: "개인정보") 또는 "Non-labeled"
    """
    raw_label = str(raw_label).strip()
    
    # 문자열 포함 여부로 카테고리화
    if "개인정보" in raw_label:
        return "개인정보"
    elif "준식별자" in raw_label:
        return "준식별자"
    elif "기밀정보" in raw_label:
        return "기밀정보"
    else:
        # 시각화 대상이 아닌 라벨(일반 정보 등)
        return "Non-labeled"

def create_z_score_dataframe(data_root: str, model_name: str) -> pd.DataFrame:
    """
    [Data Preparation: 시각화 데이터 생성]
    계산된 'z_score.json' (점수 데이터)과 'answer_sheet.csv' (라벨 데이터)를 병합합니다.
    
    [핵심 로직: 토크나이저 일치]
    - Z-Score는 'Subword Token' 단위로 계산되어 있습니다. (예: '##전자')
    - 하지만 CSV 정답지는 'Word' 단위로 되어 있습니다. (예: '삼성전자')
    - 따라서 CSV 단어를 '모델 토크나이저'로 분해하여 JSON의 Key와 매칭시킵니다.
    
    Args:
        data_root (str): 데이터 저장 최상위 경로 (data/train_data)
        model_name (str): HuggingFace 모델명 (예: "klue/roberta-base")
        
    Returns:
        pd.DataFrame: ['Domain', 'Type', 'Word', 'Score', 'Label'] 컬럼을 가진 DF
    """
    logger.info("[Visualization] Preparing data for plotting...")
    
    all_records = []
    
    # 1. 모델 토크나이저 로드
    # Z-Score 계산 시 사용한 모델과 반드시 동일해야 Key 매칭이 가능합니다.
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        logger.warning(f"Could not load tokenizer '{model_name}': {e}. Visualization mapping might fail.")
        return pd.DataFrame()

    # 시각화할 핵심 대분류 라벨 정의
    TARGET_CATEGORIES = {'개인정보', '준식별자', '기밀정보'}

    if not os.path.exists(data_root):
        logger.error(f"Data root not found: {data_root}")
        return pd.DataFrame()

    # 각 도메인 디렉토리 순회 (예: data/train_data/01_finance)
    for domain_dir in os.listdir(data_root):
        domain_path = os.path.join(data_root, domain_dir)
        if not os.path.isdir(domain_path) or domain_dir.startswith('.'):
            continue

        # --- [Step A] Label Mapping (CSV -> Subword Tokens) ---
        # 목표: CSV 단어를 쪼개서 { '토큰': '라벨' } 형태의 맵을 만듭니다.
        token_label_map = {}
        csv_path = os.path.join(domain_path, 'answer_sheet.csv')
        
        if os.path.exists(csv_path):
            try:
                df_csv = pd.read_csv(csv_path)
                # 컬럼명 유연성 확보
                word_col = 'word' if 'word' in df_csv.columns else df_csv.columns[0]
                label_col = 'label' if 'label' in df_csv.columns else df_csv.columns[1]

                for _, row in df_csv.iterrows():
                    raw_word = str(row[word_col])
                    raw_label = str(row[label_col])
                    
                    # 1. 라벨 정규화 (예: '개인정보_1' -> '개인정보')
                    category_label = normalize_label(raw_label)
                    final_label = category_label if category_label in TARGET_CATEGORIES else 'Non-labeled'

                    # 2. [핵심] 단어를 모델 토큰으로 분해
                    # 예: "삼성전자" -> ['삼성', '##전자']
                    tokens = tokenizer.tokenize(raw_word)
                    special_tokens = set(tokenizer.all_special_tokens)
                    
                    # 3. 분해된 토큰 각각에 라벨 부여
                    for t in tokens:
                        if t not in special_tokens:
                            # 중복 토큰 발생 시, 최신 라벨로 덮어씌웁니다.
                            token_label_map[t] = final_label
                            
            except Exception as e:
                logger.warning(f"Failed to process CSV in {domain_dir}: {e}")

        # --- [Step B] Z-Score JSON Merge ---
        # 목표: JSON 점수 데이터를 읽고, 위에서 만든 Map으로 라벨을 붙입니다.
        json_path = os.path.join(domain_path, 'z_score.json')
        if not os.path.exists(json_path):
            continue

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                z_data = json.load(f)

            for doc_id, scores in z_data.items():
                # Global Scores 데이터 추가
                for word, score in scores.get('global', {}).items():
                    all_records.append({
                        'Domain': domain_dir,
                        'Type': 'Global Z-Score',
                        'Word': word,
                        'Score': score,
                        # 토큰 맵에 없으면 'Non-labeled' (기계가 찾았지만 정답지엔 없는 토큰 등)
                        'Label': token_label_map.get(word, 'Non-labeled')
                    })
                
                # Local Scores 데이터 추가
                for word, score in scores.get('local', {}).items():
                    all_records.append({
                        'Domain': domain_dir,
                        'Type': 'Local Z-Score',
                        'Word': word,
                        'Score': score,
                        'Label': token_label_map.get(word, 'Non-labeled')
                    })
        except Exception as e:
            logger.error(f"Error reading JSON {json_path}: {e}")

    return pd.DataFrame(all_records)


def main():
    """
    [Main Execution Flow]
    1. Config Load: 모델명과 데이터 경로 확인
    2. Calculation: ZScoreCalculator 실행 (점수 갱신)
    3. Visualization: DataFrame 생성 및 그래프 저장
    """
    
    # 1. Argument Parsing
    parser = argparse.ArgumentParser(description="Update Global/Local Z-scores and Plot Distribution.")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Path to base config yaml.")
    args = parser.parse_args()

    # 2. Config Loading (기본값 설정)
    model_name = "klue/roberta-base"
    train_data_root = "data/train_data"

    if os.path.exists(args.config):
        try:
            config = load_yaml(args.config)
            
            # YAML 파일 구조에 따라 경로 파싱
            if 'path' in config and 'train_data_root' in config['path']:
                train_data_root = config['path']['train_data_root']
            
            if 'train' in config and 'model_name' in config['train']:
                model_name = config['train']['model_name']
                
            logger.info(f"Loaded Config - Model: {model_name}, Data Root: {train_data_root}")
            
        except Exception as e:
            logger.warning(f"Failed to parse config file: {e}. Using defaults.")
    else:
        logger.warning(f"Config file not found at {args.config}. Using defaults.")

    # 상대 경로를 절대 경로로 변환
    if not os.path.isabs(train_data_root):
        train_data_root = os.path.join(project_root, train_data_root)

    # 3. Run Calculator (Z-Score 계산)
    logger.info("=== Starting Z-Score Update Process ===")
    try:
        # Config에서 읽은 모델명을 전달하여 Tokenizer 일치 보장
        calculator = ZScoreCalculator(data_root_dir=train_data_root, model_name=model_name)
        calculator.run() 
        logger.info("✅ Z-Score Calculation Completed.")
    except Exception as e:
        logger.critical(f"Error during calculation: {e}", exc_info=True)
        return # 계산 실패 시 중단

    # 4. Run Visualization (시각화)
    logger.info("=== Starting Z-Score Visualization ===")
    try:
        # 시각화 데이터 준비 (모델명 전달 필수)
        df_result = create_z_score_dataframe(data_root=train_data_root, model_name=model_name)
        
        if not df_result.empty:
            # 그래프 저장
            plot_z_score_distribution(df=df_result, save_dir=train_data_root)
            logger.info("✅ Visualization Completed. Check 'z_score_distribution.png' in train_data folder.")
        else:
            logger.warning("No data available to plot (Check if JSON files are empty).")
            
    except Exception as e:
        logger.error(f"Error during visualization: {e}", exc_info=True)

if __name__ == "__main__":
    main()