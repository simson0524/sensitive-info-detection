# scripts/run_update_z_score.py

import sys
import os
import argparse
import json
import pandas as pd
from transformers import AutoTokenizer

# [변경] Sudo 없이 사용 가능한 Mecab
from mecab import MeCab 

# Path Setup
current_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.dirname(current_dir)              
sys.path.append(project_root)

from src.modules.z_score_calculator import ZScoreCalculator
from src.utils.common import load_yaml
from src.utils.logger import setup_experiment_logger 
from src.utils.visualizer import plot_z_score_distribution

# 전역 로거 설정
logger = setup_experiment_logger("Z_SCORE_UPDATE")

def normalize_label(raw_label: str) -> str:
    """
    세부 라벨(예: 개인정보_1)을 대분류(개인정보)로 통합
    """
    raw_label = str(raw_label).strip()
    if "개인정보" in raw_label: return "개인정보"
    elif "준식별자" in raw_label: return "준식별자"
    elif "기밀정보" in raw_label: return "기밀정보"
    else: return "Non-labeled"

def process_tokens_with_mecab(tokenizer, mecab, text: str, cache: dict) -> list:
    """
    [Data Matching Helper]
    ZScoreCalculator의 _smart_tokenizer와 동일한 로직을 수행합니다.
    CSV의 단어를 이 함수로 처리해야 JSON의 Key와 정확히 매칭됩니다.
    
    1. RoBERTa Tokenize -> 2. Merge '##' -> 3. MeCab POS Filter
    """
    # 1. Tokenize
    raw_tokens = tokenizer.tokenize(str(text))
    special_tokens = set(tokenizer.all_special_tokens)
    merged_chunks = []
    
    # 2. Merge
    for t in raw_tokens:
        if t in special_tokens: continue
        if t.startswith("##"):
            if merged_chunks: merged_chunks[-1] += t[2:]
            else: merged_chunks.append(t[2:])
        else:
            merged_chunks.append(t)
            
    final_tokens = []
    # 3. MeCab POS Filter
    TARGET_TAGS = {'NNG', 'NNP', 'NNB', 'NR', 'SL', 'SN'}
    
    for chunk in merged_chunks:
        # 캐시 확인 (속도 최적화)
        if chunk in cache:
            final_tokens.extend(cache[chunk])
            continue
            
        try:
            valid_words = []
            pos_results = mecab.pos(chunk)
            
            for word, tag in pos_results:
                if tag in TARGET_TAGS:
                    valid_words.append(word)
            
            # 캐시 저장
            cache[chunk] = valid_words
            final_tokens.extend(valid_words)
        except:
            pass
            
    return final_tokens

def create_z_score_dataframe(data_root: str, model_name: str) -> pd.DataFrame:
    logger.info("[Visualization] Preparing data for plotting...")
    all_records = []
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        mecab = MeCab() # Mecab 인스턴스 생성
    except Exception as e:
        logger.warning(f"Could not load resources: {e}")
        return pd.DataFrame()

    # 시각화용 로컬 캐시 (함수 실행 동안만 유지)
    viz_pos_cache = {} 
    
    TARGET_CATEGORIES = {'개인정보', '준식별자', '기밀정보'}

    if not os.path.exists(data_root): return pd.DataFrame()

    for domain_dir in os.listdir(data_root):
        domain_path = os.path.join(data_root, domain_dir)
        if not os.path.isdir(domain_path) or domain_dir.startswith('.'): continue

        token_label_map = {}
        csv_path = os.path.join(domain_path, 'answer_sheet.csv')
        
        # --- [Step A] Label Mapping ---
        if os.path.exists(csv_path):
            try:
                df_csv = pd.read_csv(csv_path)
                word_col = 'word' if 'word' in df_csv.columns else df_csv.columns[0]
                label_col = 'label' if 'label' in df_csv.columns else df_csv.columns[1]

                for _, row in df_csv.iterrows():
                    raw_word = str(row[word_col])
                    raw_label = str(row[label_col])
                    
                    cat_label = normalize_label(raw_label)
                    final_label = cat_label if cat_label in TARGET_CATEGORIES else 'Non-labeled'

                    # [핵심] CSV 단어 -> 정제된 토큰으로 변환 (캐시 사용)
                    refined_tokens = process_tokens_with_mecab(tokenizer, mecab, raw_word, viz_pos_cache)
                    
                    # 정제된 토큰들에 라벨 부여
                    for t in refined_tokens:
                        token_label_map[t] = final_label
                            
            except Exception as e:
                logger.warning(f"Failed to process CSV in {domain_dir}: {e}")

        # --- [Step B] JSON Merge ---
        json_path = os.path.join(domain_path, 'z_score.json')
        if not os.path.exists(json_path): continue

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                z_data = json.load(f)

            for doc_id, scores in z_data.items():
                # Global Z-Score
                for word, score in scores.get('global', {}).items():
                    all_records.append({
                        'Domain': domain_dir, 'Type': 'Global Z-Score',
                        'Word': word, 'Score': score,
                        'Label': token_label_map.get(word, 'Non-labeled')
                    })
                # Local Z-Score
                for word, score in scores.get('local', {}).items():
                    all_records.append({
                        'Domain': domain_dir, 'Type': 'Local Z-Score',
                        'Word': word, 'Score': score,
                        'Label': token_label_map.get(word, 'Non-labeled')
                    })
        except Exception: pass

    return pd.DataFrame(all_records)

def main():
    # 1. Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    args = parser.parse_args()

    # 2. Config Loading
    model_name = "klue/roberta-base"
    train_data_root = "data/train_data"

    if os.path.exists(args.config):
        try:
            config = load_yaml(args.config)
            if 'path' in config: train_data_root = config['path'].get('train_data_root', train_data_root)
            if 'train' in config: model_name = config['train'].get('model_name', model_name)
            logger.info(f"Loaded Config - Model: {model_name}, Data Root: {train_data_root}")
        except: pass

    if not os.path.isabs(train_data_root):
        train_data_root = os.path.join(project_root, train_data_root)

    # 3. Run Calculation
    logger.info("=== Starting Z-Score Update (Mecab + Cache) ===")
    try:
        calculator = ZScoreCalculator(data_root_dir=train_data_root, model_name=model_name)
        calculator.run()
        logger.info("✅ Z-Score Calculation Completed.")
    except Exception as e:
        logger.critical(f"Error: {e}", exc_info=True)
        return

    # 4. Run Visualization
    logger.info("=== Starting Visualization ===")
    try:
        df_result = create_z_score_dataframe(data_root=train_data_root, model_name=model_name)
        if not df_result.empty:
            plot_z_score_distribution(df=df_result, save_dir=train_data_root)
            logger.info("✅ Visualization Completed.")
        else:
            logger.warning("No data available to plot.")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)

if __name__ == "__main__":
    main()