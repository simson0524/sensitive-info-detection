# scripts/run_update_z_score.py

import sys
import os
import argparse
import json
import pandas as pd
from transformers import AutoTokenizer
from konlpy.tag import Okt # [필수] Okt 추가

current_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.dirname(current_dir)              
sys.path.append(project_root)

from src.modules.z_score_calculator import ZScoreCalculator
from src.utils.common import load_yaml
from src.utils.logger import setup_experiment_logger 
from src.utils.visualizer import plot_z_score_distribution

logger = setup_experiment_logger("Z_SCORE_UPDATE")

def normalize_label(raw_label: str) -> str:
    raw_label = str(raw_label).strip()
    if "개인정보" in raw_label: return "개인정보"
    elif "준식별자" in raw_label: return "준식별자"
    elif "기밀정보" in raw_label: return "기밀정보"
    else: return "Non-labeled"

def process_tokens_with_okt(tokenizer, okt, text: str) -> list:
    """
    [Helper] Calculator의 _smart_tokenizer와 동일한 로직
    Tokenize -> Merge -> Okt POS Check -> Filtering
    """
    # 1. RoBERTa Tokenize
    raw_tokens = tokenizer.tokenize(str(text))
    special_tokens = set(tokenizer.all_special_tokens)
    merged_chunks = []
    
    # 2. Merge ##
    for t in raw_tokens:
        if t in special_tokens: continue
        if t.startswith("##"):
            if merged_chunks: merged_chunks[-1] += t[2:]
            else: merged_chunks.append(t[2:])
        else:
            merged_chunks.append(t)
            
    # 3. Okt POS & Filter
    final_tokens = []
    TARGET_TAGS = {'Noun', 'Number', 'Alpha', 'Foreign'}
    
    for chunk in merged_chunks:
        try:
            # Calculator와 동일 옵션 (stem=True, norm=True)
            pos_results = okt.pos(chunk, stem=True, norm=True)
            for word, tag in pos_results:
                if tag in TARGET_TAGS:
                    final_tokens.append(word)
        except:
            pass
            
    return final_tokens

def create_z_score_dataframe(data_root: str, model_name: str) -> pd.DataFrame:
    logger.info("[Visualization] Preparing data for plotting...")
    all_records = []
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        okt = Okt() # 매핑을 위한 Okt 인스턴스
    except Exception as e:
        logger.warning(f"Could not load resources: {e}")
        return pd.DataFrame()

    TARGET_CATEGORIES = {'개인정보', '준식별자', '기밀정보'}

    if not os.path.exists(data_root): return pd.DataFrame()

    for domain_dir in os.listdir(data_root):
        domain_path = os.path.join(data_root, domain_dir)
        if not os.path.isdir(domain_path) or domain_dir.startswith('.'): continue

        token_label_map = {}
        csv_path = os.path.join(domain_path, 'answer_sheet.csv')
        
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

                    # [핵심] CSV 단어도 (Merge -> Okt) 과정을 거쳐서 알맹이만 추출
                    refined_tokens = process_tokens_with_okt(tokenizer, okt, raw_word)
                    
                    for t in refined_tokens:
                        token_label_map[t] = final_label
                            
            except Exception as e:
                logger.warning(f"Failed to process CSV in {domain_dir}: {e}")

        json_path = os.path.join(domain_path, 'z_score.json')
        if not os.path.exists(json_path): continue

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                z_data = json.load(f)

            for doc_id, scores in z_data.items():
                for word, score in scores.get('global', {}).items():
                    all_records.append({
                        'Domain': domain_dir, 'Type': 'Global Z-Score',
                        'Word': word, 'Score': score,
                        'Label': token_label_map.get(word, 'Non-labeled')
                    })
                for word, score in scores.get('local', {}).items():
                    all_records.append({
                        'Domain': domain_dir, 'Type': 'Local Z-Score',
                        'Word': word, 'Score': score,
                        'Label': token_label_map.get(word, 'Non-labeled')
                    })
        except Exception: pass

    return pd.DataFrame(all_records)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base_config.yaml")
    args = parser.parse_args()

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

    logger.info("=== Starting Z-Score Update (Hybrid Tokenization) ===")
    try:
        calculator = ZScoreCalculator(data_root_dir=train_data_root, model_name=model_name)
        calculator.run()
        logger.info("✅ Z-Score Calculation Completed.")
    except Exception as e:
        logger.critical(f"Error: {e}", exc_info=True)
        return

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