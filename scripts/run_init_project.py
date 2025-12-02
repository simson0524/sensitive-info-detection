# scripts/run_init_project.py

import sys
import os
import pandas as pd
import traceback
from datetime import datetime

# 프로젝트 루트 경로 추가 (src 모듈 인식을 위해)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Modules
from src.database.connection import db_manager
from src.database import crud
from src.database.init_db import init_database # 기존 DB 초기화 로직 재사용

# Utils
from src.utils.common import load_yaml
from src.utils.logger import setup_experiment_logger

def main():
    """
    [Init Project] 프로젝트 초기화 스크립트
    1. 데이터베이스 테이블 생성 (init_db)
    2. 정답지(CSV)를 로드하여 초기 사전(Dictionary) 구축
    """
    
    # 1. 설정 로드
    config_path = os.path.join(project_root, "configs", "init_project_config.yaml")
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return

    config = load_yaml(config_path)
    
    init_code = config['project']['init_code']
    dict_conf = config['dictionary_init']
    
    # 로거 설정
    logger = setup_experiment_logger(init_code, config['project']['log_dir'])
    logger.info("="*60)
    logger.info(f"🚀 Starting Project Initialization")
    logger.info("="*60)

    try:
        # ----------------------------------------------------------------------
        # [Step 1] 데이터베이스 및 테이블 생성
        # ----------------------------------------------------------------------
        logger.info("Step 1: Initializing Database Tables...")
        
        # 기존 init_db.py의 함수를 호출하여 테이블 생성 (Idempotent하므로 안전)
        try:
            init_database() 
            logger.info("✅ Database tables verified/created.")
        except Exception as e:
            logger.error(f"❌ DB Init Failed: {e}")
            raise e

        # ----------------------------------------------------------------------
        # [Step 2] 정답지 CSV 로드 및 전처리
        # ----------------------------------------------------------------------
        csv_path = os.path.join(project_root, dict_conf['source_csv_path'])
        target_domain_id = str(dict_conf['target_domain_id'])
        
        logger.info(f"Step 2: Loading Dictionary Data from {csv_path}...")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Answer sheet not found: {csv_path}")
        
        # CSV 읽기
        df = pd.read_csv(csv_path)
        
        # 필수 컬럼 확인
        required_cols = ['word', 'label']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")
        
        # 중복 제거 (동일한 단어+라벨이 여러 번 나올 수 있으므로)
        initial_count = len(df)
        df = df.drop_duplicates(subset=['word', 'label'])
        logger.info(f"Loaded {initial_count} rows -> {len(df)} unique words (Deduplicated).")

        # ----------------------------------------------------------------------
        # [Step 3] 사전 데이터 구축 (Bulk Insert 준비)
        # ----------------------------------------------------------------------
        logger.info("Step 3: Preparing Data for DB Insert...")
        
        dict_items = []
        
        for _, row in df.iterrows():
            word = str(row['word']).strip()
            label = str(row['label']).strip() # 이것이 data_category가 됨
            
            if not word or not label:
                continue

            # DB 스키마에 맞춘 딕셔너리 생성
            item = {
                "annotated_word": word,
                "data_category": label, # CSV의 label 컬럼 사용
                "domain_id": target_domain_id, # Config 값 사용
                
                "first_inserted_experiment_code": "init",
                "insertion_counts": 1,
                "deletion_counts": 0,
                "z_score_of_the_word": {} # 빈 JSON
            }
            dict_items.append(item)

        # ----------------------------------------------------------------------
        # [Step 4] DB 저장 (Bulk Insert)
        # ----------------------------------------------------------------------
        if dict_items:
            with db_manager.get_db() as session:
                # 기존 crud에 만든 bulk insert 함수 활용
                # (주의: crud.py에 bulk_insert_dictionary_items 함수가 있어야 함. 
                #  아까 process_0 만들 때 추가해드렸습니다.)
                crud.bulk_insert_dictionary_items(session, dict_items)
                
            logger.info(f"✅ Successfully inserted {len(dict_items)} items into InfoDictionary.")
        else:
            logger.warning("⚠️ No valid items to insert.")

        logger.info("="*60)
        logger.info("🎉 Project Initialization Completed Successfully.")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"❌ Initialization Failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()