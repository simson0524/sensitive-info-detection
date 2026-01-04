# scripts/run_update_z_score.py

import sys
import os
import argparse
import time
import pandas as pd
from sqlalchemy.orm import Session

# 프로젝트 루트 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.dirname(current_dir)              
sys.path.append(project_root)

# DB 및 모듈 임포트
from src.database.connection import SessionLocal
from src.modules.dtm_initializer import DTMInitializer
from src.modules.tf_idf_updater import TFIDFUpdater
from src.modules.z_score_updater import ZScoreUpdater

from src.utils.common import load_yaml
from src.utils.logger import setup_experiment_logger 
from src.utils.visualizer import plot_z_score_distribution
from src.database.crud import get_all_dtm_for_viz


# 전역 로거 설정
logger = setup_experiment_logger("DB_STAT_PIPELINE")

def main():
    """
    [Main Execution Pipeline]
    1. 환경 설정 로드 (YAML/Argparse)
    2. DB 초기화 및 도메인 스캔 (Phase 1)
    3. TF-IDF 계산 및 업데이트 (Phase 2)
    4. Z-Score 산출 및 정규화 (Phase 3)
    """
    
    # 1. Argument Parsing
    parser = argparse.ArgumentParser(description="DB-based TF-IDF & Z-Score Update Pipeline")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml", help="Path to config file")
    args = parser.parse_args()

    # 2. Config & Path 로드 (기존 스타일 유지)
    model_name = "klue/roberta-base"
    train_data_root = "data/train_data"

    if os.path.exists(args.config):
        try:
            config = load_yaml(args.config)
            if 'path' in config:
                train_data_root = config['path'].get('train_data_root', train_data_root)
            if 'train' in config:
                model_name = config['train'].get('model_name', model_name)
            logger.info(f"[Config] Loaded - Model: {model_name}, Data Root: {train_data_root}")
        except Exception as e:
            logger.warning(f"[Config] Failed to load config, using defaults: {e}")

    # 상대 경로를 절대 경로로 변환
    if not os.path.isabs(train_data_root):
        train_data_root = os.path.join(project_root, train_data_root)

    # 3. 파이프라인 실행
    logger.info("=" * 60)
    logger.info("🚀 Starting DB-based Statistical Analysis Pipeline")
    logger.info("=" * 60)
    
    start_all = time.time()
    session: Session = SessionLocal()
    
    try:
        # --- [Phase 1] DTM Initialization & Data Scanning ---
        logger.info("[Phase 1] Initializing Tables & Scanning train_data...")
        p1_start = time.time()
        initializer = DTMInitializer(session, model_name=model_name)
        initializer.initialize_and_scan(train_data_root)
        session.commit() # 트랜잭션 확정
        logger.info(f"✅ Phase 1 Completed ({time.time() - p1_start:.2f}s)")

        # --- [Phase 2] Global TF-IDF Calculation ---
        logger.info("[Phase 2] Computing Global TF-IDF Scores...")
        p2_start = time.time()
        tfidf_up = TFIDFUpdater(session)
        tfidf_up.update_tfidf_scores()
        session.commit() # 트랜잭션 확정
        logger.info(f"✅ Phase 2 Completed ({time.time() - p2_start:.2f}s)")

        # --- [Phase 3] Local Z-Score Normalization ---
        logger.info("[Phase 3] Normalizing Z-Scores per Domain...")
        p3_start = time.time()
        z_up = ZScoreUpdater(session)
        z_up.update_z_scores()
        session.commit() # 트랜잭션 확정
        logger.info(f"✅ Phase 3 Completed ({time.time() - p3_start:.2f}s)")

        total_elapsed = time.time() - start_all
        logger.info("=" * 60)
        logger.info(f"✨ Pipeline Finished Successfully! (Total: {total_elapsed:.2f}s)")
        logger.info("=" * 60)

    except Exception as e:
        session.rollback() # 오류 발생 시 모든 변경사항 되돌림
        logger.critical(f"❌ Pipeline Failed due to Error: {e}", exc_info=True)
        sys.exit(1)
        
    finally:
        session.close() # DB 세션 반납

    # 4. Visualization
    logger.info("[Phase 4] Generating Statistical Distributions...")
    try:        
        # 1. DB에서 데이터 로드 (yield_per를 사용하여 안전하게 리스트화)
        # 시각화에 필요한 2개 컬럼(z_score, is_sensitive_label)만 가져옵니다.
        session_viz = SessionLocal() # 새 세션을 열거나 기존 로직 유지
        viz_data_gen = get_all_dtm_for_viz(session_viz, batch_size=10000)
        
        # 2. DataFrame 변환 (모든 샘플을 메모리에 적재)
        viz_df = pd.DataFrame(viz_data_gen)
        
        if not viz_df.empty:
            # 3. 시각화 실행 (그림 그리기 전용 함수 호출)
            report_dir = os.path.join(train_data_root, "reports")
            plot_z_score_distribution(viz_df, report_dir)
            logger.info(f"✅ Phase 4 Completed. Check {report_dir} for results.")
        else:
            logger.warning("⚠️ No data found in DomainTermMatrix to plot.")
            
    except Exception as e:
        logger.error(f"❌ Visualization failed: {e}")
    finally:
        session_viz.close()

if __name__ == "__main__":
    main()