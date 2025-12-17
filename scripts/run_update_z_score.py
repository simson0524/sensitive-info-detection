# scripts/run_update_z_score.py

import sys
import os
import argparse

# -----------------------------------------------------------------------------
# [Path Setup]
# 이 스크립트는 scripts/ 폴더 내에 위치하지만, 프로젝트 루트(src/)의 모듈을 참조해야 합니다.
# 따라서 현재 파일(__file__) 기준으로 상위 상위 디렉토리를 찾아 sys.path에 추가합니다.
# -----------------------------------------------------------------------------
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../scripts
project_root = os.path.dirname(current_dir)              # .../sensitive-info-detector
sys.path.append(project_root)

# Path 설정 후 모듈 임포트
from src.modules.z_score_calculator import ZScoreCalculator
from src.utils.common import load_yaml  # Config 로더 (프로젝트 내 유틸리티 사용 가정)
from src.utils.logger import logger

def main():
    """
    [Main Execution Function]
    1. 커맨드 라인 인자 파싱 (Config 파일 경로 설정)
    2. Config 파일 로드 (train_data_root 경로 확인)
    3. ZScoreCalculator 인스턴스 생성 및 실행
    """
    
    # 1. 인자 파서 설정
    parser = argparse.ArgumentParser(description="Update Global/Local Z-scores for all domains.")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/base_config.yaml", 
        help="Path to the base configuration yaml file."
    )
    args = parser.parse_args()

    # 2. Config 로드 및 데이터 경로 결정
    # 기본값 설정: Config 파일에 값이 없거나 파일 로드 실패 시 사용
    train_data_root = 'data/train_data' 
    
    if os.path.exists(args.config):
        try:
            config = load_yaml(args.config)
            
            # base_config.yaml 내부에 'train_data_root' 키가 있는지 확인
            if config and 'train_data_root' in config:
                train_data_root = config['train_data_root']
                logger.info(f"Loaded train_data_root from config: {train_data_root}")
            else:
                logger.info(f"Key 'train_data_root' not found in config. Using default: {train_data_root}")
                
        except Exception as e:
            logger.warning(f"Failed to load config file ({args.config}): {e}. Using default path.")
    else:
        logger.info(f"Config file not found at {args.config}. Using default path: {train_data_root}")

    # [Path Safety] 상대 경로일 경우 프로젝트 루트 기준으로 보정 (선택 사항)
    if not os.path.isabs(train_data_root):
        train_data_root = os.path.join(project_root, train_data_root)

    # 3. Z-Score Calculator 실행
    logger.info("=== Starting Z-Score Update Process ===")
    logger.info(f"Target Data Root: {train_data_root}")
    
    try:
        # 모듈 인스턴스 생성 및 전체 파이프라인(run) 실행
        calculator = ZScoreCalculator(data_root_dir=train_data_root)
        calculator.run()
        logger.info("=== Z-Score Update Completed Successfully ===")
        
    except Exception as e:
        logger.critical(f"An error occurred during Z-Score update: {e}", exc_info=True)

if __name__ == "__main__":
    main()