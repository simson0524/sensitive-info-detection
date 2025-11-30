# src/utils/logger.py

import logging
import os
import sys

def setup_experiment_logger(experiment_code: str, log_dir: str = "outputs/logs"):
    """
    실험별로 별도의 폴더를 만들고 로그 파일을 생성하는 로거를 설정합니다.
    
    최종 경로: {log_dir}/{experiment_code}/{experiment_code}_experiment_log.txt
    예시: outputs/logs/EXP_001/EXP_001_experiment_log.txt
    
    Args:
        experiment_code (str): 실험 식별 코드 (예: EXP_001)
        log_dir (str): 로그 기본 디렉토리 경로 (기본값: outputs/logs)
    
    Returns:
        logging.Logger: 설정된 로거 객체
    """
    # 1. 실험 코드별 전용 디렉토리 경로 생성 (ex: outputs/logs/EXP_001)
    experiment_save_dir = os.path.join(log_dir, experiment_code)
    
    # 2. 디렉토리가 없으면 생성 (재귀적으로 생성됨)
    if not os.path.exists(experiment_save_dir):
        os.makedirs(experiment_save_dir, exist_ok=True)
        # print(f"📂 [Logger] Created log directory: {experiment_save_dir}")

    # 3. 로그 파일 전체 경로 설정
    log_file_name = f"{experiment_code}_experiment_log.txt"
    log_file_path = os.path.join(experiment_save_dir, log_file_name)

    # 4. 로거 생성 (이름을 실험 코드로 설정하여 구분)
    logger = logging.getLogger(experiment_code)
    logger.setLevel(logging.INFO)
    
    # 5. 중복 핸들러 방지 (이미 핸들러가 있으면 추가하지 않고 반환)
    if logger.hasHandlers():
        return logger

    # 6. 포맷 설정 (시간 - 레벨 - 메시지)
    formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 7. 파일 핸들러 (txt 파일에 저장)
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 8. 콘솔 핸들러 (터미널에 출력)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger