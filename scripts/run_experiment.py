# scripts/run_experiment.py

import sys
import os
from datetime import datetime
import traceback

# 프로젝트 루트 경로를 path에 추가 (src 모듈 인식을 위해)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Utils
from src.utils.common import load_yaml
from src.utils.logger import setup_experiment_logger

# Database
from src.database.connection import db_manager
from src.database import crud

# Processes
from src.processes.process_0 import run_process_0
from src.processes.process_1 import run_process_1
from src.processes.process_2 import run_process_2
from src.processes.process_3 import run_process_3
from src.processes.process_4 import run_process_4

def main():
    """
    [Main Script] 실험 전체 파이프라인 지휘 (Orchestrator)
    
    1. 설정 로드 및 DB 실험 등록
    2. Process 0: 준비 (데이터셋, 모델, 가중치 로드)
    3. Process 1~4: 모드(Train/Test)에 따라 순차적 실행
    4. 종료 처리 및 시간 기록
    """
    
    # --------------------------------------------------------------------------
    # [Step 1] 설정 로드 및 초기화
    # --------------------------------------------------------------------------
    config_path = os.path.join(project_root, "configs", "experiment_config.yaml")
    config = load_yaml(config_path)

    exp_conf = config['experiment']
    path_conf = config['path']
    
    experiment_code = exp_conf['experiment_code']
    run_mode = exp_conf.get('run_mode', 'train') # 'train' or 'test'
    
    # 전역 로거 설정
    logger = setup_experiment_logger(experiment_code, path_conf['log_dir'])
    logger.info("="*60)
    logger.info(f"🎬 Experiment Started: {experiment_code} (Mode: {run_mode.upper()})")
    logger.info("="*60)

    try:
        # --------------------------------------------------------------------------
        # [Step 2] DB에 실험 정보 등록 (Experiment Table)
        # --------------------------------------------------------------------------
        logger.info("Step 0: Registering Experiment to DB...")
        
        with db_manager.get_db() as session:
            # 실험 정보가 이미 존재하는지 확인
            existing_exp = crud.get_experiment(session, experiment_code)
            
            if existing_exp:
                logger.warning(f"⚠️ Experiment {experiment_code} already exists. Updating start time...")
                crud.update_experiment(session, experiment_code, {
                    "experiment_start_time": datetime.now(),
                    "experiment_config": config,
                    "run_mode": run_mode 
                })
            else:
                # 신규 실험 생성
                exp_data = {
                    "experiment_code": experiment_code,
                    "previous_experiment_code": exp_conf.get('previous_experiment_code'),
                    "data_category": exp_conf.get('data_category', 'personal_data'),
                    "run_mode": run_mode,
                    "experiment_config": config,
                    "dataset_absolute_path": path_conf.get('data_dir'),
                    "experiment_start_time": datetime.now(),
                    "experiment_duration": 0.0,
                    "dataset_info": {} 
                }
                crud.create_experiment(session, exp_data)
                logger.info("✅ Experiment record created.")

        # --------------------------------------------------------------------------
        # [Step 3] 프로세스 순차 실행
        # --------------------------------------------------------------------------
        
        # [Process 0] 준비 단계
        # - 데이터셋 생성, 모델 초기화, 가중치 로드(Resume/Inference)가 모두 여기서 수행됨
        # - 준비된 객체들이 담긴 context 딕셔너리를 반환
        context = run_process_0(config)
        
        # [Process 1] Run Mode에 따른 실행 흐름 제어
        if run_mode == "train":
            # [Train Mode] Process 1 (학습 & 검증) 필수 실행
            if exp_conf.get('run_process_1', True):
                logger.info("▶️ Running Process 1 (Training)...")
                # 학습된 모델은 context에 업데이트되어 반환됨
                context = run_process_1(config, context)
            else:
                logger.info("⏭️ Process 1 skipped by config.")
                
        elif run_mode == "test":
            # [Test Mode] Process 1 (학습) 건너뜀
            # 모델 가중치는 이미 Process 0에서 'inference_checkpoint'로 로드되었음
            logger.info("⏭️ Skipping Process 1 (Training) due to TEST mode.")
        
        # [Process 2] 사전 매칭 검증
        if exp_conf.get('run_process_2', True):
            logger.info("▶️ Running Process 2 (Dictionary Matching)...")
            run_process_2(config, context)
            
        # [Process 3] 정규식 매칭 검증
        if exp_conf.get('run_process_3', True):
            logger.info("▶️ Running Process 3 (Regex Matching)...")
            run_process_3(config, context)
            
        # [Process 4] 모델 보완 추론 (Hybrid Logic)
        # 규칙 기반(Process_2, Process_3)이 놓친 데이터를 모델이 찾아내는지 검증
        if exp_conf.get('run_process_4', True):
            logger.info("▶️ Running Process 4 (Model Complementary Inference)...")
            run_process_4(config, context)

        # --------------------------------------------------------------------------
        # [Step 4] 실험 종료 처리
        # --------------------------------------------------------------------------
        with db_manager.get_db() as session:
            end_time = datetime.now()
            
            # 시작 시간 조회 (Duration 계산용)
            exp_obj = crud.get_experiment(session, experiment_code)
            start_time = exp_obj.experiment_start_time if exp_obj else end_time
            
            # Timezone 고려한 Duration 계산
            if start_time.tzinfo:
                duration = (end_time - start_time).total_seconds()
            else:
                duration = (end_time - start_time.replace(tzinfo=None)).total_seconds()

            # DB에 종료 시간 및 소요 시간 업데이트
            crud.update_experiment(session, experiment_code, {
                "experiment_end_time": end_time,
                "experiment_duration": duration
            })
            
        logger.info("="*60)
        logger.info(f"🏁 Experiment Finished Successfully. (Duration: {duration:.2f}s)")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"❌ Experiment Failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()