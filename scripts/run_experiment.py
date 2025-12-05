# scripts/run_experiment.py

import sys
import os
import json
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
    4. 종료 처리 및 상세 결과 리포트(TXT) 생성
    """
    
    # --------------------------------------------------------------------------
    # [Step 1] 설정 로드(base & exp config merge load) 및 초기화
    # --------------------------------------------------------------------------
    base_conf_path = os.path.join(project_root, "configs", "base_config.yaml")
    exp_conf_path = os.path.join(project_root, "configs", "experiment_config.yaml")
    
    # 두 설정 파일을 로드 후 병합
    base_config = load_yaml(base_conf_path)
    exp_config = load_yaml(exp_conf_path)

    config = base_config.copy()
    
    for section, values in exp_config.items():
        if section in config and isinstance(config[section], dict) and isinstance(values, dict):
            # 하위 키(train, path 등) 업데이트
            config[section].update(values)
        else:
            # 새로운 섹션이면 통째로 추가
            config[section] = values

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
        context = run_process_0(config)
        
        # [Process 1] Run Mode에 따른 실행 흐름 제어
        if run_mode == "train":
            if exp_conf.get('run_process_1', True):
                logger.info("▶️ Running Process 1 (Training)...")
                context = run_process_1(config, context)
            else:
                logger.info("⏭️ Process 1 skipped by config.")
                
        elif run_mode == "test":
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
        if exp_conf.get('run_process_4', True):
            logger.info("▶️ Running Process 4 (Model Complementary Inference)...")
            run_process_4(config, context)

        # --------------------------------------------------------------------------
        # [Step 4] 실험 종료 처리 및 리포트 생성
        # --------------------------------------------------------------------------
        with db_manager.get_db() as session:
            end_time = datetime.now()
            exp_obj = crud.get_experiment(session, experiment_code)
            start_time = exp_obj.get('experiment_start_time') if exp_obj else end_time
            
            # crud.get_experiment가 dict를 반환하므로 .get() 사용 (이전 row_to_dict 적용됨)
            # 만약 datetime 객체라면 바로 사용
            
            if isinstance(start_time, datetime):
                # start_time이 timezone 정보(Aware)를 가지고 있다면 제거하여 Naive로 변환
                # end_time은 datetime.now()로 생성되어 기본적으로 Naive 상태임
                if start_time.tzinfo is not None:
                    start_time = start_time.replace(tzinfo=None)
                
                duration = (end_time - start_time).total_seconds()

            # DB 업데이트
            crud.update_experiment(session, experiment_code, {
                "experiment_end_time": end_time,
                "experiment_duration": duration
            })
            
            # [NEW] 상세 결과 리포트 생성 (TXT)
            generate_experiment_report(session, experiment_code, path_conf['log_dir'])
            
        logger.info("="*60)
        logger.info(f"🏁 Experiment Finished Successfully. (Duration: {duration:.2f}s)")
        logger.info("="*60)

    except Exception as e:
        logger.error(f"❌ Experiment Failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


def generate_experiment_report(session, experiment_code: str, log_dir: str):
    """
    DB에서 실험 정보와 모든 프로세스 결과를 조회하여 상세 리포트(TXT)를 생성합니다.
    outputs/logs/{code}/{code}_all_process_results.txt
    
    포함 내용:
    1. Experiment Table 정보 (설정, 시간 등 전체)
    2. Process 1: Epoch별 상세 지표 (Confusion Matrix 포함) 및 Best Epoch (F1, Loss)
    3. Process 2~4: 각 프로세스 결과 요약
    """
    report_lines = []
    
    # ---------------------------------------------------------
    # 1. Experiment General Info (Table Dump)
    # ---------------------------------------------------------
    exp_data = crud.get_experiment(session, experiment_code) # returns dict
    
    report_lines.append("="*80)
    report_lines.append(f"📊 EXPERIMENT REPORT: {experiment_code}")
    report_lines.append(f"Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("="*80 + "\n")
    
    report_lines.append("[1] General Information (Experiment Table)")
    report_lines.append("-" * 80)
    
    if exp_data:
        for key, value in exp_data.items():
            # Config 같은 큰 JSON은 보기 좋게 포맷팅
            if key == 'experiment_config' or key == 'dataset_info':
                val_str = json.dumps(value, indent=4, ensure_ascii=False)
                report_lines.append(f"* {key}:\n{val_str}")
            else:
                report_lines.append(f"* {key}: {value}")
    else:
        report_lines.append("Error: Experiment data not found.")
    report_lines.append("\n")

    # ---------------------------------------------------------
    # 2. Process 1 (Model Training) Details
    # ---------------------------------------------------------
    # process_code가 'process_1' 또는 'model_train' 인 것 필터링
    all_results = crud.get_process_results(session, experiment_code)
    p1_results = [r for r in all_results if r['process_code'] in ['process_1', 'model_train']]
    
    if p1_results:
        report_lines.append("[2] Process 1: Model Training & Validation")
        report_lines.append("-" * 80)
        
        # Best Metric Tracking
        best_f1 = -1.0
        best_f1_epoch = -1
        min_valid_loss = float('inf')
        min_loss_epoch = -1

        # Epoch별 상세 기록
        for res in p1_results:
            epoch = res['process_epoch']
            metrics = res['process_results']
            
            # 주요 지표 추출
            train_loss = metrics.get('train_loss', 0.0)
            valid_loss = metrics.get('valid_loss', 0.0)
            valid_f1 = metrics.get('valid_f1', 0.0)
            valid_prec = metrics.get('valid_precision', 0.0)
            valid_rec = metrics.get('valid_recall', 0.0)
            conf_matrix = metrics.get('confusion_matrix') # 2D List

            # Best Update
            if valid_f1 > best_f1:
                best_f1 = valid_f1
                best_f1_epoch = epoch
            
            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                min_loss_epoch = epoch
            
            # Line Writing
            report_lines.append(f"Epoch {epoch:02d}")
            report_lines.append(f"  - Train Loss: {train_loss:.5f} | Valid Loss: {valid_loss:.5f}")
            report_lines.append(f"  - F1: {valid_f1:.5f} | Precision: {valid_prec:.5f} | Recall: {valid_rec:.5f}")
            
            if conf_matrix:
                # Confusion Matrix 예쁘게 출력
                cm_str = json.dumps(conf_matrix) # 한 줄로 보거나
                # cm_str = json.dumps(conf_matrix, indent=2) # 여러 줄로 보거나 (여기선 한줄)
                report_lines.append(f"  - Confusion Matrix: {cm_str}")
            report_lines.append("") # 빈 줄

        # 요약 정보
        report_lines.append("-" * 40)
        report_lines.append("🏆 Process 1 Summary")
        report_lines.append(f"  - Best Model (Max F1): Epoch {best_f1_epoch} (F1: {best_f1:.5f})")
        report_lines.append(f"  - Best Model (Min Loss): Epoch {min_loss_epoch} (Loss: {min_valid_loss:.5f})")
        report_lines.append("-" * 80 + "\n")

    # ---------------------------------------------------------
    # 3. Other Processes (2, 3, 4) Results
    # ---------------------------------------------------------
    # 이 프로세스들은 보통 1회성 실행(Epoch 1)이므로 간단히 출력
    for proc_code in ["process_2", "process_3", "process_4"]:
        # 최신 결과 1개만 가져옴 (혹시 여러 번 돌렸을 수 있으니)
        results = [r for r in all_results if r['process_code'] == proc_code]
        if results:
            last_result = results[-1] # 가장 최근 것
            
            proc_name = proc_code.upper().replace("_", " ")
            report_lines.append(f"[{proc_name[-1]}th Step] {proc_name} Results")
            report_lines.append("-" * 80)
            
            metrics = last_result['process_results']
            formatted_json = json.dumps(metrics, indent=4, ensure_ascii=False)
            report_lines.append(formatted_json)
            report_lines.append("\n")

    # ---------------------------------------------------------
    # 4. 파일 저장
    # ---------------------------------------------------------
    save_path = os.path.join(log_dir, experiment_code, f"{experiment_code}_all_process_results.txt")
    
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(report_lines))
        print(f"📄 Final Report generated: {save_path}")
    except Exception as e:
        print(f"❌ Failed to write report: {e}")


if __name__ == "__main__":
    main()