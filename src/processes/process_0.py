# src/processes/process_0.py

import torch
import os
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split

# Modules: Ner모델과 해당 모델에 데이터셋을 로드하기 위한 전처리 모듈
from src.modules.ner_preprocessor import NerPreprocessor
from src.models.ner_roberta import RobertaNerModel

# Utils: 공통 유틸리티 함수
from src.utils.common import set_seed
from src.utils.logger import setup_experiment_logger

def run_process_0(config: dict) -> dict:
    """
    [Process 0] 학습 환경 및 객체 초기화 프로세스 (Setup Phase)
    
    이 함수의 역할:
    1. 데이터 로드: 하나의 원본 폴더에서 데이터를 읽어 Train/Valid로 자동 분할합니다.
    2. 모델 초기화: 설정된 아키텍처(RoBERTa 등)로 모델 껍데기를 만듭니다.
    3. 가중치 로드: 만약 이어서 학습(Resume)해야 한다면 저장된 가중치를 불러옵니다.
    4. 도구 준비: Optimizer, Scheduler 등을 준비하여 패키징(Context)합니다.
    
    Args:
        config (dict): experiment_config.yaml에서 로드한 설정값
        
    Returns:
        dict: 학습에 필요한 모든 객체가 담긴 Context
    """
    
    # ==============================================================================
    # [Step 1] 설정 로드 및 로거 초기화
    # ==============================================================================
    exp_conf = config['experiment']
    train_conf = config['train']
    path_conf = config['path']
    experiment_code = exp_conf['experiment_code']
    run_mode = exp_conf.get('run_mode', 'train') # 무조건 'train' or 'test'
    
    # 로거 생성 (이미 존재하면 가져오고, 없으면 파일과 함께 생성)
    logger = setup_experiment_logger(experiment_code, path_conf['log_dir'])
    logger.info(f"🛠️ [Process 0] Initializing Experiment: {experiment_code}")

    # 재현성을 위해 랜덤 시드 고정 (데이터 분할 결과가 매번 같아야 함)
    set_seed(train_conf.get('seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # ==============================================================================
    # [Step 2] 데이터 전처리 및 로드 (Data Preparation)
    # ==============================================================================
    logger.info("Step 1: Loading Data & Splitting Train/Valid...")
    
    # HuggingFace Tokenizer 로드
    tokenizer = AutoTokenizer.from_pretrained(train_conf['model_name'])
    
    # 2-1. 전처리기(Preprocessor) 초기화
    # 이 친구가 JSON 로드, BIO 태깅 변환 등을 담당합니다.
    preprocessor = NerPreprocessor(
        tokenizer=tokenizer, 
        max_len=train_conf['max_len'], 
        label2id=train_conf['label_map']
    )
    
    # 2-2. 전체 Raw Data 로드
    # 지정된 폴더(path_conf['data_dir']) 내의 모든 JSON 파일을 읽어옵니다.
    all_samples, all_annos = preprocessor.load_data(path_conf['data_dir'])
    
    total_count = len(all_samples)
    if total_count == 0:
        # 데이터가 없으면 더 이상 진행할 수 없으므로 에러 발생
        raise ValueError(f"❌ No data found in {path_conf['data_dir']}")

    # 2-3. Train / Valid 자동 분할 (sklearn 사용)
    # 별도의 검증 폴더를 두지 않고, 전체 데이터에서 일정 비율을 떼어내어 검증용으로 씁니다.
    val_ratio = train_conf.get('validation_split', 0.2) # 기본값 20%
    all_ids = list(all_samples.keys())
    
    # ID 리스트를 섞어서 나눕니다. (random_state가 고정되어 있어 매번 결과가 같음)
    train_ids, valid_ids = train_test_split(
        all_ids, 
        test_size=val_ratio, 
        random_state=train_conf.get('seed', 42), 
        shuffle=True
    )
    
    # ID를 기준으로 실제 데이터를 딕셔너리에서 추출하여 재구성합니다.
    train_samples = {uid: all_samples[uid] for uid in train_ids}
    train_annos = {uid: all_annos[uid] for uid in train_ids}
    
    valid_samples = {uid: all_samples[uid] for uid in valid_ids}
    valid_annos = {uid: all_annos[uid] for uid in valid_ids}

    logger.info(f"📊 Data Split Result: Total({total_count}) -> Train({len(train_ids)}) / Valid({len(valid_ids)})")

    # 2-4. Dataset 객체 생성 (실제 토큰화 및 BIO 태깅 수행)
    # 개인정보/기밀정보 여부에 따라 필터링 옵션(data_category)을 적용합니다.
    data_category = exp_conf.get('data_category', 'personal_data')
    
    logger.info("Creating Train Dataset...")
    train_dataset = preprocessor.create_dataset(train_samples, train_annos, data_category=data_category)
    
    logger.info("Creating Valid Dataset...")
    valid_dataset = preprocessor.create_dataset(valid_samples, valid_annos, data_category=data_category)
    
    # 2-5. DataLoader 생성 (Batch 단위 공급기)
    train_loader = DataLoader(train_dataset, batch_size=train_conf['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=train_conf['batch_size'], shuffle=False)


    # ==============================================================================
    # [Step 3] 모델 초기화 및 가중치 로드 (Model Setup)
    # ==============================================================================
    logger.info("Step 2: Building Model & Optimizer...")
    
    # 기본 Encoder (RoBERTa) 로드
    encoder = AutoModel.from_pretrained(train_conf['model_name'])
    num_labels = len(preprocessor.ner_label2id) # BIO 태그 개수 자동 계산
    
    # 우리가 정의한 Custom NER 모델 생성
    model = RobertaNerModel(
        encoder=encoder,
        num_classes=num_labels,
        use_focal=train_conf.get('use_focal', False) # Focal Loss 사용 여부
    ).to(device)

    # --------------------------------------------------------------------------
    # [중요] 학습 재개 (Resume Training) 로직
    # config에 'resume_checkpoint' 경로가 있고, 파일이 실제로 존재하면 가중치를 덮어씌웁니다.
    # --------------------------------------------------------------------------
    target_ckpt_path = None
    
    if run_mode == 'test':
        # Test 모드: inference_checkpoint 로드 (필수)
        target_ckpt_path = path_conf.get('inference_checkpoint')
        if not target_ckpt_path:
            logger.warning("⚠️ [Test Mode] 'inference_checkpoint' is not set in config!")
    else:
        # Train 모드: resume_checkpoint 로드 (선택)
        target_ckpt_path = path_conf.get('resume_checkpoint')

    # 경로가 존재하면 로드 수행
    if target_ckpt_path and os.path.exists(target_ckpt_path):
        logger.info(f"📥 Loading Weights from: {target_ckpt_path}")
        try:
            state_dict = torch.load(target_ckpt_path, map_location=device)
            model.load_state_dict(state_dict)
            logger.info("✅ Weights loaded successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint: {e}")
            raise e
    else:
        if run_mode == 'test':
            logger.warning("⚠️ Running TEST mode with RANDOM weights (Checkpoint not found).")
        else:
            logger.info("🆕 Initialized model with Base Weights (No resume checkpoint found).")

    # ==============================================================================
    # [Step 4] 학습 도구 설정 (Optimizer & Scheduler)
    # ==============================================================================
    optimizer = AdamW(model.parameters(), lr=float(train_conf['learning_rate']))
    
    total_steps = len(train_loader) * train_conf['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * 0.1), # 전체 스텝의 10% 동안 Warmup
        num_training_steps=total_steps
    )

    logger.info("✅ [Process 0] Setup Completed Successfully.")

    # ==============================================================================
    # [Step 5] Context 패키징 및 반환
    # ==============================================================================
    # 다음 프로세스(Process 1, 2...)에서 사용할 객체들을 딕셔너리에 담아 보냅니다.
    context = {
        "experiment_code": experiment_code,
        "device": device,
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        
        # Preprocessor 객체 (토크나이저, 라벨맵 포함)는 후속 프로세스에서도 계속 필요함
        "preprocessor": preprocessor, 
        
        # Dataset 객체 (상태 유지용, Process 4 등에서 재활용)
        "train_dataset": train_dataset, 
        "valid_dataset": valid_dataset
    }
    
    return context