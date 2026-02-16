# src/processes/process_0.py

import torch
import os
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
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
    1. 라벨 맵 선택: data_category(개인/기밀)에 따라 적절한 라벨 맵을 로드합니다.
    2. 데이터 로드: 하나의 원본 폴더에서 데이터를 읽어 Train/Valid로 자동 분할합니다.
    3. 모델 초기화: 설정된 아키텍처(RoBERTa 등)로 모델 껍데기를 만듭니다.
    4. 가중치 로드: 만약 이어서 학습(Resume)해야 한다면 저장된 가중치를 불러옵니다.
    5. 도구 준비: Optimizer, Scheduler 등을 준비하여 패키징(Context)합니다.
    
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
    label_settings = config['label_settings'] # [NEW] base_config에서 로드된 라벨 설정
    target_device = train_conf['device']
    
    experiment_code = exp_conf['experiment_code']
    run_mode = exp_conf.get('run_mode', 'train') # 'train' or 'test'
    
    # 로거 생성 (이미 존재하면 가져오고, 없으면 파일과 함께 생성)
    logger = setup_experiment_logger(experiment_code, path_conf['log_dir'])
    logger.info(f"🛠️ [Process 0] Initializing Experiment: {experiment_code} (Mode: {run_mode.upper()})")

    # 재현성을 위해 랜덤 시드 고정
    set_seed(train_conf.get('seed', 42))
    device = torch.device(target_device if torch.cuda.is_available() else 'cpu')

    # [핵심] 데이터 카테고리에 따른 라벨 맵 선택
    data_category = exp_conf.get('data_category', 'personal_data')
    
    if data_category not in label_settings:
        raise ValueError(f"❌ Unknown data_category: '{data_category}'. Check config files.")
        
    # 해당 카테고리의 라벨 맵 가져오기 (예: personal_data -> {'일반':0, '개인':1, '준식별':2})
    current_label_map = label_settings[data_category]['label_map']
    
    logger.info(f"🎯 Target Category: {data_category}")
    logger.info(f"🏷️  Active Label Map: {current_label_map}")


    # ==============================================================================
    # [Step 2] 데이터 전처리 및 로드 (Data Preparation)
    # ==============================================================================
    logger.info("Step 1: Loading Data & Splitting Train/Valid...")
    
    # HuggingFace Tokenizer 로드
    tokenizer = AutoTokenizer.from_pretrained(train_conf['model_name'])
    
    # 2-1. 전처리기(Preprocessor) 초기화
    # 선택된 라벨 맵(current_label_map)을 주입하여 BIO 태깅 규칙을 생성합니다.
    preprocessor = NerPreprocessor(
        tokenizer=tokenizer, 
        max_len=train_conf['max_len'], 
        label2id=current_label_map # [수정됨] 동적으로 선택된 맵 사용
    )
    
    # 2-2. 전체 All(train, valid) Data + Test Data로드
    all_samples, all_annos = preprocessor.load_data(path_conf['data_dir'])
    
    total_count = len(all_samples)
    if total_count == 0:
        # 학습 데이터가 없으면 더 이상 진행할 수 없으므로 에러 발생
        raise ValueError(f"❌ No data found in {path_conf['data_dir']}")

    # 2-3. Train / Valid 자동 분할
    val_ratio = train_conf.get('validation_split', 0.2)
    all_ids = list(all_samples.keys())
    
    train_ids, valid_ids = train_test_split(
        all_ids, 
        test_size=val_ratio, 
        random_state=train_conf.get('seed', 42), 
        shuffle=True
    )
    
    # 학습 데이터
    train_samples = {uid: all_samples[uid] for uid in train_ids}
    train_annos = {uid: all_annos[uid] for uid in train_ids}
    
    # 검증 데이터
    valid_samples = {uid: all_samples[uid] for uid in valid_ids}
    valid_annos = {uid: all_annos[uid] for uid in valid_ids}

    # 테스트 데이터
    if path_conf['test_data_dir']:
        test_samples, test_annos = preprocessor.load_data(path_conf['test_data_dir'])
    else:
        test_samples = valid_samples
        test_annos = valid_annos

    logger.info(f"📊 Data Split Result: Total({total_count}) -> Train({len(train_ids)}) / Valid({len(valid_ids)})")
    logger.info(f"📊 Test Data Result: Test({len(test_annos)})")

    # 2-4. Dataset 객체 생성
    # data_category를 전달하여 해당 카테고리에 맞는 라벨만 필터링하도록 함
    logger.info("Creating Train Dataset...")
    train_dataset = preprocessor.create_dataset(train_samples, train_annos, data_category=data_category)
    
    logger.info("Creating Valid Dataset...")
    valid_dataset = preprocessor.create_dataset(valid_samples, valid_annos, data_category=data_category)

    logger.info("Creating Test Dataset...")
    test_dataset = preprocessor.create_dataset(test_samples, test_annos, data_category=data_category)
    
    # 2-5. DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=train_conf['batch_size'], shuffle=True, collate_fn=smart_collate_fn)
    valid_loader = DataLoader(valid_dataset, batch_size=train_conf['batch_size'], shuffle=False, collate_fn=smart_collate_fn)
    test_loader  = DataLoader(test_dataset, batch_size=train_conf['batch_size'], shuffle=False, collate_fn=smart_collate_fn)


    # ==============================================================================
    # [Step 3] 모델 초기화 및 가중치 로드 (Model Setup)
    # ==============================================================================
    logger.info("Step 2: Building Model & Loading Weights...")
    
    encoder = AutoModel.from_pretrained(train_conf['model_name'])
    num_labels = len(preprocessor.ner_label2id) # BIO 태그 개수 (자동 계산됨)
    
    # Custom NER 모델 생성 (출력 클래스 개수는 num_labels에 맞춰짐)
    model = RobertaNerModel(
        encoder=encoder,
        num_classes=num_labels,
        use_focal=train_conf.get('use_focal', False)
    ).to(device)

    # --------------------------------------------------------------------------
    # [중요] 가중치 로드 통합 로직 (Train/Test 공통)
    # run_mode에 따라 적절한 체크포인트 경로를 선택하여 로드합니다.
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

    # 경로가 유효하면 가중치 로드 수행
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
        num_warmup_steps=int(total_steps * 0.1),
        num_training_steps=total_steps
    )

    logger.info("✅ [Process 0] Setup Completed Successfully.")

    # ==============================================================================
    # [Step 5] Context 패키징 및 반환
    # ==============================================================================
    context = {
        "experiment_code": experiment_code,
        "device": device,
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "test_loader": test_loader,
        "preprocessor": preprocessor, 
        "train_dataset": train_dataset, 
        "valid_dataset": valid_dataset,
        "test_dataset": test_dataset,
        "best_epoch": 0
    }
    
    return context


import torch

def smart_collate_fn(batch):
    keys = batch[0].keys()
    output = {}

    for key in keys:
        sample = batch[0][key]

        # [Case 1] 텐서인 경우 -> Stack
        if isinstance(sample, torch.Tensor):
            output[key] = torch.stack([item[key] for item in batch])
            
        # [Case 2] 숫자(int, float)인 경우 -> Try Tensor conversion
        elif isinstance(sample, (int, float)):
            try:
                # 일단 텐서 변환을 시도합니다.
                output[key] = torch.tensor([item[key] for item in batch])
            except (ValueError, TypeError, RuntimeError):
                # 앗! 중간에 문자열이나 None이 섞여있어서 실패했네요.
                # 그러면 그냥 안전하게 리스트로 저장합니다.
                output[key] = [item[key] for item in batch]
            
        # [Case 3] 그 외 (문자열 등) -> List
        else:
            output[key] = [item[key] for item in batch]
            
    return output