# src/processes/process_0.py

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split

# Modules
from src.modules.ner_preprocessor import NerPreprocessor
from src.models.ner_roberta import RobertaNerModel

# Utils
from src.utils.common import set_seed
from src.utils.logger import setup_experiment_logger

def run_process_0(config: dict) -> dict:
    """
    [Process 0] 학습 환경 및 객체 초기화 (Setup Phase)
    - 단일 데이터 디렉토리 로드 -> Train/Valid 자동 분할
    - 데이터셋, 모델, 옵티마이저 생성
    """
    # 1. 설정 및 로거
    exp_conf = config['experiment']
    train_conf = config['train']
    path_conf = config['path']
    experiment_code = exp_conf['experiment_code']
    
    logger = setup_experiment_logger(experiment_code, path_conf['log_dir'])
    logger.info(f"🛠️ [Process 0] Initializing Experiment: {experiment_code}")

    # 시드 고정 (데이터 분할의 재현성을 위해 매우 중요)
    seed = train_conf.get('seed', 42)
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 2. 전처리기 및 데이터셋 생성
    logger.info("Step 1: Loading Data & Splitting Train/Valid...")
    tokenizer = AutoTokenizer.from_pretrained(train_conf['model_name'])
    
    # 2-1. Preprocessor 초기화
    preprocessor = NerPreprocessor(
        tokenizer=tokenizer, 
        max_len=train_conf['max_len'], 
        label2id=train_conf['label_map']
    )
    
    # 2-2. 전체 Raw Data 로드 (단일 디렉토리)
    # config['path']['data_dir']에 모든 json 파일이 있어야 합니다.
    all_samples, all_annos = preprocessor.load_data(path_conf['data_dir'])
    
    total_count = len(all_samples)
    if total_count == 0:
        raise ValueError(f"No data found in {path_conf['data_dir']}")

    # 2-3. Train / Valid 자동 분할
    # validation_split 설정이 없으면 기본값 0.2 (20%) 사용
    val_ratio = train_conf.get('validation_split', 0.2)
    
    # 샘플 ID(키)를 기준으로 분할
    all_ids = list(all_samples.keys())
    train_ids, valid_ids = train_test_split(
        all_ids, 
        test_size=val_ratio, 
        random_state=seed, 
        shuffle=True
    )
    
    # ID를 이용해 딕셔너리 재구성
    train_samples = {uid: all_samples[uid] for uid in train_ids}
    train_annos = {uid: all_annos[uid] for uid in train_ids}
    
    valid_samples = {uid: all_samples[uid] for uid in valid_ids}
    valid_annos = {uid: all_annos[uid] for uid in valid_ids}

    logger.info(f"Data Split: Total({total_count}) -> Train({len(train_ids)}), Valid({len(valid_ids)})")

    # 2-4. Dataset 생성
    data_category = exp_conf.get('data_category', 'personal_data')
    
    logger.info("Creating Train Dataset...")
    train_dataset = preprocessor.create_dataset(train_samples, train_annos, data_category=data_category)
    
    logger.info("Creating Valid Dataset...")
    valid_dataset = preprocessor.create_dataset(valid_samples, valid_annos, data_category=data_category)
    
    # 2-5. DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=train_conf['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=train_conf['batch_size'], shuffle=False)

    # 3. 모델 및 학습 도구 초기화
    logger.info("Step 2: Building Model & Optimizer...")
    
    encoder = AutoModel.from_pretrained(train_conf['model_name'])
    num_labels = len(preprocessor.ner_label2id)
    
    model = RobertaNerModel(
        encoder=encoder,
        num_classes=num_labels,
        use_focal=train_conf.get('use_focal', False)
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=float(train_conf['learning_rate']))
    
    total_steps = len(train_loader) * train_conf['epochs']
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * 0.1), 
        num_training_steps=total_steps
    )

    logger.info("✅ [Process 0] Setup Completed.")

    # 4. Context 반환
    context = {
        "experiment_code": experiment_code,
        "device": device,
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "preprocessor": preprocessor,
        "train_dataset": train_dataset,
        "valid_dataset": valid_dataset
    }
    
    return context