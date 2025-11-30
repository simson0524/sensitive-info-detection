# src/utils/common.py

import os
import random
import yaml
import torch
import numpy as np

def load_yaml(path: str) -> dict:
    """
    YAML 설정 파일을 로드하여 딕셔너리로 반환합니다.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"YAML file not found: {path}")
        
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def set_seed(seed: int = 42):
    """
    실험 재현성을 위해 Python, Numpy, PyTorch의 랜덤 시드를 고정합니다.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # 멀티 GPU 사용 시
        
        # 성능보다 재현성을 우선시하는 설정
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    print(f"🔒 [Common] Global Seed set to {seed}")

def ensure_dir(path: str):
    """
    디렉토리가 존재하지 않으면 생성합니다.
    """
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"📂 [Common] Created directory: {path}")