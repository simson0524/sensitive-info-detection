# src/utils/common.py

import os
import random
import yaml
import torch
import numpy as np
import pandas as pd

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

def save_logs_to_csv(logs: list, save_path: str):
    """
    로그 리스트(Dict)를 CSV 파일로 저장합니다.
    JSON 구조의 필드는 문자열로 변환되어 저장될 수 있습니다.
    """
    if not logs:
        return

    try:
        # 데이터프레임 생성
        df = pd.DataFrame(logs)
        
        # CSV 저장 (utf-8-sig: 엑셀 한글 깨짐 방지)
        df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"📄 Log saved to: {save_path}")
        
    except Exception as e:
        print(f"❌ Failed to save CSV: {e}")

def normalize_label(label: str) -> str:
    """
    라벨 이름에서 접미사(_숫자)를 제거하여 원본 카테고리명을 반환합니다.
    
    [동작 예시]
    - "개인정보_1" -> "개인정보"
    - "개인정보_2" -> "개인정보"
    - "개인정보"   -> "개인정보" (변화 없음)
    - "전화번호"   -> "전화번호"
    """
    if not label:
        return label
        
    # "_"가 있고 마지막 부분이 숫자인 경우에만 분리
    if "_" in label and label.rsplit("_", 1)[-1].isdigit():
        return label.rsplit("_", 1)[0]
    
    return label