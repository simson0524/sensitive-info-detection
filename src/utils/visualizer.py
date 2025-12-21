# src/utils/visualizer.py

import os
import platform
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib import rc
from typing import List, Dict, Optional

def set_korean_font():
    """
    OS(Mac/Linux)에 따라 Matplotlib 한글 폰트를 설정합니다.
    """
    system_name = platform.system()
    
    # 1. 폰트 설정
    if system_name == 'Darwin': # Mac OS
        rc('font', family='AppleGothic')
    elif system_name == 'Linux': # Linux (Ubuntu)
        # 나눔고딕 경로 확인 (일반적인 Ubuntu 경로)
        # 폰트가 설치되어 있어야 함 (sudo apt-get install fonts-nanum)
        rc('font', family='NanumGothic')
    else:
        # Windows 등 기타 (필요시 추가)
        pass
        
    # 2. 마이너스(-) 기호 깨짐 방지
    rc('axes', unicode_minus=False)

# 모듈 임포트 시 자동으로 폰트 설정 실행
set_korean_font()


def plot_loss_graph(train_losses: list, valid_losses: list, save_dir: str, experiment_code: str):
    """
    Train/Valid Loss 추이를 그래프로 그려 저장합니다.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss')
    if valid_losses and len(valid_losses) == len(train_losses):
        plt.plot(epochs, valid_losses, 'r-s', label='Validation Loss')
    
    plt.title(f'Loss Trend - {experiment_code}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(save_dir, f"{experiment_code}_loss_graph.png")
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Loss Graph saved to {save_path}")


def plot_confusion_matrix_trends(cm_history: list, id2label: dict, save_dir: str, experiment_code: str):
    """
    [NEW] Epoch별 Confusion Matrix 변화 추이를 그래프로 그립니다.
    각 GT(정답) 라벨별로, 모델이 어떻게 예측했는지 비율(%) 변화를 보여줍니다.
    
    Args:
        cm_history: List[List[List[int]]] (Epochs x Pred x GT) 구조의 CM 리스트
        id2label: {0: 'O', 1: 'B-PER', ...}
    """
    if not cm_history:
        return

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    num_epochs = len(cm_history)
    epochs = range(1, num_epochs + 1)
    num_labels = len(id2label)
    
    # 그래프 설정: 라벨 개수만큼 Subplot 생성 (세로로 배치)
    # constrained_layout=True를 사용하여 한글 제목 겹침 방지
    fig, axes = plt.subplots(num_labels, 1, figsize=(10, 5 * num_labels), constrained_layout=True)
    if num_labels == 1: axes = [axes] # 라벨이 1개일 경우 리스트로 변환

    # 각 GT 라벨(Target)에 대해 반복
    for gt_idx in range(num_labels):
        gt_name = id2label[gt_idx]
        ax = axes[gt_idx]
        
        # 해당 GT에 대한 Epoch별 예측 분포 수집
        # history_per_pred: {pred_idx: [epoch1_pct, epoch2_pct, ...]}
        history_per_pred = {p_idx: [] for p_idx in range(num_labels)}
        
        for epoch_cm in cm_history:
            # epoch_cm 구조: row=Pred, col=GT
            # 현재 GT 컬럼의 총합 계산 (해당 Epoch의 해당 라벨 총 샘플 수)
            total_samples = sum(row[gt_idx] for row in epoch_cm)
            
            for pred_idx in range(num_labels):
                count = epoch_cm[pred_idx][gt_idx]
                # 0으로 나누기 방지
                percent = (count / total_samples * 100) if total_samples > 0 else 0.0
                history_per_pred[pred_idx].append(percent)
        
        # 그래프 그리기
        has_plotted = False
        for pred_idx, pct_list in history_per_pred.items():
            # 모든 Epoch에서 0%인 예측 라벨은 그리지 않음 (가독성 향상)
            if all(p == 0.0 for p in pct_list):
                continue
                
            pred_name = id2label[pred_idx]
            ax.plot(epochs, pct_list, marker='.', label=f"Pred: {pred_name}")
            has_plotted = True

        ax.set_title(f"Ground Truth: {gt_name}")
        ax.set_ylabel("Prediction Ratio (%)")
        ax.set_xlabel("Epoch")
        ax.set_ylim(-5, 105) # 0~100% 범위 고정
        ax.grid(True, linestyle='--', alpha=0.7)
        
        if has_plotted:
            # 범례 위치 조정
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        else:
            ax.text(0.5, 0.5, "No Samples Found", ha='center', va='center')

    fig.suptitle(f"Label Prediction Distribution Over Epochs - {experiment_code}", fontsize=16)
    
    save_path = os.path.join(save_dir, f"{experiment_code}_label_count_graph.png")
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Label Count Graph saved to {save_path}")


def plot_z_score_distribution(df: pd.DataFrame, save_dir: str):
    """
    Z-Score 분포를 시각화하여 저장합니다. (Global vs Local 비교)
    
    Args:
        df (pd.DataFrame): 컬럼 ['Type', 'Score', 'Label', 'Domain']을 가진 데이터프레임
                           - Type: 'Global Z-Score' or 'Local Z-Score'
                           - Label: '개인정보', '준식별자' 등
        save_dir (str): 저장할 디렉토리 경로 (보통 data/train_data/)
    """
    if df.empty:
        print("⚠️ [Visualizer] Empty DataFrame provided for Z-Score plotting.")
        return
        
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Binning (점수 구간화)
    bins = [-float('inf'), -1, 0, 1, 2, 3, 4, float('inf')]
    labels = ['< -1', '-1~0', '0~1', '1~2', '2~3', '3~4', '4+']
    df['Score_Bin'] = pd.cut(df['Score'], bins=bins, labels=labels)

    # 스타일 설정
    sns.set_style("whitegrid")
    set_korean_font() # seaborn 스타일 적용 후 폰트 재설정

    # 색상 팔레트
    palette = {
        '개인정보': '#ff6b6b',    # Red
        '준식별자': '#feca57',    # Yellow
        '기밀정보': '#48dbfb',    # Blue
        'Non-labeled': '#c8d6e5', # Grey
        'Other': '#c8d6e5'
    }
    
    # 캔버스 생성
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # 1. Global Plot
    sns.countplot(
        data=df[df['Type'] == 'Global Z-Score'],
        x='Score_Bin', hue='Label', palette=palette,
        ax=axes[0], edgecolor='black', linewidth=0.5
    )
    axes[0].set_title("Global Z-Score Distribution (Entire Corpus)", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("Word Count")

    # 2. Local Plot
    sns.countplot(
        data=df[df['Type'] == 'Local Z-Score'],
        x='Score_Bin', hue='Label', palette=palette,
        ax=axes[1], edgecolor='black', linewidth=0.5
    )
    axes[1].set_title("Local Z-Score Distribution (Per Domain)", fontsize=14, fontweight='bold')
    axes[1].set_ylabel("Word Count")

    plt.tight_layout()
    
    save_path = os.path.join(save_dir, "z_score_distribution.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 [Visualizer] Z-Score Distribution saved to {save_path}")