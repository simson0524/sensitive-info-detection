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
    system_name = platform.system()
    if system_name == 'Darwin':
        rc('font', family='AppleGothic')
    elif system_name == 'Linux':
        rc('font', family='NanumGothic')
    rc('axes', unicode_minus=False)

set_korean_font()

def plot_loss_graph(train_losses: list, valid_losses: list, save_dir: str, experiment_code: str):
    """기존 로직 유지"""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss')
    if valid_losses:
        plt.plot(epochs, valid_losses, 'r-s', label='Validation Loss')
    plt.title(f'Loss Trend - {experiment_code}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    save_path = os.path.join(save_dir, f"{experiment_code}_loss_graph.png")
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix_trends(cm_history: list, save_dir: str, experiment_code: str):
    """
    [수정] 에포크별 의미 단위(Pure Label) 예측 추이를 시각화합니다.
    각 GT 라벨별로 어떤 Pred 라벨로 분류되었는지 Stacked Bar 형태로 보여줍니다.
    """
    if not cm_history: return

    # cm_history[0] 구조: {"labels": ["PER", "ORG", "O"], "values": [[...]]}
    labels = cm_history[0]['labels']
    num_labels = len(labels)
    num_epochs = len(cm_history)
    epochs = np.arange(1, num_epochs + 1)

    # 각 GT 라벨별로 Subplot 생성
    fig, axes = plt.subplots(num_labels, 1, figsize=(12, 4 * num_labels), constrained_layout=True)
    if num_labels == 1: axes = [axes]

    # 색상 팔레트 생성 (라벨별 고유 색상)
    colors = sns.color_palette("husl", num_labels)
    label_to_color = {label: colors[i] for i, label in enumerate(labels)}

    for g_idx, gt_label in enumerate(labels):
        ax = axes[g_idx]
        
        # 데이터를 쌓기 위한 준비 (Stacked Bar)
        bottom = np.zeros(num_epochs)
        
        for p_idx, pred_label in enumerate(labels):
            ratios = []
            for epoch_data in cm_history:
                matrix = epoch_data['values']
                # 행: GT, 열: Pred 기준 (Evaluator에서 만든 구조에 맞춤)
                total = sum(matrix[g_idx])
                val = matrix[g_idx][p_idx]
                ratios.append((val / total * 100) if total > 0 else 0)
            
            ax.bar(epochs, ratios, bottom=bottom, label=f"Pred: {pred_label}", color=label_to_color[pred_label])
            bottom += ratios

        ax.set_title(f"GT Label 추이: {gt_label}", fontsize=14, fontweight='bold')
        ax.set_ylabel("비중 (%)")
        ax.set_xlabel("Epoch")
        ax.set_ylim(0, 100)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    save_path = os.path.join(save_dir, f"{experiment_code}_cm_trends.png")
    plt.savefig(save_path)
    plt.close()

def plot_label_relation_matrix(cm_data: dict, save_dir: str, filename_prefix: str):
    """
    [NEW] 특정 에포크(Best Epoch)의 의미 단위 Confusion Matrix 히트맵
    """
    labels = cm_data['labels']
    values = cm_data['values']
    
    df_cm = pd.DataFrame(values, index=labels, columns=labels)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues', cbar=True)
    plt.title(f"Entity-level Confusion Matrix ({filename_prefix})")
    plt.xlabel("Predicted Label")
    plt.ylabel("Ground Truth Label")
    
    save_path = os.path.join(save_dir, f"{filename_prefix}_relation_matrix.png")
    plt.savefig(save_path)
    plt.close()

def plot_label_accuracy_histograms(accuracy_dist: dict, save_dir: str, filename_prefix: str):
    """
    [NEW] 라벨별 엔티티 정확도 분포 백분율 히스토그램
    X축: 0.0 ~ 1.0 (정확도 점수)
    """
    num_labels = len(accuracy_dist)
    if num_labels == 0: return

    fig, axes = plt.subplots(num_labels, 1, figsize=(10, 4 * num_labels), constrained_layout=True)
    if num_labels == 1: axes = [axes]

    for i, (label, scores) in enumerate(accuracy_dist.items()):
        ax = axes[i]
        if not scores:
            ax.text(0.5, 0.5, f"{label}: No Data", ha='center')
            continue
            
        # 0.0, 0.5, 0.75, 1.0 등 특정 구간이 강조되도록 bin 설정
        sns.histplot(scores, bins=20, binrange=(0, 1), ax=ax, kde=False, color='skyblue', edgecolor='black')
        
        ax.set_title(f"정확도 분포 (B성공 0.5 + I비율 0.5): {label}", fontsize=12)
        ax.set_xlabel("Accuracy Score")
        ax.set_ylabel("Entity Count")
        ax.set_xlim(-0.05, 1.05)
        
        # 평균선 추가
        avg_score = np.mean(scores)
        ax.axvline(avg_score, color='red', linestyle='--', label=f'Avg: {avg_score:.2f}')
        ax.legend()

    save_path = os.path.join(save_dir, f"{filename_prefix}_accuracy_dist.png")
    plt.savefig(save_path)
    plt.close()


def plot_z_score_distribution(df: pd.DataFrame, save_dir: str):
    """
    [Visualization Only] 
    전달받은 DataFrame(z_score, is_sensitive_label 포함)을 바탕으로 
    0.2 단위 구간 분포를 그립니다.
    """
    if df.empty:
        print("⚠️ [Visualizer] 시각화할 데이터가 없습니다 (DataFrame empty).")
        return

    # 1. Binning 설정 (0.2 단위)
    # 범위: -2.0 ~ 3.0, 그 외 구간은 < -2.0, 3.0+ 로 처리
    bin_edges = np.arange(-2.0, 3.2, 0.05) 
    bins = [-float('inf')] + list(bin_edges) + [float('inf')]
    
    labels = ['< -2.0']
    for i in range(len(bin_edges)-1):
        labels.append(f"{bin_edges[i]:.1f}~{bin_edges[i+1]:.1f}")
    labels.append('3.0+')
    
    # 데이터 구간화 (Score와 Is_Sensitive 컬럼명 기준)
    # 만약 원본 DF 컬럼명이 다르면 여기서 맞춰줍니다.
    plot_df = df.copy()
    plot_df['Score_Bin'] = pd.cut(plot_df['z_score'], bins=bins, labels=labels)
    plot_df['Label_Group'] = plot_df['is_sensitive_label'].map({True: 'Sensitive (민감)', False: 'Normal (일반)'})

    # 2. 스타일 설정
    sns.set_style("whitegrid")
    # 한글 깨짐 방지 (시스템에 따라 폰트명은 수정될 수 있습니다)
    plt.rcParams['font.family'] = 'NanumGothic' 
    plt.rcParams['axes.unicode_minus'] = False

    # 3. 그래프 그리기 (2단 구성)
    # 두 라벨 간의 데이터 편차가 크므로 sharey=False로 설정
    fig, axes = plt.subplots(2, 1, figsize=(20, 12))
    
    palette = {'Sensitive (민감)': '#ff6b6b', 'Normal (일반)': '#54a0ff'}
    groups = ['Normal (일반)', 'Sensitive (민감)']

    for i, group_name in enumerate(groups):
        group_data = plot_df[plot_df['Label_Group'] == group_name]
        
        sns.countplot(
            data=group_data,
            x='Score_Bin',
            ax=axes[i],
            color=palette[group_name],
            edgecolor='black',
            linewidth=0.5,
            order=labels # 모든 구간이 표시되도록 순서 고정
        )
        
        axes[i].set_title(f"Z-Score Distribution: {group_name}", fontsize=16, fontweight='bold')
        axes[i].set_ylabel("Count")
        axes[i].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    
    # 4. 저장
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "z_score_distribution.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"📊 [Visualizer] 시각화 이미지가 저장되었습니다: {save_path}")