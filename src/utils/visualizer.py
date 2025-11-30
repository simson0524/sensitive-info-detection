# src/utils/visualizer.py

import matplotlib.pyplot as plt
import os

def plot_loss_graph(train_losses: list, valid_losses: list, save_dir: str, experiment_code: str):
    """
    Train Loss와 Validation Loss의 변화를 그래프로 그려 저장합니다.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Train Loss
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss')
    
    # Valid Loss (있을 경우에만)
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