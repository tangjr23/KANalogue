import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def train_process_visualize(epochs, 
                            train_losses, val_losses, 
                            train_accuracies, val_accuracies, 
                            dataset, title, parent_folder):

    fig, ax1 = plt.subplots(figsize=(8, 6))

    ax1.plot(epochs, train_losses, label='Train Loss', linestyle='-', color="#ECA625")
    ax1.plot(epochs, val_losses, label='Val Loss', linestyle='--', color="#8f56bd")
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.grid(True, axis='both')

    ax2 = ax1.twinx()
    ax2.plot(epochs, train_accuracies, label='Train Acc', linestyle='-', color="#68d3a5")
    ax2.plot(epochs, val_accuracies, label='Val Acc', linestyle='--', color="#239ef6")
    ax2.set_ylabel('Accuracy')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1+lines2, labels1+labels2, loc='best')

    plt.title(f'{dataset}_{title}')
    plt.tight_layout()

    save_path = f'{parent_folder}/{dataset}/train_process'
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(f'{save_path}/{title}.png', bbox_inches='tight')
    plt.close(fig)

def plot_batch_of_exp_result(csv_data, dataset, title, parent_folder):
        # 创建结果DataFrame
        results_df = pd.DataFrame(csv_data)
        
        # 提取学习率和批次大小作为坐标轴
        learning_rates = sorted(results_df['learning_rate'].unique())
        batch_sizes = sorted(results_df['batch_size'].unique())
        
        # 创建热力图数据矩阵
        heatmap_data = np.zeros((len(batch_sizes), len(learning_rates)))
        
        # 填充数据
        for i, bs in enumerate(batch_sizes):
            for j, lr in enumerate(learning_rates):
                mask = (results_df['batch_size'] == bs) & (results_df['learning_rate'] == lr)
                if mask.any():
                    heatmap_data[i, j] = results_df.loc[mask, 'test_acc'].iloc[0]
                else:
                    heatmap_data[i, j] = np.nan
        
        # 使用纯matplotlib绘制热力图
        plt.figure(figsize=(10, 8))
        im = plt.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
        
        # 设置坐标轴
        plt.yticks(range(len(learning_rates)), [f'{lr:.5f}' for lr in learning_rates], rotation=45)
        plt.xticks(range(len(batch_sizes)), batch_sizes)
        plt.ylabel('Learning Rate')
        plt.xlabel('Batch Size')
        plt.title(f'Test Accuracy Heatmap\n{dataset} - {title}')
        
        # 添加颜色条
        cbar = plt.colorbar(im)
        cbar.set_label('Test Accuracy')
        
        # 添加数值标注
        for i in range(len(batch_sizes)):
            for j in range(len(learning_rates)):
                if not np.isnan(heatmap_data[i, j]):
                    plt.text(j, i, f'{heatmap_data[i, j]:.3f}', 
                            ha='center', va='center', fontsize=8,
                            color='white' if heatmap_data[i, j] > np.nanmean(heatmap_data) else 'black')
        
        plt.tight_layout()
        
        # 保存热力图
        heatmap_path = os.path.join(parent_folder, dataset, f"{title}_heatmap.png")
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Heatmap saved to: {heatmap_path}")
