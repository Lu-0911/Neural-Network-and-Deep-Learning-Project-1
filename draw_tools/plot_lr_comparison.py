import pickle
import matplotlib.pyplot as plt

# 加载三个不同学习率的历史记录
lrs = [0.1, 0.01, 0.001]
colors = {0.1: 'red', 0.01: 'blue', 0.001: 'green'}
histories = {}

for lr in lrs:
    filename = f"History_CNN_SGD_LR{lr}_None.pickle"
    try:
        with open(filename, 'rb') as f:
            histories[lr] = pickle.load(f)
    except FileNotFoundError:
        print(f"未找到实验数据文件: {filename}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss 对比
for lr, hist in histories.items():
    axes[0].plot(hist['train_loss'], label=f'LR={lr} (Train)', color=colors[lr], alpha=0.3)
    axes[0].plot(hist['dev_loss'], label=f'LR={lr} (Dev)', color=colors[lr], linestyle='--')
axes[0].set_title('Loss Comparison under Different Learning Rates', fontsize=12)
axes[0].set_xlabel('Iterations')
axes[0].set_ylabel('Loss')
axes[0].grid(True, linestyle=':', alpha=0.6)
axes[0].legend()

# Accuracy 对比
for lr, hist in histories.items():
    axes[1].plot(hist['train_acc'], label=f'LR={lr} (Train)', color=colors[lr], alpha=0.3)
    axes[1].plot(hist['dev_acc'], label=f'LR={lr} (Dev)', color=colors[lr], linestyle='--')
axes[1].set_title('Accuracy Comparison under Different Learning Rates', fontsize=12)
axes[1].set_xlabel('Iterations')
axes[1].set_ylabel('Accuracy')
axes[1].grid(True, linestyle=':', alpha=0.6)
axes[1].legend()

plt.tight_layout()
plt.savefig('lr_comparison.png', dpi=300)
plt.show()