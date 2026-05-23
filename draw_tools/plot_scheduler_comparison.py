import pickle
import matplotlib.pyplot as plt

# 学习率调度器对比
files = {
    'None': 'History_CNN_SGD_LR0.1_None.pickle',
    'Step': 'History_CNN_SGD_LR0.1_Step.pickle',
    'MultiStep': 'History_CNN_SGD_LR0.06_MultiStep.pickle',
    'Exponential': 'History_CNN_SGD_LR0.1_Exponential.pickle'
}
labels = {'None': 'Constant LR (0.1)', 'Step': 'Step', 'MultiStep': 'MultiStepLR', 'Exponential': 'ExponentialLR'}
colors = {'None': 'gray', 'Step': 'green', 'MultiStep': 'blue', 'Exponential': 'red'}

histories = {}
for sch, filename in files.items():
    try:
        with open(filename, 'rb') as f:
            histories[sch] = pickle.load(f)
    except FileNotFoundError:
        print(f"未找到实验数据文件: {filename}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss 对比
for sch, hist in histories.items():
    axes[0].plot(hist['train_loss'], label=f'{labels[sch]} (Train)', color=colors[sch], alpha=0.3)
    axes[0].plot(hist['dev_loss'], label=f'{labels[sch]} (Dev)', color=colors[sch], linestyle='--')
axes[0].set_title('Loss Curve under Different Schedulers', fontsize=12)
axes[0].set_xlabel('Iterations')
axes[0].set_ylabel('Loss')
axes[0].grid(True, linestyle=':', alpha=0.6)
axes[0].legend()

# Accuracy 对比
for sch, hist in histories.items():
    axes[1].plot(hist['train_acc'], label=f'{labels[sch]} (Train)', color=colors[sch], alpha=0.3)
    axes[1].plot(hist['dev_acc'], label=f'{labels[sch]} (Dev)', color=colors[sch], linestyle='--')
axes[1].set_title('Accuracy Curve under Different Schedulers', fontsize=12)
axes[1].set_xlabel('Iterations')
axes[1].set_ylabel('Accuracy')
axes[1].grid(True, linestyle=':', alpha=0.6)
axes[1].legend()

plt.tight_layout()
plt.savefig('scheduler_comparison.png', dpi=300)
plt.show()