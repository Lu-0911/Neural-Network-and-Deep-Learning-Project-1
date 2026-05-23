# plot_comparison.py
import pickle
import matplotlib.pyplot as plt

try:
    with open('History_MLP_SGD_LR0.06_MultiStep.pickle', 'rb') as f:
        mlp = pickle.load(f)
except FileNotFoundError:
    mlp = None

try:
    with open('History_CNN_SGD_LR0.06_MultiStep.pickle', 'rb') as f:
        cnn = pickle.load(f)
except FileNotFoundError:
    cnn = None

try:
    with open('History_CNN_Momentum_LR0.06_MultiStep.pickle', 'rb') as f:
        cnn_mom = pickle.load(f)
except FileNotFoundError:
    cnn_mom = None

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss 对比
if mlp:
    axes[0].plot(mlp['train_loss'], label='MLP (SGD) Train', color='blue', alpha=0.3)
    axes[0].plot(mlp['dev_loss'], label='MLP (SGD) Dev', color='blue', linestyle='--')
if cnn:
    axes[0].plot(cnn['train_loss'], label='CNN (SGD) Train', color='red', alpha=0.3)
    axes[0].plot(cnn['dev_loss'], label='CNN (SGD) Dev', color='red', linestyle='--')
if cnn_mom:
    axes[0].plot(cnn_mom['train_loss'], label='CNN (Momentum) Train', color='green', alpha=0.3)
    axes[0].plot(cnn_mom['dev_loss'], label='CNN (Momentum) Dev', color='green', linestyle='--')

axes[0].set_title('Loss Convergence Curve Comparison', fontsize=12)
axes[0].set_xlabel('Iterations')
axes[0].set_ylabel('Loss')
axes[0].grid(True, linestyle=':', alpha=0.6)
axes[0].legend()

# Accuracy 对比
if mlp:
    axes[1].plot(mlp['train_acc'], label='MLP (SGD) Train', color='blue', alpha=0.3)
    axes[1].plot(mlp['dev_acc'], label='MLP (SGD) Dev', color='blue', linestyle='--')
if cnn:
    axes[1].plot(cnn['train_acc'], label='CNN (SGD) Train', color='red', alpha=0.3)
    axes[1].plot(cnn['dev_acc'], label='CNN (SGD) Dev', color='red', linestyle='--')
if cnn_mom:
    axes[1].plot(cnn_mom['train_acc'], label='CNN (Momentum) Train', color='green', alpha=0.3)
    axes[1].plot(cnn_mom['dev_acc'], label='CNN (Momentum) Dev', color='green', linestyle='--')

axes[1].set_title('Accuracy Curve Comparison', fontsize=12)
axes[1].set_xlabel('Iterations')
axes[1].set_ylabel('Accuracy')
axes[1].grid(True, linestyle=':', alpha=0.6)
axes[1].legend()

plt.tight_layout()
plt.savefig('experiment_comparison.png', dpi=300)
plt.show()