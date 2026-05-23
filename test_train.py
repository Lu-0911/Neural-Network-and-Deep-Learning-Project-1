import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
from draw_tools.plot import plot

# 超参数配置
MODEL_TYPE = 'CNN'             # 'MLP' 或 'CNN'
OPTIMIZER_TYPE = 'SGD'         # 'SGD' 或 'Momentum'
INIT_LR = 0.06                 # 初始学习率，例如: 0.1, 0.01, 0.001
SCHEDULER_TYPE = 'MultiStep'   # 选项: 'None', 'Step', 'MultiStep', 'Exponential'

# 固定随机种子
np.random.seed(309)

# 1. 读取数据集
train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    train_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(train_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    train_labs = np.frombuffer(f.read(), dtype=np.uint8)

# 划分验证集 (10000张)
idx = np.random.permutation(np.arange(num))
# 保存划分的索引
with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# 归一化
train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()

# 初始化模型
if MODEL_TYPE == 'MLP':
    model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 10], 'ReLU', [1e-4, 1e-4])
elif MODEL_TYPE == 'CNN':
    model = nn.models.Model_CNN()

# 初始化优化器
if OPTIMIZER_TYPE == 'SGD':
    optimizer = nn.optimizer.SGD(init_lr=INIT_LR, model=model)
elif OPTIMIZER_TYPE == 'Momentum':
    optimizer = nn.optimizer.MomentGD(init_lr=INIT_LR, model=model, mu=0.9)

# 学习率调度器
if SCHEDULER_TYPE == 'None':
    scheduler = None
elif SCHEDULER_TYPE == 'Step':
    scheduler = nn.lr_scheduler.StepLR(optimizer=optimizer, step_size=800, gamma=0.5)
elif SCHEDULER_TYPE == 'MultiStep':
    scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
elif SCHEDULER_TYPE == 'Exponential':
    scheduler = nn.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9997)

loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=train_labs.max()+1)

# 训练模型
save_name = f"best_model_{MODEL_TYPE}_{OPTIMIZER_TYPE}_LR{INIT_LR}_{SCHEDULER_TYPE}"
runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)
runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=5, log_iters=100, save_name=save_name)

# 保存实验数据
history = {
    'train_loss': runner.train_loss,
    'train_acc': runner.train_scores,
    'dev_loss': runner.dev_loss,
    'dev_acc': runner.dev_scores
}
history_filename = f"History_{MODEL_TYPE}_{OPTIMIZER_TYPE}_LR{INIT_LR}_{SCHEDULER_TYPE}.pickle"
with open(history_filename, 'wb') as f:
    pickle.dump(history, f)

_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)

plt.show()
