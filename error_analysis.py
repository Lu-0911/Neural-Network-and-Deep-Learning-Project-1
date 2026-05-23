import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt

# 加载测试集
test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
    magic, num, rows, cols = unpack('>4I', f.read(16))
    test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28, 28) # 转为 28x28
    
with gzip.open(test_labels_path, 'rb') as f:
    magic, num = unpack('>2I', f.read(8))
    test_labs = np.frombuffer(f.read(), dtype=np.uint8)

# 归一化并调整形状
test_imgs_norm = test_imgs / test_imgs.max()
test_imgs_input = test_imgs_norm.reshape(-1, 1, 28, 28)

# 加载训练好的 CNN 模型
model = nn.models.Model_CNN()

model.load_model(r'.\best_model\best_models_CNN_SGD_LR0.06_MultiStep')

# 进行预测评估
logits = model(test_imgs_input)
preds = np.argmax(logits, axis=-1)

accuracy = np.mean(preds == test_labs)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# 绘制混淆矩阵 (Confusion Matrix)
cm = np.zeros((10, 10), dtype=int)
for p, t in zip(preds, test_labs):
    cm[t, p] += 1

plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(10)
plt.xticks(tick_marks, tick_marks)
plt.yticks(tick_marks, tick_marks)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

for i in range(10):
    for j in range(10):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > cm.max() / 2. else "black")
plt.tight_layout()
plt.savefig('confusion_matrix.png')

# 找出预测错误的索引
errors = np.where(preds != test_labs)[0]

# 可视化错分样本
plt.figure(figsize=(10, 5))
for i in range(10):
    if i >= len(errors): break
    idx = errors[i]
    plt.subplot(2, 5, i+1)
    plt.imshow(test_imgs[idx], cmap='gray')
    plt.title(f"True: {test_labs[idx]}\nPred: {preds[idx]}", color='red')
    plt.axis('off')
plt.tight_layout()
plt.savefig('misclassified.png')
plt.show()

