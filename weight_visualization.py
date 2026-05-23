# import mynn as nn
# import numpy as np
# from struct import unpack
# import gzip
# import matplotlib.pyplot as plt
# import pickle

# model = nn.models.Model_MLP()
# model.load_model(r'.\saved_models\best_model_1.pickle')

# test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
# test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

# with gzip.open(test_images_path, 'rb') as f:
#         magic, num, rows, cols = unpack('>4I', f.read(16))
#         test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
# with gzip.open(test_labels_path, 'rb') as f:
#         magic, num = unpack('>2I', f.read(8))
#         test_labs = np.frombuffer(f.read(), dtype=np.uint8)

# test_imgs = test_imgs / test_imgs.max()

# # logits = model(test_imgs)

# mats = []
# mats.append(model.layers[0].params['W'])
# mats.append(model.layers[2].params['W'])

# # _, axes = plt.subplots(30, 20)
# # _.set_tight_layout(1)
# # axes = axes.reshape(-1)
# # for i in range(600):
# #         axes[i].matshow(mats[0].T[i].reshape(28,28))
# #         axes[i].set_xticks([])
# #         axes[i].set_yticks([])

# plt.figure()
# plt.matshow(mats[1])
# plt.xticks([])
# plt.yticks([])
# plt.show()

import mynn as nn
import numpy as np
import matplotlib.pyplot as plt

# 可视化 MLP 的隐藏层权重
try:
    mlp_model = nn.models.Model_MLP()
    # 载入 MLP 权重
    mlp_model.load_model(r'./best_model/best_models_MLP_SGD_LR0.06_MultiStep')
    
    # 提取第一层全连接的权重 W: 形状为 (784, 600)
    W_mlp = mlp_model.layers[0].params['W']
    
    # 可视化前 20 个隐层神经元对应的感受野
    fig, axes = plt.subplots(4, 5, figsize=(10, 8))
    axes = axes.reshape(-1)
    for i in range(20):
        # 转置后取第 i 个隐层神经元，并 reshape 回 28x28 图像
        neuron_weight = W_mlp.T[i].reshape(28, 28)
        axes[i].imshow(neuron_weight, cmap='RdBu', vmin=-0.1, vmax=0.1)
        axes[i].set_title(f'Neuron {i}')
        axes[i].axis('off')
    plt.suptitle("MLP Hidden Layer Weight Visualization (First 20 Neurons)", fontsize=14)
    plt.tight_layout()
    plt.savefig('mlp_weights_visualization.png', dpi=300)
except FileNotFoundError:
    print("未找到 MLP 权重文件，跳过 MLP 权重可视化。")


# 可视化 CNN 的卷积核
try:
    cnn_model = nn.models.Model_CNN()
    # 载入 CNN 权重
    cnn_model.load_model(r'./best_model/best_models_CNN_SGD_LR0.06_MultiStep')
    
    # 提取第一个卷积层的权重 W: 形状为 (8, 1, 5, 5)
    W_conv = cnn_model.layers[0].params['W']
    
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    axes = axes.reshape(-1)
    for i in range(8):
        # 提取第 i 个通道的 5x5 卷积核
        kernel = W_conv[i, 0]
        axes[i].imshow(kernel, cmap='gray')
        axes[i].set_title(f'Kernel {i}')
        axes[i].axis('off')
    plt.suptitle("CNN First Layer Convolutional Kernels Visualization", fontsize=14)
    plt.tight_layout()
    plt.savefig('cnn_kernels_visualization.png', dpi=300)
except FileNotFoundError:
    print("未找到 CNN 权重文件，跳过卷积核可视化。")

plt.show()