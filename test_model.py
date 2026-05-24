import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

model = nn.models.Model_MLP()
model.load_model(r'.\best_model\best_model_MLP_SGD_LR0.06_MultiStep')

test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)

test_imgs = test_imgs / test_imgs.max()

logits = model(test_imgs)
print(nn.metric.accuracy(logits, test_labs))



# import mynn as nn
# import numpy as np
# from struct import unpack
# import gzip
# import matplotlib.pyplot as plt
# import pickle

# model = nn.models.Model_CNN()

# model.load_model(r'.\best_model\best_model_CNN_SGD_LR0.06_MultiStep')

# test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
# test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

# with gzip.open(test_images_path, 'rb') as f:
#         magic, num, rows, cols = unpack('>4I', f.read(16))
#         # 读出来的原始形状是 (10000, 784)
#         test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
# with gzip.open(test_labels_path, 'rb') as f:
#         magic, num = unpack('>2I', f.read(8))
#         test_labs = np.frombuffer(f.read(), dtype=np.uint8)

# test_imgs = test_imgs / test_imgs.max()

# # 将输入 Reshape 适配 CNN
# test_imgs_cnn = test_imgs.reshape(-1, 1, 28, 28)

# logits = model(test_imgs_cnn)
# print(f"Test Accuracy: {nn.metric.accuracy(logits, test_labs) * 100:.2f}%")
