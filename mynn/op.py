from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        # 使用正态分布初始化，乘以 0.01 缩小方差，防止初始梯度爆炸
        self.W = initialize_method(size=(in_dim, out_dim)) * 0.01
        self.b = np.zeros((1, out_dim)) # 偏置初始化为0
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X  # 保存输入，用于反向传播时计算梯度
        output = np.dot(X, self.W) + self.b
        return output

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        # 计算对输入 X 的梯度，传给上一层: dX = grad * W^T
        grad_input = np.dot(grad, self.W.T)
        
        # 计算对权重 W 和偏置 b 的梯度
        # dW = X^T * grad
        self.grads['W'] = np.dot(self.input.T, grad)
        # db = sum(grad, axis=0)
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        
        return grad_input
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None} # 梯度清零

class conv2D(Layer):
    """
    The 2D convolutional layer.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # 初始化参数
        self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size)) * 0.1
        self.b = np.zeros(out_channels)
        
        self.grads = {'W' : None, 'b' : None}
        self.params = {'W' : self.W, 'b' : self.b}
        self.optimizable = True
        
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
        
        self.X_pad = None

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [1, out, in, k, k]
        """
        batch_size, in_c, H, W = X.shape
        out_c, _, k, _ = self.W.shape
        
        H_out = (H + 2 * self.padding - k) // self.stride + 1
        W_out = (W + 2 * self.padding - k) // self.stride + 1
        
        # 填充
        if self.padding > 0:
            self.X_pad = np.pad(X, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')
        else:
            self.X_pad = X.copy()
            
        out = np.zeros((batch_size, out_c, H_out, W_out))
        
        # 把 W 扩展一维 [1, out_c, in_c, k, k]
        W_broad = self.W[np.newaxis, ...]
        
        # 前向传播，计算卷积
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * self.stride
                h_end = h_start + k
                w_start = j * self.stride
                w_end = w_start + k
                
                # 切片并扩展维度: [batch, 1, in_c, k, k]
                X_slice = self.X_pad[:, np.newaxis, :, h_start:h_end, w_start:w_end]
                
                # 元素级相乘求和
                out[:, :, i, j] = np.sum(X_slice * W_broad, axis=(2, 3, 4))
                
        # 加上偏置
        out += self.b.reshape(1, out_c, 1, 1)
        return out

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, H_out, W_out]
        """
        batch_size, out_c, H_out, W_out = grads.shape
        _, _, k, _ = self.W.shape
        
        dX_pad = np.zeros_like(self.X_pad)
        dW = np.zeros_like(self.W)
        db = np.sum(grads, axis=(0, 2, 3))
        
        W_broad = self.W[np.newaxis, ...]

        # 反向传播，计算梯度
        for i in range(H_out):
            for j in range(W_out):
                h_start = i * self.stride
                h_end = h_start + k
                w_start = j * self.stride
                w_end = w_start + k
                
                # g_slice: [batch, out_c, 1, 1, 1]
                g_slice = grads[:, :, i, j].reshape(batch_size, out_c, 1, 1, 1)
                
                # X_slice: [batch, 1, in_c, k, k]
                X_slice = self.X_pad[:, np.newaxis, :, h_start:h_end, w_start:w_end]
                
                # 累加 W 的梯度 [out_c, in_c, k, k]
                dW += np.sum(g_slice * X_slice, axis=0)
                
                # 累加 X 的梯度 [batch, in_c, k, k]
                dX_slice = np.sum(g_slice * W_broad, axis=1)
                dX_pad[:, :, h_start:h_end, w_start:w_end] += dX_slice
                
        self.grads['W'] = dW
        self.grads['b'] = db
        
        # 去除 Padding
        if self.padding > 0:
            dX = dX_pad[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dX = dX_pad
        return dX
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}


class Flatten(Layer):
    """
    Used to flatten the 4D output of CNN into 2D for the Linear layer.
    """
    def __init__(self):
        super().__init__()
        self.input_shape = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1)

    def backward(self, grads):
        # 梯度 reshape 回 4D 张量的形状
        return grads.reshape(self.input_shape)


class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.predicts = None
        self.labels = None

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        """
        if self.has_softmax:
            self.predicts = softmax(predicts)  # softmax 函数已定义
        else:
            self.predicts = predicts
            
        self.labels = labels
        batch_size = predicts.shape[0]
        
        # 提取真实标签对应的预测概率，加上 1e-7 防止 log(0) 导致 NaN
        correct_class_probs = self.predicts[np.arange(batch_size), labels]
        loss = -np.mean(np.log(correct_class_probs + 1e-7))
        
        return loss
    
    def backward(self):
        # first compute the grads from the loss to the input
        batch_size = self.predicts.shape[0]
        
        # 梯度的计算公式: (p - y) / batch_size
        self.grads = self.predicts.copy()
        # 只有在真实标签对应的位置减去 1 (相当于减去 One-hot 编码的 1)
        self.grads[np.arange(batch_size), self.labels] -= 1
        
        # 梯度关于批次大小平均
        self.grads = self.grads / batch_size
        
        # 反向传播到模型
        if self.model is not None:
            self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition