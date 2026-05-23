from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] -= self.init_lr * layer.grads[key] # 原地减法更新参数



class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu=0.9):
        super().__init__(init_lr, model)
        self.mu = mu
        self.v = {}  # 记录每个参数的历史速度
        
        # 初始化速度字典 v 为全 0 矩阵
        for i, layer in enumerate(self.model.layers):
            if layer.optimizable:
                self.v[i] = {}
                for key in layer.params.keys():
                    self.v[i][key] = np.zeros_like(layer.params[key])
    
    def step(self):
        for i, layer in enumerate(self.model.layers):
            if layer.optimizable:
                for key in layer.params.keys():
                    # 处理权重衰减 (L2正则化)
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    
                    # 动量公式: v_new = mu * v_old + grad
                    self.v[i][key] = self.mu * self.v[i][key] + layer.grads[key]
                    
                    # 参数更新: W_new = W_old - lr * v_new
                    layer.params[key] -= self.init_lr * self.v[i][key] # 原地减法更新参数