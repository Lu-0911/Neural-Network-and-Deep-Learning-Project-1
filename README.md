# Neural Network & Deep Learning - Project 1

**基于 NumPy 实现的多层感知机 (MLP) 与卷积神经网络 (CNN)**

本项目是《神经网络与深度学习》课程的 Project-1，目标是依靠 Python 和科学计算库 NumPy，从底层推导并构建简单的深度学习框架（内置于 `mynn` 模块），并在 MNIST 手写数字数据集上完成图像分类、模型对比、优化器测试及特征可视化任务。

## 项目结构 (Repository Structure)

```text
├── dataset/
│   └── MNIST/                         # MNIST 数据集存放目录
├── mynn/                              # 核心框架包 (自定义的底层算子与模型)
│   ├── __init__.py
│   ├── op.py                          # 核心网络层与损失函数实现 (Linear, Conv2D, ReLU, Loss)
│   ├── models.py                      # 网络架构 (Model_MLP, Model_CNN)
│   ├── optimizer.py                   # 优化算法 (SGD, MomentGD)
│   ├── lr_scheduler.py                # 学习率调度器 (MultiStep, Exponential 等)
│   ├── metric.py                      # 评估指标 (Accuracy)
│   └── runner.py                      # 训练验证循环控制器
├── draw_tools/                        # 可视化代码
│   ├── draw.py
│   ├── plot.py
│   ├── plot_comparison.py             # 绘制多模型收敛曲线对比图脚本
│   ├── plot_lr_comparison.py          # 绘制不同初始学习率对比图脚本
│   └── plot_scheduler_comparison.py   # 绘制不同调度器对比图脚本
├── test_train.py                      # 主训练脚本
├── test_models.py                     # 模型测试脚本
├── error_analysis.py                  # 混淆矩阵与错分样本分析脚本
├── weight_visualization.py            # 权重与卷积核可视化脚本
└── README.md                          # 项目说明文档
```

## 环境依赖 (Requirements)

本项目轻量，使用 Python 3.8+，可直接用 CPU 运行。

```bash
pip install numpy matplotlib
```

## 代码运行 (Running Codes)

### 1. 模型训练与配置
本项目的主训练入口为 `test_train.py`，可在顶部配置模型与超参数

```python
# 超参数配置
MODEL_TYPE = 'CNN'             # 选项: 'MLP' 或 'CNN'
OPTIMIZER_TYPE = 'Momentum'    # 选项: 'SGD' 或 'Momentum'
INIT_LR = 0.06                 # 初始学习率，例如: 0.1, 0.06, 0.01
SCHEDULER_TYPE = 'MultiStep'   # 选项: 'None', 'Step', 'MultiStep', 'Exponential'
```

修改完成后，直接在终端运行：

```bash
python test_train.py
```
训练完成后，代码会自动保存最优模型权重，并生成一个 `.pickle` 文件用于保存 Loss 和 Accuracy 的训练历史。

### 2. 绘制对比曲线
当运行了多组不同的配置实验后，可以运行相应的画图脚本将它们绘制在同一张图表上：

```bash
# 绘制模型架构与优化算法对比图 (Loss & Accuracy)
python plot_comparison.py

# 绘制不同学习率对比图
python plot_lr_comparison.py

# 绘制学习率调度策略对比图
python plot_scheduler_comparison.py
```

### 3. 错误分析

生成测试集的混淆矩阵并查看错分样本，可运行：

```bash
python error_analysis.py
```

### 4. 权重与卷积核可视化

想要深入了解网络底层权重，可运行：

```bash
python weight_visualization.py
```

