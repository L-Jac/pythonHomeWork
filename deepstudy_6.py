import numpy as np


# TODO 深度学习与python——6  与学习相关的技巧

# TODO momentun优化
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
                # 创建一个与params值形状和类型相同的零数组作为新字典的值。
                # 例子
                # params = {'w1': np.array([[1, 2], [3, 4]]), 'b1': np.array([1])}
                # v = {}
                # for key, val in params.items():
                #     v[key] = np.zeros_like(val)
                # print(v)
                # {'w1': array([[0, 0],[0, 0]]),
                #  'b1': array([0])}
        for key in params.keys():
            # 修正权重
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


# TODO AdaGrad优化
class AdaGrad:
    """AdaGrad"""

    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            # 修正权重
            self.h[key] += grads[key] * grads[key]
            # 变量h它保存了以前的所有梯度值的平方和
            # 在更新参数时，通过乘以1/√h就可以调整学习的尺度。
            # 这意味着,参数的元素中变动较大（被大幅更新）的元素的学习率将变小。
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
            # 最后一行加上了微小值1e-7。
            # 防止当 self.h[key]中有0时，将0用作除数的情况。

# TODO Adam优化
#  结合Momentum和AdaGrad的方法
#  common/optimizer.py中将其实现为了Adam类


# TODO Xavier初始值：
#   与前一层有n个节点连接时，初始值(初始权重)使用标准差为1/√n的分布
#   这是以线性函数为前提而推导出来的。
#   因为sigmoid函数和tanh函数左右对称，且中央附近可以视作线性函数，
#   所以适合使用Xavier初始值。

# TODO ReLU的权重初始值：
#   ReLU专用的初始值，也也称为“He初始值”。
#   当前一层的节点数为n时，He初始值使用标准差为√(2/n)的高斯分布。

# TODO 基于MNIST数据集的权重初始值的比较 ch06/weight_init_compare.py

# TODO Batch Norm
#  Batch Norm的思路是调整各层的激活值分布使其拥有适当的广度
#  common/multi_layer_net_extend.py
#  ch06/batch_norm_test.py

# TODO 正则化
#  过拟合后使用权值衰减抑制 ch06/overfit_weight_decay.py
#  Dropout Dropout是一种在学习的过程中随机删除神经元的方法（抑制过拟合）
#  ch06/overfit_dropout.py

# TODO 超参数的验证
#  不能使用测试数据评估超参数的性能，否则超参数的值会对测试数据发生过拟合
#  用于调整超参数的数据，一般称为验证数据（validation data）。
#  ch06/hyperparameter_optimization.py

# TODO Dropout
#  Dropout是一种在学习的过程中随机删除神经元的方法
#  通过使用Dropout，即便是表现力强的网络，也可以抑制过拟合。
#  ch06/overfit_dropout.py
