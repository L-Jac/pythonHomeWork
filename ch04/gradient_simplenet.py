# coding: utf-8
import sys, os

sys.path.append(os.pardir)  # 为了导入父目录中的文件而进行的设定
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet()

f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)

print(dW)
# [[ 0.21924763 0.14356247 -0.36281009]
#  [ 0.32887144 0.2153437 -0.54421514]]
# 观察一下dW的内容，
# 会发现w11的值大约是0.2，这表示如果将w11增加h，那么损失函数的值会增加0.2h。
# 再如w23对应的值大约是−0.5，这表示如果将w23增加h，损失函数的值将减小0.5h。
# 因此，从减小损失函数值的观点来看，w23应向正方向更新，w11应向负方向更新。
# 至于更新的程度，w23比w11的贡献要大。
