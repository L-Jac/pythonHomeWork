import os
import sys

import numpy as np

from common.util import im2col

sys.path.append(os.pardir)


# TODO 深度学习与python——7  卷积神经网络

# 填充，步幅 使用填充主要是为了调整输出的大小。
# 增大步幅后，输出大小会变小。而增大填充后，输出大小会变大。


# TODO 输出大小计算
#  设输入大小为(H, W)，滤波器(卷积核)大小为(FH, FW)，
#  输出大小为(OH, OW)，填充为P，步幅为S。
#  OH = (H + 2P - FH) / S + 1
#  OW = (H + 2P - FW) / S + 1
#  但是所设定的值必须使式可以除尽否则报错

# TODO 池化层
#  处理：
#  一般来说，池化的窗口大小会和步幅设定成相同的值
#  Max池化是获取最大值的运算
#  Average池化则是计算目标区域的平均值
#  在图像识别领域，主要使用Max池化。
#  特征：
#  1没有要学习的参数 池化层和卷积层不同，没有要学习的参数。
#  2通道数不发生变化 经过池化运算，输入数据和输出数据的通道数不会发生变化。
#  3对微小的位置变化具有鲁棒性（健壮） 输入数据发生微小偏差时，池化仍会返回相同的结果。


# TODO 卷积层的实现
class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W  # 卷积核
        self.b = b  # 偏置
        self.stride = stride  # 步长
        self.pad = pad  # 填充

    def forward(self, x):
        # 卷积核
        FN, C, FH, FW = self.W.shape
        # 输入
        N, C, H, W = x.shape
        # 输出大小的计算
        out_h = int(1 + (H + 2 * self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2 * self.pad - FW) / self.stride)
        # 将输入数据展开为2维数组
        col = im2col(x, FH, FW, self.stride, self.pad)
        # 滤波器的展开
        col_W = self.W.reshape(FN, -1).T
        # 设(4,1,3,2) --> (2,-1) => (2,12) 即-1取12
        # 首先，使用reshape方法将W的形状从(FN, C, FH, FW)转换为(FN, -1)，
        # 其中-1表示自动计算该维度的大小。
        # 然后，使用转置操作将其转换为(C-FH-FW, FN)的形状。
        # 这样，每一列就对应一个卷积核,每一行对应一个位置上的权重。
        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        # .transpose(0, 3, 1, 2): (A,B,C,D) => (A,D,B,C)
        # (4,6) --> (2,3,2,-1) => (2,2,3,2) 即-1取2
        # (4,6) --> (4,3,2,-1) => (4,1,3,2) 即-1取1
        # 综上得(A,B) <--> (C,D,E,F) 充分必要条件:A*B = C*D*E*F
        return out


# TODO 池化层的实现
#  书:7.4.4　池化层的实现 P222
class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)
        # 展开(1)
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)
        # 最大值(2)
        out = np.max(col, axis=1)
        # 转换(3)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)
        return out

# TODO 简单CNN的实现
#  ch07/simple_convnet.py
