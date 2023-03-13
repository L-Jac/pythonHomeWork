import numpy as np


# TODO 深度学习与python——2 感知机

# TODO 简单实现
def AND1(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1


print(AND1(0, 0),  # 输出0
      AND1(1, 0),  # 输出0
      AND1(0, 1),  # 输出0
      AND1(1, 1))  # 输出1


# TODO 使用权重和偏置的实现
# 这里把−θ命名为偏置b,w1和w2是控制输入信号的重要性的参数，而偏置是调整神经元被激活
# 的容易程度,若b为−0.1，则只要输入信号的加权总和超过0.1，神经元就会被激活。但是如果b
# 为−20.0，则输入信号的加权总和必须超过20.0，神经元才会被激活。

# 与门
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


# 与非门
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


# 或门
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


# TODO 异或门的实现
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


print(XOR(0, 0),  # 输出0
      XOR(1, 0),  # 输出1
      XOR(0, 1),  # 输出1
      XOR(1, 1))  # 输出0
