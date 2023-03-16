import pickle
from common.functions import sigmoid, softmax
import numpy as np
import sys
import os
from dataset.mnist import load_mnist

sys.path.append(os.pardir)


# MNIST数据集导入处理12行-33行

def init_network():
    with open("ch03/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


network = init_network()


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


# 将当前文件所在目录的父目录添加到 sys.path

# TODO 深度学习与python——4 神经网络的学习

# TODO 损失函数
# 1.均方误差
def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


# 设“2”为正确解
t1 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# 将正确解标签表示为1，其他标签表示为0的表示方法称为one-hot表示。
# 例1：“2”的概率最高的情况（0.6）
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
# 例2：“7”的概率最高的情况（0.6）
y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(mean_squared_error(np.array(y1), np.array(t1)))  # 0.09750000000000003
print(mean_squared_error(np.array(y2), np.array(t1)))  # 0.5975


# 这里举了两个例子。第一个例子中，正确解是“2”，神经网络的输出的最大值是“2”；
# 第二个例子中，正确解是“2”，神经网络的输出的最大值是“7”。
# 如实验结果所示，我们发现第一个例子的损失函数的值更小，和监督数据之间的误差较小。
# 也就是说，均方误差显示第一个例子的输出结果与监督数据更加吻合。

# 2.交叉熵误差
def cross_entropy_error1(y, t):
    delta = 1e-7
    # 是一个科学计数法，表示 1 * 10^-7，也就是 0.0000001。
    # 这个值被加到 y 上是为了防止当 y 的值为 0 时，对其取对数会出现无穷大的情况。
    return -np.sum(t * np.log(y + delta))


print(cross_entropy_error1(np.array(y1), np.array(t1)))  # 0.51
print(cross_entropy_error1(np.array(y2), np.array(t1)))  # 2.3
# 正确解标签对应的输出越大，交叉熵误差越接近0，正确率越高
# 第一个例子中，正确解标签对应的输出为0.6，此时的交叉熵误差大约为0.51。
# 第二个例子中，正确解标签对应的输出为0.1的低值，此时的交叉熵误差大约为2.3。

# TODO mini-batch学习
(x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)
# load_mnist函数是用于读入MNIST数据集的函数
# 设定参数one_hot_label=True，可以得到one-hot表示
print(x_train.shape)  # (60000, 784)
print(t_train.shape)  # (60000, 10)
train_size = x_train.shape[0]  # 60000
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
# np.random.choice(60000, 10)会从0到59999之间随机选择10个数字
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]


# mini-batch版交叉熵误差函数
def cross_entropy_error2(y, t):
    # y是神经网络的输出，t是监督数据。
    # y的维度为1时，即求单个数据的交叉熵误差时，需要改变数据的形状。
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_Size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7), axis=1) / batch_Size
    # axis=1参数告诉np.sum函数沿着数组的第二个维度（即列）进行求和。
    # t * np.log(y + 1e-7)是一个二维数组，其中每一行表示一个样本的交叉熵误差。
    # 当我们使用np.sum(t * np.log(y + 1e-7), axis=1)时，
    # 它会对每一行进行求和(因为t是one-hot形式，非正确解为零不影响结果)，
    # 得到一个一维数组，其中每个元素表示一个样本的交叉熵误差之和。


y = predict(network, x_batch)
print(cross_entropy_error2(y, t_batch))


# 当监督数据是标签形式
# （非one-hot表示，而是像“2”“7”这样的标签）时，交叉熵误差可通过如下代码实现。
def cross_entropy_error3(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_Size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) \
        / batch_Size
    # y[np.arange(batch_size), t]
    # 对于每一行（即每一个样本），从 y 中选择第 t 列的元素。
    # 也就是说，对于每一个样本，我们都选择其对应标签的预测概率。
