# coding: utf-8
import os
import sys
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax

sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


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


x, t = get_data()
network = init_network()
accuracy_cnt1 = 0
accuracy_cnt2 = 0

# TODO 普通推理处理
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)  # 获取概率最高的元素的索引
    if p == t[i]:
        accuracy_cnt1 += 1
    # accuracy_cnt1 += (p == t[i])

# TODO 批处理
batch_size = 100  # 批数量
for i in range(0, len(x), batch_size):
    x_batch = x[i: i + batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)  # y_batch二维数组,axis=1/0按行/列取最大
    accuracy_cnt2 += np.sum(p == t[i:i + batch_size])


print("Accuracy:" + str(float(accuracy_cnt1) / len(x)))
print("Accuracy:" + str(float(accuracy_cnt2) / len(x)))
print(x.shape)  # 输入的数据的形状
print(x[0].shape)
W1, W2, W3 = network['W1'], network['W2'], network['W3']
print(W1.shape, W2.shape, W3.shape)  # 各个权重的形状
