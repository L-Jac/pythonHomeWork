import numpy as np
import matplotlib.pylab as plt


# TODO 深度学习与python——3 神经网络
# 一般而言，神经网络只把输出值最大的神经元所对应的类别作为识别结果。
# 简单阶跃函数
def normal_step_function(x1):
    if x1 > 0:
        return 1
    else:
        return 0


# TODO 支持numpy的简单阶跃函数
def step_function(x):
    return np.array(x > 0, dtype=np.int_)  # dtype() 返回数组中元素的数据类型


a = np.array([-1.0, 1.0, 2.0])
b = a > 0
print(b.astype(np.int_))  # astype() 对数据类型进行转换


# TODO　阶跃函数的图形
# x = np.arange(-5.0, 5.0, 0.1)  # 在−5.0到5.0内，以0.1为单位，生成NumPy数组
# y = step_function(x)
# plt.plot(x, y)
# plt.ylim(-0.1, 1.1)  # 指定y轴的范围
# plt.show()


# TODO sigmoid函数的实现
def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # exp(−x)表示e的-x次方
    # NumPy 的广播功能


z = np.array([-1.0, 1.0, 2.0])
print(sigmoid(z))


# sigmoid图像
n = np.arange(-5.0, 5.0, 0.1)
plt.plot(n, sigmoid(n))
plt.ylim(-0.1, 1.1)  # 指定y轴的范围
plt.show()

# TODO sigmoid与阶跃对比
# x = np.arange(-6, 6, 0.1)  # 以0.1为单位，生成0到6的数据
# y1 = sigmoid(x)
# y2 = step_function(x)
#
# # 绘制图形
# plt.plot(x, y1, label="sigmoid()")
# plt.plot(x, y2, linestyle=":", label="step_function()")
# plt.title('sigmoid与阶跃对比')  # 标题
# plt.legend()
# plt.show()

# TODO ReLU函数
def relu(x):
    return np.maximum(0, x)


# 图像
# n = np.arange(-5.0, 5.0, 0.1)
# plt.plot(n, relu(n))
# plt.ylim(-1, 6)  # 指定y轴的范围
# plt.show()

# TODO 多维数组的运算

A = np.array([[[1, 3], [2, 3]],
              [[1, 1], [2, 1]],
              [[3, 6], [3, 7]]])
print(np.ndim(A))  # np.ndim()获取数组维度
print(A.shape)  # shape数组形状
B = np.array([[1, 2], [3, 4]])
C = np.array([[5, 6], [7, 8]])
print(np.dot(C, B))  # dot()矩阵点乘
print(np.dot(A, B))  # TODO 异维数组相乘


# TODO 3层神经网络的实现
def identity_function(x):
    return x


# 第0层=>第1层(输入层=>隐藏层
X1 = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])
A1 = np.dot(X1, W1) + B1
Z1 = sigmoid(A1)
# 第1层=>第2层
W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])
A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)
# 第2层=>第三层(隐藏层=>输出层
W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])
A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)  # 或者Y = A3
print(Y)


# 优化
# def init_network():
#     network = {}
#     network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
#     network['b1'] = np.array([0.1, 0.2, 0.3])
#     network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
#     network['b2'] = np.array([0.1, 0.2])
#     network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
#     network['b3'] = np.array([0.1, 0.2])
#     return network
# def forward(network, x):
#     W1, W2, W3 = network['W1'], network['W2'], network['W3']
#     b1, b2, b3 = network['b1'], network['b2'], network['b3']
#     a1 = np.dot(x, W1) + b1
#     z1 = sigmoid(a1)
#     a2 = np.dot(z1, W2) + b2
#     z2 = sigmoid(a2)
#     a3 = np.dot(z2, W3) + b3
#     y = identity_function(a3)
#     return y
# network = init_network()
# x = np.array([1.0, 0.5])
# y = forward(network, x)
# print(y)  # [ 0.31682708 0.69627909]

# TODO 输出层的设计
# 一般而言，回归问题用恒等函数，分类问题用softmax函数

# TODO　恒等函数
# identity_function 会将输入按原样输出

# TODO softmax函数
def softmax(index):
    return np.exp(index) / np.sum(np.exp(index))


print(softmax(A3))


# 优化 softmax函数
def softmax_better(index):
    # 防止溢出 保证正确运算
    exp_ai = np.exp(index - np.max(index))
    exp_ai_sum = np.sum(np.exp(index - np.max(index)))
    return exp_ai / exp_ai_sum


print(softmax_better(A3))

# TODO MNIST数据集 test1.py
# TODO 神经网络的推理处理  ch03/neuralnet_mnist.py
