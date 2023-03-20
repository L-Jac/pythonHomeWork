import numpy as np


# TODO 深度学习与python——5 误差反向传播法

# TODO 乘法层的实现 ch05/layer_naive.py, ch05/buy_apple.py
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        return out

    def backward(self, dout):
        dx = dout * self.y  # 翻转x,y
        dy = dout * self.x
        return dx, dy


apple = 100
apple_num = 2
tax = 1.1
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()
# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)
print(price)  # 220
# backward
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
print(dapple, dapple_num, dtax)  # 2.2 110 200


# TODO 加法层的实现 ch05/buy_apple_orange.py
class AddLayer:
    def __init__(self):
        pass  # pass语句表示“什么也不运行

    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy


# TODO ReLU层(激活函数ReLU)

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        # 布尔类型的数组self.mask是一个掩码数组
        out = x.copy()
        # 创建输入数据x的副本,在修改out时不会影响x。
        out[self.mask] = 0
        # 使用掩码数组来索引out，并将选中的元素设为0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


# TODO Sigmoid层
class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx
