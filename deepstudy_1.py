import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

# TODO 深度学习与python——1， numpy,Matplotlib
x = np.array([1, 5, 3])
print(x)
print(type(x))
y = np.array([2, 4, 6])
print(x / y)
print(x * y)  # 对应元素的乘法
print(x / 2)
# NumPy的N维数
A = np.array([[1, 2, 4], [2, 5, 3], [2, 2, 4]])
print(A)
print(A.shape)
print(A.dtype)
B = np.array([[3, 0, 3], [0, 6, 2], [1, 3, 2]])
print(A * B)
# numpy的广播
C = np.array([10, 20, 30])
D = np.array([[10], [20], [0]])
print(A * C)
print(A * D)
# 对元素的访问
print(A[1])
print(A[0][1])
for i in A:
    print(i)
print(A.flatten())  # 把A转为一维数组
print(A.flatten()[x])  # 获取索引x(x = np.array([1, 5, 3]))的元素
N = A.flatten()
print(N > 2)  # 判断并输出布尔数组
print(N[N > 2])  # 使用这个布尔型数组取出了数组的各个元素（取出True对应的元素）

# TODO Matplotlib绘制简单图形
# import numpy as np
# import matplotlib.pyplot as plt
# https://www.runoob.com/w3cnote/matplotlib-tutorial.html

# 生成数据
x = np.arange(0, 6, 0.1)  # 以0.1为单位，生成0到6的数据
y1 = np.sin(x)
y2 = np.cos(x)

# 绘制图形
plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle=":", label="cos")
plt.xlabel("x")  # x轴标签
plt.ylabel("y")  # y轴标签
plt.title('sin & cos')  # 标题
plt.legend()
plt.show()

# 显示图像
# import matplotlib.pyplot as plt
# from matplotlib.image import imread
img = imread('D:/pythonworkspace/image/myQR.png')  # 读入图像（设定合适的路径！）
plt.imshow(img)
plt.show()
