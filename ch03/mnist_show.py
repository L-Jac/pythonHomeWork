# coding: utf-8
import os
import sys
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)  # 5
print(img.shape)  # (784,)
# flatten=True时读入的图像是以一列（一维）NumPy数组的形式保存的。
# 因此，显示图像时，需要把它变为原来的28像素 × 28像素的形状。
# 可以通过reshape()方法的参数指定期望的形状，更改NumPy数组的形状。
img = img.reshape(28, 28)  # 把图像的形状变为原来的尺寸
print(img.shape)  # (28, 28)

img_show(img)
