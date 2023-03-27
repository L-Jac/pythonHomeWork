# coding: utf-8
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from common.multi_layer_net import MultiLayerNet
from common.trainer import Trainer
from common.util import shuffle_dataset
from dataset.mnist import load_mnist

sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 为了实现高速化，减少训练数据
x_train = x_train[:500]
t_train = t_train[:500]

# 分割验证数据
validation_rate = 0.20
validation_num = int(x_train.shape[0] * validation_rate)
x_train, t_train = shuffle_dataset(x_train, t_train)
x_val = x_train[:validation_num]
t_val = t_train[:validation_num]
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]


def __train(lr, weight_decay, epocs=50):
    network = MultiLayerNet(input_size=784,
                            hidden_size_list=[100, 100, 100, 100, 100, 100],
                            output_size=10,
                            weight_decay_lambda=weight_decay)
    trainer = Trainer(network, x_train, t_train, x_val, t_val,
                      epochs=epocs,
                      mini_batch_size=100,
                      optimizer='sgd',
                      optimizer_param={'lr': lr},
                      verbose=False)
    trainer.train()

    return trainer.test_acc_list, trainer.train_acc_list


# 超参数的随机搜索======================================
optimization_trial = 100
results_val = {}
results_train = {}
for _ in range(optimization_trial):
    # 指定搜索的超参数的范围===============
    weight_decay = 10 ** np.random.uniform(-8, -4)
    lr = 10 ** np.random.uniform(-6, -2)
    # ================================================

    val_acc_list, train_acc_list = __train(lr, weight_decay)
    print("val acc:" + str(val_acc_list[-1]) + " | lr:" + str(lr) + ", weight decay:" + str(weight_decay))
    # val_acc_list[-1]将返回val_acc_list列表中的最后一个元素。
    key = "lr:" + str(lr) + ", weight decay:" + str(weight_decay)
    results_val[key] = val_acc_list
    results_train[key] = train_acc_list

# 绘制图形========================================================
print("=========== Hyper-Parameter Optimization Result ===========")
graph_draw_num = 20
col_num = 5
row_num = int(np.ceil(graph_draw_num / col_num))
# ceil 函数通常用来对一个数进行向上取整
i = 0

for key, val_acc_list in sorted(results_val.items(), key=lambda x: x[1][-1], reverse=True):
    # sorted 函数对字典 results_val 的键值对进行排序。
    # key=lambda x: x[1][-1] 指定了排序的依据，即字典的值列表的最后一个元素。
    # reverse=True 表示按照降序排序。
    # 按照字典 results_val 的值列表的最后一个元素（即模型在验证数据集上的最终准确率）对字典进行降序排序。
    print("Best-" + str(i + 1) + "(val acc:" + str(val_acc_list[-1]) + ") | " + key)
    # 某次结果：
    # Best-1 (val acc:0.83) | lr:0.0092, weight decay:3.86e-07
    # Best-2 (val acc:0.78) | lr:0.00956, weight decay:6.04e-07
    # Best-3 (val acc:0.77) | lr:0.00571, weight decay:1.27e-06
    # Best-4 (val acc:0.74) | lr:0.00626, weight decay:1.43e-05
    # Best-5 (val acc:0.73) | lr:0.0052, weight decay:8.97e-06
    # 可知 学习率在0.001到0.01,权值衰减系数在10−8到10−6之间时，学习可以顺利进行。
    plt.subplot(row_num, col_num, i + 1)
    plt.title("Best-" + str(i + 1))
    plt.ylim(0.0, 1.0)
    if i % 5:
        plt.yticks([])
    plt.xticks([])
    x = np.arange(len(val_acc_list))
    plt.plot(x, val_acc_list)
    plt.plot(x, results_train[key], "--")
    i += 1

    if i >= graph_draw_num:
        break

plt.show()
