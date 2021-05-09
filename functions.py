from numpy import *
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

def create_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    data = np.array(df.iloc[:100, [0, 1, -1]])
    for i in range(len(data)):
        if data[i, -1] == 0:
            data[i, -1] = -1
    return data[:, :2], data[:, -1]


# 计算w
def calcWs(alphas, datas, Labels):
    X = mat(datas)
    labelMat = mat(Labels).transpose()
    m, n = shape(X)
    print(m)
    print(n)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * labelMat[i], X[i, :].T)
    return w


# 计算b
def calcb(w, index_list, datas, labels):
    length = len(index_list)
    b = 0
    for i in range(length):
        j = index_list[i]
        b += labels[j] - np.dot( datas[j],w)

    return b / length

# 可视化
def show_figure(train_data, label_data, b, w, index_list):
    # 支持向量
    x_cord_sv0 = []
    y_cord_sv0 = []
    x_cord_sv1 = []
    y_cord_sv1 = []

    # 非支持向量
    x_cord_nsv0 = []
    y_cord_nsv0 = []
    x_cord_nsv1 = []
    y_cord_nsv1 = []
    for i, datas in enumerate(train_data):
        xPt = float(datas[0])
        yPt = float(datas[1])
        label = int(label_data[i])
        if i in index_list:
            if label_data[i] == -1:
                x_cord_sv0.append(xPt)
                y_cord_sv0.append(yPt)
            else:
                x_cord_sv1.append(xPt)
                y_cord_sv1.append(yPt)
        else:
            if label_data[i] == -1:
                x_cord_nsv0.append(xPt)
                y_cord_nsv0.append(yPt)
            else:
                x_cord_nsv1.append(xPt)
                y_cord_nsv1.append(yPt)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title('Support Vectors')
    # * 号表示支持向量
    ax.scatter(x_cord_sv0, y_cord_sv0, marker='*', s=50)
    ax.scatter(x_cord_sv1, y_cord_sv1, marker='*', s=50, c='red')
    ax.scatter(x_cord_nsv0, y_cord_nsv0, marker='s', s=50, c = 'blue')
    ax.scatter(x_cord_nsv1, y_cord_nsv1, marker='o', s=50, c='red')
    w0 = w[0]
    w1 = w[1]
    x = arange(-2.0, 12.0, 0.1)
    y = (-w0 * x - b) / w1
    ax.plot(x, y)
    ax.axis([3, 7.5, 0, 6])
    plt.show()