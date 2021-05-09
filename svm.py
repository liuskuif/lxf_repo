import numpy as np
from numpy import *
import functions as fc


# SMO算法
class SVM:
    #     定义最大迭代次数，核函数
    def __init__(self, max_iter, kernel='linear', C = 1):
        self.max_iter = max_iter
        self._kernel = kernel
        self.C = C  # 惩罚参数

    #     m样本量，n维度，X样本， Y样本类别，b,alpha拉格朗日乘子,E,C
    def init_args(self, features, labels):
        self.m, self.n = features.shape
        self.X = features
        self.Y = labels
        self.b = 0.0
        self.alpha = np.zeros(self.m)

        # Ei是g(x)预测值-实际值,保存至列表
        # self.E = [self._E(i) for i in range(self.m)]
        self.E = - self.Y # 将拉格朗日乘子全部初始化为0，则相应的预测值初始化为0，预测误差就是-Y

    # 核函数
    def kernel(self, x1, x2):
        if self._kernel == 'linear':  # 线性分类器 k(x,y)=x*y
            return sum([x1[k] * x2[k] for k in range(self.n)])
        elif self._kernel == 'poly':
            return (sum([x1[k] * x2[k] for k in range(self.n)]) + 1) ** 2  # d阶多项式分类器 k(x,y)={(x*y)+1}d
        return 0

    # KKT条件
    def _KKT(self, i):
        y_g = self._g(i) * self.Y[i]
        if self.alpha[i] == 0:
            return y_g >= 1
        elif 0 < self.alpha[i] < self.C:
            return y_g == 1
        else:
            return y_g <= 1

    # g(x)预测值，输入（X[i]）
    def _g(self, i):
        r = self.b
        for j in range(self.m):
            r += self.alpha[j] * self.Y[j] * self.kernel(self.X[i], self.X[j])  # p145 公式7.105   7.117
        return r

    # E（x）为g(x)对输入x的预测值和实际值y的差
    def _E(self, i):
        return self._g(i) - self.Y[i]

    def _init_alpha(self):
        # 外层循环首先遍历所有满足0<a<C的样本点，检验是否满足KKT
        index_list = [i for i in range(self.m) if 0 < self.alpha[i] < self.C]
        # 否则遍历整个训练集
        non_satisfy_list = [i for i in range(self.m) if i not in index_list]
        index_list.extend(non_satisfy_list)
        for i in index_list:
            if self._KKT(i):
                continue
            E1 = self.E[i]
            # 如果E1是+，选择最小的；如果E1是负的，选择最大的
            if E1 >= 0:
                j = min(range(self.m), key=lambda x: self.E[x])
            else:
                j = max(range(self.m), key=lambda x: self.E[x])
            return i, j
        return -1,-1

    def _compare(self, _alpha, L, H):
        if _alpha > H:
            return H
        elif _alpha < L:
            return L
        else:
            return _alpha

    def fit(self, features, labels):
        self.init_args(features, labels)
        for t in range(self.max_iter):  # 迭代
            # train   变量的选择 i1 i2
            # print(self.alpha)
            # print(self.E)
            i1, i2 = self._init_alpha()
            if i1 == -1:
                break
            # print(i1)
            # print(i2)
            # 边界
            if self.Y[i1] == self.Y[i2]:
                L = max(0, self.alpha[i1] + self.alpha[i2] - self.C)
                H = min(self.C, self.alpha[i1] + self.alpha[i2])
            else:
                L = max(0, self.alpha[i2] - self.alpha[i1])
                H = min(self.C, self.C + self.alpha[i2] - self.alpha[i1])

            E1 = self.E[i1]
            E2 = self.E[i2]
            # eta = k11+k22-2k12
            eta = self.kernel(self.X[i1], self.X[i1]) + self.kernel(self.X[i2], self.X[i2]) - 2 * self.kernel(
                self.X[i1], self.X[i2])
            if eta <= 0:
                continue

            # 求alpha2剪辑前的解
            alpha2_new_unc = self.alpha[i2] + self.Y[i2] * (E1 - E2) / eta
            # 求alpha2 剪辑后的解
            alpha2_new = self._compare(alpha2_new_unc, L, H)
            # 更新alpha1
            alpha1_new = self.alpha[i1] + self.Y[i1] * self.Y[i2] * (self.alpha[i2] - alpha2_new)

            # 更新b1
            b1_new = -E1 - self.Y[i1] * self.kernel(self.X[i1], self.X[i1]) * (
                    alpha1_new - self.alpha[i1]) - self.Y[i2] * self.kernel(
                self.X[i2], self.X[i1]) * (alpha2_new - self.alpha[i2]) + self.b

            # 更新b2
            b2_new = -E2 - self.Y[i1] * self.kernel(self.X[i1], self.X[i2]) * (
                    alpha1_new - self.alpha[i1]) - self.Y[i2] * self.kernel(
                self.X[i2], self.X[i2]) * (alpha2_new - self.alpha[i2]) + self.b

            if 0 < alpha1_new < self.C:
                b_new = b1_new
            elif 0 < alpha2_new < self.C:
                b_new = b2_new
            else:
                # 选择中点
                b_new = (b1_new + b2_new) / 2

            # 更新参数
            self.alpha[i1] = alpha1_new
            self.alpha[i2] = alpha2_new
            self.b = b_new

            self.E[i1] = self._E(i1)
            self.E[i2] = self._E(i2)


    def predict(self, data, b):
        # g(xi)
        r = b.copy()
        # print(r)
        for i in range(self.m):
            r += self.alpha[i] * self.Y[i] * self.kernel(data, self.X[i])
        return 1 if r > 0 else -1

    def score(self, X, y, b):
        right_count = 0
        for i in range(len(X)):
            result = self.predict(X[i], b)
            if result == y[i]:
                right_count += 1
        return right_count / len(X)

