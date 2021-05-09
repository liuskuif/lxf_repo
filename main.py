import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from svm import SVM
import functions as fc

X, y = fc.create_data()

S = SVM(max_iter=200, kernel='linear', C = 0.4)
S.fit(X, y)


index_list = [] # 支持向量标号
for i in range(len(X)):
    if S.alpha[i] > 0:
        print("支持向量为:", X[i],y[i])
        index_list.append(i)
w = fc.calcWs(S.alpha, X, y)
# print(S.alpha)
print("w: {}".format(w))
b = fc.calcb(w, index_list, X, y)
print("b: {}".format(b))
# print(S.E)

fc.show_figure(X, y, b, w, index_list)
score = S.score(X, y, b)
print("score = {}".format(score))
