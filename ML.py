import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from sklearn import linear_model
from scipy import stats
from statsmodels.tsa.stattools import pacf
import pandas as pd


iv = np.array([[0.2, 0.3], [0.75, 0.25], [0.6, 0.8]])
ov = np.array([[0.2, 0.8], [0.5, 0.5], [0.3, 0.7]])
x = np.array([[0], [1], [0]])
xi = np.array([[0], [0], [1]])
t = np.array([[1], [0], [0]])

for no in range(40):
    if no % 2 == 0:
        helper = 0
        plug = x
    else:
        helper = 1
        plug = xi
    t = np.array([[1], [0], [0]])
    h = np.dot(np.transpose(plug),iv)
    u = np.dot(h, np.transpose(ov))
    scalar = 1/np.sum(u)
    y = u*scalar
    e = np.transpose(y) - t
    update_int = np.dot(np.transpose(h), np.transpose(e))
    ov = ov - 0.1*(np.transpose(update_int))
    in_int = np.dot(np.transpose(e), ov)
    if helper == 0:
        rower = iv[1:2] - in_int
        iv[1:2] = rower
    if helper == 1:
        rower = iv[2:3] - in_int
        iv[2:3] = rower

        XO = [0,0,0,0,0]
        YO = [0,0,0,0,0]

        X = [iv[1:2,0], iv[2:3, 0], ov [0:1, 0], ov [1:2, 0], ov [2:3, 0]]
        Y = [iv[1:2, 1], iv[2:3, 1], ov[0:1, 1], ov[1:2, 1], ov[2:3, 1]]

        fig, ax = plt.subplots(figsize=(12, 7))
        ax.quiver(XO, YO, X, Y, color=['r','b','g', 'black', 'orange'],
                  scale=5)
        ax.axis([-1.5, 1.5, -1.5, 1.5])
        plt.show(block=False)

plt.show()


# class Network(object):
#
#     def __init__(self, sizes):
#         self.num_layers = len(sizes)
#         self.sizes = sizes
#         self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
#         self.weights = [np.random.randn(y, x)
#                         for x, y in zip(sizes[:-1], sizes[1:])]
#     def sigmoid(z):
#         return 1.0/(1.0+np.exp(-z))
#
#
#     def feedforward(self, a):
#         print(self.sizes)
#         print(self.biases)
#         print(self.weights)
#         for b, w in zip(self.biases, self.weights):
#             a = (np.dot(w, a) + b)
#             print(a)
#             print(b)
#
# test = Network([3,2,1])
# test.feedforward(np.array([5,6,7]))

# np.random.seed(1)
# n_samples = int(50)
# a = -0.5
# x = w = np.random.normal(size=n_samples)


# plt.hist(x, bins=100, density=True, alpha=1, color='g')
# plt.show()

# dicto = {}
# d = 0
# for t in range(n_samples):
#     x[t] = a * x[t - 1] + w[t]
#     # x[t] = a * x[t - 1]
#     dicto[d] = x[t]
#     d += 1
#
# plt.bar(*zip(*dicto.items()))
# plt.show()
# _ = tsplot(x, lags=30)


# test1 = [15, 20, 25, 15, 20, 25, 15, 20, 25, 15, 20, 25, 15, 20, 25, 15, 20, 25, 15, 20, 25, 15, 20, 25]
# dta_list = np.arange(24)
# data=pd.DataFrame(test1, index=dta_list)
# data.columns = ['test1']
# data['L1']=data['test1'].shift(1)
# data['L2']=data['test1'].shift(2)
# data['L3']=data['test1'].shift(3)
# data['L4']=data['test1'].shift(4)
# data = data.drop(data.index[[0,1,2,3]])
# lm = linear_model.LinearRegression()
# df_1 = data[['L1']]
# df_2 = data[['L2']]
# df_3 = data[['L3']]
# df_y = data['test1']
# model = lm.fit(df_3,df_y)
# data['PredictedT|L1'] = lm.predict(df_3)
# print(model.score(df_3,df_y))
# print(model.coef_)

# answer = stats.pearsonr(diff[1:], ddiff)
# print(answer)
# print(pacf(test1, nlags=3)[3])

# plt.scatter(x=for2, y=nt2)
# plt.show()