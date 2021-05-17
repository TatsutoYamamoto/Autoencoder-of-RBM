# Restricted Boltzmann Machine/Autoencoder

import numpy as np
import pylab as pyl
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1,data_home = ".")

class RBM:

    def __init__(self, n_v, n_h):
        self.w = np.random.randn(n_h, n_v) #可視層と隠れ層の結合定数
        self.b = np.random.randn(n_h, 1)   #隠れ層の磁場
        self.a = np.random.randn(n_v, 1)   #可視層の磁場

    def train(self, V, epsilon, epoch, T):
        for epo in range(epoch):
            for (n, v_0) in enumerate(V): #２次元配列Vにおいて,nは画像データのindex,v_0は画像データ
                v_0 = np.copy(v_0).reshape(-1, 1)#縦ベクトルに変更している(-1は片方の大きさがわからないときに使う,v_0=v_0,reshape()だと、参照がコピーされるため値だけコピーするcopy()を使うみたい？
                p_h_0 = np.copy(self.sigmoid(self.w.dot(v_0) + self.b)) #p(hj=1∣v,θ)

                v, p_h = self.encode_decode(np.copy(v_0), T)
                self.update(v_0, v, p_h_0, p_h, epsilon)

        return (self.w,self.b,self.a)

    def encode_decode(self, v, T):
        for t in range(T):#Tはモンテカルロの最大回数
            # visible
            p_h = self.sigmoid(self.w.dot(v) + self.b) #p(hj=1∣v,θ)
            h = (np.random.rand(n_h, 1) < p_h).astype('float64')#i成分について、乱数<ph_i なら1, 乱数>ph_iなら０を値に持つベクトル→推定した重みから可視層の配位を生成している

            # hidden
            p_v = self.sigmoid(self.w.T.dot(h) + self.a) #p(vi=1∣h,θ)
            v = (np.random.rand(n_v, 1) < p_v).astype('float64')#i成分について、乱数_i<pv_i なら1, 乱数_i>pv_iなら０を値に持つベクトル

        return (v, p_h)

    def update(self, v_0, v, p_h_0, p_h, epsilon):
        self.w += epsilon * (v_0.T * p_h_0 - v.T * p_h)
        self.a += epsilon * (v_0 - v)
        self.b += epsilon * (p_h_0 - p_h)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

# init parameters
N = 10000
size = 28
n_v = size ** 2 #可視層のサイト数
n_h = 32 #隠れ層のサイト数
THRESHOLD = 127

MNIST_all = len(mnist.data)
indices = np.random.permutation(range(MNIST_all))[:N]

V = mnist.data.values[indices]

# binalization
for i in range(N):
    bin_npy = np.zeros(size*size)
    bin_npy[np.where(V[i] > THRESHOLD)] = 1
    V[i] = bin_npy

# train
rbm = RBM(n_v, n_h)
epsilon = 0.1
epoch = 200
T = 5#サンプリングの回数
w,b,a = rbm.train(V, epsilon, epoch, T)

# test
test_pic_zero =  mnist.data.values[0]
test_pic = np.zeros(size*size)
test_pic[np.where(test_pic_zero > THRESHOLD)] = 1
plt.imshow(test_pic.reshape(size, size), cmap='Greys')
plt.show()

lambda_h = b.reshape(-1) + np.dot(w,test_pic)
generated_h_zero = 1.0 / (1.0 + np.exp(-lambda_h))

generated_h = np.zeros(n_h)
generated_h[np.where(generated_h_zero > 0.5)] = 1

lambda_v_gen = a.reshape(-1) + np.dot(w.T,generated_h)

generated_pic = 1.0 / (1.0 + np.exp(-lambda_v_gen))

plt.imshow(generated_pic.reshape(size, size), cmap='Greys')
plt.show()