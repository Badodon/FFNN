# coding: utf-8
import six 
import sys 
import argparse
import numpy as np
from sklearn.cross_validation import train_test_split

import chainer
import chainer.links as L
from chainer import optimizers, cuda, serializers
import chainer.functions as F

from gensim import corpora, matutils

import util
from perceptron import Perceptron

"""
Movie Review Classification

多層パーセプトロンで映画レビューのポジネガ分類
 - 入力層のユニット数：単語タイプ数
 - 中間層 1000
 - 出力層 2ユニット 

@Badodon                                                                                                                                                                             
"""

#引数の設定
parser = argparse.ArgumentParser()
parser.add_argument('--gpu  '       , dest='gpu'        , type=int, default=0           , help='1: use gpu, 0: use cpu')
parser.add_argument('--epoch'       , dest='epoch'      , type=int, default=10000       , help='number of epochs to learn')
parser.add_argument('--data'        , dest='data'       , type=str, default='input.dat' , help='an input data file')
parser.add_argument('--batchsize'   , dest='batchsize'  , type=int, default='50'        , help='learning minibatch size')

args = parser.parse_args()
n_epoch     = args.epoch        # エポック数(パラメータ更新回数)
batchsize   = args.batchsize    # minibatch size

# Preared dataset
dataset = util.load_data(args.data)

dataset['source'] = dataset['source'].astype(np.float32)   # BOW
dataset['target'] = dataset['target'].astype(np.float32)   # ラベル

x_train, x_test, y_train, y_test = train_test_split(dataset['source'], dataset['target'], test_size=0.10)
N_test = y_test.size        # test data size
N = len(x_train)

in_units = x_train.shape[1]     # 入力層のユニット数
n_units = 1000                  # 隠れ層のユニット数
out_units = 1                   # 出力層のユニット数

model = L.Classifier(Perceptron(in_units, n_units, out_units), lossfun=F.mean_squared_error)
model.compute_accuracy = False #accuracyを計算しない

#GPUを使うかどうか
if args.gpu > 0:
    cuda.check_cuda_available()
    cuda.get_device(args.gpu).use()
    model.to_gpu()
xp = np if args.gpu <= 0 else cuda.cupy #args.gpu <= 0: use cpu, otherwise: use gpu

# Setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model)
for epoch in six.moves.range(1, n_epoch + 1):
                                                                                                                                                                                     
    print 'epoch', epoch

    # training
    perm = np.random.permutation(N)
    sum_train_loss          = 0.0
    sum_train_accccuracy    = 0

    for i in six.moves.range(0, N, batchsize):
    
        # perm を使い x_train, y_trainからデータセットを選択 (毎回対象となるデータは異なる)
        x = chainer.Variable(xp.asarray(x_train[perm[i:i + batchsize]])) #source
        t = chainer.Variable(xp.asarray(y_train[perm[i:i + batchsize]])) #target
    
        model.zerograds()   # 勾配をゼロ初期化
        loss = model(x, t)        # 損失の計算

        sum_train_loss += loss.data * len(t)

        # 最適化を実行
        loss.backward()              # 誤差逆伝播
        optimizer.update()           # 最適化 
    
    
    print('train mean loss={}'.format(sum_train_loss / N))  #平均誤差
    
    # evaluation
    sum_test_loss = 0.0
    sum_test_accuracy = 0.0

    for i in six.moves.range(0, N_test, batchsize):

        # all test data
        x = chainer.Variable(xp.asarray(x_test[i:i + batchsize]))
        t = chainer.Variable(xp.asarray(y_test[i:i + batchsize]))

        loss = model(x, t) # 損失の計算
        #y = model(x) # 損失の計算
        #loss = F.mean_squared_error(y, t)
        sum_test_loss += loss.data * len(t)
        #sum_test_accuracy += model.accuracy.data * len(t)

    print(' test mean loss={}'.format(sum_test_loss / N_test)) #平均誤差

    sys.stdout.flush()

#modelとoptimizerを保存
print 'save the model'
serializers.save_npz('sc_cnn.model', model)
print 'save the optimizer'
serializers.save_npz('sc_cnn.state', optimizer)
