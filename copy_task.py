from random import random as r
import numpy as np
from plotting import plotting
from ntm import *
import sys
import pickle as pic
def copy_task():
    n_train = 1000
    for i in range(0,n_train):
        seq_size = 5#int(9* r())+1
        seq = np.random.binomial(1, 0.5, size=(seq_size, y_size-1)).astype(theano.config.floatX)   
        x_ = np.zeros((seq_size*2+1, y_size)).astype(theano.config.floatX)
        x_[:seq_size,:y_size-1] = seq
        x_[seq_size, y_size-1] = 1.0
        y = np.zeros((seq_size*2+1, y_size)).astype(np.int32)
        y[seq_size+1:,:y_size-1] = seq
        cost, ypred = train(x_,y)
        print "cost =", cost 
        # pic.dump(ypred, open("output.txt", "w+"))
        # print f(x_,y)
        # sys.exit()
    n_test = 10   
    p = plotting()
    for i in range(0,n_test):
        seq_size = 5#int(9* r())+1
        seq = np.random.binomial(1, 0.5, size=(seq_size, y_size-1)).astype(theano.config.floatX)   
        x_ = np.zeros((seq_size*2+1, y_size)).astype(theano.config.floatX)
        x_[:seq_size,:y_size-1] = seq
        x_[seq_size, y_size-1] = 1.0
        y = np.zeros((seq_size*2+1, y_size)).astype(np.int32)
        y[seq_size+1:,:y_size-1] = seq
        y_pred= test(x_)
        print "cost =", cost
        p.draw([np.transpose(x_), np.transpose(y), np.transpose(y_pred)])
if __name__ == '__main__':
    print 'runniong copy task'
    copy_task()