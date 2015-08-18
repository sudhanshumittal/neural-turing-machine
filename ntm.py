#neural turing machine by deep mind
import os
os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=cpu,floatX=float32,exception_verbosity=high'
import theano
from theano import tensor as T
import numpy as np
from scipy.linalg import circulant
#normalize each row
def vector_softmax(w):
    return w / T.sum(w)
    # e = T.exp(w)
    # dist = e / T.sum(e)
    # return dist


M = 20
N = 128
nHeads = 1
shiftwidth = 3
y_size = 8
insize, hsize, outsize = y_size+M, 200,  y_size+nHeads*(3*M+3+shiftwidth)

mem0 = theano.shared(2.0*(np.random.rand(N,M).astype(theano.config.floatX))- 0.5)
w_init = theano.shared(np.random.rand(N,).astype(theano.config.floatX))
w0 =  T.nnet.softmax(w_init)[0]
#controller params
w_xh = theano.shared(np.random.uniform(-1,1, (insize, hsize)).astype(theano.config.floatX))
bh = theano.shared(np.random.rand(hsize,).astype(theano.config.floatX))
w_ho = theano.shared(np.random.uniform(-1,1, (hsize, outsize)).astype(theano.config.floatX))
bo = theano.shared(np.random.rand(outsize,).astype(theano.config.floatX))

params = [mem0, w_init, w_xh, bh, w_ho, bo]
x = T.matrix('x')
y = T.imatrix('y')


def read(mem, w):
    return T.dot(w, mem)


def write(mem, w, e, a):
    w = w.dimshuffle(0, 'x')
    e = e.dimshuffle('x',0)
    a = a.dimshuffle('x',0)
    return mem*(1-T.dot(w,e)) + T.dot(w, a) 

def get_head_params(h):
    #define w_ho, bo
    raw_outputs = T.dot(h, w_ho) + bo
    output_raw = raw_outputs[:y_size]
    beta_raw =raw_outputs[y_size]
    gate_raw  =raw_outputs[y_size+1]
    gamma_raw = raw_outputs[y_size+2]
    key = raw_outputs[y_size+3:y_size+3+M]
    erase_raw =raw_outputs[y_size+3+M:y_size+3+M*2]
    add = raw_outputs[y_size+3+M*2:y_size+3+M*3]
    shift_raw = raw_outputs[y_size+3+M*3:y_size+3+M*3+shiftwidth]
    output = T.nnet.sigmoid(output_raw)
    
    shift = T.nnet.softmax(shift_raw)[0]#reshape((1,shift_raw.shape[0])))[0] 
    erase = T.nnet.sigmoid(erase_raw)
    beta = T.nnet.softplus(beta_raw)#T.dot(x, w_beta) + b_beta)
    gate = T.nnet.sigmoid(gate_raw)#T.dot(x, w_gate) + b_gate)
    gamma = T.nnet.softplus(gamma_raw)+1#T.dot(x, w_gamma) + b_gamma )+1
    return key, shift, erase, add, beta, gate, gamma, output


def controller(x, r):
    inlayer = T.concatenate([x, r])
    hlayer = T.nnet.sigmoid(T.dot(inlayer, w_xh)+bh)
    return get_head_params(hlayer)

def shift_convolve(weight,shift):
    shift = shift.dimshuffle((0,'x'))
    shift_conv = circulant(np.arange(N)).T[np.arange(-(shiftwidth//2),(shiftwidth//2)+1)][::-1]
    return T.sum(shift * weight[shift_conv],axis=0)

def step(x_t, w_tm1, mem_tm1):
    #read
    r_t = read(mem_tm1, w_tm1)
    #calculate head parameters 
    k_t, s_t, e_t, a_t, beta_t, g_t, gamma_t, y_t = controller(x_t, r_t)
    wc_t = T.nnet.softmax(beta_t*T.dot(mem_tm1,k_t))[0]
    wg_t = g_t*wc_t + (1.0-g_t)* w_tm1
    wshift_t = shift_convolve(wg_t, s_t)
    w_t = wshift_t ** gamma_t
    w_t = w_t /T.sum(w_t)
    #write to memory
    mem_t = write(mem_tm1, w_t, e_t, a_t)
    return [w_t, mem_t, y_t, beta_t, s_t, gamma_t, g_t, a_t]


def RMSprop(cost, params, lr=0.01, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

[w, mem, ypred, beta, s, gamma, g, a_t],_ = theano.scan(fn = step, sequences = [x],  \
    outputs_info= [w0, mem0, None, None, None, None, None, None])
cost = T.sum(T.nnet.binary_crossentropy(5e-6 + ypred,y), axis=1).sum()
# cost = T.sqrt(T.mean((y-ypred)**2))
# f = theano.function(inputs = [x,y], outputs = cost)
train = theano.function(inputs = [x,y], outputs = [cost, ypred], updates = RMSprop(cost = cost, params = params))

test = theano.function(inputs = [x], outputs = ypred)