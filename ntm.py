#neural turing machine by deep mind
import os
os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=cpu,floatX=float32'
import theano
from theano import tensor as T
import numpy as np
from scipy.linalg import circulant
from plotting import plotting
#normalize each row
def vector_softmax(w):
	e = T.exp(w)
	dist = e / T.sum(e)
	return dist


M = 20
N = 128
nHeads = 1
shiftwidth = 3
y_size = 8
lr = 0.001
insize, hsize, outsize = y_size+M, 200,  2*y_size+3*M+3+shiftwidth

mem0 = theano.shared(np.random.rand(N,M).astype(theano.config.floatX))
w_init = theano.shared(np.random.rand(N,).astype(theano.config.floatX))
w0 =  vector_softmax(w_init)
#controller params
w_xh = theano.shared(np.random.uniform(-1,1, (insize, hsize)).astype(theano.config.floatX))
bh = theano.shared(np.zeros((hsize,), dtype = theano.config.floatX))
w_ho = theano.shared(np.random.uniform(-1,1, (hsize, outsize)).astype(theano.config.floatX))
bo = theano.shared(np.zeros((outsize,), dtype = theano.config.floatX))

#k = theano.shared(np.random.uniform(0,1, (M,)).astype(theano.config.floatX))
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
	output_raw = raw_outputs[:2*y_size]
	beta_raw =raw_outputs[2*y_size]
	gate_raw  =raw_outputs[2*y_size+1]
	gamma_raw = raw_outputs[2*y_size+2]
	key = raw_outputs[2*y_size+3:2*y_size+3+M]
	erase_raw =raw_outputs[2*y_size+3+M:2*y_size+3+M*2]
	add = raw_outputs[2*y_size+3+M*2:2*y_size+3+M*3]
	shift_raw = raw_outputs[2*y_size+3+M*3:2*y_size+3+M*3+shiftwidth]
	output = T.nnet.softmax(output_raw.reshape((y_size,2)))
	
	shift = T.nnet.softmax(shift_raw.dimshuffle(0, 'x'))[0]#reshape((1,shift_raw.shape[0])))[0] 
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
	shift_conv = circulant(np.arange(M)).T[np.arange(-(shiftwidth//2),(shiftwidth//2)+1)][::-1]
	return T.sum(shift * weight[shift_conv],axis=0)
	

def step(x_t, w_tm1, mem_tm1):
	#read
	r_t = read(mem_tm1, w_tm1)
	#calculate head parameters 
	k_t, s_t, e_t, a_t, beta_t, g_t, gamma_t, y_t = controller(x_t, r_t)
	wc_t = T.nnet.softmax(beta_t*T.dot(k_t, mem_tm1))
	wg_t = g_t*wc_t + (1.0-g_t)* w_tm1
	wshift_t = shift_convolve(w_tm1, s_t)
	w_t = w_tm1 ** gamma_t
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

[w, mem, y_, beta, s, gamma, g, a_t],_ = theano.scan(fn = step, sequences = [x],  \
	outputs_info= [w0, mem0, None, None, None, None, None, None])
#prob_y_given_x =prob_y_given_x[:,0,:]
ypred = T.argmax(y_, axis= 2)
y_ = y_.reshape((x.shape[0]*y_size, 2))
yflat = T.TensorVariable.flatten(y)
y_ = (y_[T.arange(yflat.shape[0]), yflat])
y_ = y_.reshape((x.shape[0], y_size))
#f = theano.function(inputs = [x,y], outputs = [cost])
cost = T.mean(-T.sum(T.log(y_), axis = 1))
#cost = T.mean(cost_arr)
#grad = T.grad(cost = cost, wrt = params)
#updates = [(p, p-lr*gp) for p, gp in zip(params, grad)]
train = theano.function(inputs = [x,y], outputs = [cost, y_], updates = RMSprop(cost = cost, params = params))

test = theano.function(inputs = [x], outputs = ypred)