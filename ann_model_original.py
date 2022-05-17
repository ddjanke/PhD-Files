import numpy as np
from numba import njit,vectorize,cuda
import matplotlib.pyplot as plt
import math
#from copy import deepcopy
from scipy import optimize
import threading
import queue
import multiprocessing
nthreads = multiprocessing.cpu_count()
import ergence as erg



@cuda.jit
def tanh_grad(z):
    x, y = cuda.grid(2)
    if x < z.shape[0] and y < z.shape[1]:
       z[x, y] = 1/(math.cosh(z[x, y])**2)
	   
@cuda.jit
def tanh(z):
    x, y = cuda.grid(2)
    if x < z.shape[0] and y < z.shape[1]:
       z[x, y] = math.tanh(z[x,y])

@vectorize(['float64(float64,float64,float64)'], target='parallel')
def parallel_tanh_grad(x,G,R):
	return G*R/(math.cosh(R*x))**2

@vectorize(['float64(float64,)'], target='parallel')
def parallel_tanh(x):
	return math.tanh(x)

def numpifyWeights(weights,nodes):
	maxdim = max(nodes[1:])
	tot = sum(nodes[:-1]+1)
	weightrix = np.zeros((maxdim,tot))
	prog = 0
	for w in weights:
		weightrix[:w.shape[0],prog:prog+w.shape[1]] = w
		prog +=w.shape[1]
	return weightrix

def deNumpifyWeights(weights,nodes):
	w = []
	tot = 0
	for i,n in enumerate(nodes[1:]):
		w += [weights[:n,tot:tot+nodes[i]+1]]
		tot += nodes[i]+1
	return w

@njit(nogil=True)
def dotMultiply(X,gain):
	return X.dot(gain.T)

@vectorize(['float64(float64,)'], target='parallel')
def parallel_log(y):
	if y==0: y = 0.0001
	return math.log(y)

@njit
def makeYk(y,nout):
	m = len(y)
	ys = np.ones((m,nout))*np.arange(1,nout+1)
	cat = (np.ones((m,nout)).T*y).T
	yk = 0.0+ (cat==ys)

	return yk

@njit(nogil=True, cache = True)
def npjitGradient(yk,yglog,z,a,weights,nodes):
	J = 0
	Delta = np.zeros(weights.shape)
	
	d = np.zeros((np.sum(nodes[1:]),))
	Dtemp = np.zeros(weights.shape)
	m = z.shape[0]
	layers = len(nodes)
	
	a_nodes = nodes+1
	a_nodes[-1] -= 1
	
	for i in range(m):
		ys = np.hstack((yk[i],1-yk[i]))
		J -= ys.dot(yglog[i])/m
		
		ai = a[i,:]
		zi = z[i,:]

		d[-nodes[-1]:] = (ai[-nodes[-1]:] - yk[i])*2
		for j in range(2,layers):
			w1_j = weights[:nodes[1-j],-np.sum(a_nodes[-j:-1]):weights.shape[1]-np.sum(a_nodes[1-j:-1])]
			d1_j = d[-np.sum(nodes[1-j:]):d.shape[0]-np.sum(nodes[layers+2-j:layers])]
			a_ji = ai[-np.sum(a_nodes[-j:]):ai.shape[0]-np.sum(a_nodes[1-j:])]
			z_ji = zi[-np.sum(nodes[-j:]):zi.shape[0]-np.sum(nodes[1-j:])]
			
			Dtemp[:nodes[1-j],-np.sum(a_nodes[-j:-1]):Dtemp.shape[1]-np.sum(a_nodes[1-j:-1])] = np.outer(d1_j,a_ji)/m
			d[-np.sum(nodes[-j:]):d.shape[0]-np.sum(nodes[1-j:])] = np.multiply((np.ascontiguousarray(w1_j.T)).dot(d1_j)[1:],z_ji/2)
			
		Dtemp[:nodes[1],:a_nodes[0]] = np.outer(d[:nodes[1]],ai[:a_nodes[0]])/m
		Delta += Dtemp
		
	return J,Delta

def parallelBackProp(y,z,a,weights,nodes,m):
	nthreads = multiprocessing.cpu_count()
	yguess = a[-1]
	yglog = parallel_log(np.hstack((yguess,1-yguess)))
	nout = nodes[-1]
	yk = makeYk(y,nout)
	a = np.hstack(a)
	z = np.hstack(z)
	z = parallel_tanh_grad(z,1,1)
	
	#threadsperblock = (32, 32)
	#blockspergrid_x = math.ceil(z.shape[0] / threadsperblock[0])
	#blockspergrid_y = math.ceil(z.shape[1] / threadsperblock[1])
	#blockspergrid = (blockspergrid_x, blockspergrid_y)
	#tanh_grad[blockspergrid, threadsperblock](z)
	
	in_arrays = [yk,yglog,z,a]
	
	weightrix = numpifyWeights(weights,np.array(nodes))
	Delta = np.zeros(weightrix.shape,dtype=np.float64)
	J = 0
	
	chunklen = (m + nthreads - 1) // nthreads
	chunks = [[arr[i * chunklen:(i + 1) * chunklen] for arr in in_arrays]+[weightrix,np.array(nodes)] for i in range(nthreads)]
	
	que = queue.Queue()
	#threads = [threading.Thread(target = npjitGradient,args = chunk) for chunk in chunks]
	threads = [threading.Thread(target = lambda q, args: q.put(npjitGradient(*args)),args = (que,chunk)) for chunk in chunks]
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()
	while not que.empty():
		result = que.get()
		J += result[0]/nthreads
		Delta += result[1]/nthreads
	
	Delta = deNumpifyWeights(Delta,nodes)
	return J,Delta


def unflattenWeights(weights,nodes):
	tot = 0
	unrolled = []
	for i in range(len(nodes)-1):
		ni = (nodes[i]+1)*nodes[i+1]
		if tot + ni > len(weights):
			tot += len(weights[tot:])
		else:
			unrolled += [weights[tot:tot+ni].reshape((nodes[i+1],nodes[i]+1))]
			tot += ni

	if len(unrolled) != len(nodes)-1:
		print(len(unrolled),len(nodes)-1)
		raise ValueError('Number of weights does not match total synapses: '+str(len(weights))+' vs. '+str(tot))
	return unrolled

def flattenAll(weights):
	allweights = np.array([])
	for w in weights:
		allweights = np.concatenate([allweights,np.ravel(w)])
		
	return allweights

def countSynapses(nodes):
	tot = 0
	for i,n in enumerate(nodes[:-1]):
		tot += (n+1)*nodes[i+1]
	return tot

class ANN_Model:
	
	### Initialization functions ###
	
	def __init__(self, nodes):
		if len(nodes) < 2:
			print('Parameter \'nodes\' must be a tuple of length >= 2')
			return
		
		self.nodes = nodes
		self.synapses = countSynapses(nodes)
		self.voltages = np.random.uniform(-0.1,0.1,(self.synapses,))
		self.rates = 2*np.ones((self.synapses,))
		self.maxGain = 10*np.ones((self.synapses,))
		self.calcWeights()
		self.j = math.inf
		
		
	### get/set methods ###
	
	def initialize(self):
		self.voltages = np.random.uniform(-0.1,0.1,(self.synapses,))
		self.calcWeights(inplace = True)
	
	def calcWeights(self,inplace=True,maxGainIn=1,ratesIn=1):
		weights = self.maxGain*maxGainIn*parallel_tanh(self.rates*ratesIn*self.voltages)
		if inplace:
			self.weights = weights
		else:
			return weights
		
	def calcVoltages(self,inplace=True):
		voltages = np.arctanh(self.weights/self.maxGain)/self.rates
		if inplace:
			self.voltages = voltages
		else:
			return voltages
		
	def size_check(self,weights):
		if len(weights) != self.synapses:
			print('Incorrect lengths. Expected {}, got {}'.format(self.synapses, len(weights)))
			return False
		return True
			
	def setVoltages(self,voltages):
		if not self.size_check(voltages): return
		self.voltages = voltages
		self.calcWeights()
		
	def setWeights(self,weights):
		if not self.size_check(weights): return
		self.weights = weights
		self.calcVoltages()
			
	def setMaxGain(self,maxGain):
		if not self.size_check(maxGain): return
		self.maxGain = maxGain
		self.calcWeights()
			
	def setRates(self,rates):
		if not self.size_check(rates): return
		self.rates = rates
		self.calcWeights()
		
	### Train/Predict ###
	
	def fit(self,X,y,runs=1,retrain = False):
		if not retrain:
			bound = list(zip(flattenAll(self.maxGain)*-0.98,flattenAll(self.maxGain)*0.98))
		else:
			up_weight = flattenAll(self.maxGain)*parallel_tanh(flattenAll(self.rates)*(flattenAll(self.voltages)+0.02))
			down_weight = flattenAll(self.maxGain)*parallel_tanh(flattenAll(self.rates)*(flattenAll(self.voltages)-0.02))
			bound = list(zip(down_weight,up_weight))
		
		j = math.inf
		best = None
		for i in range(runs):
			if i > 0 : self.initialize()
			result = optimize.fmin_tnc(func=self.nnCostFunction, x0=self.weights, args=(X, y), bounds = bound, disp = 0)
			if self.j < j:
				j = self.j
				best = result
		
		self.setWeights(best[0])
		return best
	
	def predict(self,X):
		z,a,unflatWeights = self.forwardProp(X,self.weights)
		if a[-1].shape[1] == 1: return np.ravel(0+(a[-1]>0.5))
		else: return np.concatenate([np.where(a==np.amax(a)) for a in a[-1]])+1
		
	def predictAccuracy(self,X,y):
		Result = []
		for x,yi in zip(X,y):
			yo = self.predict(x)
			acc = erg.confusionMat(yi,yo)
			Result += [acc[0]]
		return np.array(Result)
			
	def forwardProp(self,X,weights):
		m,n = X.shape
		z = []; a = [X]
		unflat_weights = unflattenWeights(weights,self.nodes)
		for g in unflat_weights:
			a[-1] = np.hstack((np.ones((m,1)),a[-1]))
			j = dotMultiply(a[-1],g)
			k = (parallel_tanh(j)+1)/2
			z += [j]; a += [k]
		return z,a,unflat_weights
			
	### These are a set of functions used for the cost/gradient of the network
	
	def nnCostFunction(self,weights,X,y):
		# Forward Propagation
		m,n = X.shape
		p = np.random.permutation(m)
		X = X[p,:]; y = y[p]

		z,a,unflat_weights = self.forwardProp(X,weights)

		J,Delta = parallelBackProp(y,z,a,unflat_weights,self.nodes,m)
		
		Delta = flattenAll(Delta)
		if J < self.j:
			print("{0:7.4f}".format(J),end=', ')
			self.j = J
			self.grad = Delta
		
		return J,Delta

class ANN_Model_Parallel(ANN_Model):
	
	def __init__(self, nodes,noise,directory):
		ANN_Model.__init__(self,nodes)
		self.directory = directory

		if noise.shape[1] != (5*nodes[0]+self.synapses*2):
			print('Not enough noise variables. Expected {}, got {}'.format(5*nodes[0]+self.synapses*2,noise.shape[1]))
			return
		self.analog_noise = noise[:,:5*nodes[0]]
		self.weight_noise = noise[:,5*nodes[0]:]
		self.variations = noise.shape[0]
		
		
	#### Parameter get/set methods ####			
	def applyWeightNoise(self,voltages,WeightNoise,inplace=True):
		if not self.size_check(voltages): return
		if WeightNoise.size/2 != self.synapses:
			print('Size mismatch: WeightNoise')
			return
		maxGain = self.maxGain*WeightNoise[:self.synapses]
		rates = self.rates*WeightNoise[self.synapses:]
		weights = parallel_tanh(rates*voltages)*maxGain
		if inplace:
			self.weights = weights
			return maxGain,rates
		else:
			return weights,maxGain,rates
	
	### NN training/testing methods ###
	def train(self,runs=1, dropout=False, dropout_p=0):
		bound = list(zip(np.ones((self.synapses,))*-1,np.ones((self.synapses,))))
		#bound = list(zip(flattenAll(self.maxGain)*-0.98,flattenAll(self.maxGain)*0.98))
		j = math.inf
		best = None
		for i in range(runs):
			if i > 0 : self.voltages = np.random.uniform(-0.1,0.1,(self.synapses,))
			if dropout:
				options = {}
				result = optimize.minimize(fun = self.nnCostFunction, x0 = self.voltages, args = (dropout,dropout_p), method = 'SLSQP', jac = True, bounds = bound, options = options).x
			else:
				result = optimize.fmin_tnc(func=self.nnCostFunction, x0=self.voltages, args = (dropout,dropout_p), bounds = bound, disp=0)[0]
			if self.j < j:
				j = self.j
				best = result
		self.setVoltages(best)
		return best
	
	def predict(self,*args):
		if len(args) == 0:
			i = np.random.randint(0,1001)
			filename = 'ErgFeatures'+str(i)+'.npz'
			data = np.load(self.directory+filename)
			X = data['Xtrain']
			y = data['ytrain']
		else:
			X,y = args
		z,a,weights = self.forwardProp(X,self.weights)
		if a[-1].shape[1] == 1: return y,np.ravel(0+(a[-1]>0.5))
		else: return y,np.concatenate([np.where(a==np.amax(a)) for a in a[-1]])+1
		
	def predictAll(self):
		Result = np.zeros((self.variations,7))
		for i in range(self.variations):
			self.applyWeightNoise(self.voltages,self.weight_noise[i,:])
			filename = 'ErgFeatures'+str(i)+'.npz'
			failed = True
			while failed:
				try:
					data = np.load(self.directory+filename)
					Xtrain = data['Xtrain']
					ytrain = data['ytrain']
					Xval1 = data['Xval1']
					yval1 = data['yval1']
					Xval2 = data['Xval2']
					yval2 = data['yval2']
					X1t = data['X1t']
					X2t = data['X2t']
					X3t = data['X3t']
					X4t = data['X4t']
					yat = data['yat']
					ybt = data['ybt']
					yct = data['yct']
					ydt = data['ydt']
					failed = False
				except:
					print('Retry.')
					erg.createFeatures(self.analog_noise[i], True, self.directory, i)
			
			X = [Xtrain, Xval1, Xval2, X1t, X2t, X3t, X4t]
			y = [ytrain, yval1, yval2, yat, ybt, yct, ydt]
			
			Result[i,:] = self.predictAccuracy(X,y)
			print(i,'/',self.variations,' predicted')
		return Result
	
	def predictAccuracy(self,X,y):
		Result = []
		for x,yi in zip(X,y):
			pred = self.predict(x,yi)
			acc = erg.confusionMat(*pred)
			Result += [acc[0]]
		return np.array(Result)
		
	### These are a set of functions used for the cost/gradient of the network
	def nnCostFunction(self,voltages,dropout=False,dropout_p = 0):
		# Forward Propagation
		J_tot = 0
		Grad_tot = 0
		
		if dropout:
			keepers = (np.random.uniform(0,1,(np.sum(self.nodes[:-1]),)) >= dropout_p)+0
			onesVolt = unflattenWeights(np.ones(voltages.shape),self.nodes)
			tot = 0
			dropList = []
			for i,n in enumerate(self.nodes[:-1]):
				keep = keepers[tot:tot+n]
				dropList += [onesVolt[i]*np.concatenate(([1],keep))]
				tot += n
			drop = flattenAll(dropList)
			test_voltages = voltages * drop
		else: test_voltages = np.copy(voltages)
			
		
		for i in range(self.variations):
			filename = 'ErgFeatures'+str(i)+'.npz'
			failed = True
			while failed:
				try:
					data = np.load(self.directory+filename)
					failed = False
				except:
					erg.createFeatures(self.analog_noise[i], True, self.directory, i)
			X = data['Xtrain']
			y = data['ytrain']
			m,n = X.shape
			weights, maxGain, rates = self.applyWeightNoise(test_voltages,self.weight_noise[i,:],inplace=False)
			z,a,unflat_weights = self.forwardProp(X,weights)

			J,Delta = parallelBackProp(y,z,a,unflat_weights,self.nodes,m)
			J_tot += J
			flat_grad = flattenAll(Delta)
			
			vgrad = flat_grad*parallel_tanh_grad(test_voltages,self.maxGain,self.rates)
			Grad_tot += vgrad
		
		J = J_tot/self.variations
		Delta = Grad_tot/self.variations
		if J < self.j:
			print("{0:7.4f}".format(J),end=', ')
			self.j = J
		
		if dropout:
			Delta *= drop
		
		return J,Delta