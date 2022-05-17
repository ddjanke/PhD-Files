import numpy as np
#from scipy.io.wavfile import read as readwav
import ergence as erg
import ANN_Model as ann
import ANNModelTorch as annt
import gc
import os

Audio_directory = "/home/djanke3/Documents/Audio/"
feature_directory_1 = '/hdd2/features/'
feature_directory_2 = '/hdd3/features/'

numfil = 6

erg.importAudio(Audio_directory)

nodes = (numfil, numfil*3, numfil*2, numfil, 1)

'''
### Train ideal network ###
var = np.ones(5*numfil)
Xtrain, ytrain, Xval1, yval1, Xval2, yval2,\
 X1t, yat, X2t, ybt, X3t, yct, X4t, ydt = erg.createFeatures(var,False)
	
print('Initializing Classifier...')
clf = ann.ANN_Model((numfil,2*numfil,1))

X = [Xtrain, Xval1, Xval2, X1t, X2t, X3t, X4t]
y = [ytrain, yval1, yval2, yat, ybt, yct, ydt]

print('Training...')
clf.fit(Xtrain, ytrain, 5)
print("Done")

pred1 = clf.predict(Xtrain)

Result = clf.predictAccuracy(X,y)

new_volt = clf.voltages
#print(clf.voltages)
np.savetxt('TrainedVoltsLarge.csv',new_volt,delimiter=',')

input()
'''
#############################################################################################################################

nodes = (numfil, numfil*3, numfil*2, numfil, 1)
synapses = ann.countSynapses(nodes)
#erg.createFeatures(np.ones((5*numfil,)))
'''
feat_rand = np.vstack([np.random.normal(1,x/3,(1000,5*numfil)) for x in [0.05,0.1,0.15,0.2]])
print(feat_rand.shape)
np.savetxt('FeatureVars.csv',feat_rand,delimiter=',')

w0_rand = np.vstack([np.random.normal(1,x/3,(1000,2*(nodes[0]+1)*nodes[1])) for x in [0.05,0.1,0.15,0.2]])
print(w0_rand.shape)
np.savetxt('Weight0VarsLarge.csv',w0_rand,delimiter=',')

w1_rand = np.vstack([np.random.normal(1,x/3,(1000,2*(nodes[1]+1)*nodes[2])) for x in [0.05,0.1,0.15,0.2]])
print(w1_rand.shape)
np.savetxt('Weight1VarsLarge.csv',w1_rand,delimiter=',')

w2_rand = np.vstack([np.random.normal(1,x/3,(1000,2*(nodes[2]+1)*nodes[3])) for x in [0.05,0.1,0.15,0.2]])
print(w2_rand.shape)
np.savetxt('Weight2VarsLarge.csv',w2_rand,delimiter=',')

w3_rand = np.vstack([np.random.normal(1,x/3,(1000,2*(nodes[3]+1)*nodes[4])) for x in [0.05,0.1,0.15,0.2]])
print(w3_rand.shape)
np.savetxt('Weight3VarsLarge.csv',w2_rand,delimiter=',')

w01_rand = np.hstack([w0_rand,w1_rand])
print(w01_rand.shape)
np.savetxt('Weight01VarsLarge.csv',w01_rand,delimiter=',')

w23_rand = np.hstack([w2_rand,w3_rand])
print(w23_rand.shape)
np.savetxt('Weight23VarsLarge.csv',w23_rand,delimiter=',')

all_rand = np.hstack([feat_rand,w01_rand,w23_rand])
print(all_rand.shape)
np.savetxt('AllVarsLarge.csv',all_rand,delimiter=',')
input('Done')

#Variations in Features only

#voltages = np.loadtxt('TrainedVoltsLarge.csv',delimiter=',')
'''
random_vars = np.loadtxt('FeatureVars.csv',delimiter = ',')
random_vars = np.vstack((np.ones((1,5*numfil)),random_vars))
tot = len(random_vars)

Result = np.zeros((tot,2*7+2))

dir_full = False
feat_dir = feature_directory_1

for i in range(tot):
	
	if not dir_full:
		stats = os.statvfs('/hdd2')
		remain = stats.f_frsize * stats.f_bavail/10**9
		if remain < 5:
			dir_full = True
			feat_dir = feature_directory_2
	
	
	var = random_vars[i,:]
	#Xtrain, ytrain, Xval1, yval1, Xval2, yval2, X1t, yat, X2t, ybt, X3t, yct, X4t, ydt = erg.createFeatures(var,False)
	erg.createFeatures(var,True, feat_dir, i)
	'''
	print('Initializing Classifier...')
	clf = ann.ANN_Model(nodes)
	clf.setVoltages(voltages)
	print("Predicting...")
	
	X = [Xtrain, Xval1, Xval2, X1t, X2t, X3t, X4t]
	y = [ytrain, yval1, yval2, yat, ybt, yct, ydt]
	
	Result[i,:7] = erg.predictAccuracy(clf,X,y)
	
	print('Training...')
	clf.fit(Xtrain, ytrain, 1, True)
	print("Done")
	
	new_volt = clf.voltages
	diff_volt = np.absolute(new_volt-voltages)
	max_diff = np.amax(diff_volt)
	rms_diff = np.sqrt(np.mean(diff_volt**2))
	Result[i,7] = max_diff
	Result[i,8] = rms_diff
	
	Result[i,9:] = erg.predictAccuracy(clf,X,y)
	
	np.savetxt('Result_Features_bounded_Large.csv',Result,delimiter=',')
	'''
	print(i,'/',tot,' features only')
	
print('Done with features')

'''
# Variations in Weights

voltages = np.loadtxt('TrainedVoltsLarge.csv',delimiter=',')
random_vars = np.ones((1,synapses*2))
first = (nodes[0]+1)*nodes[1]
second = (nodes[1]+1)*nodes[2]

#First Weights only
temp = np.loadtxt('Weight0VarsLarge.csv',delimiter = ',')
tot = len(temp)
temp = np.hstack((temp[:,:first],np.ones((tot,second)),temp[:,first:],np.ones((tot,second))))
random_vars = np.vstack((random_vars,temp))

#Second Weights only
temp = np.loadtxt('Weight1VarsLarge.csv',delimiter = ',')
tot = len(temp)
temp = np.hstack((np.ones((tot,first)),temp[:,:second],np.ones((tot,first)),temp[:,second:]))
random_vars = np.vstack((random_vars,temp))

#All Weights
temp = np.loadtxt('WeightAllVarsLarge.csv',delimiter = ',')
tot = len(temp)
random_vars = np.vstack((random_vars,temp))

tot = len(random_vars)
Result = np.ones((tot,2*7+2))

var = np.ones(5*numfil)

# Import audio data
Xtrain, ytrain, Xval1, yval1, Xval2, yval2, X1t, yat, X2t, ybt, X3t, yct, X4t, ydt = erg.createFeatures(var,False)

for i in range(tot):
	
	w0var = random_vars[i,:]
	
	print('Initializing Classifier...')
	clf = ann.ANN_Model(nodes)
	clf.setVoltages(voltages)
	gains = w0var[:synapses]*10
	rates = w0var[synapses:]*2
	clf.setMaxGain(gains)
	clf.setRates(rates)
	print("Predicting...")
	
	X = [Xtrain, Xval1, Xval2, X1t, X2t, X3t, X4t]
	y = [ytrain, yval1, yval2, yat, ybt, yct, ydt]
	
	Result[i,:7] = erg.predictAccuracy(clf,X,y)
	
	print('Training...')
	clf.fit(Xtrain, ytrain, 1)
	print("Done")
	
	new_volt = clf.voltages
	diff_volt = np.absolute(new_volt-voltages)
	max_diff = np.amax(diff_volt)
	rms_diff = np.sqrt(np.mean(diff_volt**2))
	Result[i,7] = max_diff
	Result[i,8] = rms_diff
	
	Result[i,9:] = erg.predictAccuracy(clf,X,y)
	
	np.savetxt('Result_W0W1WAll_bounded_Large.csv',Result,delimiter=',')
	print(i,'/',tot, 'Weights')
	



# All variations
	
voltages = np.loadtxt('TrainedVoltsLarge.csv',delimiter=',')
random_vars = np.loadtxt('AllVarsLarge.csv',delimiter = ',')
random_vars = np.vstack((np.ones((1,random_vars.shape[1])),random_vars))
tot = len(random_vars)

Result = np.zeros((tot,2*7+2))
ifeature = 5*numfil

for i in range(tot):
	
	var = random_vars[i,:ifeature]
	w0var = random_vars[i,ifeature:]
	
	Xtrain, ytrain, Xval1, yval1, Xval2, yval2, X1t, yat, X2t, ybt, X3t, yct, X4t, ydt = erg.createFeatures(var,False)
	
	print('Initializing Classifier...')
	clf = ann.ANN_Model((Xtrain.shape[1],Xtrain.shape[1]*2,1))
	clf.setVoltages(voltages)
	gains = w0var[:synapses]*10
	rates = w0var[synapses:]*2
	clf.setMaxGain(gains)
	clf.setRates(rates)
	print("Predicting...")
	
	X = [Xtrain, Xval1, Xval2, X1t, X2t, X3t, X4t]
	y = [ytrain, yval1, yval2, yat, ybt, yct, ydt]
	Result[i,:7] = erg.predictAccuracy(clf,X,y)
	
	print('Training...')
	clf.fit(Xtrain, ytrain, 1, True)
	print("Done")
	
	new_volt = clf.voltages
	diff_volt = np.absolute(new_volt-voltages)
	max_diff = np.amax(diff_volt)
	rms_diff = np.sqrt(np.mean(diff_volt**2))
	Result[i,7] = max_diff
	Result[i,8] = rms_diff
	
	print('Re-predicting...')
	Result[i,9:] = erg.predictAccuracy(clf,X,y)
	
	np.savetxt('Result_All_bounded_Large.csv',Result,delimiter=',')
	print(i,'/',tot,' all')

'''
# Train multiple variations in parallel with pre-generated features.
dropout = True
random_vars0 = np.loadtxt('AllVars.csv',delimiter = ',')
random_vars1 = np.loadtxt('AllVarsTest.csv',delimiter = ',')
num_var = 1000
numfil = 5
ifeature = 5*numfil
for k in [4,3,2,1]: #range(4,5):
	random_vars = np.vstack((random_vars0[num_var*k:num_var*k+num_var,:],
						  random_vars1[num_var*k:num_var*k+num_var,:]))
	tot = random_vars.shape[0]
	
	for i in range(2*num_var):
		if k == 4: break
		gc.collect()
		print(str(k)+':',i,'/',tot,' saved')
		var = random_vars[i,:ifeature]
		erg.createFeatures(var, True, feature_directory, i)
	
	if random_vars.ndim ==1:
		random_vars = np.reshape(random_vars,(1,len(random_vars)))
	
	clf = ann.ANN_Model_Parallel(nodes,random_vars[:num_var],feature_directory)
	clf.train(2,dropout,0.1)
	print('Done')
	trained_volts = clf.voltages
	np.savetxt('Voltages_Parallel'+str(k)+'.csv',trained_volts,delimiter=',')
	
	# Comment this section if not enough room on HDD for test and train
	clf = ann.ANN_Model_Parallel(nodes,random_vars,feature_directory)
	#trained_volts = np.loadtxt('Voltages_Parallel'+str(k)+'.csv',delimiter = ',')
	clf.setVoltages(trained_volts)
	
	Result = clf.predictAll()
	np.savetxt('Result_Parallel'+str(k)+'.csv',Result,delimiter=',')
	
	## Use this section only if not enough room on HDD for train and test
	'''
	for i in range(num_var):
		gc.collect()
		print(i,'/',tot,' saved')
		var = random_vars[num_var+i,:ifeature]
		erg.createFeatures(var, True, feature_directory, i)
	
	if random_vars.ndim ==1:
		random_vars = np.reshape(random_vars,(1,len(random_vars)))
	
	clf = ann.ANN_Model_Parallel(nodes,random_vars[num_var:],feature_directory)
	clf.setVoltages(trained_volts)
	
	Result = clf.predictAll()
	np.savetxt('Result_Parallel'+str(k)+'1.csv',Result,delimiter=',')
	'''