import numpy as np
#from scipy.io.wavfile import read as readwav
import ergence as erg
import ANN_Model as ann
import ANNModelTorch as annt
import gc
import os
import torch

os.chdir("/home/djanke3/Documents/Spyder/")

Audio_directory = "/home/djanke3/Documents/Audio/"
feature_directory_1 = '/hdd2/features/'
feature_directory_2 = '/hdd3/features/'

numfil = 6
noise_len = 1000
noise_per = [0.20]

test = 'weights'

erg.importAudio(Audio_directory)
device = 'cuda:0'
folder = 'Tanh'

for num_archs in range(0,4):

	#for net_size in ['Small','Med','Large','Mid', 'Long']: #
	for net_size in ['L_3_3', 'L_3_6', 'L_3_9', 'L_3_12', 'L_6_3', 'L_6_6', 'L_6_9', 'L_6_12',
					   'L_9_3', 'L_9_6', 'L_9_9', 'L_9_12', 'L_12_3', 'L_12_6', 'L_12_9', 'L_12_12']:
		
		weight_directory = "Results_Torch/Weights/{}/{}/".format(folder,net_size)
		noise_directory = "Results_Torch/Noise/{}/".format(net_size)
		result_directory = "Results_Torch/{}/{}{}/".format(folder,net_size,num_archs)
		
		if not os.path.exists(weight_directory):
			os.makedirs(weight_directory)
		if not os.path.exists(noise_directory):
			os.makedirs(noise_directory)
		if not os.path.exists(result_directory):
			os.makedirs(result_directory)
			
		#net_size = 'Large'
		if net_size == 'Small':
			nodes = (numfil, numfil, 1)
		if net_size == 'Med':
			nodes = (numfil, numfil*2, numfil, 1)
		if net_size == 'Large':
			nodes = (numfil, numfil*3, numfil*2, numfil, 1)
		if net_size == 'Mid':
			nodes = (numfil, numfil, numfil, 1)
		if net_size == 'Long':
			nodes = (numfil, numfil, numfil, numfil, 1)
		
		if net_size == 'L_3_3':
			nodes = (numfil, 3, 3, 6, 1)
		if net_size == 'L_3_6':
			nodes = (numfil, 3, 6, 6, 1)
		if net_size == 'L_3_9':
			nodes = (numfil, 3, 9, 6, 1)
		if net_size == 'L_3_12':
			nodes = (numfil, 3, 12, 6, 1)
		if net_size == 'L_6_3':
			nodes = (numfil, 6, 3, 6, 1)
		if net_size == 'L_6_6':
			nodes = (numfil, 6, 6, 6, 1)
		if net_size == 'L_6_9':
			nodes = (numfil, 6, 9, 6, 1)
		if net_size == 'L_6_12':
			nodes = (numfil, 6, 12, 6, 1)
		if net_size == 'L_9_3':
			nodes = (numfil, 9, 3, 6, 1)
		if net_size == 'L_9_6':
			nodes = (numfil, 9, 6, 6, 1)
		if net_size == 'L_9_9':
			nodes = (numfil, 9, 9, 6, 1)
		if net_size == 'L_9_12':
			nodes = (numfil, 9, 12, 6, 1)
		if net_size == 'L_12_3':
			nodes = (numfil, 12, 3, 6, 1)
		if net_size == 'L_12_6':
			nodes = (numfil, 12, 6, 6, 1)
		if net_size == 'L_12_9':
			nodes = (numfil, 12, 9, 6, 1)
		if net_size == 'L_12_12':
			nodes = (numfil, 12, 12, 6, 1)
		
		
		synapses = ann.countSynapses(nodes)
		
		if test == 'weights':
			#Xtrain, ytrain, Xval1, yval1, Xval2, yval2,\
			# X1t, yat, X2t, ybt, X3t, yct, X4t, ydt = erg.createFeatures(var,False)
			#X = [Xtrain, Xval1, Xval2, X1t, X2t, X3t, X4t]
			#y = [ytrain, yval1, yval2, yat, ybt, yct, ydt]
			
			var = np.ones(5*numfil)
			Xtrain, ytrain, Xval1, yval1, Xval2, yval2 = erg.createFeatures(var,False, train_only = True)
			X = [Xtrain, Xval1, Xval2]
			y = [ytrain, yval1, yval2]
			
			weight_files = os.listdir(weight_directory)
			if 'TrainedVoltages{}.pt'.format(num_archs) not in weight_files:
				### Train ideal network ###
				print('Initializing Classifier...')
				clf = ann.ANNModelTorch(nodes, device = device, activation = folder)
				
				print('Training...')
				#losses = clf.fit_plot(Xtrain, ytrain,X[1:],y[1:], 2000,verbose = 1)
				losses = clf.fit(Xtrain, ytrain, 2000,verbose = 100)
				print("Done")
				
				#pred1 = clf.predict(Xtrain)
				
				Result = clf.predict_accuracy_all(X,y)
				print(Result)
				
				clf.save_model(weight_directory+'TrainedVoltages{}.pt'.format(num_archs))
		
		
		#############################################################################################################################

		noise_files = os.listdir(noise_directory)
		if test == 'features':
			if 'FeatureVars.csv'.format(net_size) not in noise_files:
				feat_rand = np.vstack([np.random.normal(1,x/3,(noise_len,5*numfil)) for x in noise_per])
				print(feat_rand.shape)
				np.savetxt(noise_directory+'FeatureVars.csv',feat_rand,delimiter=',')
		
		elif test == 'weights':
			if 'Weight0Vars{}.csv'.format(net_size) not in noise_files:
				w0_rand = np.vstack([np.random.normal(1,x/3,(noise_len,2*(nodes[0]+1)*nodes[1])) for x in noise_per])
				print(w0_rand.shape)
				np.savetxt(noise_directory+'Weight0Vars{}.csv'.format(net_size),w0_rand,delimiter=',')
			
			if 'Weight1Vars{}.csv'.format(net_size) not in noise_files:
				w1_rand = np.vstack([np.random.normal(1,x/3,(noise_len,2*(nodes[1]+1)*nodes[2])) for x in noise_per])
				print(w1_rand.shape)
				np.savetxt(noise_directory+'Weight1Vars{}.csv'.format(net_size),w1_rand,delimiter=',')
			
			if net_size != "Small":
				if 'Weight2Vars{}.csv'.format(net_size) not in noise_files:
					w2_rand = np.vstack([np.random.normal(1,x/3,(noise_len,2*(nodes[2]+1)*nodes[3])) for x in noise_per])
					print(w2_rand.shape)
					np.savetxt(noise_directory+'Weight2Vars{}.csv'.format(net_size),w2_rand,delimiter=',')
			
			if net_size in ["Large", "Long"]:
				if 'Weight3Vars{}.csv'.format(net_size) not in noise_files:
					w3_rand = np.vstack([np.random.normal(1,x/3,(noise_len,2*(nodes[3]+1)*nodes[4])) for x in noise_per])
					print(w3_rand.shape)
					np.savetxt(noise_directory+'Weight3Vars{}.csv'.format(net_size),w3_rand,delimiter=',')
			
		
		print('Initializing Classifier...')
		clf = ann.ANNModelTorch(nodes, activation = folder)
		clf.load_state_dict(torch.load(weight_directory+'TrainedVoltages{}.pt'.format(num_archs)))

		
		# This section to put together the noise array for weights
		#'''
		# Variations in Weights
		weight_vars = np.ones((1,synapses*2))
		
		
		first = (nodes[0]+1)*nodes[1]
		second = (nodes[1]+1)*nodes[2]
		syn_list = [first,second]
		if net_size != 'Small':
			third = (nodes[2]+1)*nodes[3]
			syn_list += [third]
			if net_size in ["Large","Long"]:
				fourth = (nodes[3]+1)*nodes[4]
				syn_list += [fourth]
		
		
		start = 0
		end = 0
		tot = noise_len * len(noise_per)
		all_layers_noise = np.ones((tot,synapses*2))
		for i,syn in enumerate(syn_list):
			if i > 1: break
			filename = 'Weight{}Vars{}.csv'.format(i,net_size)
			temp = np.loadtxt(noise_directory+filename,delimiter = ',')
			end += syn
			tot = len(temp)
			layer_vars = np.ones((tot,synapses*2))
			layer_vars[:,start:end] = temp[:,:end-start]
			layer_vars[:,synapses+start:synapses+end] = temp[:,end-start:]
			all_layers_noise[:,start:end] = temp[:,:end-start]
			all_layers_noise[:,synapses+start:synapses+end] = temp[:,end-start:]
			weight_vars = np.vstack((weight_vars,layer_vars))
			start += syn
	
		#weight_vars = np.vstack((weight_vars,all_layers_noise))
		
		if test == 'weights':
			tot = len(weight_vars)
			Result = np.ones((tot,len(X)))
			
			# This section for testing weight accuracy with noise
			for i in range(tot):
				
				w0var = weight_vars[i,:]
				gnoise = w0var[:synapses]
				rnoise = w0var[synapses:]
				clf.add_noise(gnoise,rnoise)
				
				Result[i,:] = clf.predict_accuracy_all(X,y)
				
				np.savetxt(result_directory+'Result_Weights_{}.csv'.format(net_size),Result,delimiter=',')
				print(i,'/',tot, 'Weights')
		
		elif test == 'features':
			# This section to test feature-only noise
			#'''
			feat_vars = np.loadtxt(noise_directory+'FeatureVars.csv',delimiter = ',')
			feat_vars = np.vstack((np.ones((1,5*numfil)),feat_vars))
			
			def dir_full(dir1,dir2,full = [False,False],max_file = None,file_i = 0):
				
				if max_file is None:
					max_file = [0,None]
				
				save = True
				if not full[0]:
					stats = os.statvfs('/hdd2')
					remain = stats.f_frsize * stats.f_bavail/10**9
					feat_dir = dir1
					if remain < 5:
						full[0] = True
						feat_dir = dir2
					max_file[0] = file_i
				elif not full[1]:
					stats = os.statvfs('/hdd3')
					remain = stats.f_frsize * stats.f_bavail/10**9
					feat_dir = dir2
					if remain < 5:
						full[1] = True
					max_file[1] = file_i
				else:
					save = False
					feat_dir = ''
						
				return save, feat_dir, full, max_file
			

			tot = noise_len*len(noise_per)
			Result_feat = np.zeros((tot+1,3))
			Result_all = np.zeros((tot+1,3))
			
			save, feat_dir, full, max_file = dir_full(feature_directory_1,feature_directory_2,full = [False,False],max_file = None, file_i = 0)
			save = False
			feat_dir = feature_directory_1
			
			for i in range(tot):
				
				#save, feat_dir, full, max_file = dir_full(feature_directory_1,feature_directory_2,
				#									   full = full, max_file = max_file, file_i = i)
				
				filename = 'ErgFeatures'+str(i)+'.npz'
			
				try:
					#data = np.load(feat_dir+filename)
					Xtrain = data['Xtrain']
					ytrain = data['ytrain']
					Xval1 = data['Xval1']
					yval1 = data['yval1']
					Xval2 = data['Xval2']
					yval2 = data['yval2']
					#X1t = data['X1t']
					#X2t = data['X2t']
					#X3t = data['X3t']
					#X4t = data['X4t']
					#yat = data['yat']
					#ybt = data['ybt']
					#yct = data['yct']
					#ydt = data['ydt']
				except:
					#print('Retry.')
					Xtrain, ytrain, Xval1, yval1, Xval2, yval2 = erg.createFeatures(feat_vars[i,:], save, feat_dir, i, train_only = True)
				
				X = [Xtrain, Xval1, Xval2]
				y = [ytrain, yval1, yval2]
				
				Result_feat[i,:] = clf.predict_accuracy_all(X,y)
				
				np.savetxt(result_directory+'Result_Features_{}.csv'.format(net_size),Result_feat,delimiter=',')
				
				print(i,'/',tot,'features',end = '')
				
				# This section to do all layers testing at same time

				w0var = all_layers_noise[i,:]
				gnoise = w0var[:synapses]
				rnoise = w0var[synapses:]
				clf.add_noise(gnoise,rnoise)
				
				Result_all[i,:7] = clf.predict_accuracy_all(X,y)
				clf.add_noise(np.ones(synapses),np.ones(synapses))
				
				np.savetxt(result_directory+'Result_All_{}.csv'.format(net_size),Result_all,delimiter=',')
			
				print(' and all')
	
'''	
print('Done with features')

# All variations
	
voltages = np.loadtxt('TrainedVoltsLarge.csv',delimiter=',')
random_vars = np.loadtxt('AllVarsLarge.csv',delimiter = ',')
random_vars = np.vstack((np.ones((1,random_vars.shape[1])),random_vars))
tot = len(random_vars)

Result = np.zeros((tot,7))
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
'''
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