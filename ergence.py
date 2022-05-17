import math
from scipy.signal import butter,lfilter
import matplotlib.pyplot as plt
import numpy as np
#import numba
from numba import njit, prange #,cuda
import threading
import queue
#import multiprocessing
from scipy.io.wavfile import read as readwav
from scipy.io.wavfile import write as writewav

import os
import random
import csv

### Global Vairables

f0s = [50,145,225,285,360,450,567,714,899,1132,1426,1796,2262,2850,3590]

# List of learned parameters
alpha = 0.8
alpha2 = 0.8
lamb = 0
a1 = 100
a2 = 10
b1 = 10
b2 = 100
n = 0.001
fmx = 13
fmn = 4
o = 2
q = 0.707
numfil = 6

fmax = f0s[fmx]
fmin = f0s[fmn]

file = 0

Audio_directory = "/home/djanke3/Documents/Audio/"
feature_directory_1 = '/hdd2/features/'
feature_directory_2 = '/hdd3/features/'

MS_DNS_AUDIO_DIR = "/home/djanke3/Documents/Audio/DNS-Challenge/"

MS_DNS_RAW_DIR = "/home/djanke3/Documents/Audio/DNS-Challenge/"
MS_DNS_CLEAN_RAW_DIR = MS_DNS_RAW_DIR + "clean/"
MS_DNS_NOISE_RAW_DIR = MS_DNS_RAW_DIR + "noise/"

def read_wav_normalize(file):
    sr, audio = readwav(file)
    return sr, audio/2**15


class DataSplitter:
    def __init__(self, train = 0.7, val = None, test = None):
        self.ratios = [train, val, test]
        self.check_split()
        self.n = None
        self.permutation = None
        self.data_split = None
        
    def check_split(self):
        assert all([r >= 0 for r in [0 if r is None else r for r in self.ratios]]),\
            "All ratios in the split must be >= 0."
        if sum([0 if r is None else r for r in self.ratios])  != 1:
            assert self.ratios[0] <= 1, "Training ratio must be between 0 and 1."
            if self.ratios[1] is None:
                if self.ratios[2] is None:
                    self.ratios[1:] = [1 - self.ratios[0], 0]
                else:
                    assert sum(self.ratios[::2]) < 1, "Split ratios must add to 1."
                    self.ratios[1] = 1 - sum(self.ratios[::2])
            else:
                if self.ratios[2] is None:
                    assert sum(self.ratios[:2]) < 1, "Split ratios must add to 1."
                    self.ratios[2] = 1 - sum(self.ratios[:2])
                else:
                    total = sum(self.ratios)
                    self.ratios[0] = int(self.ratios[0] / total * 100) / 100
                    self.ratios[1] = int(self.ratios[1] / total * 100) / 100
                    self.ratios[2] = 1 - sum(self.ratios[:2])
                
            print("Split ratios do not add to 1.",
                  "Ratios have been adjusted to {}"\
                      .format(str(self.ratios)))
            
    def generate_permutation(self, n):
        self.n = n
        self.permutation = random.sample(range(n), n)
        self.raw_split()
        
    def save_split(self):
        with open('data_split.csv', 'w') as f:
            wr = csv.writer(f)
            wr.writerow(self.ratios)
            wr.writerow(self.permutation)
            
    def load_split(self, file = "data_split.csv"):
        with open(file, 'r') as f:
            r = csv.reader(f)
            data = list(r)
        self.permutation = list(np.array(data[1], dtype = int))
        self.ratios = list(np.array(data[0], dtype = float))
        self.n = len(self.permutation)
        self.raw_split()
        
    def raw_split(self):
        if self.data_split is None:
            indexes = [int(round(self.n*self.ratios[0])),
                       int(round(self.n*sum(self.ratios[:2])))]
            #print(indexes)
            self.data_split = [self.permutation[:indexes[0]],
                               self.permutation[indexes[0] : indexes[1]],
                               self.permutation[indexes[1]:]]
        return self.data_split
            
    def split(self, data):
        n = len(data)
        if self.n is None or self.permutation is None or self.n != n:
            self.generate_permutation(n)
            
        tr, v, te = self.raw_split()
        
        train_perm = [data[i] for i in tr]
        val_perm = [data[i] for i in v]
        test_perm = [data[i] for i in te]
        
        return train_perm, val_perm, test_perm
        
    

class AudioData:
    def __init__(self, target_dir = "", audio_root = MS_DNS_AUDIO_DIR):
        self.audio_root = audio_root
        self.clean_dir = audio_root + "clean/"
        self.noise_dir = audio_root + "noise/"
        self.noisy_dir = audio_root + "noisy/"
        self.label_dir = audio_root + "labels/"
        self.clean_files = sorted(os.listdir(self.clean_dir))
        self.noise_files = sorted(os.listdir(self.noise_dir))
        self.noisy_files = sorted(os.listdir(self.noisy_dir))
        try:
            self.labels = sorted(os.listdir(self.label_dir))
        except:
            self.labels = None
        
    def generate_labels(self):
        tot = 0
        tot_sum = 0
        for i,c in enumerate(self.clean_files):
            vsr, voice = self.read_audio(self.clean_dir + c)
            y = (RCenv(voice,vsr,0.01) > 0.01)
            tot += len(y)
            tot_sum += y.sum()
            fileid = c.split("fileid_")[1][:-4] + ".csv"
            np.savetxt(MS_DNS_AUDIO_DIR + "labels/label_" + fileid, y, delimiter = ',')
            if i%100 == 0: print(i)
            
        print(tot_sum/tot)
        
    def read_audio(self, file):
        return read_wav_normalize(file)


def expand_signal(signal, sample_rate, time_beginning, time_end):
    '''Expands the signal equally at the beginning and end with the reversed
    signal. For example, if the signal is being expanded by n samples, the
    new signal starts with the first n samples reversed and ends with the last
    n samples reversed.    

    Parameters
    ----------
    signal : list-like
        signal to expand
    sample_rate : int
        sample rate of signal
    time : number
        time (s) to expand by

    Returns
    -------
    numpy.ndarray
        expanded audio signal
    '''
    n0 = int(sample_rate * time_beginning)
    n1 = int(sample_rate * time_end)
    begin = signal[:n0][::-1]
    end = signal[-n1:][::-1]
    return np.concatenate((begin, signal, end))

def truncate_signal(signal, sample_rate, time_beginning, time_end):
    '''Truncates the beginning and end of the signal by an equal number of
    samples
    
    Parameters
    ----------
    signal : list-like
        signal to truncate
    sample_rate : int
        sample rate of signal
    time : number
        time (s) to truncate

    Returns
    -------
    numpy.ndarray
        truncated audio signal
    '''
    n0 = int(sample_rate * time_beginning)
    n1 = int(sample_rate * time_end)
    
    if len(signal.shape) == 1: return signal[n0:-n1]
    if len(signal.shape) == 2:
        time_axis = np.argmax(signal.shape)
        if time_axis == 0: return signal[n0:-n1,:]
        else: return signal[:,n0:-n1]
        
class BandpassArray:
    def __init__(self, audio, sample_rate, bands = 5, filterQ = 0.7,
                 filter_order = 2, fmin = 360, fmax = 3590, lowpass = True):
        self.audio = audio
        self.sample_rate = sample_rate
        self.bands = bands
        self.orders = np.ones(self.bands).astype(int) * filter_order
        self.settings = {"Q" : filterQ,
                         "order" : filter_order,
                         "fmin" : fmin,
                         "fmax" : fmax}
        self.types = ["band"] * bands
        if lowpass: self.types[0] = "low"
        self.initialize_parameters()
        
    def initialize_parameters(self):
        self.f0s = np.logspace(np.log10(self.settings["fmin"]),
                               np.log10(self.settings["fmax"]),
                               self.bands) / self.sample_rate * 2
        self.filterQs = np.ones(self.bands) * self.settings["Q"]
        self.filter_gains = np.ones(self.bands)
        self.noise_added = False
        
    
    def expand(self, time_beginning, time_end):
        self.audio = expand_signal(self.audio, self.sample_rate,
                                   time_beginning, time_end)
        self.expand_time_beginning = time_beginning
        self.expand_time_end = time_end
        
    def add_filter_noise(self, sigma):
        if self.noise_added: self.initialize_parameters()
        self.f0s *= np.random.normal(loc = 1, scale = sigma, size = self.bands)
        self.filterQs *= np.random.normal(1, sigma, self.bands)
        self.filter_gains = np.random.normal(1, sigma, self.bands)
        self.noise_added = True
        
    def filter_audio(self, band_num, f0, order, Q, type = "low"):
        if type == "low":
            b,a = butter(order, f0, "low")
        elif type == "band":
            flow = f0*(math.sqrt(1+1/4/Q**2)-1/2/Q)
            fhigh = f0*(math.sqrt(1+1/4/Q**2)+1/2/Q)
            b,a = butter(order, [flow, fhigh], "band")
            
        return band_num, lfilter(b, a, self.audio)
        
    def multithread_filter(self):
        lv = len(self.audio)
        self.filtered_signals = np.zeros((self.bands, lv))
        
        chunks = [[i, self.f0s[i], self.orders[i], self.filterQs[i],
                   self.types[i]] for i in range(self.bands)]
        
        que = queue.Queue()
        threads = [threading.Thread(target = lambda q, args:\
                                 q.put(self.filter_audio(*args)),
                                 args = (que,chunk)) for chunk in chunks]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        while not que.empty():
            result = que.get()
            self.filtered_signals[result[0],:] = result[1] * self.filter_gains[result[0]]
        
        
class FeatureArray:
    def __init__(self, bp_array, envelope_tau = 0.25,
                 decay_rate_unit = 1, sp_decay_rise = 1, sp_decay_fall = 20,
                 nl_decay_rise = 0.1, nl_decay_fall = 1):
        
        self.settings = {"tau" : envelope_tau,
                         "decay_unit" : decay_rate_unit,
                         "sp_rise" : sp_decay_rise,
                         "sp_fall" : sp_decay_fall,
                         "nl_rise" : nl_decay_rise,
                         "nl_fall" : nl_decay_fall}
        
        self.bpa = bp_array
        self.initialize_parameters()
        
    def initialize_parameters(self):
        #sr450 = 450 / 2 / self.bpa.sample_rate # 450 Hz normalized to sample_rate
        #self.taus = np.array([self.settings["tau"] * sr450 / self.bpa.f0s[i] \
        #             for i in range(self.bpa.bands)])
        self.taus = np.array([self.settings["tau"] for i in range(self.bpa.bands)])
        #self.rates = np.array([self.settings["decay_unit"] * sr450 / self.bpa.f0s[i] \
        #              for i in range(self.bpa.bands)])
        self.rates = np.array([self.settings["decay_unit"] for i in range(self.bpa.bands)])
        self.sp_rise = self.rates * self.settings["sp_rise"]
        self.sp_fall = self.rates * self.settings["sp_fall"]
        self.nl_rise = self.rates * self.settings["nl_rise"]
        self.nl_fall = self.rates * self.settings["nl_fall"]
        self.noise_added = False
        
    def add_feature_noise(self, sigma):
        if self.noise_added: self.initialize_parameters()
        self.taus *= np.random.normal(loc = 1, scale = sigma,
                                        size = self.bpa.bands)
        self.sp_rise *= np.random.normal(1, sigma, self.bpa.bands)
        self.sp_fall *= np.random.normal(1, sigma, self.bpa.bands)
        self.nl_rise *= np.random.normal(1, sigma, self.bpa.bands)
        self.nl_fall *= np.random.normal(1, sigma, self.bpa.bands)
        self.noise_added = True
        
    def truncate(self):
        start = self.bpa.expand_time_beginning
        end = self.bpa.expand_time_end
        if start is None and end is None:
            print("Audio clip not expanded; canceling truncate.")
        else:
            start = 0 if start is None else start
            end = 0 if end is None else end
            self.X = truncate_signal(self.X, self.bpa.sample_rate, start, end)
            self.envelopes = truncate_signal(self.envelopes, self.bpa.sample_rate, start, end)
            self.noises = truncate_signal(self.noises, self.bpa.sample_rate, start, end)
            self.peaks = truncate_signal(self.peaks, self.bpa.sample_rate, start, end)
    
    def multithread_features(self):
        sig_shape = self.bpa.filtered_signals.shape
        samples = max(sig_shape)
        self.envelopes = np.zeros(sig_shape)
        self.noises = np.zeros(sig_shape)
        self.peaks = np.zeros(sig_shape)
        
        signals = self.bpa.filtered_signals.copy()
        
        signals[signals < 0] = 0
        
        chunks = [[i, signals[i], self.sp_rise[i],
                self.sp_fall[i], self.nl_rise[i], self.nl_fall[i],
                self.bpa.sample_rate, samples, self.taus[i]] \
               for i in range(self.bpa.bands)]
        
        
        que = queue.Queue()
        #threads = [threading.Thread(target = npjitGradient,args = chunk) for chunk in chunks]
        threads = [threading.Thread(target = lambda q, args: \
                                 q.put(get_features(*args)),
                                 args = (que,chunk)) for chunk in chunks]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        while not que.empty():
            result = que.get()
            self.envelopes[result[0],:] = result[1]
            self.peaks[result[0],:] = result[2]
            self.noises[result[0],:] = result[3]
            
        self.X = self.peaks #- self.noises #self.peaks -  #np.vstack((self.peaks - self.noises, self.noises))

@njit(cache=True)
def get_features(band_num, signal, sp_rise, sp_fall,
            nl_rise, nl_fall, sr, samples, tau):
    env = np.zeros((samples,))
    pks = np.zeros((samples,))
    noi = np.zeros((samples,))
    track_sp = 0
    track_nl = 0
    for j in range(1, samples):
        #Get next envelope step
        if signal[j] >= env[j-1]: env[j] = signal[j]
        else: env[j] = signal[j] + (env[j-1] - signal[j])*math.exp(-1/sr/tau)

        # Get next signal peak step
        if env[j] >= track_sp:
            pks[j] = env[j]
            track_sp = track_sp + (1 - track_sp)*sp_rise/sr
            #if track_sp < 0:
            #    track_sp = 0
        else:
            track_sp = track_sp + (env[j] - track_sp)*sp_fall/sr
            pks[j] = track_sp

        # Get next noise level step
        if env[j] >= track_nl:
            track_nl = track_nl + (1 - track_nl)*nl_rise/sr
        else:
            track_nl = track_nl + (0 - track_nl)*nl_fall/sr
        if track_nl < 0:
                track_nl = 0
        noi[j] = track_nl
                
    return band_num, env, pks, noi

'''

def multithreadFeatures(signals,numfil,rates,vsr,lv,taus,x,z,c,d):
    #nthreads = multiprocessing.cpu_count()
    envelopes = np.zeros((numfil,lv))
    noises = np.zeros((numfil,lv))
    peaks = np.zeros((numfil,lv))
    
    signals[signals < 0] = 0
    
    chunks = [[signals[i],rates[i],vsr,lv,taus[i],x,z,c,d,i] for i in range(signals.shape[0])]
    
    que = queue.Queue()
    #threads = [threading.Thread(target = npjitGradient,args = chunk) for chunk in chunks]
    threads = [threading.Thread(target = lambda q, args: q.put(getFeat(*args)),args = (que,chunk)) for chunk in chunks]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    while not que.empty():
        result = que.get()
        envelopes[result[0],:] = result[2]
        noises[result[0],:] = result[4]
        peaks[result[0],:] = result[3]
        
    return signals, envelopes, peaks, noises

@njit(cache=True)
def getFeat(sig,rates,vsr,lv,taus,x,z,c,d,i):
    env = np.zeros((lv,))
    noi = np.zeros((lv,))
    pks = np.zeros((lv,))
    track_sp = 0
    track_nl = 0
    for j in range(1,lv):
        #Get next envelope step
        if sig[j] >= env[j-1]: env[j] = sig[j]
        else: env[j] = sig[j] + (env[j-1] - sig[j])*math.exp(-1/vsr/taus)

        # Get next signal peak step
        if env[j] >= track_sp:
            pks[j] = env[j]
            track_sp = track_sp + x*z*rates/vsr
            if track_sp < 0:
                track_sp = 0
        else:
            track_sp = track_sp - x*rates/vsr
            pks[j] = track_sp

        # Get next noise level step
        if env[j] <= track_nl:
            track_nl = track_nl - c*d*rates/vsr
            noi[j] = track_nl
            if track_nl < 0:
                track_nl = 0
        else:
            track_nl = track_nl + c*rates/vsr
            noi[j] = track_nl
                
    return i,sig,env,pks,noi
    
'''

def importAudio(directory):
    
    global vsr1; global voice1; global y1; global nsr1; global noise1; global yy1; global yyy1
    global vsr2; global voice2; global y2; global nsr2; global noise2; global yy2; global yyy2
    global vsr3; global voice3; global y3; global nsr3; global noise3; global yy3; global yyy3
    global vsr4; global voice4; global y4; global nsr4; global noise4; global yy4
    global vsr1t; global voice1t; global y1t; global nsr1t; global noise1t; global yy1t; global yyy1t
    global vsr2t; global voice2t; global y2t; global nsr2t; global noise2t; global yy2t; global yyy2t
    global vsr3t; global voice3t; global y3t; global nsr3t; global noise3t; global yy3t; global yyy3t
    global vsr4t; global voice4t; global y4t; global nsr4t; global noise4t; global yy4t
    
    print('Getting all audio data...')
    vsr1, voice1 = readwav(directory+"Voice/WAV/KidsTeensEnglish.wav")
    y1 = (RCenv(voice1,vsr1,0.01) > 0.005)
    nsr1, noise1 = readwav(directory + 'Noise/WAV/HomeNoise.wav')
    if vsr1 != nsr1: print('Sample rates do not match (1)')
    yy1 = onoff(y1,0.5,1000/vsr1,20/vsr1)
    yyy1 = solidLabel(y1+0,int(vsr1/4))
    
    vsr2, voice2 = readwav(directory + 'Voice/WAV/spokenFrench.wav')
    y2 = (RCenv(voice2,vsr2,0.01) > 0.005)
    nsr2, noise2 = readwav(directory + 'Noise/WAV/MowerRain.wav')
    if vsr2 != nsr2: print('Sample rates do not match (2)')
    yy2 = onoff(y2,0.5,1000/vsr2,20/vsr2)
    yyy2 = solidLabel(y2+0,int(vsr2/4))
    
    vsr3, voice3 = readwav(directory + 'Voice/WAV/spokenSpanish.wav')
    y3 = (RCenv(voice3,vsr3,0.01) > 0.005)
    nsr3, noise3 = readwav(directory + 'Noise/WAV/Crowd.wav') #'StormHome.wav')
    if vsr3 != nsr3: print('Sample rates do not match (3)')
    yy3 = onoff(y3,0.5,1000/vsr3,20/vsr3)
    yyy3 = solidLabel(y3+0,int(vsr3/4))
    
    nsr4, noise4 = readwav(directory + 'Noise/WAV/KitchenBeatsShort.wav')
    voice4 = np.zeros(len(noise4)); vsr4 = nsr4
    y4 = np.zeros(len(noise4))
    yy4 = np.zeros(len(noise4))
    
    vsr1t, voice1t = readwav(directory + 'Voice/WAV/spokenEnglish.wav')
    y1t = (RCenv(voice1t,vsr1t,0.01) > 0.005)
    nsr1t, noise1t = readwav(directory + 'Noise/WAV/Crowd.wav')
    if vsr1t != nsr1: print('Sample rates do not match (4)')
    yy1t = onoff(y1t,0.5,1000/vsr1t,20/vsr1t)
    yyy1t = solidLabel(y1t+0,int(vsr1/4))
    
    vsr2t, voice2t = readwav(directory + 'Voice/WAV/spokenFrench.wav')
    y2t = (RCenv(voice2t,vsr2t,0.01) > 0.005)
    nsr2t, noise2t = readwav(directory + 'Noise/WAV/Birdwasher.wav')
    if vsr2t != nsr2t: print('Sample rates do not match (5)')
    yy2t = onoff(y2t,0.5,1000/vsr2t,20/vsr2t)
    yyy2t = solidLabel(y2t+0,int(vsr2t/4))
       
    vsr3t, voice3t = readwav(directory + 'Voice/WAV/spokenSpanishAmplified.wav')
    y3t = (RCenv(voice3t,vsr3t,0.01) > 0.005)
    nsr3t, noise3t = readwav(directory + 'Noise/WAV/HoodwayFan.wav')
    if vsr3t != nsr3t: print('Sample rates do not match (6)')
    yy3t = onoff(y3t,0.5,1000/vsr3t,20/vsr3t)
    yyy3t = solidLabel(y3t+0,int(vsr3t/4))
    
    nsr4t, noise4t = readwav(directory + 'Noise/WAV/KitchenSounds2short.wav')
    voice4t = np.zeros(len(noise4t)); vsr4t = nsr4t
    y4t = np.zeros(len(noise4t))
    yy4t = np.zeros(len(noise4t))
    
    return vsr1, voice1, y1, nsr1, noise1, yy1, yyy1,\
        vsr2, voice2, y2, nsr2, noise2, yy2, yyy2,\
        vsr3, voice3, y3, nsr3, noise3, yy3, yyy3,\
        vsr4, voice4, y4, nsr4, noise4, yy4,\
        vsr1t, voice1t, y1t, nsr1t, noise1t, yy1t, yyy1t,\
        vsr2t, voice2t, y2t, nsr2t, noise2t, yy2t, yyy2t,\
        vsr3t, voice3t, y3t, nsr3t, noise3t, yy3t, yyy3t,\
        vsr4t, voice4t, y4t, nsr4t, noise4t, yy4t


def createFeatures(var, save = False, directory = '', i = 0, train_only = False, return_permute = True, noise_adjust = 1):
    
    global f0s
    global alpha
    global alpha2
    global lamb
    global a1
    global a2
    global b1
    global b2
    global n
    global fmx
    global fmn
    global o
    global q
    global numfil
    global fmax
    global fmin
    
    ## var = [f0s,taus,Qs,rates,filter gain]
    
    if var is None: var = np.ones(5*numfil)
    
    # Import audio data
    print('Getting data set 1...')
    X1,ya = getFeature(voice1,vsr1,noise1,nsr1,y1,alpha*noise_adjust,a1,a2,b1,b2,numfil,o,q,fmax,fmin,var)
    print('Getting data set 2...')
    X2,yb = getFeature(voice2,vsr2,noise2,nsr2,y2,alpha*noise_adjust,a1,a2,b1,b2,numfil,o,q,fmax,fmin,var)
    print('Getting data set 3...')
    X3,yc = getFeature(voice3,vsr3,noise3,nsr3,y3,alpha*noise_adjust,a1,a2,b1,b2,numfil,o,q,fmax,fmin,var)
    print('Getting noise set...')
    X4,yd = getFeature(voice4,vsr4,noise4,nsr4,y4,alpha*noise_adjust,a1,a2,b1,b2,numfil,o,q,fmax,fmin,var)
    
    if not train_only:
        print('Getting test set 1...')
        X1t,yat = getFeature(voice1t,vsr1t,noise1t,nsr1t,y1t,alpha2*noise_adjust,a1,a2,b1,b2,numfil,o,q,fmax,fmin,var)
        print('Getting test set 2...')
        X2t,ybt = getFeature(voice2t,vsr2t,noise2t,nsr2t,y2t,alpha2*noise_adjust,a1,a2,b1,b2,numfil,o,q,fmax,fmin,var)
        print('Getting test set 3...')
        X3t,yct = getFeature(voice3t,vsr3t,noise3t,nsr3t,y3t,alpha2*noise_adjust,a1,a2,b1,b2,numfil,o,q,fmax,fmin,var)
        print('Getting noise test set...')
        X4t,ydt = getFeature(voice4t,vsr4t,noise4t,nsr4t,y4t,alpha2*noise_adjust,a1,a2,b1,b2,numfil,o,q,fmax,fmin,var)
    
    if return_permute:
        print('Permutating Data...')
        
        # Shuffle data and generate training, validation, and test sets
        Xz = np.vstack([X1,X2,X3,X4]); yz = np.concatenate([ya,yb,yc,yd])
        X = np.array(Xz); y = np.array(yz)
        
        m = len(y)
        j = np.random.permutation(m)
        mn = round(m*n)
        X = X[j,:]; y = y[j]
        i1 = mn
        i2 = round((mn+m)/2)
        
        Xtrain = X[:i1,:]
        ytrain = y[:i1]
        Xval1 = X[i1:i2,:]
        yval1 = y[i1:i2]
        Xval2 = X[i2:,:]
        yval2 = y[i2:]
    
    # Set this to True to generate files
    if save:
        
        if not train_only:
            np.savez(directory+'ErgFeatures'+str(i)+'.npz',
                 Xtrain = Xtrain, ytrain = ytrain, Xval1 = Xval1, yval1 = yval1, Xval2 = Xval2, yval2 = yval2,
                   X1t = X1t, X2t = X2t, X3t = X3t, X4t = X4t, yat = yat, ybt = ybt, yct = yct, ydt = ydt)
        else: 
            np.savez(directory+'ErgFeatures'+str(i)+'.npz',
                 Xtrain = Xtrain, ytrain = ytrain, Xval1 = Xval1, yval1 = yval1, Xval2 = Xval2, yval2 = yval2)

    if return_permute:
        if not train_only:
            return Xtrain, ytrain, Xval1, yval1, Xval2, yval2, X1t, yat, X2t, ybt, X3t, yct, X4t, ydt
        else:
            return Xtrain, ytrain, Xval1, yval1, Xval2, yval2
    else:
        if not train_only:
            return X1, solidLabel(ya,int(vsr1/4)), X2, solidLabel(yb,int(vsr2/4)), X3, solidLabel(yc,int(vsr3/4)),\
                X4, solidLabel(yd,int(vsr4/4)), X1t, solidLabel(yat,int(vsr1t/4)), X2t, solidLabel(ybt,int(vsr2t/4)),\
                 X3t, solidLabel(yct,int(vsr3t/4)), X4t, solidLabel(ydt,int(vsr4t/4))
        else:
             return X1, solidLabel(ya,int(vsr1/4)), X2, solidLabel(yb,int(vsr2/4)),\
                 X3, solidLabel(yc,int(vsr3/4)), X4, solidLabel(yd,int(vsr4/4))


def getFeature(voice,vsr,noise,nsr,y,alpha,x,z,c,d,numfil,order,quality,fmax,fmin,var):
    #global file
    
    lowpass = True
    #Add silence before and after voice
    lv = len(voice)
    first_sb = 10*vsr
    second_sb = 10000
    voice = np.concatenate([np.zeros(first_sb),voice,np.zeros(second_sb)])
    y = np.concatenate([y,np.zeros(second_sb)])
    
    # If the lengths of the two audio files are different, change the length of the noise vector
    lv = len(voice); ln = len(noise)
    if lv != ln:
        noise = matchLen(noise,lv,ln)
    
    sig = (voice + alpha*noise)/(2**15)
    #write('Sample'+str(file)+'.wav',rate=vsr,data = sig)
    #file += 1
    
    # Generates filtered subband versions of the voice+noise signal
    
    ## Parameters
    tau = 0.025
    rmin = 0.001
    f0s = np.logspace(np.log10(fmin),np.log10(fmax),numfil)/vsr*2
    f0s = f0s*var[0:numfil]
    os = np.ones(numfil)*order
    taus = np.ones(numfil)*tau*var[numfil:2*numfil]
    for i in range(numfil): taus[i] = taus[i-1]*450*2/vsr/f0s[i]
    Qs = np.ones(numfil)*quality*var[2*numfil:3*numfil]
    rates = np.ones(numfil)*rmin*var[3*numfil:4*numfil]
    for i in range(1,numfil): rates[i] = rates[i-1]*f0s[i]/f0s[i-1]
    signals = np.zeros((numfil,lv))
    b,a = butter(os[0], f0s[0],'low', analog=False)
    signals[0,:] = lfilter(b,a, sig)*var[4*numfil]

    
    fstart = int(lowpass+0)
    for i in range(fstart,numfil):
        fl = f0s[i]*(math.sqrt(1+1/4/Qs[i]**2)-1/2/Qs[i])
        fh = f0s[i]*(math.sqrt(1+1/4/Qs[i]**2)+1/2/Qs[i])
        b,a = butter(os[i], [fl,fh],'band', analog=False)
        signals[i,:] = lfilter(b,a, sig)*var[4*numfil+i]
        
    #signals,envelopes,pks,noises = loopFeaturesPar(signals,numfil,rates,vsr,lv,taus,x,z,c,d)
    signals,envelopes,pks,noises = multithreadFeatures(signals,numfil,rates,vsr,lv,taus,x,z,c,d)

    
    X =  pks-noises #np.vstack((pks-noises, noises)) #
    #X = [pks;noises];
    #Xa = (pks-noises);
    #Xb = (envelopes-noises);
    #X = [Xa;Xb];
    X = X[:,10*vsr:].T
    
    return X,y

def multithreadFeatures(signals,numfil,rates,vsr,lv,taus,x,z,c,d):
    #nthreads = multiprocessing.cpu_count()
    envelopes = np.zeros((numfil,lv))
    noises = np.zeros((numfil,lv))
    peaks = np.zeros((numfil,lv))
    
    signals[signals <0] = 0
    
    chunks = [[signals[i],rates[i],vsr,lv,taus[i],x,z,c,d,i] for i in range(signals.shape[0])]
    
    que = queue.Queue()
    #threads = [threading.Thread(target = npjitGradient,args = chunk) for chunk in chunks]
    threads = [threading.Thread(target = lambda q, args: q.put(getFeat(*args)),args = (que,chunk)) for chunk in chunks]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    while not que.empty():
        result = que.get()
        envelopes[result[0],:] = result[2]
        noises[result[0],:] = result[4]
        peaks[result[0],:] = result[3]
        
    return signals, envelopes, peaks, noises

@njit(cache=True)
def getFeat(sig,rates,vsr,lv,taus,x,z,c,d,i):
    env = np.zeros((lv,))
    noi = np.zeros((lv,))
    pks = np.zeros((lv,))
    track_sp = 0
    track_nl = 0
    for j in range(1,lv):
        #Get next envelope step
        if sig[j] >= env[j-1]: env[j] = sig[j]
        else: env[j] = sig[j] + (env[j-1] - sig[j])*math.exp(-1/vsr/taus)

        # Get next signal peak step
        if env[j] >= track_sp:
            pks[j] = env[j]
            track_sp = track_sp + x*z*rates/vsr
            if track_sp < 0:
                track_sp = 0
        else:
            track_sp = track_sp - x*rates/vsr
            pks[j] = track_sp

        # Get next noise level step
        if env[j] <= track_nl:
            track_nl = track_nl - c*d*rates/vsr
            noi[j] = track_nl
            if track_nl < 0:
                track_nl = 0
        else:
            track_nl = track_nl + c*rates/vsr
            noi[j] = track_nl
                
    return i,sig,env,pks,noi

'''
def multicoreFeatures(signals,numfil,rates,vsr,lv,taus,rmin,x,z,c,d):
    
    envelopes = np.zeros((numfil,lv))
    noises = np.zeros((numfil,lv))
    peaks = np.zeros((numfil,lv))
    track_sp = np.zeros((numfil,))
    track_nl = np.zeros((numfil,))
    
    threadsperblock = (2, 2)
    blockspergrid_x = math.ceil(signals.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(signals.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    
    cudaLoopFeatures[blockspergrid, threadsperblock](signals,numfil,rates,vsr,lv,taus,x,z,c,d,track_sp,track_nl,envelopes,noises,peaks)
    
    return signals, envelopes, peaks, noises
    

@cuda.jit
def cudaLoopFeatures(signals,numfil,rates,vsr,lv,taus,x,z,c,d,track_sp,track_nl,envelopes,noises,peaks):
    x, y = cuda.grid(2)
    if x < signals.shape[0] and y < signals.shape[1]:
        if signals[x][y] >= envelopes[x][y-1]:
            envelopes[x][y] = signals[x][y]
        else:
            to = max(0,signals[x][y])
            envelopes[x][y] = to + (envelopes[x][y-1] - to)*math.exp(-1/vsr/taus[x])
    
        # Get next signal peak step
        if envelopes[x][y] >= track_sp[x]:
            peaks[x][y] = envelopes[x][y]
            track_sp[x] = track_sp[x] + x*z*rates[x]/vsr
            if track_sp[x] < 0:
                track_sp[x] = 0
        else:
            track_sp[x] = track_sp[x] - x*rates[x]/vsr
            peaks[x][y] = track_sp[x]
    
        # Get next noise level step
        if envelopes[x][y] <= track_nl[x]:
            track_nl[x] = track_nl[x] - c*d*rates[x]/vsr
            noises[x][y] = track_nl[x]
            if track_nl[x] < 0:
                track_nl[x] = 0
        else:
            track_nl[x] = track_nl[x] + c*rates[x]/vsr
            noises[x][y] = track_nl[x]


@njit(nopython = True, parallel=True)
def loopFeaturesPar(signals,numfil,rates,vsr,lv,taus,x,z,c,d):
    
    envelopes = np.zeros((numfil,lv))
    noises = np.zeros((numfil,lv))
    peaks = np.zeros((numfil,lv))
    
    
    for i in prange(numfil):
        sig = signals[i,:]
        env = envelopes[i,:]
        noi = noises[i,:]
        pks = peaks[i,:]
        track_sp = 0
        track_nl = 0
        for j in range(1,lv):
            #Get next envelope step
            if sig[j] >= env[j-1]:
                env[j] = sig[j]
            else:
                to = max([0,sig[j]])
                env[j] = to + (env[j-1] - to)*math.exp(-1/vsr/taus[i])
    
            # Get next signal peak step
            if env[j] >= track_sp:
                pks[j] = env[j]
                track_sp = track_sp + x*z*rates[i]/vsr
                if track_sp < 0:
                    track_sp = 0
            else:
                track_sp = track_sp - x*rates[i]/vsr
                pks[j] = track_sp
    
            # Get next noise level step
            if env[j] <= track_nl:
                track_nl = track_nl - c*d*rates[i]/vsr
                noi[j] = track_nl
                if track_nl < 0:
                    track_nl = 0
            else:
                track_nl = track_nl + c*rates[i]/vsr
                noi[j] = track_nl
                
    return signals,envelopes,peaks,noises


#@njit(nopython=True,parallel=True)
def loopFeatures(signals,numfil,rates,vsr,lv,tau,x,z,c,d):
    
    envelopes = np.zeros((numfil,lv))
    noises = np.zeros((numfil,lv))
    pks = np.zeros((numfil,lv))
    
    for i in range(numfil):
        envelopes[i,:],pks[i,:],noises[i,:] = AllFeatures(signals[i,:],lv,vsr,tau[i],rates[i],x,z,c,d)
        
    return signals,envelopes,pks,noises

@njit(nopython=True,parallel=True)
def AllFeatures(sig,lv,sr,tau,rate,a1,a2,b1,b2):
    env = np.zeros(lv)
    
    peak = np.zeros(lv)
    track_sp = 0
    
    noise = np.zeros(lv)
    track_nl = 0
    
    for i in range(1,lv):
        #Get next envelope step
        if sig[i] >= env[i-1]:
            env[i] = sig[i]
        else:
            to = max([0,sig[i]])
            env[i] = to + (env[i-1] - to)*math.exp(-1/sr/tau)

        # Get next signal peak step
        if env[i] >= track_sp:
            track_sp = track_sp + a1*a2*rate/sr
            if track_sp < 0:
                track_sp = 0
        else:
            track_sp = track_sp - a1*rate/sr
            peak[i] = track_sp

        # Get next noise level step
        if env[i] <= track_nl:
            track_nl = track_nl - b1*b2*rate/sr
            noise[i] = track_nl
            if track_nl < 0.0001:
                track_nl = 0.0001
        else:
            track_nl = track_nl + b1*rate/sr
            noise[i] = track_nl
    
    return env, peak, noise
'''

@njit
def RCenv(sig,sr,tau):
    lv = len(sig)
    env = np.zeros(lv)
    for i,v in enumerate(sig[1:]):
        if v >= env[i-1]:
            env[i] = v
        else:
            to = max([0,v])
            env[i] = to + (env[i-1] - to)*math.exp(-1/sr/tau)
    #for i in range(1,lv):
    #    if sig[i] >= env[i-1]:
    #        env[i] = sig[i]
    #    else:
    #        to = max([0,sig[i]])
    #        env[i] = to + (env[i-1] - to)*math.exp(-1/sr/tau)
    return env


def matchLen(noise,lv,ln):
    if lv > ln:
        a = math.floor(lv/ln)
        b = lv%ln
        temp = np.tile(noise,a)
        return np.concatenate([temp,noise[:b]])
    elif lv < ln:
        return noise[:lv]

@njit(parallel=True)
def confusionMat(yact,ypred):
    m = len(yact)
    results = np.zeros((m,5))
    # tn = 1, fp = 2, fn = 3, tp = 4
    # tn = 0; fp = 0; fn = 0; tp = 0
    for i in prange(m):
        if yact[i]:
            if ypred[i]: results[i,4] = 1 #tp
            else: results[i,3] = 1 #fn
        elif ypred[i]: results[i,2]= 1 #fp
        else: results[i,1] = 1 #tn

    results = np.sum(results,axis=0)/m*100
    results[0] = results[1]+results[4]
    #tp = tp/m*100
    #fn = fn/m*100
    #tn = tn/m*100
    #fp = fp/m*100
    return results #np.array([tn+tp,tn,fp,fn,tp])

@njit
def forwardSweep(label,view):
    labelout = label+0
    i = 0
    n = len(label)
    while i < n:
        if labelout[i]:
            if np.sum(labelout[i+1:i+view]) > 1:
                labelout[i:i+view] = 1
                i += view-2
        i += 1
    return labelout

@njit
def backSweep(label,view):
    labelout = label+0
    i = -1
    n = len(label)
    while i >= -n:
        if labelout[i]:
            if np.sum(labelout[i-view:i-1]) > 1:
                #print(len(labelout[i-view:i]))
                #print(len(ones))
                labelout[i-view:i] = 1
                i -= view-2
        i -= 1
    return labelout

def solidLabel(label,view):
    view = round(view)
    lab1 = forwardSweep(label,view)
    lab2 = backSweep(label,view)
    return lab1*lab2

@njit
def onoff(ybits,vth,rrise,rfall):
    vcap = 0
    yout = np.zeros(len(ybits))==1
    for i,y in enumerate(ybits):
        if y:
            vcap += rrise
            if vcap > 1: vcap = 1
        else:
            vcap -= rfall
            if vcap < 0: vcap = 0
        if vcap > vth: yout[i] = True
    return yout
