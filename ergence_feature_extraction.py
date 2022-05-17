import math
from scipy.signal import butter,lfilter
import numpy as np
from numba import njit, prange
import threading
import queue
from scipy.io.wavfile import read as readwav

import os
import random
import csv
import torch
import matplotlib.pyplot as plt

### Global Vairables
MS_DNS_AUDIO_DIR = "/home/djanke3/Documents/Audio/DNS-Challenge/"
GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory

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
            
    def load_split(self, file = "/home/djanke3/Documents/Spyder/data_split.csv"):
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
            
    def split(self, data, tot_files = None):
        n = len(data)
        if tot_files is None:
            tot_files = n
        if self.n is None or self.permutation is None or self.n != n:
            self.generate_permutation(n)
            
        tr, v, te = self.raw_split()
        
        train_perm = [data[i] for i in tr][:int(self.ratios[0] * tot_files)]
        val_perm = [data[i] for i in v][:int(self.ratios[1] * tot_files)]
        test_perm = [data[i] for i in te][:int(self.ratios[2] * tot_files)]
        
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
            y = np.array(RCenv(voice,vsr,0.01) > 0.01, dtype = bool)
            tot += len(y)
            tot_sum += y.sum()
            fileid = c.split("fileid_")[1][:-4] + ".npy"
            np.save(MS_DNS_AUDIO_DIR + "labels/label_" + fileid, y)
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
    def __init__(self, audio, sample_rate, bands = 6, filterQ = 0.7,
                 filter_order = 2, fmin = 120, fmax = 1900, lowpass = True):
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
        self.f0s = np.clip(self.f0s, 0, 0.5)
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
    def __init__(self, bp_array, envelope_tau = 0.025,
                 decay_rate_unit = 1, sp_decay_rise = 1, sp_decay_fall = 0.5,
                 nl_decay_rise = 0.1, nl_decay_fall = 1, spread = True):
        
        self.settings = {"tau" : envelope_tau,
                         "decay_unit" : decay_rate_unit,
                         "sp_rise" : sp_decay_rise,
                         "sp_fall" : sp_decay_fall,
                         "nl_rise" : nl_decay_rise,
                         "nl_fall" : nl_decay_fall}
        self.spread = spread
        
        self.bpa = bp_array
        self.initialize_parameters()
        
    def initialize_parameters(self):
        if self.spread:
            sr450 = 450 / 2 / self.bpa.sample_rate # 450 Hz normalized to sample_rate
            self.taus = np.array([self.settings["tau"] * sr450 / self.bpa.f0s[i] \
                         for i in range(self.bpa.bands)])
            self.rates = np.array([self.settings["decay_unit"] / sr450 * self.bpa.f0s[i] \
                          for i in range(self.bpa.bands)])
        else:
            self.taus = np.array([self.settings["tau"] for i in range(self.bpa.bands)])
            self.rates = np.array([self.settings["decay_unit"] for i in range(self.bpa.bands)], dtype = 'float64')
        
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
    
    def multithread_features(self, features = ["diff", "n"]):
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
            
        X = []
        if "e" in features: X = [self.envelopes]
        if "p" in features: X.append(self.peaks)
        if "n" in features: X.append(self.noises)
        if "diff" in features: X.append(self.peaks - self.noises)
            
        self.X = np.vstack(X)

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


class AudioFeatureGenerator:
    def __init__(self, train_split, total_files = None, load_to_memory = False,
                 fa_params = {}, bpa_params = {}, features = ["diff", "n"],
                 data_fraction = 1):
        
        if total_files is None:
            self.tf = len(self.ad.noisy_files)
        else:
            self.tf = total_files
            
        self.ltm = load_to_memory
        self.data_fraction = data_fraction
        self.features = features
        
        self.ad = AudioData()
        self.ds = DataSplitter(*train_split)
        self.split_data()
        self.create_fa(bpa_params, fa_params)
        
        self.initialize()
        
    def initialize(self):
        #self.calc_pos_weight()
        self.num_batches = {}
        self.batch_size = {}
        
        sets = ["train", "val", "test"]
        for s in sets:
            self.calc_batch_size(s)
            
        if self.ltm:
            self.X = {s:[] for s in sets}
            self.y = {s:[] for s in sets}
            self.Xcombo = {s:[] for s in sets}
            self.ycombo = {s:[] for s in sets}
            self.nrand = {}
        self.frac_combo = False
            
    def split_data(self):
        try: self.ds.load_split()
        except:
            self.ds.generate_permutation()
            self.ds.save_split()
        
        train_files, val_files, test_files = \
            tuple([d for d in self.ds.split(self.ad.noisy_files, self.tf)])
            
        self.audio_files = {"train" : train_files,
                            "val" : val_files,
                            "test" : test_files}
            
        if self.ad.labels is not None:
            train_label_files, val_label_files, test_label_files = \
                tuple([d for d in self.ds.split(self.ad.labels, self.tf)])
                
            self.label_files= {"train" : train_label_files,
                               "val" : val_label_files,
                               "test" : test_label_files}
                
    #def create_bpa(self, bpa_params):
    #    sr, audio = self.ad.read_audio(self.ad.noisy_dir + self.audio_files["train"][0])
    #    self.bpa = BandpassArray(audio, sr, **bpa_params)
        
    def create_fa(self, bpa_params, fa_params):
        sr, audio = self.ad.read_audio(self.ad.noisy_dir + self.audio_files["train"][0])
        bpa = BandpassArray(audio, sr, **bpa_params)
        self.fa = FeatureArray(bpa, **fa_params)
    
    @staticmethod
    def calc_pos_weight(labels):
        return sum(labels) / len(labels)
                
    def calc_batch_size(self, data_set = "train"):
        _, X, labels = next(self.audio_generator(data_set))
        feature_label_bytes = X.nbytes + labels.nbytes
        total_files = len(self.audio_files[data_set])
        total_bytes = total_files * feature_label_bytes
        
        num_batches = int(total_bytes / (GPU_MEMORY * 0.2)) + 1
        i = 0
        while (total_files % (num_batches + i) != 0): i += 1
        num_batches += i
        num_batches = 1
        self.num_batches[data_set] = num_batches
        self.batch_size[data_set] = int(total_files / num_batches)
        self.batch_samples = self.batch_size[data_set] * len(labels)
        
    def audio_generator(self, data_set = "train"):
        data_zip = zip(self.audio_files[data_set], self.label_files[data_set])
        for i,(af, lf) in enumerate(data_zip):
            labels = np.load(self.ad.label_dir + lf)
            sr, audio = self.ad.read_audio(self.ad.noisy_dir + af)
            
            self.fa.bpa.audio = audio
            self.fa.bpa.expand(5, 1)
            self.fa.bpa.multithread_filter()
            self.fa.multithread_features(self.features)
            self.fa.truncate()
            #plt.plot(self.fa.X.T)
            #plt.show()
            #input()
            yield (i, self.fa.X, labels)
            
    def batch_audio_generator(self, data_set = "train", mode = "eval", save=False, load=False,
                              new_random = True):
        Xtemp = None
        ytemp = None
        j = 0
        
        if load and len(self.X[data_set]) == 0:
            try:
                self.X[data_set] = np.load("X{}.npy".format(data_set))
                self.y[data_set] = np.load("y{}.npy".format(data_set))
            except:
                print("failed to load")
            
        
        if self.ltm and len(self.X[data_set]) != 0:
            for i, (Xi, yi) in enumerate(zip(self.X[data_set], self.y[data_set])):
                if (self.data_fraction != 1 \
                    and (self.nrand.get(data_set) is None or new_random)):
                    print("generate nrand")
                    n = int(self.data_fraction * Xi.shape[0])
                    self.nrand[data_set] = random.sample(range(Xi.shape[0]), n)
                elif self.data_fraction == 1:
                    self.nrand[data_set] = list(np.arange(Xi.shape[0]))
            
                nrand = self.nrand[data_set]
                Xfrac = Xi[nrand, :]
                yfrac = yi[nrand]
                
                if self.frac_combo:
                    self.Xcombo[data_set].append(Xfrac)
                    self.ycombo[data_set].append(yfrac)
                    continue
                
                if (mode == "eval"):
                    yield (i, Xfrac, yfrac, Xi, yi)
                else:
                    yield (i, Xfrac, yfrac)
                    
            if self.frac_combo:
                yield (0, np.concatenate(self.Xcombo[data_set], axis = 0),
                       np.concatenate(self.ycombo[data_set]))
                self.Xcombo[data_set] = []
                self.ycombo[data_set] = []
        else:
            print("Generating {} feature batches for {} set:"\
                  .format(self.num_batches[data_set], data_set), end = " ")
            for i, X, y in self.audio_generator(data_set):
                if Xtemp is None:
                    Xtemp = X
                    ytemp = y
                else:
                    Xtemp = np.append(Xtemp, X, axis = 1)
                    ytemp = np.append(ytemp, y)
                
                #print(Xtemp.nbytes, ytemp.nbytes)
                if (i+1) % self.batch_size[data_set] == 0 \
                    or (i + 1 == len(self.audio_files[data_set])):
                    j += 1
                    print(j, end = " ")
                    
                    #random sample data for training
                    #if (data_set not in self.nrand):
                    if (self.data_fraction != 1):
                        n = int(self.data_fraction * Xtemp.shape[1])
                        self.nrand[data_set] = random.sample(range(Xtemp.shape[1]), n)
                    else: self.nrand[data_set] = list(np.arange(Xtemp.shape[1]))
                    
                    nrand = self.nrand[data_set]
                    #print(Xtemp, ytemp[nrand])
                    
                    if self.ltm:
                        self.X[data_set].append(Xtemp.T)
                        self.y[data_set].append(ytemp.T)
                        np.save("X{}all.npy".format(data_set), self.X[data_set])
                        np.save("y{}all.npy".format(data_set), self.y[data_set])
                        if save:
                            np.save("X{}.npy".format(data_set), self.X[data_set])
                            np.save("y{}.npy".format(data_set), self.y[data_set])
                        
                        self.frac_combo = self.num_batches[data_set] * self.data_fraction < 1 \
                            and mode == "train"
                        if self.frac_combo:
                            self.Xcombo[data_set].append(Xtemp[:, nrand].T)
                            self.ycombo[data_set].append(ytemp[nrand].T)
                    
                    if not self.frac_combo:
                        if (mode == "eval"):
                            yield (j-1, Xtemp[:, nrand].T, ytemp[nrand].T, Xtemp.T, ytemp.T)
                        else:
                            yield (j-1, Xtemp[:, nrand].T, ytemp[nrand].T)
                    
                    
                    Xtemp = None
                    ytemp = None
            if self.frac_combo:
                yield (0, np.concatenate(self.Xcombo[data_set], axis = 0),
                       np.concatenate(self.ycombo[data_set]))
                self.Xcombo[data_set] = []
                self.ycombo[data_set] = []
            print()


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
