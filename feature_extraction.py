
import ergence_feature_extraction as erg
import matplotlib.pyplot as plt
import numpy as np
import shutil

# Get all audio data
## Initialize locations

ad = erg.AudioData()

ds = erg.DataSplitter(0.6, 0.2, 0.2)
try: ds.load_split()
except:
    ds.generate_permutation()
    ds.save_split()

train_files, val_files, test_files = ds.split(ad.noisy_files)
if ad.labels is not None:
    train_label_files, val_label_files, test_label_files = ds.split(ad.labels)


noisy_features = np.zeros((18, 48000, 1000))
for i,tf in enumerate(train_files[:10]):
    #if i > 0: break
    print(tf)
    #shutil.copyfile(ad.noisy_dir + tf, "FeatureNoiseAnalysis/Audio{}".format(i) + tf)
    #continue
    labels = np.loadtxt(ad.label_dir + train_label_files[i], delimiter = ',')
    sr, audio = ad.read_audio(ad.noisy_dir + tf)
    
    #Plot the audio data
    #plt.plot(audio)
    #plt.ylim(-1, 1)
    #plt.savefig("FeatureNoiseAnalysis/audio_{}.png".format(i),
    #                        bbox_inches = "tight")
    #plt.clf()
    #continue
    
    bpa = erg.BandpassArray(audio = audio, sample_rate = sr, bands = 6)
    bpa.expand(5, 5)
    fa = erg.FeatureArray(bpa, envelope_tau = 0.25, decay_rate_unit = 1,
                          sp_decay_rise = 1, sp_decay_fall = 1,
                          nl_decay_rise = 0.1, nl_decay_fall = 1)
    
    fa.bpa.multithread_filter()
    fa.multithread_features()
    fa.truncate()
    
    clean_features = np.vstack((fa.envelopes[:,::10],
                                             fa.peaks[:,::10],
                                             fa.noises[:,::10]))
    np.save("clean_features{}.npy".format(i), clean_features, allow_pickle = False)
    
    #continue
    
    for j in range(1000):
        #fa.bpa.add_filter_noise(0.2/3)
        fa.bpa.multithread_filter()
        #fa.add_feature_noise(0.2/3)
        fa.multithread_features()
        fa.truncate()
        '''
        #plt.plot(bpa.filtered_signals.T)
        plt.plot(fa.envelopes[0].T)
        plt.plot(fa.peaks[0].T)
        plt.plot(fa.noises[0].T)
        #plt.plot(labels)
        plt.show()
        input()
        '''
        noisy_features[:,:,j] = np.vstack((fa.envelopes[:,::10],
                                             fa.peaks[:,::10],
                                             fa.noises[:,::10]))
    np.save("noisy_features{}.npy".format(i), noisy_features, allow_pickle = False)