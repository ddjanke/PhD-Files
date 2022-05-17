import pandas as pd
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import ergence_feature_extraction as erg
import torch
import json
import seaborn as sns

sns.set_style(rc={'font.family':"Times New Roman", 'font.size': 12, 'dpi': 300})

# Initial settings
train_split = (0.6, 0.2, 0.2)
total_files = 400
GPU_MEMORY = torch.cuda.get_device_properties(0).total_memory
features = ["diff", "n"]
bands = 6
num_features = len(features) * bands

dataset = ["voice", "power"][1]
activation = "CompTanh"
save_file = "SparsityTest_" + dataset + activation + "_{}full_HL{}_{}_.json"
sparsity = 3

with open("/home/djanke3/Documents/computer_identity.json", 'r') as ci:
    computer_identity = json.load(ci)

# List of learned parameters
fa_params = {"envelope_tau" : 0.025,
             "sp_decay_rise" : 25,
             "sp_decay_fall" : 2,
             "nl_decay_rise" : 13.5,
             "nl_decay_fall" : 45.5,
             "spread" : False}

bpa_params = {"bands" : bands}

kwargs = {"features" : features,
          "total_files" : total_files,
          "load_to_memory" : True,
          "fa_params" : fa_params,
          "bpa_params" : bpa_params}

efg = erg.AudioFeatureGenerator(train_split, data_fraction = 0.01, **kwargs)
    
print("Generating Audio:")
for g in efg.batch_audio_generator(mode = "train", data_set = "train", save=False, load=False):
    _,X,y = g
    break
'''
df = pd.read_csv("datasets/electrical_grid_stability.csv")
y = df["stabf"]
y = (y == "unstable").astype(int)

X = df.drop(["stab", "stabf"], axis = 1)
'''
X = pd. DataFrame(X[:3200])

Xcorr = X.T.corr()
print(Xcorr.shape)

plt.figure(figsize=(3,2.5))

plt0 = plt.pcolor(Xcorr.to_numpy()[:1600,:1600], cmap="twilight")
plt.colorbar()
plt.ylabel("Audio Features From Sample $n$", labelpad = 0, fontsize=12)
plt.xlabel("Audio Features From Sample $n$", labelpad = 0, fontsize=12)
fig = plt0.get_figure()
figname = "AudioFeatureCorrelation.png"
#fig.savefig(figname, dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(3,2.5))

Xdist = cdist(X.to_numpy(),X.to_numpy())
Xdist = pd.DataFrame(Xdist)
print(Xdist.shape)

plt0 = plt.pcolor(Xdist.to_numpy()[:1600,:1600])
plt.colorbar()
plt.ylabel("Audio Features From Sample $n$", labelpad = 0, fontsize=12)
plt.xlabel("Audio Features From Sample $n$", labelpad = 0, fontsize=12)
fig = plt0.get_figure()
figname = "AudioFeatureDistance.png"
#fig.savefig(figname, dpi=300, bbox_inches='tight')
plt.show()