import numpy as np
import matplotlib.pyplot as plt
import gc

import torch
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import ergence_feature_extraction as erg

def count_synapses(nodes):
    tot = 0
    for i,n in enumerate(nodes[:-1]):
        tot += (n+1)*nodes[i+1]
    return tot

def compressed_relu(x, gain = 1):
    return F.relu(x)*gain

class CompressedTanh(torch.nn.Tanh):
    def __init__(self, offset = 0, max_range = 2, compression = 1, inplace = False):
        super(CompressedTanh, self).__init__()
        self.max_range = max_range
        self.offset = offset
        self.compression = compression
    
    def forward(self,x):
        return (self.max_range/2)*torch.tanh(x*self.compression) + self.offset

class ANNSparse(torch.nn.Module):
    def __init__(self, nodes, sparsity = 1, device = 'cuda', activation = 'Tanh',
                 sparse_randomize = False, graft = False,
                 strategy = "each", mask_epochs = 800, prune_method = None,
                 mask_rate = 40, Nmasks = 5, mask_start = 0, soft_power = 0):
        
        super(ANNSparse,self).__init__()
        self.nodes = nodes
        self.synapses = count_synapses(nodes)
        self.activation = activation
        self.device = device
        
        self.sparsity = sparsity
        self.strategy = strategy
        self.Nmasks = Nmasks
        self.mask_rate = mask_rate
        self.mask_start = mask_start
        self.sparse_randomize = sparse_randomize
        self.graft = graft
        self.mask_epochs = mask_epochs
        self.soft_power = soft_power
        
        self.es = myEarlyStop(patience = 100, delta = 0.0001, mode = "loss")
        self.mask_stop = MaskStop(self.Nmasks, mode = "loss")
        self.initialize_model()
        self.set_sparse_params(mask_epochs, prune_method, mask_rate)
        
    def initialize_model(self):
        layers = []
        self.linear_layers = []
        self.activation_layers = []
        for i,ns in enumerate(self.nodes[1:]):
            if self.sparsity == 1:
                layers += [Linear_wVoltages(self.nodes[i], ns, layer = i,
                                            bias = True, device = self.device)]
            else:
                layers += [Linear_wVoltages_Sparse(self.nodes[i], ns, layer = i,
                                                  bias = True, device = self.device,
                                                  sparsity = self.sparsity)]
            self.linear_layers.append(2 * i)
            if i < len(self.nodes) - 2:
                self.activation_layers.append(2 * i + 1)
                layers += [self.get_activation()]
            
        #layers[-1] = CompressedTanh(0.5, 1, 1)
        #layers = layers[:-3] #+ [torch.nn.Sigmoid()]
                
        self.model = torch.nn.Sequential(*layers).to(self.device)
        self.initialize_weights()
        self.total_epochs = 0
        
    def get_activation(self):
        if self.activation == 'Tanh':
            self.threshold = 0.5
            return CompressedTanh() # CompressedTanh(0,0.2,1)
        elif self.activation == "CompTanh":
            self.threshold = 0.3
            return CompressedTanh(0, 1, 0.5) # 
        else:
            self.threshold = 0.05
            return CompressedTanh(0, 0.5, 0.25) # 
        
    def set_activation(self, activation):
        self.activation = activation
        for al in self.activation_layers:
            self.model[al] = self.get_activation()
        
    def initialize_weights(self):
        def initialize(m):
            if 'weight' in m.state_dict():
                torch.nn.init.xavier_uniform_(m.weight, gain = 1)
                m.bias.data.fill_(0.001)
        self.model.apply(initialize)
        
    def save_initial_state(self):
        self.initial_state = self.model.state_dict()
        
    def load_initial_state(self):
        self.model.load_state_dict(self.initial_state)
            
    def set_sparse_params(self, mask_epochs, prune_method = None,
                          mask_rate = 40):
        
        def apply_sparse_params(m):
            if 'weight' in m.state_dict():
                m.strategy = self.strategy
                m.mask_epochs = mask_epochs
                m.initialize_mask()
                if self.strategy in ['all', 'once', 'allsoft']:
                    m.mask_start = self.mask_start
                    m.mask_rate = mask_rate
                    m.Nmasks = self.Nmasks
                    m.soft_power = self.soft_power
        
        #self.mask_stop = myEarlyStop(self.Nmasks, mode = "loss")
        
        if self.sparse_randomize:
            self.random_sparse()
            self.mask_stop.stop = True
            return
        
        if prune_method == "parallel" or self.strategy in ['all', 'once', 'allsoft']:
            self.model.apply(apply_sparse_params)
        else:
            prune_per_layer = [max(n - self.sparsity, 0) for n in self.nodes[:-1]]
            prune_total = sum([max(n - self.sparsity, 0) for n in self.nodes[:-1]])
            prune_rate = max(mask_epochs // prune_total, 1)
            if prune_method == "backward":
                prune_starts = [prune_rate * sum(prune_per_layer[i:]) for i in range(1,len(self.nodes))]
            else:
                prune_starts = [prune_rate * sum(prune_per_layer[:i]) for i in range(0,len(self.nodes)-1)]
                #print(prune_starts)
        
            for i,l in enumerate(self.linear_layers):
                self.model[l].strategy = self.strategy
                self.model[l].mask_rate = prune_rate
                self.model[l].mask_start = prune_starts[i]
        
    def random_sparse(self):
        for l in self.linear_layers:
            self.model[l].random_mask()
            self.model[l].mask_rate = float('inf')
            
    def get_mask(self):
        if self.sparsity == 1: return []
        masks = []
        for l in self.linear_layers:
            masks.append(self.model[l].mask)
        return masks
            
    def set_mask(self, masks):
        if self.sparsity == 1: return
        for i,l in enumerate(self.linear_layers):
            assert masks[i].shape == self.model[l].mask.shape, "Mask shape does not match layer {}, {}".format(masks[i].shape, (self.model[l].shape[0], self.model[l].shape[0] - 1))
            self.model[l].mask = masks[i]
            
    @staticmethod
    def mask_equal(mask1, mask2):
        if len(mask1) != len(mask2): return False
        return all([np.array_equal(m1,m2) for m1,m2 in zip(mask1,mask2)])
        
    def back_prune(self):
        keep_rows = torch.tensor([True])
        i = -2
        for l in self.linear_layers[::-1]:
            m = self.model[l]
            if not all (keep_rows):
                self.model[l].mask = m.mask[keep_rows, :]
                with torch.no_grad():
                    self.model[l].weight = Parameter(m.weight[keep_rows, :])
                    self.model[l].bias = Parameter(m.bias[keep_rows])
                self.model[l].shape = (m.weight.shape[0], m.weight.shape[1] + 1)
                
            keep_rows = m.mask.sum(axis = 0) > 0
            
            if not all(keep_rows) and l != 0:
                m = self.model[l]
                self.model[l].mask = m.mask[:, keep_rows]
                with torch.no_grad():
                    self.model[l].weight = Parameter(m.weight[:, keep_rows])
                self.model[l].shape = (m.weight.shape[0], m.weight.shape[1] + 1)
                self.model[l].calc_rate()
                self.nodes[i] = m.weight.shape[1]
                i -= 1
                
        self.synapses = count_synapses(self.nodes)
                
                
    def back_connect(self):
        keep_rows = torch.tensor([True])
        for l in self.linear_layers[::-1]:
            m = self.model[l]
            mask_sum = m.mask.sum(axis = -1)
            if not torch.equal((mask_sum > 0), keep_rows):
                m.mask[keep_rows == False, :] = 0
                mask_sum[keep_rows == False] = 0
                prev = m.mask.sum(axis = 0)

                while not all(prev > 0):
                    connect_row = torch.argmin(torch.tensor([s if s > 0 else float('inf') for s in mask_sum]))
                    #connect_row = np.random.choice(np.argwhere(mask_sum.view(-1) > 0)[0])
                    #m.mask[randrom_row, torch.argmin(prev.type(torch.ByteTensor))] = 1
                    prev_min = torch.argmin(prev.type(torch.ByteTensor))
                    m.mask[connect_row, prev_min] = 1
                    with torch.no_grad():
                        m.weight[connect_row, prev_min] = 0
                    mask_sum[connect_row] += 1
                    prev = m.mask.sum(axis = 0) > 0
                self.model[l].mask = m.mask
            keep_rows = m.mask.sum(axis = 0) > 0
        
        
    def forward(self,X,device = None):
        if self.sparsity != 1:
            if not self.sparse_randomize and self.graft:
                self.back_connect()
        if device is None: device = self.device
        if type(X) is np.ndarray:
            X = torch.as_tensor(X, dtype = torch.float32).to(device)
        y = self.model.forward(X).squeeze()

        return y
    
    
    def fit(self, generator, epochs, batch_size = 1, verbose = False,
            generator_kwargs = {"mode" : "train"},
            weight_decay = 0, learning_rate = 0.001,
            early_stop_train = False, early_stop_eval = False):
        
        early_stop = early_stop_train or early_stop_eval
        self.model = self.model.to(self.device)
        
        optimizer = torch.optim.RMSprop(self.model.parameters(), lr = 0.002,
                                        weight_decay = weight_decay)
        if early_stop_eval:
            generator_kwargs["mode"] = "eval"
        
        self.model.train()
        
        for epoch in range(epochs):
            self.total_epochs += 1
            acc = 0
            for g in generator(**generator_kwargs):
                if early_stop_eval: i, Xi, yi, Xv, yv = g
                else: i, Xi, yi = g
                pos_weight = erg.AudioFeatureGenerator.calc_pos_weight(yi)
                if yi.max() == 1:
                    criterion = torch.nn.BCEWithLogitsLoss(pos_weight = torch.tensor(pos_weight,
                                                   dtype = torch.float32))
                    
                    yi = torch.as_tensor(yi, dtype = torch.float32).to(self.device)
                else:
                    criterion = torch.nn.CrossEntropyLoss()
                    yi = torch.as_tensor(yi, dtype = torch.long).to(self.device)
                Xi = torch.as_tensor(Xi, dtype = torch.float32).to(self.device)
                y_pred = self.forward(Xi)
                loss = criterion(y_pred, yi)
                loss = loss / batch_size
                loss.backward()
                
                
                del Xi
                #del yi
                #del y_pred
                gc.collect()
                
            
            if self.strategy in ["all", "once", "allsoft"]:
                if self.strategy == "all" \
                        and self.total_epochs >= self.mask_start \
                        and (self.total_epochs + 0) % self.mask_rate == 0 \
                        and not self.mask_stop.stop:
                    
                    self.mask_stop(loss.item(), self.model, self.get_mask())
                    
                    if self.mask_stop.stop:
                        self.model.load_state_dict(self.mask_stop.es_dict)
                        for l in self.linear_layers:
                            self.model[l].mask_from_weight()
                        print("mask stop")
                elif self.strategy == "once" and self.total_epochs == self.mask_start:
                    self.mask_stop.stop = True
                    print("mask stop")
                   
            elif self.total_epochs == self.mask_epochs:
                self.mask_stop.stop = True
                print("mask stop")
            
            ### This may need to be moved back into the end of the for loop 
            if (i+1) % batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            if verbose > 0:
                if (self.total_epochs) % verbose == 0:
                    print('Epoch {} train loss: {}'.format(self.total_epochs, loss.item()))
                    
            if early_stop:
                self.es(loss.item(), self.model)
                
            if self.es.stop: # or ((epoch + 1) == epochs and early_stop):
                self.model.load_state_dict(self.es.es_dict)
                #print(" Exiting training: early stop triggered.")
                break
            
                    
        return loss.item()
                    
                    
    def population_fit(self, generator, epochs, population = 1, verbose = False,
                        generator_kwargs = {"mode" : "train"}, average_gradient = True,
                        weight_decay = 0, learning_rate = 0.001,
                        early_stop_train = False, feature_noise = None,
                        feature_noise_params = [1, 0.1],
                        noise_at_input = False):
        
        if feature_noise is None: feature_noise = np.ones((self.nodes[0],))
        if type(feature_noise) is bool:
            if feature_noise == True: feature_noise = np.array([1,1])
            else: feature_noise = np.ones((self.nodes[0],))
            
        if feature_noise.shape[0] != self.nodes[0]:
            feature_noise = np.random.normal(loc = feature_noise_params[0],
                                                     scale = feature_noise_params[1],
                                                     size = self.nodes[0])
            
            
        self.model = self.model.to(self.device)
        
        optimizer = torch.optim.RMSprop(self.model.parameters(), lr = learning_rate,
                                        weight_decay = weight_decay)
        
        self.model.train()
        #print(self.model.state_dict())
        for epoch in range(epochs):
            total_loss = 0
            for p in range(population):
                if noise_at_input:
                    self.add_input_noise()
                else:
                    self.add_noise()
                for g in generator(**generator_kwargs):
                    i, Xi, yi = g
                    pos_weight = erg.AudioFeatureGenerator.calc_pos_weight(yi)
                    criterion = torch.nn.BCEWithLogitsLoss(pos_weight = torch.tensor(pos_weight,
                                               dtype = torch.float32))
                    
                    y_pred = self.predict(Xi * feature_noise)
                    
                    Xi = torch.as_tensor(Xi, dtype = torch.float32).to(self.device)
                    yi = torch.as_tensor(yi, dtype = torch.float32).to(self.device)
                    y_pred = self.forward(Xi)
                    loss = criterion(y_pred, yi)
                    loss = loss / (population ** average_gradient)
                    total_loss += loss.item()
                    loss.backward()
                    
                    
                    del Xi
                    del yi
                    del y_pred
                    gc.collect()
                
            if early_stop_train:
                self.es(total_loss, self.model)
                
            optimizer.step()
            optimizer.zero_grad()
            
            if self.es.stop or ((epoch + 1) == epochs and early_stop_train):
                self.model.load_state_dict(self.es.es_dict)
                print(" Exiting training: early stop triggered.")
                break
            
            if verbose > 0:
                if epoch % verbose == 0:
                    print('Epoch {} train loss: {}'.format(epoch, total_loss))          
            
    
    def predict_accuracy(self, generator, generator_kwargs = {"mode" : "eval"},
                         feature_noise = 1):
        self.model.eval()
        acc = torch.as_tensor([0, 0], dtype = torch.float32)
        for g in generator(**generator_kwargs):
            if (len(g) == 3):
                i, X, y = g
                
            if (len(g) == 5):
                i, X, y, Xv, yv = g
                
            y = torch.as_tensor(y, dtype = torch.float32).to(self.device)
            #X = torch.as_tensor(X, dtype = torch.float32).to(self.device)
            y_pred = self.predict(X * feature_noise)
            
            acc[0] = acc[0] + self.accuracy(y, y_pred)
            select = torch.randint(high = 4999, size = (10,))
            if (len(g) == 5):
                yv = torch.as_tensor(yv, dtype = torch.float32).to(self.device)
                yv_pred = self.predict(Xv * feature_noise)
                
                acc[1] = acc[1] + self.accuracy(yv, yv_pred)
                
        if (len(g) == 3):
            #print(torch.mean(y_pred).data, end = " ")
            return acc[0].item() / (i + 1)
        if (len(g) == 5):
            #print(torch.mean(y_pred).cpu().numpy(), end = " ")
            #print(torch.mean(yv_pred).cpu().numpy(), end = " ")
            return acc.data / (i + 1)
    
    
    def accuracy(self, y, y_pred):
        return torch.sum(torch.eq(y_pred, y).float()) / len(y)
    
    
    def population_accuracy(self, generator, generator_kwargs, population = 1,
                            feature_noise = None, feature_noise_params = [1, 0.1],
                            weight_noise = None, average = True):
        if feature_noise is None: feature_noise = np.ones((self.nodes[0],))
        if type(feature_noise) is bool:
            if feature_noise == True: feature_noise = np.array([1,1])
            else: feature_noise = np.ones((self.nodes[0],))
            
        if feature_noise.shape[0] != self.nodes[0]:
            feature_noise = np.random.normal(loc = feature_noise_params[0],
                                                     scale = feature_noise_params[1],
                                                     size = self.nodes[0])
        
        self.model.eval()
        
        if average:
            acc = torch.as_tensor(0, dtype = torch.float32)
        else:
            acc = [0] * population
        
        dset = generator_kwargs.get("data_set", "train")
        print("Testing {}...".format(dset), end = " ")
        for p in range(population):
            if weight_noise is not None:
                self.add_noise(wnoise = weight_noise[p])
            else:
                self.add_noise()
            for g in generator(**generator_kwargs):
                i, X, y = g
                    
                y_pred = self.predict(X * feature_noise)
                y = torch.as_tensor(y, dtype = torch.float32).to(self.device)
                if average:
                    acc = acc + self.accuracy(y, y_pred)
                else:
                    acc[p] = self.accuracy(y, y_pred).item()
            if (p + 1) % 100 == 0: print(p + 1, end = ' ')
        print()
        if average:    
            return acc.item() / (i + 1) / population
        else:
            return acc

    '''
    def predict_accuracy_all(self,X,y):
        self.model.eval()
        self.model.cpu()
        accs = []
        for Xi,yi in zip(X,y):
            if type(yi) is np.ndarray:
                yi = torch.as_tensor(yi, dtype = torch.float32)
            self.model.eval()
            y_pred = (self.forward(Xi,'cpu') > self.threshold).float()
            acci = torch.sum(torch.eq(y_pred,yi).float())/len(yi)
            accs += [acci.data]
        return np.array(accs)
    '''
    
    def predict(self, X = None, X_forward = None):
        self.model.eval()
        if X is not None:
            X_forward = self.forward(X)
            
        if len(X_forward.shape) == 1:
            y_pred = (X_forward > self.threshold).float()
        else:
            y_pred = torch.argmax(X_forward, dim = 1)
        
        return y_pred
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
        
    def add_noise(self, wnoise = None, how = "both", loc_scale = [1, 0.035, 0, 1 * 0.07]):
        if wnoise is not None:
            size = 2 if how == "both" else 1
            if self.synapses * size != len(wnoise):
                raise TypeError('Length of noise arrays ({}) does not match number of synapses ({}) in module.'\
                                .format(len(wnoise), self.synapses))
        elif how == "both":
            wnoise = np.random.normal(loc = loc_scale[0],
                                      scale = loc_scale[1],
                                      size = self.synapses)
            
            wnoise = np.concatenate([wnoise,
                                     np.random.normal(loc = loc_scale[2],
                                                      scale = loc_scale[3],
                                                      size = self.synapses)])
        else:
            wnoise = np.random.normal(loc_scale[0],
                                      scale = loc_scale[1],
                                      size = self.synapses)
            
        start = 0
        for module in self.model:
            if 'weight' in module.state_dict():
                synapses = module.bias.shape[0]+module.weight.shape[0]*module.weight.shape[1]
                all_noise = wnoise[start : start + synapses] if how != "both" else \
                    np.concatenate([wnoise[start : start + synapses],
                                    wnoise[start + self.synapses : start \
                                           + self.synapses + synapses]])
                        
                module.add_noise(all_noise, noise_mode = how)
                start += synapses
                
    def add_input_noise(self, wnoise = None, how = "both", loc_scale = [1, 0.035, 0, 1 * 0.07]):
        if wnoise is not None:
            size = 2 if how == "both" else 1
            if sum(self.nodes[1:-1]) * size != len(wnoise):
                raise TypeError('Length of noise arrays ({}) does not match number of hidden nodes ({}) in module.'\
                                .format(len(wnoise), sum(self.nodes[1:-1])))
        elif how == "both":
            wnoise = np.random.normal(loc = loc_scale[0],
                                      scale = loc_scale[1],
                                      size = self.synapses)
            
            wnoise = np.concatenate([wnoise,
                                     np.random.normal(loc = loc_scale[2],
                                                      scale = loc_scale[3],
                                                      size = self.synapses)])
        else:
            wnoise = np.random.normal(loc_scale[0],
                                      scale = loc_scale[1],
                                      size = self.synapses)
            
        start = 0
        for module in self.model:
            try: size = module.size
            except: continue
            all_noise = wnoise[start : start + size] if how != "both" else \
                        np.concatenate([wnoise[start : start + size],
                                        wnoise[start + self.synapses : start \
                                               + self.synapses + size]])
                            
            module.add_noise(all_noise, noise_mode = how)
            start += size
    
class Linear_wVoltages(torch.nn.Linear):
    def __init__(self,in_features, out_features, layer, bias=True, device = 'cuda'):
        super(Linear_wVoltages,self).__init__(in_features, out_features, bias)
        self.device = device
        #self.activation = torch.nn.Tanh()
        self.layer = layer
        self.shape = (out_features,in_features + 1)
        self.initialize()
        
    def initialize(self):
        self.maxgain = 10
        self.rate = 1 / self.maxgain
        
        #self.all_rates = self.rate*torch.ones(self.shape).to(self.device)
        self.all_gains = self.maxgain*torch.ones(self.shape).to(self.device)
        self.all_add = torch.zeros(self.shape).to(self.device)
        
            
    def forward(self,X):
        weight = self.all_gains[:,1:].to(X.device)*torch.tanh(self.rate*self.weight) + self.all_add[:,1:]
        bias = self.all_gains[:,0].to(X.device)*torch.tanh(self.rate*self.bias) + self.all_add[:,0]
        return F.linear(X, weight, bias)
    
    def add_noise(self, gnoise, noise_mode = 'both'):
        if type(gnoise) is np.ndarray:
            gnoise = torch.as_tensor(gnoise, dtype = torch.float32)
        
        if noise_mode == "both":
            mult = gnoise[:self.shape[0] * self.shape[1]]
            add = gnoise[self.shape[0] * self.shape[1]:]
            self.all_gains = (self.maxgain * mult.view(self.shape)).to(self.device)
            self.all_add =  add.view(self.shape).to(self.device)
        
        if noise_mode == 'add':
            gnoise = (gnoise - 1) * self.maxgain
            self.all_gains = (self.maxgain*torch.ones(self.shape)+gnoise.view(self.shape)).to(self.device)
            
        if noise_mode == 'mult':
            #self.all_rates = (self.rate*torch.ones(self.shape)*rnoise.view(self.shape)).to(self.device)
            self.all_gains = (self.maxgain*torch.ones(self.shape)*gnoise.view(self.shape)).to(self.device)

class Linear_wVoltages_Sparse(Linear_wVoltages):
    def __init__(self,in_features, out_features, layer, bias=True, device = 'cuda',
                 sparsity = 1, strategy = "each", mask_epochs = 200, soft_power = 0):
        super(Linear_wVoltages_Sparse,self).__init__(in_features, out_features, layer, bias, device)
        self.sparsity = sparsity
        self.strategy = strategy
        self.mask_epochs = mask_epochs
        self.soft_power = soft_power
        self.synapses = in_features * out_features
        self.initialize_mask()
        
        
        
    def initialize_mask(self):
        self.mask = torch.ones(self.shape[0], self.shape[1] - 1)
        self.epochs = 0
        self.mask_start = 0
        if self.sparsity > 1:
            n_remove = self.shape[1] - 1 - self.sparsity
            self.max_row = self.sparsity
        else:
            self.max_row = int(self.sparsity * (self.shape[1] - 1))
            n_remove = self.shape[1] - 1 - self.max_row
        if n_remove <= 0:
            self.mask_rate = float("inf")
        else:
            self.mask_rate = self.mask_epochs // int(n_remove + self.layer + 2)
        #print(self.mask_epochs, self.layer, n_remove, self.mask_rate)
            
            
    def calc_rate(self):
        #self.mask_rate *= 2
        pass
            
        
    def forward(self, X):
        if max(self.mask.sum(axis = 1)) == self.max_row and (self.epochs + 1) % self.mask_rate == 0:
            pass
            #self.initialize()
            #self.mask_rate = float("inf")
        self.epochs += 1
        weight = self.all_gains[:self.shape[0], 1:self.shape[1]] \
            .to(X.device)*torch.tanh(self.rate*self.weight) \
                + self.all_add[:self.shape[0], 1:self.shape[1]]
                
        bias = self.all_gains[:self.shape[0],0] \
            .to(X.device)*torch.tanh(self.rate*self.bias) \
                + self.all_add[:self.shape[0],0]
                
        
        if self.strategy == "all":
            if self.epochs % self.mask_rate == 0 and self.epochs >= self.mask_start:
                self.mask_all()
        elif self.strategy == "allsoft":
            if self.epochs % self.mask_rate == 0 and self.epochs >= self.mask_start:
                self.mask_all_soft()
        elif self.strategy == "once":
            if self.epochs == self.mask_start:
                self.mask_all()
                self.mask_from_weight()
        else:
            if (self.epochs) % self.mask_rate == 0 and self.epochs >= self.mask_start:
                self.mask_weights()
                #self.epochs = 0
        return F.linear(X, weight * self.mask.to(X.device), bias)
    
    
    def random_mask(self):
        self.mask = torch.zeros((self.shape[0], self.shape[1] - 1))
        for row in range(self.shape[0]):
            samples = min(self.shape[1] - 1, self.max_row)
            ones = np.random.choice(np.arange(self.shape[1] - 1),
                                    samples,
                                    replace=False)
            self.mask[row, ones] = 1
            
    
    
    def mask_weights(self):
        
        if self.sparsity > 1:
            if max(self.mask.sum(axis = 1)) == self.max_row:
                return
        else:
            if (self.mask.sum().item() - 1) / self.synapses  <= self.sparsity:
                return
            
        if self.strategy == "one":
            wmin = torch.argsort(torch.abs(self.weight).view(-1))
            count = 0
            for w in wmin:
                wrow = w // (self.shape[1] - 1)
                wcol = w % (self.shape[1] - 1)
                #print(w, wrow, wcol, self.shape)
                if self.sparsity > 1:
                    if self.mask.sum(axis = 1)[wrow] > self.sparsity:
                        break
                else:
                    count += 1
                    if count >= self.shape[0]: break
                
            with torch.no_grad():
                self.weight[wrow,wcol] = 100
            self.mask[wrow,wcol] = 0
            
        
        elif self.strategy == "each":
            if self.sparsity > 1:
                wmin = torch.argmin(torch.abs(self.weight), axis = 1)
                for wrow,wcol in enumerate(wmin):
                    if self.mask[wrow].sum() <= self.max_row: continue
                    with torch.no_grad():
                        self.weight[wrow,wcol] = 100
                    self.mask[wrow,wcol] = 0
            else:
                wmin = torch.argsort(torch.abs(self.weight).view(-1))
                for w in wmin[:self.shape[0]]:
                    wrow = w // (self.shape[1] - 1)
                    wcol = w % (self.shape[1] - 1)
                    with torch.no_grad():
                        self.weight[wrow,wcol] = 100
                    self.mask[wrow,wcol] = 0
                    if (self.mask.sum() - 1) / self.synapses < self.sparsity:
                        self.sparsity_done = True
                        break
                    
    def mask_all(self):
        with torch.no_grad():
            sd = self.state_dict()
            w = sd['weight'].cpu().numpy()
            mask_onehot = self.mask_from_weight(set_mask = False)
			
            #self.mask = mask
            new_weight = torch.from_numpy(w*mask_onehot).cuda()
            sd['weight'] = new_weight
            self.load_state_dict(sd)
            
    def mask_all_soft(self):
        def calc_soft_mask(w):
            mx = np.abs(w).max(axis = 1)
            new = (w.T / mx).T
            return np.abs(new ** self.soft_power)
            
        with torch.no_grad():
            sd = self.state_dict()
            w = sd['weight'].cpu().numpy()
            
            mask = np.argpartition(np.abs(w),-self.max_row)[:,-self.max_row:]
			
            mask_onehot = calc_soft_mask(w)
            for i,j in enumerate(mask):
                mask_onehot[i,j.astype(int)] = 1
            #self.mask = mask
            new_weight = torch.from_numpy(w*mask_onehot).cuda()
            sd['weight'] = new_weight
            self.load_state_dict(sd)
            
    def mask_from_weight(self, set_mask = True):
        sd = self.state_dict()
        w = sd['weight'].cpu().numpy()
        
        mask = np.argpartition(np.abs(w),-self.max_row)[:,-self.max_row:]
			
        mask_onehot = np.zeros(w.shape).astype(int)
        for i,j in enumerate(mask):
            mask_onehot[i,j.astype(int)] = 1
			
        if set_mask:
            self.mask = torch.from_numpy(mask_onehot)
        self.last_mask = mask_onehot
        return mask_onehot

class myEarlyStop():
    def __init__(self, patience = 50, delta = 0.0001, mode = "acc"):
        self.patience = patience
        self.delta = delta
        self.es_counter = 0
        self.es_dict = {}
        if mode == "acc":
            self.best_score = 0
        if mode == "loss":
            self.best_score = -np.inf
        self.mode = mode
        self.stop = False
        
    def __call__(self, score, model):
        if self.mode == "loss": score *= -1
        if score > self.best_score + self.delta:
            self.es_counter = 0
            self.stop = False
            self.best_score = score
            self.es_dict = model.state_dict()
            
        else:
            self.es_counter += 1
        if self.es_counter > self.patience:
            self.stop = True
            
class MaskStop():
    def __init__(self, patience = 50, delta = 0.0001, mode = "acc"):
        self.patience = patience
        self.delta = delta
        self.es_counter = 0
        self.mask_counter = 0
        self.es_dict = {}
        self.best_mask = []
        self.best_prev_mask = 0
        if mode == "acc":
            self.best_score = 0
        if mode == "loss":
            self.best_score = -np.inf
        self.mode = mode
        self.stop = False
        self.similarity_stop = False
        
    def mask_equal(self, mask2):
        if len(self.best_mask) != len(mask2): return False
        return all([np.array_equal(m1,m2) for m1,m2 in zip(self.best_mask,mask2)])

        
    def __call__(self, score, model, mask):
        
        def model_mask():
            return [m.last_mask for m in model if type(m) is Linear_wVoltages_Sparse]
        
        
        if self.mode == "loss": score *= -1
        #'''
        if self.similarity_stop:
            mask = model_mask()
            if self.mask_equal(mask): self.mask_counter += 1
            else:
                self.best_prev_mask = max(self.mask_counter, self.best_prev_mask)
                self.mask_counter = 0
        #'''
        if score > self.best_score + self.delta:
            self.es_counter = 0
            
            self.stop = False
            self.best_score = score
            self.best_mask = mask
            self.es_dict = model.state_dict()
            
        else:
            self.es_counter += 1
            
        
            
        if self.es_counter > self.patience:
            self.stop = True
        #'''
        elif self.mask_counter > 10 and self.similarity_stop:
            self.stop = True
            print("maskStop", self.best_prev_mask, self.mask_counter)
        #'''