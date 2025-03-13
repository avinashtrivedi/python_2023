# -*- coding: utf-8 -*-
"""
@author: 
"""

LOOKBACK = 30

NUM_LSTM_LAYERS = 2
DIM_IN = 1
DIM_OUT = 1
NUM_FFN_HIDDEN = 4
DIM_FFN_HIDDEN = 200
LEARN_RATE = 0.01
CUDA_PREF = True

TRAIN_SHARE = 0.8 
BATCH_TRAIN = 32
EPOCHS =100
FS = 8 # fontsize
NUM_EPOCHS_UPDATE = 100

NO_RELU = 0
IS_RELU_HIDDEN = 1
IS_RELU_HIDDEN_AND_OUT = 2
ACTIVATION = IS_RELU_HIDDEN_AND_OUT

FILE_PATH = 'itw_torch_lstm2.pkl'


import torch.nn as nn
import torch.nn.functional as F


class itw_torch_LSTM_class(nn.Module):
    """
    lstm model consisting of lstm input layer and number of fully connected
    hidden plus output layers
    
    built into sequential block instead of modulelist
    fully relu'd
    parametrized
    original
    """
    def __init__(self, insize, outsize, hidsize, numlstm, numffn, activation):
        super().__init__()
        # lstm input layer
        self.lstm = nn.LSTM(input_size = insize, hidden_size = hidsize,
                            num_layers = numlstm, batch_first = True) 
        # sequence of hidden layers
        self.ffn = nn.Sequential()
        for _ in range(numffn - 1):
            self.ffn.append(nn.Linear(hidsize, hidsize))
            if activation >= IS_RELU_HIDDEN:
                self.ffn.append(nn.ReLU())
        # output layer
        self.ffn.append(nn.Linear(hidsize, outsize))
        if activation >= IS_RELU_HIDDEN_AND_OUT:
            self.ffn.append(nn.ReLU())
        
    def forward(self, x):
        x, _ = self.lstm(x)
        return self.ffn(x)


class ___2____itw_torch_LSTM_class(nn.Module):
    """
    lstm model consisting of lstm input layer and number of fully connected
    hidden plus output layers
    
    fully relu'd
    parametrized
    original
    """
    def __init__(self, insize, outsize, hidsize, numlstm, numffn, activ):
        super().__init__()
        self.lstm = nn.LSTM(input_size = insize, hidden_size = hidsize,
                            num_layers = numlstm, batch_first = True) 
        # set of output layers, last is the actual ol
        self.hl = nn.ModuleList([nn.Linear(hidsize, hidsize) for _ in range(numffn - 1)])
        self.ol = nn.Linear(hidsize, outsize)
        self.activation = activ
    def forward(self, x):
        x, _ = self.lstm(x) 
        for hli in self.hl:
            x = F.relu(hli(x)) if self.activation >= IS_RELU_HIDDEN else hli(x) 
        x = F.relu(self.ol(x)) if self.activation >= IS_RELU_HIDDEN_AND_OUT else self.ol(x) 
        return x

class ___1____itw_torch_LSTM_class(nn.Module):
    """
    lstm model consisting of lstm input layer and number of fully connected
    hidden plus output layers
    
    original
    """
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size = DIM_IN, hidden_size = DIM_FFN_HIDDEN,
                            num_layers = NUM_LSTM_LAYERS, batch_first = True) 
        # set of output layers, last is the actual ol
        self.hl = nn.ModuleList([nn.Linear(DIM_FFN_HIDDEN, DIM_FFN_HIDDEN) for _ in range(NUM_FFN_HIDDEN - 1)])
        self.ol = nn.Linear(DIM_FFN_HIDDEN, DIM_OUT)
    def forward(self, x):
        x, _ = self.lstm(x) 
        for hli in self.hl:
            x = F.relu(hli(x))
        x = self.ol(x) 
        return x



def get_the_data(lb):
    """
    returns torch tensors for features and labels
    here we first get the demand stream for the selected product
    then we slice it into lb (LOOKBACK) long pieces
    features (inputs): [i, i+1, i+2, ...]
    labels (outputs): [i+2, i+2, i+3, ...]
    look https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/ for
    this i/o approach
    """
    import pickle
    from itw_tools import get_demands, no_date_gaps
    import torch
    import numpy as np
    
    # get the data
    with open("itw_d.pkl", "rb") as f:
        d = pickle.load(f)
    e = d[0]
    p, t, d = get_demands(e)
    if not no_date_gaps(t):
        """
        for now we assume there are no date gaps in the demand stream
        will have to clean this up later
        """
        pass
    # d = floatanize(d)
    X, y = [], [] 
    for i in range(len(d) - lb):
        feature = np.array([np.array([np.float32(di)]) for di in d[i:i + lb]])
        #feature = d[i:i + lb]
        target = np.array([np.array([np.float32(di)]) for di in d[i + 1:i + lb + 1]])
        #target = d[i + 1:i + lb + 1]
        X.append(feature)
        y.append(target)
    return p, t, torch.tensor(X), torch.tensor(y)


p, t, X, y = get_the_data(LOOKBACK)

if __name__ == '__main__':
            
    # go CUDA if possible!
    import torch
    print(torch.cuda.is_available())
    print(CUDA_PREF)
    device = torch.device('cuda' if torch.cuda.is_available() and CUDA_PREF else 'cpu')
    print(f'device = {device}')

    p, t, X, y = get_the_data(LOOKBACK)
    X = X.to(device)
    y = y.to(device)
    # split data in test and training cases
    train_size = int(len(y) * TRAIN_SHARE)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
 
    # Create DataLoader objects
    from torch.utils.data import TensorDataset
    from torch.utils.data import DataLoader

    BATCH_TEST = 1
    train_dataloader = DataLoader(TensorDataset(X_train, y_train), \
                                  batch_size = BATCH_TRAIN, shuffle = True)
    test_dataloader = DataLoader(TensorDataset(X_test, y_test), \
                                 batch_size = BATCH_TEST, shuffle = False)
    
    import pandas as pd
    
    try:
        res = pd.read_pickle(FILE_PATH )
    except FileNotFoundError:
        print("picle experiement file not found,new data and file generated!")
        res = pd.DataFrame({'NUM_LSTM_LAYERS': [], 
                            'NUM_FFN_HIDDEN': [],
                            'DIM_FFN_HIDDEN': [],
                            'ACTIVATION': [], 
                            'exec_time': [], 
                            'RMS0': [], 
                            'RMS': [],
                            'LR': [],
                            'LossF': [],
                            'Optim': []})
        
    for i in range(1,2): # 301):

        num_lstm_layers = [1, 2, 3, 5, 10]
        num_ffn_hidden = [1, 2, 3, 5, 10]
        dim_ffn_hidden = [10, 50, 100, 250, 500]
        activation = [0, 1, 2]
        learn_rate = [1e-4, 1e-3, 1e-2, 1e-1, 1e0]
        import torch.optim as optim
        optimizer = [optim.SGD, optim.Adam]
        lossf = [nn.MSELoss, nn.L1Loss]
        
        from random import randint as randint
        
        NUM_LSTM_LAYERS = 1 # num_lstm_layers[randint(0, 4)]
        NUM_FFN_HIDDEN = 2 # num_ffn_hidden[randint(0, 4)]
        DIM_FFN_HIDDEN = 500 # dim_ffn_hidden[randint(0, 4)]
        ACTIVATION = 2 # activation[randint(0, 2)]
        LEARN_RATE = 0.002 # learn_rate[randint(0, 4)]
        OPTIM = optim.Adam # SGD # optimizer[randint(0, 1)]
        LOSS = nn.MSELoss # lossf[randint(0, 1)]
        
        plt_title = f'run {i}: {p} w/ {LOOKBACK} days past dmnd, '\
                  + f'{len(train_dataloader)} train sets w/ {train_dataloader.batch_size} batches,\n' \
                  + f'LSTM w/ {NUM_LSTM_LAYERS} + FFN w/ {NUM_FFN_HIDDEN} layers and {DIM_FFN_HIDDEN}' \
                  + f' hidden par/layer, activation = {ACTIVATION},\n'\
                  + f'LR = {LEARN_RATE}, LossF = {LOSS}, Opt = {OPTIM}'
        print(plt_title)
        
        RMS0 = None
        RMS = None
        # Create an instance of the model
        model = itw_torch_LSTM_class(DIM_IN, DIM_OUT, DIM_FFN_HIDDEN , NUM_LSTM_LAYERS, \
                                     NUM_FFN_HIDDEN, ACTIVATION).to(device)
    
        # Define the training environment
        opt = OPTIM(model.parameters(), lr = LEARN_RATE)
        crit = LOSS().to(device)
        RMScrit = nn.MSELoss().to(device)
            
        # Training loop
        print(f'{EPOCHS} epochs')
        import time
        import numpy as np
        import torch
        losses = []
        epochs = []
        exec_time = 0
        print('Epoch   RMS(Loss)')
        for epoch in range(EPOCHS):
            exec_time -= time.time()
            for features, labels in train_dataloader:
                opt.zero_grad()                 # reset gradients
                model.zero_grad()
                predicted = model(features)         # forward: generate this pass output
                loss = crit(predicted, labels)    # evaluate loss by comparing output and label 
                loss.backward()                 # backward: calc gradients
                opt.step()                      # adjust parameters based on these gradients by 1 step
            exec_time += time.time()
            if (epoch + 1) % (NUM_EPOCHS_UPDATE) == 0: 
                print('**********************************')
                with torch.no_grad():
                    predicted = model(X_train)
                    RMS = np.sqrt(RMScrit(predicted, y_train).to('cpu'))
                    if RMS0 == None:
                        RMS0 = RMS
                    print(f'{epoch+1:<6}  {RMS:<8.3}')
                    losses.append(RMS)
                    epochs.append(epoch)
        import matplotlib.pyplot as plt                
        plt.figure(figsize=(8, 8))
        plt.plot(epochs, losses)
        plt.yscale('log')
        plt.xlabel('epoch', fontsize=FS)
        plt.ylabel('RMS(loss)', fontsize=FS)
        plt.tick_params(axis='both', labelsize=FS)
        plt.rcParams.update({'font.size': FS})
        #plt.title(plt_title, fontsize=FS)
        plt.show()
        
        this = pd.DataFrame({'NUM_LSTM_LAYERS': [NUM_LSTM_LAYERS], 
                             'NUM_FFN_HIDDEN': [NUM_FFN_HIDDEN],
                             'DIM_FFN_HIDDEN': [DIM_FFN_HIDDEN],
                             'ACTIVATION': [ACTIVATION], 
                             'exec_time': [exec_time], 
                             'RMS0': [RMS0], 
                             'RMS': [RMS],
                             'LR': [LEARN_RATE],
                             'LossF': [LOSS],
                             'Optim': [OPTIM]})
        # print(this)
        res = pd.concat([res, this], ignore_index=True)
        res.to_pickle(FILE_PATH)
    
        print(f'training execution time = {exec_time:3.1f}s')
    
        with torch.no_grad():     # deactivate gradients for better performance
            plt.title(plt_title, fontsize=FS)
            dp = y[:,-1].to('cpu')
            tp = t[LOOKBACK:]
            trainp = np.ones_like(dp) * np.nan
            trainp[:train_size] = model(X_train).to('cpu')[:,-1,:]
            testp = np.ones_like(dp) * np.nan
            testp[train_size:len(dp)] = model(X_test).to('cpu')[:,-1,:]
            plt.figure(figsize=(8, 8))
            plt.plot(tp, dp, c='b', label = 'input data')
            plt.plot(tp, trainp, c='r', label = 'prediction on train set')
            plt.plot(tp, testp, c='g', label = 'prediction on test set')
            plt.legend(loc = 'upper left', fontsize=FS)
            plt.xlabel('dfc', fontsize=FS)
            plt.ylabel('amount', fontsize=FS)
            plt.rcParams.update({'font.size': FS})
            plt.show()
            
    # convert tensors to their inner content
    res['RMS'] =  res['RMS'].apply(lambda x: x.item())
    res['RMS0'] = res['RMS0'].apply(lambda x: x.item())
    # eliminate NaN and Inf entries
    res = res[~(np.isnan(res['RMS']) | np.isinf(res['RMS']) | np.isnan(res['RMS0']) | np.isinf(res['RMS0']))]
    
    import matplotlib.pyplot as plt
    x = res['RMS']
    y = res['exec_time']
    labels = res.index.astype(str) # res['NUM_LSTM'].astype(str) + '-' + res['NUM_FFN'].astype(str)
    plt.scatter(x, y)
    for label, xi, yi in zip(labels, x, y):
        plt.annotate(label, (xi, yi))
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('RMS')
    plt.ylabel('exec_time')
    NUM_EL = 10
    MARGIN = 0.01
    min_x = min(x) * (1 - MARGIN)
    max_x = sorted(x)[NUM_EL - 1 if len(x) >= NUM_EL else -1] * (1 + MARGIN)
    plt.xlim(min_x, max_x)
    plt.show()

 
    
    pd.set_option('display.max_columns', None)
    print(res.sort_values(by=['RMS'], ascending=True))
    
    
    """
    note index is 0 based, runs are 1 based index 281 is run 282
    
     NUM_LSTM_LAYERS  NUM_FFN_HIDDEN  DIM_FFN_HIDDEN  ACTIVATION   exec_time  \
281              1.0             1.0           500.0         2.0   38.316425   
89               3.0             3.0           250.0         0.0   80.371961   
56               2.0            10.0            50.0         0.0   76.108753   
88               5.0             3.0           250.0         2.0  113.054048   
282              3.0             3.0           250.0         1.0   71.925768   
..               ...             ...             ...         ...         ...   
7                5.0            10.0            10.0         0.0   85.305632   
58               2.0             2.0            10.0         1.0   40.290123   
169              5.0             5.0           100.0         0.0   76.082772   
95               5.0            10.0           100.0         1.0  117.286477   
216              1.0            10.0            50.0         0.0   74.758132   

             RMS0           RMS     LR  \
281  3.061410e+05  3.007517e+05  0.010   
89   3.193791e+05  3.164641e+05  0.001   
56   3.198062e+05  3.164733e+05  0.001   
88   3.178738e+05  3.164964e+05  0.001   
282  3.176670e+05  3.165622e+05  0.001   
..            ...           ...    ...   
7    9.642267e+05  7.486893e+09  1.000   
58   1.086963e+11  1.087870e+11  1.000   
169  1.148917e+06  2.006295e+12  1.000   
95   6.065240e+14  6.065200e+14  1.000   
216  6.376572e+07  1.582170e+15  1.000   

                                       LossF                            Optim  
281  <class 'torch.nn.modules.loss.MSELoss'>    <class 'torch.optim.sgd.SGD'>  
89   <class 'torch.nn.modules.loss.MSELoss'>  <class 'torch.optim.adam.Adam'>  
56   <class 'torch.nn.modules.loss.MSELoss'>  <class 'torch.optim.adam.Adam'>  
88   <class 'torch.nn.modules.loss.MSELoss'>  <class 'torch.optim.adam.Adam'>  
282  <class 'torch.nn.modules.loss.MSELoss'>  <class 'torch.optim.adam.Adam'>  
..                                       ...                              ...  
7    <class 'torch.nn.modules.loss.MSELoss'>  <class 'torch.optim.adam.Adam'>  
58   <class 'torch.nn.modules.loss.MSELoss'>    <class 'torch.optim.sgd.SGD'>  
169  <class 'torch.nn.modules.loss.MSELoss'>  <class 'torch.optim.adam.Adam'>  
95   <class 'torch.nn.modules.loss.MSELoss'>  <class 'torch.optim.adam.Adam'>  
216   <class 'torch.nn.modules.loss.L1Loss'>  <class 'torch.optim.adam.Adam'>     
    """




