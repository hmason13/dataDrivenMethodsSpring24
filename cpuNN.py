#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.nn import Sequential as Seq, Linear, ReLU


# In[10]:



# Basic Dataset for CNN given non-normalized data
# This function will first normalize the data to unit variance before constructing the dataset. 
# The normalization values will be stored in self.norm_vals (these should be constant 
# across testing and training)

# input shape should be (N,C,Ny,Nx), N is the number of snapshots, C is the number of channels, 
# NY and NX are the number of points in the y and x direction respectively
# output shape follows the same convention
# data_in is the input array
# data_out is the input array
# device is the device you keep the dataset on, for training on the GPU it should be set to "cuda"
# if you are exploring on the cpu make sure to change it to "cpu"

class data_CNN_normalize(torch.utils.data.Dataset):

    def __init__(self,data_in,data_out,device = "cuda"):

        super().__init__()
        num_inputs = data_in.shape[3]
        num_outputs = data_out.shape[3]
        self.size = data_in.shape[0]
        
        data_in = np.nan_to_num(data_in)
        data_out = np.nan_to_num(data_out)
        
        std_data = np.nanstd(data_in,axis=(0,2,3))
        mean_data = np.nanmean(data_in,axis=(0,2,3)) 
        std_label = np.nanstd(data_out,axis=(0,2,3))
        mean_label = np.nanmean(data_out,axis=(0,2,3))
        
        for i in range(num_inputs):
            data_in[:,i,:,:] = (data_in[:,i,:,:,] - mean_data[i])/std_data[i]
        
        for i in range(num_outputs):
            data_out[:,i,:,:] = (data_out[:,i,:,:] - mean_label[i])/std_label[i]
            
        data_in = torch.from_numpy(data_in).type(torch.float32).to(device=device)
        data_out = torch.from_numpy(data_out).type(torch.float32).to(device=device)        
        

        std_dict = {'s_in':std_data,'s_out':std_label,'m_in':mean_data, 'm_out':mean_label}
            
        self.input = data_in
        self.output = data_out
        
        self.norm_vals = std_dict
        
    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.size

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        data_in = self.input[idx]
        label = self.output[idx]
        return data_in, label
    
# Basic Dataset for CNN given normalized data
# Same as above, but if you want to take care of normalization outside of this function.
    
    
class data_CNN(torch.utils.data.Dataset):

    def __init__(self,data_in,data_out,device = "cuda"):

        super().__init__()
        num_inputs = data_in.shape[3]
        num_outputs = data_out.shape[3]
        self.size = data_in.shape[0]
        
        data_in = np.nan_to_num(data_in)
        data_out = np.nan_to_num(data_out)
            
        data_in = torch.from_numpy(data_in).type(torch.float32).to(device=device)
        data_out = torch.from_numpy(data_out).type(torch.float32).to(device=device)        
        
            
        self.input = data_in
        self.output = data_out
                
    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.size

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        data_in = self.input[idx]
        label = self.output[idx]
        return data_in, label    
    
# Basic Dataset for ANN given non-normalized data 
# This function will first normalize the data to unit variance before constructing the dataset. 
# The normalization values will be stored in self.std (these should be constant 
# across testing and training). This assumes all inputs are of similar mean and variance. 
# If this is not the case normalize yourself and use the following function
# input shape should be (N,L), N is the number of snapshots, L is the length of the flattened array
# data_in is the input array
# data_out is the input array
# device is the device you keep the dataset on, for training on the GPU it should be set to "cuda"
# if you are exploring on the cpu make sure to change it to "cpu"    
    
class data_ANN_normalize(torch.utils.data.Dataset):

    def __init__(self,data_in,data_out,device = "cuda"):

        super().__init__()
        num_inputs = data_in.shape[1]
        num_outputs = data_out.shape[1]
        
        data_in_num = data_in
        data_out_num = data_out
        data_in = torch.from_numpy(data_in).type(torch.float32).to(device=device)
        data_out = torch.from_numpy(data_out).type(torch.float32).to(device=device)
        self.size = data_in.size()[0]
        
        std_data = torch.std(data_in,0)
        mean_data = torch.mean(data_in,0)        
        std_label = torch.std(data_out,0)
        mean_label = torch.mean(data_out,0)    
        

        data_in = (data_in - mean_data)/std_data
                          
        data_out = (data_out - mean_label)/std_label          
        
        std_dict = {'s_in':std_data,'s_out':std_label,'m_in':mean_data, 'm_out':mean_label}
        
        data_in = torch.nan_to_num(data_in)
        data_out = torch.nan_to_num(data_out)
        
        self.input = data_in
        self.output = data_out
        self.std = std_dict

    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.size

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        data_in = self.input[idx]
        label = self.output[idx]
        return data_in, label  
    
# Basic Dataset for ANN given normalized data 
# Same as above, but for if you have data that is already normalized
class data_ANN(torch.utils.data.Dataset):

    def __init__(self,data_in,data_out,device = "cuda"):

        super().__init__()
        num_inputs = data_in.shape[1]
        num_outputs = data_out.shape[1]
        
        data_in_num = data_in
        data_out_num = data_out
        data_in = torch.from_numpy(data_in).type(torch.float32).to(device=device)
        data_out = torch.from_numpy(data_out).type(torch.float32).to(device=device)
        self.size = data_in.size()[0]         
        
        
        data_in = torch.nan_to_num(data_in)
        data_out = torch.nan_to_num(data_out)
        
        self.input = data_in
        self.output = data_out

    def __len__(self):
        # Number of data point we have. Alternatively self.data.shape[0], or self.label.shape[0]
        return self.size

    def __getitem__(self, idx):
        # Return the idx-th data point of the dataset
        # If we have multiple things to return (data point and label), we can return them as tuple
        data_in = self.input[idx]
        label = self.output[idx]
        return data_in, label        


# In[11]:


# basic CNN 
# num_in = number of input channels (e.g giving u,v,T would be 3 input channels)
# num_out = number of output channels (e.g giving u,v,T would be 3 output channels)
# num_channels = number of channels in the hidden layers
# num_layers = number of hidden layers (does not include input and output)
# kernel_size = filter width
# padding mode = way to pad boundaries (in periodic domains this can be changed to circular)

class CNN(torch.nn.Module):

    def __init__(self,num_in = 2, num_out = 2,num_channels = 64, num_layers=6,
                 kernel_size = 5,padding_mode = "zeros"):
        super().__init__()
        self.N_in = num_in

        layers = []
        layers.append(torch.nn.Conv2d(num_in,num_channels,kernel_size,
                                      padding='same',padding_mode = padding_mode))
        layers.append(torch.nn.ReLU())
        for _ in range(num_layers):
            layers.append(torch.nn.Conv2d(num_channels,num_channels,kernel_size,
                                          padding='same',padding_mode = padding_mode))
            layers.append(torch.nn.ReLU())              
        layers.append(torch.nn.Conv2d(num_channels,num_out,kernel_size,
                                      padding='same',padding_mode = padding_mode))

        self.layers = nn.ModuleList(layers)
        #self.layers = nn.ModuleList(layer)

    def forward(self,fts):
        for l in self.layers:
            fts= l(fts)
        return fts
    

# basic CNN 
# num_in = size of flattened inputs 
# num_out = size of flattened outputs 
# num_hidden = number of nodes in the hidden layers
# num_layers = number of hidden layers (does not include input and output)
    
    
class ANN(torch.nn.Module):

    def __init__(self,num_in = 2, num_out = 1, num_layers=6, num_hidden = 25):
        super().__init__()
        layers = []
        layers.append(torch.nn.Linear(num_in,num_hidden))
        layers.append(torch.nn.Tanh())
        for _ in range(num_layers):
            layers.append(torch.nn.Linear(num_hidden,num_hidden))
            layers.append(torch.nn.Tanh())              
        layers.append(torch.nn.Linear(num_hidden,num_out))

        self.layers = nn.ModuleList(layers)
        #self.layers = nn.ModuleList(layer)

    def forward(self,fts):
        for l in self.layers:
            fts= l(fts)
        return fts   
    
    
    


# In[12]:


# basic training loop 
# model = the model you are training
# train_loader = dataloader from the training dataset, see below 
# test_loader = dataloader from the test dataset, see below 
# num_epochs = number of update steps of the model. In each step, the model will see the full dataset
# loss = loss function chosen of the form (scalar = loss(pred,true))
# optim = optimizer that will update your network

def train(model, train_loader, test_loader, num_epochs, loss_fn, optim):
    # Set up the loss and the optimizer
    for epoch in range(num_epochs):
        for data, label in train_loader:
            optimizer.zero_grad()
            outs = model(data)

            loss = loss_fn(outs, label) # no train_mask!
            loss.backward()
            optimizer.step()
        for data, label in test_loader:
            outs = model(data)
            loss_val = loss_fn(outs, label) # no train_mask!
        if epoch%20==0:    
            print(f'[Epoch {epoch+1}/{num_epochs}] Loss: {loss} | Val Loss: {loss_val}')


# Load data
loadLoc = "/scratch/hcm7920/class/mitgcmData/"
dataVel = np.zeros(shape=(1200,5000))
dataEta = np.zeros(shape=(1200,2500))
testVel = np.zeros(shape=(100,5000))
testEta = np.zeros(shape=(100,2500))
exmpVel = np.zeros(shape=(100,5000))
exmpEta = np.zeros(shape=(100,2500))

for sn in range(1200):
  dataVel[sn,0:2500] = np.load(loadLoc+f"u/dataU-{sn:05}.npy")
  dataVel[sn,2500:5000] = np.load(loadLoc+f"v/dataV-{sn:05}.npy")
  dataEta[sn,:] = np.load(loadLoc+f"eta/dataEta-{sn:05}.npy")

for sn in range(1200,1300,1):
  testVel[sn-1200,0:2500] = np.load(loadLoc+f"u/dataU-{sn:05}.npy")
  testVel[sn-1200,2500:5000] = np.load(loadLoc+f"v/dataV-{sn:05}.npy")
  testEta[sn-1200,:] = np.load(loadLoc+f"eta/dataEta-{sn:05}.npy")

for sn in range(1300,1400,1):
  exmpVel[sn-1300,0:2500] = np.load(loadLoc+f"u/dataU-{sn:05}.npy")
  exmpVel[sn-1300,2500:5000] = np.load(loadLoc+f"v/dataV-{sn:05}.npy")
  exmpEta[sn-1300,:] = np.load(loadLoc+f"eta/dataEta-{sn:05}.npy")


# Put data into the correct form

train_data = data_ANN_normalize(dataVel,dataEta,device = "cpu")
val_data = data_ANN_normalize(testVel,testEta,device = "cpu")

# Generate a dataloader for both datasets, the batch size can be changed, often for larger data 
# this may need to be kept smaller
train_loader = torch.utils.data.DataLoader(train_data, batch_size=10)
test_loader = torch.utils.data.DataLoader(val_data, batch_size=10)


# In[14]:


# Creates the instance of the model. Change the device to correspond to GPU ("cuda") vs "cpu"

model = ANN(num_in=5000, num_out=2500, num_layers=3, num_hidden=400).to(device = "cpu")

# Creates an optimizer
# lr = learning rate of the model, this might need to be tweaked
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

# define a loss function
loss = torch.nn.MSELoss()

# train the model (just pass in everthing that we have defined previously)
# Note that your problems will likely require far fewer epochs for convergence

train(model, train_loader, test_loader, 400, loss, optimizer)


# In[15]:


# Get the testing data

test_data = data_ANN_normalize(exmpVel,exmpEta,device = "cpu")


# In[16]:


# Let's see how the network did
model = model.eval()

for idex in range(100):
  with torch.no_grad():
      pred = np.reshape(model(test_data[:][0][idex,:]), (50,50))
  
  myTruth = np.reshape(test_data[:][1][idex,:], (50,50))
  fig,ax = plt.subplots(1,2,layout='constrained')
  im = ax[0].imshow(pred, cmap="seismic", vmax=3, vmin=-3)
  ax[0].set_title("Neural net output")
  ax[1].imshow(myTruth, cmap="seismic", vmax=3, vmin=-3)
  ax[1].set_title("MITgcm output")
  plt.colorbar(im, ax=ax[1])
  plt.savefig(f"/scratch/hcm7920/class/plots/nnOutput-{idex:03}.png")
  plt.close('all')




