#!/usr/bin/env python3

"""
#Magda Arnal
#30/01/2023
#FFN model classes 
"""

### import libraries
# for DL modeling
import torch
import torch.nn as nn
import torch.nn.functional as F

# for number-crunching
import numpy as np

# for data visualization
import matplotlib.pyplot as plt
import os
import re

#Important to set a random seed!
torch.manual_seed(1)
np.random.seed(1)
##############################################################################
##############################################################################
def FFN_4hidden(learningRate, selfdr,nUnits,nLayers, inUnits):

  class ANNclass(nn.Module):
    def __init__(self, selfdr,nUnits,nLayers, inUnits):
      super().__init__()

      # create dictionary to store the layers
      self.layers = nn.ModuleDict()
      self.nLayers = nLayers

      ### input layer
      self.layers['input'] = nn.Linear(inUnits,200)
      
      ### hidden layer fixed
      self.layers['hiddenPre'] = nn.Linear(200,nUnits)

      ### hidden layers to iterate
      for i in range(nLayers):
        self.layers[f'hidden{i}'] = nn.Linear(nUnits,nUnits)
      
      ### hidden layer fixed
      self.layers['hiddenPost0'] = nn.Linear(nUnits,50)
      self.layers['hiddenPost1'] = nn.Linear(50,50)

      ### output layer
      self.layers['output'] = nn.Linear(50,1)

      # parameters
      self.dr = selfdr
      
      # forward pass
    def forward(self,x):
      
      x = F.leaky_relu( self.layers['input'](x) )
      x = F.dropout(x,p=self.dr,training=self.training)

      x = F.leaky_relu( self.layers['hiddenPre'](x) )
      x = F.dropout(x,p=self.dr,training=self.training)

      # hidden layers
      for i in range(self.nLayers):
        x = F.leaky_relu( self.layers[f'hidden{i}'](x) )
        x = F.dropout(x,p=self.dr,training=self.training)
      
      x = F.leaky_relu( self.layers['hiddenPost0'](x) )
      x = F.dropout(x,p=self.dr,training=self.training)
      
      x = F.leaky_relu( self.layers['hiddenPost1'](x) )
      x = F.dropout(x,p=self.dr,training=self.training)

      # return output layer
      x = self.layers['output'](x)
      return x
    

  # create the model instance
  net = ANNclass(selfdr,nUnits,nLayers, inUnits)
  
  # loss function
  lossfun = nn.BCEWithLogitsLoss()

  # optimizer
  optimizer = torch.optim.Adam(net.parameters(),lr=learningRate)

  return net,lossfun,optimizer

##########################################################################
##########################################################################
def PlotPerformance(trainAcc,devAcc, devAcc0, devAcc1, devFscore, losses, outpath):
  
  #Select last rows for all matrices, corresponding to the last epoch to get the last accuracy
  trainAcc_mean = np.mean(trainAcc[-1,:])
  devAcc_mean = np.mean(devAcc[-1,:])
  devAcc0_mean = np.mean(devAcc0[-1,:])
  devAcc1_mean = np.mean(devAcc1[-1,:])
  devFscore_mean = np.mean(devFscore[-1,:])

  #Make the plot for each combination of characters
  fig,ax = plt.subplots(2,3,figsize=(16,8))
  for c in range(devFscore.shape[1]):
    ax[0,0].plot(trainAcc[:,c], label=f'c {c:.0f}')
    ax[0,1].plot(devFscore[:,c], label=f'c {c:.0f}')
    ax[0,2].plot(losses[:,c], label=f'c {c:.0f}')
    ax[1,0].plot(devAcc[:,c], label=f'c {c:.0f}')
    ax[1,1].plot(devAcc0[:,c], label=f'c {c:.0f}')
    ax[1,2].plot(devAcc1[:,c], label=f'c {c:.0f}')
                    
  ax[0,0].set_title(f'Train set accuracies mean={trainAcc_mean:.2f}')
  ax[0,0].set_xlabel('Epoch number')
  ax[0,0].set_ylabel('Accuracy')
  ax[0,0].legend()

  ax[0,1].set_title(f'F-beta in the Dev set mean={devFscore_mean:.2f}')
  ax[0,1].set_xlabel('Epoch number')
  ax[0,1].set_ylabel('F-beta')
  ax[0,1].legend()

  ax[0,2].set_title('Losses')
  ax[0,2].set_xlabel('Epoch number')
  ax[0,2].set_ylabel('Loss')
  ax[0,2].legend()

  ax[1,0].set_title(f'Accuracy in the dev set mean={devAcc_mean:.2f}')
  ax[1,0].set_xlabel('Epoch number')
  ax[1,0].set_ylabel('Accuracy')
  ax[1,0].legend()

  ax[1,1].set_title(f'Specificity in the dev set mean={devAcc0_mean:.2f}')
  ax[1,1].set_xlabel('Epoch number')
  ax[1,1].set_ylabel('Specificity')
  ax[1,1].legend()

  ax[1,2].set_title(f'Sensitivity in the dev set mean={devAcc1_mean:.2f}')
  ax[1,2].set_xlabel('Epoch number')
  ax[1,2].set_ylabel('Sensitivity')
  ax[1,2].legend()

  plt.tight_layout()
  #plt.show()
  plt.savefig(outpath, format='pdf' )
  file_name = os.path.basename(outpath)
  print(f'Finished the file: {file_name}')
  plt.close(fig)
  plt.clf()#Clean all the figures

################################################################################
################################################################################
#Define a FFN class that applies a sigmoid activation function to the output logits.
class ANNclassO(nn.Module):
    def __init__(self, selfdr,nUnits,nLayers, inUnits):
      super().__init__()

      # create dictionary to store the layers
      self.layers = nn.ModuleDict()
      self.nLayers = nLayers

      ### input layer
      self.layers['input'] = nn.Linear(inUnits,200)
      
      ### hidden layer fixed
      self.layers['hiddenPre'] = nn.Linear(200,nUnits)

      ### hidden layers to iterate
      for i in range(nLayers):
        self.layers[f'hidden{i}'] = nn.Linear(nUnits,nUnits)
      
      ### hidden layer fixed
      self.layers['hiddenPost0'] = nn.Linear(nUnits,50)
      self.layers['hiddenPost1'] = nn.Linear(50,50)

      ### output layer
      self.layers['output'] = nn.Linear(50,1)

      # parameters
      self.dr = selfdr
      
      # forward pass
    def forward(self,x):
      
      x = F.leaky_relu( self.layers['input'](x) )
      x = F.dropout(x,p=self.dr,training=self.training)

      x = F.leaky_relu( self.layers['hiddenPre'](x) )
      x = F.dropout(x,p=self.dr,training=self.training)

      # hidden layers
      for i in range(self.nLayers):
        x = F.leaky_relu( self.layers[f'hidden{i}'](x) )
        x = F.dropout(x,p=self.dr,training=self.training)
      
      x = F.leaky_relu( self.layers['hiddenPost0'](x) )
      x = F.dropout(x,p=self.dr,training=self.training)
      
      x = F.leaky_relu( self.layers['hiddenPost1'](x) )
      x = F.dropout(x,p=self.dr,training=self.training)

      # return output layer
      x = self.layers['output'](x)
      return x

class ANNclassWithSigmoid(ANNclassO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        # Call the parent forward method to get the output tensor
        output = super().forward(x)
        
        # Apply the sigmoid activation function to the output tensor
        output = torch.sigmoid(output)
        
        return output

#######################################################################################
#######################################################################################
