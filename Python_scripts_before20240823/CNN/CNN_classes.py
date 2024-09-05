#!/usr/bin/env python3

"""
#Magda Arnal
#30/01/2023
#CNN model classes 
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

#####################################################################################
#####################################################################################
def CNN_WithCV(learningRate, selfdr,nUnits,nLayers,inUnits):
  class CNNclass(nn.Module):
      def __init__(self, selfdr, nUnits, nLayers,inUnits):
          super().__init__()
          self.layers = nn.ModuleDict()
          self.nLayers = nLayers
          ################################################
          #First Conv Layer
          l1=6#Out channels (number of layers/vectors representing each individual)
          ks=3 #Kernel or number of rows to summarize in one
          st=1#the step size that the convolutional filter(kernel) moves across the input data when computing each new output. In other words, it determines the amount of overlap between the filter(kernel) and the input as the filter(kernel) moves from one position to the next.
          pa=2#padding are the extra rows to be added to avoid trimming of the first and last rows, I added the same padding as the kernel so I make sure the first and last variants are not trimmed
          self.conv1 = nn.Conv1d(in_channels=3,out_channels=l1,kernel_size=ks,stride=st,padding=pa)
          a=np.floor((inUnits+2*pa-ks)/st)+1
          ouLyier_a=np.floor((a+(2*1))/2)#post-pooling divided by 2 but adding padding 1
          ##################################################
          #Second Conv Layer
          l2=12#Out channels (number of layers/vectors representing each individual)
          ks=3#Kernel or number of rows to summarize in one
          st=1#the step size that the convolutional filter(kernel) moves across the input data when computing each new output. In other words, it determines the amount of overlap between the filter(kernel) and the input as the filter(kernel) moves from one position to the next.
          pa=2#padding are the extra rows to be added to avoid trimming of the first and last rows, I added the same padding as the kernel so I make sure the first and last variants are not trimmed
          self.conv2 = nn.Conv1d(in_channels=l1,out_channels=l2,kernel_size=ks,stride=st,padding=pa)
          b=np.floor((ouLyier_a+2*pa-ks)/st)+1
          ouLyier_b=int(np.floor((b+(2*1))/2))#post-pooling divided by 2 but adding padding 1
          ####################################################
          # Flatten all the convolutional layers: compute the number of units in the fully connected layer (number of outputs of conv2)
          infcLayer=ouLyier_b*l2
          ####################################################
          #Define the fully connected layers
          ### input layer
          self.layers['input'] = nn.Linear(infcLayer,200)
          
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
          
          #First conv-pool set
          conv1act = F.leaky_relu(self.conv1(x))
          x = F.avg_pool1d(conv1act,kernel_size=2, padding=1)
          
          #Second  conv-pool set
          conv2act = F.leaky_relu(self.conv2(x))
          x = F.avg_pool1d(conv2act,kernel_size=2, padding=1)
          
          # fully connected part
          x = x.reshape(x.shape[0],-1)
          
          x = F.leaky_relu( self.layers['input'](x) )
          x = F.dropout(x,p=self.dr,training=self.training)
          
          x = F.leaky_relu( self.layers['hiddenPre'](x) )
          x = F.dropout(x,p=self.dr,training=self.training)
          
          # hidden layers
          for i in range(self.nLayers):
            x = F.leaky_relu( self.layers[f'hidden{i}'](x) )
            x = F.dropout(x,p=self.dr,training=self.training)
            
          #output
          x = F.leaky_relu( self.layers['hiddenPost0'](x) )
          x = F.dropout(x,p=self.dr,training=self.training)

          x = F.leaky_relu( self.layers['hiddenPost1'](x) )
          x = F.dropout(x,p=self.dr,training=self.training)

          x = self.layers['output'](x)
          return x
    
  # create the model instance
  net = CNNclass(selfdr,nUnits,nLayers,inUnits)
  
  # loss function
  lossfun = nn.BCEWithLogitsLoss()
  
  # optimizer
  optimizer = torch.optim.Adam(net.parameters(),lr=learningRate)
  
  return net,lossfun,optimizer

##########################################################################
##########################################################################
def PlotPerformance(trainAcc,devAcc, devAcc0, devAcc1, devFscore, losses, outpath):
  
  #Select last rows for all matrices, corresponding to the last epoch
  trainAcc_mean = np.mean(trainAcc[-1,:])
  devAcc_mean = np.mean(devAcc[-1,:])
  devAcc0_mean = np.mean(devAcc0[-1,:])
  devAcc1_mean = np.mean(devAcc1[-1,:])
  devFscore_mean = np.mean(devFscore[-1,:])

  #Make the plot for each evaluation metric
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
#Define a CNN class that applies a sigmoid activation function to the output logits.

class CNNclassO(nn.Module):
  def __init__(self, selfdr, nUnits, nLayers,inUnits):
    super().__init__()
    
    self.layers = nn.ModuleDict()
    self.nLayers = nLayers
    ################################################
    #First Conv Layer
    l1=6#Out channels (number of layers/vectors representing each individual)
    ks=3 #Kernel or number of rows to summarize in one
    st=1#the step size that the convolutional filter(kernel) moves across the input data when computing each new output. In other words, it determines the amount of overlap between the filter(kernel) and the input as the filter(kernel) moves from one position to the next.
    pa=2#padding are the extra rows to be added to avoid trimming of the first and last rows, I added the same padding as the kernel so I make sure the first and last variants are not trimmed
    self.conv1 = nn.Conv1d(in_channels=3,out_channels=l1,kernel_size=ks,stride=st,padding=pa)
    a=np.floor((inUnits+2*pa-ks)/st)+1
    ouLyier_a=np.floor((a+(2*1))/2)#post-pooling divided by 2 but adding padding 1
    ##################################################
    #Second Conv Layer
    l2=12#Out channels (number of layers/vectors representing each individual)
    ks=3#Kernel or number of rows to summarize in one
    st=1#the step size that the convolutional filter(kernel) moves across the input data when computing each new output. In other words, it determines the amount of overlap between the filter(kernel) and the input as the filter(kernel) moves from one position to the next.
    pa=2#padding are the extra rows to be added to avoid trimming of the first and last rows, I added the same padding as the kernel so I make sure the first and last variants are not trimmed
    self.conv2 = nn.Conv1d(in_channels=l1,out_channels=l2,kernel_size=ks,stride=st,padding=pa)
    b=np.floor((ouLyier_a+2*pa-ks)/st)+1
    ouLyier_b=int(np.floor((b+(2*1))/2))#post-pooling divided by 2 but adding padding 1
    ####################################################
    # Flatten all the convolutional layers: compute the number of units in FClayer (number of outputs of conv2)
    infcLayer=ouLyier_b*l2
    ####################################################
    #Define the fully connected layers
    ### input layer
    self.layers['input'] = nn.Linear(infcLayer,200)

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
            
    #First conv-pool set
    conv1act = F.leaky_relu(self.conv1(x))
    x = F.avg_pool1d(conv1act,kernel_size=2, padding=1)

    #Second  conv-pool set
    conv2act = F.leaky_relu(self.conv2(x))
    x = F.avg_pool1d(conv2act,kernel_size=2, padding=1)

    # ANN part
    x = x.reshape(x.shape[0],-1)

    x = F.leaky_relu( self.layers['input'](x) )
    x = F.dropout(x,p=self.dr,training=self.training)

    x = F.leaky_relu( self.layers['hiddenPre'](x) )
    x = F.dropout(x,p=self.dr,training=self.training)

    # hidden layers
    for i in range(self.nLayers):
      x = F.leaky_relu( self.layers[f'hidden{i}'](x) )
      x = F.dropout(x,p=self.dr,training=self.training)
          
    #output
    x = F.leaky_relu( self.layers['hiddenPost0'](x) )
    x = F.dropout(x,p=self.dr,training=self.training)

    x = F.leaky_relu( self.layers['hiddenPost1'](x) )
    x = F.dropout(x,p=self.dr,training=self.training)
    
    x = self.layers['output'](x)
    return x


class CNNclassWithSigmoid(CNNclassO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, x):
        # Call the parent forward method to get the output tensor
        output = super().forward(x)
        
        # Apply the sigmoid activation function to the output tensor
        output = torch.sigmoid(output)
        
        return output

#########################################################################################
#########################################################################################