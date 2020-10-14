#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 11:17:35 2020

just a bunch of CE losses
"""
import torch

"""
A CCE loss with added weight on boundaries, implemented from 
Wachinger et al. 2017 DeepNAT
"""
EPS=1e-10 # log offset to avoid log(0)

Z1=torch.tensor([[[ 1,  1, 1],
                     [ 1,  2, 1],
                     [ 1,  1, 1]],
            
                    [[ 0,  0, 0],
                     [ 0,  0, 0],
                     [ 0,  0, 0]],
            
                    [[ -1,  -1, -1],
                     [ -1,  -2, -1],
                     [ -1,  -1, -1]]],
                     requires_grad=False)#.type(torch.DoubleTensor)
    
X1=torch.tensor([[[ 1,  0, -1],
                     [ 1,  0, -1],
                     [ 1,  0, -1]],
            
                    [[ 1,  0, -1],
                     [ 2,  0, -2],
                     [ 1,  0, -1]],
            
                    [[ 1,  0, -1],
                     [ 1,  0, -1],
                     [ 1,  0, -1]]],
                     requires_grad=False)#.type(torch.DoubleTensor)
Y1=torch.tensor([[[ 1,  1, 1],
                     [ 0,  0, 0],
                     [ -1,  -1, -1]],
            
                    [[ 1,  2, 1],
                     [ 0,  0, 0],
                     [ -1,  -2, -1]],
            
                    [[ 1,  1, 1],
                     [ 0,  0, 0],
                     [ -1,  -1, -1]]],
                     requires_grad=False)#.type(torch.DoubleTensor)
    
def Sobel(Convolveme):
    
    X=X1.reshape(1,1,3,3,3).type(torch.cuda.FloatTensor).expand(Convolveme.shape[1], -1,-1,-1,-1)
    Y=Y1.reshape(1,1,3,3,3).type(torch.cuda.FloatTensor).expand(Convolveme.shape[1], -1,-1,-1,-1)
    
    Z=Z1.reshape(1,1,3,3,3).type(torch.cuda.FloatTensor).expand(Convolveme.shape[1], -1,-1,-1,-1)
    Xconv=torch.nn.functional.conv3d(Convolveme,X,groups=Convolveme.shape[1])
    Yconv=torch.nn.functional.conv3d(Convolveme,Y,groups=Convolveme.shape[1])
    Zconv=torch.nn.functional.conv3d(Convolveme,Z,groups=Convolveme.shape[1])
    conv=torch.abs(torch.nn.functional.pad(Xconv+Yconv+Zconv,(1,1,1,1,1,1)))
    conv[conv>0]=1
    
        
    return conv

def CCEBoundary(Ytrue,Ypred,CatW,SobW=0):
    shape=Ytrue.shape
    CCE=-torch.mul(Ytrue,torch.log(Ypred + EPS))
    W=torch.tensor(CatW).reshape((1,len(CatW),1,1,1)).expand(shape).float()
    W=W*(1+Sobel(Ytrue)*SobW)
    
    W.requires_grad=False
    
    wCCE=torch.mul(W,CCE)
    
    return torch.mean(wCCE)

def BCEBoundary(Ytrue,Ypred,W0,W1):
    
    ''' 
    Returns binary cross entropy + dice loss for one 3D volume, normalized
    W0: added weight on region border
    W1: base class weight for Binary Cross Entropy, should depend on frequency
    '''
    shape=Ytrue.shape
    BCE = - torch.mul((torch.ones(shape,requires_grad=False)-Ytrue), torch.log(torch.ones(shape,requires_grad=False)-Ypred + torch.ones(shape,requires_grad=False)*EPS))-torch.mul(Ytrue,torch.log(Ypred + torch.ones(shape,requires_grad=False)*EPS))
    W1b=torch.tensor(W1).reshape((1,1,1,1)).expand(shape)
    W0b=torch.tensor(W0).reshape((1,1,1,1)).expand(shape)
    
    W=W0b*Sobel(Ytrue)+W1b*torch.ones(shape,requires_grad=False)
    wBCE=torch.mul(W,BCE)
    
    return torch.mean(wBCE)