#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pasted on Tue Oct 13 18:59:03 2020

"""

import torch

class GeneralizedDice():
    """
    Using the implementation from Kervadec et al. 2019
    https://github.com/LIVIAETS/surface-loss/blob/master/losses.py
class GeneralizedDice():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor, _: Tensor) -> Tensor:
        assert simplex(probs) and simplex(target)

        pc = probs[:, self.idc, ...].type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        w: Tensor = 1 / ((einsum("bcwh->bc", tc).type(torch.float32) + 1e-10) ** 2)
        intersection: Tensor = w * einsum("bcwh,bcwh->bc", pc, tc)
        union: Tensor = w * (einsum("bcwh->bc", pc) + einsum("bcwh->bc", tc))

        divided: Tensor = 1 - 2 * (einsum("bc->b", intersection) + 1e-10) / (einsum("bc->b", union) + 1e-10)

        loss = divided.mean()

        return loss
    """
    def __init__(self, classs=(0,1)):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc = classs

    def __call__(self, probs, target):
        
        

        pc = probs[:, self.idc, ...].type(torch.cuda.FloatTensor)
        tc = target[:, self.idc, ...].type(torch.cuda.FloatTensor)

        
        w = 1 / ((torch.einsum("bcdwh->bc", tc).type(torch.cuda.FloatTensor) + 1e-10) ** 2)
        intersection = w * torch.einsum("bcdwh,bcdwh->bc", pc, tc)
        union = w * (torch.einsum("bcdwh->bc", pc) + torch.einsum("bcdwh->bc", tc))

        divided = 1 - 2 * (torch.einsum("bc->b", intersection) + 1e-10) / (torch.einsum("bc->b", union) + 1e-10)

        loss = divided.mean()

        return loss
    
def DiceLoss(Ytrue,Ypred):
    '''
    Just a basic Dice loss implementation
    '''

    DICE = -torch.div( torch.sum(torch.mul(torch.mul(Ytrue,Ypred),2)), torch.sum(torch.mul(Ypred,Ypred)) + torch.sum(torch.mul(Ytrue,Ytrue))+1)
    
    return DICE