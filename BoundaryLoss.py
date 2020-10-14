#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pasted on Tue Oct 13 18:59:03 2020
From Kervadec et al. 2019
https://github.com/LIVIAETS/surface-loss/blob/master/losses.py

Original:

class SurfaceLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, dist_maps: Tensor, _: Tensor) -> Tensor:
        assert simplex(probs)
        assert not one_hot(dist_maps)

        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multipled = einsum("bcwh,bcwh->bcwh", pc, dc)

        loss = multipled.mean()

        return loss
    
"""

import torch
import numpy as np
from scipy.ndimage import distance_transform_edt as distance

class SurfaceLoss():
    def __init__(self, classs=1):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc = classs

    def __call__(self, probs, ground_truth):

        pc = probs[:, self.idc, ...].type(torch.cuda.FloatTensor)
        dc = ground_truth[:, self.idc, ...].type(torch.cuda.FloatTensor)

        multipled = torch.einsum("bcwh,bcwh->bcwh", pc, dc)

        loss = multipled.mean()

        return loss

# fun fact: it can be useful even omitting this part
def one_hot2dist(seg):
    # assert one_hot(torch.Tensor(seg), axis=0)
    C: int = len(seg)

    res = np.zeros_like(seg)
    for c in range(C):
        posmask = seg[c].astype(np.bool)

        if posmask.any():
            negmask = ~posmask
            res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res
