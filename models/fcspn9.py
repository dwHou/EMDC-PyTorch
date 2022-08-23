#!/usr/bin/env python
#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Author  : Devonn Hou
@Email   : devonn.hou@zoom.us
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class AffinityPropagate1(nn.Module):

    def __init__(self, prop_time):
        super(AffinityPropagate1, self).__init__()
        self.times = prop_time
        # self.dilation = dilation

    def forward(self, guided1,guided2,guided3, fuse, x, sparse_depth=None):
        """
        :param x:        Feature maps, N,C,H,W
        :param guided:   guided Filter, N, K^2, H, W, K is kernel size
        :return:         returned feature map, N, C, H, W
        """

        self.fuse = fuse # B,3,H,W
        B, C, H, W = guided1.size()
        K = int(math.sqrt(C))

        # Normalization
        guided1 = F.softmax(guided1, dim=1)
        guided2 = F.softmax(guided2, dim=1)
        guided3 = F.softmax(guided3, dim=1)

        kernel1 = guided1
        kernel2 = guided2
        kernel3 = guided3
        

        # kernel = kernel.unsqueeze(dim=1).reshape(B, 1, K, K, H, W)
        kernel1 = kernel1.unsqueeze(dim=1).reshape(B, K*K, H*W)
        kernel2 = kernel2.unsqueeze(dim=1).reshape(B, K*K, H*W)
        kernel3 = kernel3.unsqueeze(dim=1).reshape(B, K*K, H*W)


        if sparse_depth is not None:
            sparse_mask = sparse_depth.sign()
            _x = x

        for _ in range(self.times):
            '''
            Convolution is equivalent with Unfold + Matrix Multiplication + Fold 
            Einsum can act as Matrix Multiplication + Fold 
            '''
            inp = x 
            inp_unf1 = torch.nn.functional.unfold(inp, kernel_size = (3, 3), dilation=1, padding=1, stride=1) 
            inp_unf2 = torch.nn.functional.unfold(inp, kernel_size = (3, 3), dilation=2, padding=2, stride=1) 
            inp_unf3 = torch.nn.functional.unfold(inp, kernel_size = (3, 3), dilation=3, padding=3, stride=1) 
            
            output1 = torch.einsum('ijk,ijk->ik', (inp_unf1, kernel1))
            output2 = torch.einsum('ijk,ijk->ik', (inp_unf2, kernel2))
            output3 = torch.einsum('ijk,ijk->ik', (inp_unf3, kernel3))
            
            x = output1.view(B, 1, H, W) * self.fuse[:, 0:1, ...] + output2.view(B, 1, H, W) * self.fuse[:, 1:2, ...] + output3.view(B, 1, H, W) * self.fuse[:, 2:, ...]

            if sparse_depth is not None:
                no_sparse_mask = 1 - sparse_mask
                x = sparse_mask * _x + no_sparse_mask * x
        return x
    
    
class AffinityPropagate2(nn.Module):

    def __init__(self, prop_time):
        super(AffinityPropagate2, self).__init__()
        self.times = prop_time
        # self.dilation = dilation

    def forward(self, guided1,guided2, fuse, x, sparse_depth=None):
        """
        :param x:        Feature maps, N,C,H,W
        :param guided:   guided Filter, N, K^2, H, W, K is kernel size
        :return:         returned feature map, N, C, H, W
        """

        self.fuse = fuse # B,3,H,W
        B, C, H, W = guided1.size()
        K = int(math.sqrt(C))

        guided1 = F.softmax(guided1, dim=1)
        guided2 = F.softmax(guided2, dim=1)

        kernel1 = guided1
        kernel2 = guided2

        kernel1 = kernel1.unsqueeze(dim=1).reshape(B, K*K, H*W)
        kernel2 = kernel2.unsqueeze(dim=1).reshape(B, K*K, H*W)

        if sparse_depth is not None:
            sparse_mask = sparse_depth.sign()
            _x = x

        for _ in range(self.times):
            '''
            Convolution is equivalent with Unfold + Matrix Multiplication + Fold 
            Einsum can act as Matrix Multiplication + Fold 
            '''
            inp = x 
            inp_unf1 = torch.nn.functional.unfold(inp, kernel_size = (3, 3), dilation=1, padding=1, stride=1) 
            inp_unf2 = torch.nn.functional.unfold(inp, kernel_size = (3, 3), dilation=2, padding=2, stride=1) 

            output1 = torch.einsum('ijk,ijk->ik', (inp_unf1, kernel1))
            output2 = torch.einsum('ijk,ijk->ik', (inp_unf2, kernel2))
            x = output1.view(B, 1, H, W) * self.fuse[:, 0:1, ...] + output2.view(B, 1, H, W) * self.fuse[:, 1:2, ...]

            if sparse_depth is not None:
                no_sparse_mask = 1 - sparse_mask
                x = sparse_mask * _x + no_sparse_mask * x
      
        return x
    
    
class AffinityPropagate3(nn.Module):

    def __init__(self, prop_time):
        super(AffinityPropagate3, self).__init__()
        self.times = prop_time

    def forward(self, guided, x, sparse_depth=None):
        """
        :param x:        Feature maps, N,C,H,W
        :param guided:   guided Filter, N, K^2, H, W, K is kernel size
        :return:         returned feature map, N, C, H, W
        """
        B, C, H, W = guided.size()
        K = int(math.sqrt(C))

        # Normalization
        guided = F.softmax(guided, dim=1)
        kernel = guided
        
        kernel = kernel.unsqueeze(dim=1).reshape(B, K*K, H*W)


        if sparse_depth is not None:
            sparse_mask = sparse_depth.sign()
            _x = x

        for _ in range(self.times):
            '''
            Convolution is equivalent with Unfold + Matrix Multiplication + Fold 
            Einsum can act as Matrix Multiplication + Fold 
            '''
            inp = x
            inp_unf = torch.nn.functional.unfold(inp, kernel_size = (3, 3), dilation=1, padding=1, stride=1) 
            output = torch.einsum('ijk,ijk->ik', (inp_unf, kernel))
            x = output.view(B, 1, H, W)

            if sparse_depth is not None:
                no_sparse_mask = 1 - sparse_mask
                x = sparse_mask * _x + no_sparse_mask * x
                
        return x
