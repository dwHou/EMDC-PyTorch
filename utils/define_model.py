import os, fnmatch
import sys
import os.path as ops
import scipy.io as scio
from math import log10

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import random

def define_model(arch):
    # 1. define model structure
    if arch == 'EMDC':
        from models.EMDC import dcmodel
        model = dcmodel().cuda()
    
    elif arch == 'CSPN':
        from models.CSPN import dcmodel
        model = dcmodel().cuda()
        
    elif arch == 'PENet':
        from models.PENet import dcmodel
        model = dcmodel().cuda()

    else:
        print('Unsupported model structure!')
        sys.exit(1)
    
    return model