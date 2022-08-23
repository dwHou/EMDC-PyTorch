import os
import os.path as ops
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import scipy.io as scio
import random

import numpy as np
# from Model_define_pytorch import DatasetFolder
from torchvision import transforms
from utils import common
import torch.utils.data as data
import pickle
import random
import cv2
from PIL import Image
# from generateSpotMask import GenerateSpotMask
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torch


def get_training_set():
    train_txt = '/containers-shared/MIPI/data/train/train_lst/data_train.list'
    return mipi_dataset(train_txt, 'train')

def get_test_set():
    test_txt = '/containers-shared/MIPI/data/train/train_lst/data_fixedtest.list'
    return mipi_dataset(test_txt, 'test')

# MIPI Dataset
class mipi_dataset(data.Dataset):
    def __init__(self, txt_path, flag):
        super(mipi_dataset, self).__init__()
        
        self.flag = flag

        if self.flag != 'test':
            with open(txt_path, 'r') as fh:
                pairs = []
                for line in fh:
                    line = line.rstrip()
                    words = line.split()
                    pairs.append((words[0], words[1]))
                self.pairs = pairs
        elif self.flag == 'test':
            with open(txt_path, 'r') as fh:
                pairs = []
                for line in fh:
                    line = line.rstrip()
                    words = line.split()
                    pairs.append((words[0], words[1], words[2]))
                self.pairs = pairs
        
        
        height, width = (288, 384)
        # height, width = (360, 480) # Reserve resolution space for the scale jitter; 256 < height < 480
        crop_size = (256, 256)

        self.height = height
        self.width = width
        self.crop_size = crop_size
        
        self.max_depth = 10.0
        self.cutmask = True
        self.rgb_noise = 0.05
        self.noise = 0.01
        self.grid_spot = True
        self.num_spot = 1000
        self.cutmix = False
        
    def __getitem__(self, index):
        
        self.index = index
        
        if self.flag != 'test':
            np_rgb, np_tar = self._load_png(self.index)

            # from nparray to PIL
            rgb = Image.fromarray(np_rgb)
            dep = np_tar.astype(np.float32)
            dep[dep>self.max_depth] = 0
            dep = Image.fromarray(dep)
            
        elif self.flag == 'test':
            np_rgb, np_sp, np_tar = self._load_png(self.index)

            # from nparray to PIL
            rgb = Image.fromarray(np_rgb)
            
            dep = np_tar.astype(np.float32)
            dep[dep>self.max_depth] = 0
            dep = Image.fromarray(dep)
            
            dep_sp = np_sp.astype(np.float32)
            dep_sp = Image.fromarray(dep_sp)
        
        
        if self.flag == 'train':
            # data augment
            _scale = 1.0 # not by _scale, and hardcode by height, width = (288, 384)
            # _scale = np.random.uniform(0.75, 1.25)
            scale = np.int(self.height * _scale)
            angle = (np.random.rand()-0.5)*70

            rgb = TF.rotate(rgb, angle)
            dep = TF.rotate(dep, angle)

            hflip = np.random.uniform(0.0, 1.0)
            vflip = np.random.uniform(0.0, 1.0)
            if hflip > 0.5:
                rgb = TF.hflip(rgb)
                dep = TF.hflip(dep)
            if vflip > 0.5:
                rgb = TF.vflip(rgb)
                dep = TF.vflip(dep)

            t_rgb = T.Compose([
                T.Resize(scale),
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

            t_dep = T.Compose([
                # T.Resize(scale, Image.NEAREST),
                T.Resize(scale, T.InterpolationMode.NEAREST),
                self.ToNumpy(),
                T.ToTensor()
            ])

            rgb = t_rgb(rgb)
            dep = t_dep(dep)

            _, height_nocrop, width_nocrop = rgb.shape
            top = np.random.randint(0, height_nocrop - self.crop_size[0])
            left = np.random.randint(0, width_nocrop - self.crop_size[1])
            rgb = rgb[:, top:(top+self.crop_size[0]), left:(left+self.crop_size[1])]
            dep = dep[:, top:(top+self.crop_size[0]), left:(left+self.crop_size[1])]

            # _scale_true = _scale * 0.75 # (480 / 360)
            # Foreshortening effects
            # dep = dep / _scale_true
            
            rgb_n = np.random.uniform(0.0, 1.0)
            if rgb_n > 0.2 and self.rgb_noise > 0:
                rgb_noise = torch.normal(mean=torch.zeros_like(rgb), std=self.rgb_noise * np.random.uniform(0.5, 1.5))
                rgb = rgb + rgb_noise

            if self.noise:
                reflection = np.clip(np.random.normal(1, scale=0.333332, size=(1,1)), 0.01, 3)[0,0]
                noise = torch.normal(mean=0.0, std=dep * reflection * self.noise)
                dep_noise = dep + noise
                dep_noise[dep_noise < 0] = 0
            else:
                dep_noise = dep.clone()

            if self.grid_spot:
                dep_sp = self.get_sparse_depth_grid(dep_noise)
            else:
                dep_sp = self.get_sparse_depth(dep_noise, self.num_spot)

            if self.cutmask:
                dep_sp = self.cut_mask(dep_sp)

            gt = dep
            
            
        elif self.flag == 'test':
            t_rgb = T.Compose([
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            t_dep = T.Compose([
                self.ToNumpy(),
                T.ToTensor()
            ])

            rgb = t_rgb(rgb)
            gt = t_dep(dep)
            dep_sp = t_dep(dep_sp)
        
        # output = {'rgb': rgb, 'dep': dep_sp, 'gt': dep}
        return rgb, dep_sp, gt

    def __len__(self):
        self.length= len(self.pairs)
        return len(self.pairs)


    def _load_png(self, index):
        if self.flag != 'test':
            input, target = self.pairs[index]
            input = ops.join('/containers-shared/MIPI/data/train', input)
            target = ops.join('/containers-shared/MIPI/data/train', target)
            
            np_in = cv2.imread(input)
            np_in = cv2.cvtColor(np_in, cv2.COLOR_BGR2RGB)
            np_tar = cv2.imread(target, cv2.IMREAD_ANYDEPTH)
        
        elif self.flag == 'test':
            input, dep_sp, target = self.pairs[index]
            input = ops.join('/containers-shared/MIPI/data/train', input)
            dep_sp = ops.join('/containers-shared/MIPI/data/train', dep_sp)
            target = ops.join('/containers-shared/MIPI/data/train', target)
            
            np_in = cv2.imread(input)
            np_in = cv2.cvtColor(np_in, cv2.COLOR_BGR2RGB)
            np_sp = cv2.imread(dep_sp, cv2.IMREAD_ANYDEPTH)
            np_tar = cv2.imread(target, cv2.IMREAD_ANYDEPTH)
            return np_in, np_sp, np_tar
            
            
        if self.cutmix and self.flag == 'train':
            idx = random.randint(0, self.length - 1)
            input_, target_ = self.pairs[idx]
            input_ = ops.join('/containers-shared/MIPI/data/train', input_)
            target_ = ops.join('/containers-shared/MIPI/data/train', target_)
            
            np_in_ = cv2.imread(input_)
            np_in_ = cv2.cvtColor(np_in_, cv2.COLOR_BGR2RGB)
            np_tar_ = cv2.imread(target_, cv2.IMREAD_ANYDEPTH)
             
            H, W, C = np_in.shape 
            H2, W2, C2 = np_in_.shape

            # random crop
            cut_ratio = np.sqrt(1. - np.random.beta(1., 1.)) * 0.8
            cut_w, cut_h = np.int(W * cut_ratio), np.int(H * cut_ratio)
            
            scale = 1
            # define x, y 
            cx, cy = np.random.randint(W), np.random.randint(H)
            
            bbx1, bby1 = np.clip(cx - cut_w // 2, 0, W), np.clip(cy - cut_h // 2, 0, H)
            bbx1, bby1 = np.clip(bbx1, 0, W2), np.clip(bby1, 0, H2)
            bbx2, bby2 = np.clip(cx + cut_w // 2, 0, W), np.clip(cy + cut_h // 2, 0, H)
            bbx2, bby2 = np.clip(bbx2, 0, W2), np.clip(bby2, 0, H2)

            np_in[bby1 // scale: bby2 // scale, bbx1 // scale: bbx2 // scale, :] = np_in_[bby1 // scale: bby2 // scale, bbx1 // scale: bbx2 // scale, :]  
            # np_tar[bby1: bby2, bbx1: bbx2, :] = np_tar_[bby1: bby2, bbx1: bbx2, :] 
            np_tar[bby1: bby2, bbx1: bbx2] = np_tar_[bby1: bby2, bbx1: bbx2]    
        
        return np_in, np_tar
    
    def get_sparse_depth(self, dep, num_spot):
        channel, height, width = dep.shape

        assert channel == 1

        idx_nnz = torch.nonzero(dep.view(-1) > 0.0001)

        num_idx = len(idx_nnz)
        idx_sample = torch.randperm(num_idx)[:num_spot]

        idx_nnz = idx_nnz[idx_sample[:]]

        mask = torch.zeros((channel*height*width))
        mask[idx_nnz] = 1.0
        mask = mask.view((channel, height, width))

        dep_sp = dep * mask.type_as(dep)

        return dep_sp

    def get_sparse_depth_grid(self, dep):
        '''
        Simulate pincushion distortion:
        --stride: 
        It controls the distance between neighbor spots7
        Suggest stride value:       5~10

        --dist_coef:
        It controls the curvature of the spot pattern
        Larger dist_coef distorts the pattern more.
        Suggest dist_coef value:    0 ~ 5e-5

        --noise:
        standard deviation of the spot shift
        Suggest noise value:        0 ~ 0.5
        '''
        # Generate Grid points
        channel, img_h, img_w = dep.shape
        assert channel == 1

        stride = np.random.randint(5,7)

        dist_coef = np.random.rand()*4e-5 + 1e-5
        noise = np.random.rand() * 0.3

        x_odd, y_odd = np.meshgrid(np.arange(stride//2, img_h, stride*2), np.arange(stride//2, img_w, stride))
        x_even, y_even = np.meshgrid(np.arange(stride//2+stride, img_h, stride*2), np.arange(stride, img_w, stride))
        x_u = np.concatenate((x_odd.ravel(),x_even.ravel()))
        y_u = np.concatenate((y_odd.ravel(),y_even.ravel()))
        x_c = img_h//2 + np.random.rand()*50-25
        y_c = img_w//2 + np.random.rand()*50-25
        x_u = x_u - x_c
        y_u = y_u - y_c       
        

        # Distortion
        r_u = np.sqrt(x_u**2+y_u**2)
        r_d = r_u + dist_coef * r_u**3
        num_d = r_d.size
        sin_theta = x_u/r_u
        cos_theta = y_u/r_u
        x_d = np.round(r_d * sin_theta + x_c + np.random.normal(0, noise, num_d))
        y_d = np.round(r_d * cos_theta + y_c + np.random.normal(0, noise, num_d))
        idx_mask = (x_d<img_h) & (x_d>0) & (y_d<img_w) & (y_d>0)
        x_d = x_d[idx_mask].astype('int')
        y_d = y_d[idx_mask].astype('int')

        spot_mask = np.zeros((img_h, img_w))
        spot_mask[x_d,y_d] = 1

        dep_sp = torch.zeros_like(dep)
        dep_sp[:, x_d, y_d] = dep[:, x_d, y_d]

        return dep_sp

    def cut_mask(self, dep):
        _, h, w = dep.size()
        c_x = np.random.randint(h/4, h/4*3)
        c_y = np.random.randint(w/4, w/4*3)
        r_x = np.random.randint(h/4, h/4*3)
        r_y = np.random.randint(h/4, h/4*3)

        mask = torch.zeros_like(dep)
        min_x = max(c_x-r_x, 0)
        max_x = min(c_x+r_x, h)
        min_y = max(c_y-r_y, 0)
        max_y = min(c_y+r_y, w)
        mask[0, min_x:max_x, min_y:max_y] = 1
        
        return dep * mask
    
    class ToNumpy:
        def __call__(self, sample):
            return np.array(sample)
    
    
    
