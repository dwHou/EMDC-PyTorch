import os
import random

import torch
import numpy as np
from torchvision import transforms as T
from torchvision.transforms import functional as F
import cv2
import shutil

def backup_source_code(backup_directory):
    ignore_hidden = shutil.ignore_patterns(
        ".", "..", ".git*", "*pycache*", "*build", "*.fuse*", "*_drive_*",
        "*pretrained*", "*pth*", "*backup*", "*checkpoint*", "*png", "*jpg", ".exr", "*logs*")

    if os.path.exists(backup_directory):
        shutil.rmtree(backup_directory)

    shutil.copytree('.', backup_directory, ignore=ignore_hidden)
    os.system("chmod -R g+w {}".format(backup_directory))

def save_checkpoint(prefix: str, state: dict, is_best: bool, is_best_s: bool, n_epoch: int, ext: str='checkpoint.pth.tar'):
    file_name = '_'.join((prefix, ext))
    torch.save(state, file_name)
    if is_best:
        best_name = '_'.join((prefix, 'best', ext))
        shutil.copyfile(file_name, best_name)
    if is_best_s:
        best_name_s = '_'.join((prefix, 'best_static', ext))
        shutil.copyfile(file_name, best_name_s)
    # if (n_epoch % 25) == 0:
    #     ckpstone_name = '_'.join((prefix, f'epoch{n_epoch}', ext))
    #     shutil.copyfile(file_name, ckpstone_name)
    if (n_epoch > 1) :
        ckpstone_name = '_'.join((prefix, f'epoch{n_epoch}', ext))
        shutil.copyfile(file_name, ckpstone_name)

def set_random_seed(random_seed: int=0):
    """Set random seed to reproduce the training.
    
    After Pytorch updates in 3/19/21, DDP has already made initial states the same across multi-gpus.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(random_seed)    # set gpu seed deterministic
        # fix convolution calculate methods
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  

def get_patch(img_in, img_tar, patch_size, scale=3, multi_scale=False):
    # T H W C
    multi_frames = len(img_in.shape) == 4

    ih, iw = img_in.shape[-3:-1]
    p = scale if multi_scale else 1  
    tp = p * patch_size
    ip = tp // scale

    ix = random.randrange(0, iw - ip + 1)
    iy = random.randrange(0, ih - ip + 1)
    tx, ty = scale * ix, scale * iy

    if multi_frames:
        img_in = img_in[:, iy:iy + ip, ix:ix + ip, :]
    else:
        img_in = img_in[iy:iy + ip, ix:ix + ip, :]

    img_tar = img_tar[ty:ty + tp, tx:tx + tp, :]

    return img_in, img_tar

def np2Tensor(l, rgb_range):
    def _np2Tensor(img):
        multi_frames = len(img.shape) == 4
        if multi_frames:
            np_transpose = np.ascontiguousarray(img.transpose((0, 3, 1, 2)))
        else:
            np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))  # https://zhuanlan.zhihu.com/p/59767914
        tensor = torch.from_numpy(np_transpose).float() / rgb_range
        return tensor

    return [_np2Tensor(_l) for _l in l]

def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img

class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target
    
class RandomResize:
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size)
        target = F.resize(target, size)
        return image, target

'''
def augment(l, flip=True, rot=True, crop=True, resize=True):
    hflip = flip and random.random() < 0.5
    vflip = flip and random.random() < 0.5
    rot90 = rot and random.random() < 0.5
    crop = crop and random.random() < 0.5
    resize = resize and random.random() < 0.5

    def _augment(img):
        multi_frames = len(img.shape) == 4
        if multi_frames:
            if hflip: img = img[:, :, ::-1, :]
            if vflip: img = img[:, ::-1, :, :]
            if rot90: img = img.transpose(0, 2, 1, 3)
        else:
            if hflip: img = img[:, ::-1, :]
            if vflip: img = img[::-1, :, :]
            if rot90: img = img.transpose(1, 0, 2)

        return img

    return [_augment(_l) for _l in l]
'''

def augment(l, flip=True, rot=True, crop=True, resize=True):
    hflip = flip and random.random() < 0.5
    vflip = flip and random.random() < 0.5
    rotate = rot and random.random() < 0.5
    crop = crop and random.random() < 0.5
    resize = resize and random.random() < 0.5

    def _augment(img):
        multi_frames = len(img.shape) == 4
        if multi_frames:
            if hflip: img = img[:, :, ::-1, :]
            if vflip: img = img[:, ::-1, :, :]
            if rotate: img = img.transpose(0, 2, 1, 3)
        else:
            if hflip: img = img[:, ::-1, :]
            if vflip: img = img[::-1, :, :]
            if rotate: img = img.transpose(1, 0, 2)

        return img

    return [_augment(_l) for _l in l]


if __name__ == '__main__':
    pass