#!/usr/bin/env python

import torch.nn as nn
import torch
import math
import os, fnmatch
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import argparse
import time
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from define_model import define_model


def plt_visualize(rgb, depth, depsp, jet_path):
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    depth[depth>20] = 0
    vmin, vmax = depth.min(), depth.max()
    spot_depth = depsp
    
    x_sp, y_sp = np.where(spot_depth>0)
    d_sp = spot_depth[x_sp, y_sp]

    title_names = ['RGB', 'EMDC', 'spot depth']
    fig = plt.figure(figsize=(14,4))
    axs = ImageGrid(fig, 111,
                    nrows_ncols = (1,3),
                    axes_pad = 0.05,
                    cbar_location = "right",
                    cbar_mode="single",
                    cbar_size="5%",
                    cbar_pad=0.05
                    )
    axs[0].imshow(rgb)
    axs[1].imshow(depth, cmap='jet_r', vmin=vmin, vmax=vmax)
    imc = axs[2].scatter(y_sp,x_sp,np.ones_like(x_sp)*0.1,c=d_sp,cmap='jet_r', vmin=vmin, vmax=vmax)
    axs[2].axis([0,256,192,0])
    asp = abs(np.diff(axs[2].get_xlim())[0] / np.diff(axs[2].get_ylim())[0]) *192/256
    axs[2].set_aspect(asp)
    for ii, title_name in enumerate(title_names):
        axs[ii].set_title(title_name, fontsize=12)
        axs[ii].set_xticks([])
        axs[ii].set_yticks([])
    cbar = plt.colorbar(imc, cax=axs.cbar_axes[0], ticks = np.linspace(vmin, vmax, 5), format='%.1f')
    cbar.ax.set_ylabel('Depth (m)')

    plt.savefig(jet_path)
    plt.close('all')


def visual(args):
    
    # 1. define model structure
    model = define_model(args.arch)
             
    print("this checkpoint reachs a score of: ", torch.load(f"{args.ckp_path}")['best_acc'])
    weight={k.replace('module.',''):v for k,v in torch.load(f"{args.ckp_path}")['state_dict'].items()}
    model.load_state_dict(weight, strict=False)
    model.eval()
    
    class ToNumpy:
        def __call__(self, sample):
            return np.array(sample)
        

    with open(args.txt_path, 'r') as fh:
        pairs = []
        for line in fh:
            line = line.rstrip()
            words = line.split()
            pairs.append((args.txt_path[:-9]+words[0], args.txt_path[:-9]+words[1]))
    
    
    with torch.no_grad():
        cost_time = 0
        flist = []
        for pair in pairs:
            print(pair)
            rgb_path, depsp_path = pair
            
            bgr = cv2.imread(rgb_path)
            np_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            np_dep = cv2.imread(depsp_path, cv2.IMREAD_ANYDEPTH)

            # from nparray to PIL
            rgb = Image.fromarray(np_rgb)
            dep = np_dep.astype(np.float32)
            dep[dep>20] = 0
            dep = Image.fromarray(dep)
            
            t_rgb = T.Compose([
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            t_dep = T.Compose([
                ToNumpy(),
                T.ToTensor()
            ])

            rgb = t_rgb(rgb).unsqueeze(0).cuda()
            dep_sp = t_dep(dep).unsqueeze(0).cuda()

            torch.cuda.synchronize()
            begin = time.time()
            if args.output_num == 1:
                output = model(rgb, dep_sp) 
            elif args.output_num == 2:
                output, _ = model(rgb, dep_sp)
            elif args.seemap == 1:
                output, fus = model(rgb, dep_sp) 
            elif args.output_num == 3:
                output, _, _ = model(rgb, dep_sp) 
            elif args.output_num == 5:
                output, _, _, _, _ = model(rgb, dep_sp) 
            
            torch.cuda.synchronize()
            end = time.time()
            cost_time += end - begin 
            output = output[0, 0, :, :].cpu().numpy().astype(np.float32)

            '''
            # output dense depth image
            exr_name = depsp_path.split('/')[-1]
            output_path = os.path.join('results', f'{args.out_path}', exr_name)
            cv2.imwrite(output_path, output)
            '''
            
            # output dense depth image
            exr_dir = depsp_path.split('/')[-2]
            exr_name = depsp_path.split('/')[-1]
            if not os.path.exists(os.path.join('results', f'{args.out_path}', exr_dir)):
                os.system(f"mkdir {os.path.join('results', f'{args.out_path}', exr_dir)}")
            output_path = os.path.join('results', f'{args.out_path}', exr_dir, exr_name)
            cv2.imwrite(output_path, output)
            
            if args.visualization:
                jet_path_o = output_path[:-4] + '_jet' + '.png'
                
                plt_visualize(np_rgb, output, np_dep, jet_path_o)
                
                if args.seemap == 1:
                    confidence_map = fus
                    
                    confidence_map1 = confidence_map * 255
                    confidence_map1 = confidence_map1[0, 0, :, :].cpu().numpy().astype(np.float32)
                    confidence_map2 = (1 - fus) * 255
                    confidence_map2 = confidence_map2[0, 0, :, :].cpu().numpy().astype(np.float32)
                    
                    output_jet_path = output_path[:-4] + '_map1' + '.png'
                    cv2.imwrite(output_jet_path,confidence_map1)
                    output_jet_path = output_path[:-4] + '_map2' + '.png'
                    cv2.imwrite(output_jet_path,confidence_map2)
                

            flist.append(exr_name) 
        
        with open(f'./results/{args.out_path}/data.list', 'w') as f:
            for item in flist:
                f.write("%s\n" % item)
            
        print(f"done! costtime {cost_time} for {len(pairs)} frames")
        print(f"{cost_time * 1000 / len(pairs)} ms per frames")
        
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        with open(f'./results/{args.readme_path}/readme.txt', 'w') as f:
            f.write(f"Runtime per image [s] : {cost_time / len(pairs)}\n")
            f.write(f"Parameters : {pytorch_total_params}\n")
            f.write(f"Extra Data [1] / No Extra Data [0] : 0\n")
            f.write(f"Other description : GPU: A6000; Pretraind model: from https://github.com/tonylins/pytorch-mobilenet-v2")     
            

if __name__ == "__main__":
    
    # retrieve necessary hyper-parameters
    parser = argparse.ArgumentParser()

    # define model and log path
    parser.add_argument("--arch", required=True, help="Decide the model structure")
    parser.add_argument("--txt_path", default='./Inputs/test.txt', help="The path of test.txt.")
    parser.add_argument("--out_path", default='./results/test.txt', help="The path of out.txt.")
    parser.add_argument("--readme_path", default='./results/test.txt', help="The path of readme.txt.")
    parser.add_argument("--ckp_path", default=None, help="The path of ckp file.")
    parser.add_argument("--gpu", default='0', type=str, help="Select single GPU idx.")
    parser.add_argument("--visualization", required=True, default=0, type=int, help="use lsgan")
    parser.add_argument("--seemap", required=True, default=0, type=int, help="see map")
    parser.add_argument("--output_num", required=True, default=1, type=int, help="num of outputs")
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu    # set the running GPU
    args.local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else -1    # retain for further DDP
    
    visual(args)
