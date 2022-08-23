import os, fnmatch
import sys
import os.path as ops
import scipy.io as scio
from math import log10

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import argparse
import numpy as np

from utils.common import *
from utils.loss import *
from utils.dataset import *
from utils.scheduler import *
from utils.define_model import define_model
from utils.metric import Metric
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
from torch_ema import ExponentialMovingAverage
import random


def main(args):
    print('Hyper-parameter Settings:', args)

    # 0. tensorboard log and training code backup
    log_name = '_'.join((args.arch, args.name))
    writer = SummaryWriter(log_dir=ops.join('logs', log_name))
    backup_source_code(args.save_dir)
    try:
        os.makedirs(args.save_dir, exist_ok=True)
    except OSError:
        pass
    
    # 1. define model structure
    model = define_model(args.arch)
    
    # 2. define dataset and dataloader
    train_set = get_training_set()
    test_set = get_test_set()

    print('Size of train and val dataset:', len(train_set))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.train_batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.val_batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 3. define loss, optimizer, scheduler
    criterion1 = L1Loss()
    criterion1_ = BerhuLoss()
    criterion1__ = BMCLoss(init_noise_sigma=1., device='cuda')
    criterion2 = L2Loss()
    criterion3 = RMAEloss()
    criterion4 = GradientLoss()
    criterion5 = Sparse_Loss()
    
    if args.c2f:
        ignored_params = list(map(id, model.refine.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params, model.parameters())
    
        optimizer = torch.optim.AdamW(
                [
                    {'params': base_params, 'lr': 0.01 * args.learning_rate},
                    {'params': model.refine.parameters(), 'lr': args.learning_rate},
                    ])
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    n_iters_every_epoch = len(train_loader)    # iterations of every epoch
    
    if args.ckp_path != None:
        if args.c2f:
            print("loading coarse model")
            ckp_path = "./checkpoints/" + args.ckp_path
            ckp = torch.load(ckp_path)
            model.trunk.load_state_dict(ckp['state_dict'])
            
        else:
            print("loading depth completion model")
            ckp_path = "./checkpoints/" + args.ckp_path
            ckp = torch.load(ckp_path)
            model.load_state_dict(ckp['state_dict'])

    # EMA setting change 
    # pip show torch_ema (1 + self.num_updates) / (3600 + self.num_updates)
    if args.ckp_path != None:
        ema = ExponentialMovingAverage(model.parameters(), decay=0.9995, use_num_updates=False)
    else:
        ema = ExponentialMovingAverage(model.parameters(), decay=0.9995)
           
    if args.decay_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs * n_iters_every_epoch)
    elif args.decay_type == 'warmup_cosine':
        scheduler = WarmupCosineSchedule(
            optimizer, warmup_steps=args.num_warmup_epochs * n_iters_every_epoch, t_total=args.num_epochs * n_iters_every_epoch,
        )
    elif args.decay_type == 'step':
        scheduler = MultiStepLR(
            optimizer, milestones=[
                (args.num_epochs // 3) * n_iters_every_epoch, 
                (args.num_epochs // 3) * 2 * n_iters_every_epoch, 
                (args.num_epochs // 3) * 3 * n_iters_every_epoch,
            ], gamma=0.1)
    else:
        print('Unsupport learning rate scheculer!')
        sys.exit(1)
    
    def train(n_epoch):
        # model training
        model.train()
        total_loss = 0.
        epoch_iterator = tqdm(train_loader, bar_format="{l_bar}|[{elapsed}]", dynamic_ncols=True, disable=args.local_rank not in [-1, 0])
        for i, batch_x in enumerate(epoch_iterator):
            input, dep, target =  batch_x[0].cuda(), batch_x[1].cuda(), batch_x[2].cuda()

            optimizer.zero_grad()
            
            # forward
            output, y_loc, y_glb = model(input, dep)
            
            if 'criterion1' in args.loss:
                loss1 = criterion1(output, target)
                loss = loss1
                loss_sam1 = criterion1(y_loc, target)
                loss += loss_sam1/(loss_sam1/loss1).detach()
                loss_sam2 = criterion1(y_glb, target)
                loss += loss_sam2/(loss_sam2/loss1).detach()
                
            if 'criterion2' in args.loss:
                loss2 = criterion2(output, target)
                loss += loss2
                
            if 'criterion3' in args.loss:
                loss3 = criterion3(output, target) 
                loss += loss3
                
            if 'criterion4' in args.loss:
                loss4 = criterion4(output, target) 
                loss += 0.7 * loss4/(loss4/loss1).detach() # default 0.2
            
            # backpropagate
            if loss < 1000: 
                loss.backward()
                optimizer.step()
                ema.update()
                total_loss += (loss.detach().item()) * target.shape[0]
                
            elif loss > 1000:
                print("loss>1000, and value is", loss)
            
            scheduler.step()
            
            if (i + 1) % 5 == 0 or (i + 1) == n_iters_every_epoch:
                epoch_iterator.set_description(
                    'Training epoch {}: {} / {}, training Loss {loss:.8f}\t lr: {lr:.5f}'.format(
                        n_epoch, i + 1, n_iters_every_epoch, loss=total_loss / len(train_set), lr=optimizer.param_groups[-1]['lr'],
                    )
                )
        # recording train metrics
        writer.add_scalar('train/loss', scalar_value=total_loss / len(train_set), global_step=n_epoch)
        writer.add_scalar('train/lr', scalar_value=optimizer.param_groups[-1]['lr'], global_step=n_epoch)
        if args.c2f:
            writer.add_scalar('train/lr_pretrained_trunk', scalar_value=optimizer.param_groups[-2]['lr'], global_step=n_epoch)

    def test(n_epoch):
        
        metric = Metric()
        
        # model evaluating
        with ema.average_parameters():
            
            cost_time = 0
            avg_metric = 0
            
            invalid_cnt = 0
            model.eval()
            total_loss = 0.
            preds, gts = list(), list()
            with torch.no_grad():
                for i, batch_x in enumerate(val_loader):
                    input, dep, target =  batch_x[0].cuda(), batch_x[1].cuda(), batch_x[2].cuda()
                    
                    torch.cuda.synchronize()
                    begin = time.time()
                    output, _, _ = model(input, dep)
                    torch.cuda.synchronize()  
                    end = time.time()
                    cost_time += end - begin 
                    
                    total_loss += criterion1(output, target).item() * target.shape[0]                  
                    
                    metric_val = metric.evaluate(output, dep, target)
                    if (i % 100) == 0:
                        print('Official_score:', metric_val)
                    if (metric_val < -1000):
                        output = output[0, 0, :, :].cpu().numpy().astype(np.float32)
                        depth=output
                        depth[depth>20] = 0 
                        vmin, vmax = depth.min(), depth.max()
                        depth_int = (depth*255/(vmax - vmin)).astype(np.uint8)
                        im_color = cv2.applyColorMap(depth_int, cv2.COLORMAP_JET)
                        output_jet_path =  './debug/' + f'{i}' + '.png'
                        cv2.imwrite(output_jet_path,im_color)

                    if metric_val > -100.0:
                        avg_metric += metric_val
                    else:
                        invalid_cnt += 1
                
                '''
                same_scene = []
                for i, batch_x in enumerate(val_loader2):
                    input, dep =  batch_x[0].cuda(), batch_x[1].cuda()
                    
                    torch.cuda.synchronize()
                    begin = time.time()
                    output, _, _ = model(input, dep)
                    torch.cuda.synchronize()  
                    end = time.time()
                    cost_time += end - begin 
                    
                    total_loss += 0   

                    same_scene.append(output)
                    preddeps = torch.cat(same_scene, dim=0)
                    metric_val = metric.evaluate(output, dep, dep, isStaticRDS=True)
                    if metric_val > -100.0:
                        avg_metric_rds += metric_val
                    else:
                        invalid_cnt += 1
   
                    if (i+1) % 25 == 0:
                        metric_val = metric.evaluate(preddeps, preddeps, preddeps, isStaticRTSD=True)
                        print('RTSD_score:', metric_val)
                        avg_metric_rtsd += metric_val
                        same_scene = []
                '''
                        
                print("valid sample number is: ", (len(test_set) - invalid_cnt))
                # evaluate on validation dataset
                metric_score = avg_metric / (len(test_set) - invalid_cnt)
                average_loss = total_loss / len(test_set) 
                average_time = cost_time * 1000 / len(test_set)
                
                print('Epoch {} Validation Loss: {:.3f}, score:{:.4f}, time:{:.2f}ms per frame'.format(n_epoch, average_loss, metric_score, average_time))
                
        # record validation metrics
        writer.add_scalar('val/loss', scalar_value=average_loss, global_step=n_epoch)
        writer.add_scalar('val/Metric', scalar_value=metric_score, global_step=n_epoch)
        return metric_score

    # Start Training
    best_metric = 0    # set initial score
    for n_epoch in range(args.num_epochs):
        train(n_epoch)
        with ema.average_parameters():
            metric_score = test(n_epoch)
            
        with ema.average_parameters():
            is_best = metric_score > best_metric
            is_best_s = False
            best_metric = max(metric_score, best_metric)
            state = {
                'epoch': n_epoch,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc': best_metric
            }
            save_checkpoint(ops.join('checkpoints', log_name), state, is_best, is_best_s, n_epoch)

    print('Done! Best MIPI score: {:.5f}'.format(best_metric))

if __name__ == '__main__':
    # retrieve necessary hyper-parameters
    parser = argparse.ArgumentParser()

    # define model and log path
    parser.add_argument("--arch", required=True, help="Decide the model structure")
    parser.add_argument("--name", required=True, help="Identifier of this train mission.")
    parser.add_argument("--output_dir", default="checkpoints", type=str, help="The output directory where checkpoints will be written.")

    # training hyper-parameters
    parser.add_argument("--train_batch_size", default=128, type=int, help="Total batch size for training.")
    parser.add_argument("--val_batch_size", default=1, type=int, help="Total batch size for eval.")
    parser.add_argument("--ckp_path", default=None, help="The path of ckp file.")
    parser.add_argument("--c2f", default=0, type=int, help="coarse to fine.")
    # parser.add_argument("--topk", default=0, type=int, help="top K loss.")

    # learning scheduler and optimizer parameters
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--num_epochs", default=300, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "warmup_cosine", "warmup_linear", "step"], default="cosine", help="How to decay the learning rate.")
    parser.add_argument("--num_warmup_epochs", default=5, type=int, help="Steps of training to perform learning rate warmup for.")
    parser.add_argument("--gpu", default='0', type=str, help="Select single GPU idx.")
    
    # loss function
    parser.add_argument("--loss", required=True, action='append', help="losses used")
    
    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu    # set the running GPU
    args.local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else -1    # retain for further DDP
    # torch.manual_seed(1)

    set_random_seed(42)
    
    current_time = time.strftime('%y%m%d_%H%M%S_')
    save_dir = os.path.join('./code_backup/', current_time+'_'.join((args.arch, args.name)))
    args.save_dir = save_dir

    main(args)