import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class GradientLoss(nn.Module):

    def __init__(self):
        super(GradientLoss, self).__init__()
        self.depth_valid1 = 0.001

    def forward(self, sr, hr):
    
        mask1 = (hr > self.depth_valid1).type_as(sr).detach()
        
        km = torch.Tensor([[1, 1, 1], 
                           [1, 1, 1],
                           [1, 1, 1]]).view(1, 1, 3, 3).to(hr)
        # km = torch.Tensor([[1, 1, 1, 1, 1], 
        #                    [1, 1, 1, 1, 1],
        #                    [1, 1, 1, 1, 1],
        #                    [1, 1, 1, 1, 1],
        #                    [1, 1, 1, 1, 1]]).view(1, 1, 5, 5).to(hr)
        
        kx = torch.Tensor([[1, 0, -1], [2, 0, -2],
                           [1, 0, -1]]).view(1, 1, 3, 3).to(hr)
        ky = torch.Tensor([[1, 2, 1], [0, 0, 0],
                           [-1, -2, -1]]).view(1, 1, 3, 3).to(hr)

        erode = F.conv2d(mask1, km, padding=1)
        # erode = F.conv2d(mask1, km, padding=2)
        mask1_erode = (erode == 9).type_as(sr).detach()
        # mask1_erode = (erode == 25).type_as(sr).detach()
        pred_grad_x = F.conv2d(sr, kx, padding=1)
        pred_grad_y = F.conv2d(sr, ky, padding=1)
        target_grad_x = F.conv2d(hr, kx, padding=1)
        target_grad_y = F.conv2d(hr, ky, padding=1)
        
        d = torch.abs(pred_grad_x - target_grad_x) * mask1_erode 
        d = torch.sum(d, dim=[1, 2, 3])
        num_valid = torch.sum(mask1_erode, dim=[1, 2, 3])
        loss_x = d / (num_valid + 1e-8)

        d = torch.abs(pred_grad_y - target_grad_y) * mask1_erode 
        d = torch.sum(d, dim=[1, 2, 3])
        num_valid = torch.sum(mask1_erode, dim=[1, 2, 3])
        loss_y = d / (num_valid + 1e-8)
        
        loss = loss_x.sum() + loss_y.sum()
        return loss



class Sparse_Loss(nn.Module):
    def __init__(self):
        super(Sparse_Loss, self).__init__()

        # self.args = args
        self.depth_valid1 = 0.001
        # self.depth_valid2 = 20

    def forward(self, pred, depsp, gt):
        
        zeros = torch.zeros(depsp.size(), device=depsp.device)
        pred_ = torch.where(depsp > 0.001, pred, zeros)
        depsp_ = torch.where(depsp > 0.001, gt, zeros)
        error = torch.abs(pred_ - depsp_)
        loss = torch.mean(error)
        return loss


class L1Loss(nn.Module):
    def __init__(self):
        super(L1Loss, self).__init__()

        # self.args = args
        self.depth_valid1 = 0.001
        # self.depth_valid2 = 20


    def forward(self, pred, gt):

        mask1 = (gt > self.depth_valid1).type_as(pred).detach()
        # mask2 = (gt < self.depth_valid2).type_as(pred).detach()

        d = torch.abs(pred - gt) * mask1 # * mask2

        d = torch.sum(d, dim=[1, 2, 3])
        num_valid = torch.sum(mask1, dim=[1, 2, 3])

        loss = d / (num_valid + 1e-8)

        loss = loss.sum()

        return loss
    
class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-9

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss
    


from torch.distributions import MultivariateNormal as MVN
# import torch.nn.functional as F


class BMCLoss(nn.Module):
    def __init__(self, init_noise_sigma, device):
        super(BMCLoss, self).__init__()
        self.device = device
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma, device=device))

    def bmc_loss_md(self, pred, target, noise_var):
        """Compute the Multidimensional Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
        Args:
        pred: A float tensor of size [batch, d].
        target: A float tensor of size [batch, d].
        noise_var: A float number or tensor.
        Returns:
        loss: A float tensor. Balanced MSE Loss.
        """
        I = torch.eye(pred.shape[-1], device=self.device)
        logits = MVN(pred.unsqueeze(1), noise_var*I).log_prob(target.unsqueeze(0))  # logit size: [batch, batch]
        loss = F.cross_entropy(logits, torch.arange(pred.shape[0], device=self.device))     # contrastive-like loss
        loss = loss * (2 * noise_var).detach()  # optional: restore the loss scale, 'detach' when noise is learnable 
        
        return loss

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        return self.bmc_loss_md(pred, target, noise_var)


class BerhuLoss(nn.Module):
    def __init__(self, delta=0.05):
        super(BerhuLoss, self).__init__()
        self.delta = delta

    def forward(self, prediction, gt):
        
        err = torch.abs(prediction - gt)
        mask = (gt > 0.001).detach()
        err = torch.abs(err[mask])
        c = self.delta*err.max().item()
        squared_err = (err**2+c**2)/(2*c)
        linear_err = err
        return torch.mean(torch.where(err > c, squared_err, linear_err))

    

class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

        self.depth_valid1 = 0.001

    def forward(self, pred, gt):

        mask1 = (gt > self.depth_valid1).type_as(pred).detach()

        d = torch.pow(pred - gt, 2) * mask1 # * mask2

        d = torch.sum(d, dim=[1, 2, 3])
        num_valid = torch.sum(mask1, dim=[1, 2, 3])

        loss = d / (num_valid + 1e-8)

        loss = loss.sum()

        return loss
    

class L1_Charbonnier_loss(torch.nn.Module):
    """L1 Charbonnierloss."""""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-9

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt(diff * diff + self.eps)
        loss = torch.mean(error)
        return loss


class RMAEloss(nn.Module):
    def __init__(self, reduction='mean'):
        super(RMAEloss, self).__init__()
        self.reduction = reduction
        
        self.depth_valid1 = 0.15
        self.depth_valid2 = 20

    def forward(self, pred_dep, gt):
        
        ''' If forget to add parentheses to (gt+1e-6), it will cause nan loss
        mask1 = (gt > self.depth_valid1).type_as(pred_dep).detach()
        mask2 = (gt < self.depth_valid2).type_as(pred_dep).detach()      
        loss = torch.abs((pred_dep-gt)/(gt+1e-6)) * mask1 * mask2
        '''
        loss = torch.abs((pred_dep[gt>0.15]-gt[gt>0.15])/gt[gt>0.15])
        
        if self.reduction == 'mean':
            rmae = torch.mean(loss) 
        else:
            rmae = torch.sum(loss)
        return rmae

class Self_Consistency_loss(torch.nn.Module):
    """Self Consistency loss."""""
    def __init__(self, overlap_size, scale):
        super(Self_Consistency_loss, self).__init__()
        self.overlap_size = overlap_size
        self.scale = scale
    def forward(self, x, net):
        out1, out2 = get_overlap_region(x, net, overlap_size=self.overlap_size, scale=self.scale)
        loss = F.l1_loss(out1, out2)
        return loss
    
# https://nonint.com/2021/01/01/translational-regularization-for-image-super-resolution/
def get_overlap_region(img_in, net, overlap_size=24, scale=3):
    
    s = overlap_size // 2
    ih, iw = img_in.shape[-2:]
    
    t = random.randrange(3, 8) * 2
    l = random.randrange(3, 8) * 2
    b = random.randrange(3, 8) * 2
    r = random.randrange(3, 8) * 2
    
    subimg1 = img_in[:, :, ih // 2 - s - t :ih // 2 + s, iw // 2 - s - l:iw // 2 + s]
    subimg2 = img_in[:, :, ih // 2 - s :ih // 2 + s + b, iw // 2 - s:iw // 2 + s + r]

    outimg1 = net(subimg1)
    outimg2 = net(subimg2)
    
    outimg1_overlap = outimg1[:, :, t * scale :(t+ 2*s)*scale, l*scale:(l + 2*s)*scale]
    outimg2_overlap = outimg2[:, :, 0:2*s*scale, 0:2*s*scale]

    return outimg1_overlap, outimg2_overlap