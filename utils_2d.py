import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR, MultiStepLR

def make_affine2d(r, t, device):
    """Differentiable batch version, so will only optimize rotation and translation,
    If translation==None, treats it as fixed identity translation
    """
    n, _ = r.size()
    r = r.to(device)
    if t is not None: # ignore translation entirely, don't create params
        t = t.to(device)
        tx = t[:, 0].view(-1,1,1).repeat(1,2,3)*torch.tensor([[0,0,1],
                                                             [0,0,0]], dtype=torch.float32, device=device
                                                            ).view(1,2,3).repeat(n, 1, 1)
        ty = t[:, 1].view(-1,1,1).repeat(1,2,3)*torch.tensor([[0,0,0],
                                                             [0,0,1]], dtype=torch.float32, device=device
                                                              ).view(1,2,3).repeat(n, 1, 1)
    sin_mask = torch.tensor([
      [0,-1,0],
      [1,0,0],
      [0,0,0]], dtype=torch.float32, device = device).view(1,3,3).repeat(n, 1, 1)
    cos_mask = torch.tensor([
      [1,0,0],
      [0,1,0],
      [0,0,0]], dtype=torch.float32, device = device).view(1,3,3).repeat(n, 1, 1)

    affine_mat = (torch.cos(r.view(-1, 1, 1).repeat(1,3,3)) * cos_mask + 
              torch.sin(r.view(-1, 1, 1).repeat(1,3,3)) * sin_mask)[:, :2, :3]
    
    if t is not None: 
        affine_mat = affine_mat + tx + tyy
    return affine_mat


class TransformedMNIST(torchvision.datasets.MNIST):
    def __init__(self, root, transform_type, train):
        super(TransformedMNIST, self).__init__(root, train=train, download=True)
        self.transform_type = transform_type
        self.id_mat = torch.zeros(2, 3)
        self.id_mat[:, :2] = torch.FloatTensor([[1, 0], [0, 1]])
        assert transform_type in ['none', 'rand_90_rot', 'rand_rot', 'rand_rot_trans']
    
    def affine(self, x, affine_params, padding_mode='zeros'):
        grid = F.affine_grid(affine_params, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, padding_mode=padding_mode, align_corners=False)
        return x
    
    def random_affine(self, x, trans=False):
        r_init = torch.ones([1, 1], dtype=torch.float32, device=x.device).uniform_(0, 2*math.pi)
        if trans:
            t_init = torch.ones([1, 2], dtype=torch.float32, device=x.device).uniform_(-.2, .2)
        else:
            t_init = None
        affine_params = make_affine2d(r=r_init, t=t_init, device=x.device)
        x = self.affine(x, affine_params, padding_mode='zeros')
        return x, affine_params

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = {}
        attributes = {}
        x = self.data[index].float().unsqueeze(0)
        x = F.pad(x, (6, 6, 6, 6), 'constant', 0)
#         print(x.min().item(), x.max().item(), x.mean().item())
#         x = transforms.Normalize((0.1307,), (0.3081,))(x).unsqueeze(0)
        x = x/255.
#         print(x.min().item(), x.max().item(), x.mean().item())
        if self.transform_type == 'rand_rot':
            x, attributes['affine_params'] = self.random_affine(x, trans=False)
        if self.transform_type == 'rand_rot_trans':
            x, attributes['affine_params'] = self.random_affine(x,trans=True)
        elif self.transform_type == 'none':
            attributes['affine_params'] = self.id_mat.clone()
        return {'data': x.squeeze(), 'attributes': attributes}