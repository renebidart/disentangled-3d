import os
import math
import random
import pickle
import argparse
import numpy as np
from pathlib import Path

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import StepLR, MultiStepLR

import kaolin as kal
from kaolin.datasets import ModelNet, ModelNetVoxels

from models.avae import AVAE3d, VAE
from utils import progress_bar


parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=3e-3, type=float)
parser.add_argument("--bs", type=int, default=32)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument("--dataset", type=str, default='modelnet10')
parser.add_argument('--data_path', type=str, default='./data/ModelNet10')
parser.add_argument('--save_path', default="./results", type=str)

parser.add_argument('--model_type', type=str, default='avae3d')
parser.add_argument("--n_classes", type=int, default=1)
parser.add_argument("--category", type=str, default='dresser')


parser.add_argument('--data_augmentation', type=str, default='none')
parser.add_argument('--val_augmentation', type=str, default='none')
parser.add_argument('--opt_method', type=str, default='none')
parser.add_argument('--latent_size', type=int, default=8)

# parser.add_argument("--channel_factor", type=int)
# parser.add_argument("--ws_factor", type=int, default=1)
# parser.add_argument("--ws_sp_factor", type=int, default=1)
# parser.add_argument("--ws_ch_factor", type=int, default=1)
# parser.add_argument("--downsample_repeat", action='store_true')
args = parser.parse_args()
print(args)


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
assert args.data_augmentation in ['none', 'random_90_rot']
assert args.opt_method in ['none', 'all_90_rot']


class TransformedModelNetVoxels(ModelNetVoxels):
    def __init__(self, modelnet_path, transform_type=None, categories=['chair'], 
                 device='cuda', split='train', resolutions=[32]):
        super(TransformedModelNetVoxels, self).__init__(modelnet_path, categories=categories, 
                                                        resolutions=resolutions, split=split, device=device)
        self.transform_type = transform_type
        self.id_mat = torch.zeros(3, 4)
        self.id_mat[:, :3] = torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        
        def get_all_rot_mat():
            def rot90_x(mat): return torch.mm(torch.FloatTensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]]), mat)
            def rot90_y(mat): return torch.mm(torch.FloatTensor([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]), mat)
            mat = torch.zeros(3, 4)
            mat[:, :3] = torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            all_rotations = []
            for cycle in range(2):
                for step in range(3):
                    mat[:, :3] = rot90_x(mat[:, :3])
                    all_rotations.append(mat.clone())
                    for i in range(3):
                        mat[:, :3] = rot90_y(mat[:, :3])
                        all_rotations.append(mat.clone())
                mat[:, :3] = rot90_x(rot90_y(rot90_x(mat[:, :3])))
            return torch.stack(all_rotations)
        self.all_90_rot = get_all_rot_mat()
    
    def affine(self, x, affine_params, padding_mode='zeros'):
        grid = F.affine_grid(affine_params, x.size(), align_corners=False).to(x.device)
        x = F.grid_sample(x, grid, padding_mode=padding_mode, align_corners=False)
        return x
    
    def random_90(self, x):
        affine_params = self.all_90_rot[random.randint(0, 23), :, :]
        x = self.affine(x, affine_params.unsqueeze(0), padding_mode='zeros')
        return x, affine_params

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        """Returns the item at index idx. """
        data = dict()
        attributes = dict()
        name = self.names[index]

        for res in self.params['resolutions']:
            data[str(res)] = self.cache_transforms[res](name)
        attributes['name'] = name
        attributes['category'] = self.categories[self.cat_idxs[index]]
        
        if self.transform_type == 'random_90_rot':
            data[str(res)], attributes['affine_params'] = self.random_90(data[str(res)].unsqueeze(0).unsqueeze(0))
            data[str(res)] = data[str(res)].squeeze()
        elif self.transform_type == 'none':
            attributes['affine_params'] = self.id_mat.clone()
        return {'data': data, 'attributes': attributes}        
    
    
if args.dataset == 'modelnet10':
    categories = ['dresser', 'desk', 'night_stand',  'bathtub', 'chair', 
                   'sofa', 'monitor', 'table', 'toilet', 'bed']
    if args.n_classes > 1:
        categories = categories[:args.n_classes]
    else:
        categories = [args.category]
    print(f'Using categories: {categories}')
    trainset = TransformedModelNetVoxels(args.data_path, transform_type=args.data_augmentation, categories=categories, 
                                         resolutions=[32], split='train', device=device)
    valset = TransformedModelNetVoxels(args.data_path, transform_type=args.val_augmentation, categories=categories, 
                                       resolutions=[32], split='test', device=device)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, num_workers=2, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=2*args.bs, num_workers=2, shuffle=False)
        

if args.model_type=='avae3d':
    model = AVAE3d(VAE=VAE(latent_size=args.latent_size, img_size=32), 
                   opt_method=args.opt_method).to(device)
    
model.to(device)
model.train()
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# if args.resume:
#     print('==> Resuming from checkpoint..')
#     assert os.path.isdir(args.save_path), 'Error: no checkpoint directory found!'
#     checkpoint = torch.load(args.save_path + '/ckpt.pth')
#     net.load_state_dict(checkpoint['net'])
#     best_acc = checkpoint['acc']
#     start_epoch = checkpoint['epoch']


optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = StepLR(optimizer, step_size=int(args.epochs/3), gamma=0.2)
    
    
def vae_loss(recon_x, x, mu):
    BCE = F.binary_cross_entropy(recon_x.squeeze(), x.squeeze(), reduction='sum')
    KLD = -0.5 * torch.sum(1 + 0 - mu.pow(2) - 1)
    MSE = F.mse_loss(recon_x.squeeze(), x.squeeze(), reduction='sum')
    return {'loss': BCE + KLD,
            'mse': MSE}

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        x = batch['data']['32'].to(device).unsqueeze(1)
        optimizer.zero_grad()
        output = model(x)
        losses = vae_loss(output['recon_x'], x, output['mu'])
        losses['loss'].backward()
        train_loss += losses['loss'].item()
        optimizer.step()
        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f' % (train_loss/(batch_idx+1)))
        

def val(epoch, best_loss, save_freq):
    model.eval()
    val_loss = 0
    val_mse = 0
    model_ap_list = []
    data_ap_list = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
                x = batch['data']['32'].to(device).unsqueeze(1)
                output = model(x)
                losses = vae_loss(output['recon_x'], x, output['mu'])
                val_loss += losses['loss'].item()
                val_mse += losses['mse'].item()
                data_ap_list.append(batch['attributes']['affine_params'])
                model_ap_list.append(output['affine_params'])
                
        val_loss /= len(val_loader.dataset)
        progress_bar(batch_idx, len(val_loader), 'Loss: %.3f' % (val_loss/(batch_idx+1)))
        
    # Save checkpoint.
    print(f'Epoch: {epoch}')
    if (val_loss < best_loss) or (epoch % save_freq == 0):
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'val_loss': val_loss,
            'val_mse': val_mse,
            'epoch': epoch,
            'data_ap_list': torch.cat(data_ap_list, dim=0).detach().cpu().numpy() ,
            'model_ap_list': torch.cat(model_ap_list, dim=0).detach().cpu().numpy(),
            'num_params': num_params
        }
        if not os.path.isdir(args.save_path):
            os.makedirs(args.save_path)
        if (val_loss < best_loss):
            name = '/best_ckpt.pth'
            torch.save(state, args.save_path + name)
        if (epoch % save_freq == 0) :
            name = '/' + str(int(epoch/save_freq)) +'_ckpt.pth'
            torch.save(state, args.save_path + name)
        best_loss = val_loss
    return best_loss
      
best_loss = 1000000000  # best val accuracy

for epoch in range(args.epochs):
    train(epoch)
    best_loss = val(epoch, best_loss, save_freq=int(args.epochs/10))
    scheduler.step()
