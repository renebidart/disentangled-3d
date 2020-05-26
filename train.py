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
from utils_3d import pad_3d, make_affine
from utils import progress_bar
from dataset.trans_model_net_voxels import TransformedModelNetVoxels

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

args = parser.parse_args()
print(args)

# device = 'cpu'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
assert args.data_augmentation in ['none', 'rand_90_rot', 'rand_rot', 'rand_rot_trans', 'rand_rot_1d']
assert args.opt_method in ['none', 'all_90_rot',  'rand_sgd_rot', 'rand_sgd_rot_trans', 'opt1', 'rand_sgd_1d']


    
if args.dataset == 'modelnet10':
    categories = ['dresser', 'desk', 'night_stand',  'bathtub', 'chair', 
                   'sofa', 'monitor', 'table', 'toilet', 'bed']
    if args.n_classes > 1:
        categories = categories[:args.n_classes]
    else:
        categories = [args.category]
    print(f'Using categories: {categories}')
    trainset = TransformedModelNetVoxels(args.data_path, transform_type=args.data_augmentation, categories=categories, 
                                         resolutions=[32], split='train', device='cpu')
    valset = TransformedModelNetVoxels(args.data_path, transform_type=args.val_augmentation, categories=categories, 
                                       resolutions=[32], split='test', device='cpu')
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, num_workers=4, pin_memory=True, shuffle=True)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=args.bs, num_workers=4, pin_memory=True, shuffle=False)
    # Do all data preprocessing is done on cpu (i dont understand on gpu), with mutiple workers
        

img_size = 48 # consistency between tests is most important
    
model = AVAE3d(VAE=VAE(latent_size=args.latent_size, img_size=img_size), 
                opt_method=args.opt_method).to(device)
    
model.to(device)
model.train()
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)


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
    return train_loss
        

def val(epoch, train_loss, results, save_freq):
    model.eval()
    val_loss = 0
    val_mse = 0
    model_ap_list = []
    data_ap_list = []
#     with torch.no_grad(): Normally use no_grad for eval, but here we need gradients
    for batch_idx, batch in enumerate(val_loader):
            x = batch['data']['32'].to(device).unsqueeze(1)
            output = model(x)
            losses = vae_loss(output['recon_x'], x, output['mu'])
            val_loss += losses['loss'].item()
            val_mse += losses['mse'].item()
            data_ap_list.append(batch['attributes']['affine_params'])
            model_ap_list.append(output['affine_params'])

    val_loss /= len(val_loader.dataset)
    results['val_loss'].append(val_loss)
    results['val_mse'].append(val_mse)
    results['train_loss'].append(train_loss)
    progress_bar(batch_idx, len(val_loader), 'Loss: %.3f' % (val_loss/(batch_idx+1)))

    # Save checkpoint.
    print(f'Epoch: {epoch}')
    if (val_loss < results['best_loss']) or (epoch % save_freq == 0):
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'val_loss': val_loss,
            'val_mse': val_mse,
            'epoch': epoch,
            'data_ap_list': torch.cat(data_ap_list, dim=0).detach().cpu().numpy() ,
            'model_ap_list': torch.cat(model_ap_list, dim=0).detach().cpu().numpy(),
            'num_params': num_params,
            'results': results
        }
        if not os.path.isdir(args.save_path):
            os.makedirs(args.save_path)
        if (val_loss < results['best_loss']):
            name = '/best_ckpt.pth'
            torch.save(state, args.save_path + name)
        if (epoch % save_freq == 0) :
            name = '/' + str(int(epoch/save_freq)) +'_ckpt.pth'
            torch.save(state, args.save_path + name)
        results['best_loss'] = val_loss
    return results
      
results = {}
results['val_mse'] = []
results['val_loss'] = []
results['train_loss'] = []
results['best_loss'] = 1000000000  # best val loss
for epoch in range(args.epochs):
    train_loss = train(epoch)
    results = val(epoch, train_loss, results, save_freq=int(args.epochs/20))
    scheduler.step()
