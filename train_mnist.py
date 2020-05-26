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

from models.avae2d import AVAE2d, VAE2d
from utils_2d import make_affine2d, TransformedMNIST
from utils import progress_bar

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--bs', type=int, default=128)
parser.add_argument('--epochs', default=100, type=int)
parser.add_argument('--data_path', type=str, default='./data/')
parser.add_argument('--save_path', default="./results", type=str)

parser.add_argument('--data_augmentation', type=str, default='none')
parser.add_argument('--val_augmentation', type=str, default='none')
parser.add_argument('--opt_method', type=str, default='none')
parser.add_argument('--latent_size', type=int, default=8)

args = parser.parse_args()
print(args)

# device = 'cpu'
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
assert args.data_augmentation in ['none', 'rand_90_rot', 'rand_rot', 'rand_rot_trans']
assert args.opt_method in ['none', 'all_90_rot',  'rand_sgd_rot', 'rand_sgd_rot_trans', 'opt1']


trainset = TransformedMNIST(args.data_path, transform_type=args.data_augmentation, train=True)
valset = TransformedMNIST(args.data_path, transform_type=args.val_augmentation, train=False)
train_loader = DataLoader(trainset, batch_size=args.bs, num_workers=4, pin_memory=True, shuffle=True)
val_loader = DataLoader(valset, batch_size=args.bs, num_workers=4, pin_memory=True, shuffle=False)

img_size = 40 # consistency between tests is important
model = AVAE2d(VAE=VAE2d(latent_size=args.latent_size, img_size=img_size), 
                opt_method=args.opt_method).to(device)
    
model.to(device)
model.train()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
scheduler = StepLR(optimizer, step_size=int(args.epochs/3), gamma=0.2)
    
    
# def vae_loss(recon_x, x, mu):
#     KLD = -0.5 * torch.sum(-mu.pow(2))
#     recon = 0.5 * F.mse_loss(recon_x.squeeze(), x.squeeze(), reduction='sum')
#     MSE = F.mse_loss(recon_x.squeeze(), x.squeeze(), reduction='sum')
#     return {'loss': KLD + recon,
#             'mse': MSE}


def vae_loss(recon_x, x, mu_logvar):
    """loss is BCE + KLD. target is original x"""
    mu = mu_logvar[:, 0:int(mu_logvar.size()[1]/2)]
    logvar = mu_logvar[:, int(mu_logvar.size()[1]/2):]
#     KLD = -0.5 * torch.sum(1 + 2 * logvar - mu.pow(2) - (2 * logvar).exp())
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
#     BCE = F.binary_cross_entropy(recon_x.squeeze(), x.squeeze(), reduction='sum')
    MSE = F.mse_loss(recon_x.squeeze(), x.squeeze(), reduction='sum')
    return {'loss': KLD + MSE,
            'mse': MSE}

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        x = batch['data'].to(device).unsqueeze(1)
        optimizer.zero_grad()
        output = model(x, deterministic=True)
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
    y_list = []
    for batch_idx, batch in enumerate(val_loader):
        x = batch['data'].to(device).unsqueeze(1)
        output = model(x, deterministic=True)
        losses = vae_loss(output['recon_x'], x, output['mu'])
        val_loss += losses['loss'].item()
        val_mse += losses['mse'].item()
        data_ap_list.append(batch['attributes']['affine_params'])
        model_ap_list.append(output['affine_params'])
        y_list.append(batch['y'])

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
            'label_list': torch.cat(y_list, dim=0).detach().cpu().numpy(),
            'data_ap_list': torch.cat(data_ap_list, dim=0).detach().cpu().numpy() ,
            'model_ap_list': torch.cat(model_ap_list, dim=0).detach().cpu().numpy(),
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
    results = val(epoch, train_loss, results, save_freq=1)
    scheduler.step()
