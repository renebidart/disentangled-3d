import math
import pickle
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, random_split, DataLoader
from torch.nn import functional as F

def make_affine2d(r, t, device):
    """Differentiable batch version, so will only optimize rotation and translation,
    If translation==None, treats it as fixed identity translation
    """
    n, _ = r.size()
    r = r.to(device)
    if t is None: # ignore translation entirely, don't create params
        pass
    else:
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

    rot_mat = (torch.cos(r.view(-1, 1, 1).repeat(1,3,3)) * cos_mask + 
              torch.sin(r.view(-1, 1, 1).repeat(1,3,3)) * sin_mask)[:, :2, :3]
    if t is None: 
        affine_mat = rot_mat
    else:  
        affine_mat = rot_mat + tx + ty
    return affine_mat


class AVAE2d(nn.Module):
    def __init__(self, VAE, opt_method='rand_sgd_rot'):
        super(AVAE2d, self).__init__()
        assert opt_method in ['none', 'rand_sgd_rot', 'rand_sgd_rot_trans']
        self.VAE = VAE
        self.opt_method = opt_method
        
    def affine(self, x, affine_params, padding_mode='zeros'):
        if len(affine_params.size()) == 2:
            affine_params = affine_params.unsqueeze(0)
        grid = F.affine_grid(affine_params, x.size(), align_corners=False).to(x.device)
        x = F.grid_sample(x, grid, padding_mode=padding_mode, align_corners=False)
        return x
    
    def affine_inv(self, x, affine_params, padding_mode='zeros'):
        if len(affine_params.size()) == 2:
            affine_params = affine_params.unsqueeze(0)
        if self.opt_method in ['none', 'rand_sgd_rot']: # inverse is transpose if rot
            zeros = torch.FloatTensor(affine_params.size()).fill_(0)[:, :, 0].unsqueeze(-1).to(x.device)
            inv_affine_params = torch.cat([torch.transpose(affine_params[:, :, :2], 1, 2), zeros], dim=-1)
        else:
            M_inv = torch.inverse(affine_params[:, :, :2])
            t_inv = torch.bmm(-M_inv, affine_params[:, :, 2].unsqueeze(-1))
            inv_affine_params = torch.cat([M_inv, t_inv], dim=-1)
        grid = F.affine_grid(inv_affine_params, x.size(), align_corners=False).to(x.device)
        x = F.grid_sample(x, grid, padding_mode=padding_mode, align_corners=False)
        return x

    def forward(self, x, deterministic=False):
        if self.opt_method=='none': # affine is identity
            affine_params = torch.zeros(2, 3).to(x.device)
            affine_params[:, :2] = torch.FloatTensor([[1, 0], [0, 1]]).to(x.device)
            affine_params = affine_params.unsqueeze(0).repeat(x.size(0), 1, 1)
            mu = self.VAE.encode(x)
            z = self.VAE.reparameterize(mu, deterministic)
            recon_x = self.VAE.decode(z)
        elif 'rand_sgd' in self.opt_method:
            recon_x, mu, loss, affine_params = self.forward_opt(x, n_sgd=8, n_total_affine=32, deterministic=True)        
            recon_x, mu = self.affine_forward(x, affine_params=affine_params, deterministic=deterministic)
        return {'recon_x':recon_x, 
                'mu': mu,
                'affine_params': affine_params}
        
    def affine_forward(self, x, affine_params=None, deterministic=False):
        x_affine = self.affine(x, affine_params)
        mu = self.VAE.encode(x_affine)
        z = self.VAE.reparameterize(mu, deterministic)
        recon_x = self.VAE.decode(z)
        recon_x = self.affine_inv(recon_x, affine_params)
        return recon_x, mu

    
    def forward_opt(self, x, n_sgd, n_total_affine, deterministic=True):
        bs, ch, _, _, _ = x.size()
        r_init = torch.ones([n_total_affine*bs, 1], dtype=torch.float32, device=x.device).uniform_(0, 2*math.pi)
        if 'trans' in self.opt_method:
            t_init = torch.ones([n_total_affine*bs, 2], dtype=torch.float32, device=x.device).uniform_(-.2, .2)
        else:
            t_init = None
        with torch.no_grad():
            affine_params = make_affine(r=r_init, t=t_init, device=x.device)
            x_rep = x.repeat(n_total_affine, 1, 1, 1, 1)
            recon_x, mu = self.affine_forward(x_rep, affine_params=affine_params, deterministic=deterministic)
            loss = self.vae_loss_unreduced((recon_x, mu), x_rep)
            loss = loss.view(n_total_affine, bs, 1)
            best_loss, best_param_idx = torch.topk(loss, k=n_sgd, dim=0, largest=False)
            
        # SGD TIME - Select best params and get rid of old params so no grad or memory issues
        del recon_x, mu, x_rep
        r_init = r_init[best_param_idx, :].view(n_sgd*bs, 1).clone().detach().requires_grad_(True)
        if 'trans' in self.opt_method:
            t_init = t_init[best_param_idx, :].view(n_sgd*bs, 2).clone().detach().requires_grad_(True)
            optimizer = optim.Adam([r_init, t_init], lr=.03)
        else:
            t_init = None
            optimizer = optim.Adam([r_init], lr=.03)

        x_rep = x.repeat(n_sgd, 1, 1, 1, 1)

        for i in range(15):
            affine_params = make_affine(r=r_init, t=t_init, device=x.device)
            recon_x, mu = self.affine_forward(x_rep, affine_params=affine_params, deterministic=deterministic)
            loss = self.vae_loss_unreduced((recon_x, mu), x_rep).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        affine_params = make_affine(r=r_init, t=t_init, device=x.device)
        recon_x, mu = self.affine_forward(x_rep, affine_params=affine_params, deterministic=deterministic)
        loss = self.vae_loss_unreduced((recon_x, mu), x_rep)
        best_loss, best_idx = torch.min(loss.view(n_sgd, bs), dim=0)
        
        recon_x = recon_x.view(n_sgd, bs, 42, 42, 42)[best_idx, :, :, :]
        mu = mu.view(n_sgd, bs, -1)[best_idx, :]
        affine_params = affine_params.view(n_sgd, bs, 2, 3)[best_idx, torch.arange(bs), :, :].squeeze()
        return recon_x, mu, best_loss, affine_params    

    def vae_loss_unreduced(self, output, x, KLD_weight=1):
        recon_x, mu  = output
        BCE = F.binary_cross_entropy(recon_x.squeeze(), x.squeeze(), reduction='none')
        BCE = torch.sum(BCE, dim=(1, 2, 3))
        KLD = -0.5 * torch.sum(1 + 0 - mu.pow(2) - 1)
        return BCE + KLD_weight*KLD


class VAE2d(nn.Module):
    """VAE based off https://arxiv.org/pdf/1805.09190v3.pdf"""
    def __init__(self, input_dim=1, output_dim=1, latent_size=8, img_size=40):
        super(VAE2d, self).__init__()
        self.latent_size = latent_size
        self.img_size = img_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear_size = int(16*(img_size/4)**2)

        self.elu = nn.ELU()
        self.enc_conv1 = nn.Conv2d(self.input_dim, 32, kernel_size=5, stride=1, padding=2, bias=False)
        self.enc_conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.enc_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.enc_conv4 = nn.Conv2d(64, 16, kernel_size=5, stride=1, padding=2, bias=True)
        self.enc_bn1 = nn.BatchNorm2d(32)
        self.enc_bn2 = nn.BatchNorm2d(32)
        self.enc_bn3 = nn.BatchNorm2d(64)

        self.dec_conv1 = nn.ConvTranspose2d(16, 32, kernel_size=4, stride=1, padding=2,  output_padding=0, bias=False)
        self.dec_conv2 = nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=1,  output_padding=1, bias=False)
        self.dec_conv3 = nn.ConvTranspose2d(16, 16, kernel_size=5, stride=2, padding=2,  output_padding=1, bias=False)
        self.dec_conv4 = nn.ConvTranspose2d(16, self.output_dim, kernel_size=3, stride=1, padding=1,  output_padding=0, bias=True)


        self.dec_bn1 = nn.BatchNorm2d(32)
        self.dec_bn2 = nn.BatchNorm2d(16)
        self.dec_bn3 = nn.BatchNorm2d(16)
        
        self.fc_mu = nn.Linear(self.linear_size, self.latent_size)
        self.fc_logvar= nn.Linear(self.linear_size, self.latent_size)
        self.fc_dec = nn.Linear(self.latent_size, self.linear_size)

        
    def forward(self, x, deterministic=False):
        mu_logvar = self.encode(x)
        z = self.reparameterize(mu_logvar, deterministic)
        recon_x = self.decode(z)
        return recon_x, mu_logvar

    def encode(self, x):
        x = self.elu(self.enc_bn1(self.enc_conv1(x)))
        x = self.elu(self.enc_bn2(self.enc_conv2(x)))
        x = self.elu(self.enc_bn3(self.enc_conv3(x)))
        x = self.enc_conv4(x)
        x = x.view(x.size(0), -1)
        mu_logvar = torch.cat((self.fc_mu(x), self.fc_logvar(x)), dim=1)
        return mu_logvar

    def decode(self, x):
        x = self.fc_dec(x)
        x = x.view((-1, 16, int(self.img_size/4), int(self.img_size/4)))
        x = self.elu(self.dec_bn1(self.dec_conv1(x)))
        x = self.elu(self.dec_bn2(self.dec_conv2(x)))
        x = self.elu(self.dec_bn3(self.dec_conv3(x)))
        x = self.dec_conv4(x)
        if self.input_dim==1: return torch.sigmoid(x)
        else: return x

    def reparameterize(self, mu_logvar, deterministic=False):
        mu = mu_logvar[:, 0:int(mu_logvar.size()[1]/2)]
        if deterministic: # return mu 
            return mu
        else: # return mu + random
            logvar = mu_logvar[:, int(mu_logvar.size()[1]/2):]
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        
#         self.fc_mu = nn.Linear(self.linear_size, self.latent_size)
#         self.fc_dec = nn.Linear(self.latent_size, self.linear_size)

#     def forward(self, x, deterministic=False):
#         mu = self.encode(x)
#         z = self.reparameterize(mu, deterministic)
#         recon_x = self.decode(z)
#         return recon_x, mu

#     def encode(self, x):
#         x = self.elu(self.enc_bn1(self.enc_conv1(x)))
#         x = self.elu(self.enc_bn2(self.enc_conv2(x)))
#         x = self.elu(self.enc_bn3(self.enc_conv3(x)))
#         x = self.enc_conv4(x)
#         x = x.view(x.size(0), -1)
#         return self.fc_mu(x)

#     def decode(self, x):
#         x = self.fc_dec(x)
#         x = x.view((-1, 16, int(self.img_size/4), int(self.img_size/4)))
#         x = self.elu(self.dec_bn1(self.dec_conv1(x)))
#         x = self.elu(self.dec_bn2(self.dec_conv2(x)))
#         x = self.elu(self.dec_bn3(self.dec_conv3(x)))
#         x = self.dec_conv4(x)
#         return torch.sigmoid(x)

#     def reparameterize(self, mu, deterministic=False):
#         if deterministic: 
#             return mu
#         else:
#             return mu.add_(torch.randn_like(mu))