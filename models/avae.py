import math
import pickle
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, random_split, DataLoader
from torch.nn import functional as F

from utils_3d import make_affine


class AVAE3d(nn.Module):
    def __init__(self, VAE, opt_method='all_90_rot'):
        super(AVAE3d, self).__init__()
        assert opt_method in ['none', 'all_90_rot', 'rand_sgd_rot', 'rand_sgd_rot_trans', 'opt1']
        self.VAE = VAE
        self.opt_method = opt_method
        if self.opt_method == 'all_90_rot':
            self.all_90_rot = self.get_all_rot_mat()
            
    def get_all_rot_mat(self):
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
            
        
    def affine(self, x, affine_params, padding_mode='zeros'):
        if len(affine_params.size()) == 2:
            affine_params = affine_params.unsqueeze(0)
        grid = F.affine_grid(affine_params, x.size(), align_corners=False).to(x.device)
        x = F.grid_sample(x, grid, padding_mode=padding_mode, align_corners=False)
        return x
    
    def affine_inv(self, x, affine_params, padding_mode='zeros'):
        if len(affine_params.size()) == 2:
            affine_params = affine_params.unsqueeze(0)
        if self.opt_method in ['none', 'all_90_rot', 'rand_sgd_rot']: # inverse is transpose if rot
            zeros = torch.FloatTensor(affine_params.size()).fill_(0)[:, :, 0].unsqueeze(-1).to(x.device)
            inv_affine_params = torch.cat([torch.transpose(affine_params[:, :, :3], 1, 2), zeros], dim=-1)
        else:
            M_inv = torch.inverse(affine_params[:, :, :3])  
            t_inv = torch.bmm(-M_inv, affine_params[:, :, 3].unsqueeze(-1))
            inv_affine_params = torch.cat([M_inv, t_inv], dim=-1)
        grid = F.affine_grid(inv_affine_params, x.size(), align_corners=False).to(x.device)
        x = F.grid_sample(x, grid, padding_mode=padding_mode, align_corners=False)
        return x

    
    def forward(self, x, deterministic=True):
        if self.opt_method=='none': # affine is identity
            affine_params = torch.zeros(3, 4).to(x.device)
            affine_params[:, :3] = torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).to(x.device)
            affine_params = affine_params.unsqueeze(0).repeat(x.size(0), 1, 1)
        elif self.opt_method=='all_90_rot':
            recon_x, mu, loss, affine_params = self.forward_best_90_rot(x, deterministic=True)
        elif 'rand_sgd' in self.opt_method:
            recon_x, mu, loss, affine_params = self.forward_opt(x, n_sgd=8, n_total_affine=32, deterministic=True)
        elif 'opt1' in self.opt_method:
            recon_x, mu, loss, affine_params = self.forward_opt1(x, n_sgd=8, n_total_affine=32, deterministic=True)            
        # Shouldn't need this:
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

    
    def forward_best_90_rot(self, x, deterministic=True):
        """ Try all 24 rotations.
        
        Loss for all afffine params and imgs in parallel. 
        Duplicate imgs & params since can only apply one affine per img.
        returns: loss, affine_params(3x4)
        """                                
        with torch.no_grad():
            affine_params = self.all_90_rot.clone().detach().to(x.device) # be safe
            bs, ch, _, _, _ = x.size()
            n_affine, _, _ = affine_params.size()
            x_repeated = x.repeat(n_affine, 1, 1, 1, 1)
            affine_params_repeat = affine_params.repeat(bs, 1, 1).view(bs*n_affine, 3, 4)

            recon_x, mu = self.affine_forward(x_repeated, affine_params=affine_params_repeat, deterministic=deterministic)
            loss = self.vae_loss_unreduced((recon_x, mu), x_repeated)
            loss = loss.view(n_affine, bs, 1)
            best_loss, best_param_idx = torch.min(loss, dim=0)

            # select the params, recon_x, mu corresponding to lowest loss
            mu = mu.view(n_affine, bs, -1)
            affine_params_repeat = affine_params_repeat.view(n_affine, bs, 3, 4)
            recon_x = recon_x.view(n_affine, bs, 1, recon_x.size(-3), recon_x.size(-2), recon_x.size(-1))
            recon_x_best = recon_x[best_param_idx.squeeze(), torch.arange(bs), :, :, :, :]
            mu_best = mu[best_param_idx.squeeze(), torch.arange(bs), :]

            best_affine_params = affine_params_repeat[best_param_idx.squeeze(), torch.arange(bs), :, :]
        return recon_x_best, mu_best, best_loss, best_affine_params

    
    def forward_opt(self, x, n_sgd, n_total_affine, deterministic=True):
        bs, ch, _, _, _ = x.size()
        r_init = torch.ones([n_total_affine*bs, 3], dtype=torch.float32, device=x.device).uniform_(0, 2*math.pi)
        if 'trans' in self.opt_method:
            t_init = torch.ones([n_total_affine*bs, 3], dtype=torch.float32, device=x.device).uniform_(-.2, .2)
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
        r_init = r_init[best_param_idx, :].view(n_sgd*bs, 3).clone().detach().requires_grad_(True)
        if 'trans' in self.opt_method:
            t_init = t_init[best_param_idx, :].view(n_sgd*bs, 3).clone().detach().requires_grad_(True)
            optimizer = optim.Adam([r_init, t_init], lr=.03)
        else:
            t_init = None
            optimizer = optim.Adam([r_init], lr=.03)

        x_rep = x.repeat(n_sgd, 1, 1, 1, 1)

        for i in range(20):
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
        
        recon_x = recon_x.view(n_sgd, bs, 48, 48, 48)[best_idx, :, :, :]
        mu = mu.view(n_sgd, bs, -1)[best_idx, :]
        affine_params = affine_params.view(n_sgd, bs, 3, 4)[best_idx, torch.arange(bs), :, :].squeeze()
        return recon_x, mu, best_loss, affine_params    

    
#     def forward_opt(self, x, n_sgd, n_total_affine, deterministic=True):
#         bs, ch, _, _, _ = x.size()
#         r_init = torch.ones([n_total_affine, 3], dtype=torch.float32, device=x.device).uniform_(0, 2*math.pi)
#         if 'trans' in self.opt_method:
#             t_init = torch.ones([n_total_affine, 3], dtype=torch.float32, device=x.device).uniform_(-.2, .2)
#         else:
#             t_init = None
#         with torch.no_grad():
#             affine_params = make_affine(r=r_init, t=t_init, device=x.device).repeat(bs, 1, 1)
#             x_rep = x.repeat(n_total_affine, 1, 1, 1, 1)
#             recon_x, mu = self.affine_forward(x_rep, affine_params=affine_params, deterministic=deterministic)
#             loss = self.vae_loss_unreduced((recon_x, mu), x_rep)
#             loss = loss.view(n_total_affine, bs, 1)            
#             best_loss, best_param_idx = torch.topk(loss, k=n_sgd, dim=0, largest=False)
            
#         # SGD TIME - Select best params and get rid of old params so no grad or memory issues
#         del recon_x, mu, x_rep
#         r_init = r_init[best_param_idx, :].view(n_sgd*bs, 3).clone().detach().requires_grad_(True)
#         if 'trans' in self.opt_method:
#             t_init = t_init[best_param_idx, :].view(n_sgd*bs, 3).clone().detach().requires_grad_(True)
#             optimizer = optim.Adam([r_init, t_init], lr=.03)
#         else:
#             t_init = None
#             optimizer = optim.Adam([r_init], lr=.03)

#         x_rep = x.repeat(n_sgd, 1, 1, 1, 1)

#         for i in range(15):
#             affine_params = make_affine(r=r_init, t=t_init, device=x.device)
#             recon_x, mu = self.affine_forward(x_rep, affine_params=affine_params, deterministic=deterministic)
#             loss = self.vae_loss_unreduced((recon_x, mu), x_rep).sum()
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
#         affine_params = make_affine(r=r_init, t=t_init, device=x.device)
#         recon_x, mu = self.affine_forward(x_rep, affine_params=affine_params, deterministic=deterministic)
#         loss = self.vae_loss_unreduced((recon_x, mu), x_rep)
#         best_loss, best_idx = torch.min(loss.view(n_sgd, bs), dim=0)
        
# #         print(recon_x.size(), mu.size(), affine_params.size())
#         recon_x = recon_x.view(n_sgd, bs, 48, 48, 48)[best_idx, :, :, :]
#         mu = mu.view(n_sgd, bs, -1)[best_idx, :]
#         affine_params = affine_params.view(n_sgd, bs, 3, 4)[best_idx, torch.arange(bs), :, :].squeeze()
#         return recon_x, mu, best_loss, affine_params
    

    def forward_opt1(self, x, n_sgd, n_total_affine, deterministic=True):
        """ BS=1 !
        n_affine is number of random restarts
        """
        bs, ch, _, _, _ = x.size()
        assert bs==1
        r_init = torch.ones([n_total_affine, 3], dtype=torch.float32, device=x.device).uniform_(0, 2*math.pi)
        if 'trans' in self.opt_method:
            t_init = torch.ones([n_total_affine, 3], dtype=torch.float32, device=x.device).uniform_(-.2, .2)
        else:
            t_init = None

        with torch.no_grad():
            affine_params = make_affine(r=r_init, t=t_init, device=x.device)
            x_rep = x.repeat(n_total_affine, 1, 1, 1, 1)
            recon_x, mu = self.affine_forward(x_rep, affine_params=affine_params, deterministic=deterministic)
            loss = self.vae_loss_unreduced((recon_x, mu), x_rep)
            best_loss, best_param_idx = torch.topk(loss, k=n_sgd, largest=False)
            
        # SGD TIME - Select best params and get rid of old params so no grad or memory issues
        del recon_x, mu, x_rep
        r_init = r_init[best_param_idx, :].clone().detach().requires_grad_(True)

        if 'trans' in self.opt_method:
            t_init = t_init[best_param_idx, :].clone().detach().requires_grad_(True)
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
        best_loss, best_idx = torch.min(loss, dim=0)
        return recon_x[best_idx, :, :, :], mu[best_idx, :], best_loss, affine_params[best_idx, :, :]

    
    def vae_loss_unreduced(self, output, x, KLD_weight=1):
        recon_x, mu  = output
        BCE = F.binary_cross_entropy(recon_x.squeeze(), x.squeeze(), reduction='none')
        BCE = torch.sum(BCE, dim=(1, 2, 3))
        KLD = -0.5 * torch.sum(1 + 0 - mu.pow(2) - 1)
        return BCE + KLD_weight*KLD



# import math
# import pickle
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import TensorDataset, random_split, DataLoader
# from torch.nn import functional as F

# # Use only 90 as a flag when creating the model, or only when optimizing?
# # where to add the flexibility?

# class AVAE3d(nn.Module):
#     def __init__(self, VAE, opt_method='all_90_rot'):
#         super(AVAE3d, self).__init__()
#         self.VAE = VAE
#         self.opt_method = opt_method
#         self.rotation_only = True
#         if self.opt_method == 'all_90_rot':
#             self.all_90_rot = self.get_all_rot_mat()
            
#     def get_all_rot_mat(self):
#         def rot90_x(mat): return torch.mm(torch.FloatTensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]]), mat)
#         def rot90_y(mat): return torch.mm(torch.FloatTensor([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]), mat)
#         mat = torch.zeros(3, 4)
#         mat[:, :3] = torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
#         all_rotations = []
#         for cycle in range(2):
#             for step in range(3):
#                 mat[:, :3] = rot90_x(mat[:, :3])
#                 all_rotations.append(mat.clone())
#                 for i in range(3):
#                     mat[:, :3] = rot90_y(mat[:, :3])
#                     all_rotations.append(mat.clone())
#             mat[:, :3] = rot90_x(rot90_y(rot90_x(mat[:, :3])))
#         return torch.stack(all_rotations)
            
        
#     def affine(self, x, affine_params, padding_mode='zeros'):
#         grid = F.affine_grid(affine_params, x.size(), align_corners=False).to(x.device)
#         x = F.grid_sample(x, grid, padding_mode=padding_mode, align_corners=False)
#         return x
    
#     def affine_inv(self, x, affine_params, padding_mode='zeros'):
#         inv_affine_params = torch.FloatTensor(affine_params.size()).fill_(0).to(x.device)
#         if self.rotation_only: # inverse is transpose
#             inv_affine_params[:, :, :3] = torch.transpose(affine_params[:, :, :3], 1, 2) 
#         else:
#             pass
#         grid = F.affine_grid(inv_affine_params, x.size(), align_corners=False).to(x.device)
#         x = F.grid_sample(x, grid, padding_mode=padding_mode, align_corners=False)
#         return x
    
    
#     def forward(self, x, deterministic=False, opt_method=None):
#         # ??? this opt_method overrides the one when creating the model, 
# #         if opt_method==None:
#         opt_method = self.opt_method
#         if opt_method=='none': # affine is identity
#             affine_params = torch.zeros(3, 4).to(x.device)
#             affine_params[:, :3] = torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
#             affine_params = affine_params.unsqueeze(0).repeat(x.size(0), 1, 1).to(x.device)
#             recon_x, mu = self.affine_forward(x, affine_params=affine_params, deterministic=deterministic)
#         elif opt_method=='all_90_rot':
#             recon_x, mu, loss, affine_params = self.forward_best_90_rot(x, deterministic=True)
#             # Shouldn't need this:
#             recon_x, mu = self.affine_forward(x, affine_params=affine_params, deterministic=False)
#         return {'recon_x':recon_x, 
#                 'mu': mu,
#                 'affine_params': affine_params}
        
        
#     def affine_forward(self, x, affine_params=None, deterministic=False):
#         x_affine = self.affine(x, affine_params)
#         mu = self.VAE.encode(x_affine)
#         z = self.VAE.reparameterize(mu, deterministic)
#         recon_x = self.VAE.decode(z)
#         recon_x = self.affine_inv(recon_x, affine_params)
#         return recon_x, mu

    
#     def forward_best_90_rot(self, x, deterministic=True):
#         """ Try all 24 rotations.
        
#         Loss for all afffine params and imgs in parallel. 
#         Duplicate imgs & params since can only apply one affine per img.
#         returns: loss, affine_params(3x4)
#         """                                
#         with torch.no_grad():
#             affine_params = self.all_90_rot.clone().detach().to(x.device) # be safe
#             bs, ch, _, _, _ = x.size()
#             n_affine, _, _ = affine_params.size()
#             x_repeated = x.repeat(n_affine, 1, 1, 1, 1)
# #             print(affine_params.size())
# #             print(affine_params[0, :, :])
# #             affine_params_repeat = affine_params.repeat_interleave(bs).view(bs*n_affine, 3, 4).to(x.device)
#             affine_params_repeat = affine_params.repeat(bs, 1, 1).view(bs*n_affine, 3, 4).to(x.device)

# #             print('affine_params_repeat.size()', affine_params_repeat.size())
# #             print(affine_params_repeat[0, :, :])
            
#             recon_x, mu = self.affine_forward(x_repeated, affine_params=affine_params_repeat, deterministic=deterministic)
            
#             loss = self.vae_loss_unreduced((recon_x, mu), x_repeated)
#             loss = loss.view(n_affine, bs, 1)
#             best_loss, best_param_idx = torch.min(loss, dim=0)

#             # select the params, recon_x, mu corresponding to lowest loss
#             mu = mu.view(n_affine, bs, -1)
#             affine_params_repeat = affine_params_repeat.view(n_affine, bs, 3, 4)
# #             print(affine_params_repeat.size())
# #             print(affine_params_repeat[0, 0, :, :])

#             recon_x = recon_x.view(n_affine, bs, 1, recon_x.size(-3), recon_x.size(-2), recon_x.size(-1))
            
#             recon_x_best = recon_x[best_param_idx.squeeze(), torch.arange(bs), :, :, :, :]
#             mu_best = mu[best_param_idx.squeeze(), torch.arange(bs), :]
# #             print(best_param_idx)
# #             print(affine_params_repeat[0, :, :])

#             best_affine_params = affine_params_repeat[best_param_idx.squeeze(), torch.arange(bs), :, :]
# #             print(affine_params_repeat[0, :, :])
# #             print(best_affine_params)

#         return recon_x_best, mu_best, best_loss, best_affine_params
            
            
#     def vae_loss_unreduced(self, output, x, KLD_weight=1):
#         recon_x, mu  = output
#         BCE = F.binary_cross_entropy(recon_x.squeeze(), x.squeeze(), reduction='none')
#         BCE = torch.sum(BCE, dim=(1, 2, 3))
#         KLD = -0.5 * torch.sum(1 + 0 - mu.pow(2) - 1)
#         return BCE + KLD_weight*KLD

    
class VAE(nn.Module):
    """3d VAE. Force sigma=1"""
    def __init__(self, latent_size=8, img_size=32):
        super(VAE, self).__init__()
        self.latent_size = latent_size
        self.img_size = img_size
        self.linear_size = int(8*(img_size/8)**3)

        self.elu = nn.ELU()
        self.enc_conv1 = nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.enc_conv2 = nn.Conv3d(16, 16, kernel_size=5, stride=2, padding=2, bias=False)
        self.enc_conv3 = nn.Conv3d(16, 32, kernel_size=5, stride=2, padding=2, bias=False)
        self.enc_conv4 = nn.Conv3d(32, 8, kernel_size=3, stride=2, padding=1, bias=True)
        self.enc_bn1 = nn.BatchNorm3d(16)
        self.enc_bn2 = nn.BatchNorm3d(16)
        self.enc_bn3 = nn.BatchNorm3d(32)

        self.dec_conv1 = nn.ConvTranspose3d(8, 32, kernel_size=3, stride=2, padding=1,  output_padding=1, bias=False)
        self.dec_conv2 = nn.ConvTranspose3d(32, 16, kernel_size=5, stride=2, padding=2,  output_padding=1, bias=False)
        self.dec_conv3 = nn.ConvTranspose3d(16, 16, kernel_size=5, stride=2, padding=2,  output_padding=1, bias=False)
        self.dec_conv4 = nn.ConvTranspose3d(16, 1, kernel_size=3, stride=1, padding=1,  output_padding=0, bias=True)
        self.dec_bn1 = nn.BatchNorm3d(32)
        self.dec_bn2 = nn.BatchNorm3d(16)
        self.dec_bn3 = nn.BatchNorm3d(16)

        self.fc_enc = nn.Linear(self.linear_size, self.latent_size)
        self.fc_dec = nn.Linear(self.latent_size, self.linear_size)

    def forward(self, x, deterministic=False):
        mu = self.encode(x)
        z = self.reparameterize(mu, deterministic)
        recon_x = self.decode(z)
        return recon_x, mu

    def encode(self, x):
        x = self.elu(self.enc_bn1(self.enc_conv1(x)))
        x = self.elu(self.enc_bn2(self.enc_conv2(x)))
        x = self.elu(self.enc_bn3(self.enc_conv3(x)))
        x = self.enc_conv4(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_enc(x)
        return mu

    def decode(self, x):
        x = self.fc_dec(x)
        x = x.view((-1, 8, int(self.img_size/8), int(self.img_size/8), int(self.img_size/8)))
        x = self.elu(self.dec_bn1(self.dec_conv1(x)))
        x = self.elu(self.dec_bn2(self.dec_conv2(x)))
        x = self.elu(self.dec_bn3(self.dec_conv3(x)))
        x = self.dec_conv4(x)
        return torch.sigmoid(x)

    def reparameterize(self, mu, deterministic=False):
        if deterministic: 
            return mu
        else:
            return mu.add_(torch.randn_like(mu))