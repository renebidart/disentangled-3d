import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def pad_3d(x, pad_factor=1.5):
    pad_amt = ((pad_factor-1)*torch.tensor(x.size()[-3:]))/2
    pad_amt = [[int(x.item()), int(x.item())] for x in pad_amt]
    pad_amt = [item for sublist in pad_amt for item in sublist]
    img = F.pad(x, pad_amt, 'constant', 0)
    return img

def pad_for_transform(x, max_trans=.4, rot_factor=1.1):
    """pad the input and adjust the scaling and translation so the object stays withing the boundary
    Default is 40% translation, and scale object by this. 
    Factor of 1.1 added because of rotations (worst case would need 1.4)
    Assume scaling==1
    weirdly in pytorch you use scale as 1/scale, and translation as relative amount of translation
    """
    assert transform['scale'] == 1.0
    pad_factor = np.round(rot_factor * (1.0+max_trans), decimals=1)
    x_pad = pad_3d(x, pad_factor=pad_factor)
    # Final translation is adjusted to be in terms of final object which is increased by pad_factor
    transform['trans'] =  np.array(transform['trans']) / pad_factor
    return x_pad


def affine_transform(x, affine_mat, padding_mode='zeros'):
    grid = F.affine_grid(affine_mat, x.size(), align_corners=False).cuda()
    x = F.grid_sample(x, grid, padding_mode=padding_mode, align_corners=False)
    return x


def make_affine(r, t, device):
    """Differentiable batch version, so will only optimize rotation and translation,
    If translation==None, treats it as fixed identity translation
    """
    n, _ = r.size()
    r = r.to(device)
    if t is None: # ignore translation entirely, don't create params
        pass
    else:
        assert r.size(0) == t.size(0)
        t = t.to(device)
        tx = t[:, 0].view(-1,1,1).repeat(1,3,4)*torch.tensor([[0,0,0,1],
                                                             [0,0,0,0],
                                                             [0,0,0,0]], dtype=torch.float32, device=device
                                                            ).view(1,3,4).repeat(n, 1, 1)
        ty = t[:, 1].view(-1,1,1).repeat(1,3,4)*torch.tensor([[0,0,0,0],
                                                             [0,0,0,1],
                                                             [0,0,0,0]], dtype=torch.float32, device=device
                                                              ).view(1,3,4).repeat(n, 1, 1)
        tz = t[:, 2].view(-1,1,1).repeat(1,3,4)*torch.tensor([[0,0,0,0],
                                                             [0,0,0,0],
                                                             [0,0,0,1]], dtype=torch.float32, device=device
                                                            ).view(1,3,4).repeat(n, 1, 1)
    identity = torch.tensor([[1,0,0,0],
                             [0,1,0,0],
                             [0,0,1,0]], dtype=torch.float32, device=device).view(1,3,4).repeat(n, 1, 1)
    
    x_sin_mask = torch.tensor([[0,0,0,0],\
      [0,0,-1,0],
      [0,1,0,0],
      [0,0,0,0]], dtype=torch.float32, device = device).view(1,4,4).repeat(n, 1, 1)
    x_cos_mask = torch.tensor([[0,0,0,0],\
      [0,1,0,0],
      [0,0,1,0],
      [0,0,0,0]], dtype=torch.float32, device = device).view(1,4,4).repeat(n, 1, 1)
    y_sin_mask = torch.tensor([[0,0,1,0],\
      [0,0,0,0],
      [-1,0,0,0],
      [0,0,0,0]], dtype=torch.float32, device = device).view(1,4,4).repeat(n, 1, 1)
    y_cos_mask = torch.tensor([[1,0,0,0],\
      [0,0,0,0],
      [0,0,1,0],
      [0,0,0,0]], dtype=torch.float32, device = device).view(1,4,4).repeat(n, 1, 1)
    z_sin_mask = torch.tensor([[0,-1,0,0],\
      [1,0,0,0],
      [0,0,0,0],
      [0,0,0,0]], dtype=torch.float32, device = device).view(1,4,4).repeat(n, 1, 1)
    z_cos_mask = torch.tensor([[1,0,0,0],\
      [0,1,0,0],
      [0,0,0,0],
      [0,0,0,0]], dtype=torch.float32, device = device).view(1,4,4).repeat(n, 1, 1)
    base_x = torch.tensor([[1,0,0,0],\
      [0,0,0,0],\
      [0,0,0,0],\
      [0,0,0,1]], dtype=torch.float32, device = device).view(1,4,4).repeat(n, 1, 1)
    base_y = torch.tensor([[0,0,0,0],\
      [0,1,0,0],\
      [0,0,0,0],\
      [0,0,0,1]], dtype=torch.float32, device = device).view(1,4,4).repeat(n, 1, 1)
    base_z = torch.tensor([[0,0,0,0],\
      [0,0,0,0],\
      [0,0,1,0],\
      [0,0,0,1]], dtype=torch.float32, device = device).view(1,4,4).repeat(n, 1, 1)

    rot_x = torch.cos(r[:, 0].view(-1, 1, 1).repeat(1,4,4)) * x_cos_mask + \
            torch.sin(r[:, 0].view(-1, 1, 1).repeat(1,4,4)) * x_sin_mask + base_x
    rot_y = torch.cos(r[:, 1].view(-1, 1, 1).repeat(1,4,4)) * y_cos_mask + \
            torch.sin(r[:, 1].view(-1, 1, 1).repeat(1,4,4)) * y_sin_mask + base_y
    rot_z = torch.cos(r[:, 2].view(-1, 1, 1).repeat(1,4,4)) * z_cos_mask + \
            torch.sin(r[:, 2].view(-1, 1, 1).repeat(1,4,4)) * z_sin_mask + base_z

    rot_mat = torch.bmm(torch.bmm(rot_x, rot_y), rot_z)[:, :3, :4]   
    if t is None: 
        affine_mat = rot_mat
    else:  
        affine_mat = rot_mat + tx + ty + tz
    return affine_mat


def show_rotations(x, rot_mat_list):
    nrows, ncols = int(len(rot_mat_list)/2), 2
    fig = plt.figure(figsize=(10, nrows*4))
    for idx in range(0, nrows*ncols, 1):
        rotated_x = affine_transform(x, rot_mat_list[idx].to(device))    
        rotated_x = rotated_x > .5
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection='3d')
        ax.voxels(rotated_x.squeeze(), facecolors='blue', edgecolor='k')
        
def visualize_optimization(x, affine_trans_dicts, device):
    nrows, ncols = len(affine_trans_dicts), 3
    fig = plt.figure(figsize=(10, nrows*4))
    for idx in range(0, nrows, 1): 
        # Grab the new x
        x_pad = pad_3d(x, pad_factor=1.5).detach()
        r_init = torch.tensor([0., 0., 0.], requires_grad=True)
        x_goal = affine_transform(x_pad, affine_mat=make_rot(torch.FloatTensor(affine_trans_dicts[idx]['rot']), device))
        loss, rot_mat, rot  = optimize_rotation(x_pad, x_goal, r_init)
        x_final = affine_transform(x_pad, affine_mat=rot_mat)
        x_pad = x_pad > .5
        x_goal = x_goal > .5
        x_final = x_final > .5   
        rot = (180.*rot/math.pi).detach().numpy()
        objective_rot = (180.*np.array(affine_trans_dicts[idx]["rot"])/math.pi)
        
        ax = fig.add_subplot(nrows, ncols, 3*idx + 1, projection='3d')
        ax.voxels(x_pad.squeeze(), facecolors='blue', edgecolor='k')
        ax.set_title('Initial', fontsize='large')
        ax.axis('off')
        ax = fig.add_subplot(nrows, ncols, 3*idx + 2, projection='3d')
        ax.voxels(x_final.squeeze(), facecolors='blue', edgecolor='k')    
        ax.set_title(f'Result: {rot[0]:.1f}$^\circ$', fontsize='large')
        ax.axis('off')
        ax = fig.add_subplot(nrows, ncols, 3*idx + 3, projection='3d')
        ax.voxels(x_goal.squeeze(), facecolors='blue', edgecolor='k')
        ax.set_title(f'Objective: {objective_rot[0]:.1f}$^\circ$', fontsize='large')
        ax.axis('off')
    fig.tight_layout()
        
def rotationMatrixToEulerAngles(R) :
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    singular = sy < 1e-6
    if  not singular:
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
    return np.round(np.array([x, y, z]), 4)

        
    

# Should be deleted but used in notebooks:
def get_affine_matrix(rot, trans, scale=1):
    """ Not differentiable version
    rotation[x, y, z], translation[x, y, z], scale[s] -> affine transform matrix"""
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(rot[0]), -math.sin(rot[0]) ],
                    [0,         math.sin(rot[0]), math.cos(rot[0])  ]
                   ])
    R_y = np.array([[math.cos(rot[1]),    0,      math.sin(rot[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(rot[1]),   0,      math.cos(rot[1])  ]
                   ])
    R_z = np.array([[math.cos(rot[2]),    -math.sin(rot[2]),    0],
                    [math.sin(rot[2]),    math.cos(rot[2]),     0],
                    [0,                     0,                      1]])
    S = np.array([[scale,      0,                  0],
                  [0,             scale,           0],
                  [0,             0,             scale]])
    R = np.dot(R_z, np.dot( R_y, R_x ))
    RS = np.dot(R, S)
    affine_mat = torch.zeros(1, 3, 4)
    affine_mat[:, :, :3] = torch.FloatTensor(RS)
    affine_mat[:, :, 3] = torch.FloatTensor([trans[0],
                                             trans[1],
                                             trans[2]])
    return affine_mat