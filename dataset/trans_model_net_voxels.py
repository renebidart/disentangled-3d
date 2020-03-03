import os
import math
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from kaolin.datasets import ModelNet, ModelNetVoxels


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
    