import torch
from torch.utils.data import Dataset
from kornia.utils.grid import create_meshgrid, create_meshgrid3d
from einops import repeat
import numpy as np
from patterns import einops_f 


class CoordinateDataset(Dataset):
    def __init__(self, output, hparams, active_blocks = None):
        """
        output: Output Resized to current level
                subtracted by the upsampled reconstruction of the previous level
                in finer levels
        active_blocks: torch.tensor, None to return all blocks, otherwise specify the blocks to take
        """
        self.size = np.prod(hparams.patch_size)

        # split into patches
        output = einops_f(output, hparams.patterns['reshape'][3], hparams)
        self.output = torch.tensor(output)

        if hparams.task == 'image':
            input = create_meshgrid(hparams.p2, hparams.p1)
        elif hparams.task == 'mesh':
            input = create_meshgrid3d(hparams.p3, hparams.p2, hparams.p1)
        self.input = einops_f(input, hparams.patterns['reshape'][7])

        if active_blocks is None:
            self.input = repeat(self.input, '1 p c -> n p c', n = len(self.output))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return {'inp': self.input[:,idx], 'out': self.output[:,idx]}
