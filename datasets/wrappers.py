import random
from torch.utils.data import Dataset
from scipy.ndimage import zoom
from datasets import register
import numpy as np
from scipy import ndimage as nd
import utils


@register('sr-implicit-paired')
class SRImplicitPaired(Dataset):

    def __init__(self, dataset, scale_min=1, scale_max=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        patch_src_hr, patch_tgt_hr, seq_src, seq_tgt = self.dataset[idx]
        patch_src_hr = utils.percentile_clip(patch_src_hr)
        patch_tgt_hr = utils.percentile_clip(patch_tgt_hr)
        non_zero = np.nonzero(patch_src_hr)
        min_indice = np.min(non_zero, axis=1)
        max_indice = np.max(non_zero, axis=1)
        patch_src_hr = patch_src_hr[min_indice[0]:max_indice[0]+1, min_indice[1]:max_indice[1]+1, min_indice[2]:max_indice[2]+1]
        patch_tgt_hr = patch_tgt_hr[min_indice[0]:max_indice[0]+1, min_indice[1]:max_indice[1]+1, min_indice[2]:max_indice[2]+1]
        s = np.round(random.uniform(self.scale_min, self.scale_max), 1)
        size = (20 * s).astype(int)
        h0 = random.randint(0, patch_src_hr.shape[0] - size)
        w0 = random.randint(0, patch_src_hr.shape[1] - size)
        d0 = random.randint(0, patch_src_hr.shape[2] - size)
        patch_src_hr = patch_src_hr[h0:h0 + size, w0:w0 + size, d0:d0 + size]
        patch_tgt_hr = patch_tgt_hr[h0:h0 + size, w0:w0 + size, d0:d0 + size]
        patch_src_lr = zoom(patch_src_hr, 1 / s)
        patch_tgt_lr = zoom(patch_tgt_hr, 1 / s)
        coord_hr = utils.make_coord(patch_src_hr.shape, flatten=True)
        patch_src_hr = patch_src_hr.reshape(-1, 1)
        patch_tgt_hr = patch_tgt_hr.reshape(-1, 1)

        if self.sample_q is not None:
            sample_indices = np.random.choice(len(coord_hr), self.sample_q, replace=False)
            coord_hr = coord_hr[sample_indices]
            patch_src_hr = patch_src_hr[sample_indices]
            patch_tgt_hr = patch_tgt_hr[sample_indices]
        

        return {
            'src_lr': patch_src_lr,
            'tgt_lr': patch_tgt_lr,
            'src_hr': patch_src_hr,
            'tgt_hr': patch_tgt_hr,
            'coord_hr': coord_hr,
            'seq_src': seq_src,
            'seq_tgt': seq_tgt
        }


