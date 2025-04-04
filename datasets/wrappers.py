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
        patch_src_hr = utils.percentile_clip(patch_src_hr)  # Image normalization
        patch_tgt_hr = utils.percentile_clip(patch_tgt_hr)
        non_zero = np.nonzero(patch_src_hr) # Remove slices that are completely black
        min_indice = np.min(non_zero, axis=1)
        max_indice = np.max(non_zero, axis=1)
        patch_src_hr = patch_src_hr[min_indice[0]:max_indice[0]+1, min_indice[1]:max_indice[1]+1, min_indice[2]:max_indice[2]+1]
        patch_tgt_hr = patch_tgt_hr[min_indice[0]:max_indice[0]+1, min_indice[1]:max_indice[1]+1, min_indice[2]:max_indice[2]+1]
        s = np.round(random.uniform(self.scale_min, self.scale_max), 1) #Extract 3D patches from the full volume; patch sizes range from 20×20×20 to 60×60×60
        size = (20 * s).astype(int)
        h0 = random.randint(0, patch_src_hr.shape[0] - size)
        w0 = random.randint(0, patch_src_hr.shape[1] - size)
        d0 = random.randint(0, patch_src_hr.shape[2] - size)
        patch_src_hr = patch_src_hr[h0:h0 + size, w0:w0 + size, d0:d0 + size] #Obtain high-resolution patches from the extracted volumes
        patch_tgt_hr = patch_tgt_hr[h0:h0 + size, w0:w0 + size, d0:d0 + size]
        patch_src_lr = zoom(patch_src_hr, 1 / s) # Downsample each high-resolution patch to a fixed size of 20×20×20 to ensure consistent input dimensions
        patch_tgt_lr = zoom(patch_tgt_hr, 1 / s)
        coord_hr = utils.make_coord(patch_src_hr.shape, flatten=True) # Make coordinates at grid centers.
        patch_src_hr = patch_src_hr.reshape(-1, 1) # Flatten the high-resolution patch so that its shape matches the coordinate grid
        patch_tgt_hr = patch_tgt_hr.reshape(-1, 1)

        if self.sample_q is not None: # Select the target number of pixels and their coordinates from the high-resolution patch for prediction
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


