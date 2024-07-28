import models
import torch
import SimpleITK as sitk
import utils
from utils_clip.simple_tokenizer import SimpleTokenizer
import numpy as np
import os
from itertools import product
from CLIP.model import CLIP
from utils_clip import load_config_file
import time
from torch.utils.data import DataLoader, Dataset
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
checkpoint_path = 'checkpoint_CLIP.pt'
MODEL_CONFIG_PATH = 'CLIP/model_config.yaml'
model_config = load_config_file(MODEL_CONFIG_PATH)

tokenizer = SimpleTokenizer()
model_params = dict(model_config.RN50)
model_params['vision_layers'] = tuple(model_params['vision_layers'])
model_params['vision_patch_size'] = None
model = CLIP(**model_params)
checkpoint = torch.load(checkpoint_path)
state_dict = checkpoint['model_state_dict']

model.load_state_dict(state_dict)
model = model.cuda()
model.eval()


def tokenize(texts, tokenizer, context_length=90):
    if isinstance(texts, str):
        texts = [texts]

    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)
    return result

def img_pad(img, target_shape):
    current_shape = img.shape
    pads = [(0, max(0, target_shape[i] - current_shape[i])) for i in range(len(target_shape))]
    padded_img = np.pad(img, pads, mode='constant', constant_values=0)
    current_shape_2 = padded_img.shape
    crops = []
    for i in range(len(target_shape)):
        if current_shape_2[i] > target_shape[i]:
            crops.append(
                slice((current_shape_2[i] - target_shape[i]) // 2, (current_shape_2[i] + target_shape[i]) // 2))
        else:
            crops.append(slice(None))
    cropped_img = padded_img[tuple(crops)]
    return cropped_img

def calculate_patch_index(target_size, patch_size, overlap_ratio=0.25):
    shape = target_size

    gap = int(patch_size[0] * (1 - overlap_ratio))
    index1 = [f for f in range(shape[0])]
    index_x = index1[::gap]
    index2 = [f for f in range(shape[1])]
    index_y = index2[::gap]
    index3 = [f for f in range(shape[2])]
    index_z = index3[::gap]

    index_x = [f for f in index_x if f < shape[0] - patch_size[0]]
    index_x.append(shape[0] - patch_size[0])
    index_y = [f for f in index_y if f < shape[1] - patch_size[1]]
    index_y.append(shape[1] - patch_size[1])
    index_z = [f for f in index_z if f < shape[2] - patch_size[2]]
    index_z.append(shape[2] - patch_size[2])

    start_pos = list()
    loop_val = [index_x, index_y, index_z]
    for i in product(*loop_val):
        start_pos.append(i)
    return start_pos

def patch_slicer(img_vol_0, overlap_ratio, crop_size, scale0, scale1, scale2):
    W, H, D = img_vol_0.shape
    pos = calculate_patch_index((W, H, D), crop_size, overlap_ratio)
    scan_patches = []
    patch_idx = []
    for start_pos in pos:
        img_0_lr_patch = img_vol_0[start_pos[0]:start_pos[0] + crop_size[0], start_pos[1]:start_pos[1] + crop_size[1],
                         start_pos[2]:start_pos[2] + crop_size[2]]
        #print(img_0_lr_patch.shape)
        scan_patches.append(torch.tensor(img_0_lr_patch).float().unsqueeze(0))
        patch_idx.append([int(start_pos[0]), int(start_pos[0])+int(crop_size[0] * scale0), int(start_pos[1]), int(start_pos[1])+int(crop_size[1] * scale1), int(start_pos[2]), int(start_pos[2])+int(crop_size[2] * scale2)])
    return scan_patches, patch_idx

class PatchDataset(Dataset):
    def __init__(self, patches):
        self.patches = patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx]

def _get_pred(model, dataloader, coord_hr, seq_tgt):
    model.eval()
    results = []
    with torch.no_grad():
        for batch in dataloader:
            batchsize = batch.size(0)
            input_patch = batch.cuda()
            batch_coord_hr = coord_hr.repeat(batchsize, 1, 1)
            tgt_prompt = seq_tgt.repeat(batchsize, 1)
            pred_0_1_patch = model(input_patch, batch_coord_hr, tgt_prompt.cuda().float())
            results.extend(pred_0_1_patch)
    return results
torch.multiprocessing.set_start_method('spawn', force=True)
if __name__ == '__main__':
    psnr_0_1_list = []
    psnr_1_0_list = []
    ssim_0_1_list = []
    ssim_1_0_list = []
    model_pth = '.../TUMSyn/save/checkpoint.pth'
    model_img = models.make(torch.load(model_pth)['model_G'], load_sd=True).cuda()
    img_path_0 = r'.../TUMSyn/Experimental_data/image'
    img_path_1 = r'.../TUMSyn/Experimental_data/image' # Using to provide target image spacing, it is not necessary, the target image spacing can be manually set
    img_list_0 = sorted(os.listdir(img_path_0))
    img_list_1 = sorted(os.listdir(img_path_1))
    prompt_M1 = r'.../TUMSyn/Experimental_data/test_HCPD_T2w.txt'
    with open(prompt_M1) as f1:
        lines_M1 = f1.readlines()

    img_0 = sitk.ReadImage(os.path.join(img_path_0, 'test_HCPD_T1w.nii.gz'))
    img_0_spacing = img_0.GetSpacing()
    img_vol_0 = sitk.GetArrayFromImage(img_0)
    H, W, D = img_vol_0.shape
    img_vol_0 = img_pad(img_vol_0, target_shape=(H, W, D))
    img_vol_0 = utils.percentile_clip(img_vol_0)
    coord_size = [60, 60, 60]
    coord_hr = utils.make_coord(coord_size, flatten=True)
    coord_hr = torch.tensor(coord_hr).cuda().float()
    text_tgt = lines_M1[0].replace('"', '')
    text_tgt = text_tgt.strip((text_tgt.strip().split(':'))[0])
    text_tgt = text_tgt.strip(text_tgt[0])
    seq_tgt = tokenize(text_tgt, tokenizer).cuda()
    with torch.no_grad():
        seq_tgt = model.encode_text(seq_tgt)
    crop_size = (60, 60, 60)
    scale0 = coord_size[0] / crop_size[0]
    scale1 = coord_size[1] / crop_size[1]
    scale2 = coord_size[2] / crop_size[2]
    patches, _ = patch_slicer(img_vol_0, 0.5, crop_size, scale0, scale1, scale2)
    dataset = PatchDataset(patches)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    pred_0_1 = _get_pred(model_img, dataloader, coord_hr, seq_tgt)
    utils.write_img(pred_0_1, os.path.join('save_path'),
                    os.path.join(img_path_1, 'test_HCPD_T2w.nii.gz'), new_spacing=None)