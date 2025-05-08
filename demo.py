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

def set_new_spacing(ori_spacing, coord_size, crop_size):
    scale0 = coord_size[0] / crop_size[0]
    scale1 = coord_size[1] / crop_size[1]
    scale2 = coord_size[2] / crop_size[2]
    new_spacing = (ori_spacing[0]/scale0, ori_spacing[1]/scale1, ori_spacing[2]/scale2)
    return new_spacing
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

def _get_pred(crop_size, overlap_ratio, model, img_vol_0, coord_size, coord_hr, seq_tgt):
    W, H, D = img_vol_0.shape
    W_po, H_po, D_po = crop_size[0], crop_size[1], crop_size[2]
    W_pt, H_pt, D_pt = coord_size[0], coord_size[1], coord_size[2]
    scale0 = W_pt/W_po
    scale1 = H_pt/H_po
    scale2 = D_pt/D_po
    W_t = int(W * scale0)
    H_t = int(H * scale1)
    D_t = int(D * scale2)
    pos = calculate_patch_index((W, H, D), crop_size, overlap_ratio)
    pred_0_1 = np.zeros((W_t, H_t, D_t))
    freq_rec = np.zeros((W_t, H_t, D_t))
    start_time = time.time()
    for start_pos in pos:
        img_0_lr_patch = img_vol_0[start_pos[0]:start_pos[0] + crop_size[0], start_pos[1]:start_pos[1] + crop_size[1], start_pos[2]:start_pos[2] + crop_size[2]]
        img_0_lr_patch = torch.tensor(img_0_lr_patch).cuda().float().unsqueeze(0).unsqueeze(0)
        
        model.eval()
        with torch.no_grad():
            pred_0_1_patch = model(img_0_lr_patch, coord_hr, seq_tgt.cuda().float())
        pred_0_1_patch = pred_0_1_patch.squeeze(0).squeeze(-1).cpu().numpy().reshape(W_pt, H_pt, D_pt)
        
        target_pos0 = int(start_pos[0] * scale0)
        target_pos1 = int(start_pos[1] * scale1)
        target_pos2 = int(start_pos[2] * scale2)
        pred_0_1[target_pos0:target_pos0 + W_pt, target_pos1:target_pos1 + H_pt, target_pos2:target_pos2 + D_pt] += pred_0_1_patch[:, :, :]
        freq_rec[target_pos0:target_pos0 + W_pt, target_pos1:target_pos1 + H_pt, target_pos2:target_pos2 + D_pt] += 1
    end_time = time.time()
    print(end_time-start_time)
    pred_0_1_img = pred_0_1 / freq_rec

    return pred_0_1_img
    
img_model_pth = 'checkpoint.pt'
checkpoint_path = 'checkpoint_CLIP.pt'
img_path_0 = 'input image filefolder'
img_path_1 = 'resolution reference image filefolder'
prompt = 'target prompt file' #txt file, each row represents a target image corresponding to an input image
save_path = 'save dataroot'

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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
img_model = models.make(torch.load(img_model_pth)['model_G'], load_sd=True).cuda()

img_list_0 = sorted(os.listdir(img_path_0))
img_list_1 = sorted(os.listdir(img_path_1))
with open(prompt) as f1:
    lines_M1 = f1.readlines()

for idx, (i, j) in enumerate(zip(img_list_0, img_list_1)):  # img_list_0: input image; img_list_1: resolution reference image
    img_0 = sitk.ReadImage(os.path.join(img_path_0, i))
    img_0_spacing = img_0.GetSpacing()
    img_vol_0 = sitk.GetArrayFromImage(img_0)
    H, W, D = img_vol_0.shape
    img_vol_0 = img_pad(img_vol_0, target_shape=(H, W, D))
    img_1 = sitk.ReadImage(os.path.join(img_path_1, j))
    img_1_spacing = img_1.GetSpacing()
    img_vol_1 = sitk.GetArrayFromImage(img_1)
    img_vol_1 = img_pad(img_vol_1, target_shape=(H, W, D))
    img_vol_0 = utils.percentile_clip(img_vol_0)
    coord_size = [60, 60, 60]
    coord_hr = utils.make_coord(coord_size, flatten=True)
    coord_hr = torch.tensor(coord_hr).cuda().float().unsqueeze(0)
    text_tgt = lines_M1[idx].replace('"', '')
    text_tgt = text_tgt.strip((text_tgt.strip().split(':'))[0])
    text_tgt = text_tgt.strip(text_tgt[0])
    seq_tgt = tokenize(text_tgt, tokenizer).cuda()
    with torch.no_grad():
        seq_tgt = model.encode_text(seq_tgt)
    crop_size = (60, 60, 60)
    pred_0_1 = _get_pred(crop_size, 0.5, model_img, img_vol_0, coord_size, coord_hr, seq_tgt) # (input size; overlap ratio; model weight; input image; target size of patch; shape of target patch feature (don't need to change); target prompt)

    new_spacing_1 = set_new_spacing(img_1_spacing, coord_size, crop_size)
    
    utils.write_img(pred_0_1, save_path, os.path.join(img_path_1, j),new_spacing=new_spacing_1)
