# dataloader here
from torch.utils.data import Dataset

from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import os.path as op
import SimpleITK as sitk
import json
from utils import load_from_yaml_file, read_json, load_config_file
import torch
import numpy as np


class MRI_dataset(Dataset):
    "MRI_dataset"

    def __init__(self, config, text_tokenizer):
        
        super(MRI_dataset, self).__init__()
        self.config = config
        self.filenames1, self.captions1 = get_files(self.config.text_dir_1)
        self.filenames2, self.captions2 = get_files(self.config.text_dir_2)
        self.filenames3, self.captions3 = get_files(self.config.text_dir_3)
        self.filenames4, self.captions4 = get_files(self.config.text_dir_4)
        self.filenames5, self.captions5 = get_files(self.config.text_dir_5)
        self.filenames6, self.captions6 = get_files(self.config.text_dir_6)
        self.filenames7, self.captions7 = get_files(self.config.text_dir_7)
        self.filenames8, self.captions8 = get_files(self.config.text_dir_8)
        self.filenames9, self.captions9 = get_files(self.config.text_dir_9)
        self.filenames10, self.captions10 = get_files(self.config.text_dir_10)
        self.img_dir_1 = self.config.img_dir_1
        self.img_dir_2 = self.config.img_dir_2
        self.img_dir_3 = self.config.img_dir_3
        self.img_dir_4 = self.config.img_dir_4
        self.img_dir_5 = self.config.img_dir_5
        self.img_dir_6 = self.config.img_dir_6
        self.img_dir_7 = self.config.img_dir_7
        self.img_dir_8 = self.config.img_dir_8
        self.img_dir_9 = self.config.img_dir_9
        self.img_dir_10 = self.config.img_dir_10
        self._tokenizer = text_tokenizer
        self.context_length =self.config.context_length

    def tokenize(self, text):
        sot_token = self._tokenizer.encoder["<|startoftext|>"]
        eot_token = self._tokenizer.encoder["<|endoftext|>"]
        tokens = [sot_token] + self._tokenizer.encode(text) + [eot_token]
        result = torch.zeros(self.context_length, dtype=torch.long)
        result[:len(tokens)] = torch.tensor(tokens)
        return result.unsqueeze(0)

    def __len__(self):
        return len(self.filenames1)

    def __getitem__(self, idx):
        file1 = self.filenames1[idx]
        file2 = self.filenames2[idx]
        file3 = self.filenames3[idx%(len(self.filenames3))]
        file4 = self.filenames4[idx%(len(self.filenames4))]
        file5 = self.filenames5[idx%(len(self.filenames5))]
        file6 = self.filenames6[idx%(len(self.filenames6))]
        file7 = self.filenames7[idx%(len(self.filenames7))]
        file8 = self.filenames8[idx%(len(self.filenames8))]
        file9 = self.filenames9[idx%(len(self.filenames9))]
        file10 = self.filenames10[idx%(len(self.filenames10))]
        img_1 = read_img(op.join(self.img_dir_1, file1))
        img_2 = read_img(op.join(self.img_dir_2, file2))
        img_3 = read_img(op.join(self.img_dir_3, file3))
        img_4 = read_img(op.join(self.img_dir_4, file4))
        img_5 = read_img(op.join(self.img_dir_5, file5))
        img_6 = read_img(op.join(self.img_dir_6, file6))
        img_7 = read_img(op.join(self.img_dir_7, file7))
        img_8 = read_img(op.join(self.img_dir_8, file8))
        img_9 = read_img(op.join(self.img_dir_9, file9))
        img_10 = read_img(op.join(self.img_dir_10, file10))
        text_1 = self.tokenize(self.captions1[idx])
        text_2 = self.tokenize(self.captions2[idx])
        text_3 = self.tokenize(self.captions3[idx%(len(self.filenames3))])
        text_4 = self.tokenize(self.captions4[idx%(len(self.filenames4))])
        text_5 = self.tokenize(self.captions5[idx%(len(self.filenames5))])
        text_6 = self.tokenize(self.captions6[idx%(len(self.filenames6))])
        text_7 = self.tokenize(self.captions7[idx%(len(self.filenames7))])
        text_8 = self.tokenize(self.captions8[idx%(len(self.filenames8))])
        text_9 = self.tokenize(self.captions9[idx%(len(self.filenames9))])
        text_10 = self.tokenize(self.captions10[idx%(len(self.filenames10))])
   
        imgs = torch.cat((img_1,img_2,img_3,img_4,img_5,img_6,img_7,img_8,img_9,img_10),dim=0)
        texts = torch.cat((text_1,text_2,text_3,text_4,text_5,text_6,text_7,text_8,text_9,text_10),dim=0)
    
        return imgs,texts

def percentile_clip(input_tensor, reference_tensor=None, p_min=0.01, p_max=99.9, strictlyPositive=True):
    if(reference_tensor == None):
        reference_tensor = input_tensor
    v_min, v_max = np.percentile(reference_tensor, [p_min,p_max]) 
    if( v_min < 0 and strictlyPositive): 
        v_min = 0
    output_tensor = np.clip(input_tensor,v_min,v_max) 
    output_tensor = (output_tensor - v_min)/(v_max-v_min) 
    return output_tensor

def get_files(text_path):
    filenames = []
    captions = []

    with open(text_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        splits = line.split(':',1)
        filename = splits[0].strip('""')
        caption = splits[1].strip('""')
        
        filenames.append(filename)
        captions.append(caption)
    return filenames, captions

def read_img(img_path):
    img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
    img = torch.tensor(img)
    img_input = percentile_clip(img)
    img_input = padding_or_crop(img_input)
    return img_input.unsqueeze(0)

def padding_or_crop(x,length=128):
    len = length
    # 计算需要crop或padding的数量
    d0 = len - x.shape[0]
    d1 = len - x.shape[1] 
    d2 = len - x.shape[2]
    # 根据正负选择crop或padding
    if d0 < 0: 
        x = x[abs(d0//2):-abs(d0//2)] # crop
    else:
        padding0 = torch.zeros(d0//2, x.shape[1], x.shape[2])
        x = torch.cat([padding0, x, padding0], dim=0) # padding

    if d1 < 0:
        x = x[:, abs(d1//2):-abs(d1//2),:] 
    else:
        padding1 = torch.zeros(x.shape[0], d1//2, x.shape[2])
        x = torch.cat([padding1, x, padding1], dim=1)

    if d2 < 0:
        x = x[:, :, abs(d2//2):-abs(d2//2)]
    else:
        padding2 = torch.zeros(x.shape[0], x.shape[1], d2//2) 
        x = torch.cat([padding2, x, padding2], dim=2)
    return x