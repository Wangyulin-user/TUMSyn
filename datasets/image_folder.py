import torch
from torch.utils.data import Dataset
from datasets import register
from utils_clip.simple_tokenizer import SimpleTokenizer
import numpy as np
from CLIP.model import CLIP
from utils_clip import load_config_file
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
model.eval()

@register('image-folder')
class ImageFolder(Dataset):

    def __init__(self, root_path_1, root_path_2, prompt_D1_M1, prompt_D1_M2, repeat=1, cache='none'):
        self.repeat = repeat
        self.cache = cache
        self.root_path_1 = root_path_1
        self.root_path_2 = root_path_2
        self.prompt_D1_M1 = prompt_D1_M1
        self.prompt_D1_M2 = prompt_D1_M2

        with open(self.root_path_1) as f1, open(self.root_path_2) as f2, open(self.prompt_D1_M1) as f3, open(
                self.prompt_D1_M2) as f4:
            img_M1 = f1.readlines()
            img_M2 = f2.readlines()
            prompt_M1 = f3.readlines()
            prompt_M2 = f4.readlines()
        self.img_M1 = img_M1
        self.img_M2 = img_M2
        self.prompt_M1 = prompt_M1
        self.prompt_M2 = prompt_M2

    def tokenize(self, texts, tokenizer, context_length=90):
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
    def __len__(self):
        return len(self.img_M1) * self.repeat

    def __getitem__(self, idx):
        patch_src_hr = self.img_M1[idx % len(self.img_M1)]
        patch_tgt_hr = self.img_M2[idx % len(self.img_M1)]

        text_src = self.prompt_M1[idx % len(self.img_M1)]
        text_tgt = self.prompt_M2[idx % len(self.img_M1)]
        text_src = text_src.replace('"', '')
        text_src = text_src.strip((text_src.strip().split(':'))[0])
        text_src = text_src.strip(text_src[0])
        text_tgt = text_tgt.replace('"', '')
        text_tgt = text_tgt.strip((text_tgt.strip().split(':'))[0])
        text_tgt = text_tgt.strip(text_tgt[0])

        seq_src = self.tokenize(text_src, tokenizer)
        with torch.no_grad():
            seq_src = model.encode_text(seq_src)
        seq_tgt = self.tokenize(text_tgt, tokenizer)
        with torch.no_grad():
            seq_tgt = model.encode_text(seq_tgt)

        img_vol_src_hr = np.load(patch_src_hr.strip())  
        img_vol_tgt_hr = np.load(patch_tgt_hr.strip())

        return img_vol_src_hr, img_vol_tgt_hr, seq_src, seq_tgt  # last_hidden_state


@register('paired-image-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, prompt_D1_M1, prompt_D1_M2, repeat, cache, **kwargs):
        self.dataset = ImageFolder(root_path_1, root_path_2, prompt_D1_M1, prompt_D1_M2, repeat, cache, **kwargs)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]