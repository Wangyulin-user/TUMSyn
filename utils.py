import os
import time
import shutil
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
from torch.optim import SGD, Adam
from tensorboardX import SummaryWriter
import random
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Variable

def eval_psnr(loader, model):
    model.eval()
    metric_fn = calc_psnr
    val_res1 = Averager()
    val_res0 = Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda().float()
        src_lr = batch['src_lr'].unsqueeze(1)
        tgt_lr = batch['tgt_lr'].unsqueeze(1)
        with torch.no_grad():
            pre_src_tgt, pre_tgt_src, _, _ = model(src_lr, tgt_lr, batch['coord_hr'], batch['seq_src'], batch['seq_tgt'])
                 
        res0 = metric_fn(pre_src_tgt, batch['tgt_hr'])
        res1 = metric_fn(pre_tgt_src, batch['src_hr'])
 
        val_res0.add(res0.item(), batch['src_hr'].shape[0])
        val_res1.add(res1.item(), batch['src_hr'].shape[0])

    return val_res0.item(),val_res1.item()

def percentile_clip(input_tensor, reference_tensor=None, p_min=0.01, p_max=99.9, strictlyPositive=True):
    if(reference_tensor == None):
        reference_tensor = input_tensor
    v_min, v_max = np.percentile(reference_tensor, [p_min,p_max]) #get p_min percentile and p_max percentile
    if( v_min < 0 and strictlyPositive): #set lower bound to be 0 if it would be below
        v_min = 0
    output_tensor = np.clip(input_tensor,v_min,v_max) #clip values to percentiles from reference_tensor
    output_tensor = (output_tensor - v_min)/(v_max-v_min) #normalizes values to [0;1]
    return output_tensor

def random_selection(input_list):
    num_to_select = random.randint(1, 3)  # 随机选择1到3个数字
    selected_numbers = random.sample(input_list, num_to_select)  # 从列表中选择随机数字
    return selected_numbers

class Loss_CC(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, m):
        b, c, h, w = m.shape
        m = m.reshape(b, c, h*w)
        m = torch.nn.functional.normalize(m, dim=2, p=2)
        m_T = torch.transpose(m, 1, 2)
        m_cc = torch.matmul(m, m_T)
        mask = torch.eye(c).unsqueeze(0).repeat(b,1,1).cuda()
        m_cc = m_cc.masked_fill(mask==1, 0)
        loss = torch.sum(m_cc**2)/(b*c*(c-1))
        return loss
    
class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer_G(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd_G'])
    return optimizer

def make_optimizer_D(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd_D'])
    return optimizer

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret



def calc_psnr(sr, hr):
    diff = (sr - hr) 
    mse = diff.pow(2).mean()
    return -10 * torch.log10(mse)

def write_middle_feature(intermediate_output):
    for i in range(intermediate_output.shape[1]):
        activation = intermediate_output[0, i, :, :, :]
        plt.savefig(f'./save/layer_{i}_activation_{activation}.png')  # Save each activation as a PNG file
        plt.clf()
def write_img(vol, out_path, ref_path, new_spacing=None):
    img_ref = sitk.ReadImage(ref_path)
    img = sitk.GetImageFromArray(vol)
    img.SetDirection(img_ref.GetDirection())
    if new_spacing is None:
        img.SetSpacing(img_ref.GetSpacing())
    else:
        img.SetSpacing(tuple(new_spacing))
    img.SetOrigin(img_ref.GetOrigin())
    sitk.WriteImage(img, out_path)
    print('Save to:', out_path)

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

