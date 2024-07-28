import argparse
import os
import yaml
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import utils
import datasets
import models

def make_data_loader(spec, tag=''):
    if spec is None:
        return None
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})

    log('{} dataset: size={}'.format(tag, len(dataset)))
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        shuffle=(tag == 'train'), num_workers=4, pin_memory=True)
    return loader


def make_data_loaders():
    train_loader = make_data_loader(config.get('train_dataset'), tag='train')
    val_loader = make_data_loader(config.get('val_dataset'), tag='val')
    return train_loader, val_loader


def prepare_training():
    if config.get('resume') is not None:
        sv_file = torch.load(config['resume'])
        model_G = models.make(sv_file['model_G'], load_sd=True).cuda()

        optimizer_G = utils.make_optimizer_G(
            model_G.parameters(), sv_file['optimizer_G'], load_sd=True)

        epoch_start = sv_file['epoch'] + 1
        if config.get('multi_step_lr') is None:
            lr_scheduler_G = None
        else:
            lr_scheduler_G = MultiStepLR(optimizer_G, **config['multi_step_lr'])

        for _ in range(epoch_start - 1):
            lr_scheduler_G.step()
    else:
        model_G = models.make(config['model_G']).cuda()

        optimizer_G = utils.make_optimizer_G(
            model_G.parameters(), config['optimizer_G'])

        epoch_start = 1
        if config.get('multi_step_lr') is None:
            lr_scheduler_G = None

        else:
            lr_scheduler_G = MultiStepLR(optimizer_G, **config['multi_step_lr'])

    log('model_G: #params={}'.format(utils.compute_num_params(model_G, text=True)))

    return model_G, optimizer_G, epoch_start, lr_scheduler_G


def train(train_loader, model_G, optimizer_G):
    model_G.train()

    loss_fn = nn.L1Loss()
    loss_0 = utils.Averager()
    loss_1 = utils.Averager()

    for batch in tqdm(train_loader, leave=False, desc='train'):
        for k, v in batch.items():
            batch[k] = v.cuda().float()
        src_lr = batch['src_lr'].cuda().unsqueeze(1)
        tgt_lr = batch['tgt_lr'].cuda().unsqueeze(1)
        coord_hr = batch['coord_hr'].cuda()
        seq_src = batch['seq_src'].cuda()
        seq_tgt = batch['seq_tgt'].cuda()
        tgt_hr = batch['tgt_hr'].cuda()
        src_hr = batch['src_hr'].cuda()
        pre_src_tgt, pre_tgt_src, feat_src_lr, feat_tgt_lr = model_G(src_lr, tgt_lr, coord_hr, seq_src, seq_tgt)
        loss_src = loss_fn(pre_src_tgt, tgt_hr)
        loss_tgt = loss_fn(pre_tgt_src, src_hr)

        loss_G = loss_src * 0.5 + loss_tgt * 0.5
        loss_0.add(loss_src.item())
        loss_1.add(loss_tgt.item())

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()
        grad_norm = 0
        for name, param in model_G.named_parameters():
            #print('name:', name)
            grad_norm += param.grad.norm(2) ** 2
        if grad_norm > 1e5:
            print('gradient explosion')
    return loss_0.item(), loss_1.item()


def main(config_, save_path):
    global config, log
    config = config_
    log, _ = utils.set_save_path(save_path)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, val_loader = make_data_loaders()
    model_G, optimizer_G, epoch_start, lr_scheduler_G = prepare_training()
    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model_G.cuda()
        model_G = nn.parallel.DistributedDataParallel(model_G)
    epoch_max = config['epoch_max']
    epoch_val = config.get('epoch_val')
    epoch_save = config.get('epoch_save')
    max_val_v = -1e18

    timer = utils.Timer()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        optimizer_G.param_groups[0]['lr'] = 0.00001
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]
        log_info.append('lr_G={:.6f}'.format(optimizer_G.param_groups[0]['lr']))

        train_loss = train(train_loader, model_G, optimizer_G)
        if lr_scheduler_G is not None:
            lr_scheduler_G.step()
        log_info.append('loss0={:.4f}'.format(train_loss[0]))
        log_info.append('loss1={:.4f}'.format(train_loss[1]))

        if n_gpus > 1:
            model_G_ = model_G.module
        else:
            model_G_ = model_G
        model_G_spec = config['model_G']
        model_G_spec['sd_G'] = model_G_.state_dict()
        optimizer_G_spec = config['optimizer_G']
        optimizer_G_spec['sd_G'] = optimizer_G.state_dict()
        sv_file = {
            'model_G': model_G_spec,
            'optimizer_G': optimizer_G_spec,
            'epoch': epoch
        }

        torch.save(sv_file, os.path.join(save_path, 'epoch-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file,
                       os.path.join(save_path, 'epoch-{}.pth'.format(epoch)))

        if (epoch_val is not None) and (epoch % epoch_val == 0):
            val_res0, val_res1 = utils.eval_psnr(val_loader, model_G_)

            log_info.append('val: psnr0={:.4f}'.format(val_res0))
            log_info.append('val: psnr1={:.4f}'.format(val_res1))

            if val_res0 + val_res1 > max_val_v:
                max_val_v = val_res0 + val_res1
                torch.save(sv_file, os.path.join(save_path, 'epoch-best.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/train_lccd_sr.yaml')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = '_' + args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    # save_path = os.path.join('./save', save_name)
    save_path = os.path.join('./save', save_name)
    main(config, save_path)