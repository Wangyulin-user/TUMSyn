import torch
import torch.nn as nn
from models.cross_att import Basic_block
import torch.nn.functional as F
from models.linear import Linear
from models.linear import Linear_CNN
import models
from models import register
from einops import rearrange
@register('lccd')
class LCCD(nn.Module):

    def __init__(self, encoder_spec, no_imnet):
        super().__init__()
        self.encoder = models.make(encoder_spec)
        self.f_dim = self.encoder.out_dim
        self.fusion = Basic_block(dim=self.f_dim, num_heads=8)

        if no_imnet:
            self.imnet = None
        else:
            self.imnet = Linear(in_dim=self.f_dim+3,out_dim=1,hidden_list=[3072, 3072, 768, 256])

    def gen_feat(self, inp):
        feat = self.encoder(inp)
        return feat

    
    #tarin together
    def forward(self, src_lr, tgt_lr, coord_hr, prompt_src, prompt_tgt):
        N, K = coord_hr.shape[:2]
        feat_src_lr = self.gen_feat(src_lr) # Extract features from the low-resolution input image
        feat_src_lr_tgt = self.fusion(feat_src_lr, prompt_tgt) #Fuse extracted features with the target image prompt
        vector_src_tgt = F.grid_sample(feat_src_lr_tgt, coord_hr.flip(-1).unsqueeze(1).unsqueeze(1), # Sample local feature vectors from the fused low-resolution feature map at the given high-resolution coordinates
                                       mode='bilinear',
                                       align_corners=False)[:, :, 0, 0, :].permute(0, 2, 1)
        vector_src_tgt_with_coord = torch.cat([vector_src_tgt, coord_hr], dim=-1) # Concatenate the sampled feature vectors with their corresponding normalized coordinates
        pre_src_tgt = self.imnet(vector_src_tgt_with_coord.view(N * K, -1)).view(N, K, -1) # Predict intensity values at queried coordinates using the implicit decoder
        
        feat_tgt_lr = self.gen_feat(tgt_lr)
        feat_tgt_lr_src = self.fusion(feat_tgt_lr, prompt_src)
        vector_tgt_src = F.grid_sample(feat_tgt_lr_src, coord_hr.flip(-1).unsqueeze(1).unsqueeze(1),
                                       mode='bilinear',
                                       align_corners=False)[:, :, 0, 0, :].permute(0, 2, 1)
        vector_tgt_src_with_coord = torch.cat([vector_tgt_src, coord_hr], dim=-1)
        pre_tgt_src = self.imnet(vector_tgt_src_with_coord.view(N * K, -1)).view(N, K, -1)
        return pre_src_tgt, pre_tgt_src, feat_src_lr, feat_tgt_lr
        

    # test forward
    def forward(self, src_lr, coord_hr, prompt_tgt):
        N, K = coord_hr.shape[:2]
        feat_src_lr = self.gen_feat(src_lr)
        feat_src_lr_tgt = self.fusion(feat_src_lr, prompt_tgt)
        vector_src_tgt = F.grid_sample(feat_src_lr_tgt, coord_hr.flip(-1).unsqueeze(1).unsqueeze(1),
                                       mode='bilinear',
                                       align_corners=False)[:, :, 0, 0, :].permute(0, 2, 1)
        vector_src_tgt_with_coord = torch.cat([vector_src_tgt, coord_hr], dim=-1)
        pre_src_tgt = self.imnet(vector_src_tgt_with_coord.view(N * K, -1)).view(N, K, -1)
        return pre_src_tgt


       
