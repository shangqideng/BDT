# -*- encoding: utf-8 -*-
import math

import torch
from torch import optim
# from UDL.Basis.criterion_metrics import *
# from UDL.pansharpening.common.evaluate import analysis_accu
# from UDL.Basis.module import PatchMergeModule
# from UDL.Basis.pytorch_msssim.cal_ssim import SSIM
from bidirection_x8 import *
from torchsummary import summary
import torch.nn.functional as F

def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):   ## initialization for Conv2d
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):   ## initialization for BN
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):     ## initialization for nn.Linear
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

def init_w(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                # torch.nn.init.uniform_(m.weight, a=0, b=1)
            elif isinstance(m, nn.Conv2d):   ## initialization for Conv2d
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


class Bidinet(nn.Module):
    def __init__(self, args):
        super(Bidinet, self).__init__()
        self.args = args
        # self.img_size = 1024
        self.img_size = 512
        self.in_channels = 31
        self.embed_dim = 48  # w-msa
        self.dim = 32  # w-xca
        self.t = Merge(img_size=self.img_size, patch_size=1, in_chans1=34, in_chans2=31, embed_dim=48, num_heads1=[8, 8, 8, 8], window_size=8, group=8,  mlp_ratio=4.,dim=32,
                 num_heads2=[8, 8, 8, 8],  ffn_expansion_factor=2.66,  LayerNorm_type = 'WithBias', bias=False)
        self.visual_corresponding_name = {}
        init_w(self.t)
        self.visual_corresponding_name['gt'] = 'result'
        self.visual_names = ['gt', 'result']

    def forward(self, rgb, lms, ms):
        '''
        :param pan:
        :param ms:
        :return:
        '''
        self.rgb = rgb
        self.lms = lms
        self.ms = ms # Bx31X10x10
        xt = torch.cat((self.lms, self.rgb), 1)  # Bx34X80x80

        # # padding
        # pad = 8
        # xt = F.pad(xt, (pad, pad, pad, pad))
        # # xt = F.pad(xt, (pad, pad, pad, pad), mode='reflect')
        # self.lms = F.pad(self.lms, (pad, pad, pad, pad))
        # self.ms = F.pad(self.ms, (pad//8, pad//8, pad//8, pad//8))

        w_out = self.t(xt, self.ms)
        # padding
        self.result = w_out + self.lms
        # self.result = self.result[:, :, pad:80+pad, pad:80+pad]

        return self.result

    def name(self):
        return ' net'
    #
    # def train_step(self, batch, *args, **kwargs):
    #     gt, up, hsi, msi = batch['gt'].cuda(), \
    #                        batch['up'].cuda(), \
    #                        batch['lrhsi'].cuda(), \
    #                        batch['rgb'].cuda()
    #     # x = torch.cat((up, msi), 1)
    #     sr = self(msi, up, hsi)
    #     loss = self.criterion(sr, gt, *args, **kwargs)
    #     log_vars = {}
    #     with torch.no_grad():
    #         metrics = analysis_accu(gt, sr, 4, choices=4)
    #         log_vars.update(metrics)
    #
    #     return {'loss': loss , 'log_vars': log_vars}
    #
    # def eval_step(self, batch, *args, **kwargs):
    #     gt, up, hsi, msi = batch['gt'].cuda(), \
    #                        batch['up'].cuda(), \
    #                        batch['lrhsi'].cuda(), \
    #                        batch['rgb'].cuda()
    #
    #     sr1 = self.forward(msi, up, hsi)
    #     with torch.no_grad():
    #         metrics = analysis_accu(gt[0].permute(1, 2, 0), sr1[0].permute(1, 2, 0), 4)
    #         metrics.update(metrics)
    #
    #
    #     return sr1, metrics
    #
    # def set_metrics(self, criterion, rgb_range=1.0):
    #     self.rgb_range = rgb_range
    #     self.criterion = criterion

#
# def build(args):
#     scheduler = None
#     scale = 2
#     mode = "one"
#     g_ssim = SSIM(size_average=True)
#     loss1 = nn.L1Loss().cuda()
#     loss2 = g_ssim.cuda()
#     weight_dict = {'Loss': 1, 'ssim_loss': 0.1}
#     losses = {'Loss': loss1, 'ssim_loss':loss2}
#     criterion = SetCriterion(losses, weight_dict)
#     model = Bidinet(args).cuda()
#     num_params = 0
#     for param in Bidinet(args).parameters():
#         num_params += param.numel()
#     print('[Network %s] Total number of parameters : %.3f M' % ('Bottleneck', num_params / 1e6))
#     # summary(model, input_size=[(3, 64, 64), (31, 64, 64), (31, 16, 16)], batch_size=1)
#     model.set_metrics(criterion)
#     optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)  ## optimizer 1: Adam
#
#     return model, criterion, optimizer, scheduler