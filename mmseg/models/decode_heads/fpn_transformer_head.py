import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from mmseg.ops import resize
from mmcv.cnn import ConvModule
from ..backbones.swin import SwinBlock

from ..builder import HEADS
from .decode_head import BaseDecodeHead

@HEADS.register_module()
class FpnTransformerHead(BaseDecodeHead):

    def __init__(self, num_heads=8,
                 **kwargs):
        super(FpnTransformerHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        self.lateral_convs = nn.ModuleList()
        self.fpn_swins = nn.ModuleList()
        self.l_conv4 = ConvModule(self.in_channels[-1],
                                  self.channels,
                                  1,
                                  conv_cfg=self.conv_cfg,
                                  norm_cfg=dict(type='SyncBN', requires_grad=True),
                                  act_cfg=self.act_cfg,
                                  inplace=False)
        for in_channels in self.in_channels[:-1]:
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            fpn_swin = SwinBlock(self.channels, num_heads, self.channels)
            self.fpn_swins.append(fpn_swin)


    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        # build laterals
        laterals = [
            l(inputs[i])
            for i, l in enumerate(self.lateral_convs)]
        laterals.append(self.l_conv4(inputs[-1]))
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels-1, 0, -1):
            prev_shape = laterals[i-1].shape[2:]
            laterals[i - 1] += resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build fpn outputs
        fpn_outs = []
        for i in range(used_backbone_levels-1):
            B, C, H, W = laterals[i].shape
            x = laterals[i].view(B, C, H*W).permute(0, 2, 1) #B,L,C
            #x = self.fpn_swins[i](x, (H, W)) #B,L,self.channels
            x = checkpoint(self.fpn_swins[i], x, (H, W)) 
            x = x.permute(0, 2, 1).view(B, self.channels, H, W)
            fpn_outs.append(x)
        fpn_outs.append(laterals[-1])
        # fuze multi-features
        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_out = torch.stack(fpn_outs, dim=0).sum(dim=0)
        output = self.cls_seg(fpn_out)
        return output

