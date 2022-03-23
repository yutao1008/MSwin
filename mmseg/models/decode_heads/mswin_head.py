import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch.nn.modules.utils import _pair as to_2tuple
from mmseg.ops import resize
from mmcv.runner.base_module import BaseModule
from mmcv.cnn import ConvModule, build_norm_layer
from mmcv.cnn.bricks.transformer import FFN
from ..backbones.swin import WindowMSA, ShiftWindowMSA, SwinBlock

from ..builder import HEADS
from .decode_head import BaseDecodeHead

'''
class PatchSplit(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 up_scales=2,
                 bias=False,
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_scales = up_scales

        self.upsample_dim = in_channels//(up_scales**2)
        assert in_channels%(up_scales**2) == 0, 'the channel dimension must be up_scales^2 times'
        if norm_cfg is not None:
            self.norm = build_norm_layer(norm_cfg, self.upsample_dim)[1]
        else:
            self.norm = None
        self.expansion = nn.Linear(self.upsample_dim, out_channels, bias=bias)

    def forward(self, x):
        B, L, C = x.shape
        assert C == self.in_channels, 'wrong size of in_channel'

        x = x.view(B, L, C//self.upsample_dim, C//self.upsample_dim)
        x = x.view(B, L*C//self.upsample_dim, C//self.upsample_dim)
        x = self.norm(x) if self.norm else x
        x = self.expansion(x)
        return x
'''

class PartialWindowMSA(WindowMSA):
    """
    This class is used for cross-attention.
    """
    def __init__(self, 
                 embed_dims,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 init_cfg=None):
        super().__init__(embed_dims=embed_dims,
                         num_heads=num_heads,
                         window_size=window_size,
                         qkv_bias=qkv_bias,
                         qk_scale=qk_scale,
                         attn_drop_rate=attn_drop_rate,
                         proj_drop_rate=proj_drop_rate,
                         init_cfg=init_cfg)
        self.qkv = nn.Linear(embed_dims, embed_dims * 2, bias=qkv_bias)

    def forward(self, x, mask=None):
        """
        Args:
            x (tensor): external input features with shape of (num_windows*B, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        B, N, C = x.shape
        kv = self.qkv(x).reshape(B, N, 2, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)
        x = x.reshape(B, N, self.num_heads,
                      C // self.num_heads).permute(0, 2, 1, 3)
        q = x * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PartialShiftWindowMSA(ShiftWindowMSA):
    def __init__(self, embed_dims,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0,
                 proj_drop_rate=0,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 init_cfg=None):
        super().__init__(embed_dims=embed_dims,
                         num_heads=num_heads,
                         window_size=window_size,
                         shift_size=shift_size,
                         qkv_bias=qkv_bias,
                         qk_scale=qk_scale,
                         attn_drop_rate=attn_drop_rate,
                         proj_drop_rate=proj_drop_rate,
                         dropout_layer=dropout_layer,
                         init_cfg=init_cfg)
        self.w_msa = PartialWindowMSA(embed_dims=embed_dims,
                         num_heads=num_heads,
                         window_size=to_2tuple(window_size),
                         qkv_bias=qkv_bias,
                         qk_scale=qk_scale,
                         attn_drop_rate=attn_drop_rate,
                         proj_drop_rate=proj_drop_rate,
                         init_cfg=None)


class MultiSwinBlock(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 mode = 'seq',
                 window_sizes = [5,7,12],
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 init_cfg=None):
        super(MultiSwinBlock, self).__init__(init_cfg)
        assert mode in {'seq', 'par', 'crs'}
        self.mode = mode
        self.embed_dims = embed_dims
        self.window_sizes = window_sizes
        self.attns = nn.ModuleList()
        if mode in {'seq', 'par'}:
            for window_size in window_sizes:
                self.attns.append(ShiftWindowMSA(
                              embed_dims=embed_dims,
                              num_heads=num_heads,
                              window_size=window_size,
                              shift_size=0,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop_rate=attn_drop_rate,
                              proj_drop_rate=drop_rate,
                              dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                              init_cfg=init_cfg))
                self.attns.append(ShiftWindowMSA(
                              embed_dims=embed_dims,
                              num_heads=num_heads,
                              window_size=window_size,
                              shift_size=window_size // 2,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop_rate=attn_drop_rate,
                              proj_drop_rate=drop_rate,
                              dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                              init_cfg=init_cfg))
        else: # cross-attention case
            for window_size in window_sizes:
                self.attns.append(PartialShiftWindowMSA(
                              embed_dims=embed_dims,
                              num_heads=num_heads,
                              window_size=window_size,
                              shift_size=0,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop_rate=attn_drop_rate,
                              proj_drop_rate=drop_rate,
                              dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                              init_cfg=init_cfg))
                self.attns.append(PartialShiftWindowMSA(
                              embed_dims=embed_dims,
                              num_heads=num_heads,
                              window_size=window_size,
                              shift_size=window_size // 2,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop_rate=attn_drop_rate,
                              proj_drop_rate=drop_rate,
                              dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                              init_cfg=init_cfg))
        if mode=='par':
            self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
            self.reduce = nn.Linear(embed_dims*2*len(window_sizes), embed_dims, bias=False)
            self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
            self.ffn = FFN(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                num_fcs=2,
                ffn_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                act_cfg=act_cfg,
                add_identity=True,
                init_cfg=None)
        else:
            self.norm1s = nn.ModuleList([build_norm_layer(norm_cfg, embed_dims)[1] 
                                         for _ in range(2*len(window_sizes))])
            self.norm2s = nn.ModuleList([build_norm_layer(norm_cfg, embed_dims)[1] 
                                         for _ in range(2*len(window_sizes))])
            self.ffns = nn.ModuleList([FFN(embed_dims=embed_dims,
                                           feedforward_channels=feedforward_channels,
                                           num_fcs=2, ffn_drop=drop_rate,
                                           dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
                                           act_cfg=act_cfg, add_identity=True,
                                           init_cfg=None) for _ in range(2*len(window_sizes))])
        
    def par_forward(self, x, hw_shape):
        identity = x
        x = self.norm1(x)
        #xs = [attn(x, hw_shape)+identity for attn in self.attns]
        xs = [checkpoint(attn, x, hw_shape)+identity for attn in self.attns]
        x = torch.cat(xs, dim=2)
        x = self.reduce(x)
        identity = x
        x = self.norm2(x)
        x = self.ffn(x, identity=identity)
        return x

    def seq_forward(self, x, hw_shape):
        for i in range(len(self.window_sizes)*2):
            identity = x
            x = self.norm1s[i](x)
            #x = self.attns[i](x, hw_shape) + identity
            x = checkpoint(self.attns[i], x, hw_shape) + identity
            identity = x
            x = self.norm2s[i](x)
            x = self.ffns[i](x, identity=identity)
        return x

    def crs_forward(self, x, hw_shape):
        inputs = [x]
        for i in range(len(self.window_sizes)*2):
            x = torch.stack(inputs, dim=0).sum(dim=0)
            identity = x
            x = self.norm1s[i](x)
            #x = self.attns[i](x, hw_shape) + identity
            x = checkpoint(self.attns[i], x, hw_shape) + identity
            identity = x
            x = self.norm2s[i](x)
            x = self.ffns[i](x, identity=identity)
            inputs.append(x)
        return x

    def forward(self, x, hw_shape):
        if self.mode == 'seq':
            return self.seq_forward(x, hw_shape)
        elif self.mode == 'par':
            return self.par_forward(x, hw_shape)
        else:
            return self.crs_forward(x, hw_shape)



@HEADS.register_module()
class MultiSwinHead(BaseDecodeHead):

    def __init__(self, num_heads=8,
                 window_sizes = [5,7,12],
                 mode = 'seq',
                 **kwargs):
        super(MultiSwinHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        self.mode = mode
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

        self.multi_swin = MultiSwinBlock(self.channels, num_heads,
                                         feedforward_channels=self.channels,
                                         mode = mode,
                                         window_sizes=window_sizes)

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
        # multi-swin head
        B, C, H, W = fpn_out.shape
        fpn_out = fpn_out.view(B, C, H*W).permute(0, 2, 1)
        output = self.multi_swin(fpn_out, (H, W))
        output = output.permute(0, 2, 1).view(B, self.channels, H, W)
        output = self.cls_seg(output)
        return output

