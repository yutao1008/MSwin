_base_ = ['./mswin_par_base_patch4_512x1024_80k_cityscapes_pretrain_384x384_22K.py']

model = dict(
    decode_head=dict(
        mode='crs',
    ))