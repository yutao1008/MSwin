_base_ = ['./mswin_par_small_patch4_480x480_40k_cocostuff_pretrain_224x224_1K.py']

model = dict(
    decode_head=dict(
        mode='crs',
    ))