_base_ = ['./mswin_par_base_patch4_480x480_40k_cocostuff_pretrain_384x384_22K.py']

model = dict(
    decode_head=dict(
        mode='seq',
    ))