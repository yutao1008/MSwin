_base_ = ['./mswin_par_base_patch4_512x512_80k_voc2012_pretrain_384x384_22K.py']

model = dict(
    decode_head=dict(
        mode='seq',
    ))