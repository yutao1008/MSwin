_base_ = ['./mswin_par_small_patch4_512x512_160k_ade20k_pretrain_224x224_1K.py']

model = dict(
    decode_head=dict(
        mode='seq',
    ))

data = dict(samples_per_gpu=10)
