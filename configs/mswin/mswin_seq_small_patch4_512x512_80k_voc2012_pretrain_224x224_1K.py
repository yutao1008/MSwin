_base_=['./mswin_par_small_patch4_512x512_80k_voc2012_pretrain_224x224_1K.py']

model = dict(
    decode_head=dict(
        mode='seq',
    ))

data = dict(samples_per_gpu=10)
