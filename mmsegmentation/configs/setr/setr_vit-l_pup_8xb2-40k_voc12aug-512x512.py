_base_ = ['./setr_vit-l_pup_8xb2-20k_voc12aug-512x512.py']

# 40k training schedule
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=40000,
        by_epoch=False)
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
default_hooks = dict(checkpoint=dict(by_epoch=False, interval=4000))
