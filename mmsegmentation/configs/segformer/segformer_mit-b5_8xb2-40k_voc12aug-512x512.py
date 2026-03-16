_base_ = ['./segformer_mit-b5_8xb2-20k_voc12aug-512x512.py']

# 40k training schedule
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=40000,
        by_epoch=False)
]

train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=4000)
default_hooks = dict(checkpoint=dict(by_epoch=False, interval=4000))
