_base_ = ['./setr_vit-l_pup_8xb1-20k_voc12aug-512x512.py']

# num_gpus: 8 -> batch_size: 16
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
