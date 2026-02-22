import torch
import torch.nn as nn
# import torchvision.transforms as transforms
# import dataset, transform
from PIL import Image
import numpy as np
import cv2
import argparse
import os
import sys
import evaluate
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import random
# ddcat 모델들 import
from model.pspnet import PSPNet, DeepLabV3, PSPNet_DDCAT, DeepLabV3_DDCAT
import logging
# 상위 디렉토리를 Python path에 추가
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(_parent_dir)
# mmsegmentation 디렉토리도 추가 (dataset.py가 여기에 있음)
_mmseg_dir = os.path.join(_parent_dir, 'mmsegmentation')
if os.path.isdir(_mmseg_dir):
    sys.path.append(_mmseg_dir)
# dataset 클래스들 import
from dataset import CitySet, ADESet, VOCSet
cv2.ocl.setUseOpenCL(False)
import torch, torch.nn.functional as F, numpy as np, cv2, math
from contextlib import nullcontext
# 랜덤성 제거를 위한 시드 설정
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False





def load_model(config):
        """모델 타입에 따라 모델 초기화"""
        if config["model"] == 'deeplabv3_sat':
            return DeepLabV3(layers=config["layers"], classes=config["num_class"], zoom_factor=config["zoom_factor"], pretrained=False)
        elif config["model"] == 'deeplabv3_ddcat':
            return DeepLabV3_DDCAT(layers=config["layers"], classes=config["num_class"], zoom_factor=config["zoom_factor"], pretrained=False)
        elif config["model"] == 'pspnet_sat':
            return PSPNet(layers=config["layers"], classes=config["num_class"], zoom_factor=config["zoom_factor"], pretrained=False)
        elif config["model"] == 'pspnet_ddcat':
            return PSPNet_DDCAT(layers=config["layers"], classes=config["num_class"], zoom_factor=config["zoom_factor"], pretrained=False)
        else:
            raise ValueError(f"지원하지 않는 모델 타입: {config['model']}")
        
    








# ────────── 공통 세팅 ──────────
torch.backends.cudnn.benchmark = True      # 입력 해상도가 일정할 때 필수


@torch.no_grad()
def fast_net_process(model, img_np, mean, std=None, do_flip=True, amp=True, config=None):
    """
    • numpy → GPU tensor 한 번만
    • mean / std 브로드캐스팅으로 루프 제거
    • AMP(Auto Mixed Precision) optional
    """
    # (H,W,C) → (1,C,H,W) & float32
    inp = torch.from_numpy(img_np).permute(2,0,1).unsqueeze(0).to(config["device"], dtype=torch.float32)

    if std is None:
        inp -= torch.tensor(mean, device=config["device"])[:, None, None]
    else:
        inp = (inp - torch.tensor(mean, device=config["device"])[:, None, None]) / \
              torch.tensor(std,  device=config["device"])[:, None, None]

    if do_flip:                       # test-time augmentation
        inp = torch.cat([inp, inp.flip(3)], dim=0)

    autocast_ctx = torch.cuda.amp.autocast() if amp else nullcontext()
    with autocast_ctx:
        out = model(inp)              # (B,C,H',W')

    if out.shape[-2:] != inp.shape[-2:]:
        out = F.interpolate(out, inp.shape[-2:], mode='bilinear', align_corners=False)

    out = F.softmax(out, dim=1)
    if do_flip:
        out = (out[0] + out[1].flip(2)) * 0.5
    else:
        out = out[0]

    return out.permute(1,2,0).cpu().float().numpy()   # (H,W,C)

def fast_scale_process(model, image, classes,
                       crop_h, crop_w, h, w, mean, std=None,
                       stride_rate=2/3, batch_size=8, amp=True, config=None):

    ori_h, ori_w, _ = image.shape
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    ph1, pw1 = pad_h // 2, pad_w // 2
    if pad_h or pad_w:
        image = cv2.copyMakeBorder(image, ph1, pad_h - ph1, pw1, pad_w - pw1,
                                   cv2.BORDER_CONSTANT, value=mean)

    new_h, new_w = image.shape[:2]
    sh, sw = math.ceil(crop_h * stride_rate), math.ceil(crop_w * stride_rate)
    gh, gw = math.ceil((new_h - crop_h) / sh) + 1, math.ceil((new_w - crop_w) / sw) + 1

    # ───── 모든 crop 좌표 미리 계산
    coords = []
    for ih in range(gh):
        for iw in range(gw):
            s_h = max(min(ih * sh, new_h - crop_h), 0)
            s_w = max(min(iw * sw, new_w - crop_w), 0)
            coords.append((s_h, s_w, s_h + crop_h, s_w + crop_w))

    pred_map = np.zeros((new_h, new_w, classes), dtype=np.float32)
    cnt_map  = np.zeros((new_h, new_w), dtype=np.float32)

    # ───── 배치 단위 추론
    model.eval()
    for i in range(0, len(coords), batch_size):
        batch_imgs = [image[y1:y2, x1:x2] for (y1,x1,y2,x2) in coords[i:i+batch_size]]
        batch_outs = [fast_net_process(model, im, mean, std, amp=amp, config=config) for im in batch_imgs]

        for (y1,x1,y2,x2), out in zip(coords[i:i+batch_size], batch_outs):
            pred_map[y1:y2, x1:x2] += out
            cnt_map [y1:y2, x1:x2] += 1

    pred_map /= cnt_map[..., None]
    pred_map  = pred_map[ph1:ph1+ori_h, pw1:pw1+ori_w]
    pred_map  = cv2.resize(pred_map, (w, h), interpolation=cv2.INTER_LINEAR)
    return pred_map


def model_predict(model, image, config):
    image = image[:,:,::-1]             # bgr -> rgb
    h, w, _ = image.shape

    if config["dataset"] == "cityscapes":
        image = cv2.resize(image, (1024, 512), interpolation=cv2.INTER_LINEAR)

    
    


    confidence = np.zeros((h, w, config["num_class"]), dtype=float)
    for scale in config["scales"]:
        long_size = round(scale * config["base_size"])
        new_h = long_size
        new_w = long_size
        if h > w:
            new_w = round(long_size/float(h)*w)
        else:
            new_h = round(long_size/float(w)*h)

        image_scale = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        confidence += fast_scale_process(model, image_scale, config["num_class"], config["crop_h"], config["crop_w"], h, w, config["mean"], config["std"], config=config)


    confidence = torch.from_numpy(confidence / len(config["scales"])).to(config["device"])
    confidence = confidence.permute(2, 0, 1)
    prediction = torch.argmax(confidence, dim=0)
    return confidence, prediction

def main(config):
    # 랜덤성 제거
    set_seed(42)
    
    model = load_model(config)
    checkpoint = torch.load(config["model_path"])
    # 모델을 cuda로 이동 후 DataParallel 적용
    model = model.cuda()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['state_dict'])
    
    
    model.eval()
    
    # CUDA 벤치마크 비활성화 (결정적 동작을 위해)
    cudnn.benchmark = False


    mean_iou = evaluate.load("mean_iou", "segmentation")

    if config["dataset"] == "cityscapes":
        dataset = CitySet("../datasets/cityscapes", use_gt=False)
        
    elif config["dataset"] == "VOC2012":
        dataset = VOCSet(config)
        adv_files = dataset.adv_files

    # Normalization parameters
    value_scale = 255

    mean_rgb = [0.485, 0.456, 0.406]  # [R, G, B]
    mean_rgb = [item * value_scale for item in mean_rgb]
    std_rgb = [0.229, 0.224, 0.225]   # [R, G, B]
    std_rgb = [item * value_scale for item in std_rgb]
    


    pred_list = []
    gt_list = []

    for i in range(len(dataset)):
        image, _, gt = dataset[i]
        confidence, prediction = model_predict(model, image, config)
        # confidence (H,W) according to prediction
        # prediction (H,W)
        pred_list.append(prediction)
        gt_list.append(gt)
    
    iou = mean_iou.compute(
        predictions=pred_list,
        references=gt_list,
        num_labels=config["num_class"],
        ignore_index=255,
    )
    print(f"Mean IoU: {iou}")




if __name__ == "__main__":
    config = {
        "model": "pspnet_sat",
        "model_path": "/workspace/ckpt/pretrained_model/pretrain/voc2012/pspnet/sat/train_epoch_50.pth",
        "device": "cuda",
        "dataset": "VOC2012",
        "layers": 50,
        "num_class": 19,
        "zoom_factor": 8,
        "base_size": 1024,
        "crop_h": 449,
        "crop_w": 449,
        "scales": [1.0],
        "mean":[255*0.485, 255*0.456, 255*0.406],   # [R, G, B]
        "std":[255*0.229, 255*0.224, 255*0.225]     # [R, G, B]
    }
    main(config)