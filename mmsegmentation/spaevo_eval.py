"""
SpaEvO Attack Evaluation for MMSegmentation
Adapted from PointWise evaluation script.
"""

import os
import sys
import torch
import json
from tqdm import tqdm
import datetime
import importlib
import numpy as np
from PIL import Image
import setproctitle
import argparse

# 상위 디렉토리와 현재 디렉토리를 sys.path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)
# Robust-Semantic-Segmentation 경로 추가
sys.path.insert(0, os.path.join(parent_dir, 'Robust-Semantic-Segmentation'))

from mmseg.apis import init_model, inference_model
from dataset import CitySet, ADESet, VOCSet
from spaevo_attack import SpaEvoAttack
from utils_se import salt_pepper_noise, rand_img_upscale, l0

from function import *
from evaluation import *
from utils import save_experiment_results

# PointWise에서 사용한 gen_starting_point_seg 재사용 (로컬 정의)

def gen_starting_point_seg(attack, oimg, original_pred_labels, seed=None, init_mode='salt_pepper', dataset_name='voc2012'):
    """
    세그멘테이션용 Starting Point 생성 함수.
    utils_se.py의 salt_pepper_noise와 rand_img_upscale을 사용합니다.
    
    원본 gen_starting_point와 동일한 로직:
    - dataset_name에 따라 scale 설정
    - while True 루프로 scale 단위 반복
    - D(distance history) 반환
    """
    if len(oimg.shape) == 3:
        oimg = oimg.unsqueeze(0)
    
    nquery = 0
    i = 0
    rndtype = 'normal'  # 원본과 동일하게 'normal' 사용
    
    # 데이터셋에 따른 scale 설정 (이미지 크기 기반)
    # cityscapes (2048x1024): 고해상도 - scale 64까지 가능
    # imagenet/voc2012/ade20k (~224-500px): 중해상도 - scale 32까지
    # cifar10/cifar100 (32x32): 저해상도 - scale 16까지
    if dataset_name == 'cityscapes':
        scales = [1, 2, 4, 8, 16, 32, 64]  # 고해상도 (2048x1024)
    elif dataset_name in ['imagenet', 'voc2012', 'VOC2012', 'ade20k']:
        scales = [1, 2, 4, 8, 16, 32]      # 중해상도 (~224-500px)
    else:  # cifar10, cifar100
        scales = [1, 2, 4, 8, 16]          # 저해상도 (32x32)
    
    if init_mode == 'salt_pepper':
        while True:
            current_seed = seed + i if seed is not None else None
            timg = salt_pepper_noise(oimg / 255.0, rndtype, scales[i], current_seed)
            timg = timg * 255.0
            
            nquery += 1
            is_adv, changed_ratio = attack.check_adv_status(timg, original_pred_labels)
            
            if is_adv:
                D = torch.ones(nquery, dtype=int).cuda() * l0(oimg, timg)
                print(f'Starting point found: mode={init_mode}, scale={scales[i]}, changed_ratio={changed_ratio:.4f}')
                return timg, nquery, D
            elif i + 1 < len(scales):
                i += 1
            else:
                # Fallback: 모든 scale 실패 → 전체 랜덤 이미지 사용
                print(f'Fallback: using full random image as starting point')
                timg = torch.rand_like(oimg).cuda() * 255.0
                nquery += 1
                D = torch.ones(nquery, dtype=int).cuda() * l0(oimg, timg)
                return timg, nquery, D
    
    elif init_mode == 'gauss_rand' or init_mode == 'random':
        while True:
            current_seed = seed + i if seed is not None else None
            timg = rand_img_upscale(oimg / 255.0, rndtype, scales[i], current_seed)
            timg = timg * 255.0
            
            nquery += 1
            is_adv, changed_ratio = attack.check_adv_status(timg, original_pred_labels)
            
            if is_adv:
                D = torch.ones(nquery, dtype=int).cuda() * l0(oimg, timg)
                print(f'Starting point found: mode={init_mode}, scale={scales[i]}, changed_ratio={changed_ratio:.4f}')
                return timg, nquery, D
            elif i + 1 < len(scales):
                i += 1
            else:
                # Fallback: 모든 scale에서 adversarial starting point를 찾지 못함
                print(f'Warning: No adversarial starting point found. Using last scale={scales[i]} as fallback.')
                D = torch.ones(nquery, dtype=int).cuda() * l0(oimg, timg)
                return timg, nquery, D


def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config


def init_model_for_process(model_configs, dataset, model_name, device):
    if model_name == "setr":
        model = init_model(model_configs["config"], None, 'cuda')
        checkpoint = torch.load(model_configs["checkpoint"], map_location='cuda', weights_only=False)
        model.backbone.patch_embed.projection.bias = torch.nn.Parameter(
            torch.zeros(checkpoint["state_dict"]["backbone.patch_embed.projection.weight"].shape[0], device='cuda')
        )
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model = init_model(model_configs["config"], None, device)
        checkpoint = torch.load(model_configs["checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])

    del checkpoint
    torch.cuda.empty_cache()
    return model


def process_single_image(args):
    (img_bgr, filename, gt, model_configs, config, base_dir, idx, total_images) = args
    
    model = init_model_for_process(model_configs, config["dataset"], config["model"], config["device"])
    setproctitle.setproctitle(f"({idx+1}/{total_images})_SpaEvO_{config['dataset']}_{config['model']}_{config['success_threshold']}")

    # Prepare inputs
    img_tensor_bgr = torch.from_numpy(img_bgr.copy()).unsqueeze(0).permute(0, 3, 1, 2).float().to(config["device"])
    
    # Original prediction
    ori_result = inference_model(model, img_bgr.copy())
    ori_pred = ori_result.pred_sem_seg.data.squeeze().cpu().numpy()
    original_pred_labels = ori_result.pred_sem_seg.data.squeeze().cuda()

    # n_pix 자동 계산 (0이거나 지정되지 않은 경우 0.004 × H × W)
    H, W = img_bgr.shape[:2]
    n_pix = config.get("n_pix", 0)
    if n_pix == 0 or n_pix is None:
        n_pix = int(0.004 * H * W)
        print(f"[Auto n_pix] Image size: {H}×{W}, n_pix = 0.004 × {H} × {W} = {n_pix}")
    
    # Initialize SpaEvO Attack
    attack = SpaEvoAttack(
        model=model,
        n_pix=n_pix,
        pop_size=config.get("pop_size", 10),
        cr=config.get("cr", 0.9),
        mu=config.get("mu", 0.01),
        seed=config.get("seed", 0),
        success_threshold=config.get("success_threshold", 0.01),
        verbose=config.get("verbose", False),
        device=config["device"],
        is_mmseg=True
    )
    attack.set_ignore_index(config["dataset"], include_bg=config.get("include_bg", False))  # 데이터셋별 ignore index 설정

    # Generate Starting Point (timg)
    print(f"\n[{idx+1}/{total_images}] {filename}: Generating starting point...")
    timg, init_nqry, init_D = gen_starting_point_seg(
        attack, img_tensor_bgr, original_pred_labels, 
        seed=config.get("seed", 0), 
        init_mode=config.get("init_mode", "salt_pepper"),
        dataset_name=config["dataset"]
    )

    print(f"[{idx+1}/{total_images}] {filename}: Running SpaEvO attack...")
    
    # Run Attack
    # SpaEvO modifies timg to look like oimg while maintaining adversarial status
    snapshot_interval = 200
    attack_max_query = config["max_query"] - init_nqry
    adv_np, nquery, D, snapshots = attack.evo_perturb(
        img_tensor_bgr.squeeze(0), timg.squeeze(0), original_pred_labels,
        max_query=attack_max_query,
        snapshot_interval=snapshot_interval,
        query_offset=init_nqry
    )
    
    total_query = init_nqry + nquery
    
    # Save results
    current_img_save_dir = os.path.join(base_dir, os.path.splitext(os.path.basename(filename))[0])
    os.makedirs(current_img_save_dir, exist_ok=True)
    
    # Save final adv image
    adv_img_bgr = torch.from_numpy(adv_np.reshape(img_tensor_bgr.squeeze(0).shape)).unsqueeze(0).float()
    adv_img_np = adv_img_bgr.squeeze(0).permute(1, 2, 0).numpy().astype(np.uint8)
    
    Image.fromarray(img_bgr[:, :, ::-1]).save(os.path.join(current_img_save_dir, "original.png"))
    visualize_segmentation(img_bgr, gt,
                        save_path=os.path.join(current_img_save_dir, "gt.png"),
                        alpha=1.0, dataset=config["dataset"])
    

    
    # Calculate metrics
    l0_norm = calculate_l0_norm(img_bgr, adv_img_np)
    pixel_ratio = calculate_pixel_ratio(img_bgr, adv_img_np)
    
    adv_result = inference_model(model, adv_img_np)
    adv_pred = adv_result.pred_sem_seg.data.squeeze().cpu().numpy()
    impact = calculate_impact(img_bgr, adv_img_np, ori_pred, adv_pred)
    
    # 최종 success_ratio 계산 (배경/ignore 제외, 원본 예측 대비 변경된 픽셀 비율)
    if config["dataset"].lower() in ["cityscapes", "ade20k"]:
        ignore_index = 255
    elif config["dataset"] == "VOC2012":
        ignore_index = None if config.get("include_bg", False) else 0
    else:
        ignore_index = None
    if ignore_index is not None:
        foreground_mask = ori_pred != ignore_index
    else:
        foreground_mask = np.ones_like(ori_pred, dtype=bool)
    if foreground_mask.sum() > 0:
        success_ratio = ((adv_pred != ori_pred) & foreground_mask).sum() / foreground_mask.sum()
    else:
        success_ratio = 0.0
    
    print(f"[{idx+1}/{total_images}] {filename}: Completed. L0={l0_norm}, Ratio={pixel_ratio:.4f}, Impact={impact:.4f}, Success Ratio={success_ratio:.4f}")
    
    # Visualizations
    visualize_segmentation(img_bgr, ori_pred,
                        save_path=os.path.join(current_img_save_dir, "ori_seg.png"),
                        alpha=0.5, dataset=config["dataset"])
    visualize_segmentation(img_bgr, ori_pred,
                        save_path=os.path.join(current_img_save_dir, "ori_seg_only.png"),
                        alpha=1.0, dataset=config["dataset"])

    # 각 이미지별 결과를 JSON으로 저장
    # foreground 픽셀 수 및 공격 실패 픽셀 수 계산
    foreground_pixels = int(foreground_mask.sum())
    unsuccess_pixel = int(((adv_pred == ori_pred) & foreground_mask).sum())
    success_pixel = int(((adv_pred != ori_pred) & foreground_mask).sum())
    
    # 쿼리별 결과 생성 (스냅샷 활용)
    query_history = []
    D_numpy = D.cpu().numpy() if isinstance(D, torch.Tensor) else D
    
    # 정수형 키만 추출하여 정렬
    snapshot_queries = sorted([k for k in snapshots.keys() if isinstance(k, int)])
    
    for query_num in snapshot_queries:
        snapshot_img = snapshots[query_num]
        snapshot_np = np.transpose(snapshot_img, (1, 2, 0)).astype(np.uint8)  # (C,H,W) -> (H,W,C)
        
        # 해당 쿼리 시점의 메트릭 계산
        if query_num == 0:
            # 0번 쿼리는 초기 상태, L0 직접 계산
            diff = np.abs(snapshot_np.astype(int) - img_bgr.astype(int))
            l0_at_q = int(np.sum(np.sum(diff, axis=2) > 0))
        else:
            l0_at_q = int(D_numpy[query_num-1]) if query_num-1 < len(D_numpy) else int(D_numpy[-1])
            
        pixel_ratio_at_q = calculate_pixel_ratio(img_bgr, snapshot_np)
        
        # 예측 및 success_ratio 계산
        snapshot_result = inference_model(model, snapshot_np)
        snapshot_pred = snapshot_result.pred_sem_seg.data.squeeze().cpu().numpy()
        
        success_pixel_at_q = int(((snapshot_pred != ori_pred) & foreground_mask).sum())
        unsuccess_pixel_at_q = int(((snapshot_pred == ori_pred) & foreground_mask).sum())
        impact_at_q = calculate_impact(img_bgr, snapshot_np, ori_pred, snapshot_pred)
        success_ratio_at_q = success_pixel_at_q / foreground_pixels if foreground_pixels > 0 else 0.0

        # 이미지 저장 (0 또는 200 단위)
        if query_num == 0 or query_num % 200 == 0:
            save_q_dir = os.path.join(current_img_save_dir, f"{query_num}query")
            os.makedirs(save_q_dir, exist_ok=True)
            # snapshot_np is (H, W, C) in BGR, convert to RGB for PIL
            Image.fromarray(snapshot_np[:, :, ::-1]).save(os.path.join(save_q_dir, "adv.png"))
            
            # Delta 이미지 저장
            delta_img = np.abs(img_bgr.astype(np.int16) - snapshot_np.astype(np.int16)).astype(np.uint8)
            Image.fromarray(delta_img).save(os.path.join(save_q_dir, "delta.png"))
            
            # Adversarial 세그멘테이션 저장
            visualize_segmentation(snapshot_np, snapshot_pred, 
                save_path=os.path.join(save_q_dir, "adv_seg.png"),
                alpha=0.5, dataset=config["dataset"])
            visualize_segmentation(snapshot_np, snapshot_pred, 
                save_path=os.path.join(save_q_dir, "adv_seg_only.png"),
                alpha=1.0, dataset=config["dataset"])
        
        query_history.append({
            'query': query_num,
            'l0': l0_at_q,
            'pixel_ratio': float(pixel_ratio_at_q),
            'impact': float(impact_at_q),
            'success_ratio': float(success_ratio_at_q),
            'success_pixel': success_pixel_at_q,
            'unsuccess_pixel': unsuccess_pixel_at_q
        })

    # Final snapshot 처리
    if 'final' in snapshots:
        final_query = snapshots.get('final_query', nquery)

        if final_query not in snapshot_queries:
            snapshot_img = snapshots['final']
            snapshot_np = np.transpose(snapshot_img, (1, 2, 0)).astype(np.uint8)
            
            l0_at_q = int(D_numpy[final_query-1]) if final_query > 0 and final_query-1 < len(D_numpy) else int(D_numpy[-1] if len(D_numpy) > 0 else 0)
            if final_query == 0:
                 diff = np.abs(snapshot_np.astype(int) - img_bgr.astype(int))
                 l0_at_q = int(np.sum(np.sum(diff, axis=2) > 0))

            pixel_ratio_at_q = calculate_pixel_ratio(img_bgr, snapshot_np)
            
            # 예측 및 success_ratio 계산
            snapshot_result = inference_model(model, snapshot_np)
            snapshot_pred = snapshot_result.pred_sem_seg.data.squeeze().cpu().numpy()
            
            success_pixel_at_q = int(((snapshot_pred != ori_pred) & foreground_mask).sum())
            unsuccess_pixel_at_q = int(((snapshot_pred == ori_pred) & foreground_mask).sum())
            impact_at_q = calculate_impact(img_bgr, snapshot_np, ori_pred, snapshot_pred)
            success_ratio_at_q = success_pixel_at_q / foreground_pixels if foreground_pixels > 0 else 0.0

            # 이미지 저장 (Final은 무조건 저장)
            final_save_query = init_nqry + final_query
            save_q_dir = os.path.join(current_img_save_dir, f"{final_save_query}query")
            os.makedirs(save_q_dir, exist_ok=True)
            Image.fromarray(snapshot_np[:, :, ::-1]).save(os.path.join(save_q_dir, "adv.png"))
            
            # Delta 이미지 저장
            delta_img = np.abs(img_bgr.astype(np.int16) - snapshot_np.astype(np.int16)).astype(np.uint8)
            Image.fromarray(delta_img).save(os.path.join(save_q_dir, "delta.png"))
            
            # Adversarial 세그멘테이션 저장
            visualize_segmentation(snapshot_np, snapshot_pred, 
                save_path=os.path.join(save_q_dir, "adv_seg.png"),
                alpha=0.5, dataset=config["dataset"])
            visualize_segmentation(snapshot_np, snapshot_pred, 
                save_path=os.path.join(save_q_dir, "adv_seg_only.png"),
                alpha=1.0, dataset=config["dataset"])
            
            query_history.append({
                'query': final_save_query,
                'l0': l0_at_q,
                'pixel_ratio': float(pixel_ratio_at_q),
                'impact': float(impact_at_q),
                'success_ratio': float(success_ratio_at_q),
                'success_pixel': success_pixel_at_q,
                'unsuccess_pixel': unsuccess_pixel_at_q
            })
    
    # max_query 이하에서 공격 성공한 가장 큰 쿼리 찾기
    max_query_limit = config['max_query']
    attack_success_query = None
    attack_success = False
    
    # 1. Snapshots에서 정확한 값 확인 (spaevo_attack.py 수정됨)
    if 'last_success_query' in snapshots and snapshots['last_success_query'] is not None:
        last_succ = snapshots['last_success_query']
        if last_succ <= max_query_limit and last_succ > 0:
            attack_success = True
            attack_success_query = last_succ + init_nqry
    image_result = {
        'filename': filename,
        'total_query': total_query,
        'l0': int(l0_norm),
        'pixel_ratio': float(pixel_ratio),
        'impact': float(impact),
        'success_ratio': float(success_ratio),
        'foreground_pixels': foreground_pixels,
        'success_pixel': success_pixel,
        'unsuccess_pixel': unsuccess_pixel,
        'max_query': config['max_query'],
        'pop_size': config.get('pop_size', 10),
        'n_pix': config.get('n_pix', 196),
        'success_threshold': config['success_threshold'],
        'attack_success': attack_success,
        'attack_success_query': attack_success_query,
        'query_history': query_history
    }
    with open(os.path.join(current_img_save_dir, 'result.json'), 'w') as f:
        json.dump(image_result, f, indent=2)

    del model
    del attack
    torch.cuda.empty_cache()

    return {
        'img_bgr': img_bgr,
        'gt': gt,
        'adv_img': adv_img_np,
        'l0': l0_norm,
        'ratio': pixel_ratio,
        'impact': impact,
        'queries': total_query,
        'distance_history': D,
        'success_ratio': success_ratio,
        'attack_success': attack_success,
        'attack_success_query': attack_success_query,
        'query_history': query_history,
        'snapshots': snapshots
    }

def main(config):
    # Model configs (same as pw_eval.py)
    model_configs = {
        "cityscapes": {
            "mask2former": {
                "config": 'configs/mask2former/mask2former_swin-b-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024.py',
                "checkpoint": '../ckpt/mask2former_swin-b-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024_20221203_045030-9a86a225.pth'
            },
            "segformer": {
                "config": 'configs/segformer/segformer_mit-b5_8xb1-160k_cityscapes-1024x1024.py',
                "checkpoint": '../ckpt/segformer_mit-b5_8x1_1024x1024_160k_cityscapes_20211206_072934-87a052ec.pth'
            },
            "deeplabv3": {
                "config": 'configs/deeplabv3/deeplabv3_r101-d8_4xb2-80k_cityscapes-512x1024.py',
                "checkpoint": '../ckpt/deeplabv3_r101-d8_512x1024_80k_cityscapes_20200606_113503-9e428899.pth'
            },
            "pspnet": {
                "config": 'configs/pspnet/pspnet_r101-d8_4xb2-80k_cityscapes-512x1024.py',
                "checkpoint": '../ckpt/pspnet_r101-d8_512x1024_80k_cityscapes_20200606_112211-e1e1100f.pth'
            },
            "setr": {
                "config": 'configs/setr/setr_vit-l_pup_8xb1-80k_cityscapes-768x768.py',
                "checkpoint": '../ckpt/setr_pup_vit-large_8x1_768x768_80k_cityscapes_20211122_155115-f6f37b8f.pth'
            }
        },
        "ade20k": {
            "mask2former": {
                "config": 'configs/mask2former/mask2former_swin-b-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py',
                "checkpoint": '../ckpt/mask2former_swin-b-in22k-384x384-pre_8xb2-160k_ade20k-640x640_20221203_235230-7ec0f569.pth'
            },
            "segformer": {
                "config": 'configs/segformer/segformer_mit-b5_8xb2-160k_ade20k-640x640.py',
                "checkpoint": '../ckpt/segformer_mit-b5_640x640_160k_ade20k_20210801_121243-41d2845b.pth'
            },
            "deeplabv3": {
                "config": 'configs/deeplabv3/deeplabv3_r101-d8_4xb4-80k_ade20k-512x512.py',
                "checkpoint": '../ckpt/deeplabv3_r101-d8_512x512_160k_ade20k_20200615_105816-b1f72b3b.pth'
            },
            "pspnet": {
                "config": 'configs/pspnet/pspnet_r101-d8_4xb4-160k_ade20k-512x512.py',
                "checkpoint": '../ckpt/pspnet_r101-d8_512x512_160k_ade20k_20200615_100650-967c316f.pth'
            },
            "setr": {
                "config": 'configs/setr/setr_vit-l_pup_8xb2-160k_ade20k-512x512.py',
                "checkpoint": '../ckpt/setr_pup_512x512_160k_b16_ade20k_20210619_191343-7e0ce826.pth'
            }
        },
        "VOC2012": {
            "deeplabv3": {
                "config": 'configs/deeplabv3/deeplabv3_r101-d8_4xb4-20k_voc12aug-512x512.py',
                "checkpoint": '../ckpt/deeplabv3_r101-d8_512x512_20k_voc12aug_20200617_010932-8d13832f.pth'
            },
            "pspnet": {
                "config": 'configs/pspnet/pspnet_r101-d8_4xb4-40k_voc12aug-512x512.py',
                "checkpoint": '../ckpt/pspnet_r101-d8_512x512_20k_voc12aug_20200617_102003-4aef3c9a.pth'
            }
        }
    }

    device = config["device"]
    model_cfg = model_configs[config["dataset"]][config["model"]]

    # Load dataset
    data_dir = config["data_dir"]
    if not os.path.isabs(data_dir):
        data_dir = os.path.abspath(os.path.join(current_dir, data_dir))

    if config["dataset"] == "cityscapes":
        dataset = CitySet(dataset_dir=data_dir)
    elif config["dataset"] == "ade20k":
        dataset = ADESet(dataset_dir=data_dir)
    elif config["dataset"] == "VOC2012":
        dataset = VOCSet(dataset_dir=data_dir)
    else:
        raise ValueError(f"Unsupported dataset: {config['dataset']}")

    num_images = config["num_images"]
    dataset.images = dataset.images[:min(len(dataset.images), num_images)]
    dataset.filenames = dataset.filenames[:min(len(dataset.filenames), num_images)]
    dataset.gt_images = dataset.gt_images[:min(len(dataset.gt_images), num_images)]

    # Result directory
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = os.path.join(config["base_dir"], current_time)
    os.makedirs(base_dir, exist_ok=True)

    process_args = []
    for idx, (img_bgr, filename, gt) in enumerate(zip(dataset.images, dataset.filenames, dataset.gt_images)):
        process_args.append((img_bgr, filename, gt, model_cfg, config, base_dir, idx, len(dataset.images)))

    # Init model for mIoU calc
    model = init_model_for_process(model_cfg, config["dataset"], config["model"], device)

    # Run
    print(f"\nProcessing {len(process_args)} images with SpaEvO Attack...")
    
    img_list = []
    gt_list = []
    
    snapshot_interval = config.get("snapshot_interval", 200)
    levels = config["max_query"] // snapshot_interval + 1
    adv_img_lists = [[] for _ in range(levels)]
    final_adv_img_list = []
    l0_metrics = []
    ratio_metrics = []
    impact_metrics = []
    query_metrics = []
    success_ratio_metrics = []
    all_query_histories = []
    
    # 공격 성공 쿼리 추적
    attack_success_list = []  # 각 이미지별 공격 성공 여부
    attack_success_queries = []  # 공격 성공한 이미지들의 성공 쿼리

    for args in tqdm(process_args, desc="SpaEvO Attack"):
        result = process_single_image(args)
        
        img_list.append(result['img_bgr'])
        gt_list.append(result['gt'])
        
        snapshots = result['snapshots']
        for i in range(levels):
            q = i * snapshot_interval
            # Use snapshot if available, else final (attack ended)
            if q in snapshots and q <= result['queries']: 
                 # Note: result['queries'] is init+nqry, but snapshot keys are nqry (evo only).
                 # Wait, snapshot keys are 0, 200 etc. (evo steps).
                 # So we rely on snapshot keys. 
                 pass
            
            # Logic: 
            # keys: 0, 200, ... 
            # If q in snapshots, use it. 
            # If not in snapshots (e.g. q=1000 but finished at 452), use final.
            
            target_img = None
            if q in snapshots:
                target_img = snapshots[q]
            else:
                target_img = snapshots['final']
            
            adv_img_lists[i].append(np.transpose(target_img, (1, 2, 0)).astype(np.uint8))
        
        # Collect final image for final evaluation
        final_img = result['snapshots']['final']
        final_adv_img_list.append(np.transpose(final_img, (1, 2, 0)).astype(np.uint8))
        
        l0_metrics.append(result['l0'])
        ratio_metrics.append(result['ratio'])
        impact_metrics.append(result['impact'])
        query_metrics.append(result['queries'])
        success_ratio_metrics.append(result['success_ratio'])
        all_query_histories.append(result['query_history'])
        
        # 공격 성공한 이미지의 성공 쿼리 수집 (max_query 이하)
        attack_success_list.append(result['attack_success'])
        if result['attack_success'] and result['attack_success_query'] is not None:
            attack_success_queries.append(result['attack_success_query'])

    # 쿼리별 평균 메트릭 계산 (max_query 이하만)
    max_query_limit = config['max_query']
    query_step_averages = {}
    for qh in all_query_histories:
        for entry in qh:
            q = entry['query']
            if q > max_query_limit:  # max_query 초과 쿼리 제외
                continue
            if q not in query_step_averages:
                query_step_averages[q] = {'l0': [], 'pixel_ratio': [], 'impact': [], 
                                          'success_ratio': [], 'success_pixel': [], 'unsuccess_pixel': []}
            query_step_averages[q]['l0'].append(entry['l0'])
            query_step_averages[q]['pixel_ratio'].append(entry['pixel_ratio'])
            query_step_averages[q]['impact'].append(entry['impact'])
            query_step_averages[q]['success_ratio'].append(entry['success_ratio'])
            query_step_averages[q]['success_pixel'].append(entry['success_pixel'])
            query_step_averages[q]['unsuccess_pixel'].append(entry['unsuccess_pixel'])
    
    per_query_avg = []
    for q in sorted(query_step_averages.keys()):
        avg_entry = {
            'query': q,
            'avg_l0': float(np.mean(query_step_averages[q]['l0'])),
            'avg_pixel_ratio': float(np.mean(query_step_averages[q]['pixel_ratio'])),
            'avg_impact': float(np.mean(query_step_averages[q]['impact'])),
            'avg_success_ratio': float(np.mean(query_step_averages[q]['success_ratio'])),
            'avg_success_pixel': float(np.mean(query_step_averages[q]['success_pixel'])),
            'avg_unsuccess_pixel': float(np.mean(query_step_averages[q]['unsuccess_pixel']))
        }
        per_query_avg.append(avg_entry)

    # mIoU calc
    # mIoU calc
    _, init_mious = eval_miou(model, img_list, img_list, gt_list, config)
    
    benign_to_adv_mious = []
    gt_to_adv_mious = []
    acc_benign = []
    oacc_benign = []
    acc_gt = []
    oacc_gt = []
    avg_miou_no0_benign = []
    avg_miou_no0_gt = []
    per_cat_iou_benign_list = []
    per_cat_iou_gt_list = []
    per_cat_acc_benign_list = []
    per_cat_acc_gt_list = []
    query_labels = []

    for i in range(levels):
        if adv_img_lists[i]:
            benign_to_adv_miou, gt_to_adv_miou = eval_miou(model, img_list, adv_img_lists[i], gt_list, config)
            
            benign_to_adv_mious.append(benign_to_adv_miou['mean_iou'])
            acc_benign.append(benign_to_adv_miou['mean_accuracy'])
            oacc_benign.append(benign_to_adv_miou['overall_accuracy'])
            
            per_cat = np.array(benign_to_adv_miou['per_category_iou'])
            avg_miou_no0_benign.append(np.nanmean(per_cat[1:]) if len(per_cat) > 1 else per_cat[0])
            per_cat_iou_benign_list.append([round(float(v), 4) if not np.isnan(v) else None for v in per_cat])
            per_cat_acc = np.array(benign_to_adv_miou['per_category_accuracy'])
            per_cat_acc_benign_list.append([round(float(v), 4) if not np.isnan(v) else None for v in per_cat_acc])

            gt_to_adv_mious.append(gt_to_adv_miou['mean_iou'])
            acc_gt.append(gt_to_adv_miou['mean_accuracy'])
            oacc_gt.append(gt_to_adv_miou['overall_accuracy'])
            
            per_cat_gt = np.array(gt_to_adv_miou['per_category_iou'])
            avg_miou_no0_gt.append(np.nanmean(per_cat_gt[1:]) if len(per_cat_gt) > 1 else per_cat_gt[0])
            per_cat_iou_gt_list.append([round(float(v), 4) if not np.isnan(v) else None for v in per_cat_gt])
            per_cat_acc_gt = np.array(gt_to_adv_miou['per_category_accuracy'])
            per_cat_acc_gt_list.append([round(float(v), 4) if not np.isnan(v) else None for v in per_cat_acc_gt])
            
            query_labels.append(i * snapshot_interval)
            
    # Calculate Final mIoU (skip if last level already covers max_query)
    last_level_q = (levels - 1) * snapshot_interval
    if final_adv_img_list and last_level_q != config["max_query"]:
        final_benign_res, final_gt_res = eval_miou(model, img_list, final_adv_img_list, gt_list, config)
        
        benign_to_adv_mious.append(final_benign_res['mean_iou'])
        acc_benign.append(final_benign_res['mean_accuracy'])
        oacc_benign.append(final_benign_res['overall_accuracy'])
        
        per_cat = np.array(final_benign_res['per_category_iou'])
        avg_miou_no0_benign.append(np.nanmean(per_cat[1:]) if len(per_cat) > 1 else per_cat[0])
        per_cat_iou_benign_list.append([round(float(v), 4) if not np.isnan(v) else None for v in per_cat])
        per_cat_acc = np.array(final_benign_res['per_category_accuracy'])
        per_cat_acc_benign_list.append([round(float(v), 4) if not np.isnan(v) else None for v in per_cat_acc])

        gt_to_adv_mious.append(final_gt_res['mean_iou'])
        acc_gt.append(final_gt_res['mean_accuracy'])
        oacc_gt.append(final_gt_res['overall_accuracy'])
        
        per_cat_gt = np.array(final_gt_res['per_category_iou'])
        avg_miou_no0_gt.append(np.nanmean(per_cat_gt[1:]) if len(per_cat_gt) > 1 else per_cat_gt[0])
        per_cat_iou_gt_list.append([round(float(v), 4) if not np.isnan(v) else None for v in per_cat_gt])
        per_cat_acc_gt = np.array(final_gt_res['per_category_accuracy'])
        per_cat_acc_gt_list.append([round(float(v), 4) if not np.isnan(v) else None for v in per_cat_acc_gt])
        
        query_labels.append(config["max_query"])

    # Extract lists for PointWise-like format (Snapshot steps + Final Average)
    # Filter snapshots up to max_query to avoid sparse tails
    valid_queries = [entry for entry in per_query_avg if entry['query'] <= config["max_query"]]
    
    l0_list = [entry['avg_l0'] for entry in valid_queries]
    l0_list.append(np.mean(l0_metrics))
    
    ratio_list = [entry['avg_pixel_ratio'] for entry in valid_queries]
    ratio_list.append(np.mean(ratio_metrics))
    
    impact_list = [entry['avg_impact'] for entry in valid_queries]
    impact_list.append(np.mean(impact_metrics))

    # 공격 성공 통계 계산
    attack_success_count = sum(attack_success_list)
    attack_success_rate = attack_success_count / len(attack_success_list) if len(attack_success_list) > 0 else 0.0
    avg_attack_success_query = np.mean(attack_success_queries) if len(attack_success_queries) > 0 else None
    
    final_results = {
        "Attack Method": "SpaEvO",
        "Init mIoU": init_mious['mean_iou'],
        "Adversarial mIoU(benign)": benign_to_adv_mious,
        "Adversarial mIoU(gt)": gt_to_adv_mious,
        "Accuracy(benign)": acc_benign,
        "Overall Accuracy(benign)": oacc_benign,
        "Accuracy(gt)": acc_gt,
        "Overall Accuracy(gt)": oacc_gt,
        "L0": l0_list,
        "Ratio": ratio_list,
        "Impact": impact_list,
        "Per-category IoU(benign)": per_cat_iou_benign_list,
        "Per-category IoU(gt)": per_cat_iou_gt_list,
        "Per-category Accuracy(benign)": per_cat_acc_benign_list,
        "Per-category Accuracy(gt)": per_cat_acc_gt_list,
        "Average mIoU excluding label 0 (benign)": avg_miou_no0_benign,
        "Average mIoU excluding label 0 (gt)": avg_miou_no0_gt,
        "Query Labels": query_labels,
        "Average Queries": np.mean(query_metrics),
        "Max Query Limit": config["max_query"],
        "Pop Size": config.get("pop_size", 10),
        "NPix": config.get("n_pix", 196),
        "Success Ratios (per image)": success_ratio_metrics,
        "Mean Success Ratio": np.mean(success_ratio_metrics),
        "Attack Success Count": attack_success_count,
        "Attack Success Rate": attack_success_rate,
        "Attack Success Queries (per image)": attack_success_queries,
        "Average Attack Success Query": float(avg_attack_success_query) if avg_attack_success_query is not None else None,
        "Per Query Averages": per_query_avg
    }

    print("\n--- Experiment Summary ---")
    for key, value in final_results.items():
        print(f"{key}: {value}")

    save_experiment_results(final_results,
                            config,
                            sweep_config=None,
                            timestamp=current_time,
                            save_dir=base_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run SpaEvO attack evaluation.")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument('--device', type=str, default='cuda', help='Device to use.')
    parser.add_argument('--max_query', type=int, default=1000, help='Maximum queries for attack.')
    parser.add_argument('--num_images', type=int, default=100, help='Number of images to evaluate.')
    parser.add_argument('--pop_size', type=int, default=10, help='Population size.')
    parser.add_argument('--n_pix', type=int, default=0, help='Number of pixels to remove (sparseness). 0 = auto (0.004 × H × W).')
    parser.add_argument('--cr', type=float, default=0.9, help='Crossover rate.')
    parser.add_argument('--mu', type=float, default=0.004, help='Mutation rate.')
    parser.add_argument('--success_threshold', type=float, default=0.01, help='Threshold for attack success.')
    parser.add_argument('--init_mode', type=str, default='random', help='Starting point initialization mode.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')
    parser.add_argument('--include_bg', action='store_true',
                        help='Include background class in attack (do not exclude label 0 for VOC2012).')
    args = parser.parse_args()

    config = load_config(args.config)
    config["attack_method"] = "SpaEvO"
    config["device"] = args.device
    config["max_query"] = args.max_query
    config["num_images"] = args.num_images
    config["pop_size"] = args.pop_size
    config["n_pix"] = args.n_pix
    config["cr"] = args.cr
    config["mu"] = args.mu
    config["success_threshold"] = args.success_threshold
    config["init_mode"] = args.init_mode
    config["seed"] = args.seed
    config["verbose"] = args.verbose
    config["include_bg"] = args.include_bg
    config["base_dir"] = f"./data/SpaEvO/results/{config['dataset']}/{config['model']}"

    main(config)
