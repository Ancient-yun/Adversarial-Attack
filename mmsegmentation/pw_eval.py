"""
PointWise Attack Evaluation for MMSegmentation
Based on the original PointWise attack for image classification,
adapted for semantic segmentation models.
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
import multiprocessing as mp

# 상위 디렉토리와 현재 디렉토리를 sys.path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from mmseg.apis import init_model, inference_model
from dataset import CitySet, ADESet, VOCSet
from pointwise_attack import PointWiseAttack, l0
from utils_se import salt_pepper_noise, rand_img_upscale

from function import *
from evaluation import *
from utils import save_experiment_results

import argparse
import setproctitle


def load_config(config_path):
    """Load and return config dictionary from a python file at config_path."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module.config


def init_model_for_process(model_configs, dataset, model_name, device):
    """각 프로세스에서 모델을 초기화하는 함수"""
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
        scales = [1, 2, 4, 8, 16, 32, 64]      # 중해상도 (~224-500px)
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
                # Fallback: 모든 scale 실패 → 전체 랜덤 이미지 사용
                print(f'Fallback: using full random image as starting point')
                timg = torch.rand_like(oimg).cuda() * 255.0
                nquery += 1
                D = torch.ones(nquery, dtype=int).cuda() * l0(oimg, timg)
                return timg, nquery, D


def process_single_image(args):
    """단일 이미지를 처리하는 함수"""
    (img_bgr, filename, gt, model_configs, config, base_dir, idx, total_images, save_steps) = args
    
    # 프로세스별 모델 초기화
    model = init_model_for_process(model_configs, config["dataset"], config["model"], config["device"])
    
    setproctitle.setproctitle(f"({idx+1}/{total_images})_PointWise_Attack_{config['dataset']}_{config['model']}_{config['success_threshold']}")

    img_tensor_bgr = torch.from_numpy(img_bgr.copy()).unsqueeze(0).permute(0, 3, 1, 2).float().to(config["device"])
    gt_tensor = torch.from_numpy(gt.copy()).unsqueeze(0).long().to(config["device"])

    ori_result = inference_model(model, img_bgr.copy()) 
    ori_pred = ori_result.pred_sem_seg.data.squeeze().cpu().numpy()
    original_pred_labels = ori_result.pred_sem_seg.data.squeeze().cuda()

    # PointWise Attack 객체 생성
    attack = PointWiseAttack(
        model=model,
        cfg=config,
        is_mmseg=True,
        is_detectron2=False,
        success_threshold=config.get("success_threshold", 0.01),
        verbose=config.get("verbose", False)
    )
    attack.set_ignore_index(config["dataset"])  # 데이터셋별 ignore index 설정

    # Starting Point 생성
    print(f"\n[{idx+1}/{total_images}] {filename}: Generating starting point...")
    timg, init_nqry, _ = gen_starting_point_seg(
        attack, img_tensor_bgr, original_pred_labels, 
        seed=config.get("seed", 0), 
        init_mode=config.get("init_mode", "salt_pepper"),
        dataset_name=config["dataset"]
    )

    levels = len(save_steps)
    adv_img_bgr_list = []
    adv_query_list = []
    total_nquery = init_nqry

    adv_img_bgr_list = []
    adv_query_list = []
    total_nquery = init_nqry

    # 0번(원본)은 나중에 채움

    # PointWise Attack 실행
    print(f"[{idx+1}/{total_images}] {filename}: Running PointWise attack (mode={config['attack_mode']})...")
    
    snapshot_interval = 200
    if config["attack_mode"] == "single":
        x, nquery, D, snapshots = attack.pw_perturb(
            img_tensor_bgr.squeeze(0), timg.squeeze(0), original_pred_labels,
            max_query=config["max_query"],
            snapshot_interval=snapshot_interval
        )
    elif config["attack_mode"] == "multiple":
        x, nquery, D, snapshots = attack.pw_perturb_multiple(
            img_tensor_bgr.squeeze(0), timg.squeeze(0), original_pred_labels,
            npix=config.get("npix", 196),
            max_query=config["max_query"],
            snapshot_interval=snapshot_interval
        )
    elif config["attack_mode"] == "scheduling":
        x, nquery, D, snapshots = attack.pw_perturb_multiple_scheduling(
            img_tensor_bgr.squeeze(0), timg.squeeze(0), original_pred_labels,
            npix=config.get("npix", 196),
            max_query=config["max_query"],
            snapshot_interval=snapshot_interval
        )
    else:
        raise ValueError(f"Unknown attack mode: {config['attack_mode']}")

    total_nquery += nquery
    
    # Reshape result to tensor
    adv_img_bgr = torch.from_numpy(x.reshape(img_tensor_bgr.squeeze(0).shape)).unsqueeze(0).float().cuda()
    
    # Save results by steps
    for step in save_steps:
        target_q = step - init_nqry
        img_to_add = None
        current_query = step
        
        if step == 0:
            img_to_add = img_tensor_bgr
        else:
             if target_q < 0:
                 # Init 단계
                 img_to_add = timg.unsqueeze(0) if timg.dim()==3 else timg
                 # Check dim
                 if img_to_add.dim() == 3: img_to_add = img_to_add.unsqueeze(0)
             else:
                 if target_q > nquery:
                     # Finished early
                     img_to_add = adv_img_bgr
                     current_query = total_nquery
                 else:
                     key = (target_q // snapshot_interval) * snapshot_interval
                     if key in snapshots:
                         snap_np = snapshots[key]
                         img_to_add = torch.from_numpy(snap_np).unsqueeze(0).float().to(config["device"])
                     else:
                         img_to_add = adv_img_bgr 
                         current_query = total_nquery

        adv_img_bgr_list.append(img_to_add)
        adv_query_list.append(current_query)

    # Ensure final result is included if not covered
    if total_nquery > adv_query_list[-1]:
         adv_img_bgr_list.append(adv_img_bgr)
         adv_query_list.append(total_nquery)

    # 결과 저장
    current_img_save_dir = os.path.join(base_dir, os.path.splitext(os.path.basename(filename))[0])
    os.makedirs(current_img_save_dir, exist_ok=True)

    Image.fromarray(img_bgr[:, :, ::-1]).save(os.path.join(current_img_save_dir, "original.png"))
    # GT 세그멘테이션 저장 (Segmentation Visualization)
    visualize_segmentation(img_bgr, gt,
                        save_path=os.path.join(current_img_save_dir, "gt.png"),
                        alpha=1.0, dataset=config["dataset"])
    
    # 원본 세그멘테이션 저장 (메인 디렉토리)
    visualize_segmentation(img_bgr, ori_pred,
                        save_path=os.path.join(current_img_save_dir, "ori_seg.png"),
                        alpha=0.5, dataset=config["dataset"])
    visualize_segmentation(img_bgr, ori_pred,
                        save_path=os.path.join(current_img_save_dir, "ori_seg_only.png"),
                        alpha=1.0, dataset=config["dataset"])

    print(f"[{idx+1}/{total_images}] {filename}: Completed with {total_nquery} queries")
    
    # 메트릭 계산
    l0_metrics = []
    ratio_metrics = []
    impact_metrics = []
    
    for i, adv_img in enumerate(adv_img_bgr_list):
        query_val = adv_query_list[i]
        adv_img_np = adv_img.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        if i == 0:
            l0_norm = 0
            pixel_ratio = 0.0
            impact = 0.0
        else:
            query_img_save_dir = os.path.join(current_img_save_dir, f"{query_val}query")
            os.makedirs(query_img_save_dir, exist_ok=True)

            adv_result = inference_model(model, adv_img_np)
            adv_pred = adv_result.pred_sem_seg.data.squeeze().cpu().numpy()
            delta_img = np.abs(img_bgr.astype(np.int16) - adv_img_np.astype(np.int16)).astype(np.uint8)

            l0_norm = calculate_l0_norm(img_bgr, adv_img_np)
            pixel_ratio = calculate_pixel_ratio(img_bgr, adv_img_np)
            impact = calculate_impact(img_bgr, adv_img_np, ori_pred, adv_pred)
            
            # 최종 success_ratio 계산 (배경/ignore 제외, 원본 예측 대비 변경된 픽셀 비율)
            ignore_index = 255 if config["dataset"].lower() == "cityscapes" else 0
            foreground_mask = ori_pred != ignore_index
            if foreground_mask.sum() > 0:
                success_ratio = ((adv_pred != ori_pred) & foreground_mask).sum() / foreground_mask.sum()
            else:
                success_ratio = 0.0

            Image.fromarray(adv_img_np[:, :, ::-1]).save(os.path.join(query_img_save_dir, "adv.png"))
            Image.fromarray(delta_img).save(os.path.join(query_img_save_dir, "delta.png"))

            visualize_segmentation(adv_img_np, adv_pred,
                                save_path=os.path.join(query_img_save_dir, "adv_seg.png"),
                                alpha=0.5, dataset=config["dataset"])
            visualize_segmentation(adv_img_np, adv_pred,
                                save_path=os.path.join(query_img_save_dir, "adv_seg_only.png"),
                                alpha=1.0, dataset=config["dataset"])

        if i == 0:
            success_ratio = 0.0
        print(f"  L0 norm: {l0_norm}, Pixel ratio: {pixel_ratio:.4f}, Impact: {impact:.4f}, Success Ratio: {success_ratio:.4f}")

        l0_metrics.append(l0_norm)
        ratio_metrics.append(pixel_ratio)
        impact_metrics.append(impact)

    # 각 이미지별 결과를 JSON으로 저장
    # foreground 픽셀 수 및 공격 실패 픽셀 수 계산
    ignore_index = 255 if config["dataset"].lower() == "cityscapes" else 0
    foreground_mask = ori_pred != ignore_index
    foreground_pixels = int(foreground_mask.sum())
    
    # 마지막 adversarial 이미지의 예측 사용
    if len(adv_img_bgr_list) > 1:
        final_adv_np = adv_img_bgr_list[-1].squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        final_adv_result = inference_model(model, final_adv_np)
        final_adv_pred = final_adv_result.pred_sem_seg.data.squeeze().cpu().numpy()
        unchanged_pixels = int(((final_adv_pred == ori_pred) & foreground_mask).sum())
        changed_pixels = int(((final_adv_pred != ori_pred) & foreground_mask).sum())
    else:
        unchanged_pixels = foreground_pixels
        changed_pixels = 0
    
    # 쿼리별 결과 생성 (스냅샷 활용)
    query_history = []
    D_numpy = D.cpu().numpy() if isinstance(D, torch.Tensor) else D
    
    # 정수형 키(쿼리 번호)만 추출하여 정렬
    snapshot_queries = sorted([k for k in snapshots.keys() if isinstance(k, int)])
    
    for query_num in snapshot_queries:
        snapshot_img = snapshots[query_num]
        snapshot_np = np.transpose(snapshot_img, (1, 2, 0)).astype(np.uint8)  # (C,H,W) -> (H,W,C)
        
        # 해당 쿼리 시점의 메트릭 계산
        if query_num == 0:
            # 0번 쿼리는 초기 상태이므로 L0 직접 계산 (또는 D의 첫 값?)
            # D는 공격 진행 중 기록되므로 0번 쿼리 시점(공격 전)과 다를 수 있음
            # 스냅샷 이미지로 직접 계산
            l0_at_q = int((snapshot_np != img_bgr).sum() / 3) # 간단한 L0 계산 (RGB 채널 고려)
            # 정확한 L0 함수 사용 권장: from pointwise_attack import l0_distance
            # 여기서는 간단히 diff로 계산. (img_bgr은 원본)
            # l0_distance 함수가 있다면 사용:
            # l0_at_q = l0_distance(img_bgr, snapshot_np)
            # 임시로 numpy 계산
            diff = np.abs(snapshot_np.astype(int) - img_bgr.astype(int))
            # 픽셀 단위 L0 (채널 중 하나라도 다르면 1)
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
        final_query = snapshots.get('final_query', total_nquery)
        # 이미 정수형 키에 포함되어 있는지 확인 (중복 방지)
        if final_query not in snapshot_queries:
            snapshot_img = snapshots['final']
            snapshot_np = np.transpose(snapshot_img, (1, 2, 0)).astype(np.uint8)
            
            # 메트릭 계산 - D의 마지막 값 사용
            l0_at_q = int(D_numpy[final_query-1]) if final_query > 0 and final_query-1 < len(D_numpy) else int(D_numpy[-1] if len(D_numpy) > 0 else 0)
            if final_query == 0: # 혹시 0이면 직접 계산
                 diff = np.abs(snapshot_np.astype(int) - img_bgr.astype(int))
                 l0_at_q = int(np.sum(np.sum(diff, axis=2) > 0))

            pixel_ratio_at_q = calculate_pixel_ratio(img_bgr, snapshot_np)
            
            snapshot_result = inference_model(model, snapshot_np)
            snapshot_pred = snapshot_result.pred_sem_seg.data.squeeze().cpu().numpy()
            
            success_pixel_at_q = int(((snapshot_pred != ori_pred) & foreground_mask).sum())
            unsuccess_pixel_at_q = int(((snapshot_pred == ori_pred) & foreground_mask).sum())
            impact_at_q = calculate_impact(img_bgr, snapshot_np, ori_pred, snapshot_pred)
            success_ratio_at_q = success_pixel_at_q / foreground_pixels if foreground_pixels > 0 else 0.0
            
            # 이미지 저장 (Final은 무조건 저장)
            save_q_dir = os.path.join(current_img_save_dir, f"{final_query}query")
            os.makedirs(save_q_dir, exist_ok=True)
            Image.fromarray(snapshot_np[:, :, ::-1]).save(os.path.join(save_q_dir, "adv.png"))
            
            # Delta 이미지 저장
            delta_img = np.abs(img_bgr.astype(np.int16) - snapshot_np.astype(np.int16)).astype(np.uint8)
            Image.fromarray(delta_img).save(os.path.join(save_q_dir, "delta.png"))
            
            visualize_segmentation(snapshot_np, snapshot_pred, 
                save_path=os.path.join(save_q_dir, "adv_seg.png"),
                alpha=0.5, dataset=config["dataset"])
            visualize_segmentation(snapshot_np, snapshot_pred, 
                save_path=os.path.join(save_q_dir, "adv_seg_only.png"),
                alpha=1.0, dataset=config["dataset"])
            
            query_history.append({
                'query': final_query,
                'l0': l0_at_q,
                'pixel_ratio': float(pixel_ratio_at_q),
                'impact': float(impact_at_q),
                'success_ratio': float(success_ratio_at_q),
                'success_pixel': success_pixel_at_q,
                'unsuccess_pixel': unsuccess_pixel_at_q
            })
    
    image_result = {
        'filename': filename,
        'total_query': total_nquery,
        'l0': l0_metrics[-1] if l0_metrics else 0,
        'pixel_ratio': ratio_metrics[-1] if ratio_metrics else 0.0,
        'impact': impact_metrics[-1] if impact_metrics else 0.0,
        'success_ratio': float(success_ratio),
        'foreground_pixels': foreground_pixels,
        'success_pixel': changed_pixels,
        'unsuccess_pixel': unchanged_pixels,
        'attack_mode': config['attack_mode'],
        'max_query': config['max_query'],
        'success_threshold': config['success_threshold'],
        'query_history': query_history
    }
    with open(os.path.join(current_img_save_dir, 'result.json'), 'w') as f:
        json.dump(image_result, f, indent=2)

    # 모델 메모리 정리
    del model
    del attack
    torch.cuda.empty_cache()
    
    return {
        'img_bgr': img_bgr,
        'gt': gt,
        'filename': filename,
        'adv_img_bgr_list': adv_img_bgr_list,
        'adv_query_list': adv_query_list,
        'total_query': total_nquery,
        'l0_metrics': l0_metrics,
        'ratio_metrics': ratio_metrics,
        'impact_metrics': impact_metrics,
        'distance_history': D,
        'success_ratio': success_ratio,
        'query_history': query_history,
        'snapshots': snapshots
    }


def main(config):
    # Model configs (rs_eval.py와 동일)
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
                "config": 'configs/setr/setr_vit-l_pup_8xb1-80k_ade20k-768x768.py',
                "checkpoint": '../ckpt/setr_pup_vit-large_8x1_768x768_80k_ade20k_20211122_155115-f6f37b8f.pth'
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

    if config["dataset"] not in model_configs:
        raise ValueError(f"Unsupported dataset: {config['dataset']}")
    if config["model"] not in model_configs[config["dataset"]]:
        raise ValueError(f"Unsupported model: {config['model']} for dataset {config['dataset']}")

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

    # 결과 저장 디렉토리
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{config['dataset']}_{config['model']}_pointwise_{current_time}"
    base_dir = os.path.join(config["base_dir"], current_time)
    os.makedirs(base_dir, exist_ok=True)

    snapshot_interval = config.get("snapshot_interval", 200)
    save_steps = list(range(0, config["max_query"] + 1, snapshot_interval))
    levels = len(save_steps)

    # 처리 데이터 준비
    process_args = []
    for idx, (img_bgr, filename, gt) in enumerate(zip(dataset.images, dataset.filenames, dataset.gt_images)):
        process_args.append((img_bgr, filename, gt, model_cfg, config, base_dir, idx, len(dataset.images), save_steps))

    # 모델 초기화 (메트릭 계산용)
    model = init_model_for_process(model_cfg, config["dataset"], config["model"], device)

    # 순차 처리
    print(f"\nProcessing {len(process_args)} images with PointWise Attack...")
    results = []
    
    img_list = []
    gt_list = []
    filename_list = []
    adv_img_lists = [[] for _ in range(levels)]
    final_adv_img_list = []
    
    all_l0_metrics = [[] for _ in range(levels)]
    final_l0_list = []
    
    all_ratio_metrics = [[] for _ in range(levels)]
    final_ratio_list = []
    
    all_impact_metrics = [[] for _ in range(levels)]
    final_impact_list = []
    
    all_queries = []
    all_success_ratios = []
    all_query_histories = []

    for args in tqdm(process_args, desc="PointWise Attack"):
        result = process_single_image(args)
        results.append(result)
        
        img_list.append(result['img_bgr'])
        gt_list.append(result['gt'])
        filename_list.append(result['filename'])
        all_queries.append(result['total_query'])
        all_success_ratios.append(result['success_ratio'])
        all_query_histories.append(result['query_history'])

        for i, adv_img in enumerate(result['adv_img_bgr_list']):
            if i < levels:
                adv_img_lists[i].append(adv_img.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8))
                all_l0_metrics[i].append(result['l0_metrics'][i])
                all_ratio_metrics[i].append(result['ratio_metrics'][i])
                all_impact_metrics[i].append(result['impact_metrics'][i])

        # Collect final results
        if 'snapshots' in result and 'final' in result['snapshots']:
            final_img = result['snapshots']['final']
            final_adv_img_list.append(np.transpose(final_img, (1, 2, 0)).astype(np.uint8))
            
            # Get final metrics from query_history
            if result['query_history']:
                last_hist = result['query_history'][-1]
                final_l0_list.append(last_hist['l0'])
                final_ratio_list.append(last_hist['pixel_ratio'])
                final_impact_list.append(last_hist['impact'])

    # 쿼리별 평균 메트릭 계산
    query_step_averages = {}
    for qh in all_query_histories:
        for entry in qh:
            q = entry['query']
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

    # mIoU 계산
    _, init_mious = eval_miou(model, img_list, img_list, gt_list, config)
    
    benign_to_adv_mious = []
    gt_to_adv_mious = []
    acc_benign = []
    oacc_benign = []
    acc_gt = []
    oacc_gt = []
    avg_miou_no0_benign = []
    avg_miou_no0_gt = []

    mean_l0 = []
    mean_ratio = []
    mean_impact = []
    
    for i in range(levels):
        if adv_img_lists[i]:
            benign_to_adv_miou, gt_to_adv_miou = eval_miou(model, img_list, adv_img_lists[i], gt_list, config)
            
            benign_to_adv_mious.append(benign_to_adv_miou['mean_iou'])
            acc_benign.append(benign_to_adv_miou['mean_accuracy'])
            oacc_benign.append(benign_to_adv_miou['overall_accuracy'])
            
            per_cat = np.array(benign_to_adv_miou['per_category_iou'])
            per_cat = np.array(benign_to_adv_miou['per_category_iou'])
            avg_miou_no0_benign.append(np.nanmean(per_cat[1:]) if len(per_cat) > 1 else per_cat[0])

            gt_to_adv_mious.append(gt_to_adv_miou['mean_iou'])
            acc_gt.append(gt_to_adv_miou['mean_accuracy'])
            oacc_gt.append(gt_to_adv_miou['overall_accuracy'])
            
            per_cat_gt = np.array(gt_to_adv_miou['per_category_iou'])
            avg_miou_no0_gt.append(np.nanmean(per_cat_gt[1:]) if len(per_cat_gt) > 1 else per_cat_gt[0])
            
        mean_l0.append(np.mean(all_l0_metrics[i]).item() if all_l0_metrics[i] else 0)
        mean_ratio.append(np.mean(all_ratio_metrics[i]).item() if all_ratio_metrics[i] else 0)
        mean_impact.append(np.mean(all_impact_metrics[i]).item() if all_impact_metrics[i] else 0)
            
    # Process Final Results
    if final_adv_img_list:
        final_benign_res, final_gt_res = eval_miou(model, img_list, final_adv_img_list, gt_list, config)
        
        benign_to_adv_mious.append(final_benign_res['mean_iou'])
        acc_benign.append(final_benign_res['mean_accuracy'])
        oacc_benign.append(final_benign_res['overall_accuracy'])
        
        per_cat = np.array(final_benign_res['per_category_iou'])
        avg_miou_no0_benign.append(np.nanmean(per_cat[1:]) if len(per_cat) > 1 else per_cat[0])

        gt_to_adv_mious.append(final_gt_res['mean_iou'])
        acc_gt.append(final_gt_res['mean_accuracy'])
        oacc_gt.append(final_gt_res['overall_accuracy'])
        
        per_cat_gt = np.array(final_gt_res['per_category_iou'])
        avg_miou_no0_gt.append(np.nanmean(per_cat_gt[1:]) if len(per_cat_gt) > 1 else per_cat_gt[0])

    # Append final metrics averages
    # Append final metrics averages
    if final_l0_list:
        mean_l0.append(np.mean(final_l0_list))
        mean_ratio.append(np.mean(final_ratio_list))
        mean_impact.append(np.mean(final_impact_list))

    final_results = {
        "Attack Method": "PointWise",
        "Attack Mode": config["attack_mode"],
        "Init mIoU": init_mious['mean_iou'],
        "Adversarial mIoU(benign)": benign_to_adv_mious,
        "Adversarial mIoU(gt)": gt_to_adv_mious,
        "Accuracy(benign)": acc_benign,
        "Overall Accuracy(benign)": oacc_benign,
        "Accuracy(gt)": acc_gt,
        "Overall Accuracy(gt)": oacc_gt,
        "L0": mean_l0,
        "Ratio": mean_ratio,
        "Impact": mean_impact,
        "Average mIoU excluding label 0 (benign)": avg_miou_no0_benign,
        "Average mIoU excluding label 0 (gt)": avg_miou_no0_gt,
        "Average Queries": np.mean(all_queries).item(),
        "Max Query Limit": config["max_query"],
        "NPix": config.get("npix", 196),
        "Success Ratios (per image)": all_success_ratios,
        "Mean Success Ratio": np.mean(all_success_ratios).item(),
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
    parser = argparse.ArgumentParser(description="Run PointWise attack evaluation.")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument('--device', type=str, default='cuda', help='Device to use.')
    parser.add_argument('--max_query', type=int, default=1000, help='Maximum queries for attack.')
    parser.add_argument('--num_images', type=int, default=10, help='Number of images to evaluate.')
    parser.add_argument('--attack_mode', type=str, default='scheduling', 
                        choices=['single', 'multiple', 'scheduling'],
                        help='Attack mode: single, multiple, or scheduling.')
    parser.add_argument('--npix', type=int, default=196, help='Pixels per group for multiple mode.')
    parser.add_argument('--success_threshold', type=float, default=0.01, 
                        help='Threshold for attack success (ratio of changed pixels).')
    parser.add_argument('--init_mode', type=str, default='random',
                        choices=['salt_pepper', 'random'],
                        help='Starting point initialization mode.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')
    args = parser.parse_args()

    config = load_config(args.config)
    config["attack_method"] = "PointWise"
    config["device"] = args.device
    config["max_query"] = args.max_query
    config["num_images"] = args.num_images
    config["attack_mode"] = args.attack_mode
    config["npix"] = args.npix
    config["success_threshold"] = args.success_threshold
    config["init_mode"] = args.init_mode
    config["seed"] = args.seed
    config["verbose"] = args.verbose
    config["base_dir"] = f"./data/PointWise/results/{config['dataset']}/{config['model']}"

    main(config)
