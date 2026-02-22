"""
PointWise Attack Evaluation for Robust-Semantic-Segmentation Models
Uses adv_setting.py's load_model/model_predict for model loading and prediction.
Result saving structure adapted from mmsegmentation/pw_eval.py.
"""

import os
import sys
import torch
import json
from tqdm import tqdm
import datetime
import numpy as np
from PIL import Image
import setproctitle
import argparse

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from adv_setting.py
from adv_setting import model_predict, load_model, set_seed

# Import from parent directory
from dataset import CitySet, ADESet, VOCSet
from pointwise_attack import PointWiseAttack, l0
from utils_se import salt_pepper_noise, rand_img_upscale
from function import *
from evaluation import *
from utils import save_experiment_results


def gen_starting_point_seg(attack, oimg, original_pred_labels, seed=None, init_mode='salt_pepper', dataset_name='voc2012'):
    """
    세그멘테이션용 Starting Point 생성 함수.
    utils_se.py의 salt_pepper_noise와 rand_img_upscale을 사용합니다.
    """
    if len(oimg.shape) == 3:
        oimg = oimg.unsqueeze(0)
    
    nquery = 0
    i = 0
    rndtype = 'normal'
    
    if dataset_name == 'cityscapes':
        scales = [1, 2, 4, 8, 16, 32, 64]
    elif dataset_name in ['imagenet', 'voc2012', 'VOC2012', 'ade20k']:
        scales = [1, 2, 4, 8, 16, 32, 64]
    else:
        scales = [1, 2, 4, 8, 16]
    
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
                print(f'Fallback: using full random image as starting point')
                timg = torch.rand_like(oimg).cuda() * 255.0
                nquery += 1
                D = torch.ones(nquery, dtype=int).cuda() * l0(oimg, timg)
                return timg, nquery, D


def init_robust_model(model_config, device):
    """Robust 모델을 adv_setting.py 방식으로 초기화하는 함수"""
    model = load_model(model_config)
    checkpoint = torch.load(model_config["model_path"], map_location=device, weights_only=False)
    model = model.to(device)
    model = torch.nn.DataParallel(model)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    del checkpoint
    torch.cuda.empty_cache()
    return model


def process_single_image(args):
    """단일 이미지를 처리하는 함수"""
    (img_bgr, filename, gt, model_config, config, base_dir, idx, total_images) = args
    
    model = init_robust_model(model_config, config["device"])
    setproctitle.setproctitle(f"({idx+1}/{total_images})_PointWise_Attack_{config['dataset']}_{config['model']}_{config['success_threshold']}_{config['npix']}")

    img_tensor_bgr = torch.from_numpy(img_bgr.copy()).unsqueeze(0).permute(0, 3, 1, 2).float().to(config["device"])
    gt_tensor = torch.from_numpy(gt.copy()).unsqueeze(0).long().to(config["device"])

    # Original prediction using model_predict
    ori_confidence, ori_pred = model_predict(model, img_bgr.copy(), model_config)
    ori_pred_np = ori_pred.cpu().numpy()
    original_pred_labels = ori_pred.cuda()

    # PointWise Attack 객체 생성 (is_mmseg=False for Robust models)
    attack = PointWiseAttack(
        model=model,
        cfg=model_config,
        is_mmseg=False,
        is_detectron2=False,
        success_threshold=config.get("success_threshold", 0.01),
        verbose=config.get("verbose", False)
    )
    attack.set_ignore_index(config["dataset"], include_bg=config.get("include_bg", False))

    # Starting Point 생성
    print(f"\n[{idx+1}/{total_images}] {filename}: Generating starting point...")
    timg, init_nqry, _ = gen_starting_point_seg(
        attack, img_tensor_bgr, original_pred_labels, 
        seed=config.get("seed", 0), 
        init_mode=config.get("init_mode", "salt_pepper"),
        dataset_name=config["dataset"]
    )

    # PointWise Attack 실행
    print(f"[{idx+1}/{total_images}] {filename}: Running PointWise attack (mode={config['attack_mode']})...")
    
    # Calculate npix
    npix_input = config.get("npix", 196)
    if npix_input >= 1:
        npix = int(npix_input)
    else:
        _, _, H, W = img_tensor_bgr.shape
        total_pixels = H * W
        npix = max(1, int(total_pixels * npix_input))
        print(f"    -> Using npix ratio={npix_input}: {npix} pixels (image: {H}x{W}={total_pixels})")
    
    snapshot_interval = 200
    attack_max_query = config["max_query"] - init_nqry
    if config["attack_mode"] == "single":
        x, nquery, D, snapshots = attack.pw_perturb(
            img_tensor_bgr.squeeze(0), timg.squeeze(0), original_pred_labels,
            max_query=attack_max_query,
            snapshot_interval=snapshot_interval,
            query_offset=init_nqry
        )
    elif config["attack_mode"] == "multiple":
        x, nquery, D, snapshots = attack.pw_perturb_multiple(
            img_tensor_bgr.squeeze(0), timg.squeeze(0), original_pred_labels,
            npix=npix,
            max_query=attack_max_query,
            snapshot_interval=snapshot_interval,
            query_offset=init_nqry
        )
    elif config["attack_mode"] == "scheduling":
        x, nquery, D, snapshots = attack.pw_perturb_multiple_scheduling(
            img_tensor_bgr.squeeze(0), timg.squeeze(0), original_pred_labels,
            npix=npix,
            max_query=attack_max_query,
            snapshot_interval=snapshot_interval,
            query_offset=init_nqry
        )
    else:
        raise ValueError(f"Unknown attack mode: {config['attack_mode']}")

    total_nquery = init_nqry + nquery
    
    # Reshape result to tensor
    adv_img_bgr = torch.from_numpy(x.reshape(img_tensor_bgr.squeeze(0).shape)).unsqueeze(0).float().cuda()

    # 결과 저장
    current_img_save_dir = os.path.join(base_dir, os.path.splitext(os.path.basename(filename))[0])
    os.makedirs(current_img_save_dir, exist_ok=True)

    Image.fromarray(img_bgr[:, :, ::-1]).save(os.path.join(current_img_save_dir, "original.png"))
    visualize_segmentation(img_bgr, gt,
                        save_path=os.path.join(current_img_save_dir, "gt.png"),
                        alpha=1.0, dataset=config["dataset"])
    
    visualize_segmentation(img_bgr, ori_pred_np,
                        save_path=os.path.join(current_img_save_dir, "ori_seg.png"),
                        alpha=0.5, dataset=config["dataset"])
    visualize_segmentation(img_bgr, ori_pred_np,
                        save_path=os.path.join(current_img_save_dir, "ori_seg_only.png"),
                        alpha=1.0, dataset=config["dataset"])

    # Calculate final metrics
    adv_img_np = adv_img_bgr.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    
    l0_norm = calculate_l0_norm(img_bgr, adv_img_np)
    pixel_ratio = calculate_pixel_ratio(img_bgr, adv_img_np)
    
    _, adv_pred = model_predict(model, adv_img_np, model_config)
    adv_pred_np = adv_pred.cpu().numpy()
    impact = calculate_impact(img_bgr, adv_img_np, ori_pred_np, adv_pred_np)
    
    # 최종 success_ratio 계산
    if config["dataset"].lower() in ["cityscapes", "ade20k"]:
        ignore_index = 255
    elif config["dataset"] == "VOC2012":
        ignore_index = None if config.get("include_bg", False) else 0
    else:
        ignore_index = None
    if ignore_index is not None:
        foreground_mask = ori_pred_np != ignore_index
    else:
        foreground_mask = np.ones_like(ori_pred_np, dtype=bool)
    if foreground_mask.sum() > 0:
        success_ratio = ((adv_pred_np != ori_pred_np) & foreground_mask).sum() / foreground_mask.sum()
    else:
        success_ratio = 0.0
    
    print(f"[{idx+1}/{total_images}] {filename}: Completed. L0={l0_norm}, Ratio={pixel_ratio:.4f}, Impact={impact:.4f}, Success Ratio={success_ratio:.4f}")

    # 각 이미지별 결과를 JSON으로 저장
    foreground_pixels = int(foreground_mask.sum())
    unsuccess_pixel = int(((adv_pred_np == ori_pred_np) & foreground_mask).sum())
    success_pixel = int(((adv_pred_np != ori_pred_np) & foreground_mask).sum())
    
    # 쿼리별 결과 생성 (스냅샷 활용)
    query_history = []
    D_numpy = D.cpu().numpy() if isinstance(D, torch.Tensor) else D
    
    snapshot_queries = sorted([k for k in snapshots.keys() if isinstance(k, int)])
    
    for query_num in snapshot_queries:
        snapshot_img = snapshots[query_num]
        snapshot_np = np.transpose(snapshot_img, (1, 2, 0)).astype(np.uint8)
        
        if query_num == 0:
            diff = np.abs(snapshot_np.astype(int) - img_bgr.astype(int))
            l0_at_q = int(np.sum(np.sum(diff, axis=2) > 0))
        else:
            l0_at_q = int(D_numpy[query_num-1]) if query_num-1 < len(D_numpy) else int(D_numpy[-1])
            
        pixel_ratio_at_q = calculate_pixel_ratio(img_bgr, snapshot_np)
        
        _, snapshot_pred = model_predict(model, snapshot_np, model_config)
        snapshot_pred_np = snapshot_pred.cpu().numpy()
        
        success_pixel_at_q = int(((snapshot_pred_np != ori_pred_np) & foreground_mask).sum())
        unsuccess_pixel_at_q = int(((snapshot_pred_np == ori_pred_np) & foreground_mask).sum())
        impact_at_q = calculate_impact(img_bgr, snapshot_np, ori_pred_np, snapshot_pred_np)
        success_ratio_at_q = success_pixel_at_q / foreground_pixels if foreground_pixels > 0 else 0.0
        
        # 이미지 저장 (0 또는 200 단위)
        if query_num == 0 or query_num % 200 == 0:
            save_q_dir = os.path.join(current_img_save_dir, f"{query_num}query")
            os.makedirs(save_q_dir, exist_ok=True)
            Image.fromarray(snapshot_np[:, :, ::-1]).save(os.path.join(save_q_dir, "adv.png"))
            
            delta_img = np.abs(img_bgr.astype(np.int16) - snapshot_np.astype(np.int16)).astype(np.uint8)
            Image.fromarray(delta_img).save(os.path.join(save_q_dir, "delta.png"))
            
            visualize_segmentation(snapshot_np, snapshot_pred_np, 
                save_path=os.path.join(save_q_dir, "adv_seg.png"),
                alpha=0.5, dataset=config["dataset"])
            visualize_segmentation(snapshot_np, snapshot_pred_np, 
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
        snapshot_img = snapshots['final']
        snapshot_np = np.transpose(snapshot_img, (1, 2, 0)).astype(np.uint8)
        
        l0_at_q = int(D_numpy[-1]) if len(D_numpy) > 0 else 0
        pixel_ratio_at_q = calculate_pixel_ratio(img_bgr, snapshot_np)
        
        _, snapshot_pred = model_predict(model, snapshot_np, model_config)
        snapshot_pred_np = snapshot_pred.cpu().numpy()
        
        success_pixel_at_q = int(((snapshot_pred_np != ori_pred_np) & foreground_mask).sum())
        unsuccess_pixel_at_q = int(((snapshot_pred_np == ori_pred_np) & foreground_mask).sum())
        impact_at_q = calculate_impact(img_bgr, snapshot_np, ori_pred_np, snapshot_pred_np)
        success_ratio_at_q = success_pixel_at_q / foreground_pixels if foreground_pixels > 0 else 0.0
        
        # 이미지 저장 (Final은 무조건 저장)
        final_query = total_nquery
        save_q_dir = os.path.join(current_img_save_dir, f"{final_query}query")
        os.makedirs(save_q_dir, exist_ok=True)
        Image.fromarray(snapshot_np[:, :, ::-1]).save(os.path.join(save_q_dir, "adv.png"))
        delta_img = np.abs(img_bgr.astype(np.int16) - snapshot_np.astype(np.int16)).astype(np.uint8)
        Image.fromarray(delta_img).save(os.path.join(save_q_dir, "delta.png"))
        visualize_segmentation(snapshot_np, snapshot_pred_np, 
            save_path=os.path.join(save_q_dir, "adv_seg.png"),
            alpha=0.5, dataset=config["dataset"])
        visualize_segmentation(snapshot_np, snapshot_pred_np, 
            save_path=os.path.join(save_q_dir, "adv_seg_only.png"),
            alpha=1.0, dataset=config["dataset"])
        
        # 일찍 끝난 이미지: 이후 정규 snapshot query에도 final 상태의 메트릭을 기록
        snapshot_interval_val = config.get("snapshot_interval", 200)
        max_query_limit = config['max_query']
        recorded_queries = set(entry['query'] for entry in query_history)
        
        for q in range(0, max_query_limit + 1, snapshot_interval_val):
            if q not in recorded_queries:
                query_history.append({
                    'query': q,
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
    
    if 'last_success_query' in snapshots and snapshots['last_success_query'] is not None:
        last_succ = snapshots['last_success_query']
        if last_succ <= max_query_limit:
            attack_success = True
            attack_success_query = last_succ + init_nqry

    image_result = {
        'filename': filename,
        'total_query': total_nquery,
        'l0': int(l0_norm),
        'pixel_ratio': float(pixel_ratio),
        'impact': float(impact),
        'success_ratio': float(success_ratio),
        'foreground_pixels': foreground_pixels,
        'success_pixel': success_pixel,
        'unsuccess_pixel': unsuccess_pixel,
        'attack_mode': config['attack_mode'],
        'max_query': config['max_query'],
        'success_threshold': config['success_threshold'],
        'attack_success': attack_success,
        'attack_success_query': attack_success_query,
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
        'adv_img': adv_img_np,
        'l0': l0_norm,
        'ratio': pixel_ratio,
        'impact': impact,
        'queries': total_nquery,
        'distance_history': D,
        'success_ratio': success_ratio,
        'attack_success': attack_success,
        'attack_success_query': attack_success_query,
        'query_history': query_history,
        'snapshots': snapshots
    }


def main(config, model_config):
    set_seed(42)
    device = config["device"]

    # Load dataset
    data_dir = config["data_dir"]

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
    base_dir = os.path.join(config["base_dir"], current_time)
    os.makedirs(base_dir, exist_ok=True)

    # 처리 데이터 준비
    process_args = []
    for idx, (img_bgr, filename, gt) in enumerate(zip(dataset.images, dataset.filenames, dataset.gt_images)):
        process_args.append((img_bgr, filename, gt, model_config, config, base_dir, idx, len(dataset.images)))

    # 모델 초기화 (메트릭 계산용)
    model = init_robust_model(model_config, device)

    # 순차 처리
    print(f"\nProcessing {len(process_args)} images with PointWise Attack...")
    results = []
    
    img_list = []
    gt_list = []
    filename_list = []
    
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
    
    attack_success_list = []
    attack_success_queries = []

    for args in tqdm(process_args, desc="PointWise Attack"):
        result = process_single_image(args)
        results.append(result)
        
        img_list.append(result['img_bgr'])
        gt_list.append(result['gt'])
        filename_list.append(result['filename'])
        
        snapshots = result['snapshots']
        for i in range(levels):
            q = i * snapshot_interval
            target_img = None
            if q in snapshots:
                target_img = snapshots[q]
            elif 'final' in snapshots:
                target_img = snapshots['final']
            else:
                available_keys = [k for k in snapshots.keys() if isinstance(k, int)]
                if available_keys:
                    target_img = snapshots[max(available_keys)]
                else:
                    target_img = result['adv_img'].transpose(2, 0, 1)
            
            adv_img_lists[i].append(np.transpose(target_img, (1, 2, 0)).astype(np.uint8))
        
        if 'final' in snapshots:
            final_img = snapshots['final']
        else:
            available_keys = [k for k in snapshots.keys() if isinstance(k, int)]
            if available_keys:
                final_img = snapshots[max(available_keys)]
            else:
                final_img = result['adv_img'].transpose(2, 0, 1)
        final_adv_img_list.append(np.transpose(final_img, (1, 2, 0)).astype(np.uint8))
        
        l0_metrics.append(result['l0'])
        ratio_metrics.append(result['ratio'])
        impact_metrics.append(result['impact'])
        query_metrics.append(result['queries'])
        success_ratio_metrics.append(result['success_ratio'])
        all_query_histories.append(result['query_history'])
        
        attack_success_list.append(result['attack_success'])
        if result['attack_success'] and result['attack_success_query'] is not None:
            attack_success_queries.append(result['attack_success_query'])

    # 쿼리별 평균 메트릭 계산 (정규 snapshot query만)
    max_query_limit = config['max_query']
    regular_queries = set(i * snapshot_interval for i in range(levels))
    query_step_averages = {}
    for qh in all_query_histories:
        for entry in qh:
            q = entry['query']
            if q > max_query_limit:
                continue
            if q not in regular_queries:
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
            
    # Process Final Results
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

    # Extract lists
    valid_queries = [entry for entry in per_query_avg if entry['query'] in regular_queries and entry['query'] <= config["max_query"]]
    
    l0_list = [entry['avg_l0'] for entry in valid_queries]
    l0_list.append(np.mean(l0_metrics))
    
    ratio_list = [entry['avg_pixel_ratio'] for entry in valid_queries]
    ratio_list.append(np.mean(ratio_metrics))
    
    impact_list = [entry['avg_impact'] for entry in valid_queries]
    impact_list.append(np.mean(impact_metrics))

    # 공격 성공 통계
    attack_success_count = sum(attack_success_list)
    attack_success_rate = attack_success_count / len(attack_success_list) if len(attack_success_list) > 0 else 0.0
    avg_attack_success_query = np.mean(attack_success_queries) if len(attack_success_queries) > 0 else None
    
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
        "NPix": config.get("npix", 196),
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
    parser = argparse.ArgumentParser(description="Run PointWise attack evaluation using Robust models.")
    parser.add_argument("--config", type=str, required=True, 
                        choices=["pspnet_sat_voc", "pspnet_sat_city", "pspnet_vanilla_voc", "pspnet_vanilla_city",
                               "pspnet_ddcat_voc", "pspnet_ddcat_city",
                               "deeplabv3_sat_voc", "deeplabv3_sat_city", "deeplabv3_vanilla_voc", "deeplabv3_vanilla_city",
                               "deeplabv3_ddcat_voc", "deeplabv3_ddcat_city"],
                        help="Config file to use (without .py extension).")
    parser.add_argument('--device', type=str, default='cuda', help='Device to use.')
    parser.add_argument('--max_query', type=int, default=1000, help='Maximum queries for attack.')
    parser.add_argument('--num_images', type=int, default=100, help='Number of images to evaluate.')
    parser.add_argument('--attack_mode', type=str, default='scheduling', 
                        choices=['single', 'multiple', 'scheduling'],
                        help='Attack mode: single, multiple, or scheduling.')
    parser.add_argument('--npix', type=float, default=0.1, help='Pixels per group. If >= 1: fixed count. If < 1: ratio of image pixels.')
    parser.add_argument('--success_threshold', type=float, default=0.1, 
                        help='Threshold for attack success (ratio of changed pixels).')
    parser.add_argument('--init_mode', type=str, default='random',
                        choices=['salt_pepper', 'random'],
                        help='Starting point initialization mode.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output.')
    parser.add_argument('--include_bg', action='store_true',
                        help='Include background class in attack (do not exclude label 0 for VOC2012).')
    parser.add_argument('--data_dir', type=str, default=None, help='Path to dataset directory (overrides config file if specified).')
    args = parser.parse_args()

    # Import config from the specified config file
    import importlib.util
    config_path = f"configs/{args.config}.py"
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    model_config = config_module.config.copy()
    
    # Create runtime config with attack parameters
    config = {
        "dataset": model_config["dataset"],
        "num_class": model_config["num_class"],
        "device": args.device,
        "data_dir": args.data_dir if args.data_dir is not None else model_config["data_dir"],
        "attack_method": "PointWise",
        "model": model_config["model"],
        "max_query": args.max_query,
        "num_images": args.num_images,
        "attack_mode": args.attack_mode,
        "npix": args.npix,
        "success_threshold": args.success_threshold,
        "init_mode": args.init_mode,
        "seed": args.seed,
        "verbose": args.verbose,
        "include_bg": args.include_bg,
    }
    
    # Copy model-specific parameters
    config.update(model_config)
    if args.data_dir is not None:
        config["data_dir"] = args.data_dir
    config["base_dir"] = f"../data/PointWise/results/{config['dataset']}/{config['model']}"

    main(config, model_config)
