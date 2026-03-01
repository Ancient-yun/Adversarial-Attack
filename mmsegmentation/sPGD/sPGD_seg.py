import math
import os
import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .mask import MaskingA, MaskingB
except ImportError:
    from mask import MaskingA, MaskingB


MODEL_CONFIGS = {
    "VOC2012": {
        "deeplabv3": {
            "config": "configs/deeplabv3/deeplabv3_r101-d8_4xb4-20k_voc12aug-512x512.py",
            "checkpoint": "../ckpt/deeplabv3_r101-d8_512x512_20k_voc12aug_20200617_010932-8d13832f.pth",
        },
        "pspnet": {
            "config": "configs/pspnet/pspnet_r101-d8_4xb4-20k_voc12aug-512x512.py",
            "checkpoint": "../ckpt/pspnet_r101-d8_512x512_20k_voc12aug_20200617_102003-4aef3c9a.pth",
        },
    },
    "cityscapes": {
        "deeplabv3": {
            "config": "configs/deeplabv3/deeplabv3_r101-d8_4xb2-80k_cityscapes-512x1024.py",
            "checkpoint": "../ckpt/deeplabv3_r101-d8_512x1024_80k_cityscapes_20200606_113503-9e428899.pth",
        },
        "pspnet": {
            "config": "configs/pspnet/pspnet_r101-d8_4xb2-40k_cityscapes-512x1024.py",
            "checkpoint": "../ckpt/pspnet_r101-d8_512x1024_80k_cityscapes_20200606_112211-e1e1100f.pth",
        },
    },
    "ade20k": {
        "deeplabv3": {
            "config": "configs/deeplabv3/deeplabv3_r101-d8_4xb4-160k_ade20k-512x512.py",
            "checkpoint": "../ckpt/deeplabv3_r101-d8_512x512_160k_ade20k_20200615_105816-b1f72b3b.pth",
        },
        "pspnet": {
            "config": "configs/pspnet/pspnet_r101-d8_4xb4-160k_ade20k-512x512.py",
            "checkpoint": "../ckpt/pspnet_r101-d8_512x512_160k_ade20k_20200615_100650-967c316f.pth",
        },
    },
}

DATASET_CONFIGS = {
    "VOC2012": {
        "num_class": 21,
        "default_data_dir": "../datasets/VOC2012",
        "default_base_dir": "./data/sPGD/results/VOC2012/spgd_seg",
    },
    "cityscapes": {
        "num_class": 19,
        "default_data_dir": "../datasets/cityscapes",
        "default_base_dir": "./data/sPGD/results/cityscapes/spgd_seg",
    },
    "ade20k": {
        "num_class": 150,
        "default_data_dir": "../datasets/ade20k",
        "default_base_dir": "./data/sPGD/results/ade20k/spgd_seg",
    },
}


def normalize_dataset_name(dataset_name: str) -> str:
    key = dataset_name.lower()
    if key in ("voc", "voc2012"):
        return "VOC2012"
    if key in ("city", "cityscapes"):
        return "cityscapes"
    if key in ("ade", "ade20k"):
        return "ade20k"
    raise ValueError(
        f"Unsupported dataset: {dataset_name}. Use one of: voc, voc2012, city, cityscapes, ade, ade20k"
    )


def preprocess_gt_for_seg_loss(
    y_gt: np.ndarray,
    dataset_name: str,
    num_class: int,
    ignore_index: int = 255,
) -> Tuple[np.ndarray, int]:
    """
    Convert GT labels into the label space expected by CE loss.
    Returns (processed_gt, num_clamped_out_of_range_labels).
    """
    y = y_gt.astype(np.int64, copy=True)

    # ADE20K configs in this project use reduce_zero_label=True:
    # raw 0 -> ignore, raw 1..150 -> 0..149.
    if dataset_name == "ade20k":
        y[y == 0] = ignore_index
        valid = y != ignore_index
        y[valid] -= 1

    valid = y != ignore_index
    out_of_range = valid & ((y < 0) | (y >= num_class))
    bad_count = int(out_of_range.sum())
    if bad_count > 0:
        y[out_of_range] = ignore_index
    return y, bad_count


def load_seg_model(dataset_name_or_model: str, model_name_or_device, device: Optional[torch.device] = None):
    """
    Load segmentation model/checkpoint.

    Backward compatibility:
    - load_seg_model(model_name, device) -> assumes VOC2012
    - load_seg_model(dataset_name, model_name, device)
    """
    from mmseg.apis import init_model

    if device is None:
        dataset_name = "VOC2012"
        model_name = dataset_name_or_model
        device = model_name_or_device
    else:
        dataset_name = normalize_dataset_name(dataset_name_or_model)
        model_name = model_name_or_device

    if dataset_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Available: {list(MODEL_CONFIGS.keys())}")
    if model_name not in MODEL_CONFIGS[dataset_name]:
        raise ValueError(
            f"Unsupported model: {model_name} for dataset {dataset_name}. "
            f"Available: {list(MODEL_CONFIGS[dataset_name].keys())}"
        )

    model_cfg = MODEL_CONFIGS[dataset_name][model_name]
    model = init_model(model_cfg["config"], None, device=str(device))
    checkpoint = torch.load(model_cfg["checkpoint"], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def load_voc_model(model_name: str, device: torch.device):
    """Backward-compatible VOC-specific loader."""
    return load_seg_model("VOC2012", model_name, device)


def load_seg_dataset(dataset_name: str, data_dir: str, use_gt: bool = True):
    from dataset import ADESet, CitySet, VOCSet

    dataset_name = normalize_dataset_name(dataset_name)

    dataset_map = {
        "VOC2012": VOCSet,
        "cityscapes": CitySet,
        "ade20k": ADESet,
    }
    if dataset_name not in dataset_map:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Available: {list(dataset_map.keys())}")

    dataset = dataset_map[dataset_name](dataset_dir=data_dir, use_gt=use_gt)
    original_images = [img.copy() for img in dataset.images]
    x_nat_list = [img.astype(np.float32) / 255.0 for img in original_images]
    num_class = DATASET_CONFIGS[dataset_name]["num_class"]
    y_nat_list = []
    total_bad_labels = 0
    for gt in dataset.gt_images:
        y_proc, bad_count = preprocess_gt_for_seg_loss(gt, dataset_name=dataset_name, num_class=num_class)
        y_nat_list.append(y_proc)
        total_bad_labels += bad_count
    if total_bad_labels > 0:
        print(
            f"[warning] Clamped {total_bad_labels} out-of-range GT labels to ignore_index=255 "
            f"for dataset={dataset_name}"
        )
    return dataset, original_images, x_nat_list, y_nat_list


def load_voc_dataset(data_dir: str, use_gt: bool = True):
    """Backward-compatible VOC-specific dataset loader."""
    return load_seg_dataset("VOC2012", data_dir, use_gt=use_gt)


def _extract_mmseg_logits(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output

    if isinstance(output, dict):
        for key in ("logits", "seg_logits", "out", "pred", "outputs"):
            if key in output:
                return _extract_mmseg_logits(output[key])
        raise TypeError(f"Cannot extract logits from dict keys: {list(output.keys())}")

    if isinstance(output, (list, tuple)):
        if len(output) == 0:
            raise TypeError("Cannot extract logits from empty list/tuple output")
        first = output[0]
        if torch.is_tensor(first):
            if first.dim() == 4:
                return first
            if first.dim() == 3:
                return torch.stack(list(output), dim=0)
        if hasattr(first, "seg_logits"):
            logits_list = []
            for sample in output:
                seg_logits = sample.seg_logits
                seg_logits = seg_logits.data if hasattr(seg_logits, "data") else seg_logits
                if seg_logits.dim() == 4 and seg_logits.size(0) == 1:
                    seg_logits = seg_logits.squeeze(0)
                logits_list.append(seg_logits)
            return torch.stack(logits_list, dim=0)
        raise TypeError("Cannot extract logits from list/tuple output")

    if hasattr(output, "seg_logits"):
        seg_logits = output.seg_logits
        seg_logits = seg_logits.data if hasattr(seg_logits, "data") else seg_logits
        if seg_logits.dim() == 3:
            seg_logits = seg_logits.unsqueeze(0)
        return seg_logits

    raise TypeError(f"Unsupported mmseg output type: {type(output)}")


def build_mmseg_logits(model, x_input: torch.Tensor) -> torch.Tensor:
    """
    Build segmentation logits using mmseg data_preprocessor + inference flow.

    Args:
        x_input: [B,C,H,W] (recommended) or [B,H,W,C] float tensor.
    Returns:
        logits: [B, num_classes, H, W]
    """
    try:
        from mmseg.structures import SegDataSample as _SegDataSample
    except ImportError:
        _SegDataSample = None

    if x_input.dim() != 4:
        raise ValueError(f"x_input must have 4 dims, got {tuple(x_input.shape)}")

    if x_input.size(1) in (1, 3):
        x_nchw = x_input
    elif x_input.size(-1) in (1, 3):
        x_nchw = x_input.permute(0, 3, 1, 2).contiguous()
    else:
        raise ValueError(f"Cannot infer channel dimension from shape: {tuple(x_input.shape)}")

    # Most mmseg configs normalize with mean/std in 0..255 scale.
    # Our attack tensors are often in 0..1, so rescale when needed.
    dp = getattr(model, "data_preprocessor", None)
    dp_mean = getattr(dp, "mean", None) if dp is not None else None
    if torch.is_tensor(dp_mean):
        try:
            mean_scalar = float(dp_mean.detach().float().mean().item())
        except Exception:
            mean_scalar = None
    else:
        mean_scalar = None
    if mean_scalar is not None:
        # Heuristic: mean around ImageNet scale (>10) implies 0..255 input expectation.
        # If current tensor is 0..1 range, lift to 0..255 before data_preprocessor.
        with torch.no_grad():
            x_max = float(x_nchw.detach().amax().item())
        if mean_scalar > 10.0 and x_max <= 1.5:
            x_nchw = x_nchw * 255.0

    n, _, h, w = x_nchw.shape
    data_samples = []
    chw_inputs = []
    for i in range(n):
        chw_inputs.append(x_nchw[i].contiguous())
        if _SegDataSample is not None:
            sample = _SegDataSample()
        else:
            class _FallbackSample:
                def __init__(self):
                    self.metainfo = {}

                def set_metainfo(self, metainfo):
                    self.metainfo = dict(metainfo)

            sample = _FallbackSample()
        sample.set_metainfo(
            {
                "ori_shape": (h, w),
                "img_shape": (h, w),
                "pad_shape": (h, w),
                "padding_size": [0, 0, 0, 0],
            }
        )
        data_samples.append(sample)

    processed = model.data_preprocessor({"inputs": chw_inputs, "data_samples": data_samples}, training=False)
    batch_inputs = processed["inputs"]
    batch_metas = [s.metainfo for s in processed["data_samples"]]
    raw_output = model.inference(batch_inputs, batch_metas)
    logits = _extract_mmseg_logits(raw_output)

    if logits.dim() == 3:
        logits = logits.unsqueeze(0)
    if logits.dim() != 4:
        raise ValueError(f"Expected logits shape [B, C, H, W], got {tuple(logits.shape)}")
    return logits


def compute_miou_from_predictions(
    predictions: List[np.ndarray],
    references: List[np.ndarray],
    config: Dict[str, Any],
    metric=None,
):
    if metric is None:
        import evaluate

        metric = evaluate.load("mean_iou")
    reduce_labels = config["dataset"] == "ade20k"
    return metric.compute(
        predictions=predictions,
        references=references,
        num_labels=config["num_class"],
        ignore_index=255,
        reduce_labels=reduce_labels,
    )


def save_segmentation_checkpoint(
    model,
    config: Dict[str, Any],
    dataset,
    original_images: List[np.ndarray],
    adv_images_float: List[np.ndarray],
    iteration: int,
    start_time: datetime.datetime,
    start_timestamp: str,
    base_dir: str,
    max_iter: int,
    ori_preds_cache: Optional[List[np.ndarray]] = None,
    benign_miou_cache: Optional[Dict[str, Any]] = None,
    miou_metric=None,
) -> Dict[str, Any]:
    """
    Save attack outputs and metrics with pgd0_voc_attack.py-compatible structure.

    Expected inputs:
    - original_images: uint8 HWC(BGR) list
    - adv_images_float: [0,1] float HWC list
    """
    from PIL import Image
    from mmseg.apis import inference_model

    from evaluation import calculate_impact, calculate_l0_norm, calculate_pixel_ratio
    from function import visualize_segmentation
    from utils import save_experiment_results

    save_dir = base_dir if iteration == max_iter else os.path.join(base_dir, f"intermediate_bound_{iteration}")
    adv_dir = os.path.join(save_dir, "adv")
    delta_dir = os.path.join(save_dir, "delta")
    adv_seg_dir = os.path.join(save_dir, "adv_seg")
    ori_seg_dir = os.path.join(save_dir, "ori_seg")
    os.makedirs(adv_dir, exist_ok=True)
    os.makedirs(delta_dir, exist_ok=True)
    os.makedirs(adv_seg_dir, exist_ok=True)
    os.makedirs(ori_seg_dir, exist_ok=True)

    if len(original_images) != len(adv_images_float):
        raise ValueError(
            f"Length mismatch: original_images={len(original_images)}, adv_images_float={len(adv_images_float)}"
        )
    if len(dataset.filenames) != len(original_images):
        raise ValueError(
            f"Length mismatch: dataset.filenames={len(dataset.filenames)}, original_images={len(original_images)}"
        )

    adv_examples = [np.clip((x * 255.0).round(), 0, 255).astype(np.uint8) for x in adv_images_float]
    l0_list: List[int] = []
    ratio_list: List[float] = []
    impact_list: List[float] = []
    adv_predictions: List[np.ndarray] = []
    ori_predictions: Optional[List[np.ndarray]] = [] if ori_preds_cache is None else None

    for i, filename in enumerate(dataset.filenames):
        name = filename.rsplit(".", 1)[0] + ".png"
        adv_path = os.path.join(adv_dir, name)
        delta_path = os.path.join(delta_dir, name)
        adv_seg_path = os.path.join(adv_seg_dir, name)
        ori_seg_path = os.path.join(ori_seg_dir, name)
        os.makedirs(os.path.dirname(adv_path), exist_ok=True)
        os.makedirs(os.path.dirname(delta_path), exist_ok=True)
        os.makedirs(os.path.dirname(adv_seg_path), exist_ok=True)
        os.makedirs(os.path.dirname(ori_seg_path), exist_ok=True)

        original = original_images[i].astype(np.uint8)
        adv_uint8 = adv_examples[i]
        delta = np.abs(original.astype(np.int16) - adv_uint8.astype(np.int16)).astype(np.uint8)

        Image.fromarray(adv_uint8[:, :, ::-1]).save(adv_path, "PNG")
        Image.fromarray(delta).save(delta_path, "PNG")

        adv_result = inference_model(model, adv_uint8)
        adv_pred = adv_result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.uint8)
        if ori_preds_cache is None:
            ori_result = inference_model(model, original)
            ori_pred = ori_result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.uint8)
            ori_predictions.append(ori_pred)
        else:
            ori_pred = ori_preds_cache[i]

        visualize_segmentation(original, ori_pred, ori_seg_path, alpha=0.5, dataset=config["dataset"])
        visualize_segmentation(adv_uint8, adv_pred, adv_seg_path, alpha=0.5, dataset=config["dataset"])

        l0_list.append(int(calculate_l0_norm(original, adv_uint8)))
        ratio_list.append(float(calculate_pixel_ratio(original, adv_uint8)))
        impact_list.append(float(calculate_impact(original, adv_uint8, ori_pred, adv_pred)))
        adv_predictions.append(adv_pred)

    metric = miou_metric
    if metric is None:
        import evaluate

        metric = evaluate.load("mean_iou")

    if benign_miou_cache is None:
        benign_source = ori_predictions if ori_preds_cache is None else ori_preds_cache
        benign_miou_score = compute_miou_from_predictions(benign_source, dataset.gt_images, config, metric=metric)
    else:
        benign_miou_score = benign_miou_cache
    adv_miou_score = compute_miou_from_predictions(adv_predictions, dataset.gt_images, config, metric=metric)

    results = {
        "start_time": start_timestamp,
        "current_bound": iteration,
        "elapsed_time": (datetime.datetime.now() - start_time).total_seconds(),
        "iteration": [iteration] * len(dataset.filenames),
        "iteration_mean": float(iteration),
        "l0": l0_list,
        "l0_mean": float(np.mean(l0_list)),
        "ratio": ratio_list,
        "ratio_mean": float(np.mean(ratio_list)),
        "impact": impact_list,
        "impact_mean": float(np.mean(impact_list)),
        "benign_miou_score": benign_miou_score,
        "adv_miou_score": adv_miou_score,
    }
    if iteration == max_iter:
        results["end_time"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    save_experiment_results(results, config, f"{start_timestamp}", save_dir=save_dir)
    return results


def save_checkpoint(
    model,
    config: Dict[str, Any],
    dataset,
    original_images: List[np.ndarray],
    adv_images_float: List[np.ndarray],
    iteration: int,
    start_time: datetime.datetime,
    start_timestamp: str,
    base_dir: str,
    max_iter: int,
    ori_preds_cache: Optional[List[np.ndarray]] = None,
    benign_miou_cache: Optional[Dict[str, Any]] = None,
    miou_metric=None,
) -> Dict[str, Any]:
    """Backward-compatible alias matching pgd0_voc_attack.py naming."""
    return save_segmentation_checkpoint(
        model=model,
        config=config,
        dataset=dataset,
        original_images=original_images,
        adv_images_float=adv_images_float,
        iteration=iteration,
        start_time=start_time,
        start_timestamp=start_timestamp,
        base_dir=base_dir,
        max_iter=max_iter,
        ori_preds_cache=ori_preds_cache,
        benign_miou_cache=benign_miou_cache,
        miou_metric=miou_metric,
    )


class SegSparsePGD(object):
    """
    SparsePGD variant for semantic segmentation.

    API style is kept compatible with sPGD/sPGD.py:
    - __call__(x, y, seed=-1, targeted=False, target=None)
    - perturb(x, y)
    - change_masking()
    """

    def __init__(
        self,
        model,
        epsilon=255,
        k=10,
        t=30,
        random_start=True,
        patience=3,
        classes=10,
        alpha=0.25,
        beta=0.25,
        unprojected_gradient=False,
        verbose=False,
        verbose_interval=100,
        early_stop=True,
        attack_mode="pixel",
    ):
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.t = t
        self.random_start = random_start
        self.alpha = epsilon * alpha
        self.beta = beta
        self.patience = patience
        self.classes = classes
        self.unprojected_gradient = unprojected_gradient
        self.masking = MaskingA() if self.unprojected_gradient else MaskingB()
        self.weight_decay = 0.0
        self.p_init = 1.0
        self.verbose = verbose
        self.verbose_interval = verbose_interval
        self.early_stop = early_stop
        assert attack_mode in ["pixel", "feature"], "attack_mode shoule be either pixel or feature"
        self.attack_mode = attack_mode

        # Segmentation defaults (kept out of constructor to preserve existing args).
        self.ignore_index = 255
        self.acc_threshold = 0.0
        self.forward_mode = "auto"  # auto | model | mmseg
        self.current_k: Optional[int] = None
        self.bounds_cfg: Optional[Dict[str, Any]] = None
        self.logits_extractor: Optional[Callable[[Any], torch.Tensor]] = None
        self.enable_constraints_check = False
        self._unit_bounds_cache: Dict[Tuple[str, Optional[int], torch.dtype, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        self._auto_use_mmseg_forward: Optional[bool] = None

    def configure_segmentation(
        self,
        ignore_index: Optional[int] = None,
        acc_threshold: Optional[float] = None,
        forward_mode: Optional[str] = None,
        bounds: Optional[Tuple[Any, Any]] = None,
        logits_extractor: Optional[Callable[[Any], torch.Tensor]] = None,
        enable_constraints_check: Optional[bool] = None,
    ):
        """Optional runtime config without changing constructor args."""
        if ignore_index is not None:
            self.ignore_index = int(ignore_index)
        if acc_threshold is not None:
            self.acc_threshold = float(acc_threshold)
        if forward_mode is not None:
            if forward_mode not in ("auto", "model", "mmseg"):
                raise ValueError(f"forward_mode must be one of ['auto', 'model', 'mmseg'], got: {forward_mode}")
            self.forward_mode = forward_mode
            self._auto_use_mmseg_forward = None
        if bounds is not None:
            if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
                raise ValueError("bounds must be (low, high)")
            self.set_bounds(bounds[0], bounds[1])
        if logits_extractor is not None:
            self.logits_extractor = logits_extractor
        if enable_constraints_check is not None:
            self.enable_constraints_check = bool(enable_constraints_check)

    def set_bounds(self, low: Any, high: Any):
        self.bounds_cfg = {"low": low, "high": high}
        self._unit_bounds_cache.clear()

    def _to_bound_tensor(self, bound: Any, x: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(bound):
            t = bound.to(device=x.device, dtype=x.dtype)
        else:
            t = torch.as_tensor(bound, device=x.device, dtype=x.dtype)

        if t.dim() == 0:
            return t.view(1, 1, 1, 1).expand(1, x.size(1), 1, 1)
        if t.dim() == 1:
            if t.numel() != x.size(1):
                raise ValueError(f"Channel bound size mismatch: {t.numel()} vs {x.size(1)}")
            return t.view(1, -1, 1, 1)
        if t.dim() == 3 and t.size(0) == x.size(1):
            return t.unsqueeze(0)
        if t.dim() == 4:
            if t.size(1) != x.size(1):
                raise ValueError(f"Channel bound size mismatch: {t.size(1)} vs {x.size(1)}")
            return t
        raise ValueError(f"Unsupported bound shape: {tuple(t.shape)}")

    def _resolve_bounds(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.bounds_cfg is None:
            key = (x.device.type, x.device.index, x.dtype, x.size(1))
            cached = self._unit_bounds_cache.get(key, None)
            if cached is not None:
                return cached
            low = torch.zeros((1, x.size(1), 1, 1), device=x.device, dtype=x.dtype)
            high = torch.ones((1, x.size(1), 1, 1), device=x.device, dtype=x.dtype)
            self._unit_bounds_cache[key] = (low, high)
            return low, high
        low = self._to_bound_tensor(self.bounds_cfg["low"], x)
        high = self._to_bound_tensor(self.bounds_cfg["high"], x)
        return low, high

    def _clamp_like_input(self, x_adv: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
        low, high = self._resolve_bounds(x_ref)
        return torch.max(torch.min(x_adv, high), low)

    def _clamp_perturbation(self, perturb: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
        low, high = self._resolve_bounds(x_ref)
        perturb = torch.clamp(perturb, min=-self.epsilon, max=self.epsilon)
        perturb = torch.max(torch.min(perturb, high - x_ref), low - x_ref)
        return perturb

    def _extract_logits(self, model_output: Any) -> torch.Tensor:
        if self.logits_extractor is not None:
            logits = self.logits_extractor(model_output)
            if not torch.is_tensor(logits):
                raise TypeError("logits_extractor must return a torch.Tensor")
            return logits

        if torch.is_tensor(model_output):
            return model_output

        if isinstance(model_output, dict):
            for key in ("logits", "seg_logits", "out", "pred", "outputs"):
                if key in model_output:
                    return self._extract_logits(model_output[key])
            raise TypeError(f"Cannot extract logits from dict keys: {list(model_output.keys())}")

        if isinstance(model_output, (list, tuple)):
            if len(model_output) == 0:
                raise TypeError("Cannot extract logits from empty list/tuple")
            first = model_output[0]
            if torch.is_tensor(first):
                if first.dim() == 4:
                    return first
                if len(model_output) == 1 and first.dim() == 3:
                    return first.unsqueeze(0)
            if hasattr(first, "seg_logits"):
                logits_list = []
                for sample in model_output:
                    seg_logits = sample.seg_logits
                    seg_logits = seg_logits.data if hasattr(seg_logits, "data") else seg_logits
                    logits_list.append(seg_logits)
                return torch.stack(logits_list, dim=0)
            raise TypeError("Cannot extract logits from list/tuple output")

        if hasattr(model_output, "seg_logits"):
            seg_logits = model_output.seg_logits
            seg_logits = seg_logits.data if hasattr(seg_logits, "data") else seg_logits
            if seg_logits.dim() == 3:
                seg_logits = seg_logits.unsqueeze(0)
            return seg_logits

        raise TypeError(f"Unsupported model output type for logits extraction: {type(model_output)}")

    def _forward_logits(self, x_adv: torch.Tensor) -> torch.Tensor:
        if self.forward_mode == "mmseg":
            use_mmseg_forward = True
        elif self.forward_mode == "model":
            use_mmseg_forward = False
        else:
            if self._auto_use_mmseg_forward is None:
                self._auto_use_mmseg_forward = hasattr(self.model, "data_preprocessor") and hasattr(
                    self.model, "inference"
                )
            use_mmseg_forward = self._auto_use_mmseg_forward

        if use_mmseg_forward:
            logits = build_mmseg_logits(self.model, x_adv)
        else:
            logits = self._extract_logits(self.model(x_adv))
        if logits.dim() != 4:
            raise ValueError(f"Expected logits shape [B, C, H, W], got {tuple(logits.shape)}")
        return logits

    def _align_logits(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if logits.shape[-2:] != y.shape[-2:]:
            logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
        return logits

    def _pixel_accuracy(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        valid = y != self.ignore_index
        pred = logits.argmax(dim=1)
        correct = ((pred == y) & valid).sum(dim=(1, 2)).float()
        denom = valid.sum(dim=(1, 2)).clamp(min=1).float()
        return correct / denom

    def _robust_indicator(self, pixel_acc: torch.Tensor, targeted: bool) -> torch.Tensor:
        if targeted:
            return (pixel_acc < self.acc_threshold).float()
        # Untargeted success is pixel_acc <= acc_threshold, so robust means strictly greater.
        return (pixel_acc > self.acc_threshold).float()

    def loss_fn(
        self,
        logits: torch.Tensor,
        y: torch.Tensor,
        targeted: bool = False,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = target if targeted else y
        if labels is None:
            raise ValueError("target labels are required when targeted=True")

        logits = self._align_logits(logits, labels)
        if targeted:
            loss_map = -F.cross_entropy(logits, labels, reduction="none", ignore_index=self.ignore_index)
        else:
            loss_map = F.cross_entropy(logits, labels, reduction="none", ignore_index=self.ignore_index)

        valid = (labels != self.ignore_index).float()
        valid_count = valid.sum(dim=(1, 2)).clamp(min=1.0)
        loss = (loss_map * valid).sum(dim=(1, 2)) / valid_count
        pixel_acc = self._pixel_accuracy(logits, labels)
        return loss, pixel_acc

    def _mask_l0(self, proj_mask: torch.Tensor) -> torch.Tensor:
        if self.attack_mode == "pixel":
            return (proj_mask.sum(dim=1) > 0).flatten(1).sum(dim=1)
        return proj_mask.flatten(1).norm(p=0, dim=1)

    def _resolve_k(self, c: int, h: int, w: int) -> int:
        if self.attack_mode == "pixel":
            budget_space = h * w
        else:
            budget_space = c * h * w

        k_raw = float(self.k)
        if k_raw <= 0:
            raise ValueError(f"k must be positive, got {self.k}")

        if k_raw <= 1.0:
            k_eff = int(budget_space * k_raw)
            k_eff = max(1, k_eff)
        else:
            k_eff = int(k_raw)
            k_eff = max(1, k_eff)

        k_eff = min(k_eff, budget_space)
        return k_eff

    def _assert_constraints(self, x_ref: torch.Tensor, x_adv: torch.Tensor, proj_mask: torch.Tensor, k: int):
        l0 = self._mask_l0(proj_mask)
        if torch.any(l0 > k):
            raise AssertionError("projection error: L0(mask) exceeds k")
        low, high = self._resolve_bounds(x_ref)
        if torch.any(x_adv < (low - 1e-6)) or torch.any(x_adv > (high + 1e-6)):
            raise AssertionError("x_adv violates input bounds")

    def initial_perturb(self, x, seed=-1):
        if self.random_start:
            if seed != -1:
                torch.random.manual_seed(seed)
            perturb = x.new(x.size()).uniform_(-self.epsilon, self.epsilon)
        else:
            perturb = x.new(x.size()).zero_()
        perturb = self._clamp_perturbation(perturb, x)
        return perturb

    def update_perturbation(self, perturb, grad, x, low_conf_idx=None):
        if low_conf_idx is None or low_conf_idx.numel() == 0:
            perturb1 = perturb + self.alpha * grad.sign()
            return self._clamp_perturbation(perturb1, x)

        b, _, _, _ = perturb.size()
        step_size = torch.full((b,), self.alpha, device=perturb.device, dtype=perturb.dtype)
        step_size[low_conf_idx] = 0.1 * step_size[low_conf_idx]
        perturb1 = perturb + step_size.view(b, 1, 1, 1) * grad.sign()
        perturb1 = self._clamp_perturbation(perturb1, x)
        return perturb1

    def update_mask(self, mask, grad, low_conf_idx=None):
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        b, c, h, w = mask.size()
        grad_norm = torch.norm(grad, p=2, dim=(1, 2, 3), keepdim=True)
        d = grad / (grad_norm + 1e-10)
        base_step = math.sqrt(h * w * c) * self.beta
        zero_grad_idx = (grad_norm.view(-1) < 2e-10).nonzero(as_tuple=False).flatten()
        if zero_grad_idx.numel() == 0 and (low_conf_idx is None or low_conf_idx.numel() == 0):
            return mask + base_step * d

        step_size = torch.full((b,), base_step, device=mask.device, dtype=mask.dtype)
        if zero_grad_idx.numel() > 0:
            step_size[zero_grad_idx] = 0
        if low_conf_idx is not None and low_conf_idx.numel() > 0:
            step_size[low_conf_idx] = 0.1 * base_step
        mask = mask + step_size.view(b, 1, 1, 1) * d

        return mask

    def initial_mask(self, x, it=0, prev_mask=None, seed=-1):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        b, c, h, w = x.size()

        if seed != -1:
            torch.random.manual_seed(seed)
        if self.attack_mode == "pixel":
            mask = torch.randn(b, 1, h, w, device=x.device)
        else:
            mask = torch.randn(b, c, h, w, device=x.device)

        return mask

    def project_mask(self, mask, k: Optional[int] = None):
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        b, c, h, w = mask.size()
        if k is None:
            k = self._resolve_k(c, h, w)
        flat = mask.view(b, -1)
        k_eff = min(max(1, int(k)), flat.size(1))
        idx = torch.topk(flat, k=k_eff, dim=1, largest=True, sorted=False).indices
        mask_proj = torch.zeros_like(flat).scatter_(1, idx, 1).view(b, c, h, w)
        return mask_proj

    def check_shape(self, x):
        return x if len(x.shape) == 4 else x.unsqueeze(0)

    def __call__(self, x, y, seed=-1, targeted=False, target=None, checkpoint_steps: Optional[List[int]] = None):
        if x.dim() != 4:
            raise ValueError(f"x must have shape [B,C,H,W], got {tuple(x.shape)}")
        if y.dim() != 3:
            raise ValueError(f"y must have shape [B,H,W], got {tuple(y.shape)}")
        y = y.long()
        if target is not None:
            target = target.long()

        checkpoint_steps_sorted: List[int] = []
        checkpoint_steps_set = set()
        checkpoint_records: Dict[int, Dict[str, torch.Tensor]] = {}
        if checkpoint_steps is not None:
            for s in checkpoint_steps:
                si = int(s)
                if 1 <= si <= self.t:
                    checkpoint_steps_set.add(si)
            checkpoint_steps_sorted = sorted(checkpoint_steps_set)

        b, _, _, _ = x.size()
        _, c, h, w = x.size()
        current_k = self._resolve_k(c, h, w)
        self.current_k = current_k
        it = torch.zeros(b, dtype=torch.long, device=x.device)

        training = self.model.training
        if training:
            self.model.eval()

        with torch.no_grad():
            clean_logits = self._forward_logits(x)
            clean_loss, clean_pixel_acc = self.loss_fn(clean_logits, y, targeted, target)
            robust_flags = self._robust_indicator(clean_pixel_acc, targeted)

        if self.t == 0:
            if training:
                self.model.train()
            if checkpoint_steps is not None:
                return x.clone(), robust_flags, it, checkpoint_records
            return x.clone(), robust_flags, it

        perturb = self.initial_perturb(x, seed)
        mask = self.initial_mask(x, seed=seed)

        mask_best = mask.clone()
        perturb_best = perturb.clone()
        x_adv_best = x.clone()
        loss_best = clean_loss.detach().clone()

        ind_all = torch.arange(b, device=x.device)
        reinitial_count = torch.zeros(b, dtype=torch.long, device=x.device)

        if self.early_stop:
            ind_fail = (robust_flags == 1).nonzero(as_tuple=False).flatten()
            if ind_fail.numel() == 0:
                if training:
                    self.model.train()
                if checkpoint_steps_set:
                    final_record = {
                        "x_adv": x.detach().clone(),
                        "robust": robust_flags.detach().clone(),
                        "it": it.detach().clone(),
                        "loss": clean_loss.detach().clone(),
                        "clean_loss": clean_loss.detach().clone(),
                    }
                    for step in checkpoint_steps_sorted:
                        checkpoint_records[step] = {
                            "x_adv": final_record["x_adv"].clone(),
                            "robust": final_record["robust"].clone(),
                            "it": final_record["it"].clone(),
                            "loss": final_record["loss"].clone(),
                            "clean_loss": final_record["clean_loss"].clone(),
                        }
                    return x.clone(), robust_flags, it, checkpoint_records
                return x.clone(), robust_flags, it

            x = x[ind_fail]
            perturb = perturb[ind_fail]
            mask = mask[ind_fail]
            y = y[ind_fail]
            ind_all = ind_all[ind_fail]
            reinitial_count = reinitial_count[ind_fail]
            if target is not None:
                target = target[ind_fail]

        if self.verbose:
            acc_list = []
            ind_fail_list = []

        perturb.requires_grad_()
        mask.requires_grad_()
        proj_perturb, proj_mask = self.masking.apply(perturb, torch.sigmoid(mask), current_k)
        x_adv = self._clamp_like_input(x + proj_perturb, x)
        if self.enable_constraints_check:
            with torch.no_grad():
                self._assert_constraints(x, x_adv, proj_mask, current_k)
        logits = self._forward_logits(x_adv)
        loss, pixel_acc = self.loss_fn(logits, y, targeted, target)

        grad_perturb, grad_mask = torch.autograd.grad(loss.sum(), (perturb, mask), retain_graph=False, create_graph=False)
        grad_perturb = grad_perturb.detach()
        grad_mask = grad_mask.detach()

        for i in range(self.t):
            it[ind_all] += 1

            perturb = perturb.detach()
            mask = mask.detach()

            prev_mask = mask
            mask = self.update_mask(mask, grad_mask)
            perturb = self.update_perturbation(perturb=perturb, grad=grad_perturb, x=x)

            perturb.requires_grad_()
            mask.requires_grad_()
            proj_perturb, proj_mask = self.masking.apply(perturb, torch.sigmoid(mask), current_k)
            x_adv = self._clamp_like_input(x + proj_perturb, x)
            if self.enable_constraints_check:
                with torch.no_grad():
                    self._assert_constraints(x, x_adv, proj_mask, current_k)

            logits = self._forward_logits(x_adv)
            loss, pixel_acc = self.loss_fn(logits, y, targeted, target)

            grad_perturb, grad_mask = torch.autograd.grad(
                loss.sum(), (perturb, mask), retain_graph=False, create_graph=False
            )
            grad_perturb = grad_perturb.detach()
            grad_mask = grad_mask.detach()

            with torch.no_grad():
                acc = self._robust_indicator(pixel_acc.detach(), targeted)
                robust_flags[ind_all] = acc

                loss_det = loss.detach()
                loss_improve_idx = (loss_det >= loss_best[ind_all]).nonzero(as_tuple=False).flatten()
                if loss_improve_idx.numel() > 0:
                    global_idx = ind_all[loss_improve_idx]
                    loss_best[global_idx] = loss_det[loss_improve_idx]
                    x_adv_best[global_idx] = x_adv[loss_improve_idx].detach()
                    mask_best[global_idx] = mask[loss_improve_idx].detach()
                    perturb_best[global_idx] = perturb[loss_improve_idx].detach()

                ind_success = (acc == 0).nonzero(as_tuple=False).flatten()
                if ind_success.numel() > 0:
                    global_success = ind_all[ind_success]
                    x_adv_best[global_success] = x_adv[ind_success].detach()
                    loss_best[global_success] = loss_det[ind_success]

            ind_fail = (acc == 1).nonzero(as_tuple=False).flatten()
            if ind_fail.numel() > 0:
                with torch.no_grad():
                    curr_flat = mask[ind_fail].reshape(ind_fail.numel(), -1)
                    prev_flat = prev_mask[ind_fail].reshape(ind_fail.numel(), -1)
                    k_eff = min(max(1, int(current_k)), curr_flat.size(1))
                    curr_topk = torch.topk(curr_flat, k=k_eff, dim=1, largest=True, sorted=False).indices
                    prev_topk = torch.topk(prev_flat, k=k_eff, dim=1, largest=True, sorted=False).indices
                    curr_topk_sorted = curr_topk.sort(dim=1).values
                    prev_topk_sorted = prev_topk.sort(dim=1).values
                    delta_mask_norm = (curr_topk_sorted != prev_topk_sorted).any(dim=1).long()
                reinitial_count[ind_fail] = 0
                ind_unchange = (delta_mask_norm <= 0).nonzero(as_tuple=False).flatten()
                if ind_unchange.numel() > 0:
                    reinitial_count[ind_fail[ind_unchange]] += 1

                ind_reinit = (reinitial_count >= self.patience).nonzero(as_tuple=False).flatten()
                if ind_reinit.numel() > 0:
                    with torch.no_grad():
                        mask[ind_reinit] = self.initial_mask(x[ind_reinit])
                    reinitial_count[ind_reinit] = 0
                    grad_perturb[ind_reinit] = 0
                    grad_mask[ind_reinit] = 0

                if self.early_stop:
                    x = self.check_shape(x[ind_fail])
                    perturb = self.check_shape(perturb[ind_fail])
                    mask = self.check_shape(mask[ind_fail])
                    grad_perturb = self.check_shape(grad_perturb[ind_fail])
                    grad_mask = self.check_shape(grad_mask[ind_fail])
                    y = y[ind_fail]
                    ind_all = ind_all[ind_fail]
                    reinitial_count = reinitial_count[ind_fail]
                    if target is not None:
                        target = target[ind_fail]

            if self.verbose and (i + 1) % self.verbose_interval == 0:
                acc_list.append(acc.sum().item())
                ind_fail_list.append(ind_all.clone().detach().cpu().numpy())
                print(
                    f"[SegSparsePGD] it={i + 1}, mean_ce={loss.detach().mean().item():.6f}, "
                    f"pixel_acc={pixel_acc.detach().mean().item():.6f}, "
                    f"active={int(ind_all.numel())}, restarts={int((reinitial_count > 0).sum().item())}"
                )

            current_step = i + 1
            if current_step in checkpoint_steps_set:
                checkpoint_records[current_step] = {
                    "x_adv": x_adv_best.detach().clone(),
                    "robust": robust_flags.detach().clone(),
                    "it": it.detach().clone(),
                    "loss": loss_best.detach().clone(),
                    "clean_loss": clean_loss.detach().clone(),
                }

            if self.early_stop and (ind_fail.numel() == 0 or ind_all.numel() == 0):
                break

        if training:
            self.model.train()

        if checkpoint_steps_set:
            final_record = {
                "x_adv": x_adv_best.detach().clone(),
                "robust": robust_flags.detach().clone(),
                "it": it.detach().clone(),
                "loss": loss_best.detach().clone(),
                "clean_loss": clean_loss.detach().clone(),
            }
            for step in checkpoint_steps_sorted:
                if step not in checkpoint_records:
                    checkpoint_records[step] = {
                        "x_adv": final_record["x_adv"].clone(),
                        "robust": final_record["robust"].clone(),
                        "it": final_record["it"].clone(),
                        "loss": final_record["loss"].clone(),
                        "clean_loss": final_record["clean_loss"].clone(),
                    }
        if self.verbose:
            expected_len = self.t // self.verbose_interval if self.verbose_interval > 0 else 0
            if expected_len > 0:
                if len(acc_list) == 0:
                    acc_list = [0] * expected_len
                    ind_fail_list = [torch.empty(0).cpu().numpy()] * expected_len
                if len(acc_list) != expected_len:
                    acc_list += [acc_list[-1]] * (expected_len - len(acc_list))
                    ind_fail_list += [ind_fail_list[-1]] * (expected_len - len(ind_fail_list))
            if checkpoint_steps is not None:
                return x_adv_best, robust_flags, it, acc_list, ind_fail_list, checkpoint_records
            return x_adv_best, robust_flags, it, acc_list, ind_fail_list
        if checkpoint_steps is not None:
            return x_adv_best, robust_flags, it, checkpoint_records
        return x_adv_best, robust_flags, it

    def perturb(self, x, y):
        if self.verbose:
            x_adv, acc, it, acc_list, ind_fail_list = self.__call__(x, y, targeted=False)
            return x_adv, acc.sum(), it, acc_list, ind_fail_list
        x_adv, acc, it = self.__call__(x, y, targeted=False)
        self.model.zero_grad()
        return x_adv, acc.sum(), it

    def change_masking(self):
        if isinstance(self.masking, MaskingA):
            self.masking = MaskingB()
        else:
            self.masking = MaskingA()


# Compatibility alias so existing import sites can keep class name if needed.
SparsePGD = SegSparsePGD
