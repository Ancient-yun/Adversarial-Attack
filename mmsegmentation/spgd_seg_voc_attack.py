import argparse
import datetime
import os

import numpy as np
import torch
from tqdm import tqdm

from sPGD.sPGD_seg import (
    DATASET_CONFIGS,
    SegSparsePGD,
    compute_miou_from_predictions,
    load_seg_dataset,
    load_seg_model,
    save_checkpoint,
)
from utils import seed_all


def str2bool(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in ("1", "true", "yes", "y", "t")


def save_attack_diagnostics(diagnostics, save_path, args):
    total = len(diagnostics)
    if total == 0:
        return None

    still_robust_count = sum(1 for d in diagnostics if d["robust"] >= 0.5)
    success_count = total - still_robust_count
    unchanged_float_count = sum(1 for d in diagnostics if d["float_changed_pixels"] == 0)
    no_loss_improve_count = sum(
        1 for d in diagnostics if np.isfinite(d["loss_gain"]) and d["loss_gain"] <= 0.0
    )
    no_improve_and_unchanged_count = sum(
        1
        for d in diagnostics
        if np.isfinite(d["loss_gain"]) and d["loss_gain"] <= 0.0 and d["float_changed_pixels"] == 0
    )

    lines = [
        "Attack Diagnostics",
        "========================================",
        f"dataset: {args.dataset}",
        f"model: {args.model}",
        f"max_iter: {args.max_iter}",
        f"attack_pixel: {args.attack_pixel}",
        f"unprojected_gradient: {args.unprojected_gradient}",
        f"total_samples: {total}",
        f"success_count(robust=0): {success_count}",
        f"still_robust_count(robust=1): {still_robust_count}",
        f"unchanged_float_count: {unchanged_float_count}",
        f"no_loss_improve_count: {no_loss_improve_count}",
        f"no_improve_and_unchanged_count: {no_improve_and_unchanged_count}",
        "",
        "[Per-sample @ max_iter]",
        "idx\tfilename\trobust\titer\tclean_loss\tbest_loss\tloss_gain\tfloat_changed_pixels",
    ]

    for d in diagnostics:
        clean_loss_str = f"{d['clean_loss']:.6f}" if np.isfinite(d["clean_loss"]) else "nan"
        best_loss_str = f"{d['best_loss']:.6f}" if np.isfinite(d["best_loss"]) else "nan"
        loss_gain_str = f"{d['loss_gain']:.6f}" if np.isfinite(d["loss_gain"]) else "nan"
        lines.append(
            (
                f"{d['idx']}\t{d['filename']}\t{int(d['robust'])}\t{d['iter']}\t"
                f"{clean_loss_str}\t{best_loss_str}\t{loss_gain_str}\t{d['float_changed_pixels']}"
            )
        )

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return {
        "total_samples": total,
        "success_count": success_count,
        "still_robust_count": still_robust_count,
        "unchanged_float_count": unchanged_float_count,
        "no_loss_improve_count": no_loss_improve_count,
        "no_improve_and_unchanged_count": no_improve_and_unchanged_count,
    }


def print_run_config(config):
    print("Run Config")
    print("========================================")
    for key in sorted(config.keys()):
        print(f"{key}: {config[key]}")
    print("========================================")


def run_experiment(args):
    seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.save_interval <= 0:
        raise ValueError(f"save_interval must be > 0, got {args.save_interval}")
    if args.max_iter <= 0:
        raise ValueError(f"max_iter must be > 0, got {args.max_iter}")
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    dataset_cfg = DATASET_CONFIGS[args.dataset]
    data_dir = args.data_dir if args.data_dir is not None else dataset_cfg["default_data_dir"]
    base_dir_root = args.base_dir if args.base_dir is not None else dataset_cfg["default_base_dir"]

    model = load_seg_model(args.dataset, args.model, device)
    from mmseg.apis import inference_model
    dataset, original_images, x_nat_list, y_nat_list = load_seg_dataset(args.dataset, data_dir, use_gt=True)

    attacker = SegSparsePGD(
        model=model,
        epsilon=args.epsilon,
        k=args.attack_pixel,
        t=args.max_iter,
        random_start=args.random_start,
        patience=args.patience,
        alpha=args.alpha,
        beta=args.beta,
        unprojected_gradient=args.unprojected_gradient,
        verbose=args.verbose,
        verbose_interval=args.verbose_interval,
        early_stop=args.early_stop,
        attack_mode=args.attack_mode,
    )
    attacker.configure_segmentation(
        ignore_index=args.ignore_index,
        acc_threshold=args.correct_threshold,
        forward_mode=args.forward_mode,
        bounds=(0.0, 255.0),
        enable_constraints_check=args.enable_constraints_check,
    )

    start_time = datetime.datetime.now()
    start_timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    unproj_tag = "true" if args.unprojected_gradient else "false"

    base_dir = os.path.join(
        base_dir_root,
        (
            f"{start_timestamp}_bound_{args.max_iter}_use_gt_True_reward_type_L0_sPGD_"
            f"model_{args.model}_l0_ratio_{args.attack_pixel}_unprojected_{unproj_tag}"
        ),
    )

    config = {
        "task": "segmentation",
        "dataset": args.dataset,
        "model": args.model,
        "data_dir": data_dir,
        "base_dir": base_dir_root,
        "output_dir": base_dir,
        "input_value_range": "0..255",
        "num_class": dataset_cfg["num_class"],
        "device": str(device),
        "type_attack": "L0-sPGD",
        "attack_pixel": args.attack_pixel,
        "unprojected_gradient": bool(args.unprojected_gradient),
        "max_iter": args.max_iter,
        "save_interval": args.save_interval,
    }
    print_run_config(config)

    save_points = list(range(args.save_interval, args.max_iter + 1, args.save_interval))
    if len(save_points) == 0 or save_points[-1] != args.max_iter:
        save_points.append(args.max_iter)

    attacker.t = args.max_iter
    step_adv_images = {s: [] for s in save_points}
    step_robust_flags = {s: [] for s in save_points}
    step_iter_used = {s: [] for s in save_points}
    step_loss_values = {s: [] for s in save_points}
    final_step_diagnostics = []

    for idx, (x_nat, y_nat) in enumerate(
        tqdm(zip(x_nat_list, y_nat_list), total=len(x_nat_list), desc=f"sPGD-{args.dataset}-{args.model}-t{args.max_iter}")
    ):
        x = torch.from_numpy(np.transpose(x_nat, (2, 0, 1))).unsqueeze(0).to(device=device, dtype=torch.float32)
        y = torch.from_numpy(y_nat).unsqueeze(0).to(device=device, dtype=torch.long)

        sample_seed = args.seed + idx
        _, _, _, checkpoint_records = attacker(x, y, seed=sample_seed, checkpoint_steps=save_points)

        for step_t in save_points:
            ckpt = checkpoint_records[step_t]
            x_adv_t = ckpt["x_adv"]
            robust_t = ckpt["robust"]
            it_t = ckpt["it"]

            loss_t = ckpt.get("loss", None)
            if loss_t is None:
                with torch.no_grad():
                    logits_adv = attacker._forward_logits(x_adv_t)
                    loss_t, _ = attacker.loss_fn(logits_adv, y, targeted=False, target=None)

            adv = x_adv_t.squeeze(0).detach().permute(1, 2, 0).cpu().numpy()
            adv_clipped = np.clip(adv, 0.0, 255.0)
            step_adv_images[step_t].append(adv_clipped)
            step_robust_flags[step_t].append(float(robust_t.item()))
            step_iter_used[step_t].append(int(it_t.item()))
            step_loss_values[step_t].append(float(loss_t.mean().item()))

            if args.save_diagnostics and step_t == args.max_iter:
                clean_loss_t = ckpt.get("clean_loss", None)
                clean_loss_scalar = (
                    float(clean_loss_t.reshape(-1)[0].item())
                    if clean_loss_t is not None
                    else float("nan")
                )
                best_loss_scalar = float(loss_t.reshape(-1)[0].item())
                loss_gain = (
                    best_loss_scalar - clean_loss_scalar
                    if np.isfinite(clean_loss_scalar)
                    else float("nan")
                )
                float_changed_pixels = int(np.any(np.abs(adv_clipped - x_nat) > 1e-12, axis=2).sum())
                final_step_diagnostics.append(
                    {
                        "idx": idx,
                        "filename": dataset.filenames[idx],
                        "robust": float(robust_t.item()),
                        "iter": int(it_t.item()),
                        "clean_loss": clean_loss_scalar,
                        "best_loss": best_loss_scalar,
                        "loss_gain": loss_gain,
                        "float_changed_pixels": float_changed_pixels,
                    }
                )

    if args.save_diagnostics:
        os.makedirs(base_dir, exist_ok=True)
        diagnostics_path = os.path.join(base_dir, "attack_diagnostics.txt")
        summary = save_attack_diagnostics(final_step_diagnostics, diagnostics_path, args)
        if summary is not None:
            print(
                f"[{args.dataset}/{args.model}] diagnostics: still_robust={summary['still_robust_count']}/"
                f"{summary['total_samples']}, no_loss_improve={summary['no_loss_improve_count']}, "
                f"unchanged_float={summary['unchanged_float_count']}"
            )
            print(f"[{args.dataset}/{args.model}] saved diagnostics: {diagnostics_path}")

    print(f"[{args.dataset}/{args.model}] caching benign predictions once for checkpoint reuse...")
    ori_preds_cache = []
    for image in tqdm(original_images, total=len(original_images), desc=f"benign-cache-{args.dataset}-{args.model}"):
        with torch.no_grad():
            ori_result = inference_model(model, image)
        ori_pred = ori_result.pred_sem_seg.data.squeeze().cpu().numpy().astype(np.uint8)
        ori_preds_cache.append(ori_pred)

    import evaluate

    miou_metric = evaluate.load("mean_iou")
    benign_miou_cache = compute_miou_from_predictions(ori_preds_cache, dataset.gt_images, config, metric=miou_metric)

    for step_t in save_points:
        robust_acc = float(np.mean(step_robust_flags[step_t])) if step_robust_flags[step_t] else 0.0
        mean_iter = float(np.mean(step_iter_used[step_t])) if step_iter_used[step_t] else 0.0
        mean_loss = float(np.mean(step_loss_values[step_t])) if step_loss_values[step_t] else 0.0
        print(
            f"[{args.dataset}/{args.model}] t={step_t}, robust_acc={robust_acc:.6f}, "
            f"mean_iter={mean_iter:.2f}, mean_loss={mean_loss:.6f}"
        )

        save_checkpoint(
            model=model,
            config=config,
            dataset=dataset,
            original_images=original_images,
            adv_images_float=step_adv_images[step_t],
            iteration=step_t,
            start_time=start_time,
            start_timestamp=start_timestamp,
            base_dir=base_dir,
            max_iter=args.max_iter,
            ori_preds_cache=ori_preds_cache,
            benign_miou_cache=benign_miou_cache,
            miou_metric=miou_metric,
        )
        print(f"[{args.dataset}/{args.model}] saved checkpoint at t={step_t}: {base_dir}")

    print(f"sPGD attack completed. Results saved under: {base_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Segmentation L0-sPGD attack runner")
    parser.add_argument("--dataset", type=str, default="VOC2012", choices=sorted(DATASET_CONFIGS.keys()))
    parser.add_argument("--model", type=str, default="deeplabv3", choices=["deeplabv3", "pspnet"])
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--max_iter", type=int, default=1000, help="sPGD iterations (t)")
    parser.add_argument("--save_interval", type=int, default=200, help="metric save interval")
    parser.add_argument("--attack_pixel", type=float, default=1e-2, help="L0 ratio budget per image")
    parser.add_argument("--epsilon", type=float, default=255.0)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--attack_mode", type=str, default="pixel", choices=["pixel", "feature"])

    parser.add_argument("--correct_threshold", type=float, default=0.0)
    parser.add_argument("--ignore_index", type=int, default=255)
    parser.add_argument("--forward_mode", type=str, default="auto", choices=["auto", "model", "mmseg"])
    parser.add_argument("--enable_constraints_check", type=str2bool, default=False)
    parser.add_argument("--random_start", type=str2bool, default=True)
    parser.add_argument("--unprojected_gradient", type=str2bool, default=False)
    parser.add_argument("--early_stop", type=str2bool, default=False)
    parser.add_argument("--verbose", type=str2bool, default=False)
    parser.add_argument("--verbose_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--save_diagnostics", type=str2bool, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
