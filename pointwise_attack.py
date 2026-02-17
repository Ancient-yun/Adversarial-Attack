# Copyright (c) 2020-present
# PointWise Attack adapted for Semantic Segmentation (mmseg/openvoca/SED)
# Based on the original PointWise Attack implementation

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import numpy as np
import random
import time
import sys
import os

try:
    from mmseg.apis import inference_model
except ImportError:
    inference_model = None

sys.path.append(os.path.join(os.path.dirname(__file__), 'Robust-Semantic-Segmentation'))
from adv_setting import model_predict


def l0(img1, img2):
    """
    Calculate L0 distance (number of differing pixels) between two images.
    Matches the original utils_se.py l0 function.
    
    Args:
        img1: Image tensor (1, C, H, W) or (C, H, W)
        img2: Image tensor (1, C, H, W) or (C, H, W)
        
    Returns:
        int: Number of pixels where any channel differs
    """
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1)
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2)
    
    # Move tensors to cuda
    img1 = img1.cuda()
    img2 = img2.cuda()
    
    # Handle batch dimension
    if len(img1.shape) == 3:
        img1 = img1.unsqueeze(0)
    if len(img2.shape) == 3:
        img2 = img2.unsqueeze(0)
    
    # Calculate absolute difference
    xo = torch.abs(img1 - img2)
    
    # Check if any channel differs for each pixel (original utils_se style)
    d = torch.zeros(xo.shape[2], xo.shape[3]).bool().cuda()
    for i in range(xo.shape[1]):
        tmp = (xo[0, i] > 0.).bool().cuda()
        d = tmp | d  # OR operation
    
    return d.sum().item()


# Alias for compatibility
l0_distance = l0


class PointWiseAttack:
    """
    PointWise Attack adapted for Semantic Segmentation models.
    
    Supports:
    - MMSegmentation models (mmseg)
    - Detectron2 models
    - Standard PyTorch segmentation models
    
    Arguments:
        model: The segmentation model to attack
        cfg: Configuration dictionary for the model
        is_mmseg: Whether the model uses mmseg API
        is_detectron2: Whether the model uses Detectron2 API
        success_threshold: Ratio of changed pixels to consider attack successful
        verbose: Print progress information
    """

    def __init__(
            self,
            model,
            cfg,
            is_mmseg=False,
            is_detectron2=False,
            success_threshold=0.01,
            verbose=False
    ):
        self.model = model
        self.cfg = cfg
        self.is_mmseg = is_mmseg
        self.is_detectron2 = is_detectron2
        self.success_threshold = success_threshold
        self.verbose = verbose
        self.device = torch.device('cuda')
        self.ignore_index = 255  # Default: Cityscapes uses 255, can be changed via set_ignore_index

    def set_ignore_index(self, dataset_name, include_bg=False):
        """Set ignore index based on dataset name.

        Args:
            dataset_name: Name of the dataset.
            include_bg: If True, do not exclude background (ignore_index=None for VOC2012).
        """
        if dataset_name.lower() == 'cityscapes':
            self.ignore_index = 255
        elif dataset_name.lower() in ['voc2012']:
            self.ignore_index = None if include_bg else 0
        elif dataset_name.lower() in ['ade20k']:
            self.ignore_index = 255
        else:
            self.ignore_index = None

    def _get_pred_labels(self, img):
        """Get prediction labels from model for the given image tensor."""
        with torch.no_grad():
            if self.is_mmseg:
                # mmseg expects (H, W, C) numpy array
                if isinstance(img, torch.Tensor):
                    img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
                else:
                    img_np = img
                result = inference_model(self.model, img_np)
                return result.pred_sem_seg.data.squeeze().cuda()
            
            elif self.is_detectron2:
                # Detectron2 expects tensor
                if isinstance(img, np.ndarray):
                    img_tensor = torch.from_numpy(img).cuda()
                else:
                    img_tensor = img.cuda()
                if len(img_tensor.shape) == 4:
                    img_tensor = img_tensor.squeeze(0)
                inputs = [{"image": img_tensor}]
                outputs = self.model(inputs)
                return outputs[0]["sem_seg"].argmax(dim=0).cuda()
            
            else:
                # Standard PyTorch model via model_predict
                if isinstance(img, torch.Tensor):
                    img_np = img.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                else:
                    img_np = img
                _, pred_labels = model_predict(self.model, img_np, self.cfg)
                return pred_labels.cuda() if isinstance(pred_labels, torch.Tensor) else torch.from_numpy(pred_labels).cuda()

    def check_adv_status(self, img, original_pred_labels, target_labels=None, targeted=False):
        """
        Check if the current image is adversarial.
        
        Args:
            img: Current perturbed image (C, H, W) or (1, C, H, W)
            original_pred_labels: Original prediction labels (H, W)
            target_labels: Target labels for targeted attack (H, W)
            targeted: Whether this is a targeted attack
            
        Returns:
            is_adv: Boolean indicating if attack is successful
            changed_ratio: Ratio of pixels that changed prediction (excluding background)
        """
        pred_labels = self._get_pred_labels(img)
        
        # 배경/ignore class 제외 마스크
        if self.ignore_index is not None:
            foreground_mask = original_pred_labels != self.ignore_index
        else:
            foreground_mask = torch.ones_like(original_pred_labels, dtype=torch.bool)
        
        if targeted and target_labels is not None:
            # Targeted attack: success if prediction matches target
            if foreground_mask.sum() > 0:
                match_ratio = ((pred_labels == target_labels) & foreground_mask).float().sum() / foreground_mask.float().sum()
            else:
                match_ratio = (pred_labels == target_labels).float().mean()
            is_adv = match_ratio > self.success_threshold
            return is_adv, match_ratio.item()
        else:
            # Untargeted attack: success if prediction differs from original (excluding background)
            if foreground_mask.sum() > 0:
                changed_ratio = ((pred_labels != original_pred_labels) & foreground_mask).float().sum() / foreground_mask.float().sum()
            else:
                changed_ratio = (pred_labels != original_pred_labels).float().mean()
            is_adv = changed_ratio > self.success_threshold
            return is_adv, changed_ratio.item()

    def binary_search(self, x, indices, adv_value, non_adv_value, shape, original_pred_labels, target_labels=None, targeted=False, n_iterations=10):
        """
        Binary search to find minimal perturbation that maintains adversarial status.
        
        Args:
            x: Flattened image array
            indices: Pixel indices to modify (can be single int or array)
            adv_value: Value(s) that produce adversarial example
            non_adv_value: Value(s) that don't produce adversarial example
            shape: Original image shape (C, H, W)
            original_pred_labels: Original prediction labels
            target_labels: Target labels for targeted attack
            targeted: Whether this is a targeted attack
            n_iterations: Number of binary search iterations
            
        Returns:
            best_value: Best adversarial value found
            nquery: Number of queries used
        """
        nquery = 0
        
        for _ in range(n_iterations):
            next_value = (adv_value + non_adv_value) / 2
            x[indices] = next_value
            nquery += 1
            
            img_reshaped = x.reshape(shape)
            is_adv, _ = self.check_adv_status(
                torch.from_numpy(img_reshaped).unsqueeze(0).to(self.device),
                original_pred_labels,
                target_labels,
                targeted
            )
            
            if is_adv:
                adv_value = next_value
            else:
                non_adv_value = next_value
                
        return adv_value, nquery

    def pw_perturb(self, oimg, timg, original_pred_labels, target_labels=None, targeted=False, max_query=1000, snapshot_interval=100, query_offset=0):
        """
        Single-pixel PointWise perturbation attack.
        
        Args:
            oimg: Original image tensor (C, H, W) or numpy array
            timg: Target/starting adversarial image tensor (C, H, W) or numpy array
            original_pred_labels: Original prediction labels (H, W)
            target_labels: Target labels for targeted attack
            targeted: Whether this is a targeted attack
            max_query: Maximum number of queries
            snapshot_interval: Interval for saving snapshots (default: 100)
            
        Returns:
            x: Perturbed image (flattened)
            nquery: Total queries used
            D: L0 distance history
            snapshots: Dict of {query: image_copy} at each snapshot interval
        """
        # Convert to numpy if tensor
        if isinstance(oimg, torch.Tensor):
            oimg = oimg.cpu().numpy()
        if isinstance(timg, torch.Tensor):
            timg = timg.cpu().numpy()
        if isinstance(original_pred_labels, torch.Tensor):
            original_pred_labels = original_pred_labels.to(self.device)
        else:
            original_pred_labels = torch.from_numpy(original_pred_labels).to(self.device)

        shape = oimg.shape
        N = oimg.size
        start_qry = 0
        end_qry = 0

        # Flatten images
        original = oimg.copy().reshape(-1)
        x = timg.copy().reshape(-1)
        nquery = 0
        D = np.zeros(max_query + 500).astype(int)
        d = l0_distance(oimg, x.reshape(shape))
        
        # Snapshot storage
        snapshots = {}
        next_snapshot = snapshot_interval
        
        # Save initial state (query=0)
        snapshots[0] = x.copy().reshape(shape)

        terminate = False

        # Phase 1: Greedy restoration
        while not terminate:
            indices = list(range(N))
            random.shuffle(indices)

            for index in indices:
                old_value = x[index]
                new_value = original[index]
                
                if old_value == new_value:
                    continue
                    
                x[index] = new_value
                nquery += 1

                is_adv, changed_ratio = self.check_adv_status(
                    torch.from_numpy(x.reshape(shape)).unsqueeze(0).to(self.device),
                    original_pred_labels,
                    target_labels,
                    targeted
                )

                if is_adv:
                    start_qry = end_qry
                    end_qry = nquery
                    D[start_qry:end_qry] = d
                    d = l0_distance(oimg, x.reshape(shape))
                    
                    if self.verbose and nquery % 200 == 0:
                        print(f'nqry = {nquery}; Reset value -> L0 = {d}; changed_ratio = {changed_ratio:.4f}')
                else:
                    x[index] = old_value

                # Save snapshot at intervals
                if nquery + query_offset >= next_snapshot:
                    snapshots[next_snapshot] = x.copy().reshape(shape)
                    next_snapshot += snapshot_interval

                if nquery >= max_query:
                    terminate = True
                    break
            else:
                # No successful restoration in this pass
                terminate = True

        # Phase 2: Binary search refinement
        if nquery >= max_query:
            terminate = True
        else:
            terminate = False

        while not terminate:
            indices = list(range(N))
            random.shuffle(indices)
            improved = False

            for index in indices:
                old_value = x[index]
                original_value = original[index]
                
                if old_value == original_value:
                    continue
                    
                x[index] = original_value
                nquery += 1

                is_adv, changed_ratio = self.check_adv_status(
                    torch.from_numpy(x.reshape(shape)).unsqueeze(0).to(self.device),
                    original_pred_labels,
                    target_labels,
                    targeted
                )

                if is_adv:
                    start_qry = end_qry
                    end_qry = nquery
                    D[start_qry:end_qry] = d
                    d = l0_distance(oimg, x.reshape(shape))
                    improved = True
                else:
                    # Binary search for best value
                    best_adv_value, nqry = self.binary_search(
                        x, index, old_value, original_value,
                        shape, original_pred_labels, target_labels, targeted
                    )
                    nquery += nqry

                    if old_value != best_adv_value:
                        x[index] = best_adv_value
                        improved = True
                        start_qry = end_qry
                        end_qry = nquery
                        D[start_qry:end_qry] = d
                        d = l0_distance(oimg, x.reshape(shape))
                        
                        if self.verbose:
                            print(f'nquery = {nquery}; Binary search at {index}: {old_value:.2f} -> {best_adv_value:.2f}; L0 = {d}')
                    else:
                        x[index] = old_value

                # Save snapshot at intervals
                if nquery + query_offset >= next_snapshot:
                    snapshots[next_snapshot] = x.copy().reshape(shape)
                    next_snapshot += snapshot_interval

                if nquery >= max_query:
                    terminate = True
                    break

            if not improved:
                terminate = True

        d = l0_distance(oimg, x.reshape(shape))
        D[end_qry:nquery] = d
        
        # Save final snapshot
        snapshots['final'] = x.copy().reshape(shape)
        snapshots['final_query'] = nquery

        return x, nquery, D[:nquery], snapshots

    def pw_perturb_multiple(self, oimg, timg, original_pred_labels, target_labels=None, targeted=False, npix=196, max_query=1000, snapshot_interval=100, query_offset=0):
        """
        Multiple-pixel PointWise perturbation attack.
        Perturbs groups of pixels instead of single pixels for efficiency.
        
        Args:
            oimg: Original image tensor (C, H, W)
            timg: Target/starting adversarial image tensor (C, H, W)
            original_pred_labels: Original prediction labels (H, W)
            target_labels: Target labels for targeted attack
            targeted: Whether this is a targeted attack
            npix: Number of pixels per group
            max_query: Maximum number of queries
            snapshot_interval: Interval for saving snapshots
            
        Returns:
            x: Perturbed image (flattened)
            nquery: Total queries used
            D: L0 distance history
            snapshots: Dict of {query: image_copy} at each snapshot interval
        """
        # Convert to numpy if tensor
        if isinstance(oimg, torch.Tensor):
            oimg = oimg.cpu().numpy()
        if isinstance(timg, torch.Tensor):
            timg = timg.cpu().numpy()
        if isinstance(original_pred_labels, torch.Tensor):
            original_pred_labels = original_pred_labels.to(self.device)
        else:
            original_pred_labels = torch.from_numpy(original_pred_labels).to(self.device)

        shape = oimg.shape
        N = oimg.size
        start_qry = 0
        end_qry = 0

        original = oimg.copy().reshape(-1)
        x = timg.copy().reshape(-1)
        nquery = 0
        D = np.zeros(max_query + 500).astype(int)
        d = l0_distance(oimg, x.reshape(shape))
        ngroup = N // npix
        
        # Snapshot storage
        snapshots = {}
        next_snapshot = snapshot_interval
        
        # Save initial state (query=0)
        snapshots[0] = x.copy().reshape(shape)

        terminate = False

        # Phase 1: Greedy group restoration
        while not terminate:
            indices = list(range(N))
            random.shuffle(indices)

            for group_idx in range(ngroup):
                idx = indices[group_idx * npix:(group_idx + 1) * npix]
                old_value = x[idx].copy()
                new_value = original[idx]
                
                if np.abs(old_value - new_value).sum() == 0:
                    continue
                    
                x[idx] = new_value
                nquery += 1

                is_adv, changed_ratio = self.check_adv_status(
                    torch.from_numpy(x.reshape(shape)).unsqueeze(0).to(self.device),
                    original_pred_labels,
                    target_labels,
                    targeted
                )

                if is_adv:
                    start_qry = end_qry
                    end_qry = nquery
                    D[start_qry:end_qry] = d
                    d = l0_distance(oimg, x.reshape(shape))
                    
                    if self.verbose and nquery % 200 == 0:
                        print(f'nqry = {nquery}; Group reset -> L0 = {d}; changed_ratio = {changed_ratio:.4f}')
                else:
                    x[idx] = old_value

                # Save snapshot at intervals
                if nquery + query_offset >= next_snapshot:
                    snapshots[next_snapshot] = x.copy().reshape(shape)
                    next_snapshot += snapshot_interval

                if nquery >= max_query:
                    terminate = True
                    break
            else:
                terminate = True

        # Phase 2: Refinement with binary search
        if nquery >= max_query:
            terminate = True
        else:
            terminate = False

        if self.verbose:
            print('Refine stage!')

        while not terminate:
            indices = list(range(N))
            random.shuffle(indices)
            improved = False

            for group_idx in range(ngroup):
                idx = indices[group_idx * npix:(group_idx + 1) * npix]
                old_value = x[idx].copy()
                original_value = original[idx]
                
                if np.abs(old_value - original_value).sum() == 0:
                    continue
                    
                x[idx] = original_value
                nquery += 1

                is_adv, changed_ratio = self.check_adv_status(
                    torch.from_numpy(x.reshape(shape)).unsqueeze(0).to(self.device),
                    original_pred_labels,
                    target_labels,
                    targeted
                )

                if is_adv:
                    start_qry = end_qry
                    end_qry = nquery
                    D[start_qry:end_qry] = d
                    d = l0_distance(oimg, x.reshape(shape))
                    improved = True
                else:
                    # Binary search for group
                    best_adv_value, nqry = self.binary_search(
                        x, idx, old_value, original_value,
                        shape, original_pred_labels, target_labels, targeted
                    )
                    nquery += nqry
                    
                    if (old_value - best_adv_value).sum() != 0:
                        x[idx] = best_adv_value
                        improved = True
                        start_qry = end_qry
                        end_qry = nquery
                        D[start_qry:end_qry] = d
                        d = l0_distance(oimg, x.reshape(shape))
                        
                        if self.verbose and nquery % 200 == 0:
                            print(f'nquery = {nquery}; Group binary search -> L0 = {d}')
                    else:
                        x[idx] = old_value

                # Save snapshot at intervals
                if nquery + query_offset >= next_snapshot:
                    snapshots[next_snapshot] = x.copy().reshape(shape)
                    next_snapshot += snapshot_interval

                if nquery >= max_query:
                    terminate = True
                    break

            if not improved:
                terminate = True

        d = l0_distance(oimg, x.reshape(shape))
        D[end_qry:nquery] = d
        
        # Save final snapshot
        snapshots['final'] = x.copy().reshape(shape)
        snapshots['final_query'] = nquery

        return x, nquery, D[:nquery], snapshots

    def pw_perturb_multiple_scheduling(self, oimg, timg, original_pred_labels, target_labels=None, targeted=False, npix=196, max_query=1000, snapshot_interval=100, query_offset=0):
        """
        Multiple-pixel PointWise attack with adaptive group size scheduling.
        Starts with larger groups and reduces size over iterations.
        
        Args:
            oimg: Original image tensor (C, H, W)
            timg: Target/starting adversarial image tensor (C, H, W)
            original_pred_labels: Original prediction labels (H, W)
            target_labels: Target labels for targeted attack
            targeted: Whether this is a targeted attack
            npix: Initial number of pixels per group
            max_query: Maximum number of queries
            snapshot_interval: Interval for saving snapshots
            
        Returns:
            x: Perturbed image (flattened)
            nquery: Total queries used
            D: L0 distance history
            snapshots: Dict of {query: image_copy} at each snapshot interval
        """
        # Convert to numpy if tensor
        if isinstance(oimg, torch.Tensor):
            oimg = oimg.cpu().numpy()
        if isinstance(timg, torch.Tensor):
            timg = timg.cpu().numpy()
        if isinstance(original_pred_labels, torch.Tensor):
            original_pred_labels = original_pred_labels.to(self.device)
        else:
            original_pred_labels = torch.from_numpy(original_pred_labels).to(self.device)

        shape = oimg.shape
        N = oimg.size
        start_qry = 0
        end_qry = 0

        original = oimg.copy().reshape(-1)
        x = timg.copy().reshape(-1)
        nquery = 0
        D = np.zeros(max_query + 500).astype(int)
        d = l0_distance(oimg, x.reshape(shape))
        
        # Snapshot storage
        snapshots = {}
        next_snapshot = snapshot_interval
        
        # Save initial state (query=0)
        snapshots[0] = x.copy().reshape(shape)
        
        last_success_query = None
        terminate = False

        # Phase 1: Greedy group restoration with scheduling
        while not terminate:
            ngroup = N // npix
            indices = list(range(N))
            random.shuffle(indices)

            for group_idx in range(ngroup):
                idx = indices[group_idx * npix:(group_idx + 1) * npix]
                old_value = x[idx].copy()
                new_value = original[idx]
                
                if np.abs(old_value - new_value).sum() == 0:
                    continue
                    
                x[idx] = new_value
                nquery += 1

                is_adv, changed_ratio = self.check_adv_status(
                    torch.from_numpy(x.reshape(shape)).unsqueeze(0).to(self.device),
                    original_pred_labels,
                    target_labels,
                    targeted
                )

                if is_adv:
                    start_qry = end_qry
                    end_qry = nquery
                    D[start_qry:end_qry] = d
                    d = l0_distance(oimg, x.reshape(shape))
                    if nquery <= max_query:
                        last_success_query = nquery
                    
                    if self.verbose and nquery % 200 == 0:
                        print(f'nqry = {nquery}; npix = {npix}; L0 = {d}; changed_ratio = {changed_ratio:.4f}')
                else:
                    x[idx] = old_value

                # Save snapshot at intervals
                if nquery + query_offset >= next_snapshot:
                    snapshots[next_snapshot] = x.copy().reshape(shape)
                    next_snapshot += snapshot_interval

                if nquery >= max_query:
                    terminate = True
                    break
            else:
                # If loop completed without breaking (no improvement found this epoch)
                if npix >= 2:
                    npix //= 2
                    terminate = False # Continue with smaller group size
                else:
                    terminate = True # Stop if cannot reduce further

        # Phase 2: Refinement
        if nquery >= max_query:
            terminate = True
        else:
            terminate = False

        if self.verbose:
            print('Refine stage!')

        ngroup = N // max(npix, 1)
        
        while not terminate:
            indices = list(range(N))
            random.shuffle(indices)
            improved = False

            for group_idx in range(ngroup):
                idx = indices[group_idx * npix:(group_idx + 1) * npix]
                old_value = x[idx].copy()
                original_value = original[idx]
                
                if np.abs(old_value - original_value).sum() == 0:
                    continue
                    
                x[idx] = original_value
                nquery += 1

                is_adv, changed_ratio = self.check_adv_status(
                    torch.from_numpy(x.reshape(shape)).unsqueeze(0).to(self.device),
                    original_pred_labels,
                    target_labels,
                    targeted
                )

                if is_adv:
                    start_qry = end_qry
                    end_qry = nquery
                    D[start_qry:end_qry] = d
                    d = l0_distance(oimg, x.reshape(shape))
                    improved = True
                else:
                    best_adv_value, nqry = self.binary_search(
                        x, idx, old_value, original_value,
                        shape, original_pred_labels, target_labels, targeted
                    )
                    nquery += nqry
                    
                    if (old_value - best_adv_value).sum() != 0:
                        x[idx] = best_adv_value
                        improved = True
                        start_qry = end_qry
                        end_qry = nquery
                        D[start_qry:end_qry] = d
                        d = l0_distance(oimg, x.reshape(shape))
                        
                        if self.verbose and nquery % 200 == 0:
                            print(f'nquery = {nquery}; Refinement -> L0 = {d}')
                            
                        # Refinement binary search에서 찾은 값은 성공한 값이므로 업데이트
                        if nquery <= max_query:
                            last_success_query = nquery
                    else:
                        x[idx] = old_value

                # Save snapshot at intervals
                if nquery + query_offset >= next_snapshot:
                    snapshots[next_snapshot] = x.copy().reshape(shape)
                    next_snapshot += snapshot_interval

                if nquery >= max_query:
                    terminate = True
                    break

            if not improved:
                terminate = True

        d = l0_distance(oimg, x.reshape(shape))
        D[end_qry:nquery] = d
        
        # Save final snapshot
        snapshots['final'] = x.copy().reshape(shape)
        snapshots['final_query'] = nquery
        snapshots['last_success_query'] = last_success_query

        return x, nquery, D[:nquery], snapshots

    def perturb(self, img, gt, starting_adv=None, mode='single', npix=196, max_query=1000):
        """
        Main entry point for PointWise attack.
        
        Args:
            img: Original image tensor (1, C, H, W) or (C, H, W)
            gt: Ground truth segmentation (H, W) - used for mask creation
            starting_adv: Starting adversarial image. If None, random initialization is used.
            mode: Attack mode - 'single', 'multiple', or 'scheduling'
            npix: Pixels per group for multiple mode
            max_query: Maximum queries
            
        Returns:
            nquery: Total queries used
            adv: Final adversarial image tensor (1, C, H, W)
            D: L0 distance history
        """
        # Ensure proper shape
        if len(img.shape) == 4:
            img = img.squeeze(0)  # (C, H, W)
        if len(gt.shape) == 3:
            gt = gt.squeeze(0)  # (H, W)

        # Get original predictions
        original_pred_labels = self._get_pred_labels(img.unsqueeze(0))

        # Create starting adversarial if not provided
        if starting_adv is None:
            # Random initialization
            starting_adv = torch.rand_like(img) * 255.0
            starting_adv = starting_adv.clamp(0, 255)

        if len(starting_adv.shape) == 4:
            starting_adv = starting_adv.squeeze(0)

        # Run attack based on mode
        if mode == 'single':
            x, nquery, D = self.pw_perturb(
                img, starting_adv, original_pred_labels,
                max_query=max_query
            )
        elif mode == 'multiple':
            x, nquery, D = self.pw_perturb_multiple(
                img, starting_adv, original_pred_labels,
                npix=npix, max_query=max_query
            )
        elif mode == 'scheduling':
            x, nquery, D = self.pw_perturb_multiple_scheduling(
                img, starting_adv, original_pred_labels,
                npix=npix, max_query=max_query
            )
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'single', 'multiple', or 'scheduling'.")

        # Reshape result to tensor
        adv = torch.from_numpy(x.reshape(img.shape)).unsqueeze(0).float().to(self.device)

        return nquery, adv, D
