
import torch
import numpy as np
import random
from utils_se import l0

class SpaEvoAttack():
    def __init__(self,
                model,
                n_pix=196,
                pop_size=10,
                cr=0.9,
                mu=0.01,
                seed=None,
                success_threshold=0.01,
                verbose=False,
                device='cuda',
                is_mmseg=True):

        self.model = model
        self.n_pix = n_pix
        self.pop_size = pop_size
        self.cr = cr
        self.mu = mu
        self.seed = seed
        self.success_threshold = success_threshold
        self.verbose = verbose
        self.device = device
        self.is_mmseg = is_mmseg
        self.ignore_index = 255  # Default: Cityscapes uses 255, can be changed via set_ignore_index

    def set_ignore_index(self, dataset_name):
        """Set ignore index based on dataset name."""
        if dataset_name.lower() == 'cityscapes':
            self.ignore_index = 255
        elif dataset_name.lower() in ['voc2012']:
            self.ignore_index = 0
        else:  # ADE20K 등: 예측에서 제외할 클래스 없음 (label 0 = wall)
            self.ignore_index = None

    def _get_pred_labels(self, img):
        if self.is_mmseg:
            from mmseg.apis import inference_model
            # img shape: (C, H, W) or (1, C, H, W) -> numpy (H, W, C)
            if isinstance(img, torch.Tensor):
                if img.dim() == 4:
                    img = img.squeeze(0)
                img_np = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            else:
                img_np = img
                
            result = inference_model(self.model, img_np)
            return result.pred_sem_seg.data.squeeze().cuda()
        else:
            # Fallback for other models (e.g. classification)
            # Not implemented for this port
            raise NotImplementedError("Only mmseg models are supported")

    def check_adv_status(self, img, original_pred_labels, target_labels=None, targeted=False):
        pred_labels = self._get_pred_labels(img)
        
        # 배경/ignore class 제외 마스크
        if self.ignore_index is not None:
            foreground_mask = original_pred_labels != self.ignore_index
        else:
            foreground_mask = torch.ones_like(original_pred_labels, dtype=torch.bool)
        
        if targeted and target_labels is not None:
            if foreground_mask.sum() > 0:
                match_ratio = ((pred_labels == target_labels) & foreground_mask).float().sum() / foreground_mask.float().sum()
            else:
                match_ratio = (pred_labels == target_labels).float().mean()
            is_adv = match_ratio > self.success_threshold
            return is_adv, match_ratio.item()
        else:
            if foreground_mask.sum() > 0:
                changed_ratio = ((pred_labels != original_pred_labels) & foreground_mask).float().sum() / foreground_mask.float().sum()
            else:
                changed_ratio = (pred_labels != original_pred_labels).float().mean()
            is_adv = changed_ratio > self.success_threshold
            return is_adv, changed_ratio.item()

    def convert1D_to_2D(self, idx, wi):
        c1 = idx // wi
        c2 = idx - c1 * wi
        return c1, c2

    def convert2D_to_1D(self, x, y, wi):
        outp = x * wi + y
        return outp

    def masking(self, oimg, timg):
        # oimg, timg: (C, H, W) tensors
        xo = torch.abs(oimg - timg)
        # Check if ANY channel differs
        d = (xo > 0).any(dim=0) # (H, W) boolean
        
        wi = oimg.shape[2]
        # Get coordinates of different pixels
        p = torch.nonzero(d, as_tuple=True)
        # p[0] is y (height), p[1] is x (width) - based on (C, H, W)
        # Wait, original code uses (C, W, H)? No, standard is (C, H, W).
        # Let's check original convert2D_to_1D logic.
        # Original: c1 = idx // wi (row index), c2 = idx % wi (col index)
        # So idx = row * wi + col. This matches standard flattening if we flatten H*W.
        
        out = self.convert2D_to_1D(p[0], p[1], wi)
        return out.cpu().numpy()

    def uni_rand(self, oimg, timg, original_pred_labels, target_labels=None, targeted=False):
        """
        원본 방식: 전체 차이(p1)에서 시작하여 n개 픽셀을 제거하면서
        공격이 여전히 성공하는 마스크를 찾습니다.
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        nqry = 0
        _, he, wi = oimg.shape  # (C, H, W)
        
        fit = torch.zeros(self.pop_size) + np.inf
        pop = []

        # p1: 전체 차이 마스크. 1 = timg 픽셀 사용, 0 = oimg 사용
        p1 = np.zeros(he * wi).astype(int)
        idxs = self.masking(oimg, timg)
        p1[idxs] = 1
        
        total_diff = p1.sum()
        
        # n_pix 조정: 전체 차이보다 클 수 없음
        if total_diff < self.n_pix:
            self.n_pix = int(total_diff)

        for i in range(self.pop_size):
            n = self.n_pix  # 제거할 픽셀 수
            j = 0
            
            while True:
                p = p1.copy()
                
                # n개 픽셀을 랜덤하게 선택하여 제거 (0으로 설정 = 원본으로 복구)
                if n > 0 and len(idxs) >= n:
                    idx_to_remove = np.random.choice(idxs, n, replace=False)
                    p[idx_to_remove] = 0
                
                nqry += 1
                fitness = self.feval(p, oimg, timg, original_pred_labels, target_labels, targeted)
                
                if fitness < fit[i]:
                    # 공격 성공! 이 마스크 저장
                    pop.append(p)
                    fit[i] = fitness
                    break
                elif n > 1:
                    # 덜 제거해봄 (더 많은 픽셀 유지)
                    n -= 1
                elif n == 1:
                    # 하나씩 제거하며 시도
                    while j < len(idxs):
                        p = p1.copy()
                        p[idxs[j]] = 0
                        nqry += 1
                        fitness = self.feval(p, oimg, timg, original_pred_labels, target_labels, targeted)
                        
                        if fitness < fit[i]:
                            pop.append(p)
                            fit[i] = fitness
                            break
                        else:
                            j += 1
                    
                    if j >= len(idxs):
                        # 아무것도 제거 못함 -> 전체 마스크 사용
                        pop.append(p1.copy())
                        fit[i] = self.feval(p1, oimg, timg, original_pred_labels, target_labels, targeted)
                    break
            
            if j >= len(idxs) - 1:
                break
        
        # 부족한 개체는 전체 마스크로 채움
        while len(pop) < self.pop_size:
            pop.append(p1.copy())
            
        return pop, nqry, fit

    def recombine(self, p0, p1, p2):
        cross_points = np.random.rand(len(p1)) < self.cr # uniform random
        if not np.any(cross_points):
            cross_points[np.random.randint(0, len(p1))] = True
        trial = np.where(cross_points, p1, p2).astype(int)
        # Constraints: trial should be subset of p0? 
        # logical_and(p0, trial)? p0 is best_idx?. 
        # In original: trial = np.logical_and(p0, trial)
        # This enforces that we only keep pixels that were present in p0 (the best one)?
        # That would strictly reduce the set of pixels over time (monotonically decreasing L0).
        # This matches "Sparse" evolution.
        trial = np.logical_and(p0, trial).astype(int) 
        return trial

    def mutate(self, p):
        outp = p.copy()
        if p.sum() != 0:
            one = np.where(outp == 1)[0]
            # Remove a fraction of pixels (reduce L0)
            n_px = int(len(one) * self.mu)
            if n_px == 0 and len(one) > 0 and self.mu > 0:
                n_px = 1
                
            if n_px > 0:
                idx = np.random.choice(one, n_px, replace=False)
                outp[idx] = 0

        return outp

    def modify(self, pop, oimg, timg):
        # pop is a binary mask (1D). 1 means use timg, 0 means use oimg.
        # Original code: p = np.where(pop == 0) -> c1, c2 ... img[...] = oimg[...]
        # This implies timg is the base, and we revert pixels where pop==0.
        # So pop==1 means KEEP timg.
        
        # oimg, timg: (C, H, W)
        _, he, wi = oimg.shape
        mask = pop.reshape(he, wi) # (H, W)
        
        # Convert to tensor mask (1, 1, H, W) or (1, H, W) for broadcasting
        mask_tensor = torch.from_numpy(mask).to(self.device).float() # 0 or 1
        
        # Result = timg * mask + oimg * (1 - mask)
        # mask=1 -> timg
        # mask=0 -> oimg
        
        img = timg * mask_tensor + oimg * (1 - mask_tensor)
        return img

    def feval(self, pop, oimg, timg, original_pred_labels, target_labels, targeted):
        xp = self.modify(pop, oimg, timg)
        
        # L2 거리 계산 (원본 코드와 동일)
        l2_val = torch.norm(oimg - xp).item()  # Python float로 변환
        
        # Check adversarial status
        is_adv, ratio = self.check_adv_status(xp, original_pred_labels, target_labels, targeted)
        
        if is_adv:
            lc = 0
            # fitness = L2
            outp = l2_val
        else:
            lc = np.inf  # Penalty
            # fitness = L2 + Penalty
            outp = l2_val + lc
            
        return outp 

    def selection(self, x1, f1, x2, f2):
        if f2 < f1:
            return x2, f2
        return x1, f1

    def evo_perturb(self, oimg, timg, original_pred_labels, target_labels=None, targeted=False, max_query=1000, snapshot_interval=100, query_offset=0):
        # 0. variable init
        if self.seed is not None:
            np.random.seed(self.seed)

        # Convert inputs to tensors if numpy
        if not isinstance(oimg, torch.Tensor):
            oimg = torch.from_numpy(oimg).to(self.device).float()
        if not isinstance(timg, torch.Tensor):
            timg = torch.from_numpy(timg).to(self.device).float()
            
        # Ensure shape (C, H, W)
        if oimg.dim() == 4: oimg = oimg.squeeze(0)
        if timg.dim() == 4: timg = timg.squeeze(0)

        D = torch.zeros(max_query + 500, dtype=int).to(self.device)
        
        # Snapshot storage
        snapshots = {}
        next_snapshot = snapshot_interval
        last_success_query = None
        
        # 1. population init
        # uni_rand now selects a SPARSE subset of timg to start with
        pop, nqry, fitness = self.uni_rand(oimg, timg, original_pred_labels, target_labels, targeted)
        
        if len(pop) > 0:
            # 2. find the worst & best
            rank = np.argsort(fitness) 
            best_idx = rank[0].item()
            worst_idx = rank[-1].item()

            best_mask = pop[best_idx]
            current_best_l0 = best_mask.sum().item()
            D[:nqry] = current_best_l0
            
            # Print initial best
            adv_img = self.modify(best_mask, oimg, timg)
            
            # Save initial snapshot (Query ~ init_nqry)
            snapshots[0] = adv_img.cpu().numpy()
            
            is_adv, ratio = self.check_adv_status(adv_img, original_pred_labels, target_labels, targeted)
            if self.verbose:
                print(f"Init: Best L0={current_best_l0}, Adv={is_adv}, Ratio={ratio:.4f}")

            # 3. evolution
            prev_best_fitness = fitness[best_idx]
            while nqry < max_query:
                # a. Crossover (recombine)
                idxs = [idx for idx in range(self.pop_size) if idx != best_idx]
                id1, id2 = np.random.choice(idxs, 2, replace=False)
                offspring = self.recombine(pop[best_idx], pop[id1], pop[id2])

                # b. mutation (diversify)
                offspring = self.mutate(offspring)
                    
                # c. fitness evaluation
                fo = self.feval(offspring, oimg, timg, original_pred_labels, target_labels, targeted)
                nqry += 1
                    
                # d. select
                pop[worst_idx], fitness[worst_idx] = self.selection(pop[worst_idx], fitness[worst_idx], offspring, fo)
                    
                # e. update best and worst
                rank = np.argsort(fitness)
                best_idx = rank[0].item()
                worst_idx = rank[-1].item()

                if nqry <= max_query:
                    # If improved or first time
                    if last_success_query is None or fitness[best_idx] < prev_best_fitness:
                        last_success_query = nqry
                        prev_best_fitness = fitness[best_idx]

                # ====== record ======
                best_mask = pop[best_idx]
                current_best_l0 = best_mask.sum().item()
                D[nqry-1] = current_best_l0 # Record history
                
                # Save snapshot at intervals
                if nqry + query_offset >= next_snapshot:
                    adv_snapshot = self.modify(best_mask, oimg, timg)
                    snapshots[next_snapshot] = adv_snapshot.cpu().numpy()
                    next_snapshot += snapshot_interval
                
                if self.verbose and nqry % 100 == 0:
                     # Check ratio occasionally for logging
                    adv_tmp = self.modify(best_mask, oimg, timg)
                    _, ratio = self.check_adv_status(adv_tmp, original_pred_labels, target_labels, targeted)
                    print(f"Query {nqry}: Best L0={current_best_l0}, Ratio={ratio:.4f}")

                if nqry >= max_query:
                    break
            
            adv = self.modify(pop[best_idx], oimg, timg)
            
            # Save final snapshot
            snapshots['final'] = adv.cpu().numpy()
            snapshots['final_query'] = nqry
            snapshots['last_success_query'] = last_success_query
        else:
            adv = timg
            nqry = 1
            D[0] = l0(oimg, timg)
            snapshots[0] = adv.cpu().numpy() # Init if logic allows
            snapshots['final'] = adv.cpu().numpy()
            snapshots['final_query'] = nqry
            snapshots['last_success_query'] = nqry if fitness[0] < 1e8 else None
            
        # Re-verify and final check (optional)
        # Ensure we return a valid adversarial example if possible
        # If the best individual is not adversarial (fitness > 1e9), warn or return timg/oimg
        if fitness[best_idx] > 1e8:
            if self.verbose:
                print("Warning: Attack failed to find adversarial example.")
            # return original or timg? 
            # If failed, usually return timg (starting point) if it was adv, or oimg if not.
            pass

        return adv.cpu().numpy(), nqry, D[:nqry], snapshots
