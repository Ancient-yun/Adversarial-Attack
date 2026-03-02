# agent.md — SparsePGD (L0) for Semantic Segmentation (CE loss)

## 0) Goal
- Extend the provided SparsePGD implementation (classification) to attack **semantic segmentation** models.
- Use **Cross Entropy (CE)** as the loss.
- Preserve the original SparsePGD structure:
  - perturbation update (PGD on `perturb`)
  - mask update (continuous `mask` -> sigmoid -> top-k projection)
  - projection constraints: `||mask||_0 <= k` and image range `[0,1]` (or dataset-specific normalization handling)
  - optional `unprojected_gradient` masking variants

## 1) Non-goals / Must-not-break
- Do NOT change training code or model weights.
- Do NOT silently change normalization assumptions.
- Do NOT remove early-stop / restart logic unless you re-implement equivalent behavior.
- Do NOT introduce new dependencies without explicit need.

## 2) Interface Contract (Segmentation)
### Inputs
- `x`: Float tensor, shape `[B, C, H, W]`
- `y`: Long tensor, shape `[B, H, W]` (per-pixel class labels)
- optional:
  - `ignore_index` (e.g., 255) for unlabeled pixels
  - `targeted` and `target` (target labels per pixel: `[B,H,W]`) if implemented

### Model output
- Expect segmentation logits: `[B, num_classes, H, W]`
- If the model returns a dict or a structure (common in mmseg), implement a small adaptor:
  - `logits = extract_logits(model(x_adv))`
  - Ensure logits are raw (pre-softmax) scores.

## 3) Loss Definition (CE)
### Untargeted
- Use per-pixel CE:
  - `loss_map = F.cross_entropy(logits, y, reduction='none', ignore_index=ignore_index)`
  - `loss = loss_map.mean(dim=(1,2))`  # per-sample scalar loss for attack bookkeeping
- Attack objective is to **maximize** CE (gradient ascent), consistent with current code path using `loss.sum().backward()` and updating `perturb` with `+ step * sign(grad)`.

### Targeted (optional)
- If you implement targeted segmentation:
  - Maximize negative CE to the target mask:
    - `loss_map = -F.cross_entropy(logits, target, reduction='none', ignore_index=ignore_index)`
    - `loss = loss_map.mean(dim=(1,2))`

## 4) Success / Early-stop Criteria (Segmentation)
Classification code uses `argmax==y` for success. For segmentation:
- Define success per-sample using a robust rule:
  - Option A (simple): success if **pixel accuracy** drops below a threshold:
    - `acc = mean( pred==y on valid pixels )`
    - success if `acc < acc_threshold` (default e.g. 0.99 or 0.95 depending on strictness)
  - Option B (more aligned): success if **mIoU** drops below threshold (heavier compute).
- Implement **valid pixel mask**:
  - `valid = (y != ignore_index)`
- When `early_stop=True`, remove already “successful” samples from the active set like the original implementation.

Default recommendation:
- Use Option A for speed and stability; keep the threshold configurable.

## 5) Masking / Projection Rules (L0)
- Keep the core factorization:
  - `proj_perturb, proj_mask = masking.apply(perturb, sigmoid(mask), k)`
- Enforce (assert / clamp) invariants:
  - `L0(proj_mask) <= k` per sample (pixel positions)
  - `x_adv = clamp_to_valid_range(x + proj_perturb)`
- IMPORTANT: For segmentation, `k` is **#pixels** (spatial positions), not channels.
  - If `attack_mode="pixel"`: mask shape `[B,1,H,W]` and broadcasting over channels.
  - If `attack_mode="feature"`: mask shape `[B,C,H,W]` and L0 counts over all channels (usually not desired for “k pixels”).
- Default: `attack_mode="pixel"` for true L0 pixel budget.

## 6) Normalization / Data Range Handling
- Current code assumes `x` in `[0,1]` and clamps perturb so `x+perturb` stays in `[0,1]`.
- If the segmentation pipeline uses normalization (mean/std):
  - Preferred approach: attack in **input space before normalization**, or
  - If only normalized tensors are available, adjust clamping:
    - Convert bounds to normalized bounds per channel:
      - `low = (0 - mean)/std`, `high = (1 - mean)/std`
    - Clamp `x_adv` to `[low, high]` channel-wise.
- Implement this as a utility:
  - `clamp_like_input(x_adv, x, bounds_cfg)`
- Do NOT guess mean/std; require explicit config or infer from pipeline object if available.

## 7) Gradient Flow Choices
- Preserve both masking strategies:
  - `MaskingA`: unprojected gradient variant (uses soft mask in backward)
  - `MaskingB`: projected gradient variant (hard top-k in backward)
- For segmentation CE, ensure `loss.sum().backward()` yields:
  - `grad_perturb` same shape as `perturb`
  - `grad_mask` same shape as `mask`

## 8) Restart / Patience Logic
- Keep the “mask not changing” detection:
  - compare `project_mask(mask)` vs `project_mask(prev_mask)` using L0 norm
- When unchanged for `patience` iterations, reinitialize mask for those samples.
- Ensure reinit respects segmentation mask shape rules (`pixel` vs `feature`).

## 9) Code Structure Expectations
Create a new file or class without breaking the original:
- `seg_sparse_pgd.py` (recommended) containing:
  - `class SegSparsePGD(SparsePGD)` or a standalone class with shared utilities
- Minimal edits to shared code:
  - Extract common helpers (shape checks, clamp, project_mask) if needed.
- Provide an example usage snippet (no training):
  - initialize model
  - call `attacker.perturb(x, y)`
  - verify outputs and shapes

## 10) Required Tests (lightweight)
Implement quick sanity tests runnable on CPU:
1) Shape test:
- input `[2,3,64,64]`, y `[2,64,64]`, logits mocked `[2,K,64,64]`
2) Constraint test:
- after projection, count L0 pixels <= k
3) Clamp test:
- `x_adv` within bounds
4) Gradient test:
- verify `perturb.grad` and `mask.grad` are not None for a forward/backward pass

## 11) Logging / Debug
- Add optional verbose logs every `verbose_interval`:
  - current mean CE, current pixel-acc, #active samples, restart counts
- Do not print by default.

## 12) Deliverables Checklist
- [ ] Segmentation-compatible attacker class
- [ ] CE loss for segmentation (ignore_index supported)
- [ ] Success criterion and early stop for segmentation
- [ ] Correct L0 counting semantics (pixel budget)
- [ ] Normalization-aware clamping option
- [ ] Minimal tests / usage example
- [ ] Keep original API style (`perturb`, `__call__`, `change_masking`)

## 13) Coding Style
- Use torch ops; avoid numpy except for constants.
- Keep device/dtype consistent; no `.cpu()` in attack loop.
- Avoid in-place ops that break autograd on tensors requiring grad (except safe clamping on detached tensors).
- Keep functions pure where possible; document shapes in docstrings.

## 14) Questions to ask only if blocked
- What is the model forward output format? (tensor vs dict/SegDataSample)
- What is ignore_index? (common: 255)
- Are inputs normalized or in [0,1]?
- What is the chosen success threshold? (pixel-acc or mIoU)