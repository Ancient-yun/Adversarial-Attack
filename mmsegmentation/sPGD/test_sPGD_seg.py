import torch
import torch.nn as nn

from sPGD_seg import SegSparsePGD


class DummySegModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=5):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=False)

    def forward(self, x):
        return self.conv(x)


class DummyMMSegModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=5):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1, bias=False)

        class _Preprocessor:
            def __call__(self, data, training=False):
                return data

        self.data_preprocessor = _Preprocessor()

    def inference(self, inputs, metas):
        if isinstance(inputs, list):
            x = torch.stack(inputs, dim=0)
        else:
            x = inputs
        return self.conv(x)


def _build_batch(batch=2, channels=3, height=64, width=64, num_classes=5):
    torch.manual_seed(0)
    x = torch.rand(batch, channels, height, width)
    y = torch.randint(0, num_classes, (batch, height, width))
    y[:, :8, :8] = 255
    return x, y


def test_shape_and_constraints():
    model = DummySegModel()
    attacker = SegSparsePGD(
        model=model,
        epsilon=8 / 255,
        k=40,
        t=3,
        random_start=True,
        early_stop=False,
        attack_mode="pixel",
    )
    attacker.configure_segmentation(ignore_index=255, acc_threshold=0.99)
    x, y = _build_batch()
    x_adv, robust, it = attacker(x, y)

    assert x_adv.shape == x.shape
    assert robust.shape == (x.shape[0],)
    assert it.shape == (x.shape[0],)

    diff = x_adv - x
    l0 = (diff.abs().sum(dim=1) > 1e-7).flatten(1).sum(dim=1)
    assert torch.all(l0 <= attacker.k), f"L0 budget violated: {l0}"
    assert torch.all((x_adv >= 0.0) & (x_adv <= 1.0)), "x_adv out of [0,1]"


def test_gradient_flow():
    model = DummySegModel()
    attacker = SegSparsePGD(model=model, epsilon=8 / 255, k=20, t=1, attack_mode="pixel")
    x, y = _build_batch()
    perturb = attacker.initial_perturb(x)
    mask = attacker.initial_mask(x)

    perturb.requires_grad_()
    mask.requires_grad_()
    proj_perturb, _ = attacker.masking.apply(perturb, torch.sigmoid(mask), attacker.k)
    x_adv = attacker._clamp_like_input(x + proj_perturb, x)
    logits = attacker._forward_logits(x_adv)
    loss, _ = attacker.loss_fn(logits, y, targeted=False, target=None)
    loss.sum().backward()

    assert perturb.grad is not None, "perturb.grad is None"
    assert mask.grad is not None, "mask.grad is None"


def test_mmseg_forward_mode():
    model = DummyMMSegModel()
    attacker = SegSparsePGD(
        model=model,
        epsilon=8 / 255,
        k=20,
        t=2,
        random_start=True,
        early_stop=False,
        attack_mode="pixel",
    )
    attacker.configure_segmentation(ignore_index=255, acc_threshold=0.99, forward_mode="mmseg")
    x, y = _build_batch(height=32, width=32, num_classes=5)
    x_adv, robust, it = attacker(x, y)

    assert x_adv.shape == x.shape
    assert robust.shape == (x.shape[0],)
    assert it.shape == (x.shape[0],)


def test_ratio_budget_recomputed_from_image_size():
    model = DummySegModel()
    ratio = 0.05
    attacker = SegSparsePGD(
        model=model,
        epsilon=8 / 255,
        k=ratio,
        t=2,
        random_start=True,
        early_stop=False,
        attack_mode="pixel",
    )
    x, y = _build_batch(height=32, width=32, num_classes=5)
    x_adv, _, _ = attacker(x, y)

    expected_k = max(1, int(32 * 32 * ratio))
    assert attacker.current_k == expected_k, f"expected current_k={expected_k}, got {attacker.current_k}"
    l0 = (x_adv - x).abs().sum(dim=1).gt(1e-7).flatten(1).sum(dim=1)
    assert torch.all(l0 <= expected_k), f"ratio-based k violated: {l0} > {expected_k}"


if __name__ == "__main__":
    test_shape_and_constraints()
    test_gradient_flow()
    test_mmseg_forward_mode()
    test_ratio_budget_recomputed_from_image_size()
    print("sPGD_seg sanity tests passed")
