"""Unit tests for noise-robust losses.

Covers: output shape, gradient flow, numerical stability at extreme inputs,
and coarse correctness against reference formulas.

Run: pytest tests/test_losses.py
"""
import math
import pytest
import torch
import torch.nn.functional as F

from src.losses.sce import SCELoss
from src.losses.gce import GCELoss
from src.losses.peer_loss import PeerLoss
from src.losses.elr import ELRLoss
from src.losses.label_smoothing import LabelSmoothingLoss
from src.losses.composite_loss import CompositeLoss
from src.losses.base_loss import build_loss


BATCH = 8
NUM_CLASSES = 10


@pytest.fixture
def logits_targets():
    torch.manual_seed(0)
    logits = torch.randn(BATCH, NUM_CLASSES, requires_grad=True)
    targets = torch.randint(0, NUM_CLASSES, (BATCH,))
    return logits, targets


# ---------- shape + finiteness ----------

@pytest.mark.parametrize("loss_cls,kwargs", [
    (SCELoss, dict(num_classes=NUM_CLASSES, alpha=0.1, beta=1.0)),
    (GCELoss, dict(num_classes=NUM_CLASSES, q=0.7)),
    (PeerLoss, dict(num_classes=NUM_CLASSES, lam=0.5)),
    (LabelSmoothingLoss, dict(num_classes=NUM_CLASSES, smoothing=0.1)),
])
def test_scalar_output(loss_cls, kwargs, logits_targets):
    logits, targets = logits_targets
    loss = loss_cls(**kwargs)
    out = loss(logits, targets)
    assert out.ndim == 0, f"{loss_cls.__name__} should return scalar, got shape {out.shape}"
    assert torch.isfinite(out), f"{loss_cls.__name__} returned non-finite: {out}"


# ---------- gradient flow ----------

@pytest.mark.parametrize("loss_cls,kwargs", [
    (SCELoss, dict(num_classes=NUM_CLASSES, alpha=0.1, beta=1.0)),
    (GCELoss, dict(num_classes=NUM_CLASSES, q=0.7)),
    (LabelSmoothingLoss, dict(num_classes=NUM_CLASSES, smoothing=0.1)),
])
def test_gradient_flows(loss_cls, kwargs, logits_targets):
    logits, targets = logits_targets
    loss = loss_cls(**kwargs)
    out = loss(logits, targets)
    out.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all(), f"{loss_cls.__name__} produced non-finite gradient"
    assert logits.grad.abs().sum() > 0, f"{loss_cls.__name__} gradient is all zero"


# ---------- numerical stability at extremes ----------

def test_sce_extreme_confident_correct():
    """Very confident correct prediction: SCE should be finite, small."""
    logits = torch.zeros(1, NUM_CLASSES)
    logits[0, 3] = 20.0
    targets = torch.tensor([3])
    loss = SCELoss(num_classes=NUM_CLASSES, alpha=0.1, beta=1.0)
    out = loss(logits, targets)
    assert torch.isfinite(out)


def test_sce_extreme_confident_wrong():
    """Very confident wrong prediction: SCE should be finite (clamp prevents inf)."""
    logits = torch.zeros(1, NUM_CLASSES)
    logits[0, 3] = 20.0
    targets = torch.tensor([7])
    loss = SCELoss(num_classes=NUM_CLASSES, alpha=0.1, beta=1.0)
    out = loss(logits, targets)
    assert torch.isfinite(out), f"SCE should not inf on confident-wrong, got {out}"


def test_gce_extreme_confident_wrong():
    logits = torch.zeros(1, NUM_CLASSES)
    logits[0, 3] = 20.0
    targets = torch.tensor([7])
    loss = GCELoss(num_classes=NUM_CLASSES, q=0.7)
    out = loss(logits, targets)
    assert torch.isfinite(out)
    # For q in (0,1], GCE is bounded in [0, 1/q]
    assert 0.0 <= out.item() <= 1.0 / 0.7 + 1e-4


def test_label_smoothing_matches_formula():
    """LabelSmoothing(smoothing=0) should equal CE."""
    torch.manual_seed(1)
    logits = torch.randn(BATCH, NUM_CLASSES)
    targets = torch.randint(0, NUM_CLASSES, (BATCH,))
    ls = LabelSmoothingLoss(num_classes=NUM_CLASSES, smoothing=0.0)
    ce = F.cross_entropy(logits, targets)
    assert torch.allclose(ls(logits, targets), ce, atol=1e-5)


def test_gce_approaches_mae_as_q_to_one():
    """GCE with q=1 should be (1 - p_y), which equals MAE on one-hot."""
    torch.manual_seed(2)
    logits = torch.randn(BATCH, NUM_CLASSES)
    targets = torch.randint(0, NUM_CLASSES, (BATCH,))
    gce = GCELoss(num_classes=NUM_CLASSES, q=1.0)
    out = gce(logits, targets)
    pred = F.softmax(logits, dim=-1)
    p_y = pred[torch.arange(BATCH), targets]
    expected = (1.0 - p_y).mean()
    assert torch.allclose(out, expected, atol=1e-5)


# ---------- ELR (index-aware) ----------

def test_elr_shape_and_grad():
    torch.manual_seed(3)
    n = 32
    logits = torch.randn(BATCH, NUM_CLASSES, requires_grad=True)
    targets = torch.randint(0, NUM_CLASSES, (BATCH,))
    indices = torch.randint(0, n, (BATCH,))
    loss = ELRLoss(num_samples=n, num_classes=NUM_CLASSES, beta=0.9, lam=3.0, cpu_buffer=False)
    out = loss(logits, targets, indices)
    assert out.ndim == 0
    assert torch.isfinite(out)
    out.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_elr_buffer_updates():
    """After a forward, target_estimates at seen indices should be non-zero."""
    torch.manual_seed(4)
    n = 16
    logits = torch.randn(BATCH, NUM_CLASSES)
    targets = torch.randint(0, NUM_CLASSES, (BATCH,))
    indices = torch.arange(BATCH)
    loss = ELRLoss(num_samples=n, num_classes=NUM_CLASSES, beta=0.9, lam=3.0, cpu_buffer=False)
    assert loss.target_estimates[indices].abs().sum().item() == 0.0
    loss(logits, targets, indices)
    assert loss.target_estimates[indices].abs().sum().item() > 0.0
    # untouched indices remain zero
    untouched = torch.arange(BATCH, n)
    assert loss.target_estimates[untouched].abs().sum().item() == 0.0


def test_elr_cpu_buffer_parity():
    """cpu_buffer=True and False should give same loss value (up to fp noise)."""
    torch.manual_seed(5)
    n = 16
    logits = torch.randn(BATCH, NUM_CLASSES)
    targets = torch.randint(0, NUM_CLASSES, (BATCH,))
    indices = torch.arange(BATCH)
    loss_gpu = ELRLoss(num_samples=n, num_classes=NUM_CLASSES, beta=0.9, lam=3.0, cpu_buffer=False)
    loss_cpu = ELRLoss(num_samples=n, num_classes=NUM_CLASSES, beta=0.9, lam=3.0, cpu_buffer=True)
    out_gpu = loss_gpu(logits, targets, indices)
    out_cpu = loss_cpu(logits, targets, indices)
    assert torch.allclose(out_gpu, out_cpu, atol=1e-5)


# ---------- CompositeLoss ----------

def test_composite_sum_of_weighted():
    """CompositeLoss output should equal sum of w_i * L_i."""
    torch.manual_seed(6)
    logits = torch.randn(BATCH, NUM_CLASSES)
    targets = torch.randint(0, NUM_CLASSES, (BATCH,))
    sce = SCELoss(num_classes=NUM_CLASSES, alpha=0.1, beta=1.0)
    gce = GCELoss(num_classes=NUM_CLASSES, q=0.7)
    expected = 0.3 * sce(logits, targets) + 0.7 * gce(logits, targets)
    composite = CompositeLoss([sce, gce], weights=[0.3, 0.7])
    out = composite(logits, targets)
    assert torch.allclose(out, expected, atol=1e-6)


def test_composite_forwards_indices_only_to_elr():
    """CompositeLoss should pass `indices` kwarg only to losses that accept it."""
    torch.manual_seed(7)
    n = 16
    logits = torch.randn(BATCH, NUM_CLASSES, requires_grad=True)
    targets = torch.randint(0, NUM_CLASSES, (BATCH,))
    indices = torch.arange(BATCH)
    sce = SCELoss(num_classes=NUM_CLASSES, alpha=0.1, beta=1.0)
    elr = ELRLoss(num_samples=n, num_classes=NUM_CLASSES, beta=0.9, lam=3.0, cpu_buffer=False)
    composite = CompositeLoss([sce, elr], weights=[1.0, 1.0])
    out = composite(logits, targets, indices=indices)
    assert torch.isfinite(out)
    out.backward()
    assert torch.isfinite(logits.grad).all()


# ---------- build_loss factory ----------

def _cfg(loss_name, num_classes=NUM_CLASSES, loss_params=None):
    return {
        "model": {"num_classes": num_classes},
        "training": {"loss": loss_name, "loss_params": loss_params or {}, "label_smoothing": 0.0},
    }


@pytest.mark.parametrize("loss_name", ["ce", "label_smoothing", "sce", "gce", "peer_loss"])
def test_build_loss_basic(loss_name, logits_targets):
    logits, targets = logits_targets
    criterion = build_loss(_cfg(loss_name))
    out = criterion(logits, targets)
    assert out.ndim == 0
    assert torch.isfinite(out)


def test_build_loss_elr_requires_num_samples():
    criterion = build_loss(_cfg("elr"), num_samples=32)
    logits = torch.randn(BATCH, NUM_CLASSES)
    targets = torch.randint(0, NUM_CLASSES, (BATCH,))
    indices = torch.arange(BATCH)
    out = criterion(logits, targets, indices)
    assert torch.isfinite(out)


def test_build_loss_composite():
    cfg = _cfg("composite", loss_params={
        "losses": [
            {"name": "sce", "weight": 0.3, "params": {"alpha": 0.1, "beta": 1.0}},
            {"name": "gce", "weight": 0.7, "params": {"q": 0.7}},
        ]
    })
    criterion = build_loss(cfg)
    logits = torch.randn(BATCH, NUM_CLASSES)
    targets = torch.randint(0, NUM_CLASSES, (BATCH,))
    out = criterion(logits, targets)
    assert torch.isfinite(out)


def test_build_loss_unknown_raises():
    with pytest.raises(ValueError, match="Unknown loss"):
        build_loss(_cfg("not_a_real_loss"))
