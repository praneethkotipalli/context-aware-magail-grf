"""
test_discriminator_loss.py

Hand-verified checks for discriminator_loss.py, in the same spirit as
test_feature_derivation.py and the discriminator_model.py verification
already done (param count, finite gradients, near-0.5 untrained output,
wrong-shape rejection).

Uses a small stand-in discriminator with the same 137 -> 256 -> 256 -> 1
architecture. Swap the import below for the real discriminator_model.py
when running this against the actual model -- the stub exists only so
this file has zero dependency on where you're running it (Mac, no GRF
needed, matches the portable .npz-cache workflow).

Run: python test_discriminator_loss.py
"""

import torch
import torch.nn as nn

from discriminator_loss import DiscriminatorLoss, DEFAULT_ETA


INPUT_DIM = 137


class StubDiscriminator(nn.Module):
    """Mirrors discriminator_model.py's contract: forward() returns the
    pre-sigmoid logit, shape (B, 1). Replace with the real import to
    re-run these checks against the actual trained-parameter model."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x)


class ConstantGradModel(nn.Module):
    """logit = scale * sum(x). Gradient w.r.t. x is a known constant vector
    (all entries = scale), so ||grad||^2 = scale^2 * INPUT_DIM exactly.
    Used to verify the R1 formula against a hand-computable value, not
    just check that it runs."""

    def __init__(self, scale=0.1):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        return (self.scale * x.sum(dim=1, keepdim=True))


def check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}")
    assert condition, f"FAILED: {name}"


def main():
    torch.manual_seed(0)
    B = 32

    print("1. Basic forward/backward on stub discriminator")
    disc = StubDiscriminator()
    loss_fn = DiscriminatorLoss(eta=DEFAULT_ETA)

    expert_batch = torch.randn(B, INPUT_DIM)
    agent_batch = torch.randn(B, INPUT_DIM)

    out = loss_fn(disc, expert_batch, agent_batch)
    check("total_loss is a finite scalar", torch.isfinite(out.total_loss).all())
    check("total_loss requires grad", out.total_loss.requires_grad)
    check("bce_expert finite", torch.isfinite(out.bce_expert))
    check("bce_agent finite", torch.isfinite(out.bce_agent))
    check("r1_penalty finite and >= 0", torch.isfinite(out.r1_penalty) and out.r1_penalty >= 0)
    check("accuracy in [0,1]", 0.0 <= out.accuracy <= 1.0)

    out.total_loss.backward()
    grads_ok = all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in disc.parameters()
    )
    check("all discriminator params received finite gradients", grads_ok)

    print("\n2. Untrained discriminator sanity (near-random init)")
    disc2 = StubDiscriminator()
    out2 = loss_fn(disc2, torch.randn(200, INPUT_DIM), torch.randn(200, INPUT_DIM))
    # Not a strict bound -- random MLP init can land anywhere -- just
    # checking it isn't already saturated (>0.95) or degenerate (0).
    check(
        f"untrained accuracy plausible (got {out2.accuracy:.3f}, expect roughly 0.3-0.7)",
        0.2 <= out2.accuracy <= 0.8,
    )

    print("\n3. R1 penalty matches hand-computed value (ConstantGradModel)")
    scale = 0.1
    cg_model = ConstantGradModel(scale=scale)
    r1, _ = loss_fn.r1_penalty(cg_model, torch.randn(16, INPUT_DIM))
    expected_r1 = (DEFAULT_ETA / 2.0) * (scale ** 2) * INPUT_DIM
    check(
        f"R1 == (eta/2)*scale^2*dim  (got {r1.item():.4f}, expected {expected_r1:.4f})",
        abs(r1.item() - expected_r1) < 1e-4,
    )

    print("\n4. R1 scales linearly with eta")
    r1_eta1, _ = DiscriminatorLoss(eta=1.0).r1_penalty(cg_model, torch.randn(16, INPUT_DIM))
    r1_eta4, _ = DiscriminatorLoss(eta=4.0).r1_penalty(cg_model, torch.randn(16, INPUT_DIM))
    ratio = (r1_eta4 / r1_eta1).item()
    check(f"r1(eta=4)/r1(eta=1) == 4.0 (got {ratio:.4f})", abs(ratio - 4.0) < 1e-3)

    print("\n5. Accuracy metric on a perfectly separable synthetic batch")

    class PerfectSeparator(nn.Module):
        def forward(self, x):
            # first feature > 0 => "expert-like" large positive logit
            return (x[:, 0:1] * 1000.0)

    perfect = PerfectSeparator()
    expert_sep = torch.ones(10, INPUT_DIM)   # feature[0] = 1  -> large +logit
    agent_sep = -torch.ones(10, INPUT_DIM)   # feature[0] = -1 -> large -logit
    out5 = loss_fn(perfect, expert_sep, agent_sep)
    check(f"perfectly separable batch => accuracy == 1.0 (got {out5.accuracy:.3f})", out5.accuracy.item() == 1.0)

    print("\n6. Label ordering validation")
    raised = False
    try:
        DiscriminatorLoss(expert_label=0.1, agent_label=0.9)  # swapped, should reject
    except ValueError:
        raised = True
    check("swapped expert/agent labels correctly rejected", raised)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()