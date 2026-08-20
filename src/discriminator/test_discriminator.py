import torch
import numpy as np
from discriminator_model import Discriminator, INPUT_DIM


def run():
    d = Discriminator()

    n_params = sum(p.numel() for p in d.parameters())
    expected = (137*256 + 256) + (256*256 + 256) + (256*1 + 1)
    print(f"parameter count: {n_params}  (expected {expected})")
    assert n_params == expected, "architecture does not match 137->256->256->1"

    x = torch.randn(8, INPUT_DIM)
    logits = d(x)
    probs = d.probability(x)

    print(f"logits shape: {tuple(logits.shape)}  (expected (8,))")
    assert logits.shape == (8,)
    print(f"probs shape:  {tuple(probs.shape)}  (expected (8,))")
    assert probs.shape == (8,)

    assert torch.all((probs > 0) & (probs < 1)), "probabilities outside (0,1)"
    print(f"prob range: [{probs.min():.4f}, {probs.max():.4f}] -- all within (0,1)")

    # gradients must flow w.r.t. the INPUT, not just the weights --
    # the R1 penalty differentiates the logit w.r.t. x, so this must work
    x_grad = torch.randn(4, INPUT_DIM, requires_grad=True)
    out = d(x_grad).sum()
    grad = torch.autograd.grad(out, x_grad, create_graph=True)[0]
    print(f"input-gradient shape: {tuple(grad.shape)}  (expected (4, {INPUT_DIM}))")
    assert grad.shape == (4, INPUT_DIM)
    assert torch.isfinite(grad).all(), "non-finite input gradients"
    print("input gradients finite and correctly shaped -- R1 penalty will work")

    # wrong input width must fail loudly rather than silently broadcasting
    try:
        d(torch.randn(4, 100))
        raise AssertionError("should have rejected a 100-dim input")
    except ValueError as e:
        print(f"correctly rejected wrong input width: {e}")

    # untrained network should sit near P=0.5, not saturated at either extreme
    x_many = torch.randn(512, INPUT_DIM)
    p = d.probability(x_many)
    print(f"untrained mean probability: {p.mean():.4f} (want roughly 0.5, "
          f"not saturated near 0 or 1)")

    print("\nALL CHECKS PASSED -- discriminator.py verified")


if __name__ == '__main__':
    run()