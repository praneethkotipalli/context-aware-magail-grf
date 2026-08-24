"""test_discriminator.py -- FiLM version"""
import torch
from discriminator_model import Discriminator

model = Discriminator()

n_params = sum(p.numel() for p in model.parameters())
print(f"parameter count: {n_params}  (expected 167809)")
assert n_params == 167809, f"got {n_params}"

x = torch.randn(8, 137)
logits = model(x)
print(f"logits shape: {tuple(logits.shape)}  (expected (8,))")
assert logits.shape == (8,)

probs = model.probability(x)
print(f"prob range: [{probs.min():.4f}, {probs.max():.4f}]")
assert (probs > 0).all() and (probs < 1).all()

x_grad = torch.randn(4, 137, requires_grad=True)
out = model(x_grad)
grad = torch.autograd.grad(out.sum(), x_grad)[0]
print(f"input-gradient shape: {tuple(grad.shape)}, finite: {torch.isfinite(grad).all().item()}")
assert grad.shape == (4, 137) and torch.isfinite(grad).all()

try:
    model(torch.randn(4, 100))
    print("FAIL -- should have rejected wrong width")
except ValueError:
    print("correctly rejected wrong input width")

untrained_probs = model.probability(torch.randn(500, 137))
print(f"untrained mean probability: {untrained_probs.mean():.4f} (want ~0.5)")

print("\n--- FiLM-specific: context must have ZERO effect at init ---")
content = torch.randn(50, 135)
ctx_a = torch.tensor([[0.1, 2.0]] * 50)    # late-winning-ish
ctx_b = torch.tensor([[0.85, -2.0]] * 50)  # early-losing-ish
out_a = model(torch.cat([content, ctx_a], dim=1))
out_b = model(torch.cat([content, ctx_b], dim=1))
max_diff = (out_a - out_b).abs().max().item()
print(f"max |output_a - output_b| with SAME content, DIFFERENT context: {max_diff:.8f}")
assert max_diff < 1e-6, "FiLM generators are not zero-initialized correctly -- context leaking through at init"
print("PASS -- confirmed context has NO effect before any training, as designed")

print("\nALL CHECKS PASSED")