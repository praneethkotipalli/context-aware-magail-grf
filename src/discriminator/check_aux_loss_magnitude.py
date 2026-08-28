"""check_aux_loss_magnitude.py -- sanity check BEFORE any real training run."""
import torch
from discriminator_model import Discriminator
from discriminator_loss import DiscriminatorLoss, DEFAULT_BETA_AUX

torch.manual_seed(0)
model = Discriminator()

B = 128
expert_batch = torch.randn(B, 139)
agent_batch = torch.randn(B, 139)

for beta in [0.0, 1.0, DEFAULT_BETA_AUX, 5.0]:
    loss_fn = DiscriminatorLoss(beta_aux=beta)
    out = loss_fn(model, expert_batch, agent_batch)
    aux = out.aux_loss.item() if out.aux_loss is not None else 0.0
    bce_total = out.bce_expert.item() + out.bce_agent.item()
    print(f"beta_aux={beta:>4.1f}  bce_total={bce_total:.4f}  "
          f"aux_loss(raw)={aux:.4f}  beta*aux={beta*aux:.4f}  "
          f"aux_share={beta*aux/(bce_total + out.r1_penalty.item() + beta*aux)*100:.1f}%")