"""
measure_alpha_variance.py -- calibrates BOTH alpha_marg and alpha_int on
per-step variance, not full-episode sums. See the reasoning in chat: GAE's
effective horizon is ~1/(1-gamma*lambda) ~= 16.8 steps, so what reaches A_t
is a LOCAL, horizon-weighted contribution, not a full-3000-step sum -- for
a dense reward that's much closer to its std than its total sum.

Extends the user's own measure_alpha_variance.py (correct approach, D_marg
only) to also cover D_int, matching the dual-reward design.

Run:  python measure_alpha_variance.py --episodes 5
"""
import argparse, os, sys
import numpy as np
import torch

PROJECT_ROOT = os.path.expanduser("~/dissertation/context-aware-magail-grf")
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "discriminator"))

from finetune_loop import (build_env, collect_rollout, ACTOR_PATH, CRITIC_PATH,
                           DISC_CKPT, NORMALISER_PATH)
from context_conditioned_policy import ContextConditionedActor, ContextConditionedCritic
from finetune_objective import compute_style_reward
from discriminator_model import Discriminator as MargDiscriminator
from feature_normalization import FeatureNormaliser
from enhanced_LightActionMask_5 import FeatureEncoder
from discriminator_candidates import TinyInteract

D_INT_CKPT = os.path.join(PROJECT_ROOT, "src/discriminator/D_int_C_production.pt")
GAMMA, GAE_LAMBDA = 0.99, 0.95
EFFECTIVE_HORIZON = 1.0 / (1.0 - GAMMA * GAE_LAMBDA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    args = ap.parse_args()

    actor = ContextConditionedActor(torch.load(ACTOR_PATH, map_location="cpu")); actor.eval()
    critic = ContextConditionedCritic(torch.load(CRITIC_PATH, map_location="cpu")); critic.eval()

    D_marg = MargDiscriminator()
    D_marg.load_state_dict(torch.load(DISC_CKPT, map_location="cpu")["model_state_dict"])
    D_marg.eval()

    D_int = TinyInteract()
    D_int.load_state_dict(torch.load(D_INT_CKPT, map_location="cpu")["model_state_dict"])
    D_int.eval()

    norm = FeatureNormaliser(); norm.load(NORMALISER_PATH)
    env, enc = build_env(), FeatureEncoder()

    task_all, marg_all, int_all = [], [], []
    for ep in range(args.episodes):
        b = collect_rollout(actor, critic, enc, env, norm)
        feats = torch.as_tensor(b["disc_feat"], dtype=torch.float32)
        r_marg = compute_style_reward(D_marg, feats).numpy()
        r_int = compute_style_reward(D_int, feats).numpy()
        task_all.append(b["rewards"].numpy()); marg_all.append(r_marg); int_all.append(r_int)
        print(f"  ep{ep}: task std={b['rewards'].numpy().std():.5f}  "
              f"marg std={r_marg.std():.5f}  int std={r_int.std():.5f}")
    env.close()

    t = np.concatenate(task_all)
    m = np.concatenate(marg_all)
    i = np.concatenate(int_all)

    print(f"\nPER-STEP task std: {t.std():.5f}  (mean {t.mean():+.5f})")
    print(f"PER-STEP marg std: {m.std():.5f}")
    print(f"PER-STEP int  std: {i.std():.5f}")

    alpha_marg_var = t.std() / m.std()
    alpha_int_var = t.std() / i.std()
    print(f"\nvariance-matched alpha_marg = {alpha_marg_var:.4f}")
    print(f"variance-matched alpha_int  = {alpha_int_var:.4f}")

    print(f"\nGAE effective horizon ~= {EFFECTIVE_HORIZON:.1f} steps -- this is why")
    print(f"variance (not full-episode sum) is the right quantity: only a local")
    print(f"~{EFFECTIVE_HORIZON:.0f}-step window of the dense style stream reaches any given A_t.")

    print(f"\ncompare against the SUM-based estimates already in hand:")
    print(f"  sum-based   alpha_marg ~= 0.0033   alpha_int ~= 0.0021")
    print(f"  variance-based alpha_marg = {alpha_marg_var:.4f}   alpha_int = {alpha_int_var:.4f}")
    print(f"\nUSE BOTH as the two bracket points for the pilot, per the original plan:")
    print(f"  pilot setting 1 (conservative floor): ALPHA_MARG=0.0033, ALPHA_INT=0.0021")
    print(f"  pilot setting 2 (variance-matched):    ALPHA_MARG={alpha_marg_var:.4f}, ALPHA_INT={alpha_int_var:.4f}")
    print(f"  run ~150 iters of MAGAIL-C at each, watch win-rate (should hold near 0.67)")
    print(f"  and SAP (should move off 85.9%) -- neither derivation is exact enough to")
    print(f"  trust without an empirical check.")


if __name__ == "__main__":
    main()