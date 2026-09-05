"""measure_alpha_variance.py -- calibrate alpha on per-step VARIANCE, not sums."""
import os, sys, numpy as np, torch
PROJECT_ROOT = os.path.expanduser("~/dissertation/context-aware-magail-grf")
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "discriminator"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "grf_baseline"))
sys.path.insert(0, os.path.expanduser("~/dissertation/context-aware-magail-grf/src/policy"))
from finetune_loop import (build_env, collect_rollout, build_disc_input,
                            ACTOR_PATH, CRITIC_PATH, DISC_CKPT, NORMALISER_PATH)
from context_conditioned_policy import ContextConditionedActor, ContextConditionedCritic
from finetune_objective import compute_style_reward
from discriminator_model import Discriminator
from feature_normalization import FeatureNormaliser
from enhanced_LightActionMask_5 import FeatureEncoder

actor = ContextConditionedActor(torch.load(ACTOR_PATH, map_location="cpu"))
critic = ContextConditionedCritic(torch.load(CRITIC_PATH, map_location="cpu"))
disc = Discriminator(); disc.load_state_dict(torch.load(DISC_CKPT, map_location="cpu")["model_state_dict"]); disc.eval()
norm = FeatureNormaliser(); norm.load(NORMALISER_PATH)
env, enc = build_env(), FeatureEncoder()

task_all, style_all = [], []
for ep in range(3):
    b = collect_rollout(actor, critic, enc, env, norm)
    rs = compute_style_reward(disc, torch.as_tensor(b["disc_feat"], dtype=torch.float32)).numpy()
    task_all.append(b["rewards"].numpy()); style_all.append(rs)
    print(f"ep{ep}: task std={b['rewards'].numpy().std():.4f}  style std={rs.std():.4f}")
env.close()

t, s = np.concatenate(task_all), np.concatenate(style_all)
print(f"\nPER-STEP task  std: {t.std():.5f}  (mean {t.mean():+.5f})")
print(f"PER-STEP style std: {s.std():.5f}  (mean {s.mean():+.5f})")
print(f"\nalpha (variance-matched) = {t.std()/s.std():.4f}")
print(f"current alpha = 0.001  -> off by {(t.std()/s.std())/0.001:.0f}x")
print("\nAdvantage normalisation removes constant offsets, so per-step VARIANCE")
print("is what reaches the policy gradient -- not accumulated sums.")