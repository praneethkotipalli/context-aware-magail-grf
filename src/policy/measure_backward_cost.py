"""measure_backward_cost.py -- v2, with warmup and order check."""
import os, sys, time, copy
import torch

GRF_MARL_ROOT = os.path.expanduser("~/dissertation/GRF_MARL")
PROJECT_ROOT = os.path.expanduser("~/dissertation/context-aware-magail-grf")
sys.path.insert(0, GRF_MARL_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "policy"))

from context_conditioned_policy import ContextConditionedActor, ContextConditionedCritic
from finetune_objective import three_term_loss

ACTOR_PATH = os.path.join(GRF_MARL_ROOT, "light_malib/trained_models/gr_football/5_vs_5/PassingMain_v2/actor.pt")
CRITIC_PATH = os.path.join(GRF_MARL_ROOT, "light_malib/trained_models/gr_football/5_vs_5/PassingMain_v2/critic.pt")

N_STEPS = 3000 * 4


def time_update(train_base: bool, n_warmup=3, n_reps=10):
    frozen_actor = torch.load(ACTOR_PATH, map_location="cpu")
    frozen_critic = torch.load(CRITIC_PATH, map_location="cpu")
    actor = ContextConditionedActor(frozen_actor)
    critic = ContextConditionedCritic(frozen_critic)
    actor_star = copy.deepcopy(actor)
    for p in actor_star.parameters():
        p.requires_grad_(False)

    if train_base:
        params = list(actor.parameters()) + list(critic.parameters())
    else:
        params = list(actor.context_net.parameters()) + list(critic.context_net.parameters())
        # explicitly freeze everything else so autograd genuinely skips it
        for p in actor.base.parameters(): p.requires_grad_(False)
        for p in actor.out.parameters(): p.requires_grad_(False)
        for p in critic.base.parameters(): p.requires_grad_(False)
        for p in critic.out.parameters(): p.requires_grad_(False)

    n_trainable = sum(p.numel() for p in params if p.requires_grad)
    optimizer = torch.optim.Adam(params, lr=3e-4)

    obs = torch.randn(N_STEPS, 192)
    ctx = torch.randn(N_STEPS, 2)
    mask = torch.ones(N_STEPS, 19)
    actions = torch.randint(0, 19, (N_STEPS,))
    old_log_probs = torch.randn(N_STEPS)
    advantages = torch.randn(N_STEPS)
    r_style = torch.randn(N_STEPS)

    def one_update():
        _, new_lp, _, dist_theta = actor(obs, ctx, mask, explore=False, actions=actions)
        with torch.no_grad():
            _, _, _, dist_star = actor_star(obs, torch.zeros_like(ctx), mask, explore=False, actions=actions)
        loss, _ = three_term_loss(new_lp, old_log_probs, advantages, r_style, dist_theta, dist_star)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for _ in range(n_warmup):
        one_update()

    times = []
    for _ in range(n_reps):
        t0 = time.time()
        one_update()
        times.append(time.time() - t0)

    times.sort()
    return n_trainable, sum(times) / len(times), times[len(times) // 2]


print("=== ORDER A: context_net first, then full ===")
for label, tb in [("context_net ONLY", False), ("base + out + context_net", True)]:
    n, avg, med = time_update(tb)
    print(f"{label}: {n:,} params -- avg {avg:.4f}s, median {med:.4f}s")

print("\n=== ORDER B: full first, then context_net (checks ordering effects) ===")
for label, tb in [("base + out + context_net", True), ("context_net ONLY", False)]:
    n, avg, med = time_update(tb)
    print(f"{label}: {n:,} params -- avg {avg:.4f}s, median {med:.4f}s")