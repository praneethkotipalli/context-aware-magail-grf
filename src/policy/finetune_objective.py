"""
finetune_objective.py -- three-term objective, Section 3.4.1.
alpha=0.1: PROVISIONAL placeholder, same status as eta -- not tuned,
revisit once real r_style vs. task-reward magnitudes are observed from
an actual pilot rollout.
"""

import torch

ALPHA = 0.1        # provisional
LAMBDA_KL = 1.0     # locked starting value, Section 3.4.3 -- anchor dominates initially
CLIP_EPS = 0.2       # locked, Section 3.4.1


def compute_style_reward(discriminator, features):
    """r_style = logit(D) = log D - log(1-D), bias-neutral form, 3.4.2.
    discriminator.forward() already returns the pre-sigmoid logit --
    this IS log D - log(1-D) directly, no extra computation needed."""
    with torch.no_grad():
        return discriminator(features)


def three_term_loss(new_log_probs, old_log_probs, advantages, r_style, dist_theta, dist_star):
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages
    clipped_surrogate = torch.min(surr1, surr2).mean()

    style_term = ALPHA * r_style.mean()

    kl = torch.distributions.kl_divergence(dist_theta, dist_star).mean()
    kl_term = LAMBDA_KL * kl

    total = -(clipped_surrogate + style_term) + kl_term  # negative surrogate -- minimizing
    return total, {
        "clipped_surrogate": clipped_surrogate.item(),
        "style_term": style_term.item(),
        "kl": kl.item(),
        "r_style_mean": r_style.mean().item(),
        "r_style_sum_per_episode_estimate": (r_style.mean() * 3000).item(),  # rough, for the alpha sanity check
    }