"""
finetune_objective.py -- CORRECTED.

TWO CRITICAL FIXES vs. the previous version:

FIX A -- style reward had ZERO GRADIENT.
  Previously: style_term = alpha * r_style.mean(), added to the loss.
  r_style is computed under no_grad, so this was a CONSTANT -- derivative
  w.r.t. theta exactly zero. The policy never received any style signal.
  This is why SAP did not move across 160 iterations.
  Now: r_style is folded into the per-step REWARD before GAE (see
  finetune_loop.py), so it is credited to actions through the advantage.
  That is the correct GAIL formulation and what "alpha * E_t[r_style]"
  in Section 3.4.1 actually means as an expected return.
  three_term_loss no longer contains a style term -- it is already
  inside `advantages`.

FIX B -- KL was Infinity, contributing no gradient.
  torch.distributions.kl_divergence(Categorical, Categorical) does
  `t[(q.probs == 0)] = inf` internally. Action masks set logits to -1e10,
  underflowing probs to exactly 0; because pi_theta and pi_star have
  slightly different softmax normalisers, one side can land on exact zero
  while the other lands on a denormal, tripping that inf branch. The
  in-place constant assignment carries ZERO gradient, so the KL anchor
  has been inert this whole time.
  Now: computed manually from log-probs. For masked actions p == 0 and
  (log_p - log_q) is finite, so their contribution is exactly 0 with no
  special-casing needed. nan_to_num retained as a guard.
"""

import torch

ALPHA = 0.001          # measured (measure_alpha.py). Same value, now applied
                        # to the per-step reward rather than the loss.
LAMBDA_KL_INIT = 1.0    # locked, Section 3.4.3
CLIP_EPS = 0.2          # locked, Section 3.4.1
BASELINE_WIN_RATE = 0.624
STYLE_CLIP = 5.0   # logit(D) guard. With 0.9/0.1 label smoothing the optimal logit is ~±2.2

ALPHA = 0.25       # PROVISIONAL -- replace with measure_alpha_offline.py's exact output.

def compute_style_reward(discriminator, features, running_mean=None):
    """
    Centers the reward to shorten the critic's recalibration transient.
    Advantage normalization already removes constant offsets, so only variance 
    reaches the policy gradient.
    """
    with torch.no_grad():
        r = discriminator(features).clamp(-STYLE_CLIP, STYLE_CLIP)
        r = r - (r.mean() if running_mean is None else running_mean)
    return r

def safe_categorical_kl(dist_theta, dist_star):
    """KL(pi_theta || pi_star), manual, avoiding torch's inf-override on
    zero-probability (masked) actions. Categorical.logits are already
    log-softmax normalised."""
    log_p = dist_theta.logits
    log_q = dist_star.logits
    p = log_p.exp()
    kl_per_state = (p * (log_p - log_q)).sum(-1)
    kl_per_state = torch.nan_to_num(kl_per_state, nan=0.0, posinf=0.0, neginf=0.0)
    return kl_per_state.mean()


def three_term_loss(new_log_probs, old_log_probs, advantages, dist_theta, dist_star,
                     lam=LAMBDA_KL_INIT, use_kl=True):
    """
    advantages MUST already include the style reward (folded in before
    GAE). No separate style term here -- see FIX A.

    use_kl=False -> KL anchor off entirely (MAGAIL-C / -SC / -NC).
    use_kl=True  -> MAGAIL-C+KL.
    """
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages
    clipped_surrogate = torch.min(surr1, surr2).mean()

    kl = safe_categorical_kl(dist_theta, dist_star)
    kl_term = lam * kl if use_kl else torch.zeros((), device=kl.device)

    total = -clipped_surrogate + kl_term
    return total, {
        "clipped_surrogate": clipped_surrogate.item(),
        "kl": kl.item(),
        "kl_term": kl_term.item(),
        "lam": lam if use_kl else 0.0,
    }


class AlignmentTaxScheduler:
    """Section 3.4.3. Expects a REAL multi-episode win rate."""
    def __init__(self, baseline_win_rate=BASELINE_WIN_RATE, lambda_init=LAMBDA_KL_INIT,
                 anneal_factor=10.0, anneal_patience=5,
                 kill_switch_drop=0.25, hold_threshold=0.05):
        self.baseline_win_rate = baseline_win_rate
        self.lam = lambda_init
        self.anneal_factor = anneal_factor
        self.anneal_patience = anneal_patience
        self.kill_switch_drop = kill_switch_drop
        self.hold_threshold = hold_threshold
        self._hold_count = 0
        self.halted = False
        self.halt_reason = None
        self.history = []

    def update(self, current_win_rate):
        self.history.append(current_win_rate)
        if self.halted:
            return self.lam, True, self.halt_reason

        drop = (self.baseline_win_rate - current_win_rate) / max(self.baseline_win_rate, 1e-9)
        if drop > self.kill_switch_drop:
            self.halted = True
            self.halt_reason = (f"win rate {current_win_rate:.3f} is {drop*100:.1f}% below "
                                 f"baseline {self.baseline_win_rate:.3f} (>25% kill-switch)")
            return self.lam, True, self.halt_reason

        self._hold_count = self._hold_count + 1 if drop <= self.hold_threshold else 0
        if self._hold_count >= self.anneal_patience:
            old = self.lam
            self.lam /= self.anneal_factor
            self._hold_count = 0
            print(f"  [scheduler] annealing lambda {old:.5f} -> {self.lam:.5f}")
        return self.lam, False, None