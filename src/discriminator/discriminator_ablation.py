"""
discriminator_ablation.py

Controlled ablation of the discriminator's context-sensitivity design.
Replaces the retrospective development narrative (six attempts, on two
DIFFERENT gate definitions, with checkpoints overwritten) with a
reproducible one-variable-at-a-time experiment on a single fixed gate.

WHY THIS EXISTS: the development sequence cannot be reported as a clean
progression -- attempts 1-2 used late-winning <-> EARLY-losing, attempts
3-6 used late-winning <-> LATE-losing, and the final configuration
changed three things simultaneously (swap loss wired, eta 1.0->0.3,
weight decay added). This script holds everything constant except one
element per comparison.

FIVE VARIANTS, all at 139 dims, same seed, same data, same step budget:

  D1  concat  conditioning, no swap loss, eta=1.0     <- the original design
  D2  FiLM    conditioning, no swap loss, eta=1.0
  D3  FiLM    conditioning, swap loss g=6.0, eta=1.0
  D4  FiLM    conditioning, swap loss g=6.0, eta=0.3  <- the shipped config
  D5  concat  conditioning, swap loss g=6.0, eta=0.3

ADJACENT COMPARISONS -- each isolates exactly one factor:
  D1 vs D2 : does FiLM conditioning help, holding the objective fixed?
  D2 vs D3 : does the swap-consistency loss help, holding architecture fixed?
  D3 vs D4 : does relaxing the R1 penalty help? (R1 and the swap loss are in
             direct tension -- R1 penalises input gradients, the swap loss
             requires them)
  D4 vs D5 : is FiLM still NECESSARY once the swap loss is present, or does
             the objective do the work regardless of conditioning mechanism?
             (Genuinely open -- if D5 matches D4, the architectural claim
             weakens and that must be reported.)

Run:   python discriminator_ablation.py --variants all
       python discriminator_ablation.py --variants D1 D2   # split across machines
Output: discriminator_ablation_results.json, per-variant checkpoints,
        per-block gradient diagnostics for the dilution figure.
"""

import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn

from discriminator_loss import DiscriminatorLoss
from discriminator_trainer import DiscriminatorTrainer
from context_balanced_sampler import BalancedContextSampler, balanced_batch, sqrt_scaled_target
from context_shift_scoring import select_by_true_context, LATE_WINNING, LATE_LOSING
from counterfactual_gate import run_counterfactual_gate
from held_out_split import make_episode_split, split_features_by_episode
from feature_derivation import BLOCK_SLICES

INPUT_DIM, CONTEXT_DIM, HIDDEN_DIM, FILM_HIDDEN = 139, 2, 256, 64
CONTENT_DIM = INPUT_DIM - CONTEXT_DIM

EXPERT_CACHE = "expert_features_cache.npz"
RANDOM_CACHE = "random_policy_features_cache.npz"
MAPPO_CACHE = "mappo_features_cache.npz"

PHASE_A_STEPS = 10_000
PHASE_B_STEPS = 10_000
BATCH_SIZE = 128
SEED = 0

VARIANTS = {
    # name: (conditioning, gamma_swap, eta)
    "D1": ("concat", 0.0, 1.0),
    "D2": ("film",   0.0, 1.0),
    "D3": ("film",   6.0, 1.0),
    "D4": ("film",   6.0, 0.3),
    "D5": ("concat", 6.0, 0.3),
}


class FiLMGenerator(nn.Module):
    def __init__(self, context_dim=CONTEXT_DIM, hidden_dim=HIDDEN_DIM, film_hidden=FILM_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, film_hidden), nn.ReLU(),
            nn.Linear(film_hidden, hidden_dim * 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, context):
        raw_gamma, beta = self.net(context).chunk(2, dim=-1)
        return 1.0 + raw_gamma, beta


class AblationDiscriminator(nn.Module):
    """conditioning='film' is architecturally identical to the shipped
    discriminator_model.Discriminator. conditioning='concat' feeds all 139
    dims through the first layer, matching the original design. Parameter
    counts differ slightly by construction -- reported, not hidden."""

    def __init__(self, conditioning="film"):
        super().__init__()
        assert conditioning in ("film", "concat")
        self.conditioning = conditioning
        self.input_dim = INPUT_DIM

        if conditioning == "film":
            self.linear1 = nn.Linear(CONTENT_DIM, HIDDEN_DIM)
            self.film1 = FiLMGenerator()
            self.film2 = FiLMGenerator()
        else:
            self.linear1 = nn.Linear(INPUT_DIM, HIDDEN_DIM)
        self.linear2 = nn.Linear(HIDDEN_DIM, HIDDEN_DIM)
        self.linear3 = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, x):
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"expected (batch, {self.input_dim}), got {tuple(x.shape)}")
        if self.conditioning == "concat":
            h = torch.relu(self.linear1(x))
            h = torch.relu(self.linear2(h))
        else:
            content, context = x[:, :CONTENT_DIM], x[:, CONTENT_DIM:]
            g1, b1 = self.film1(context)
            h = torch.relu(g1 * self.linear1(content) + b1)
            g2, b2 = self.film2(context)
            h = torch.relu(g2 * self.linear2(h) + b2)
        return self.linear3(h).squeeze(-1)

    def probability(self, x):
        return torch.sigmoid(self.forward(x))


def held_out_accuracy(model, held_feat, batch=512):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i in range(0, len(held_feat), batch):
            p = model.probability(torch.as_tensor(held_feat[i:i + batch], dtype=torch.float32))
            correct += (p > 0.5).sum().item(); total += len(p)
    model.train()
    return correct / total


def per_block_gradients(model, feats, n=2000):
    """Mean |d(logit)/d(input)| per feature block, per dimension. This is
    the dilution diagnostic -- if the context block sits far below the
    per-dim average, context is being ignored regardless of accuracy."""
    model.eval()
    x = torch.as_tensor(feats[:n], dtype=torch.float32).requires_grad_(True)
    g = torch.autograd.grad(model(x).sum(), x)[0].abs().mean(dim=0)
    model.train()
    out = {name: float(g[sl].mean()) for name, sl in BLOCK_SLICES.items()}
    out["_overall_per_dim_mean"] = float(g.mean())
    return out


def train_variant(name, conditioning, gamma_swap, eta, data, log_every=2000):
    torch.manual_seed(SEED); np.random.seed(SEED)
    train_feat, train_bins, held_feat, rand_c, mappo_c, swap_lw, swap_ll = data

    model = AblationDiscriminator(conditioning)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    expert_sampler = BalancedContextSampler(train_feat, train_bins, name="expert_train")
    target_props = sqrt_scaled_target(expert_sampler.cell_counts)
    trainer = DiscriminatorTrainer(
        model, opt, expert_sampler=expert_sampler, eta=eta, gamma_swap=gamma_swap,
        swap_lw_features=(swap_lw if gamma_swap > 0 else None),
        swap_ll_features=(swap_ll if gamma_swap > 0 else None))

    print(f"\n{'='*74}\n{name}: conditioning={conditioning}  gamma_swap={gamma_swap}  "
          f"eta={eta}  params={n_params:,}\n{'='*74}")

    rng = np.random.default_rng(SEED)
    t0 = time.time()
    history = []

    for phase, agent_cache, n_steps in [("A", rand_c, PHASE_A_STEPS),
                                         ("B", mappo_c, PHASE_B_STEPS)]:
        agent_sampler = BalancedContextSampler(agent_cache["features"], agent_cache["bins"],
                                                name=f"agent_phase{phase}")
        for step in range(1, n_steps + 1):
            (ef, eb), (af, ab) = balanced_batch(
                expert_sampler, agent_sampler, batch_size=BATCH_SIZE,
                target_props=target_props, rng=rng, on_empty="skip")
            res = trainer.step(ef, af, expert_cells=eb, agent_cells=ab)
            if step % log_every == 0:
                acc = held_out_accuracy(model, held_feat)
                history.append({"phase": phase, "step": step, "held_out_acc": acc,
                                "loss": res["loss"], "gp": res["gp_value"],
                                "swap_shift_lw": res.get("swap_shift_lw"),
                                "swap_shift_ll": res.get("swap_shift_ll")})
                print(f"  [{name} {phase}{step:>6}] held_out_acc={acc:.4f}  loss={res['loss']:.4f}  "
                      f"gp={res['gp_value']:.4f}  swap_ll={res.get('swap_shift_ll')}")

    final_acc = held_out_accuracy(model, held_feat)
    passed, gate = run_counterfactual_gate(model, held_feat)
    grads = per_block_gradients(model, held_feat)

    torch.save({"model_state_dict": model.state_dict(), "variant": name,
                "conditioning": conditioning, "gamma_swap": gamma_swap, "eta": eta,
                "final_held_out_acc": final_acc},
               f"disc_ablation_{name}.pt")

    return {
        "variant": name, "conditioning": conditioning, "gamma_swap": gamma_swap,
        "eta": eta, "n_params": n_params, "final_held_out_acc": final_acc,
        "gate_passed": bool(passed),
        "gate_lw_shift": gate["directions"][0]["mean_abs_shift"],
        "gate_lw_frac": gate["directions"][0]["frac_exceeding_0.1"],
        "gate_ll_shift": gate["directions"][1]["mean_abs_shift"],
        "gate_ll_frac": gate["directions"][1]["frac_exceeding_0.1"],
        "per_block_gradients": grads,
        "wall_sec": time.time() - t0,
        "history": history,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["all"])
    args = ap.parse_args()
    names = list(VARIANTS) if args.variants == ["all"] else args.variants

    cache = np.load(EXPERT_CACHE, allow_pickle=True)
    tr, ho = make_episode_split(cache["episode_outcomes"], held_out_frac=0.20, seed=0)
    (train_feat, train_bins), (held_feat, _) = split_features_by_episode(
        cache["features"], cache["bins"], cache["episode_ids"], tr, ho)
    m_lw = select_by_true_context(train_feat, t_norm_max=LATE_WINNING["t_norm_max"],
                                   delta_score_sign=LATE_WINNING["delta_score_sign"])
    m_ll = select_by_true_context(train_feat, t_norm_max=LATE_LOSING["t_norm_max"],
                                   delta_score_sign=LATE_LOSING["delta_score_sign"])
    data = (train_feat, train_bins, held_feat,
            np.load(RANDOM_CACHE), np.load(MAPPO_CACHE),
            train_feat[m_lw], train_feat[m_ll])

    print(f"expert train {train_feat.shape[0]:,} / held-out {held_feat.shape[0]:,} steps")
    print(f"swap populations: late-win {m_lw.sum():,}  late-loss {m_ll.sum():,}")

    results = []
    for n in names:
        results.append(train_variant(n, *VARIANTS[n], data=data))
        with open("discriminator_ablation_results.json", "w") as f:
            json.dump(results, f, indent=2)

    print("\n" + "=" * 88)
    print("DISCRIMINATOR ABLATION -- counterfactual gate (threshold 0.1, both directions)")
    print("=" * 88)
    print(f"{'variant':8s} {'cond':8s} {'g_swap':>7s} {'eta':>5s} {'acc':>7s} "
          f"{'LW shift':>9s} {'LL shift':>9s} {'ctx grad':>9s} {'PASS':>5s}")
    print("-" * 88)
    for r in results:
        cg = r["per_block_gradients"]["context"]
        print(f"{r['variant']:8s} {r['conditioning']:8s} {r['gamma_swap']:>7.1f} {r['eta']:>5.1f} "
              f"{r['final_held_out_acc']:>7.4f} {r['gate_lw_shift']:>9.4f} {r['gate_ll_shift']:>9.4f} "
              f"{cg:>9.5f} {'YES' if r['gate_passed'] else 'no':>5s}")

    print("\nADJACENT COMPARISONS (each isolates one factor):")
    print("  D1 -> D2 : FiLM conditioning")
    print("  D2 -> D3 : swap-consistency loss")
    print("  D3 -> D4 : relaxed R1 (eta 1.0 -> 0.3)")
    print("  D4 vs D5 : is FiLM still necessary once the swap loss is present?")
    print("\nwrote discriminator_ablation_results.json + disc_ablation_*.pt")


if __name__ == "__main__":
    main()