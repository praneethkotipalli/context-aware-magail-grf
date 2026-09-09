"""
discriminator_ablation_v2.py -- v2, fixed.

BUG in v1: DiscriminatorTrainer.__init__ independently recomputes
sqrt_scaled_target(expert_sampler.cell_counts), and BalancedContextSampler
hardcodes N_CELLS=9 internally (cell_indices = [... for c in range(9)]).
Passing it 18-valued bins meant it silently ignored cells 9-17, and the
thin-cell merge (moving cell 1's examples into cell 0) left index 1
genuinely empty, which sqrt_scaled_target correctly rejects.

FIX: D7 (9-cell, unstratified) keeps using the existing, proven
BalancedContextSampler/DiscriminatorTrainer path unchanged -- that
machinery is validated by D1-D5. D6/D8/D9 (18-cell, stratified) use a
SELF-CONTAINED loop: DiscriminatorLoss directly (no hidden cell-count
assumptions) + a plain numpy stratified sampler that only ever looks at
cells that actually exist. This is appropriate for a diagnostic ablation
script -- Phase 2 can properly extend BalancedContextSampler for
production use once GATE 1 tells us which variant to ship.
"""

import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn

from discriminator_loss import DiscriminatorLoss
from discriminator_trainer import DiscriminatorTrainer
from context_balanced_sampler import BalancedContextSampler, balanced_batch, sqrt_scaled_target
from context_shift_scoring import select_by_true_context, LATE_WINNING, LATE_LOSING
from held_out_split import make_episode_split, split_features_by_episode
from build_expert_dataset import classify_bin

import sys
sys.path.insert(0, os.path.dirname(__file__))
from stratified_sampler18 import build_joint_bins, print_joint_report, SPRINT_DIM, N_JOINT_CELLS
from interaction_loss import interaction_hinge, interaction_gate, GATE_THRESHOLD, DEFAULT_GAMMA_INT

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
    # name: (conditioning, gamma_swap, gamma_int, stratified)
    "D6": ("concat", 6.0, DEFAULT_GAMMA_INT, True),
    "D7": ("concat", 6.0, DEFAULT_GAMMA_INT, False),   # unchanged 9-cell path
    "D8": ("film",   6.0, DEFAULT_GAMMA_INT, True),
    "D9": ("concat", 0.0, DEFAULT_GAMMA_INT, True),
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


class AblationDiscriminatorV2(nn.Module):
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
            p = model.probability(torch.as_tensor(held_feat[i:i+batch], dtype=torch.float32))
            correct += (p > 0.5).sum().item(); total += len(p)
    model.train()
    return correct / total


# ---------------------------------------------------------------------------
# Self-contained 18-cell sampling, no BalancedContextSampler dependency.
# Only ever iterates cells that actually exist in the passed bins array --
# no hidden N_CELLS assumption.
# ---------------------------------------------------------------------------

class SimpleJointSampler:
    def __init__(self, features, joint_bins, name="unnamed"):
        self.features = features
        self.bins = joint_bins
        self.present_cells = sorted(set(joint_bins.tolist()))
        self.cell_indices = {c: np.where(joint_bins == c)[0] for c in self.present_cells}
        self.cell_counts = {c: len(idx) for c, idx in self.cell_indices.items()}
        self.name = name

    def sample(self, per_cell_counts, rng):
        """per_cell_counts: dict {cell: n}. Cells absent from this sampler
        are silently skipped (on_empty='skip' philosophy)."""
        idx_parts = []
        for c, n_needed in per_cell_counts.items():
            if n_needed <= 0 or c not in self.cell_indices:
                continue
            available = self.cell_indices[c]
            if len(available) == 0:
                continue
            replace = n_needed > len(available)
            idx_parts.append(rng.choice(available, size=n_needed, replace=replace))
        if not idx_parts:
            return self.features[:0]
        idx = np.concatenate(idx_parts)
        return self.features[idx]


def joint_target_counts(expert_sampler, batch_size):
    """sqrt-scaled target over whatever cells the expert sampler actually
    has (post-merge), largest-remainder rounding to sum exactly to batch_size."""
    cells = expert_sampler.present_cells
    counts = np.array([expert_sampler.cell_counts[c] for c in cells], dtype=float)
    w = np.sqrt(counts)
    props = w / w.sum()
    exact = props * batch_size
    base = np.floor(exact).astype(int)
    remainder = batch_size - base.sum()
    if remainder > 0:
        order = np.argsort(-(exact - base))
        base[order[:remainder]] += 1
    return {c: int(n) for c, n in zip(cells, base)}


def balanced_batch_18(expert_sampler, agent_sampler, batch_size, rng):
    target = joint_target_counts(expert_sampler, batch_size)
    # only draw agent-side counts for cells BOTH sides actually have --
    # keeps marginals matched on whatever intersection exists this batch
    common = {c: n for c, n in target.items() if c in agent_sampler.cell_indices}
    ef = expert_sampler.sample(common, rng)
    af = agent_sampler.sample(common, rng)
    return ef, af


def train_variant_stratified(name, conditioning, gamma_swap, gamma_int, data, log_every=2000):
    """D6, D8, D9 -- self-contained loop, DiscriminatorLoss directly."""
    torch.manual_seed(SEED); np.random.seed(SEED)
    (train_feat, train_bins_9, held_feat, held_bins_9,
     rand_c, mappo_c, swap_lw, swap_ll, int_pool) = data

    model = AblationDiscriminatorV2(conditioning)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    train_sprint = train_feat[:, SPRINT_DIM]
    joint_bins, merge_report = build_joint_bins(train_bins_9, train_sprint)
    if merge_report:
        print(f"  [{name}] merged thin cells: {merge_report}")
    expert_sampler = SimpleJointSampler(train_feat, joint_bins, name="expert_train_18cell")
    print_joint_report(joint_bins, name=f"{name} expert")

    loss_fn = DiscriminatorLoss(
        eta=0.3, gamma_swap=gamma_swap,
        swap_lw_features=(swap_lw if gamma_swap > 0 else None),
        swap_ll_features=(swap_ll if gamma_swap > 0 else None))

    print(f"\n{'='*74}\n{name}: conditioning={conditioning}  gamma_swap={gamma_swap}  "
          f"gamma_int={gamma_int}  stratified=True (self-contained)  params={n_params:,}\n{'='*74}")

    rng = np.random.default_rng(SEED)
    int_rng = np.random.default_rng(SEED + 1000)
    t0 = time.time()
    history = []

    for phase, agent_cache, n_steps in [("A", rand_c, PHASE_A_STEPS), ("B", mappo_c, PHASE_B_STEPS)]:
        a_sprint = agent_cache["features"][:, SPRINT_DIM]
        a_joint, _ = build_joint_bins(agent_cache["bins"], a_sprint)
        agent_sampler = SimpleJointSampler(agent_cache["features"], a_joint, name=f"agent_{phase}")

        for step in range(1, n_steps + 1):
            ef, af = balanced_batch_18(expert_sampler, agent_sampler, BATCH_SIZE, rng)
            if len(ef) == 0 or len(af) == 0:
                continue
            ef_t = torch.as_tensor(ef, dtype=torch.float32)
            af_t = torch.as_tensor(af, dtype=torch.float32)

            out = loss_fn(model, ef_t, af_t)
            opt.zero_grad()
            out.total_loss.backward()
            opt.step()

            did_val = None
            if gamma_int > 0:
                hinge, did = interaction_hinge(model, int_pool, int_rng, batch_size=64)
                if hinge is not None:
                    opt.zero_grad()
                    (gamma_int * hinge).backward()
                    opt.step()
                    did_val = did.item()

            if step % log_every == 0:
                acc = held_out_accuracy(model, held_feat)
                history.append({"phase": phase, "step": step, "held_out_acc": acc,
                                "loss": out.total_loss.item(), "did": did_val})
                print(f"  [{name} {phase}{step:>6}] held_out_acc={acc:.4f}  "
                      f"loss={out.total_loss.item():.4f}  did={did_val}")

    final_acc = held_out_accuracy(model, held_feat)
    passed, did, sizes, warning = interaction_gate(model, held_feat, held_bins_9)
    if warning:
        print(f"  [{name}] WARNING: {warning}")

    torch.save({"model_state_dict": model.state_dict(), "variant": name,
               "conditioning": conditioning, "gamma_swap": gamma_swap,
               "gamma_int": gamma_int, "stratified": True,
               "final_held_out_acc": final_acc}, f"disc_ablation_{name}.pt")

    return {
        "variant": name, "conditioning": conditioning, "gamma_swap": gamma_swap,
        "gamma_int": gamma_int, "stratified": True, "n_params": n_params,
        "final_held_out_acc": final_acc, "gate_passed": bool(passed),
        "did": float(did), "gate_threshold": GATE_THRESHOLD,
        "group_sizes": sizes, "wall_sec": time.time() - t0, "history": history,
    }


def train_variant_unstratified(name, conditioning, gamma_swap, gamma_int, data, log_every=2000):
    """D7 -- unchanged 9-cell path via BalancedContextSampler/DiscriminatorTrainer,
    exactly as D1-D5 used it. Adds only the interaction hinge on top."""
    torch.manual_seed(SEED); np.random.seed(SEED)
    (train_feat, train_bins_9, held_feat, held_bins_9,
     rand_c, mappo_c, swap_lw, swap_ll, int_pool) = data

    model = AblationDiscriminatorV2(conditioning)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    expert_sampler = BalancedContextSampler(train_feat, train_bins_9, name="expert_train_9cell")
    target_props = sqrt_scaled_target(expert_sampler.cell_counts)
    trainer = DiscriminatorTrainer(
        model, opt, expert_sampler=expert_sampler, eta=0.3, gamma_swap=gamma_swap,
        swap_lw_features=(swap_lw if gamma_swap > 0 else None),
        swap_ll_features=(swap_ll if gamma_swap > 0 else None))

    print(f"\n{'='*74}\n{name}: conditioning={conditioning}  gamma_swap={gamma_swap}  "
          f"gamma_int={gamma_int}  stratified=False (9-cell, unchanged path)  "
          f"params={n_params:,}\n{'='*74}")

    rng = np.random.default_rng(SEED)
    int_rng = np.random.default_rng(SEED + 1000)
    t0 = time.time()
    history = []

    for phase, agent_cache, n_steps in [("A", rand_c, PHASE_A_STEPS), ("B", mappo_c, PHASE_B_STEPS)]:
        agent_sampler = BalancedContextSampler(agent_cache["features"], agent_cache["bins"],
                                               name=f"agent_phase{phase}_9cell")
        for step in range(1, n_steps + 1):
            (ef, eb), (af, ab) = balanced_batch(
                expert_sampler, agent_sampler, batch_size=BATCH_SIZE,
                target_props=target_props, rng=rng, on_empty="skip")
            res = trainer.step(ef, af, expert_cells=eb, agent_cells=ab)

            did_val = None
            if gamma_int > 0:
                hinge, did = interaction_hinge(model, int_pool, int_rng, batch_size=64)
                if hinge is not None:
                    opt.zero_grad()
                    (gamma_int * hinge).backward()
                    opt.step()
                    did_val = did.item()

            if step % log_every == 0:
                acc = held_out_accuracy(model, held_feat)
                history.append({"phase": phase, "step": step, "held_out_acc": acc,
                                "loss": res["loss"], "did": did_val})
                print(f"  [{name} {phase}{step:>6}] held_out_acc={acc:.4f}  "
                      f"loss={res['loss']:.4f}  did={did_val}")

    final_acc = held_out_accuracy(model, held_feat)
    passed, did, sizes, warning = interaction_gate(model, held_feat, held_bins_9)
    if warning:
        print(f"  [{name}] WARNING: {warning}")

    torch.save({"model_state_dict": model.state_dict(), "variant": name,
               "conditioning": conditioning, "gamma_swap": gamma_swap,
               "gamma_int": gamma_int, "stratified": False,
               "final_held_out_acc": final_acc}, f"disc_ablation_{name}.pt")

    return {
        "variant": name, "conditioning": conditioning, "gamma_swap": gamma_swap,
        "gamma_int": gamma_int, "stratified": False, "n_params": n_params,
        "final_held_out_acc": final_acc, "gate_passed": bool(passed),
        "did": float(did), "gate_threshold": GATE_THRESHOLD,
        "group_sizes": sizes, "wall_sec": time.time() - t0, "history": history,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["all"])
    args = ap.parse_args()
    names = list(VARIANTS) if args.variants == ["all"] else args.variants

    cache = np.load(EXPERT_CACHE, allow_pickle=True)
    tr, ho = make_episode_split(cache["episode_outcomes"], held_out_frac=0.20, seed=0)
    (train_feat, train_bins), (held_feat, held_bins) = split_features_by_episode(
        cache["features"], cache["bins"], cache["episode_ids"], tr, ho)

    m_lw = select_by_true_context(train_feat, t_norm_max=LATE_WINNING["t_norm_max"],
                                  delta_score_sign=LATE_WINNING["delta_score_sign"])
    m_ll = select_by_true_context(train_feat, t_norm_max=LATE_LOSING["t_norm_max"],
                                  delta_score_sign=LATE_LOSING["delta_score_sign"])

    int_pool_mask = (train_bins == 6) | (train_bins == 7)
    int_pool = train_feat[int_pool_mask]

    data = (train_feat, train_bins, held_feat, held_bins,
           np.load(RANDOM_CACHE), np.load(MAPPO_CACHE),
           train_feat[m_lw], train_feat[m_ll], int_pool)

    print(f"expert train {train_feat.shape[0]:,} / held-out {held_feat.shape[0]:,} steps")
    print(f"interaction pool (late-win + late-loss, train split): {len(int_pool):,} steps")
    print(f"GATE THRESHOLD (pre-registered) = {GATE_THRESHOLD:.4f}")

    results = []
    for n in names:
        cond, g_swap, g_int, stratified = VARIANTS[n]
        if stratified:
            r = train_variant_stratified(n, cond, g_swap, g_int, data)
        else:
            r = train_variant_unstratified(n, cond, g_swap, g_int, data)
        results.append(r)
        with open("discriminator_ablation_v2_results.json", "w") as f:
            json.dump(results, f, indent=2)

    print("\n" + "=" * 100)
    print(f"GATE 1 -- INTERACTION GATE (threshold {GATE_THRESHOLD:.4f})")
    print("=" * 100)
    print(f"{'variant':8s} {'cond':8s} {'g_swap':>7s} {'g_int':>6s} {'strat':>6s} "
          f"{'acc':>7s} {'DiD':>8s} {'PASS':>5s}")
    print("-" * 100)
    any_pass = False
    for r in results:
        p = "YES" if r["gate_passed"] else "no"
        if r["gate_passed"]:
            any_pass = True
        print(f"{r['variant']:8s} {r['conditioning']:8s} {r['gamma_swap']:>7.1f} "
              f"{r['gamma_int']:>6.1f} {str(r['stratified']):>6s} "
              f"{r['final_held_out_acc']:>7.4f} {r['did']:>8.4f} {p:>5s}")

    print()
    if any_pass:
        best = max((r for r in results if r["gate_passed"]), key=lambda r: r["did"])
        print(f"DECISION: ship {best['variant']} (highest DiD among passing variants). Proceed to Phase 2.")
    else:
        print("DECISION: NONE of D6-D9 cleared the gate. STOP.")
        print("  -> negative-result path (remediation plan Part 5): the achievable-")
        print("     ceiling gap (best learned DiD vs oracle 1.8809) is the finding.")

    print("\nwrote discriminator_ablation_v2_results.json + disc_ablation_D[6-9].pt")


if __name__ == "__main__":
    main()