"""
calibrate_and_sweep_gamma_int.py

Redoes GATE 1 properly. Two things were wrong with the first attempt:

  1. 18-CELL STRATIFICATION WAS A DESIGN ERROR, not a bug.
     The DiD *is* a difference in conditional sprint marginals between
     expert and agent:
        oracle = [exp logit P(spr|LL) - exp logit P(spr|LW)]
               - [agt logit P(spr|LL) - agt logit P(spr|LW)]
               = 1.9926 - 0.1118 = 1.8809
     Matching (context x sprint) joint cells forces
     P(spr|ctx)_expert == P(spr|ctx)_agent inside every batch, so the
     batch-level DiD available to BCE is EXACTLY ZERO. Stratification
     removed the target statistic. Observed: all four 18-cell variants
     (D6 .165, D9 .162, D10 .093, D8 -.123) lost to the single 9-cell
     variant (D7 .258).
     ==> 9-CELL ONLY from here. The sprint leak is handled at the REWARD
         level by the D_marg / D_int split, not by resampling.

  2. gamma_int was never calibrated -- set to 6.0 by analogy to
     gamma_swap and never magnitude-checked, despite beta_aux having
     been wrong by 4x when it was checked.

Stage 1 measures the hinge's share of total loss at init for each
candidate gamma_int. Stage 2 trains the calibrated candidates, 9-cell,
and reports the gate.

Run:  python calibrate_and_sweep_gamma_int.py --stage calibrate
      python calibrate_and_sweep_gamma_int.py --stage sweep --gammas 2 6 20 60
"""

import argparse, json, time
import numpy as np
import torch

from discriminator_loss import DiscriminatorLoss
from discriminator_trainer import DiscriminatorTrainer
from context_balanced_sampler import BalancedContextSampler, balanced_batch, sqrt_scaled_target
from context_shift_scoring import select_by_true_context, LATE_WINNING, LATE_LOSING
from held_out_split import make_episode_split, split_features_by_episode

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from interaction_loss import (interaction_hinge, interaction_gate, GATE_THRESHOLD,
                              DEFAULT_INT_MARGIN)
from discriminator_ablation_v2 import (AblationDiscriminatorV2, held_out_accuracy,
                                       EXPERT_CACHE, RANDOM_CACHE, MAPPO_CACHE,
                                       PHASE_A_STEPS, PHASE_B_STEPS, BATCH_SIZE, SEED)


def load_data():
    cache = np.load(EXPERT_CACHE, allow_pickle=True)
    tr, ho = make_episode_split(cache["episode_outcomes"], held_out_frac=0.20, seed=0)
    (train_feat, train_bins), (held_feat, held_bins) = split_features_by_episode(
        cache["features"], cache["bins"], cache["episode_ids"], tr, ho)
    m_lw = select_by_true_context(train_feat, t_norm_max=LATE_WINNING["t_norm_max"],
                                  delta_score_sign=LATE_WINNING["delta_score_sign"])
    m_ll = select_by_true_context(train_feat, t_norm_max=LATE_LOSING["t_norm_max"],
                                  delta_score_sign=LATE_LOSING["delta_score_sign"])
    int_pool = train_feat[(train_bins == 6) | (train_bins == 7)]
    return (train_feat, train_bins, held_feat, held_bins,
            train_feat[m_lw], train_feat[m_ll], int_pool)


def calibrate(gammas):
    """Stage 1: at init, what share of total loss does gamma_int * hinge
    represent? Target ~20-30%: meaningful but not dominant, same standard
    gamma_swap was set by."""
    (train_feat, train_bins, held_feat, held_bins,
     swap_lw, swap_ll, int_pool) = load_data()
    torch.manual_seed(SEED); np.random.seed(SEED)

    model = AblationDiscriminatorV2("concat")
    loss_fn = DiscriminatorLoss(eta=0.3, gamma_swap=6.0,
                               swap_lw_features=swap_lw, swap_ll_features=swap_ll)
    sampler_e = BalancedContextSampler(train_feat, train_bins, name="expert")
    props = sqrt_scaled_target(sampler_e.cell_counts)
    mappo = np.load(MAPPO_CACHE)
    sampler_a = BalancedContextSampler(mappo["features"], mappo["bins"], name="agent")
    rng = np.random.default_rng(SEED)
    int_rng = np.random.default_rng(SEED + 1000)

    (ef, eb), (af, ab) = balanced_batch(sampler_e, sampler_a, BATCH_SIZE, props, rng, on_empty="skip")
    out = loss_fn(model, torch.as_tensor(ef, dtype=torch.float32),
                  torch.as_tensor(af, dtype=torch.float32))
    base_loss = out.total_loss.item()
    hinge, did = interaction_hinge(model, int_pool, int_rng, batch_size=64)
    h = hinge.item()

    print(f"\nGAMMA_INT CALIBRATION (at init, concat, 9-cell)")
    print(f"  base loss (BCE + R1 + swap) = {base_loss:.4f}")
    print(f"  raw interaction hinge       = {h:.4f}   (margin {DEFAULT_INT_MARGIN}, init DiD {did.item():+.4f})")
    print(f"\n  {'gamma_int':>10s} {'g*hinge':>9s} {'total':>9s} {'share':>8s}  verdict")
    rec = []
    for g in gammas:
        contrib = g * h
        total = base_loss + contrib
        share = contrib / total
        verdict = "TARGET" if 0.20 <= share <= 0.30 else ("too small" if share < 0.20 else "dominant")
        print(f"  {g:>10.1f} {contrib:>9.4f} {total:>9.4f} {share:>7.1%}  {verdict}")
        rec.append((g, share))
    good = [g for g, s in rec if 0.20 <= s <= 0.30]
    print(f"\n  recommended gamma_int: {good if good else 'none in target band -- widen the sweep'}")
    return rec


def train_one(gamma_int, data, log_every=4000):
    """Stage 2: 9-cell, concat, calibrated gamma_int."""
    (train_feat, train_bins, held_feat, held_bins,
     swap_lw, swap_ll, int_pool) = data
    torch.manual_seed(SEED); np.random.seed(SEED)

    model = AblationDiscriminatorV2("concat")
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sampler_e = BalancedContextSampler(train_feat, train_bins, name="expert")
    props = sqrt_scaled_target(sampler_e.cell_counts)
    trainer = DiscriminatorTrainer(model, opt, expert_sampler=sampler_e, eta=0.3,
                                   gamma_swap=6.0, swap_lw_features=swap_lw,
                                   swap_ll_features=swap_ll)
    rng = np.random.default_rng(SEED)
    int_rng = np.random.default_rng(SEED + 1000)
    t0 = time.time()
    hist = []

    print(f"\n{'='*70}\n9-cell, concat, gamma_int={gamma_int}\n{'='*70}")
    for phase, cache_path, n_steps in [("A", RANDOM_CACHE, PHASE_A_STEPS),
                                       ("B", MAPPO_CACHE, PHASE_B_STEPS)]:
        ac = np.load(cache_path)
        sampler_a = BalancedContextSampler(ac["features"], ac["bins"], name=f"agent{phase}")
        for step in range(1, n_steps + 1):
            (ef, eb), (af, ab) = balanced_batch(sampler_e, sampler_a, BATCH_SIZE,
                                                props, rng, on_empty="skip")
            res = trainer.step(ef, af, expert_cells=eb, agent_cells=ab)
            hinge, did = interaction_hinge(model, int_pool, int_rng, batch_size=64)
            if hinge is not None:
                opt.zero_grad(); (gamma_int * hinge).backward(); opt.step()
            if step % log_every == 0:
                acc = held_out_accuracy(model, held_feat)
                _, gate_did, _, _ = interaction_gate(model, held_feat, held_bins)
                hist.append({"phase": phase, "step": step, "acc": acc,
                             "train_did": did.item(), "gate_did": float(gate_did)})
                print(f"  [{phase}{step:>6}] acc={acc:.4f}  train_did={did.item():.3f}  "
                      f"GATE_did={gate_did:.4f}")

    acc = held_out_accuracy(model, held_feat)
    passed, gate_did, sizes, warn = interaction_gate(model, held_feat, held_bins)
    torch.save({"model_state_dict": model.state_dict(), "gamma_int": gamma_int},
               f"disc_9cell_gint{gamma_int:g}.pt")
    return {"gamma_int": gamma_int, "held_out_acc": acc, "gate_did": float(gate_did),
            "gate_passed": bool(passed), "sizes": sizes, "warning": warn,
            "wall_sec": time.time() - t0, "history": hist}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["calibrate", "sweep"], default="calibrate")
    ap.add_argument("--gammas", type=float, nargs="+", default=[1, 2, 6, 20, 60, 200])
    args = ap.parse_args()

    if args.stage == "calibrate":
        calibrate(args.gammas)
        print("\nNext: python calibrate_and_sweep_gamma_int.py --stage sweep --gammas <recommended>")
        return

    data = load_data()
    print(f"GATE THRESHOLD = {GATE_THRESHOLD:.4f}   (30% of oracle 1.8809)")
    print(f"reference: D7 (9-cell, gamma_int=6 uncalibrated) reached 0.2577")
    results = [train_one(g, data) for g in args.gammas]
    with open("gamma_int_sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print(f"GATE 1 (redone, 9-cell only) -- threshold {GATE_THRESHOLD:.4f}")
    print("=" * 70)
    print(f"{'gamma_int':>10s} {'acc':>8s} {'GATE DiD':>10s} {'% ceiling':>10s} {'PASS':>6s}")
    for r in sorted(results, key=lambda r: -r["gate_did"]):
        print(f"{r['gamma_int']:>10.1f} {r['held_out_acc']:>8.4f} {r['gate_did']:>10.4f} "
              f"{100*r['gate_did']/1.8809:>9.1f}% {'YES' if r['gate_passed'] else 'no':>6s}")
    best = max(results, key=lambda r: r["gate_did"])
    print(f"\nbest: gamma_int={best['gamma_int']:g} at DiD={best['gate_did']:.4f}")
    if best["gate_passed"]:
        print("-> GATE 1 PASSED. Ship it, proceed to Phase 2.")
    else:
        print("-> still short of 0.5643. NOW the negative-result path is")
        print("   properly evidenced: correct sampling, calibrated hinge weight,")
        print("   swept across an order of magnitude.")


if __name__ == "__main__":
    main()