"""
run_D10.py

GATE 1 follow-up. D6-D9 all satisfied the SYNTHETIC training hinge
(DiD 0.85-0.94) but collapsed on the REAL held-out gate (DiD 0.16-0.26,
D8 sign-flipped to -0.12). Diagnosis: shortcut learning on the literal
(dim135, context) values via reused-base-row synthesis (see
interaction_loss.py's real_interaction_hinge docstring).

D10 trains directly on real-group mean differences (no synthesis at
all), removing the shortcut route by construction. Concat architecture
(D5/D7 pattern -- FiLM (D8) had the WORST synthetic/real gap, so it's
not the leading candidate here).

FIRM DECISION RULE, no further iteration after this:
  D10 clears 0.5643 on real held-out  -> ship it, proceed to Phase 2
  D10 does not clear it               -> commit to the negative-result
                                          path (remediation plan Part 5).
                                          The achievable-ceiling gap
                                          (best real DiD vs oracle 1.8809)
                                          IS the finding at that point --
                                          three independent measurements
                                          (raw 40pp spread, oracle DiD,
                                          Q3 non-redundancy) confirm the
                                          signal exists in the corpus;
                                          this would be evidence it is
                                          not learnable by this
                                          discriminator class from 52
                                          episodes, which is itself a
                                          real, reportable methodological
                                          finding.
"""

import json
import numpy as np
import torch

from context_shift_scoring import select_by_true_context, LATE_WINNING, LATE_LOSING
from held_out_split import make_episode_split, split_features_by_episode
from discriminator_loss import DiscriminatorLoss

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from interaction_loss import build_real_groups, real_interaction_hinge, interaction_gate, GATE_THRESHOLD
from discriminator_ablation_v2 import (AblationDiscriminatorV2, SimpleJointSampler,
                                       balanced_batch_18, held_out_accuracy,
                                       build_joint_bins, SPRINT_DIM,
                                       EXPERT_CACHE, RANDOM_CACHE, MAPPO_CACHE,
                                       PHASE_A_STEPS, PHASE_B_STEPS, BATCH_SIZE, SEED)

GAMMA_SWAP = 6.0
GAMMA_INT = 6.0


def main():
    torch.manual_seed(SEED); np.random.seed(SEED)

    cache = np.load(EXPERT_CACHE, allow_pickle=True)
    tr, ho = make_episode_split(cache["episode_outcomes"], held_out_frac=0.20, seed=0)
    (train_feat, train_bins), (held_feat, held_bins) = split_features_by_episode(
        cache["features"], cache["bins"], cache["episode_ids"], tr, ho)

    m_lw = select_by_true_context(train_feat, t_norm_max=LATE_WINNING["t_norm_max"],
                                  delta_score_sign=LATE_WINNING["delta_score_sign"])
    m_ll = select_by_true_context(train_feat, t_norm_max=LATE_LOSING["t_norm_max"],
                                  delta_score_sign=LATE_LOSING["delta_score_sign"])
    swap_lw, swap_ll = train_feat[m_lw], train_feat[m_ll]

    real_groups = build_real_groups(train_feat, train_bins)
    print("D10 real interaction groups (train split):")
    for k, v in real_groups.items():
        print(f"  {k}: {len(v)} steps")

    model = AblationDiscriminatorV2("concat")
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    train_sprint = train_feat[:, SPRINT_DIM]
    joint_bins, merge_report = build_joint_bins(train_bins, train_sprint)
    if merge_report:
        print(f"merged thin cells: {merge_report}")
    expert_sampler = SimpleJointSampler(train_feat, joint_bins, name="expert_train_18cell")

    loss_fn = DiscriminatorLoss(eta=0.3, gamma_swap=GAMMA_SWAP,
                               swap_lw_features=swap_lw, swap_ll_features=swap_ll)

    rng = np.random.default_rng(SEED)
    int_rng = np.random.default_rng(SEED + 1000)
    history = []

    for phase, agent_cache, n_steps in [("A", np.load(RANDOM_CACHE), PHASE_A_STEPS),
                                        ("B", np.load(MAPPO_CACHE), PHASE_B_STEPS)]:
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

            hinge, did = real_interaction_hinge(model, real_groups, int_rng, batch_size=64)
            opt.zero_grad()
            (GAMMA_INT * hinge).backward()
            opt.step()

            if step % 2000 == 0:
                acc = held_out_accuracy(model, held_feat)
                history.append({"phase": phase, "step": step, "acc": acc,
                                "loss": out.total_loss.item(), "train_did": did.item()})
                print(f"  [D10 {phase}{step:>6}] held_out_acc={acc:.4f}  "
                      f"loss={out.total_loss.item():.4f}  train_real_did={did.item():.4f}")

    final_acc = held_out_accuracy(model, held_feat)
    passed, gate_did, sizes, warning = interaction_gate(model, held_feat, held_bins)
    if warning:
        print(f"WARNING: {warning}")

    print("\n" + "=" * 70)
    print(f"D10 RESULT   held-out acc={final_acc:.4f}   "
          f"GATE DiD={gate_did:.4f}   threshold={GATE_THRESHOLD:.4f}   "
          f"PASS={'YES' if passed else 'no'}")
    print("=" * 70)

    torch.save({"model_state_dict": model.state_dict(), "variant": "D10",
               "final_held_out_acc": final_acc}, "disc_ablation_D10.pt")
    with open("D10_result.json", "w") as f:
        json.dump({"held_out_acc": final_acc, "gate_did": float(gate_did),
                   "gate_threshold": GATE_THRESHOLD, "gate_passed": bool(passed),
                   "group_sizes": sizes, "history": history}, f, indent=2)

    if passed:
        print("\n-> SHIP D10. Proceed to Phase 2 (policy-side fixes).")
    else:
        print("\n-> D10 did not clear the gate either. Per the firm decision rule,")
        print("   commit to the negative-result path now (remediation plan Part 5).")
        print(f"   Report: best real DiD achieved = {gate_did:.4f} vs oracle ceiling")
        print(f"   1.8809 -- {100*gate_did/1.8809:.1f}% of the achievable ceiling reached.")


if __name__ == "__main__":
    main()