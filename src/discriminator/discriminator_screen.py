"""
run_stage0_v2_critical.py

Stage 0 v2, corrected recipe, run on the three highest-value candidates
ONLY (not all nine) before committing to a full re-screen:

  logreg5          -- already recovered partially (0.150); test if the fix closes it
  tiny-interact     -- already the best performer (0.232, still rising); most likely to pass
  additive-split    -- the untested combination doc 30 flags: low-capacity-ish
                        interaction path + the real-group DiD hinge, which the
                        original screen deliberately withheld ("only retry if
                        a candidate fails" -- everything failed, so this
                        combination has never actually been run)

CHANGES from run_stage0_screen.py, addressing C1-C4:
  C1  eta=0.0            -- no R1 penalty (was 0.3, directly shrinks w4/w5 in LogReg5)
  C2  expert_label=1.0,
      agent_label=0.0    -- hard labels, no smoothing (was 0.9/0.1)
  C3  late-cell upweight -- target_props forces >=50% of every batch mass onto
                             cells 6+7 (late/win, late/loss), so the single global
                             linear interaction coefficient isn't fit mostly against
                             near-zero-interaction early/mid data
  C4  Phase A DROPPED    -- train expert-vs-MAPPO for the full step budget;
                             never expert-vs-random for an interaction probe
  NEW real-group DiD hinge, gamma_int swept over {6, 15, 40} -- this exact
      combination (low-capacity model + hinge + hard labels) was never run
      in the original screen or in D6-D10 (which used only content models).

Run:  python run_stage0_v2_critical.py
"""
import argparse, datetime, json, os, time, uuid
import numpy as np
import torch

import sys
sys.path.insert(0, os.path.dirname(__file__))
from discriminator_candidates import LogReg5, TinyInteract, AdditiveSplit
from discriminator_loss import DiscriminatorLoss
from context_balanced_sampler import BalancedContextSampler, balanced_batch

SPRINT_DIM = 135
LATE_WIN, LATE_LOSS = 6, 7
ORACLE_DID = 1.8809
GATE_THRESHOLD = 0.30 * ORACLE_DID
CI_LOWER_BAR = 0.40
BALANCED_ACC_BAR = 0.80
DECAY_BAR = 0.80
LATE_CELL_MIN_SHARE = 0.50   # C3 fix: force >=50% of batch mass onto cells 6+7

EXPERT_CACHE = "expert_features_cache.npz"
MAPPO_CACHE = "mappo_features_cache.npz"

BATCH_SIZE = 128
TOTAL_STEPS = 16000   # C4 fix: no Phase A, full budget on expert-vs-MAPPO
LOG_EVERY = 500
SEEDS = [0, 1, 2]
# gamma_int=0.0 FIRST -- the BCE-only baseline is now the primary lever
# (verify_logreg_claim / verify_logreg_lateonly showed 0.844 / 1.030 held-out
# DiD from hard-label BCE alone, zero hinge, once C1/C2/C4 are removed). The
# hinge is optional insurance, tested here to see whether it helps, hurts,
# or is redundant for each candidate -- NOT assumed to be the primary driver.
GAMMA_INTS = [0.0, 6.0, 15.0, 40.0]
# margin raised from 0.8 -> 1.3: at margin=0.8 the hinge gradient is exactly
# zero once did > 0.8 (relu(margin-did)=0 for did>margin), which would have
# capped any candidate below the 0.844-1.030 BCE-alone ceiling just measured.
# 1.3 sits above the late-cell-fit result (1.030) so the hinge, when active,
# cannot suppress a candidate that would otherwise reach BCE's own ceiling.
HINGE_MARGIN = 1.3

LAUNCH_ID = f"{datetime.datetime.now():%m%d}{uuid.uuid4().hex[:3]}_v2crit"
RESULT_DIR = os.path.join("results_v2", "stage0_discriminator", LAUNCH_ID)


def make_episode_split(episode_outcomes, held_out_frac=0.20, seed=0):
    rng = np.random.default_rng(seed)
    idx_by_outcome = {}
    for i, o in enumerate(episode_outcomes):
        idx_by_outcome.setdefault(str(o), []).append(i)
    train_ep, held_ep = [], []
    for o, idxs in idx_by_outcome.items():
        idxs = np.array(idxs); rng.shuffle(idxs)
        n_held = max(1, round(len(idxs) * held_out_frac))
        held_ep.extend(idxs[:n_held].tolist())
        train_ep.extend(idxs[n_held:].tolist())
    return np.array(sorted(train_ep)), np.array(sorted(held_ep))


def late_upweighted_target(cell_counts, min_late_share=LATE_CELL_MIN_SHARE):
    """sqrt-scaled baseline, then renormalise so cells 6+7 together get at
    least min_late_share of every batch -- the direct fix for C3."""
    counts = np.asarray(cell_counts, dtype=float)
    w = np.sqrt(np.maximum(counts, 1))
    base = w / w.sum()
    late_mass = base[LATE_WIN] + base[LATE_LOSS]
    if late_mass >= min_late_share:
        return base
    scale_other = (1 - min_late_share) / max(base.sum() - late_mass, 1e-9)
    out = base.copy()
    for c in range(len(out)):
        if c not in (LATE_WIN, LATE_LOSS):
            out[c] *= scale_other
    remaining = min_late_share
    out[LATE_WIN] = remaining * (base[LATE_WIN] / max(late_mass, 1e-9))
    out[LATE_LOSS] = remaining * (base[LATE_LOSS] / max(late_mass, 1e-9))
    return out / out.sum()


def target_counts_for_batch(target_props, batch_size):
    exact = np.asarray(target_props) * batch_size
    base = np.floor(exact).astype(int)
    rem = batch_size - base.sum()
    if rem > 0:
        order = np.argsort(-(exact - base))
        base[order[:rem]] += 1
    return base


def real_interaction_hinge(discriminator, groups, rng, batch_size=64, margin=HINGE_MARGIN):
    def draw(key):
        pool = groups[key]
        n = min(batch_size, len(pool))
        idx = rng.choice(len(pool), size=n, replace=(n < batch_size))
        return torch.as_tensor(pool[idx], dtype=torch.float32)
    d_l1 = discriminator(draw('ll1')).mean(); d_l0 = discriminator(draw('ll0')).mean()
    d_w1 = discriminator(draw('lw1')).mean(); d_w0 = discriminator(draw('lw0')).mean()
    did = (d_l1 - d_l0) - (d_w1 - d_w0)
    return torch.relu(margin - did), did.detach()


def build_real_groups(feat, bins):
    def g(cell, spr): return feat[(bins == cell) & ((feat[:, SPRINT_DIM] > 0.5) == spr)]
    return {"ll1": g(LATE_LOSS, True), "ll0": g(LATE_LOSS, False),
            "lw1": g(LATE_WIN, True), "lw0": g(LATE_WIN, False)}


def gate_did_point(model, feat, bins):
    def grp(cell, on): return feat[(bins == cell) & ((feat[:, SPRINT_DIM] > 0.5) == on)]
    groups = {"ll1": grp(LATE_LOSS, True), "ll0": grp(LATE_LOSS, False),
              "lw1": grp(LATE_WIN, True), "lw0": grp(LATE_WIN, False)}
    if min(len(v) for v in groups.values()) == 0:
        return np.nan
    model.eval()
    with torch.no_grad():
        m = lambda a: model(torch.as_tensor(a, dtype=torch.float32)).mean().item()
        did = (m(groups["ll1"]) - m(groups["ll0"])) - (m(groups["lw1"]) - m(groups["lw0"]))
    model.train()
    return did


def episode_bootstrap_ci(model, held_feat, held_bins, held_eids, ho_ep, n_boot=200, seed=1):
    rng = np.random.default_rng(seed)
    vals, attempts = [], 0
    while len(vals) < n_boot and attempts < n_boot * 20:
        attempts += 1
        sample_eps = rng.choice(ho_ep, size=len(ho_ep), replace=True)
        mask = np.isin(held_eids, sample_eps)
        if mask.sum() < 30:
            continue
        d = gate_did_point(model, held_feat[mask], held_bins[mask])
        if not np.isnan(d):
            vals.append(d)
    vals = np.array(vals)
    if len(vals) == 0:
        return np.nan, np.nan
    return float(np.percentile(vals, 5)), float(np.percentile(vals, 95))


def balanced_acc(model, held_expert, held_agent_sample, batch=512):
    model.eval()
    def acc(feats, want_high):
        c = t = 0
        with torch.no_grad():
            for i in range(0, len(feats), batch):
                p = model.probability(torch.as_tensor(feats[i:i+batch], dtype=torch.float32))
                c += ((p > 0.5) == want_high).sum().item(); t += len(p)
        return c / max(t, 1)
    r = 0.5 * (acc(held_expert, True) + acc(held_agent_sample, False))
    model.train()
    return r


def build_model(name):
    return {"logreg5": LogReg5, "tiny-interact": TinyInteract,
           "additive-split": AdditiveSplit}[name]()


def run(name, gamma_int, seed, data):
    (train_feat, train_bins, held_feat, held_bins, held_eids, ho_ep,
     mappo_train, mappo_held) = data
    torch.manual_seed(seed); np.random.seed(seed)

    model = build_model(name)
    n_params = sum(p.numel() for p in model.parameters())
    wd = 0.0 if n_params < 1000 else 1e-4
    opt = torch.optim.Adam(model.parameters(), lr=1e-3 if n_params < 1000 else 1e-4, weight_decay=wd)

    loss_fn = DiscriminatorLoss(eta=0.0, expert_label=0.999, agent_label=0.001,
                               gamma_swap=0.0, swap_lw_features=None, swap_ll_features=None)

    sampler_e = BalancedContextSampler(train_feat, train_bins, name="expert")
    mappo_all = np.load(MAPPO_CACHE, allow_pickle=True)
    sampler_a = BalancedContextSampler(mappo_all["features"], mappo_all["bins"], name="agent")
    props = late_upweighted_target(sampler_e.cell_counts)

    real_groups = build_real_groups(train_feat, train_bins)
    rng = np.random.default_rng(seed)
    int_rng = np.random.default_rng(seed + 1000)

    print(f"\n{'='*70}\n{name}  gamma_int={gamma_int}  seed={seed}  params={n_params}  "
          f"(no Phase A, {TOTAL_STEPS} steps, late-cell share forced >={LATE_CELL_MIN_SHARE})\n{'='*70}")

    history, peak = [], -np.inf
    t0 = time.time()
    for step in range(1, TOTAL_STEPS + 1):
        (ef, eb), (af, ab) = balanced_batch(sampler_e, sampler_a, BATCH_SIZE, props, rng, on_empty="skip")
        if len(ef) == 0 or len(af) == 0:
            continue
        out = loss_fn(model, torch.as_tensor(ef, dtype=torch.float32),
                     torch.as_tensor(af, dtype=torch.float32))
        opt.zero_grad(); out.total_loss.backward(); opt.step()

        if gamma_int > 0:
            hinge, did = real_interaction_hinge(model, real_groups, int_rng, margin=HINGE_MARGIN)
            opt.zero_grad(); (gamma_int * hinge).backward(); opt.step()

        if step % LOG_EVERY == 0:
            d = gate_did_point(model, held_feat, held_bins)
            peak = max(peak, d) if not np.isnan(d) else peak
            history.append({"step": step, "held_did": float(d) if not np.isnan(d) else None})
            print(f"  [{step:>6}] loss={out.total_loss.item():.4f}  held_did={d:+.4f}")

    final = gate_did_point(model, held_feat, held_bins)
    ci_lo, ci_hi = episode_bootstrap_ci(model, held_feat, held_bins, held_eids, ho_ep)
    bacc = balanced_acc(model, held_feat, mappo_held)
    decay_ok = (not np.isnan(peak) and peak > 0 and final >= DECAY_BAR * peak)

    return {"candidate": name, "gamma_int": gamma_int, "seed": seed, "n_params": n_params,
            "final_did": float(final), "peak_did": float(peak), "decay_ok": bool(decay_ok),
            "ci_lo": ci_lo, "ci_hi": ci_hi, "balanced_acc": bacc,
            "wall_sec": time.time() - t0, "history": history}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+", default=["logreg5", "tiny-interact", "additive-split"])
    args = ap.parse_args()

    os.makedirs(RESULT_DIR, exist_ok=True)
    print(f"LAUNCH_ID={LAUNCH_ID}  GATE={GATE_THRESHOLD:.4f}\n")

    exp = np.load(EXPERT_CACHE, allow_pickle=True)
    tr_ep, ho_ep = make_episode_split(exp["episode_outcomes"], held_out_frac=0.20, seed=0)
    eids = exp["episode_ids"]
    tr_mask, ho_mask = np.isin(eids, tr_ep), np.isin(eids, ho_ep)
    train_feat, train_bins = exp["features"][tr_mask], exp["bins"][tr_mask]
    held_feat, held_bins, held_eids = exp["features"][ho_mask], exp["bins"][ho_mask], eids[ho_mask]

    mappo = np.load(MAPPO_CACHE, allow_pickle=True)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(mappo["features"]))
    n_held_m = int(0.2 * len(idx))
    mappo_held = mappo["features"][idx[:n_held_m]]
    mappo_train = mappo["features"][idx[n_held_m:]]

    data = (train_feat, train_bins, held_feat, held_bins, held_eids, ho_ep, mappo_train, mappo_held)

    all_results = []
    gate_table = {}   # name -> {g: passed}
    for name in args.candidates:
        gate_table[name] = {}
        for g in GAMMA_INTS:
            seed_results = [run(name, g, s, data) for s in SEEDS]
            all_results.extend(seed_results)
            dids = [r["final_did"] for r in seed_results]
            ci_los = [r["ci_lo"] for r in seed_results if not np.isnan(r["ci_lo"])]
            accs = [r["balanced_acc"] for r in seed_results]
            mean_did, min_ci = float(np.mean(dids)), (float(min(ci_los)) if ci_los else float("nan"))
            passed = (mean_did > GATE_THRESHOLD and min_ci > CI_LOWER_BAR
                     and float(np.mean(accs)) >= BALANCED_ACC_BAR
                     and all(r["decay_ok"] for r in seed_results))
            gate_table[name][g] = {"passed": passed, "mean_did": mean_did, "min_ci": min_ci}
            print(f"\n{name} g_int={g}: mean_did={mean_did:+.4f}  min_CI_lo={min_ci:.4f}  "
                  f"bal_acc={np.mean(accs):.4f}  ==> {'PASS' if passed else 'fail'}\n")

    with open(os.path.join(RESULT_DIR, "v2_critical_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"wrote {RESULT_DIR}/v2_critical_results.json")

    # ---- SHIP RULE ----
    # Primary: highest-capacity candidate that passes at gamma_int=0 (BCE
    # alone, the imitation objective, with no hinge involvement to invite the
    # "trained on the metric" question). Only fall back to a hinge-assisted
    # checkpoint (gamma_int>0) if NOTHING passes BCE-only, and label it as
    # such explicitly in the report.
    print("\n" + "=" * 70)
    print("SHIP RULE  (BCE-only preferred; hinge-assisted only as fallback)")
    print("=" * 70)
    bce_only_passers = [n for n in gate_table if gate_table[n].get(0.0, {}).get("passed")]
    if bce_only_passers:
        best = max(bce_only_passers, key=lambda n: gate_table[n][0.0]["mean_did"])
        print(f"SHIP (BCE-only): {best}  "
              f"(DiD={gate_table[best][0.0]['mean_did']:+.4f} at gamma_int=0)")
        for n in gate_table:
            if n != best and gate_table[n].get(0.0, {}).get("passed"):
                print(f"  also passed BCE-only: {n} "
                      f"(DiD={gate_table[n][0.0]['mean_did']:+.4f})")
    else:
        hinge_passers = [(n, g) for n in gate_table for g in gate_table[n]
                        if g > 0 and gate_table[n][g]["passed"]]
        if hinge_passers:
            n, g = max(hinge_passers, key=lambda ng: gate_table[ng[0]][ng[1]]["mean_did"])
            print(f"No BCE-only candidate passed. SHIP (HINGE-ASSISTED, label this")
            print(f"explicitly in the write-up): {n} at gamma_int={g} "
                  f"(DiD={gate_table[n][g]['mean_did']:+.4f})")
        else:
            print("Nothing passed, BCE-only or hinge-assisted. See Gate A fallback.")

    # per-candidate: does the hinge help, hurt, or do nothing relative to g=0?
    print("\nHINGE EFFECT per candidate (DiD at g vs DiD at g=0):")
    for n in gate_table:
        base = gate_table[n].get(0.0, {}).get("mean_did")
        if base is None:
            continue
        for g in sorted(gate_table[n]):
            if g == 0.0:
                continue
            delta = gate_table[n][g]["mean_did"] - base
            tag = "helps" if delta > 0.05 else ("hurts" if delta < -0.05 else "~neutral")
            print(f"  {n:16s} g={g:>5.1f}: delta={delta:+.4f}  ({tag})")


if __name__ == "__main__":
    main()