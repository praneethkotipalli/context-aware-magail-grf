"""
diagnose_leakage.py

Answers four questions about the 139-dim discriminator input, using only
cached feature arrays. No GRF, no GPU, runs in seconds on the laptop.

  Q1  Can ONE dim (135, sprint state) separate expert from agent?
      If yes, the discriminator never needed the context channel and the
      "dilution" story has a concrete mechanism.

  Q2  How much does accuracy drop when dims 135:137 are removed?
      Quantifies how much of the 0.9606 held-out figure is sprint state.

  Q3  Is context recoverable from the 137 content dims?
      Tests the leak in the direction Section 3.3.2 worries about, and the
      reverse direction (behaviour -> context) that makes the context input
      redundant rather than diluted.

  Q4  Per-cell sprint separation: is the expert/agent gap uniform across
      context cells, or does it vary? If uniform, a context-blind reward
      is sufficient to explain all observed SAP movement.

Usage:
    python diagnose_leakage.py \
        --expert src/discriminator/expert_features_cache.npz \
        --agent  src/discriminator/agent_features_cache.npz

If the agent cache has a different filename, pass it; the script only needs
an (N, 139) float array under key 'features'.
"""

import argparse
import numpy as np

SPRINT_DIM = 135
DRIBBLE_DIM = 136
CTX = slice(137, 139)
CONTENT = slice(0, 137)


def load(path, key="features"):
    z = np.load(path, allow_pickle=True)
    if key not in z:
        raise SystemExit(f"{path} has keys {list(z.keys())}, no '{key}'")
    x = np.asarray(z[key], dtype=np.float64)
    if x.shape[1] != 139:
        raise SystemExit(f"{path}: expected 139 dims, got {x.shape[1]}")
    return x, z


def logistic(X, y, iters=400, lr=0.5, l2=1e-4):
    """Plain logistic regression, no sklearn dependency."""
    X = np.hstack([X, np.ones((len(X), 1))])
    mu, sd = X[:, :-1].mean(0), X[:, :-1].std(0) + 1e-9
    X[:, :-1] = (X[:, :-1] - mu) / sd
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
        g = X.T @ (p - y) / len(y) + l2 * w
        w -= lr * g
    return w, mu, sd


def acc(X, y, w, mu, sd):
    X = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    p = 1.0 / (1.0 + np.exp(-np.clip(X @ w, -30, 30)))
    return float(((p > 0.5).astype(int) == y).mean())


def split(n, frac=0.25, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    k = int(n * frac)
    return idx[k:], idx[:k]


def fit_and_score(Xe, Xa, cols, label):
    X = np.vstack([Xe[:, cols], Xa[:, cols]])
    y = np.concatenate([np.ones(len(Xe)), np.zeros(len(Xa))])
    tr, te = split(len(X))
    w, mu, sd = logistic(X[tr], y[tr])
    a = acc(X[te], y[te], w, mu, sd)
    print(f"  {label:52s} held-out acc = {a:.4f}")
    return a


def ridge_r2(X, Y):
    """Closed-form ridge, reports per-target held-out R^2."""
    tr, te = split(len(X))
    Xtr = np.hstack([X[tr], np.ones((len(tr), 1))])
    Xte = np.hstack([X[te], np.ones((len(te), 1))])
    A = Xtr.T @ Xtr + 1e-3 * np.eye(Xtr.shape[1])
    W = np.linalg.solve(A, Xtr.T @ Y[tr])
    P = Xte @ W
    out = []
    for j in range(Y.shape[1]):
        ss_res = float(((Y[te, j] - P[:, j]) ** 2).sum())
        ss_tot = float(((Y[te, j] - Y[te, j].mean()) ** 2).sum())
        out.append(1.0 - ss_res / max(ss_tot, 1e-12))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expert", required=True)
    ap.add_argument("--agent", required=True)
    ap.add_argument("--max-rows", type=int, default=60000)
    args = ap.parse_args()

    Xe, ze = load(args.expert)
    Xa, _ = load(args.agent)
    rng = np.random.default_rng(0)
    if len(Xe) > args.max_rows:
        Xe = Xe[rng.choice(len(Xe), args.max_rows, replace=False)]
    if len(Xa) > args.max_rows:
        Xa = Xa[rng.choice(len(Xa), args.max_rows, replace=False)]
    print(f"expert {Xe.shape}   agent {Xa.shape}\n")

    print("Q1/Q2  EXPERT vs AGENT separability by feature subset")
    print(f"  raw sprint-state mean: expert {Xe[:, SPRINT_DIM].mean():.4f}  "
          f"agent {Xa[:, SPRINT_DIM].mean():.4f}")
    a_sprint = fit_and_score(Xe, Xa, [SPRINT_DIM], "dim 135 (sprint state) ALONE")
    fit_and_score(Xe, Xa, [SPRINT_DIM, DRIBBLE_DIM], "dims 135:137 (sticky) alone")
    a_full = fit_and_score(Xe, Xa, list(range(139)), "all 139 dims")
    no_sticky = [i for i in range(137) if i not in (SPRINT_DIM, DRIBBLE_DIM)]
    a_nostick = fit_and_score(Xe, Xa, no_sticky, "content WITHOUT sticky (135 dims)")
    fit_and_score(Xe, Xa, list(range(116, 135)), "action one-hot alone (dims 116:135)")
    print(f"\n  -> sprint alone recovers {100*a_sprint/max(a_full,1e-9):.1f}% of full accuracy")
    print(f"  -> removing sticky costs {100*(a_full-a_nostick):.2f} accuracy points")
    if a_sprint > 0.85:
        print("  -> VERDICT: one dimension solves the task. The context channel is")
        print("     redundant, not diluted, and r_style is ~monotone in sprint state.")

    print("\nQ3  Is CONTEXT recoverable from the 137 content dims?")
    Xall = np.vstack([Xe, Xa])
    r2 = ridge_r2(Xall[:, CONTENT], Xall[:, CTX])
    print(f"  pooled  R^2(T_norm)={r2[0]:+.3f}   R^2(dScore)={r2[1]:+.3f}")
    r2e = ridge_r2(Xe[:, CONTENT], Xe[:, CTX])
    print(f"  expert  R^2(T_norm)={r2e[0]:+.3f}   R^2(dScore)={r2e[1]:+.3f}")
    r2a = ridge_r2(Xa[:, CONTENT], Xa[:, CTX])
    print(f"  agent   R^2(T_norm)={r2a[0]:+.3f}   R^2(dScore)={r2a[1]:+.3f}")
    print("  (high pooled R^2 with low within-class R^2 = context is a class proxy;")
    print("   high within-class R^2 = behaviour encodes context, input is redundant)")

    print("\nQ4  Per-cell expert/agent sprint gap (is the gap context-dependent?)")
    if "bins" in ze:
        be = np.asarray(ze["bins"])
        cells = sorted(set(be.tolist()))
        print(f"  {'cell':>10s} {'n_expert':>9s} {'expert sprint':>14s}")
        for c in cells:
            m = be[: len(Xe)] == c if len(be) >= len(Xe) else None
            if m is None or m.sum() == 0:
                continue
            print(f"  {str(c):>10s} {int(m.sum()):9d} {Xe[m, SPRINT_DIM].mean():14.4f}")
        print("  If the AGENT sprint rate is ~flat across cells (finding #14: 4.2pp),")
        print("  a context-blind reward explains 100% of observed SAP movement.")
    else:
        print("  no 'bins' key in the expert cache; skipping")

    print("\nAlso check in W&B, per run:")
    print("  mean(disc_update_skipped)   ~1.0 => the SC shuffle never executed")
    print("  gate/*_shift over training  identical C vs SC => same discriminator")
    print("  iterations_completed, halted (checkpoint fields)")


if __name__ == "__main__":
    main()