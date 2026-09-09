"""
r2_test.py -- v3, truncated-SVD ridge.

v2 (soft ridge shrinkage) still blew up on real data despite bounded,
outlier-free content. Diagnosis: football geometry features are far more
collinear than a single exact dependency (own_dist, own_angle, opp_dist,
opp_angle are all nonlinear functions of the same 10 numbers -- five
players' x,y). That produces MANY near-zero eigenvalues spread across a
huge magnitude range, not one clean zero. A single global alpha cannot
simultaneously regularize the near-null directions enough and leave the
informative ones alone -- soft shrinkage isn't the right tool here.

Fix: truncate. Keep only singular directions with S_i > tol * S_max, drop
the rest entirely (hard cutoff) instead of shrinking them. This is the
standard remedy for ill-conditioned/near-rank-deficient design matrices.

Usage:
    python r2_test_v3.py --cache expert_features_cache.npz
"""
import argparse
import numpy as np


def truncated_ridge_fit_predict(Xtr, ytr, Xte, alpha, tol):
    Xtr_b = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    Xte_b = np.hstack([Xte, np.ones((len(Xte), 1))])
    U, S, Vt = np.linalg.svd(Xtr_b, full_matrices=False)
    keep = S > tol * S[0]
    n_dropped = int((~keep).sum())
    S, U, Vt = S[keep], U[:, keep], Vt[keep, :]
    d = S / (S ** 2 + alpha)
    w = Vt.T @ (d * (U.T @ ytr))
    return Xte_b @ w, n_dropped, S


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def cv_r2(X, y, alpha, tol, k=5, seed=0):
    n = len(X)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    scores, drops = [], []
    for i in range(k):
        test_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
        Xtr, Xte = X[train_idx], X[test_idx]
        mu, sd = Xtr.mean(0), Xtr.std(0)
        keep_col = sd > 1e-6
        Xtr_s = (Xtr[:, keep_col] - mu[keep_col]) / sd[keep_col]
        Xte_s = (Xte[:, keep_col] - mu[keep_col]) / sd[keep_col]
        pred, n_dropped, S = truncated_ridge_fit_predict(Xtr_s, y[train_idx], Xte_s, alpha, tol)
        scores.append(r2_score(y[test_idx], pred))
        drops.append(n_dropped)
    return float(np.mean(scores)), scores, drops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--max-rows", type=int, default=60000)
    ap.add_argument("--alphas", type=float, nargs="+", default=[1, 10, 100])
    ap.add_argument("--tols", type=float, nargs="+", default=[1e-2, 1e-3, 1e-4, 1e-6])
    args = ap.parse_args()

    z = np.load(args.cache, allow_pickle=True)
    X = np.asarray(z["features"], dtype=np.float64)
    print(f"loaded {X.shape}")
    if len(X) > args.max_rows:
        rng = np.random.default_rng(0)
        X = X[rng.choice(len(X), args.max_rows, replace=False)]
        print(f"subsampled to {X.shape}")

    content = X[:, :137]

    # spectrum diagnostic on the full (standardized) content, once
    mu, sd = content.mean(0), content.std(0)
    keep = sd > 1e-6
    Cs = (content[:, keep] - mu[keep]) / sd[keep]
    S_full = np.linalg.svd(np.hstack([Cs, np.ones((len(Cs), 1))]), compute_uv=False)
    print(f"\nsingular value spectrum ({len(S_full)} total):")
    print(f"  max={S_full[0]:.2f}  top10={np.round(S_full[:10],2).tolist()}")
    print(f"  min={S_full[-1]:.6g}  bottom10={np.round(S_full[-10:],6).tolist()}")
    print(f"  condition number (max/min) = {S_full[0]/max(S_full[-1],1e-15):.3g}")
    n_below = {t: int((S_full < t*S_full[0]).sum()) for t in [1e-2,1e-3,1e-4,1e-6]}
    print(f"  directions below tol*max, by tol: {n_below}")

    for j, name in [(137, "T_norm"), (138, "dScore")]:
        y = X[:, j]
        print(f"\n{name}:")
        for tol in args.tols:
            for alpha in args.alphas:
                r2, folds, drops = cv_r2(content, y, alpha=alpha, tol=tol, k=5)
                print(f"  tol={tol:>8g}  alpha={alpha:>5g}  CV R2 = {r2:8.4f}  "
                      f"dirs dropped/fold: {drops}  per-fold: {[round(f,3) for f in folds]}")

    print("\nRead this as: find the (tol, alpha) region where R2 STOPS changing")
    print("wildly with tol -- that's the stable, trustworthy estimate. If R2 is")
    print("small and roughly constant across a range of tol/alpha, that's the answer.")


if __name__ == "__main__":
    main()