"""
build_shuffled_context_caches.py

Builds the ONE-TIME, data-level shuffled caches D_int_SC trains on. Run
ONCE, deterministic (fixed seed), reused by every SC training run.

WHY DATA-LEVEL, NOT RUNTIME: the causal question SC exists to answer is
"did the discriminator's ACCESS to context during training cause it to
learn something real" -- which requires that no genuine (context, state)
association exists ANYWHERE in its training data, not merely that its
input is corrupted at deployment. See the finetune_loop.py rewrite notes.

DESIGN CHOICE: permute context (dims 137:139) WITHIN each population
independently (expert shuffled among itself, MAPPO shuffled among
itself, random-policy shuffled among itself) -- this breaks each
population's internal context<->behaviour pairing while leaving cross-
population marginal differences untouched, which is what BCE is
supposed to be learning FROM in the first place.

SAMPLING BINS ARE NOT RECOMPUTED: `bins` stays derived from the TRUE
context (as originally computed by classify_bin), so late-cell
upweighted sampling continues to mean "this row was genuinely late in
the match" for stratification purposes. Only the context VALUES actually
written into `features[:, 137:139]` -- what the model is trained to
associate with this row -- are shuffled.

Run:  python build_shuffled_context_caches.py
"""
import numpy as np

CTX_SLICE = slice(137, 139)
SEED = 0

SOURCES = {
    "expert_features_cache.npz": "expert_features_cache_SHUFFLED.npz",
    "random_policy_features_cache.npz": "random_policy_features_cache_SHUFFLED.npz",
    "mappo_features_cache.npz": "mappo_features_cache_SHUFFLED.npz",
}


def shuffle_one(path, out_path, seed):
    z = dict(np.load(path, allow_pickle=True))
    feat = z["features"].copy()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(feat))
    feat[:, CTX_SLICE] = feat[perm, CTX_SLICE]   # context reassigned across rows
    z["features"] = feat
    # bins UNCHANGED -- still derived from each row's TRUE original context,
    # used only for sampling stratification, never fed to the model
    np.savez_compressed(out_path, **z)
    print(f"  {path} -> {out_path}  ({len(feat):,} rows, context permuted, bins unchanged)")


def main():
    for src, out in SOURCES.items():
        shuffle_one(src, out, SEED)
    print("\ndone. These SHUFFLED caches are used ONLY for D_int_SC training --")
    print("D_int_C and D_int_NC continue to use the original, unshuffled caches.")


if __name__ == "__main__":
    main()