"""
verify_all_D_int_gates.py

Runs the interaction gate against all three production D_int checkpoints.
Expected results, stated in advance so you can check against them:

  C   -- should PASS, DiD well above 0.564 (matches Stage 0 screening,
         now on the full corpus instead of the 80/20 split)
  NC  -- DiD should be EXACTLY 0 (to floating-point precision), not just
         small. This isn't a training outcome, it's a mathematical
         identity: SprintOnlyDiscriminator(x) = w*spr + b depends only on
         spr. Within each of the gate's four groups, spr is HELD FIXED
         (that's how the groups are defined), so D() returns a CONSTANT
         per group. The DiD subtraction (D(spr=1)-D(spr=0)) equals w in
         BOTH the late-loss and late-win terms, so DiD = w - w = 0
         regardless of what w actually is. If this ISN'T ~0, something
         is wrong with the gate code, not with the model.
  SC  -- should FAIL, DiD near zero (not exactly zero like NC -- SC still
         has interaction TERMS in its architecture, they just shouldn't
         have learned anything real from a corpus with the pairing
         destroyed). A passing SC would mean the shuffle failed to
         destroy the learnable signal.

Run:  python verify_all_D_int_gates.py
"""
import numpy as np
import torch

from discriminator_candidates import TinyInteract, SprintOnlyDiscriminator
from interaction_loss import interaction_gate, GATE_THRESHOLD
from held_out_split import make_episode_split, split_features_by_episode

CACHES = {
    "C":  ("expert_features_cache.npz",          TinyInteract,          "D_int_C_production.pt"),
    "NC": ("expert_features_cache.npz",          SprintOnlyDiscriminator, "D_int_NC_production.pt"),
    "SC": ("expert_features_cache_SHUFFLED.npz", TinyInteract,          "D_int_SC_production.pt"),
}


def main():
    print(f"GATE THRESHOLD = {GATE_THRESHOLD:.4f}\n")
    for variant, (cache_path, model_cls, ckpt_path) in CACHES.items():
        cache = np.load(cache_path, allow_pickle=True)
        # NOTE: production checkpoints trained on the FULL corpus (no held-out
        # split withheld). For this gate check we still need SOME held-out-like
        # split to evaluate on -- reuse the same 80/20 split construction for
        # consistency with Stage 0's methodology, understanding this data was
        # also used in training the production checkpoint (not a true
        # generalisation test here -- Stage 0 already established that on the
        # withheld split; this run is a SANITY check on the shuffle/architecture,
        # not a re-validation of generalisation).
        tr, ho = make_episode_split(cache["episode_outcomes"], held_out_frac=0.20, seed=0)
        (_, _), (held_feat, held_bins) = split_features_by_episode(
            cache["features"], cache["bins"], cache["episode_ids"], tr, ho)

        model = model_cls()
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu")["model_state_dict"])
        model.eval()

        passed, did, sizes, warn = interaction_gate(model, held_feat, held_bins)
        flag = "PASS" if passed else "fail"
        print(f"{variant:4s} ({model_cls.__name__:24s}): DiD={did:+.6f}  {flag}")
        if warn:
            print(f"      WARNING: {warn}")

    print("\nExpected: C passes clearly, NC prints ~0.000000 exactly, SC fails near 0.")
    print("If NC is not extremely close to zero, check the gate/model code before trusting anything else.")
    print("If SC PASSES, the shuffle did not destroy the learnable signal -- do not use it as a control.")


if __name__ == "__main__":
    main()