import os, glob, time
import numpy as np
from pathlib import Path

from feature_derivation import compute_raw_features, BLOCK_SLICES
from feature_normalization import normalize_features

DISS_ROOT = os.path.join(str(Path.home()), "dissertation")
EPISODES_DIR = os.path.join(DISS_ROOT, "context-aware-magail-grf", "data", "baseline_rollouts_for_verification")

def step_dict_from_episode(d, t):
    """Build the same per-step dict shape compute_raw_features expects,
    pulled from one index of a full-episode .npz array."""
    return {
        'left_team': d['left_team'][t],
        'left_team_direction': d['left_team_direction'][t],
        'right_team': d['right_team'][t],
        'right_team_direction': d['right_team_direction'][t],
        'ball': d['ball'][t],
        'ball_direction': d['ball_direction'][t],
        'action_captured': d['action_captured'][t],
        'steps_left': d['steps_left'][t],
        'score_left': d['score_left'][t],
        'score_right': d['score_right'][t],
    }


def run():
    paths = sorted(glob.glob(os.path.join(EPISODES_DIR, "*.npz")))
    print(f"Batch-verifying {len(paths)} episodes, full step-by-step...\n")

    block_min = {b: np.inf for b in BLOCK_SLICES}
    block_max = {b: -np.inf for b in BLOCK_SLICES}
    out_of_range_events = []   # (episode_file, step, block, value)
    nan_inf_events = []        # (episode_file, step, block)
    total_steps = 0

    t0 = time.time()
    for ep_idx, p in enumerate(paths):
        d = dict(np.load(p, allow_pickle=True))
        n_steps = len(d['steps_left'])
        fname = os.path.basename(p)

        for t in range(n_steps):
            step = step_dict_from_episode(d, t)
            raw = compute_raw_features(step)
            norm = normalize_features(raw)
            total_steps += 1

            for block_name, block_slice in BLOCK_SLICES.items():
                vals = norm[block_slice]

                if not np.all(np.isfinite(vals)):
                    nan_inf_events.append((fname, t, block_name))
                    continue  # skip min/max tracking for this corrupted block/step

                block_min[block_name] = min(block_min[block_name], vals.min())
                block_max[block_name] = max(block_max[block_name], vals.max())

                if vals.min() < -1.01 or vals.max() > 1.01:
                    out_of_range_events.append((fname, t, block_name, float(vals.min()), float(vals.max())))

        if (ep_idx + 1) % 10 == 0 or ep_idx == len(paths) - 1:
            elapsed = time.time() - t0
            print(f"  processed {ep_idx+1}/{len(paths)} episodes  ({total_steps} steps, {elapsed:.1f}s elapsed)")

    elapsed = time.time() - t0
    print(f"\nDone. {total_steps} total steps across {len(paths)} episodes, {elapsed:.1f}s.\n")

    print("=" * 70)
    print("PER-BLOCK RANGE OBSERVED ACROSS FULL DATASET")
    print("=" * 70)
    for block_name in BLOCK_SLICES:
        lo, hi = block_min[block_name], block_max[block_name]
        flag = "" if (lo >= -1.01 and hi <= 1.01) else "  <-- EXCEEDS EXPECTED RANGE"
        print(f"  {block_name:10s}: min={lo:+.4f}  max={hi:+.4f}{flag}")

    print("\n" + "=" * 70)
    print(f"NaN/Inf EVENTS: {len(nan_inf_events)}")
    print("=" * 70)
    if nan_inf_events:
        for fname, t, block in nan_inf_events[:20]:
            print(f"  {fname}  step={t}  block={block}")
        if len(nan_inf_events) > 20:
            print(f"  ... and {len(nan_inf_events) - 20} more")
    else:
        print("  none -- clean")

    print("\n" + "=" * 70)
    print(f"OUT-OF-RANGE EVENTS (|value| > 1.01): {len(out_of_range_events)}")
    print("=" * 70)
    if out_of_range_events:
        by_block = {}
        for fname, t, block, lo, hi in out_of_range_events:
            by_block.setdefault(block, []).append((fname, t, lo, hi))
        for block, events in by_block.items():
            print(f"\n  {block}: {len(events)} event(s)")
            for fname, t, lo, hi in events[:5]:
                print(f"    {fname}  step={t}  min={lo:+.4f}  max={hi:+.4f}")
            if len(events) > 5:
                print(f"    ... and {len(events) - 5} more in this block")
    else:
        print("  none -- every step, every block, within expected range")

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    if not nan_inf_events and not out_of_range_events:
        print("  PASS -- feature derivation + normalization verified clean")
        print("  across the full recorded dataset. Safe to proceed to the")
        print("  discriminator architecture.")
    else:
        print("  ISSUES FOUND -- do not proceed to the discriminator until")
        print("  the events above are understood and resolved (or")
        print("  deliberately accepted with a documented reason).")


if __name__ == '__main__':
    run()