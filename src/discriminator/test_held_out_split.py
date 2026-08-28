"""test_held_out_split.py"""
import numpy as np
from held_out_split import make_episode_split, split_features_by_episode

def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name

cache = np.load("expert_features_cache.npz", allow_pickle=True)
features = cache["features"]
bins = cache["bins"]
episode_ids = cache["episode_ids"]
episode_outcomes = cache["episode_outcomes"]

print(f"52 episodes, outcomes: "
      f"{(episode_outcomes=='win').sum()}W / {(episode_outcomes=='draw').sum()}D / {(episode_outcomes=='loss').sum()}L")

train_ep, held_ep = make_episode_split(episode_outcomes, held_out_frac=0.20, seed=0)

print(f"\ntrain episodes: {len(train_ep)}, held-out episodes: {len(held_ep)}")
check("train + held-out episodes sum to 52", len(train_ep) + len(held_ep) == 52)
check("no episode appears in both splits", len(set(train_ep.tolist()) & set(held_ep.tolist())) == 0)
check("held-out is roughly 20% of episodes (got {}/52)".format(len(held_ep)), 8 <= len(held_ep) <= 13)

held_outcomes = episode_outcomes[held_ep]
print(f"held-out outcome breakdown: "
      f"{(held_outcomes=='win').sum()}W / {(held_outcomes=='draw').sum()}D / {(held_outcomes=='loss').sum()}L")
check("held-out set contains at least one win episode", (held_outcomes == 'win').sum() >= 1)
check("held-out set contains at least one draw episode", (held_outcomes == 'draw').sum() >= 1)
check("held-out set contains at least one loss episode", (held_outcomes == 'loss').sum() >= 1)

(train_feat, train_bins), (held_feat, held_bins) = split_features_by_episode(
    features, bins, episode_ids, train_ep, held_ep
)

check("train + held-out steps sum to 156052",
      train_feat.shape[0] + held_feat.shape[0] == 156052)
#check("train features shape correct", train_feat.shape[1] == 137)
#check("held-out features shape correct", held_feat.shape[1] == 137)
check("train features shape correct", train_feat.shape[1] == 139)
check("held-out features shape correct", held_feat.shape[1] == 139)

print(f"\ntrain steps: {train_feat.shape[0]}, held-out steps: {held_feat.shape[0]}")

# the actual point of doing this at episode level: does held-out still
# cover multiple context cells despite being whole matches, not a
# targeted step sample?
held_cells_present = set(held_bins.tolist())
print(f"context cells present in held-out set: {sorted(held_cells_present)} "
      f"({len(held_cells_present)}/9)")
check("held-out set spans more than one context cell (episodes cross contexts)",
      len(held_cells_present) > 1)

print("\nAll checks passed.")