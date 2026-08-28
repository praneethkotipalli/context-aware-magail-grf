"""
held_out_split.py

Episode-level stratified train/held-out split for the 75% accuracy gate
(Section 3.3.3). Splits by EPISODE, not step -- adjacent steps within one
episode are near-duplicates, so a step-level split would let the model
"generalize" to a near-twin of a training example. Stratified by outcome
so a random draw can't hand back a held-out set skewed toward one result
(given the corpus's own 14W/13D/25L imbalance, this is a real risk, not
a formality).
"""

import numpy as np


def make_episode_split(episode_outcomes, held_out_frac=0.20, seed=0):
    """
    episode_outcomes: (52,) array of 'win'/'loss'/'draw', indexed by
        episode_id -- straight from expert_features_cache.npz.
    held_out_frac: target fraction of EPISODES (not steps) held out.

    Returns (train_episode_ids, held_out_episode_ids), both sorted arrays
    of episode indices. Rounds PER OUTCOME GROUP independently (not on
    the pooled 52), which is what makes this genuinely stratified rather
    than just "shuffle then split and hope."
    """
    rng = np.random.default_rng(seed)
    train_ids, held_out_ids = [], []

    for outcome in ('win', 'draw', 'loss'):
        group = np.where(episode_outcomes == outcome)[0]
        n_group = len(group)
        n_held = max(1, round(n_group * held_out_frac)) if n_group > 0 else 0

        shuffled = group.copy()
        rng.shuffle(shuffled)
        held_out_ids.extend(shuffled[:n_held].tolist())
        train_ids.extend(shuffled[n_held:].tolist())

    return np.array(sorted(train_ids)), np.array(sorted(held_out_ids))


"""def split_features_by_episode(features, bins, episode_ids, train_episode_ids, held_out_episode_ids):
    ""Applies an episode-level split to the actual (156052, 137) step
    arrays -- expands episode membership into a per-STEP boolean mask.""
    train_mask = np.isin(episode_ids, train_episode_ids)
    held_out_mask = np.isin(episode_ids, held_out_episode_ids)

    return (
        (features[train_mask], bins[train_mask]),
        (features[held_out_mask], bins[held_out_mask]),
    )"""
def split_features_by_episode(features, bins, episode_ids, train_episode_ids, held_out_episode_ids):
    """Applies an episode-level split to the actual (156052, 139) step
    arrays -- expands episode membership into a per-STEP boolean mask."""
    train_mask = np.isin(episode_ids, train_episode_ids)
    held_out_mask = np.isin(episode_ids, held_out_episode_ids)

    return (
        (features[train_mask], bins[train_mask]),
        (features[held_out_mask], bins[held_out_mask]),
    )