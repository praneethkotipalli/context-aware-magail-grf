"""
evaluate_policy.py

Runs N episodes with the CURRENT policy, explore=False (deterministic --
this is evaluation, not rollout collection for training), and computes
win rate over the full batch, per methodology's own protocol ("evaluated
over at least 50 episodes per condition") -- NOT the single-episode
proxy pilot_finetune_iteration.py used to demonstrate the scheduler API.

This is what should actually feed AlignmentTaxScheduler.update() during
real training, and what determines the kill-switch/anneal decisions for
real -- the pilot's single-episode number was explicitly flagged as
untrustworthy for exactly this reason.
"""

import numpy as np
import torch

from metrics import compute_sap, compute_win_rate, compute_mecha, compute_csi_proxy


def evaluate_policy(actor, encoder, env_factory, n_episodes=50, max_steps=3000):
    """
    actor: ContextConditionedActor, in eval mode, explore=False internally.
    env_factory: callable() -> fresh GRF env (evaluation runs its own
        episodes, separate from training rollout collection).

    Returns a dict of all metrics -- caller decides what to log/act on.
    """
    actor.eval()
    final_scores = []
    all_sap = []
    all_mecha = []
    all_t_norm, all_delta_score, all_sprint_state = [], [], []

    for ep in range(n_episodes):
        env = env_factory()
        raw_obs = env.reset()

        sticky_log, pos_log, poss_log = [], [], []
        step, done = 0, False

        while not done and step < max_steps:
            o_shared = raw_obs[0]
            t_norm = o_shared['steps_left'] / 3001.0
            delta_score = o_shared['score'][0] - o_shared['score'][1]

            obs_192, avail_masks = [], []
            for i in range(4):
                from minimal_state import MinimalState
                state = MinimalState(n_player=5); state.set_obs(raw_obs[i])
                obs_192.append(encoder.encode_each(state))
                avail_masks.append(encoder.get_available_actions(raw_obs[i], 0.0, []))
            obs_192_t = torch.as_tensor(np.stack(obs_192), dtype=torch.float32)
            mask_t = torch.as_tensor(np.stack(avail_masks), dtype=torch.float32)
            ctx = np.array([t_norm, np.clip(delta_score, -3, 3) / 3.0], dtype=np.float32)
            ctx_t = torch.as_tensor(ctx).unsqueeze(0).repeat(4, 1)

            with torch.no_grad():
                actions, _, _, _ = actor(obs_192_t, ctx_t, mask_t, explore=False)

            ref_i = int(np.argmin([
                np.linalg.norm(np.asarray(raw_obs[0]['left_team'][j+1]) - np.asarray(raw_obs[0]['ball'][:2]))
                for j in range(4)
            ]))
            sticky_log.append(raw_obs[ref_i]['sticky_actions'])
            pos_log.append(o_shared['left_team'])
            poss_log.append(bool(o_shared['ball_owned_team'] == 0))

            all_t_norm.append(t_norm)
            all_delta_score.append(delta_score)
            all_sprint_state.append(raw_obs[ref_i]['sticky_actions'][8] == 1)

            raw_obs, _, done, _ = env.step(actions.tolist())
            step += 1

        final_scores.append((raw_obs[0]['score'][0], raw_obs[0]['score'][1]))
        sticky_arr = np.array(sticky_log)
        all_sap.append(compute_sap(sticky_arr))
        mecha_val = compute_mecha(np.array(pos_log), np.array(poss_log))
        if not np.isnan(mecha_val):
            all_mecha.append(mecha_val)

        env.close()

    win_stats = compute_win_rate(final_scores)
    csi_signed, n_lw, n_ll = compute_csi_proxy(
        np.array(all_t_norm), np.array(all_delta_score),
        np.array(all_sprint_state, dtype=float),
    )

    return {
        **win_stats,
        "sap_mean": float(np.mean(all_sap)),
        "sap_std": float(np.std(all_sap)),
        "mecha_mean": float(np.mean(all_mecha)) if all_mecha else float('nan'),
        "csi_sap_proxy": csi_signed,     # see metrics.py docstring -- PROXY, not the locked controlled-scenario CSI
        "csi_proxy_n_lw": n_lw, "csi_proxy_n_ll": n_ll,
    }