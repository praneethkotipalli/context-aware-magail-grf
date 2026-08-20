import sys, os, json, glob
import numpy as np
from pathlib import Path

DISS_ROOT = os.path.join(str(Path.home()), "dissertation")
EPISODES_DIR = os.path.join(DISS_ROOT, "context-aware-magail-grf", "data", "demonstrations", "episodes")


def robust_max(values, percentile=99.9):
    """Percentile instead of raw max -- robust to a single extreme but
    legitimate event (e.g. one exceptionally powerful shot) setting the
    scale for everything else. Ball height specifically benefits: it's
    long-tailed (mostly near-ground rolling, rare aerial spikes), so a
    99.9th percentile captures 'how high the ball realistically gets'
    as a distribution property rather than chasing whichever single
    recorded value happens to be most extreme."""
    return float(np.percentile(values, percentile))


def run():
    paths = sorted(glob.glob(os.path.join(EPISODES_DIR, "*.npz")))
    print(f"Scanning {len(paths)} episodes for normalization bounds...")

    all_speeds = []
    all_ball_z = []
    all_ball_z_vel = []

    for p in paths:
        d = dict(np.load(p, allow_pickle=True))
        all_speeds.append(np.linalg.norm(d['left_team_direction'], axis=2).ravel())
        all_speeds.append(np.linalg.norm(d['right_team_direction'], axis=2).ravel())
        all_ball_z.append(d['ball'][:, 2])
        all_ball_z_vel.append(np.abs(d['ball_direction'][:, 2]))

    all_speeds_pooled = np.concatenate(all_speeds)
    all_ball_z_pooled = np.concatenate(all_ball_z)
    all_ball_z_vel_pooled = np.concatenate(all_ball_z_vel)

    max_speed_observed = float(all_speeds_pooled.max())
    max_ball_z_observed = robust_max(all_ball_z_pooled)              # <-- now 99.9th percentile, not raw max
    max_ball_z_vel_observed = float(all_ball_z_vel_pooled.max())

    constants = {
        'max_speed': max_speed_observed * 1.2,
        'max_ball_z': max_ball_z_observed * 1.2,
        'max_ball_z_vel': max_ball_z_vel_observed * 1.2,
        'max_speed_observed_raw': max_speed_observed,
        'max_ball_z_observed_raw': max_ball_z_observed,
        'max_ball_z_observed_true_max': float(all_ball_z_pooled.max()),  # kept for reference/comparison
        'max_ball_z_vel_observed_raw': max_ball_z_vel_observed,
        'n_episodes_used': len(paths),
        'ball_z_method': '99.9th percentile (robust to single extreme legitimate events)',
    }

    out_path = os.path.join(os.path.dirname(__file__), 'normalization_constants.json')
    with open(out_path, 'w') as f:
        json.dump(constants, f, indent=2)

    print(f"max_speed        observed: {max_speed_observed:.5f}  -> with margin: {constants['max_speed']:.5f}")
    print(f"max_ball_z       observed (99.9th pct): {max_ball_z_observed:.5f}  -> with margin: {constants['max_ball_z']:.5f}")
    print(f"  (true raw max was: {constants['max_ball_z_observed_true_max']:.5f} -- kept for reference, not used)")
    print(f"max_ball_z_vel   observed: {max_ball_z_vel_observed:.5f}  -> with margin: {constants['max_ball_z_vel']:.5f}")
    print(f"\nWritten to {out_path}")


if __name__ == '__main__':
    run()