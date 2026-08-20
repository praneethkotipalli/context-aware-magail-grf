# src/discriminator/feature_normalization.py
import os, json
import numpy as np
from feature_derivation import BLOCK_SLICES

_CONST_PATH = os.path.join(os.path.dirname(__file__), 'normalization_constants.json')
with open(_CONST_PATH) as f:
    _CONSTANTS = json.load(f)

MAX_SPEED = _CONSTANTS['max_speed']
MAX_BALL_Z = _CONSTANTS['max_ball_z']
MAX_BALL_Z_VEL = _CONSTANTS['max_ball_z_vel']
PITCH_DIAG = np.sqrt(2.0**2 + 0.84**2)

PITCH_X_BOUND = 1.1     # fixed physical margin, NOT data-derived. Confirmed: outfield
                         # players (not just GK) and the ball legitimately reach up to
                         # |x|=1.046 near goal areas/touchline runs, across 156k real steps.
PITCH_Y_BOUND = 0.462    # = 0.42 * 1.1, same margin factor as X for consistency.
                         # Confirmed: player 4 sustains |y|=0.446 for multiple consecutive
                         # steps (early_neutral_ep12, steps 277-286) -- a real touchline
                         # run, not a glitch (smooth, sustained values, not a single spike).


def normalize_features(raw):
    norm = raw.copy()
    s = BLOCK_SLICES

    # own positions: both axes now clipped to a fixed physical margin --
    # players legitimately run slightly past the nominal pitch boundary
    # near the touchline/goal area (verified against real recorded data)
    pos = norm[s['own_pos']].reshape(-1, 2)
    pos[:, 0] = np.clip(pos[:, 0] / PITCH_X_BOUND, -1, 1)
    pos[:, 1] = np.clip(pos[:, 1] / PITCH_Y_BOUND, -1, 1)
    norm[s['own_pos']] = pos.flatten()

    # own/opponent velocities: fixed player-speed cap
    for vel_slice in [s['own_vel'], s['opp_vel']]:
        norm[vel_slice] = np.clip(norm[vel_slice] / MAX_SPEED, -1, 1)

    # own/opponent pairwise distances: fixed pitch-diagonal cap
    for dist_slice in [s['own_dist'], s['opp_dist']]:
        norm[dist_slice] = np.clip(norm[dist_slice] / PITCH_DIAG, 0, 1)

    # own/opponent pairwise angles: fixed [-pi, pi] -> [-1, 1]
    for angle_slice in [s['own_angle'], s['opp_angle']]:
        norm[angle_slice] = norm[angle_slice] / np.pi

    # ball: x,y now use the same fixed margin as own_pos (same real
    # phenomenon -- ball follows play out toward the touchline/goal area);
    # z by its own cap; x,y velocity shares player speed scale; z velocity
    # gets its OWN cap -- a kicked/bounced ball's vertical speed is a
    # distinct physical quantity from horizontal running speed and
    # routinely exceeds it (verified: 1.98 vs 0.016 raw max)
    ball = norm[s['ball']]
    ball[0] = np.clip(ball[0] / PITCH_X_BOUND, -1, 1)
    ball[1] = np.clip(ball[1] / PITCH_Y_BOUND, -1, 1)
    ball[2] = np.clip(ball[2] / MAX_BALL_Z, 0, 1)
    ball[3:5] = np.clip(ball[3:5] / MAX_SPEED, -1, 1)
    ball[5] = np.clip(ball[5] / MAX_BALL_Z_VEL, -1, 1)
    norm[s['ball']] = ball

    # ball-relative (all 5 own players to ball): distance then angle, interleaved
    br = norm[s['ball_rel']].reshape(5, 2)
    br[:, 0] = np.clip(br[:, 0] / PITCH_DIAG, 0, 1)
    br[:, 1] = br[:, 1] / np.pi
    norm[s['ball_rel']] = br.flatten()

    # action one-hot: already {0,1}, no change

    # context: t_norm already [0,1]; delta_score per locked spec, clipped [-3,3]
    ctx = norm[s['context']]
    ctx[1] = np.clip(ctx[1], -3, 3) / 3.0
    norm[s['context']] = ctx

    return norm


if __name__ == '__main__':
    from test_feature_derivation import TEST_STEP
    from feature_derivation import compute_raw_features

    raw = compute_raw_features(TEST_STEP)
    norm = normalize_features(raw)

    print(f"raw   min={raw.min():.4f}  max={raw.max():.4f}")
    print(f"norm  min={norm.min():.4f}  max={norm.max():.4f}")
    print()

    for block_name, block_slice in BLOCK_SLICES.items():
        vals = norm[block_slice]
        flag = "" if (vals.min() >= -1.01 and vals.max() <= 1.01) else "  <-- OUT OF RANGE"
        print(f"{block_name:10s}: min={vals.min():+.4f}  max={vals.max():+.4f}{flag}")