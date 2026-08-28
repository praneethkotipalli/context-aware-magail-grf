import os, json, pickle
import numpy as np
from feature_derivation import BLOCK_SLICES

_CONST_PATH = os.path.join(os.path.dirname(__file__), 'normalization_constants.json')
with open(_CONST_PATH) as f:
    _CONSTANTS = json.load(f)

MAX_SPEED = _CONSTANTS['max_speed']
MAX_BALL_Z = _CONSTANTS['max_ball_z']
MAX_BALL_Z_VEL = _CONSTANTS['max_ball_z_vel']
PITCH_DIAG = np.sqrt(2.0**2 + 0.84**2)

PITCH_X_BOUND = 1.1     # fixed physical margin, NOT data-derived. 
PITCH_Y_BOUND = 0.462   # = 0.42 * 1.1, same margin factor as X for consistency.


class FeatureNormaliser:
    def __init__(self):
        self.fitted = False

    def fit(self, raw_data_array):
        """
        Because this normalizer uses fixed physical constants rather than empirical mean/std,
        it is mathematically immune to agent-distribution leakage.
        This fit() method simply satisfies the pipeline API and marks it ready.
        """
        self.fitted = True

    def transform(self, raw_data):
        if not self.fitted:
            print("Warning: FeatureNormaliser transform called before fit/load.")
            
        # Convert 1D single steps (live rollouts) to 2D for consistent vectorized processing
        is_single = (raw_data.ndim == 1)
        if is_single:
            norm = np.expand_dims(raw_data.copy(), 0)
        else:
            norm = raw_data.copy()

        s = BLOCK_SLICES

        # own positions
        pos = norm[:, s['own_pos']].reshape(-1, 2)
        pos[:, 0] = np.clip(pos[:, 0] / PITCH_X_BOUND, -1, 1)
        pos[:, 1] = np.clip(pos[:, 1] / PITCH_Y_BOUND, -1, 1)
        norm[:, s['own_pos']] = pos.reshape(-1, 10)

        # own/opponent velocities
        for vel_slice in [s['own_vel'], s['opp_vel']]:
            norm[:, vel_slice] = np.clip(norm[:, vel_slice] / MAX_SPEED, -1, 1)

        # own/opponent pairwise distances
        for dist_slice in [s['own_dist'], s['opp_dist']]:
            norm[:, dist_slice] = np.clip(norm[:, dist_slice] / PITCH_DIAG, 0, 1)

        # own/opponent pairwise angles
        for angle_slice in [s['own_angle'], s['opp_angle']]:
            norm[:, angle_slice] = norm[:, angle_slice] / np.pi

        # ball
        ball = norm[:, s['ball']]
        ball[:, 0] = np.clip(ball[:, 0] / PITCH_X_BOUND, -1, 1)
        ball[:, 1] = np.clip(ball[:, 1] / PITCH_Y_BOUND, -1, 1)
        ball[:, 2] = np.clip(ball[:, 2] / MAX_BALL_Z, 0, 1)
        ball[:, 3:5] = np.clip(ball[:, 3:5] / MAX_SPEED, -1, 1)
        ball[:, 5] = np.clip(ball[:, 5] / MAX_BALL_Z_VEL, -1, 1)
        norm[:, s['ball']] = ball

        # ball-relative
        br = norm[:, s['ball_rel']].reshape(-1, 2)
        br[:, 0] = np.clip(br[:, 0] / PITCH_DIAG, 0, 1)
        br[:, 1] = br[:, 1] / np.pi
        norm[:, s['ball_rel']] = br.reshape(-1, 10)

        # action one-hot AND new sticky states: already {0,1}, no change needed

        # context
        ctx = norm[:, s['context']]
        ctx[:, 1] = np.clip(ctx[:, 1], -3, 3) / 3.0
        norm[:, s['context']] = ctx

        return norm[0] if is_single else norm

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.__dict__.update(pickle.load(f))


if __name__ == '__main__':
    from test_feature_derivation import TEST_STEP
    from feature_derivation import compute_raw_features

    raw = compute_raw_features(TEST_STEP)
    
    normaliser = FeatureNormaliser()
    normaliser.fit(raw)
    norm = normaliser.transform(raw)

    print(f"raw  min={raw.min():.4f}  max={raw.max():.4f}")
    print(f"norm min={norm.min():.4f}  max={norm.max():.4f}")
    print()

    for block_name, block_slice in BLOCK_SLICES.items():
        vals = norm[block_slice]
        flag = "" if (vals.min() >= -1.01 and vals.max() <= 1.01) else "  <-- OUT OF RANGE"
        print(f"{block_name:10s}: min={vals.min():+.4f}  max={vals.max():+.4f}{flag}")