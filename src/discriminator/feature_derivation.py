import numpy as np
from itertools import combinations, product

OWN_PAIRS = list(combinations(range(5), 2))
OPP_PAIRS = list(product(range(5), range(5)))

"""BLOCK_SLICES = {
    'own_pos': slice(0, 10), 'own_vel': slice(10, 20),
    'own_dist': slice(20, 30), 'own_angle': slice(30, 40),
    'opp_dist': slice(40, 65), 'opp_angle': slice(65, 90),
    'opp_vel': slice(90, 100), 'ball': slice(100, 106),
    'ball_rel': slice(106, 116), 'action': slice(116, 135),
    'context': slice(135, 137),
}"""
BLOCK_SLICES = {
    'own_pos': slice(0, 10), 'own_vel': slice(10, 20),
    'own_dist': slice(20, 30), 'own_angle': slice(30, 40),
    'opp_dist': slice(40, 65), 'opp_angle': slice(65, 90),
    'opp_vel': slice(90, 100), 'ball': slice(100, 106),
    'ball_rel': slice(106, 116), 'action': slice(116, 135),
    'sticky': slice(135, 137),
    'context': slice(137, 139),
}

def compute_raw_features(step):
    left = np.asarray(step['left_team'])
    left_dir = np.asarray(step['left_team_direction'])
    right = np.asarray(step['right_team'])
    right_dir = np.asarray(step['right_team_direction'])
    ball = np.asarray(step['ball'])
    ball_dir = np.asarray(step['ball_direction'])

    b1 = left.flatten()
    b2 = left_dir.flatten()

    b3 = np.array([np.linalg.norm(left[j] - left[i]) for i, j in OWN_PAIRS])
    b4 = np.array([np.arctan2((left[j]-left[i])[1], (left[j]-left[i])[0]) for i, j in OWN_PAIRS])

    b5a = np.array([np.linalg.norm(right[j] - left[i]) for i, j in OPP_PAIRS])
    b5b = np.array([np.arctan2((right[j]-left[i])[1], (right[j]-left[i])[0]) for i, j in OPP_PAIRS])

    b6 = right_dir.flatten()
    b7 = np.concatenate([ball, ball_dir])

    b8 = []
    for i in range(5):
        d = ball[:2] - left[i]
        b8.append(np.linalg.norm(d))
        b8.append(np.arctan2(d[1], d[0]))
    b8 = np.array(b8)

    b9 = np.zeros(19)
    b9[int(step['action_captured'])] = 1.0

    # Extract continuous sprint and dribble states
    sticky = np.asarray(step['sticky_actions'])
    b_sticky = np.array([sticky[8], sticky[9]], dtype=float)

    t_norm = step['steps_left'] / 3001.0
    delta_score = int(step['score_left']) - int(step['score_right'])
    b10 = np.array([t_norm, delta_score])

    """raw = np.concatenate([b1, b2, b3, b4, b5a, b5b, b6, b7, b8, b9, b10])
    assert raw.shape[0] == 137, f"expected 137, got {raw.shape[0]}"
    return raw"""
    # Concatenate with b_sticky injected right before context
    raw = np.concatenate([b1, b2, b3, b4, b5a, b5b, b6, b7, b8, b9, b_sticky, b10])
    
    assert raw.shape[0] == 139, f"expected 139, got {raw.shape[0]}"
    return raw