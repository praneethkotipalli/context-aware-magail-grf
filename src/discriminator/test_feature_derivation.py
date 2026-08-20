import numpy as np
from feature_derivation import compute_raw_features, BLOCK_SLICES

TEST_STEP = {
    'left_team': [[-8.27936053e-01,-2.99008407e-05],[1.10550642e-01,-2.27038667e-01],
                  [5.35520196e-01,2.95304861e-02],[5.26552558e-01,-5.43069839e-02],
                  [-5.50163090e-02,2.47348100e-01]],
    'left_team_direction': [[0.,-0.],[0.00641285,-0.00033835],[0.00918437,-0.00023009],
                             [0.01377078,0.00024407],[0.00643117,-0.00012024]],
    'right_team': [[9.85388756e-01,-6.94927992e-04],[3.41508806e-01,1.75381273e-01],
                    [5.15769005e-01,-3.01643424e-02],[5.52794576e-01,6.55649370e-03],
                    [4.00773108e-01,-1.98589697e-01]],
    'right_team_direction': [[-0.,0.],[0.00914511,-0.00059924],[0.01356041,-0.00022402],
                              [0.00889371,-0.00150965],[0.00918122,-0.00027561]],
    'ball': [0.5430004,-0.05330953,0.11984784],
    'ball_direction': [0.01337503,-0.00020072,0.02209653],
    'action_captured': 5, 'steps_left': 656, 'score_left': 0, 'score_right': 0,
}

EXPECTED = {
    'own_dist': [0.9656, 1.3638, 1.3556, 0.8115, 0.4964, 0.4504, 0.5024, 0.0843, 0.6294, 0.6551],
    'ball_rel': [1.372, -0.0388, 0.466, 0.382, 0.0832, -1.4807, 0.0165, 0.0606, 0.6693, -0.4659],
    'context': [0.2186, 0.0],
}

def run():
    raw = compute_raw_features(TEST_STEP)
    print("shape:", raw.shape)
    assert raw.shape == (137,)

    for block, expected in EXPECTED.items():
        actual = raw[BLOCK_SLICES[block]]
        ok = np.allclose(actual, expected, atol=1e-3)
        print(f"{block:10s}: {'PASS' if ok else 'FAIL'}   got={np.round(actual,4).tolist()}")
        assert ok, f"{block} mismatch"

    print("\nALL CHECKS PASSED -- feature_derivation.py is verified correct")

if __name__ == '__main__':
    run()