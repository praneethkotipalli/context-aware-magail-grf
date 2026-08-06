"""
src/grf_baseline/check_all_architectures.py

Reads each candidate checkpoint's desc.pkl to determine which model
architecture / encoder it actually uses, BEFORE running any inference.
This is the check we skipped -- we only ever confirmed PassingMain_v2
uses enhanced_LightActionMask_5 (192-dim). Other checkpoints may use a
different architecture (e.g. plain encoder_basic_5, 133-dim).
"""

import pickle
import os

GRF_MARL_ROOT = "/home/praneeth/dissertation/GRF_MARL"

CANDIDATES = [
    "3-1_LongPass", "3-1_formation",
    "GKBug_v0", "GKBug_v1", "GKBug_v2", "GKBug_v3",
    "PassingMain_v1", "PassingMain_v2",
    "defense_v3",
]

for policy_name in CANDIDATES:
    desc_path = os.path.join(
        GRF_MARL_ROOT, "light_malib", "trained_models", "gr_football",
        "5_vs_5", policy_name, "desc.pkl"
    )
    try:
        with open(desc_path, "rb") as f:
            desc = pickle.load(f)
        model_name = desc.get("model_config", {}).get("model", "UNKNOWN")
        obs_shape = desc.get("observation_space").shape if desc.get("observation_space") is not None else "UNKNOWN"
        print(f"{policy_name:20s} model={model_name:45s} obs_shape={obs_shape}")
    except Exception as e:
        print(f"{policy_name:20s} ERROR: {e}")