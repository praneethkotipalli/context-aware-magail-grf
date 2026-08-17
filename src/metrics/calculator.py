import numpy as np
from scipy.spatial import ConvexHull

# Empirically verified via inspect_sticky_actions.py (representation='raw'):
#   SPRINT_STICKY_INDEX = 8   (flips 0->1 on SPRINT, persists through IDLE,
#                              flips back on RELEASE_SPRINT)
#   DRIBBLE_STICKY_INDEX = 9  (same test, confirmed alongside sprint)
SPRINT_STICKY_INDEX = 8
DRIBBLE_STICKY_INDEX = 9

# GRF pitch area, normalised coords: x in [-1,1], y in [-0.42,0.42] -> 2.0*0.84
GRF_PITCH_AREA = 1.68


def compute_sap(sticky_sprint_log: np.ndarray) -> float:
    sprint_state = np.array(sticky_sprint_log, dtype=float)
    total = sprint_state.size
    if total == 0:
        return 0.0
    active = np.sum(sprint_state == 1.0)
    return float((active / total) * 100.0)


def compute_sbf(sticky_sprint_log: np.ndarray) -> float:
    sprint_state = np.array(sticky_sprint_log, dtype=float)
    if sprint_state.ndim < 2 or sprint_state.shape[0] < 2:
        return 0.0
    n_agents = sprint_state.shape[1]
    if n_agents == 0:
        return 0.0
    total_transitions = 0
    for agent_idx in range(n_agents):
        agent_sprint = sprint_state[:, agent_idx]
        total_transitions += np.sum(np.abs(np.diff(agent_sprint)) > 0)
    return float(total_transitions / n_agents)


def compute_mecha(position_log: np.ndarray, possession_log: np.ndarray) -> float:
    positions = np.array(position_log)
    possession = np.array(possession_log, dtype=bool)

    assert len(positions) == len(possession), (
        f"Length mismatch: {len(positions)} positions vs "
        f"{len(possession)} possession flags -- fix upstream logging."
    )

    if not np.any(possession) or len(positions) == 0:
        return 0.0

    possession_positions = positions[possession]
    hull_areas = []
    for step_pos in possession_positions:
        unique_pos = np.unique(step_pos, axis=0)
        if len(unique_pos) >= 3:
            try:
                hull = ConvexHull(unique_pos)
                hull_areas.append(hull.volume)
            except Exception:
                pass

    if not hull_areas:
        return 0.0

    mecha = float(np.mean(hull_areas) / GRF_PITCH_AREA)
    return float(np.clip(mecha, 0.0, 1.0))


def compute_cv(position_log: np.ndarray) -> float:
    positions = np.array(position_log)
    if len(positions) == 0:
        return 0.0
    centroids = np.mean(positions, axis=1)
    var_x = np.var(centroids[:, 0])
    var_y = np.var(centroids[:, 1])
    return float(var_x + var_y)


def extract_sprint_sticky_bits(raw_obs_list):
    return np.array([
        int(agent_obs['sticky_actions'][SPRINT_STICKY_INDEX])
        for agent_obs in raw_obs_list
    ])


def extract_team_positions(raw_obs_list, n_players=5):
    left_team = np.array(raw_obs_list[0]['left_team'])
    return left_team[:n_players]


def extract_possession(raw_obs_list):
    owned_team = raw_obs_list[0]['ball_owned_team']
    return bool(owned_team == 0)