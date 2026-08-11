import sys, os, csv, argparse
import numpy as np

sys.path.insert(0, "/home/praneeth/dissertation/football")
sys.path.insert(0, "/home/praneeth/dissertation/GRF_MARL")
sys.path.insert(0, "/home/praneeth/dissertation/context-aware-magail-grf/src/grf_baseline")
sys.path.insert(0, "/home/praneeth/dissertation/context-aware-magail-grf/src/metrics")
sys.path.insert(0, "/home/praneeth/dissertation/GRF_MARL/light_malib/model/gr_football/enhanced_LightActionMask_5")

from gfootball.env import config
from gfootball.env import football_env
from minimal_state import MinimalState
from enhanced_LightActionMask_5 import FeatureEncoder
from calculator import compute_sap, compute_mecha

OUT_DIR = "/home/praneeth/dissertation/context-aware-magail-grf/data/demonstrations"
DATA_CSV = os.path.join(OUT_DIR, "all_demonstrations.csv")
SUMMARY_CSV = os.path.join(OUT_DIR, "session_summary.csv")
FIELDS = ['context_label', 'episode_id', 'step', 't_norm', 'delta_score', 'sprint_bits', 'position', 'possession']
SUM_FIELDS = ['episode_id', 'context_label', 'n_steps', 'sap', 'mecha']


def make_env(level):
    cfg = config.Config({'action_set': 'default', 'players': ['keyboard:left_players=1'],
                          'level': level, 'representation': 'raw', 'real_time': True,
                          'rewards': 'scoring'})
    env = football_env.FootballEnv(cfg)
    env.render()
    return env


def next_id():
    if not os.path.exists(SUMMARY_CSV):
        return 1
    with open(SUMMARY_CSV) as f:
        rows = list(csv.DictReader(f))
    return (max(int(r['episode_id']) for r in rows) + 1) if rows else 1


def record_episode(env, ctx, ep_id):
    obs = env.reset()
    buf = []
    step, done = 0, False
    while not done:
        sprint = [int(obs['left_agent_sticky_actions'][0][8])]
        pos = np.array(obs['left_team'])[:5]
        buf.append({'step': step, 't_norm': obs['steps_left'] / 3001.0,
                    'delta_score': obs['score'][0] - obs['score'][1],
                    'sprint_bits': sprint, 'position': pos,
                    'possession': bool(obs['ball_owned_team'] == 0)})
        obs, reward, done, info = env.step([])
        step += 1

    sap = compute_sap(np.array([r['sprint_bits'] for r in buf]))
    mecha = compute_mecha(np.array([r['position'] for r in buf]),
                           np.array([r['possession'] for r in buf]))

    ep_path = os.path.join(OUT_DIR, f"{ctx}_ep{ep_id}.csv")
    with open(ep_path, 'w', newline='') as ef:
        w = csv.writer(ef); w.writerow(FIELDS)
        for r in buf:
            w.writerow([ctx, ep_id, r['step'], r['t_norm'], r['delta_score'],
                        r['sprint_bits'], r['position'].tolist(), r['possession']])

    with open(DATA_CSV, 'a', newline='') as df:
        w = csv.writer(df)
        if os.path.getsize(DATA_CSV) == 0:
            w.writerow(FIELDS)
        for r in buf:
            w.writerow([ctx, ep_id, r['step'], r['t_norm'], r['delta_score'],
                        r['sprint_bits'], r['position'].tolist(), r['possession']])

    with open(SUMMARY_CSV, 'a', newline='') as sf:
        w = csv.writer(sf)
        if os.path.getsize(SUMMARY_CSV) == 0:
            w.writerow(SUM_FIELDS)
        w.writerow([ep_id, ctx, len(buf), sap, mecha])

    print(f"[{ctx}] ep{ep_id} steps={len(buf)} SAP={sap:.1f}% MECHA={mecha:.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--context', default='early_neutral')
    p.add_argument('--level', default='5_vs_5_earlyneutral')
    p.add_argument('--n_episodes', type=int, default=1)
    args = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    for f in [DATA_CSV, SUMMARY_CSV]:
        if not os.path.exists(f):
            open(f, 'w').close()

    env = make_env(args.level)
    sid = next_id()
    try:
        for i in range(args.n_episodes):
            record_episode(env, args.context, sid + i)
    except KeyboardInterrupt:
        print("Stopped; completed episodes saved.")
    finally:
        env.close()


if __name__ == '__main__':
    main()