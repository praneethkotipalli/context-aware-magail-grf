# src/discriminator/diagnose_position_overshoot.py
import os, glob
import numpy as np
from pathlib import Path

DISS_ROOT = os.path.join(str(Path.home()), "dissertation")
EPISODES_DIR = os.path.join(DISS_ROOT, "context-aware-magail-grf", "data", "demonstrations", "episodes")

def run():
    paths = sorted(glob.glob(os.path.join(EPISODES_DIR, "*.npz")))

    x_violations = []  # (file, step, player, x_value)
    y_violations = []  # (file, step, player, y_value)
    ball_x_violations = []
    ball_y_violations = []

    for p in paths:
        d = dict(np.load(p, allow_pickle=True))
        fname = os.path.basename(p)
        n = len(d['steps_left'])

        for t in range(n):
            for player in range(5):
                x, y = d['left_team'][t, player]
                if abs(x) > 1.0:
                    x_violations.append((fname, t, player, float(x)))
                if abs(y) > 0.42:
                    y_violations.append((fname, t, player, float(y)))

            bx, by, bz = d['ball'][t]
            if abs(bx) > 1.0:
                ball_x_violations.append((fname, t, float(bx)))
            if abs(by) > 0.42:
                ball_y_violations.append((fname, t, float(by)))

    print(f"own player X overshoot (|x|>1.0): {len(x_violations)} events")
    if x_violations:
        by_player = {}
        for fname, t, player, val in x_violations:
            by_player.setdefault(player, []).append(val)
        for player, vals in sorted(by_player.items()):
            print(f"  player {player}: {len(vals)} events, worst={max(vals, key=abs):+.4f}")

    print(f"\nown player Y overshoot (|y|>0.42): {len(y_violations)} events")
    if y_violations:
        by_player = {}
        for fname, t, player, val in y_violations:
            by_player.setdefault(player, []).append(val)
        for player, vals in sorted(by_player.items()):
            print(f"  player {player}: {len(vals)} events, worst={max(vals, key=abs):+.4f}")
        print("\n  sample events (non-GK players, if any):")
        shown = 0
        for fname, t, player, val in y_violations:
            if player != 0 and shown < 10:
                print(f"    {fname}  step={t}  player={player}  y={val:+.4f}")
                shown += 1

    print(f"\nball X overshoot (|x|>1.0): {len(ball_x_violations)} events")
    print(f"ball Y overshoot (|y|>0.42): {len(ball_y_violations)} events")

    all_x = np.array([v for _,_,_,v in x_violations] + [v for _,_,v in ball_x_violations])
    all_y = np.array([v for _,_,_,v in y_violations] + [v for _,_,v in ball_y_violations])
    if len(all_x):
        print(f"\nworst X overshoot overall: {all_x[np.argmax(np.abs(all_x))]:+.4f}")
    if len(all_y):
        print(f"worst Y overshoot overall: {all_y[np.argmax(np.abs(all_y))]:+.4f}")

if __name__ == '__main__':
    run()