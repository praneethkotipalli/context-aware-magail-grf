import glob
import json
import torch
import os
import sys

PROJECT_ROOT = os.path.expanduser("~/dissertation/context-aware-magail-grf")
GRF_MARL_ROOT = os.path.expanduser("~/dissertation/GRF_MARL")
sys.path.insert(0, GRF_MARL_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "grf_baseline"))
sys.path.insert(0, os.path.join(GRF_MARL_ROOT, "light_malib", "model", "gr_football", "enhanced_LightActionMask_5"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "policy"))

import gfootball.env as football_env
from enhanced_LightActionMask_5 import FeatureEncoder
from context_conditioned_policy import ContextConditionedActor
from evaluate_policy import evaluate_policy

def build_env():
    return football_env.create_environment(
        env_name="5_vs_5_d06", representation="raw",
        number_of_left_players_agent_controls=4,
        number_of_right_players_agent_controls=0, render=False)

encoder = FeatureEncoder()
dummy_actor_path = os.path.join(GRF_MARL_ROOT, "light_malib/trained_models/gr_football/5_vs_5/PassingMain_v2/actor.pt")
dummy_actor = torch.load(dummy_actor_path, map_location="cpu")

for pt_file in glob.glob("ablation_MAGAIL-C+KL_*.pt"):
    json_name = pt_file.replace("ablation_", "result_").replace(".pt", ".json")
    if not os.path.exists(json_name):
        continue
    
    with open(json_name, "r") as f:
        res = json.load(f)
    
    if "win_rate" in res:
        print(f"Skipping {json_name} (already patched).")
        continue

    print(f"Evaluating checkpoint for {res.get('condition', pt_file)}...")
    ckpt = torch.load(pt_file, map_location="cpu")
    
    actor = ContextConditionedActor(dummy_actor)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    ev = evaluate_policy(actor, encoder, build_env, n_episodes=50, max_steps=3000)
    
    res.update({
        "win_rate": ev["win_rate"],
        "sap_mean": ev["sap_mean"],
        "mecha_mean": ev["mecha_mean"],
        "csi_sap_proxy": ev.get("csi_sap_proxy", 0.0)
    })
    
    with open(json_name, "w") as f:
        json.dump(res, f, indent=2)

print("All missing metrics successfully evaluated and patched into JSON files.")