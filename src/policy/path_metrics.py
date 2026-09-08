import glob
import json
import torch
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

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

DUMMY_ACTOR_PATH = os.path.join(GRF_MARL_ROOT, "light_malib/trained_models/gr_football/5_vs_5/PassingMain_v2/actor.pt")

def build_env():
    return football_env.create_environment(
        env_name="5_vs_5_d06", representation="raw",
        number_of_left_players_agent_controls=4,
        number_of_right_players_agent_controls=0, render=False)

def process_checkpoint(pt_file):
    """
    Worker function executed in isolated process memory.
    Dummy actor is loaded freshly here to prevent shared-reference mutation.
    """
    json_name = pt_file.replace("ablation_", "result_").replace(".pt", ".json")
    if not os.path.exists(json_name):
        return f"Skipped {json_name} (Not found)"
    
    with open(json_name, "r") as f:
        res = json.load(f)
    
    if "win_rate" in res:
        return f"Skipped {json_name} (Already patched)"

    # Load a fresh copy of the dummy actor specifically for this worker's actor base
    encoder = FeatureEncoder()
    dummy_actor = torch.load(DUMMY_ACTOR_PATH, map_location="cpu")
    
    ckpt = torch.load(pt_file, map_location="cpu")
    
    actor = ContextConditionedActor(dummy_actor)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    # 50-episode evaluation for the specific seed
    ev = evaluate_policy(actor, encoder, build_env, n_episodes=50, max_steps=3000)
    
    res.update({
        "win_rate": ev["win_rate"],
        "sap_mean": ev["sap_mean"],
        "mecha_mean": ev["mecha_mean"],
        "csi_sap_proxy": ev.get("csi_sap_proxy", 0.0)
    })
    
    with open(json_name, "w") as f:
        json.dump(res, f, indent=2)
        
    return f"Successfully patched {json_name} (win_rate: {ev['win_rate']:.3f})"

if __name__ == "__main__":
    pt_files = glob.glob("ablation_*.pt")
    
    # Expand to capture all condition files if needed, e.g.:
    # pt_files = glob.glob("ablation_*.pt")
    
    print(f"Found {len(pt_files)} checkpoints to evaluate. Booting 37 workers...")
    
    # Launch 37 parallel processes utilizing Blackwell's cores
    with ProcessPoolExecutor(max_workers=37) as executor:
        futures = {executor.submit(process_checkpoint, pt): pt for pt in pt_files}
        
        for future in as_completed(futures):
            try:
                print(future.result())
            except Exception as e:
                print(f"Worker crashed on file {futures[future]}: {e}")
                
    print("All parallel evaluation tasks complete.")