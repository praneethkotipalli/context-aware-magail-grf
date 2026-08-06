import sys
sys.path.insert(0, "/home/praneeth/dissertation/GRF_MARL")

import torch

ckpt = torch.load(
    "/home/praneeth/dissertation/GRF_MARL/light_malib/trained_models/gr_football/5_vs_5/PassingMain_v2/actor.pt",
    map_location="cpu"
)
print("Type:", type(ckpt))

if isinstance(ckpt, dict):
    for k, v in ckpt.items():
        if hasattr(v, "shape"):
            print(f"  {k}: {tuple(v.shape)}")
        else:
            print(f"  {k}: {type(v)}")
elif hasattr(ckpt, "state_dict"):
    print("This is a live nn.Module object.")
    print(ckpt)
else:
    print(ckpt)
