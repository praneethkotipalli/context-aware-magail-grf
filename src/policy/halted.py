import torch, glob
for f in sorted(glob.glob("ablation_MAGAIL-C_seed*.pt") + glob.glob("ablation_MAGAIL-SC_seed*.pt") + glob.glob("ablation_MAGAIL-NC_seed*.pt")):
    ck = torch.load(f, map_location="cpu")
    print(f"{f}: iterations={ck['iterations_completed']}  halted={ck['halted']}")