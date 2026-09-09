import time
import numpy as np
import torch
from discriminator_loss import DiscriminatorLoss
from discriminator_trainer import DiscriminatorTrainer
from context_balanced_sampler import BalancedContextSampler, balanced_batch, sqrt_scaled_target
from interaction_loss import build_real_groups, real_interaction_hinge, interaction_gate, GATE_THRESHOLD
from discriminator_ablation_v2 import (AblationDiscriminatorV2, held_out_accuracy,
                                       EXPERT_CACHE, RANDOM_CACHE, MAPPO_CACHE,
                                       PHASE_A_STEPS, PHASE_B_STEPS, BATCH_SIZE, SEED)
from calibrate_and_sweep_gamma_int import load_data

def run_variant(name, sprint_reweight, gamma_int, data):
    (train_feat, train_bins, held_feat, held_bins, swap_lw, swap_ll, _) = data
    torch.manual_seed(SEED); np.random.seed(SEED)
    
    model = AblationDiscriminatorV2("concat")
    opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    sampler_e = BalancedContextSampler(train_feat, train_bins, name="expert")
    props = sqrt_scaled_target(sampler_e.cell_counts)
    
    # Initialize trainer with standard arguments so __init__ doesn't crash on an unknown kwarg
    trainer = DiscriminatorTrainer(model, opt, expert_sampler=sampler_e, 
                                   eta=0.3, gamma_swap=6.0,
                                   swap_lw_features=swap_lw, swap_ll_features=swap_ll)
    
    # Inject the new loss function instance directly over the default one
    trainer.loss_fn = DiscriminatorLoss(eta=0.3, gamma_swap=6.0, 
                                        swap_lw_features=swap_lw, swap_ll_features=swap_ll,
                                        sprint_reweight=sprint_reweight)
    
    rng = np.random.default_rng(SEED)
    int_rng = np.random.default_rng(SEED + 1000)
    
    real_groups = build_real_groups(train_feat, train_bins)
    
    print(f"\n{'='*60}\n{name}: Reweight={sprint_reweight}, gamma_int={gamma_int}\n{'='*60}")
    
    for phase, cache_path, n_steps in [("A", RANDOM_CACHE, PHASE_A_STEPS),
                                       ("B", MAPPO_CACHE, PHASE_B_STEPS)]:
        ac = np.load(cache_path)
        sampler_a = BalancedContextSampler(ac["features"], ac["bins"], name=f"agent{phase}")
        
        for step in range(1, n_steps + 1):
            (ef, eb), (af, ab) = balanced_batch(sampler_e, sampler_a, BATCH_SIZE, props, rng, on_empty="skip")
            
            # Standard BCE + R1 + Swap
            res = trainer.step(ef, af, expert_cells=eb, agent_cells=ab)
            
            # Real-group Interaction Hinge
            if gamma_int > 0:
                hinge, did = real_interaction_hinge(model, real_groups, int_rng, batch_size=64)
                if hinge is not None and hinge > 0:
                    opt.zero_grad()
                    (gamma_int * hinge).backward()
                    opt.step()
                    
            if step % 4000 == 0:
                acc = held_out_accuracy(model, held_feat)
                _, gate_did, _, _ = interaction_gate(model, held_feat, held_bins)
                train_did = did.item() if gamma_int > 0 else 0.0
                print(f"  [{phase}{step:>6}] acc={acc:.4f}  train_did={train_did:.3f}  GATE_did={gate_did:.4f}")

    acc = held_out_accuracy(model, held_feat)
    passed, gate_did, sizes, warn = interaction_gate(model, held_feat, held_bins)
    
    return {"name": name, "acc": acc, "gate_did": float(gate_did), "passed": passed}

if __name__ == "__main__":
    data = load_data()
    variants = [
        ("E1 (Reweight Only)", True, 0.0),
        ("E2 (Reweight + Real Hinge)", True, 1.0),
        ("E3 (Real Hinge Only)", False, 1.0)
    ]
    
    results = []
    for name, rw, g_int in variants:
        res = run_variant(name, rw, g_int, data)
        results.append(res)
        
    print("\n" + "=" * 60)
    print(f"FINAL SHOT RESULTS -- GATE THRESHOLD {GATE_THRESHOLD:.4f}")
    print("=" * 60)
    print(f"{'Variant':>28s} {'Acc':>8s} {'GATE DiD':>10s} {'PASS':>6s}")
    for r in results:
        print(f"{r['name']:>28s} {r['acc']:>8.4f} {r['gate_did']:>10.4f} {'YES' if r['passed'] else 'no':>6s}")