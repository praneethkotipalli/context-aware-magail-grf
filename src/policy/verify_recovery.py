import json
import pandas as pd

print("\n--- 1. SEED COUNT & 2. CSI MAGNITUDE (C vs SC) ---")
try:
    s = json.load(open("recovered_summary.json"))
    for cond in ["MAGAIL-C", "MAGAIL-SC"]:
        n_seeds = s[cond].get("n_seeds", "MISSING")
        seeds = s[cond].get("seeds", [])
        e = s[cond].get("eval/csi_sap_proxy", {"mean": float('nan'), "sd": float('nan'), "values": []})
        print(f"{cond} (n={n_seeds}, seeds={seeds}):")
        print(f"  CSI: mean={e['mean']:.4f} sd={e['sd']:.4f}")
        print(f"  Values: {[round(v, 4) for v in e['values']]}\n")
except Exception as e:
    print(f"Error reading recovered_summary.json: {e}")

print("--- 3. DISC SATURATION & 4. ITERATION SANITY ---")
try:
    df = pd.read_csv("recovered_metrics_final.csv")
    df_sub = df[df["condition"].isin(["MAGAIL-C", "MAGAIL-SC"])]

    # 3. Disc Health
    disc_col = "disc_health" if "disc_health" in df.columns else None
    if disc_col:
        print(f"Discriminator Status at Final Eval ({disc_col}):")
        print(df_sub.groupby("condition")[disc_col].value_counts().unstack().fillna(0))
    else:
        print("No discriminator health column found in CSV.")

    # 4. Iteration Sanity
    it_col = "iteration" if "iteration" in df_sub.columns else "_step"
    print(f"\nFinal Logged Iteration per Seed:")
    print(df_sub.groupby(["condition", "seed"])[it_col].max().unstack())

except Exception as e:
    print(f"Error reading recovered_metrics_final.csv: {e}")