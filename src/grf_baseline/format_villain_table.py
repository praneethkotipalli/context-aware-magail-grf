"""
src/grf_baseline/format_villain_table.py
Reads the villain numbers summary CSV and prints/saves a clean,
dissertation-ready formatted table (markdown + plain text).
"""
import pandas as pd
import os

PROJECT_ROOT = "/home/u5749464/dissertation/context-aware-magail-grf"
summary_path = os.path.join(PROJECT_ROOT, "results", "baseline", "villain_numbers_summary.csv")

df = pd.read_csv(summary_path, index_col=0)
df = df.sort_values("win_rate", ascending=False)

display_df = df.copy()
for col in ["win_rate", "draw_rate"]:
    display_df[col] = (display_df[col] * 100).round(1).astype(str) + "%"
for col in ["sap_mean", "sap_std", "sbf_mean"]:
    display_df[col] = display_df[col].round(2)
for col in ["mecha_mean", "cv_mean"]:
    display_df[col] = display_df[col].round(4)

print("\n" + "=" * 100)
print("VILLAIN NUMBERS -- FINAL CHARACTERIZATION (500 episodes/policy)")
print("=" * 100)
print(display_df.to_string())
print("=" * 100)

md_path = os.path.join(PROJECT_ROOT, "results", "baseline", "villain_numbers_table.md")
with open(md_path, "w") as f:
    f.write("# Villain Numbers -- Baseline Characterization (500 episodes/policy)\n\n")
    f.write(display_df.to_markdown())
    f.write("\n")

print(f"\nMarkdown table saved: {md_path}")
