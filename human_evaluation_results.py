"""
human_evaluation_results.py — MockMaster Human Evaluation Visualisations
=========================================================================
Generates all quantitative and qualitative visualisation PNG files from:
  - formula_test_results.json  (model composite scores, 11 formulas × 60 rows)
  - Human evaluator grades     (hardcoded from evaluator session)

OUTPUT FILES (saved to ./output_visuals/):
  Quantitative only:
    01_formula_accuracy_bar.png
    02_formula_mae_line.png
    03_composite_score_heatmap.png
    04_tier_avg_by_formula.png
    05_confusion_matrix_f5.png
    06_tp_fp_fn_tn_bar.png
    07_roc_style_accuracy_curve.png
    08_score_distribution_violin.png
    09_sbert_vs_llm_scatter.png
    10_discrimination_gap_line.png

  Quantitative + Qualitative combined:
    11_human_vs_composite_scatter.png
    12_human_vs_composite_heatmap.png
    13_residual_by_tc.png
    14_residual_by_tier.png
    15_pair_matrix.png
    16_accuracy_breakdown_table.png
    17_formula_rank_summary.png
    18_human_score_distribution.png
    19_f5_vs_human_bar_per_tc.png
    20_combined_performance_radar.png
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ---------------------------------------------------------------------------
# SETUP
# ---------------------------------------------------------------------------
OUTPUT_DIR = "./output_visuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)

JSON_PATH = "data/formula_test_results.json"

PALETTE = {
    "strong":    "#3B8F3B",
    "weak":      "#E08C1A",
    "irrelevant":"#C0392B",
    "f5":        "#2471A3",
    "human":     "#1A1A2E",
    "bg":        "#F8F9FA",
    "grid":      "#E0E0E0",
}

FORMULA_LABELS = [
    "F1\nS0/L1", "F2\nS.1/L.9", "F3\nS.2/L.8", "F4\nS.3/L.7",
    "F5★\nS.4/L.6", "F6\nS.5/L.5", "F7\nS.6/L.4", "F8\nS.7/L.3",
    "F9\nS.8/L.2", "F10\nS.9/L.1", "F11\nS1/L0"
]

HUMAN_GRADES = {
    ("TC-01","strong"):9, ("TC-01","weak"):4, ("TC-01","irrelevant"):2,
    ("TC-02","strong"):6, ("TC-02","weak"):4, ("TC-02","irrelevant"):3,
    ("TC-03","strong"):7, ("TC-03","weak"):4, ("TC-03","irrelevant"):2,
    ("TC-04","strong"):8, ("TC-04","weak"):4, ("TC-04","irrelevant"):3,
    ("TC-05","strong"):7, ("TC-05","weak"):3, ("TC-05","irrelevant"):2,
    ("TC-06","strong"):5, ("TC-06","weak"):3, ("TC-06","irrelevant"):2,
    ("TC-07","strong"):8, ("TC-07","weak"):3, ("TC-07","irrelevant"):2,
    ("TC-08","strong"):8, ("TC-08","weak"):5, ("TC-08","irrelevant"):3,
    ("TC-09","strong"):7, ("TC-09","weak"):4, ("TC-09","irrelevant"):2,
    ("TC-10","strong"):6, ("TC-10","weak"):3, ("TC-10","irrelevant"):2,
    ("TC-11","strong"):7, ("TC-11","weak"):4, ("TC-11","irrelevant"):4,
    ("TC-12","strong"):6, ("TC-12","weak"):3, ("TC-12","irrelevant"):4,
    ("TC-13","strong"):7, ("TC-13","weak"):3, ("TC-13","irrelevant"):4,
    ("TC-14","strong"):8, ("TC-14","weak"):5, ("TC-14","irrelevant"):3,
    ("TC-15","strong"):6, ("TC-15","weak"):4, ("TC-15","irrelevant"):4,
    ("TC-16","strong"):7, ("TC-16","weak"):4, ("TC-16","irrelevant"):3,
    ("TC-17","strong"):7, ("TC-17","weak"):3, ("TC-17","irrelevant"):2,
    ("TC-18","strong"):7, ("TC-18","weak"):5, ("TC-18","irrelevant"):3,
    ("TC-19","strong"):8, ("TC-19","weak"):4, ("TC-19","irrelevant"):3,
    ("TC-20","strong"):8, ("TC-20","weak"):4, ("TC-20","irrelevant"):3,
}

def fig_setup(figsize=(12, 7)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.grid(color=PALETTE["grid"], linewidth=0.5, zorder=0)
    return fig, ax

def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {name}")

# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------
with open(JSON_PATH) as f:
    raw = json.load(f)

# Build master dataframe
records = []
for formula in raw:
    fi = formula["formula_index"]
    sw = formula["sbert_weight"]
    lw = formula["llm_weight"]
    for r in formula["rows"]:
        key = (r["id"], r["level"])
        human = HUMAN_GRADES[key]
        comp  = r["composite"]
        comp_int = round(comp)
        records.append({
            "formula_idx":  fi,
            "sbert_w":      sw,
            "llm_w":        lw,
            "tc_id":        r["id"],
            "topic":        r["topic"],
            "tier":         r["level"],
            "sbert_score":  r["sbert_score"],
            "llm_score":    r["llm_score"],
            "composite":    comp,
            "composite_int":comp_int,
            "human":        human,
            "exact_match":  int(comp_int == human),
            "error":        abs(comp_int - human),
        })

df = pd.DataFrame(records)
df_f5 = df[df["formula_idx"] == 5].copy()

# Per-formula summary
formula_stats = df.groupby("formula_idx").agg(
    exact=("exact_match","sum"),
    mae=("error","mean"),
    sbert_w=("sbert_w","first"),
    llm_w=("llm_w","first"),
).reset_index()
formula_stats["accuracy"] = formula_stats["exact"] / 60 * 100

# Raw model accuracy
rows0 = raw[0]["rows"]
sbert_exact = sum(1 for r in rows0 if round(r["sbert_score"]) == HUMAN_GRADES[(r["id"],r["level"])])
llm_exact   = sum(1 for r in rows0 if r["llm_score"] == HUMAN_GRADES[(r["id"],r["level"])])
sbert_mae   = np.mean([abs(round(r["sbert_score"]) - HUMAN_GRADES[(r["id"],r["level"])]) for r in rows0])
llm_mae     = np.mean([abs(r["llm_score"] - HUMAN_GRADES[(r["id"],r["level"])]) for r in rows0])

print("Loading complete. Generating visualisations...\n")

# ===========================================================================
# 01 — Formula Accuracy Bar Chart
# ===========================================================================
fig, ax = fig_setup((13, 6))
colors = ["#C0392B" if i not in [4,5] else ("#2471A3" if i==4 else "#1A8F3B") for i in range(11)]
colors[4] = "#1A8F3B"  # F5 green
bars = ax.bar(range(11), formula_stats["accuracy"], color=colors, edgecolor="white", linewidth=0.8, zorder=3)
ax.axhline(formula_stats["accuracy"].max(), color="#1A8F3B", linestyle="--", linewidth=1.2, alpha=0.5, zorder=2)
for i, (bar, val) in enumerate(zip(bars, formula_stats["accuracy"])):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.8, f"{val:.1f}%",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold" if i==4 else "normal")
ax.axhline(sbert_exact/60*100, color="#8E44AD", linestyle=":", linewidth=1.5, label=f"SBERT alone ({sbert_exact/60*100:.1f}%)")
ax.axhline(llm_exact/60*100,   color="#E67E22", linestyle=":", linewidth=1.5, label=f"LLM alone ({llm_exact/60*100:.1f}%)")
ax.set_xticks(range(11)); ax.set_xticklabels(FORMULA_LABELS, fontsize=8)
ax.set_ylabel("Accuracy % (exact integer match vs human)", fontsize=10)
ax.set_title("Formula Accuracy vs Human Evaluator — All 11 Composite Weightings", fontsize=13, fontweight="bold", pad=12)
ax.set_ylim(0, 105)
ax.legend(fontsize=9)
save(fig, "01_formula_accuracy_bar.png")

# ===========================================================================
# 02 — MAE Line Chart
# ===========================================================================
fig, ax = fig_setup((13, 6))
x = formula_stats["formula_idx"].values
mae_vals = formula_stats["mae"].values
ax.plot(x, mae_vals, color="#2471A3", marker="o", linewidth=2, markersize=7, zorder=3)
ax.fill_between(x, mae_vals, alpha=0.12, color="#2471A3")
ax.scatter([5], [formula_stats[formula_stats["formula_idx"]==5]["mae"].values[0]],
           color="#1A8F3B", s=120, zorder=5, label="Formula 5 (default)")
ax.axhline(sbert_mae, color="#8E44AD", linestyle=":", linewidth=1.5, label=f"SBERT alone MAE={sbert_mae:.3f}")
ax.axhline(llm_mae,   color="#E67E22", linestyle=":", linewidth=1.5, label=f"LLM alone MAE={llm_mae:.3f}")
ax.set_xticks(x); ax.set_xticklabels(FORMULA_LABELS, fontsize=8)
ax.set_ylabel("Mean Absolute Error (vs human integer grades)", fontsize=10)
ax.set_title("MAE by Formula — Lower is Better", fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=9)
save(fig, "02_formula_mae_line.png")

# ===========================================================================
# 03 — Composite Score Heatmap (all formulas × all 60 rows)
# ===========================================================================
pivot = df.pivot_table(index=["tc_id","tier"], columns="formula_idx", values="composite")
pivot = pivot.reindex([
    (f"TC-{i:02d}", t)
    for i in range(1,21)
    for t in ["strong","weak","irrelevant"]
])
fig, ax = plt.subplots(figsize=(16, 20), facecolor=PALETTE["bg"])
ax.set_facecolor(PALETTE["bg"])
sns.heatmap(pivot, ax=ax, cmap="RdYlGn", vmin=1, vmax=10,
            linewidths=0.3, linecolor="#CCCCCC",
            cbar_kws={"label":"Composite Score", "shrink":0.5},
            annot=True, fmt=".1f", annot_kws={"size":6.5})
ax.set_title("Composite Score Heatmap — All 60 Responses × 11 Formulas", fontsize=13, fontweight="bold", pad=12)
ax.set_xlabel("Formula Index", fontsize=10)
ax.set_ylabel("")
row_labels = [f"{tc} {t[0].upper()}" for tc, t in pivot.index]
ax.set_yticklabels(row_labels, fontsize=7, rotation=0)
ax.set_xticklabels([f"F{i}" for i in range(1,12)], fontsize=8, rotation=0)
save(fig, "03_composite_score_heatmap.png")

# ===========================================================================
# 04 — Tier Average Score by Formula
# ===========================================================================
tier_avg = df.groupby(["formula_idx","tier"])["composite"].mean().reset_index()
fig, ax = fig_setup((13, 6))
for tier, col in [("strong",PALETTE["strong"]),("weak",PALETTE["weak"]),("irrelevant",PALETTE["irrelevant"])]:
    sub = tier_avg[tier_avg["tier"]==tier]
    ax.plot(sub["formula_idx"], sub["composite"], marker="o", color=col,
            linewidth=2, markersize=6, label=tier.capitalize())
ax.set_xticks(range(1,12)); ax.set_xticklabels(FORMULA_LABELS, fontsize=8)
ax.set_ylabel("Average Composite Score /10", fontsize=10)
ax.set_title("Average Composite Score by Tier Across All Formulas", fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=10)
save(fig, "04_tier_avg_by_formula.png")

# ===========================================================================
# 05 — Confusion Matrix for Formula 5 (pass/fail threshold = 6)
# ===========================================================================
threshold = 6
f5 = df_f5.copy()
f5["pred_pass"] = (f5["composite_int"] >= threshold).astype(int)
f5["human_pass"] = (f5["human"] >= threshold).astype(int)
cm = confusion_matrix(f5["human_pass"], f5["pred_pass"])
fig, ax = plt.subplots(figsize=(7, 6), facecolor=PALETTE["bg"])
ax.set_facecolor(PALETTE["bg"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fail (<6)","Pass (≥6)"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix — Formula 5 vs Human (Pass/Fail threshold = 6)", fontsize=11, fontweight="bold", pad=12)
save(fig, "05_confusion_matrix_f5.png")

# ===========================================================================
# 06 — TP / FP / FN / TN per formula bar chart
# ===========================================================================
tp_list, fp_list, fn_list, tn_list = [], [], [], []
for fi in range(1, 12):
    sub = df[df["formula_idx"]==fi].copy()
    sub["pred_pass"]  = (sub["composite_int"] >= threshold).astype(int)
    sub["human_pass"] = (sub["human"] >= threshold).astype(int)
    cm_ = confusion_matrix(sub["human_pass"], sub["pred_pass"], labels=[0,1])
    tn_, fp_, fn_, tp_ = cm_.ravel() if cm_.size == 4 else (cm_[0,0],0,0,cm_[0,0])
    tn_list.append(tn_); fp_list.append(fp_)
    fn_list.append(fn_); tp_list.append(tp_)

x = np.arange(11)
w = 0.2
fig, ax = fig_setup((14, 6))
ax.bar(x - 1.5*w, tp_list, w, label="TP", color="#1A8F3B", zorder=3)
ax.bar(x - 0.5*w, fp_list, w, label="FP", color="#E08C1A", zorder=3)
ax.bar(x + 0.5*w, fn_list, w, label="FN", color="#C0392B", zorder=3)
ax.bar(x + 1.5*w, tn_list, w, label="TN", color="#2471A3", zorder=3)
ax.set_xticks(x); ax.set_xticklabels(FORMULA_LABELS, fontsize=8)
ax.set_ylabel("Count", fontsize=10)
ax.set_title("TP / FP / FN / TN per Formula (Pass/Fail threshold = 6 vs Human)", fontsize=12, fontweight="bold", pad=12)
ax.legend(fontsize=10)
# Annotations for F5
f5_idx = 4
for j, (val, offset) in enumerate([(tp_list[f5_idx], -1.5*w),
                                     (fp_list[f5_idx], -0.5*w),
                                     (fn_list[f5_idx],  0.5*w),
                                     (tn_list[f5_idx],  1.5*w)]):
    ax.text(f5_idx + offset, val + 0.3, str(val), ha="center", fontsize=8, fontweight="bold")
save(fig, "06_tp_fp_fn_tn_bar.png")

# ===========================================================================
# 07 — Accuracy Curve (ROC-style, accuracy vs SBERT weight)
# ===========================================================================
fig, ax = fig_setup((11, 6))
sw_vals = formula_stats["sbert_w"].values
acc_vals = formula_stats["accuracy"].values
ax.plot(sw_vals, acc_vals, color="#2471A3", marker="o", linewidth=2.5, markersize=8, zorder=3)
ax.fill_between(sw_vals, acc_vals, alpha=0.1, color="#2471A3")
peak_idx = np.argmax(acc_vals)
ax.scatter([sw_vals[peak_idx]], [acc_vals[peak_idx]], color="#1A8F3B", s=150, zorder=5,
           label=f"Peak: F{peak_idx+1} @ SBERT={sw_vals[peak_idx]:.1f} ({acc_vals[peak_idx]:.1f}%)")
ax.axhline(sbert_exact/60*100, color="#8E44AD", linestyle=":", linewidth=1.5, label=f"SBERT alone")
ax.axhline(llm_exact/60*100,   color="#E67E22", linestyle=":", linewidth=1.5, label=f"LLM alone")
ax.set_xlabel("SBERT Weight (0=pure LLM → 1=pure SBERT)", fontsize=10)
ax.set_ylabel("Accuracy %", fontsize=10)
ax.set_title("Accuracy vs SBERT Weight — Composite Formula Curve", fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=9)
save(fig, "07_roc_style_accuracy_curve.png")

# ===========================================================================
# 08 — Score Distribution Violin (F5 composite by tier)
# ===========================================================================
fig, ax = fig_setup((10, 6))
tier_order = ["strong","weak","irrelevant"]
tier_colors = [PALETTE["strong"], PALETTE["weak"], PALETTE["irrelevant"]]
data_by_tier = [df_f5[df_f5["tier"]==t]["composite"].values for t in tier_order]
parts = ax.violinplot(data_by_tier, positions=[1,2,3], showmedians=True, showextrema=True)
for pc, col in zip(parts["bodies"], tier_colors):
    pc.set_facecolor(col); pc.set_alpha(0.7)
parts["cmedians"].set_color("white"); parts["cmedians"].set_linewidth(2)
parts["cmaxes"].set_color("#333"); parts["cmins"].set_color("#333")
parts["cbars"].set_color("#333")
ax.set_xticks([1,2,3]); ax.set_xticklabels(["Strong","Weak","Irrelevant"], fontsize=11)
ax.set_ylabel("Composite Score (Formula 5)", fontsize=10)
ax.set_title("Score Distribution by Tier — Formula 5 (Violin Plot)", fontsize=13, fontweight="bold", pad=12)
save(fig, "08_score_distribution_violin.png")

# ===========================================================================
# 09 — SBERT vs LLM Scatter (coloured by tier, F5)
# ===========================================================================
fig, ax = fig_setup((10, 8))
for tier, col, marker in [("strong",PALETTE["strong"],"o"),
                           ("weak",PALETTE["weak"],"s"),
                           ("irrelevant",PALETTE["irrelevant"],"^")]:
    sub = df_f5[df_f5["tier"]==tier]
    ax.scatter(sub["sbert_score"], sub["llm_score"], c=col, marker=marker,
               s=80, alpha=0.8, edgecolors="white", linewidth=0.5,
               label=tier.capitalize(), zorder=3)
ax.set_xlabel("SBERT Score /10", fontsize=10)
ax.set_ylabel("LLM Score /10", fontsize=10)
ax.set_title("SBERT vs LLM Score — Coloured by Answer Tier", fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=10)
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.plot([0,10],[0,10], "k--", linewidth=0.8, alpha=0.3, label="y=x")
save(fig, "09_sbert_vs_llm_scatter.png")

# ===========================================================================
# 10 — Discrimination Gap Line (Strong vs Irrelevant, Strong vs Weak)
# ===========================================================================
gap_si, gap_sw = [], []
for fi in range(1, 12):
    sub = df[df["formula_idx"]==fi]
    s_avg = sub[sub["tier"]=="strong"]["composite"].mean()
    w_avg = sub[sub["tier"]=="weak"]["composite"].mean()
    i_avg = sub[sub["tier"]=="irrelevant"]["composite"].mean()
    gap_si.append(s_avg - i_avg)
    gap_sw.append(s_avg - w_avg)

fig, ax = fig_setup((13, 6))
fi_vals = list(range(1, 12))
ax.plot(fi_vals, gap_si, marker="o", color="#2471A3", linewidth=2, markersize=7, label="Strong vs Irrelevant")
ax.plot(fi_vals, gap_sw, marker="s", color="#E08C1A", linewidth=2, markersize=7, label="Strong vs Weak")
ax.axhline(4.0, color="#C0392B", linestyle="--", linewidth=1.2, alpha=0.6, label="4.0 minimum target")
ax.scatter([5], [gap_si[4]], color="#2471A3", s=120, zorder=5)
ax.scatter([5], [gap_sw[4]], color="#E08C1A", s=120, zorder=5)
ax.set_xticks(fi_vals); ax.set_xticklabels(FORMULA_LABELS, fontsize=8)
ax.set_ylabel("Discrimination Gap (score points)", fontsize=10)
ax.set_title("Discrimination Gap by Formula", fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=10)
save(fig, "10_discrimination_gap_line.png")

# ===========================================================================
# 11 — Human vs Composite Scatter (F5, all 60 rows)
# ===========================================================================
fig, ax = fig_setup((9, 8))
for tier, col, marker in [("strong",PALETTE["strong"],"o"),
                           ("weak",PALETTE["weak"],"s"),
                           ("irrelevant",PALETTE["irrelevant"],"^")]:
    sub = df_f5[df_f5["tier"]==tier]
    ax.scatter(sub["human"], sub["composite_int"], c=col, marker=marker,
               s=80, alpha=0.8, edgecolors="white", linewidth=0.5,
               label=tier.capitalize(), zorder=3)
ax.plot([0,10],[0,10], "k--", linewidth=1.2, alpha=0.5, label="Perfect agreement")
ax.set_xlabel("Human Grade /10", fontsize=10)
ax.set_ylabel("Formula 5 Composite (rounded) /10", fontsize=10)
ax.set_title("Human Grade vs Formula 5 Composite Score", fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=10)
ax.set_xlim(0.5, 10.5); ax.set_ylim(0.5, 10.5)
save(fig, "11_human_vs_composite_scatter.png")

# ===========================================================================
# 12 — Human vs Composite Heatmap (frequency count)
# ===========================================================================
heatmap_data = np.zeros((11, 11), dtype=int)
for _, row in df_f5.iterrows():
    h = int(row["human"]) - 0
    c = int(row["composite_int"]) - 0
    if 0 <= h <= 10 and 0 <= c <= 10:
        heatmap_data[10 - c][h] += 1

fig, ax = plt.subplots(figsize=(9, 8), facecolor=PALETTE["bg"])
ax.set_facecolor(PALETTE["bg"])
im = ax.imshow(heatmap_data, cmap="YlOrRd", aspect="auto")
plt.colorbar(im, ax=ax, label="Count")
ax.set_xticks(range(11)); ax.set_xticklabels(range(0,11), fontsize=9)
ax.set_yticks(range(11)); ax.set_yticklabels(range(10,-1,-1), fontsize=9)
ax.set_xlabel("Human Grade", fontsize=10)
ax.set_ylabel("Formula 5 Score (rounded)", fontsize=10)
ax.set_title("Agreement Heatmap — Human Grade vs Formula 5", fontsize=13, fontweight="bold", pad=12)
for i in range(11):
    for j in range(11):
        if heatmap_data[i][j] > 0:
            ax.text(j, i, str(heatmap_data[i][j]), ha="center", va="center",
                    fontsize=9, color="black" if heatmap_data[i][j] < 5 else "white")
save(fig, "12_human_vs_composite_heatmap.png")

# ===========================================================================
# 13 — Residual by TC (F5)
# ===========================================================================
tc_residual = df_f5.groupby("tc_id").apply(
    lambda x: (x["composite_int"] - x["human"]).mean()
).reset_index(name="residual")
tc_residual = tc_residual.sort_values("tc_id")
colors_r = ["#1A8F3B" if v >= 0 else "#C0392B" for v in tc_residual["residual"]]
fig, ax = fig_setup((13, 6))
ax.bar(tc_residual["tc_id"], tc_residual["residual"], color=colors_r, edgecolor="white", zorder=3)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticklabels(tc_residual["tc_id"], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Mean Residual (F5 − Human)", fontsize=10)
ax.set_title("Mean Residual per Test Case — Formula 5 vs Human", fontsize=13, fontweight="bold", pad=12)
save(fig, "13_residual_by_tc.png")

# ===========================================================================
# 14 — Residual by Tier (all formulas)
# ===========================================================================
residual_tier = df.groupby(["formula_idx","tier"]).apply(
    lambda x: (x["composite_int"] - x["human"]).mean()
).reset_index(name="residual")
fig, ax = fig_setup((13, 6))
for tier, col in [("strong",PALETTE["strong"]),("weak",PALETTE["weak"]),("irrelevant",PALETTE["irrelevant"])]:
    sub = residual_tier[residual_tier["tier"]==tier]
    ax.plot(sub["formula_idx"], sub["residual"], marker="o", color=col,
            linewidth=2, markersize=6, label=tier.capitalize())
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_xticks(range(1,12)); ax.set_xticklabels(FORMULA_LABELS, fontsize=8)
ax.set_ylabel("Mean Residual (Composite − Human)", fontsize=10)
ax.set_title("Residual by Tier Across All Formulas", fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=10)
save(fig, "14_residual_by_tier.png")

# ===========================================================================
# 15 — Pair Matrix (SBERT, LLM, Composite F5, Human)
# ===========================================================================
pair_df = df_f5[["sbert_score","llm_score","composite","human"]].copy()
pair_df.columns = ["SBERT","LLM","F5 Composite","Human"]
fig = plt.figure(figsize=(12, 10), facecolor=PALETTE["bg"])
tier_col_map = df_f5["tier"].map({"strong":PALETTE["strong"],"weak":PALETTE["weak"],"irrelevant":PALETTE["irrelevant"]})
pd.plotting.scatter_matrix(pair_df, figsize=(12,10), color=tier_col_map.values,
                            alpha=0.7, diagonal="hist", ax=None,
                            hist_kwds={"color":"#2471A3","edgecolor":"white"},
                            marker="o")
fig = plt.gcf()
fig.suptitle("Pair Matrix — SBERT · LLM · F5 Composite · Human Grades", fontsize=13, fontweight="bold", y=1.01)
save(fig, "15_pair_matrix.png")

# ===========================================================================
# 16 — Accuracy Breakdown Table (image)
# ===========================================================================
table_data = []
for fi in range(1, 12):
    sub = df[df["formula_idx"]==fi]
    for tier in ["strong","weak","irrelevant"]:
        t_sub = sub[sub["tier"]==tier]
        exact = t_sub["exact_match"].sum()
        mae   = t_sub["error"].mean()
        table_data.append([f"F{fi}", tier.capitalize(), f"{exact}/20", f"{exact/20*100:.0f}%", f"{mae:.3f}"])

table_data.append(["SBERT", "All", f"{sbert_exact}/60", f"{sbert_exact/60*100:.1f}%", f"{sbert_mae:.3f}"])
table_data.append(["LLM",   "All", f"{llm_exact}/60",   f"{llm_exact/60*100:.1f}%",   f"{llm_mae:.3f}"])

fig, ax = plt.subplots(figsize=(10, 22), facecolor=PALETTE["bg"])
ax.axis("off")
tbl = ax.table(
    cellText=table_data,
    colLabels=["Formula","Tier","Exact /20","Accuracy","MAE"],
    cellLoc="center", loc="center"
)
tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
tbl.scale(1.2, 1.4)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor("#1A1A2E"); cell.set_text_props(color="white", fontweight="bold")
    elif table_data[row-1][0] == "F5" if row > 0 else False:
        cell.set_facecolor("#D5F5E3")
    elif row % 2 == 0:
        cell.set_facecolor("#F2F3F4")
    cell.set_edgecolor("#CCCCCC")
ax.set_title("Accuracy Breakdown by Formula and Tier", fontsize=13, fontweight="bold", pad=20)
save(fig, "16_accuracy_breakdown_table.png")

# ===========================================================================
# 17 — Formula Rank Summary (bubble chart)
# ===========================================================================
fig, ax = fig_setup((12, 7))
scatter = ax.scatter(
    formula_stats["accuracy"],
    formula_stats["mae"],
    s=formula_stats["exact"] * 8,
    c=formula_stats["formula_idx"],
    cmap="RdYlGn_r",
    edgecolors="white", linewidth=1.2, zorder=3, alpha=0.85
)
for _, row in formula_stats.iterrows():
    ax.annotate(f"F{int(row['formula_idx'])}",
                (row["accuracy"], row["mae"]),
                textcoords="offset points", xytext=(6, 4), fontsize=8,
                fontweight="bold" if row["formula_idx"]==5 else "normal")
ax.set_xlabel("Accuracy % (vs human)", fontsize=10)
ax.set_ylabel("MAE (lower = better)", fontsize=10)
ax.set_title("Formula Rank Summary — Accuracy vs MAE (bubble size = exact matches)", fontsize=12, fontweight="bold", pad=12)
plt.colorbar(scatter, ax=ax, label="Formula Index")
save(fig, "17_formula_rank_summary.png")

# ===========================================================================
# 18 — Human Score Distribution
# ===========================================================================
human_vals = list(HUMAN_GRADES.values())
fig, ax = fig_setup((10, 6))
ax.hist(human_vals, bins=range(1, 12), color="#2471A3", edgecolor="white",
        linewidth=0.8, align="left", zorder=3, alpha=0.85)
ax.axvline(np.mean(human_vals), color="#C0392B", linewidth=2, linestyle="--",
           label=f"Mean = {np.mean(human_vals):.2f}")
ax.axvline(np.median(human_vals), color="#1A8F3B", linewidth=2, linestyle="--",
           label=f"Median = {np.median(human_vals):.1f}")
ax.set_xlabel("Human Grade /10", fontsize=10)
ax.set_ylabel("Frequency", fontsize=10)
ax.set_title("Distribution of Human Evaluator Grades (60 responses)", fontsize=13, fontweight="bold", pad=12)
ax.legend(fontsize=10)
save(fig, "18_human_score_distribution.png")

# ===========================================================================
# 19 — F5 vs Human Bar per TC (grouped by tier)
# ===========================================================================
tcs = [f"TC-{i:02d}" for i in range(1, 21)]
x = np.arange(len(tcs))
w = 0.13
tier_list = ["strong","weak","irrelevant"]
fig, ax = fig_setup((18, 7))
for ti, (tier, col) in enumerate([("strong",PALETTE["strong"]),
                                    ("weak",PALETTE["weak"]),
                                    ("irrelevant",PALETTE["irrelevant"])]):
    f5_vals = [df_f5[(df_f5["tc_id"]==tc) & (df_f5["tier"]==tier)]["composite_int"].values[0] for tc in tcs]
    h_vals  = [HUMAN_GRADES[(tc, tier)] for tc in tcs]
    offset  = (ti - 1) * 2 * w
    ax.bar(x + offset - w/2, f5_vals, w, color=col, alpha=0.85, label=f"F5 {tier.capitalize()}", zorder=3)
    ax.bar(x + offset + w/2, h_vals,  w, color=col, alpha=0.4,  hatch="//", zorder=3,
           label=f"Human {tier.capitalize()}")
ax.set_xticks(x); ax.set_xticklabels(tcs, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Score /10", fontsize=10)
ax.set_title("Formula 5 vs Human Grade — Per TC Per Tier (solid=F5, hatched=Human)", fontsize=12, fontweight="bold", pad=12)
ax.legend(fontsize=7.5, ncol=3)
save(fig, "19_f5_vs_human_bar_per_tc.png")

# ===========================================================================
# 20 — Combined Performance Radar
# ===========================================================================
metrics = ["Accuracy %", "1-MAE (×10)", "TP Rate", "TN Rate", "Gap S-I", "Gap S-W"]
fi_selected = [1, 3, 5, 7, 11]
labels_sel  = ["F1 (pure LLM)", "F3", "F5★ (default)", "F7", "F11 (pure SBERT)"]
colors_sel  = ["#C0392B","#E08C1A","#1A8F3B","#2471A3","#8E44AD"]

def radar_values(fi):
    sub  = df[df["formula_idx"]==fi]
    acc  = sub["exact_match"].mean() * 100
    mae  = sub["error"].mean()
    pred_pass  = (sub["composite_int"] >= threshold).astype(int)
    human_pass = (sub["human"] >= threshold).astype(int)
    cm_  = confusion_matrix(human_pass, pred_pass, labels=[0,1])
    tn_, fp_, fn_, tp_ = cm_.ravel() if cm_.size==4 else (cm_[0,0],0,0,cm_[0,0])
    tpr  = tp_ / max(1, tp_ + fn_) * 100
    tnr  = tn_ / max(1, tn_ + fp_) * 100
    s_avg = sub[sub["tier"]=="strong"]["composite"].mean()
    w_avg = sub[sub["tier"]=="weak"]["composite"].mean()
    i_avg = sub[sub["tier"]=="irrelevant"]["composite"].mean()
    gap_si = (s_avg - i_avg) / 10 * 100
    gap_sw = (s_avg - w_avg) / 10 * 100
    return [acc, (1-mae/10)*100, tpr, tnr, gap_si, gap_sw]

N = len(metrics)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig = plt.figure(figsize=(10, 9), facecolor=PALETTE["bg"])
ax = fig.add_subplot(111, polar=True)
ax.set_facecolor(PALETTE["bg"])
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontsize=10)
ax.set_ylim(0, 100)
ax.set_yticks([20,40,60,80,100])
ax.set_yticklabels(["20","40","60","80","100"], fontsize=7, color="gray")
ax.grid(color=PALETTE["grid"], linewidth=0.6)

for fi, label, col in zip(fi_selected, labels_sel, colors_sel):
    vals = radar_values(fi)
    vals += vals[:1]
    ax.plot(angles, vals, color=col, linewidth=2, label=label)
    ax.fill(angles, vals, color=col, alpha=0.08)

ax.set_title("Combined Performance Radar — Selected Formulas", fontsize=13, fontweight="bold", pad=25)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=9)
save(fig, "20_combined_performance_radar.png")

# ===========================================================================
# DONE
# ===========================================================================
print(f"\n✓ All 20 visualisations saved to {OUTPUT_DIR}/")
print("\nFiles generated:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) // 1024
    print(f"  {f}  ({size} KB)")
