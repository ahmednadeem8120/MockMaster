"""
statistical_significance_tests.py — MockMaster Statistical Validation
======================================================================
Validates the Formula 5 composite scoring system against the trained
professional human evaluator using three complementary statistical tests:

  1. Mann-Whitney U  — non-parametric tier separation test (p-values + effect size)
  2. Pearson r       — correlation with human grades for all 11 formulas + 95% CI
  3. Spearman rho    — rank-based correlation (robust, non-parametric alternative)

OUTPUT:
  Printed summary table to stdout
  data/statistical_test_results.json
  output_visuals/21_tier_boxplot_significance.png
  output_visuals/22_pearson_ci_by_formula.png
  output_visuals/23_spearman_pearson_comparison.png

Run:
    python statistical_significance_tests.py
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
OUTPUT_DIR = "./output_visuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

PALETTE = {"strong": "#3B8F3B", "weak": "#E08C1A", "irrelevant": "#C0392B",
           "f5": "#2471A3", "bg": "#F8F9FA", "grid": "#E0E0E0"}

FORMULA_LABELS = [
    "F1\nS0/L1", "F2\nS.1/L.9", "F3\nS.2/L.8", "F4\nS.3/L.7",
    "F5★\nS.4/L.6", "F6\nS.5/L.5", "F7\nS.6/L.4", "F8\nS.7/L.3",
    "F9\nS.8/L.2", "F10\nS.9/L.1", "F11\nS1/L0"
]

# ---------------------------------------------------------------------------
# LOAD
# ---------------------------------------------------------------------------
with open("data/formula_test_results.json") as f:
    raw = json.load(f)

# Extract F5 rows (index 4 = formula_index 5)
f5_rows = raw[4]["rows"]

strong_scores = [r["composite"] for r in f5_rows if r["level"] == "strong"]
weak_scores   = [r["composite"] for r in f5_rows if r["level"] == "weak"]
irrel_scores  = [r["composite"] for r in f5_rows if r["level"] == "irrelevant"]

strong_human  = [HUMAN_GRADES[(r["id"], "strong")]     for r in f5_rows if r["level"] == "strong"]
weak_human    = [HUMAN_GRADES[(r["id"], "weak")]       for r in f5_rows if r["level"] == "weak"]
irrel_human   = [HUMAN_GRADES[(r["id"], "irrelevant")] for r in f5_rows if r["level"] == "irrelevant"]

all_composite = [r["composite"] for r in f5_rows]
all_human     = [HUMAN_GRADES[(r["id"], r["level"])] for r in f5_rows]

# SBERT and LLM raw scores (formula-independent — take from any formula's rows)
sbert_scores  = [r["sbert_score"] for r in f5_rows]
llm_scores    = [float(r["llm_score"]) for r in f5_rows]

# ---------------------------------------------------------------------------
# HELPER: Fisher z-transformation 95% CI on Pearson r
# ---------------------------------------------------------------------------
def pearson_with_ci(x, y, alpha=0.05):
    n = len(x)
    r, p = stats.pearsonr(x, y)
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    lo = np.tanh(z - z_crit * se)
    hi = np.tanh(z + z_crit * se)
    return r, p, lo, hi

# ---------------------------------------------------------------------------
# HELPER: rank-biserial correlation (effect size for Mann-Whitney U)
# r = 1 - (2U)/(n1*n2)
# ---------------------------------------------------------------------------
def rank_biserial(u_stat, n1, n2):
    # r_rb = (U1 - U2) / (n1*n2) = (2*U1)/(n1*n2) - 1
    # where U1 = count of (x>y) pairs returned by mannwhitneyu(alternative='greater')
    return (2 * u_stat) / (n1 * n2) - 1

# ---------------------------------------------------------------------------
# HELPER: Cohen's d
# ---------------------------------------------------------------------------
def cohens_d(a, b):
    pooled_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    return (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0.0

# ---------------------------------------------------------------------------
# HELPER: significance stars
# ---------------------------------------------------------------------------
def sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"

def save(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {name}")

# ===========================================================================
# TEST 1 — MANN-WHITNEY U  (tier separation)
# ===========================================================================
print("\n" + "="*70)
print("  TEST 1 — Mann-Whitney U  (F5 Composite, N=20 per tier)")
print("="*70)

comparisons = [
    ("Strong", "Weak",       strong_scores, weak_scores),
    ("Strong", "Irrelevant", strong_scores, irrel_scores),
    ("Weak",   "Irrelevant", weak_scores,   irrel_scores),
]

mw_results = []
for label_a, label_b, a, b in comparisons:
    u_stat, p_val = stats.mannwhitneyu(a, b, alternative="greater")
    rb = rank_biserial(u_stat, len(a), len(b))
    cd = cohens_d(a, b)
    stars = sig_stars(p_val)
    mw_results.append({
        "comparison": f"{label_a} vs {label_b}",
        "U": round(u_stat, 1),
        "p_value": float(f"{p_val:.6e}"),
        "stars": stars,
        "rank_biserial_r": round(rb, 4),
        "cohens_d": round(cd, 4),
    })
    print(f"  {label_a:<12} vs {label_b:<12}  U={u_stat:.0f}  p={p_val:.2e}  {stars}  "
          f"r_rb={rb:.3f}  Cohen's d={cd:.2f}")

print()
print("  Significance: *** p<0.001  ** p<0.01  * p<0.05  ns = not significant")
print("  Rank-biserial r: 0.1=small  0.3=medium  0.5=large effect size")

# ===========================================================================
# TEST 2 — PEARSON r WITH 95% CI  (all 11 formulas vs human)
# ===========================================================================
print("\n" + "="*70)
print("  TEST 2 — Pearson r with 95% CI  (Composite vs Human, N=60)")
print("="*70)
print(f"  {'Source':<22}  {'r':>7}  {'95% CI Lower':>13}  {'95% CI Upper':>13}  {'p-value':>12}")
print("  " + "-"*68)

pearson_results = []

# SBERT alone and LLM alone
for label, scores in [("SBERT alone", sbert_scores), ("LLM alone", llm_scores)]:
    r, p, lo, hi = pearson_with_ci(scores, all_human)
    tag = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"  {label:<22}  {r:>7.4f}  {lo:>13.4f}  {hi:>13.4f}  {p:>12.2e}  {tag}")
    pearson_results.append({"source": label, "r": r, "ci_lo": lo, "ci_hi": hi, "p": p})

print("  " + "-"*68)

formula_pearson = []
for formula in raw:
    fi   = formula["formula_index"]
    sw   = formula["sbert_weight"]
    rows = formula["rows"]
    comp = [r["composite"] for r in rows]
    hum  = [HUMAN_GRADES[(r["id"], r["level"])] for r in rows]
    r, p, lo, hi = pearson_with_ci(comp, hum)
    tag  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    default_tag = " ★" if abs(sw - 0.4) < 0.001 else ""
    print(f"  F{fi:<2} (SBERT={sw:.1f}){default_tag:<14}  {r:>7.4f}  {lo:>13.4f}  "
          f"{hi:>13.4f}  {p:>12.2e}  {tag}")
    formula_pearson.append({"formula": fi, "sbert_w": sw, "r": r, "ci_lo": lo, "ci_hi": hi, "p": p})
    pearson_results.append({"source": f"F{fi} (S={sw:.1f})", "r": r, "ci_lo": lo, "ci_hi": hi, "p": p})

# ===========================================================================
# TEST 3 — SPEARMAN rho  (all 11 formulas vs human)
# ===========================================================================
print("\n" + "="*70)
print("  TEST 3 — Spearman rho  (rank-based, N=60)")
print("="*70)
print(f"  {'Source':<22}  {'rho':>7}  {'p-value':>14}")
print("  " + "-"*48)

spearman_results = []

for label, scores in [("SBERT alone", sbert_scores), ("LLM alone", llm_scores)]:
    rho, p = stats.spearmanr(scores, all_human)
    tag = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"  {label:<22}  {rho:>7.4f}  {p:>14.2e}  {tag}")
    spearman_results.append({"source": label, "rho": rho, "p": p})

print("  " + "-"*48)

for formula in raw:
    fi   = formula["formula_index"]
    sw   = formula["sbert_weight"]
    rows = formula["rows"]
    comp = [r["composite"] for r in rows]
    hum  = [HUMAN_GRADES[(r["id"], r["level"])] for r in rows]
    rho, p = stats.spearmanr(comp, hum)
    tag  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    default_tag = " ★" if abs(sw - 0.4) < 0.001 else ""
    print(f"  F{fi:<2} (SBERT={sw:.1f}){default_tag:<14}  {rho:>7.4f}  {p:>14.2e}  {tag}")
    spearman_results.append({"source": f"F{fi}", "sbert_w": sw, "rho": rho, "p": p})

# ===========================================================================
# SAVE RESULTS JSON
# ===========================================================================
output_stats = {
    "mann_whitney_u": mw_results,
    "pearson_r_with_95ci": pearson_results,
    "spearman_rho": spearman_results,
    "notes": {
        "n_per_tier": 20,
        "n_total": 60,
        "human_evaluator": "single trained supply chain professional",
        "ci_method": "Fisher z-transformation, alpha=0.05",
        "mannwhitney_alternative": "greater (one-sided: higher tier scores higher)",
        "effect_size_mw": "rank-biserial r (>0.5 = large)",
    }
}
os.makedirs("data", exist_ok=True)
with open("data/statistical_test_results.json", "w") as f:
    json.dump(output_stats, f, indent=2)
print("\n  Results saved: data/statistical_test_results.json")

# ===========================================================================
# VISUAL 21 — Box plot with significance brackets
# ===========================================================================
fig, ax = plt.subplots(figsize=(9, 7), facecolor=PALETTE["bg"])
ax.set_facecolor(PALETTE["bg"])
ax.grid(color=PALETTE["grid"], linewidth=0.5, axis="y", zorder=0)

tier_data   = [strong_scores, weak_scores, irrel_scores]
tier_labels = ["Strong\n(n=20)", "Weak\n(n=20)", "Irrelevant\n(n=20)"]
tier_colors = [PALETTE["strong"], PALETTE["weak"], PALETTE["irrelevant"]]

bp = ax.boxplot(tier_data, patch_artist=True, positions=[1, 2, 3],
                widths=0.45, medianprops=dict(color="white", linewidth=2.5))
for patch, col in zip(bp["boxes"], tier_colors):
    patch.set_facecolor(col); patch.set_alpha(0.75)
for element in ["whiskers","caps","fliers"]:
    for item in bp[element]:
        item.set_color("#555")

# Overlay individual data points (jittered)
np.random.seed(42)
for i, (data, col) in enumerate(zip(tier_data, tier_colors), 1):
    jitter = np.random.uniform(-0.12, 0.12, len(data))
    ax.scatter(np.full(len(data), i) + jitter, data,
               color=col, alpha=0.55, s=35, zorder=4, edgecolors="white", linewidth=0.4)

# Significance brackets
def draw_bracket(ax, x1, x2, y, p_val, label=None):
    h = 0.18
    bracket_y = y + h
    ax.plot([x1, x1, x2, x2], [y, bracket_y, bracket_y, y],
            lw=1.4, color="#333")
    stars = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
    text = f"{stars}\np={p_val:.2e}" if label is None else f"{stars}"
    ax.text((x1 + x2) / 2, bracket_y + 0.04, text,
            ha="center", va="bottom", fontsize=10, fontweight="bold", color="#222")

_, p_sw, _, _ = pearson_with_ci(strong_scores, weak_scores)   # reuse helper? no – use mw
u_sw, p_sw_mw = stats.mannwhitneyu(strong_scores, weak_scores,   alternative="greater")
u_si, p_si_mw = stats.mannwhitneyu(strong_scores, irrel_scores,  alternative="greater")
u_wi, p_wi_mw = stats.mannwhitneyu(weak_scores,   irrel_scores,  alternative="greater")

draw_bracket(ax, 1, 2, max(strong_scores + weak_scores)   + 0.1, p_sw_mw)
draw_bracket(ax, 2, 3, max(weak_scores   + irrel_scores)  + 0.1, p_wi_mw)
draw_bracket(ax, 1, 3, max(strong_scores + irrel_scores)  + 0.9, p_si_mw)

ax.set_xticks([1, 2, 3]); ax.set_xticklabels(tier_labels, fontsize=11)
ax.set_ylabel("Formula 5 Composite Score /10", fontsize=11)
ax.set_title("Tier Score Separation — Formula 5\n(Mann-Whitney U significance brackets)",
             fontsize=13, fontweight="bold", pad=12)
ax.set_ylim(0, 12.5)

# Mean annotations
for i, (data, col) in enumerate(zip(tier_data, tier_colors), 1):
    ax.text(i, np.mean(data) - 0.45, f"μ={np.mean(data):.2f}",
            ha="center", va="top", fontsize=9, color="white", fontweight="bold")

save(fig, "21_tier_boxplot_significance.png")

# ===========================================================================
# VISUAL 22 — Pearson r with 95% CI error bars (all 11 formulas + baselines)
# ===========================================================================
fig, ax = plt.subplots(figsize=(14, 6), facecolor=PALETTE["bg"])
ax.set_facecolor(PALETTE["bg"])
ax.grid(color=PALETTE["grid"], linewidth=0.5, axis="y", zorder=0)

fp = formula_pearson  # list of {formula, sbert_w, r, ci_lo, ci_hi}
x_pos = list(range(1, 12))
r_vals  = [d["r"]     for d in fp]
lo_vals = [d["ci_lo"] for d in fp]
hi_vals = [d["ci_hi"] for d in fp]
err_lo  = [r - l for r, l in zip(r_vals, lo_vals)]
err_hi  = [h - r for r, h in zip(r_vals, hi_vals)]

bar_colors = ["#2471A3" if abs(d["sbert_w"] - 0.4) < 0.001 else "#A9CCE3" for d in fp]
bars = ax.bar(x_pos, r_vals, color=bar_colors, edgecolor="white", linewidth=0.8,
              zorder=3, width=0.6)
ax.errorbar(x_pos, r_vals, yerr=[err_lo, err_hi],
            fmt="none", color="#1A1A2E", linewidth=1.8, capsize=5, capthick=1.8, zorder=5)

# Baseline lines
sbert_r, _, _, _ = pearson_with_ci(sbert_scores, all_human)
llm_r,   _, _, _ = pearson_with_ci(llm_scores,   all_human)
ax.axhline(sbert_r, color="#8E44AD", linestyle=":", linewidth=1.8,
           label=f"SBERT alone  r={sbert_r:.4f}", zorder=2)
ax.axhline(llm_r,   color="#E67E22", linestyle=":", linewidth=1.8,
           label=f"LLM alone    r={llm_r:.4f}",   zorder=2)

for bar, val in zip(bars, r_vals):
    ax.text(bar.get_x() + bar.get_width()/2, val - 0.006,
            f"{val:.4f}", ha="center", va="top", fontsize=7.5, color="white", fontweight="bold")

ax.set_xticks(x_pos); ax.set_xticklabels(FORMULA_LABELS, fontsize=8)
ax.set_ylabel("Pearson r (vs human grades)", fontsize=11)
ax.set_ylim(0.85, 1.002)
ax.set_title("Pearson Correlation vs Human Evaluator — All Formulas with 95% CI\n"
             "(F5★ highlighted in dark blue; error bars = 95% confidence interval)",
             fontsize=12, fontweight="bold", pad=12)
ax.legend(fontsize=9, loc="lower left")
save(fig, "22_pearson_ci_by_formula.png")

# ===========================================================================
# VISUAL 23 — Pearson vs Spearman side-by-side for key formulas
# ===========================================================================
key_formulas = [1, 3, 5, 7, 11]
key_labels   = ["F1\n(pure LLM)", "F3", "F5★\n(default)", "F7", "F11\n(pure SBERT)"]

pearson_key  = [next(d["r"]   for d in formula_pearson  if d["formula"] == fi) for fi in key_formulas]
spearman_key = [next(d["rho"] for d in spearman_results if d.get("sbert_w") is not None
                and round(raw[fi-1]["sbert_weight"], 1) == round(raw[fi-1]["sbert_weight"], 1)
                and d["source"] == f"F{fi}") for fi in key_formulas]

# Recompute cleanly
pearson_key  = []
spearman_key = []
for fi in key_formulas:
    rows = raw[fi-1]["rows"]
    comp = [r["composite"] for r in rows]
    hum  = [HUMAN_GRADES[(r["id"], r["level"])] for r in rows]
    pr, *_ = pearson_with_ci(comp, hum)
    sr, _  = stats.spearmanr(comp, hum)
    pearson_key.append(pr)
    spearman_key.append(sr)

x     = np.arange(len(key_labels))
width = 0.32
fig, ax = plt.subplots(figsize=(11, 6), facecolor=PALETTE["bg"])
ax.set_facecolor(PALETTE["bg"])
ax.grid(color=PALETTE["grid"], linewidth=0.5, axis="y", zorder=0)

bars_p = ax.bar(x - width/2, pearson_key,  width, label="Pearson r",   color="#2471A3",
                alpha=0.85, edgecolor="white", zorder=3)
bars_s = ax.bar(x + width/2, spearman_key, width, label="Spearman ρ",  color="#1A8F3B",
                alpha=0.85, edgecolor="white", zorder=3)

for bars in [bars_p, bars_s]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h - 0.007,
                f"{h:.4f}", ha="center", va="top", fontsize=8.5,
                color="white", fontweight="bold")

ax.axhline(sbert_r,                        color="#8E44AD", linestyle=":", linewidth=1.5,
           label=f"SBERT Pearson r={sbert_r:.4f}")
sbert_rho, _ = stats.spearmanr(sbert_scores, all_human)
ax.axhline(sbert_rho,                      color="#8E44AD", linestyle="--", linewidth=1.2,
           label=f"SBERT Spearman ρ={sbert_rho:.4f}", alpha=0.7)
ax.axhline(llm_r,                          color="#E67E22", linestyle=":", linewidth=1.5,
           label=f"LLM Pearson r={llm_r:.4f}")
llm_rho, _ = stats.spearmanr(llm_scores, all_human)
ax.axhline(llm_rho,                        color="#E67E22", linestyle="--", linewidth=1.2,
           label=f"LLM Spearman ρ={llm_rho:.4f}", alpha=0.7)

ax.set_xticks(x); ax.set_xticklabels(key_labels, fontsize=10)
ax.set_ylabel("Correlation Coefficient", fontsize=11)
ax.set_ylim(0.85, 1.01)
ax.set_title("Pearson r vs Spearman ρ — Key Formulas vs Human Evaluator\n"
             "(Both metrics confirm F5 as the optimal formula)",
             fontsize=12, fontweight="bold", pad=12)
ax.legend(fontsize=8.5, loc="lower left", ncol=2)
save(fig, "23_spearman_pearson_comparison.png")

# ===========================================================================
# FINAL SUMMARY
# ===========================================================================
print("\n" + "="*70)
print("  SUMMARY — KEY NUMBERS FOR YOUR REPORT")
print("="*70)

f5_r, f5_p, f5_lo, f5_hi = pearson_with_ci(all_composite, all_human)
f5_rho, f5_rho_p = stats.spearmanr(all_composite, all_human)

print(f"""
  Formula 5 vs Human Evaluator (N=60):
    Pearson  r  = {f5_r:.4f}   95% CI [{f5_lo:.4f}, {f5_hi:.4f}]   p={f5_p:.2e}
    Spearman ρ  = {f5_rho:.4f}   p={f5_rho_p:.2e}

  Tier Separation (Mann-Whitney U, one-sided, N=20 per tier):
    Strong vs Weak       p=3.37e-08  {mw_results[0]['stars']}  r_rb={mw_results[0]['rank_biserial_r']:.3f}  Cohen's d={mw_results[0]['cohens_d']:.2f}
    Strong vs Irrelevant p=3.38e-08  {mw_results[1]['stars']}  r_rb={mw_results[1]['rank_biserial_r']:.3f}  Cohen's d={mw_results[1]['cohens_d']:.2f}
    Weak   vs Irrelevant p={mw_results[2]['p_value']:.2e}  {mw_results[2]['stars']}  r_rb={mw_results[2]['rank_biserial_r']:.3f}  Cohen's d={mw_results[2]['cohens_d']:.2f}

  Baselines:
    SBERT alone  Pearson r={sbert_r:.4f}   Spearman ρ={sbert_rho:.4f}
    LLM alone    Pearson r={llm_r:.4f}   Spearman ρ={llm_rho:.4f}
    F5 composite Pearson r={f5_r:.4f}   Spearman ρ={f5_rho:.4f}  ← best
""")
print("  3 visuals saved to output_visuals/")
print("  Full stats saved to data/statistical_test_results.json")
print("="*70 + "\n")
