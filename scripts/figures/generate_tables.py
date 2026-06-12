#!/usr/bin/env python3
"""Generate LaTeX tables from simulation data.

Generates:
- table_simulated_moments: Simulated moments by distribution with theoretical asymptotes
- table_predictive: Predictive regressions
- table_coverage: Inference coverage and bias

Requires: Run simulation scripts first.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FIGURES_DIR = Path(__file__).parent.parent.parent / "figures" / "generated"
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "simulations"
EMPIRICAL_DIR = Path(__file__).parent.parent.parent / "data" / "empirical"

# EVT domain classification and theoretical asymptotes
# Structure: domain -> list of distributions with their info
DOMAIN_INFO = {
    "Weibull": [
        {
            "name": "uniform",
            "params": "$[0,1]$",
            "rho_limit": "1",
            "gstar_limit": "1/3",
        },
    ],
    "Gumbel": [
        {
            "name": "exponential",
            "params": "$\\lambda{=}1$",
            "rho_limit": "1",
            "gstar_limit": "1/2",
        },
        {
            "name": "lognormal",
            "params": "$(0,1)$",
            "rho_limit": "1",
            # Population Gini for LogNormal(0,1) is 2Φ(1/√2)−1 ≈ 0.520.
            # (Depends on σ in general; we spell out the (0,1) calibration used here.)
            "gstar_limit": r"$2\Phi(1/\sqrt{2})-1 \approx 0.520$",
        },
    ],
    r"Fr\'{e}chet": [
        {
            "name": "pareto",
            "params": "$\\alpha{=}4$",
            "rho_limit": "4/5",  # α/(α+1) = 4/5
            "gstar_limit": "1/7",  # 1/(2α-1) = 1/7 for α=4
        },
    ],
}


def generate_table_6a() -> str:
    """Generate Table 6a: Simulated moments with EVT domain and asymptotes."""
    df = pl.read_parquet(DATA_DIR / "evt_domains.parquet")

    # Filter to key n values (log-spaced to show convergence)
    df = df.filter(pl.col("n_bidders").is_in([10, 100, 1000, 10000]))

    lines = [
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"Domain & Distribution & Parameters & $n$ & $E[\rho]$ & $E[H]$ & $E[G^*]$ & $E[\text{ACC}]$ \\",  # noqa: E501
        r"\midrule",
    ]

    domain_list = list(DOMAIN_INFO.keys())
    for idx, domain in enumerate(domain_list):
        distributions = DOMAIN_INFO[domain]
        first_domain = True
        for dist_info in distributions:
            dist = dist_info["name"]
            dist_data = df.filter(pl.col("distribution") == dist).sort("n_bidders")
            first_dist = True
            for row in dist_data.iter_rows(named=True):
                domain_label = domain if first_domain else ""
                dist_label = dist.capitalize() if first_dist else ""
                params_label = dist_info["params"] if first_dist else ""
                first_domain = False
                first_dist = False
                n = row["n_bidders"]
                rho = row["mean_rho"]
                h = row["mean_entropy"]
                g = row["mean_gini"]
                a = row["mean_acc"]
                lines.append(
                    f"{domain_label} & {dist_label} & {params_label} & {n} & "
                    f"{rho:.3f} & {h:.2f} & {g:.3f} & {a:.2f} \\\\"
                )
            # Add n -> infinity row for this distribution
            lines.append(
                f" & & & $\\infty$ & {dist_info['rho_limit']} & "
                f"$O(\\log n)$ & {dist_info['gstar_limit']} & $O(\\log n)$ \\\\"
            )
            lines.append(r"\addlinespace")

        # Add midrule between domains (but not after the last one)
        if idx < len(domain_list) - 1:
            lines.pop()  # Remove the \addlinespace
            lines.append(r"\midrule")

    # Remove trailing \addlinespace from last domain
    if lines[-1] == r"\addlinespace":
        lines.pop()

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )

    return "\n".join(lines)


def generate_table_mc_full() -> str:
    """Generate the full Monte Carlo table: mean ACC by distribution and n."""
    df = pl.read_parquet(DATA_DIR / "evt_domains.parquet")
    n_values = [3, 10, 100, 1000, 10000]
    labels = {
        "uniform": "Uniform",
        "exponential": "Exponential",
        "lognormal": "Lognormal",
        "pareto": r"Pareto($\alpha=4$)",
    }

    header = " & ".join(["Distribution"] + [f"$n={n}$" for n in n_values])
    lines = [
        r"\begin{tabular}{l" + "c" * len(n_values) + "}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]
    for dist, label in labels.items():
        dist_data = df.filter(pl.col("distribution") == dist)
        cells = []
        for n in n_values:
            row = dist_data.filter(pl.col("n_bidders") == n)
            cells.append(f"{row['mean_acc'][0]:.2f}")
        lines.append(label + " & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def generate_table_8() -> str:
    """Generate Table 8: Predictive regressions with slopes and robust SEs."""
    df = pl.read_parquet(DATA_DIR / "information_rent_summary.parquet")

    lines = [
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r" & & \multicolumn{3}{c}{ACC $\to$ Rent} "
        r"& \multicolumn{3}{c}{ACC $\to$ Revenue} \\",
        r"\cmidrule(lr){3-5} \cmidrule(lr){6-8}",
        r"Distribution & $n$ & $\beta$ & (SE) & $R^2$ & $\beta$ & (SE) & $R^2$ \\",
        r"\midrule",
    ]

    for dist in ["uniform", "exponential", "lognormal", "pareto"]:
        dist_data = df.filter(pl.col("distribution") == dist).sort("n_bidders")
        first = True
        for row in dist_data.iter_rows(named=True):
            dist_label = dist.capitalize() if first else ""
            first = False
            lines.append(
                f"{dist_label} & {row['n_bidders']} & "
                f"{row['slope_rent']:.3f} & ({row['se_slope_rent']:.3f}) & "
                f"{row['r2_acc_rent']:.3f} & "
                f"{row['slope_revenue']:.3f} & ({row['se_slope_revenue']:.3f}) & "
                f"{row['r2_acc_revenue']:.3f} \\\\"
            )
        if dist != "pareto":
            lines.append(r"\addlinespace")

    # Table notes
    note1 = r"\multicolumn{8}{l}{\footnotesize OLS with HC1 robust SEs.} \\"
    note2 = (
        r"\multicolumn{8}{l}{\footnotesize "
        r"Rent: $\text{Rent}_i = \alpha + \beta \text{ACC}_i + \varepsilon_i$.} \\"
    )
    note3 = (
        r"\multicolumn{8}{l}{\footnotesize "
        r"Revenue: $\text{Rev}_i = \alpha + \beta \text{ACC}_i + \gamma v_{(1)} "
        r"+ \varepsilon_i$.} \\"
    )
    lines.extend([r"\bottomrule", note1, note2, note3, r"\end{tabular}"])

    return "\n".join(lines)


def generate_table_9() -> str:
    """Generate Table 9: Inference coverage and bias."""
    df = pl.read_parquet(DATA_DIR / "coverage.parquet")

    lines = [
        r"\begin{tabular}{cccccc}",
        r"\toprule",
        r"$n$ & $T$ & True Mean & Bias & Bias/SE & Coverage \\",
        r"\midrule",
    ]

    for row in df.iter_rows(named=True):
        lines.append(
            f"{row['n_bidders']} & {row['n_auctions']} & "
            f"{row['true_mean']:.3f} & {row['bias']:.4f} & "
            f"{row['bias_se_ratio']:.3f} & {row['coverage_rate'] * 100:.1f}\\% \\\\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )

    return "\n".join(lines)


def generate_table_variance() -> str:
    """Generate variance-estimator comparison table (Appendix E)."""
    df = pl.read_parquet(DATA_DIR / "cluster_variance.parquet")

    labels = {
        "naive": r"Naive (i.i.d.)",
        "cluster_robust": r"Cluster-robust ($t_{M-1}$)",
        "wild_bootstrap": r"Wild cluster bootstrap",
    }

    def cov(design: str, n_clusters: int, estimator: str) -> str:
        row = df.filter(
            (pl.col("design") == design)
            & (pl.col("n_clusters") == n_clusters)
            & (pl.col("estimator") == estimator)
        )
        return f"{row['coverage_rate'][0] * 100:.1f}\\%"

    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Estimator & Independent & Clustered ($M=10$) & Clustered ($M=50$) \\",
        r"\midrule",
    ]
    for key, label in labels.items():
        lines.append(
            f"{label} & {cov('independent', 50, key)} & "
            f"{cov('clustered', 10, key)} & {cov('clustered', 50, key)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])

    return "\n".join(lines)


def _load_carwash_binary() -> pd.DataFrame:
    """Car Wash binary-classification sample (partly-collusive excluded)."""
    df = pd.read_csv(EMPIRICAL_DIR / "carwash" / "carwash_auctions.csv")
    return df[df["ground_truth"] >= 0].copy()


def _adjusted_auc(y: np.ndarray, score: np.ndarray) -> tuple[float, int]:
    """Direction-adjusted AUC and empirical direction.

    Returns (auc, +1) when higher scores predict collusion and
    (1 - auc, -1) when lower scores do.
    """
    raw = float(roc_auc_score(y, score))
    return (raw, 1) if raw >= 0.5 else (1.0 - raw, -1)


def _bootstrap_auc_ci(
    y: np.ndarray,
    score: np.ndarray,
    direction: int,
    n_boot: int = 10_000,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the direction-adjusted AUC."""
    rng = np.random.default_rng(seed)
    s = direction * score
    n = len(y)
    aucs = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        aucs[b] = roc_auc_score(yb, s[idx])
    aucs = aucs[np.isfinite(aucs)]
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d with the df-weighted pooled standard deviation."""
    n1, n2 = len(a), len(b)
    pooled = np.sqrt(
        ((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2)
    )
    return float((a.mean() - b.mean()) / pooled)


def _bootstrap_d_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 42
) -> tuple[float, float]:
    """Percentile bootstrap CI for Cohen's d."""
    rng = np.random.default_rng(seed)
    ds = np.empty(n_boot)
    for i in range(n_boot):
        ab = a[rng.integers(0, len(a), size=len(a))]
        bb = b[rng.integers(0, len(b), size=len(b))]
        ds[i] = _cohens_d(ab, bb)
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


_CW_PREDICTORS = [
    ("win_pte_ratio", r"Win/PTE ratio"),
    ("G_star", r"$G^*$"),
    ("ACE", r"ACE"),
    ("n_bidders", r"$n$"),
    ("H", r"$H$"),
    ("ACC", r"ACC"),
    ("rho", r"$\rho$"),
]


def generate_table_collusion_auc() -> str:
    """Generate Car Wash AUC table (Appendix H) with bootstrap CIs."""
    df = _load_carwash_binary()
    y = df["ground_truth"].to_numpy()

    rows = []
    for col, label in _CW_PREDICTORS:
        score = df[col].to_numpy().astype(float)
        auc, direction = _adjusted_auc(y, score)
        lo, hi = _bootstrap_auc_ci(y, score, direction)
        rows.append((label, direction, auc, lo, hi))

    rows.sort(key=lambda r: -r[2])

    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Predictor & Direction & AUC & 95\% CI \\",
        r"\midrule",
    ]
    for label, direction, auc, lo, hi in rows:
        arrow = r"$\uparrow$" if direction == 1 else r"$\downarrow$"
        lines.append(f"{label} & {arrow} & {auc:.3f} & [{lo:.3f}, {hi:.3f}] \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])

    return "\n".join(lines)


def generate_table_collusion_effects() -> str:
    """Generate Car Wash effect-size table (Appendix H)."""
    df = _load_carwash_binary()
    coll = df[df["ground_truth"] == 1]
    ncc = df[df["ground_truth"] == 0]

    fmt3 = {"G_star", "rho"}
    order = ["ACE", "win_pte_ratio", "n_bidders", "H", "G_star", "rho", "ACC"]
    labels = dict(_CW_PREDICTORS)

    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Component & Collusive & Not-confirmed & Cohen's $d$ & 95\% CI \\",
        r"\midrule",
    ]
    for col in order:
        a = coll[col].to_numpy().astype(float)
        b = ncc[col].to_numpy().astype(float)
        d = _cohens_d(a, b)
        lo, hi = _bootstrap_d_ci(a, b)
        prec = 3 if col in fmt3 else 2
        sign = "+" if d >= 0 else "$-$"
        lines.append(
            f"{labels[col]} & {a.mean():.{prec}f} & {b.mean():.{prec}f} & "
            f"{sign}{abs(d):.2f} & [{lo:.2f}, {hi:.2f}] \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])

    return "\n".join(lines)


def generate_table_carwash_robustness() -> str:
    """Generate the procurement-transform robustness table (Appendix H)."""
    df = _load_carwash_binary()
    y = df["ground_truth"].to_numpy()

    rows = []
    for col, label in (
        ("ACE", r"Original costs (main): $\rho$ reversed; $H, G^*$ on costs"),
        ("ACE_recip", r"Reciprocal scores $x_i = 1/b_i$ (all primitives on $x$)"),
    ):
        score = df[col].to_numpy().astype(float)
        auc, direction = _adjusted_auc(y, score)
        lo, hi = _bootstrap_auc_ci(y, score, direction)
        arrow = r"$\uparrow$" if direction == 1 else r"$\downarrow$"
        rows.append(f"{label} & {arrow} & {auc:.3f} & [{lo:.3f}, {hi:.3f}] \\\\")

    pct_infeasible = 100.0 * float((df["n_bids_at_or_above_pte"] > 0).mean())

    note1 = (
        r"\multicolumn{4}{l}{\footnotesize Cost efficiency "
        r"$x_i = \text{PTE}/b_i$ rescales $1/b_i$ within auction, so "
        r"scale-invariant ACC} \\"
    )
    note2 = (
        r"\multicolumn{4}{l}{\footnotesize coincides with the reciprocal row. "
        r"Benchmark savings $x_i = \text{PTE} - b_i$ is non-positive for} \\"
    )
    note3 = (
        r"\multicolumn{4}{l}{\footnotesize some bid in "
        + f"{pct_infeasible:.0f}"
        + r"\% of auctions (uncapped format) and is infeasible as a "
        + r"positive score.} \\"
    )

    lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"ACE under transform & Direction & AUC & 95\% CI \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        note1,
        note2,
        note3,
        r"\end{tabular}",
    ]

    return "\n".join(lines)


def generate_table_ebay_category() -> str:
    """Generate eBay category summary table (full sample, ACC_delta)."""

    df = pl.read_parquet(EMPIRICAL_DIR / "ebay" / "ebay_auctions.parquet")

    n_sat_total = int(df["SAT"].sum())
    print(f"Retained all {df.height} auctions ({n_sat_total} saturated)")

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Summary Statistics by Product Category}",
        r"\label{tab:ebay-category}",
        r"\begin{tabular}{lccccccccc}",
        r"\toprule",
        r"Category & $N$ & SAT & $\bar{n}$ & Price & $\text{ACC}_\delta$ & (std) & $\text{ACE}_\delta$ & (std) & $H/\log n$ \\",  # noqa: E501
        r"\midrule",
    ]

    def row_stats(sub: pl.DataFrame) -> tuple[float, ...]:
        return (
            sub.height,
            int(sub["SAT"].sum()),
            sub["n_bidders"].mean(),
            sub["final_price"].mean(),
            sub["ACC_delta"].mean(),
            sub["ACC_delta"].std(),
            sub["ACE_delta"].mean(),
            sub["ACE_delta"].std(),
            (sub["H"] / sub["log_n"]).mean(),
        )

    for cat in ["Xbox", "Cartier", "Palm"]:
        n, sat, mean_n, price, m_acc, s_acc, m_ace, s_ace, h_ratio = row_stats(
            df.filter(pl.col("category") == cat)
        )
        lines.append(
            f"{cat} & {n} & {sat} & {mean_n:.1f} & \\${price:.0f} & "
            f"{m_acc:.2f} & ({s_acc:.2f}) & {m_ace:.2f} & "
            f"({s_ace:.2f}) & {h_ratio:.3f} \\\\"
        )

    n, sat, mean_n, price, m_acc, s_acc, m_ace, s_ace, h_ratio = row_stats(df)
    lines.extend(
        [
            r"\midrule",
            f"All & {n} & {sat} & {mean_n:.1f} & \\${price:.0f} & "
            f"{m_acc:.2f} & ({s_acc:.2f}) & {m_ace:.2f} & "
            f"({s_ace:.2f}) & {h_ratio:.3f} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            r"",
            r"\vspace{0.5em}",
            r"\footnotesize",
            r"Notes: SAT = auctions saturated at the \$0.01 tick "
            r"(Definition~\ref{def:acc-delta}), retained rather than dropped; "
            r"$\bar{n}$ = mean number of bidders; "
            r"Price = mean final price in USD; "
            r"$\text{ACC}_\delta$ = finite-resolution Auction Competitive "
            r"Capacity; "
            r"$\text{ACE}_\delta = \text{ACC}_\delta/\log(n)$; "
            r"$H/\log n$ = entropy efficiency.",
            r"\end{table}",
        ]
    )

    return "\n".join(lines)


def generate_stats_macros() -> str:
    """Generate LaTeX macros for in-text statistics.

    Computes key statistics from eBay and carwash datasets and writes
    newcommand macros for use in the paper text.

    Returns:
        LaTeX string with macro definitions.
    """
    lines = [
        r"% Auto-generated statistics macros",
        r"% Generated by scripts/figures/generate_tables.py",
        r"",
    ]

    # =========================================================================
    # eBay Statistics (full sample; ACC_delta for index-level numbers)
    # =========================================================================
    ebay_df = pl.read_parquet(EMPIRICAL_DIR / "ebay" / "ebay_auctions.parquet")

    lines.append(rf"\newcommand{{\ebayN}}{{{ebay_df.height}}}")
    lines.append(rf"\newcommand{{\ebayNSat}}{{{int(ebay_df['SAT'].sum())}}}")
    lines.append(rf"\newcommand{{\ebayNBar}}{{{ebay_df['n_bidders'].mean():.1f}}}")
    lines.append(rf"\newcommand{{\ebayNMin}}{{{ebay_df['n_bidders'].min()}}}")
    lines.append(rf"\newcommand{{\ebayNMax}}{{{ebay_df['n_bidders'].max()}}}")

    for cat in ("Xbox", "Cartier", "Palm"):
        cat_df = ebay_df.filter(pl.col("category") == cat)
        lines.append(
            rf"\newcommand{{\ebayMeanPrice{cat}}}"
            rf"{{{cat_df['final_price'].mean():.0f}}}"
        )
        lines.append(
            rf"\newcommand{{\ebayNBar{cat}}}{{{cat_df['n_bidders'].mean():.1f}}}"
        )

    # Constrained-through-origin slope of G* on n, per category
    for cat in ("Cartier", "Palm"):
        cat_df = ebay_df.filter(pl.col("category") == cat)
        n_arr = cat_df["n_bidders"].to_numpy().astype(float)
        g_arr = cat_df["G_star"].to_numpy()
        slope = float(np.sum(n_arr * g_arr) / np.sum(n_arr * n_arr))
        lines.append(rf"\newcommand{{\ebaySlope{cat}}}{{{slope:.3f}}}")

    # Within-category Spearman(n, ACC_delta): the within/between reconciliation
    for cat in ("Xbox", "Cartier", "Palm"):
        cat_df = ebay_df.filter(pl.col("category") == cat)
        r_within, _ = stats.spearmanr(
            cat_df["n_bidders"].to_numpy(), cat_df["ACC_delta"].to_numpy()
        )
        lines.append(rf"\newcommand{{\ebayWithinSpearman{cat}}}{{{r_within:.2f}}}")

    # Median rho range across categories
    medians = [
        float(ebay_df.filter(pl.col("category") == cat)["rho"].median())
        for cat in ("Xbox", "Cartier", "Palm")
    ]
    lines.append(rf"\newcommand{{\ebayMedianRhoMin}}{{{min(medians):.2f}}}")
    lines.append(rf"\newcommand{{\ebayMedianRhoMax}}{{{max(medians):.2f}}}")

    # Segment means (ACC_delta / ACE_delta)
    def seg(cat: str, dur: str) -> pl.DataFrame:
        return ebay_df.filter((pl.col("category") == cat) & (pl.col("duration") == dur))

    lines.append(
        rf"\newcommand{{\ebayACCPalmThree}}"
        rf"{{{seg('Palm', '3-day')['ACC_delta'].mean():.2f}}}"
    )
    lines.append(
        rf"\newcommand{{\ebayACCCartierSeven}}"
        rf"{{{seg('Cartier', '7-day')['ACC_delta'].mean():.2f}}}"
    )
    lines.append(
        rf"\newcommand{{\ebayACEPalmThree}}"
        rf"{{{seg('Palm', '3-day')['ACE_delta'].mean():.2f}}}"
    )
    lines.append(
        rf"\newcommand{{\ebayACEPalmSeven}}"
        rf"{{{seg('Palm', '7-day')['ACE_delta'].mean():.2f}}}"
    )

    # Price-premium associations (ACC_delta vs log10 premium)
    acc_arr = ebay_df["ACC_delta"].to_numpy()
    price_ratio = ebay_df["price_ratio"].to_numpy()
    valid_mask = np.isfinite(price_ratio) & (price_ratio > 0)

    cats_arr = ebay_df["category"].to_numpy()
    cat_spearmans = []
    for cat in ("Xbox", "Cartier", "Palm"):
        mask = (cats_arr == cat) & valid_mask
        r_cat, _ = stats.spearmanr(acc_arr[mask], np.log(price_ratio[mask]))
        cat_spearmans.append(float(r_cat))
    lines.append(rf"\newcommand{{\ebaySpearmanMin}}{{{max(cat_spearmans):.2f}}}")
    lines.append(rf"\newcommand{{\ebaySpearmanMax}}{{{min(cat_spearmans):.2f}}}")

    # Markup drop: top vs bottom ACC_delta quartile (pooled)
    q1_cut, q3_cut = np.percentile(acc_arr[valid_mask], [25, 75])
    pr_valid = price_ratio[valid_mask]
    acc_v = acc_arr[valid_mask]
    q1_mean = float(pr_valid[acc_v <= q1_cut].mean())
    q4_mean = float(pr_valid[acc_v >= q3_cut].mean())
    lines.append(
        rf"\newcommand{{\ebayMarkupDropPct}}{{{100 * (1 - q4_mean / q1_mean):.0f}}}"
    )

    # Dispersion range (5th-95th percentile of G*, positive support)
    g_arr_all = ebay_df["G_star"].to_numpy()
    g_pos = g_arr_all[g_arr_all > 0]
    g_low, g_high = np.percentile(g_pos, [5, 95])
    lines.append(rf"\newcommand{{\ebayGstarLow}}{{{g_low:.3f}}}")
    lines.append(rf"\newcommand{{\ebayGstarHigh}}{{{g_high:.2f}}}")

    # Entropy efficiency
    mean_h_norm = (ebay_df["H"] / ebay_df["log_n"]).mean()
    lines.append(rf"\newcommand{{\ebayMeanHNorm}}{{{mean_h_norm:.3f}}}")

    lines.append(r"")

    # =========================================================================
    # Carwash Collusion Statistics
    # =========================================================================
    carwash_df = _load_carwash_binary()

    collusive = carwash_df[carwash_df["ground_truth"] == 1]
    ncc = carwash_df[carwash_df["ground_truth"] == 0]

    # Sample sizes
    lines.append(rf"\newcommand{{\collusionN}}{{{len(carwash_df)}}}")
    lines.append(rf"\newcommand{{\collusionNColl}}{{{len(collusive)}}}")
    lines.append(rf"\newcommand{{\collusionNHon}}{{{len(ncc)}}}")

    # Direction-adjusted AUC values (3 decimals, matching Table H.1)
    y_true = carwash_df["ground_truth"].to_numpy()
    for col, name in (
        ("ACE", "aucACE"),
        ("win_pte_ratio", "aucWinPTE"),
        ("G_star", "aucGstar"),
        ("ACC", "aucACC"),
        ("ACE_recip", "aucACERecip"),
    ):
        auc_val, _ = _adjusted_auc(y_true, carwash_df[col].to_numpy().astype(float))
        lines.append(rf"\newcommand{{\{name}}}{{{auc_val:.3f}}}")

    # Cohen's d (df-weighted pooled SD, matching the effects table)
    d_ace = _cohens_d(collusive["ACE"].to_numpy(), ncc["ACE"].to_numpy())
    d_gstar = _cohens_d(collusive["G_star"].to_numpy(), ncc["G_star"].to_numpy())
    lines.append(rf"\newcommand{{\cohensDAce}}{{{d_ace:.2f}}}")
    lines.append(rf"\newcommand{{\cohensDGstar}}{{{d_gstar:.2f}}}")

    # Component means and the rho Welch p-value
    lines.append(rf"\newcommand{{\cwRhoColl}}{{{collusive['rho'].mean():.3f}}}")
    lines.append(rf"\newcommand{{\cwRhoHon}}{{{ncc['rho'].mean():.3f}}}")
    lines.append(rf"\newcommand{{\cwGstarColl}}{{{collusive['G_star'].mean():.3f}}}")
    lines.append(rf"\newcommand{{\cwGstarHon}}{{{ncc['G_star'].mean():.3f}}}")
    rho_p = stats.ttest_ind(collusive["rho"], ncc["rho"], equal_var=False).pvalue
    lines.append(rf"\newcommand{{\cwRhoP}}{{{rho_p:.3f}}}")

    # Mean precision ratio rho/G* by group
    lines.append(
        rf"\newcommand{{\cwRhoGstarColl}}"
        rf"{{{(collusive['rho'] / collusive['G_star']).mean():.1f}}}"
    )
    lines.append(
        rf"\newcommand{{\cwRhoGstarHon}}"
        rf"{{{(ncc['rho'] / ncc['G_star']).mean():.1f}}}"
    )

    # Matched-sample (n in {3,4,5}) and residualized AUCs
    matched = carwash_df[carwash_df["n_bidders"].isin([3, 4, 5])]
    y_m = matched["ground_truth"].to_numpy()
    lines.append(rf"\newcommand{{\cwMatchedNColl}}{{{int((y_m == 1).sum())}}}")
    lines.append(rf"\newcommand{{\cwMatchedNHon}}{{{int((y_m == 0).sum())}}}")
    auc_m_ace, _ = _adjusted_auc(y_m, matched["ACE"].to_numpy().astype(float))
    auc_m_gstar, _ = _adjusted_auc(y_m, matched["G_star"].to_numpy().astype(float))
    lines.append(rf"\newcommand{{\cwMatchedAUCACE}}{{{auc_m_ace:.3f}}}")
    lines.append(rf"\newcommand{{\cwMatchedAUCGstar}}{{{auc_m_gstar:.3f}}}")

    log_n = np.log(carwash_df["n_bidders"].to_numpy().astype(float))
    ace_arr = carwash_df["ACE"].to_numpy().astype(float)
    slope, intercept = np.polyfit(log_n, ace_arr, 1)
    resid = ace_arr - (intercept + slope * log_n)
    auc_resid, _ = _adjusted_auc(y_true, resid)
    lines.append(rf"\newcommand{{\cwResidAUC}}{{{auc_resid:.2f}}}")

    # Share of auctions where PTE - b is infeasible as a positive score
    pct_infeasible = 100.0 * float((carwash_df["n_bids_at_or_above_pte"] > 0).mean())
    lines.append(rf"\newcommand{{\cwPTEInfeasiblePct}}{{{pct_infeasible:.0f}}}")

    # Group mean bidder counts (auction-size confound discussion)
    lines.append(rf"\newcommand{{\cwNBarColl}}{{{collusive['n_bidders'].mean():.2f}}}")
    lines.append(rf"\newcommand{{\cwNBarHon}}{{{ncc['n_bidders'].mean():.2f}}}")

    # ACE true-positive rate at 10% false-positive rate (direction-adjusted)
    ace_scores = carwash_df["ACE"].to_numpy().astype(float)
    _, direction = _adjusted_auc(y_true, ace_scores)
    fpr, tpr, _ = roc_curve(y_true, direction * ace_scores)
    tpr_at_10 = float(np.interp(0.10, fpr, tpr))
    lines.append(rf"\newcommand{{\cwTPRatTenFPR}}{{{100 * tpr_at_10:.1f}}}")

    # Two-sided ACE rule: anomaly distance from the not-confirmed median
    ncc_median = float(np.median(ncc["ACE"].to_numpy()))
    two_sided = np.abs(ace_scores - ncc_median)
    auc_two_sided = float(roc_auc_score(y_true, two_sided))
    lines.append(rf"\newcommand{{\aucACETwoSided}}{{{auc_two_sided:.3f}}}")

    # Leave-one-out CV logistic on the component vector (H, rho, G*)
    X = carwash_df[["H", "rho", "G_star"]].to_numpy().astype(float)
    loo_scores = np.empty(len(y_true))
    for i in range(len(y_true)):
        mask = np.ones(len(y_true), dtype=bool)
        mask[i] = False
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        clf.fit(X[mask], y_true[mask])
        loo_scores[i] = clf.predict_proba(X[i : i + 1])[0, 1]
    auc_components_cv = float(roc_auc_score(y_true, loo_scores))
    lines.append(rf"\newcommand{{\aucComponentsCV}}{{{auc_components_cv:.3f}}}")

    # rho under the pre-registered suppression direction (lower predicts collusion)
    rho_scores = carwash_df["rho"].to_numpy().astype(float)
    auc_rho_suppressed = float(roc_auc_score(y_true, -rho_scores))
    lines.append(rf"\newcommand{{\aucRhoSuppressed}}{{{auc_rho_suppressed:.3f}}}")

    lines.append(r"")

    # =========================================================================
    # Coverage summary (Monte Carlo inference section)
    # =========================================================================
    cov_df = pl.read_parquet(DATA_DIR / "coverage.parquet")
    worst_cov = 100 * float(cov_df["coverage_rate"].to_numpy().min())
    max_bias_se = float(np.abs(cov_df["bias_se_ratio"].to_numpy()).max())
    lines.append(rf"\newcommand{{\covWorstCoverage}}{{{worst_cov:.1f}}}")
    lines.append(rf"\newcommand{{\covMaxBiasSE}}{{{max_bias_se:.3f}}}")

    lines.append(r"")

    return "\n".join(lines)


def main() -> None:
    """Generate all tables and save to figures/generated/."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # table_simulated_moments
    try:
        table = generate_table_6a()
        path = FIGURES_DIR / "table_simulated_moments.tex"
        path.write_text(table + "\n")
        print(f"Generated {path}")
    except FileNotFoundError as e:
        print(f"Skipping table_simulated_moments: {e}")

    # table_mc_full
    try:
        table = generate_table_mc_full()
        path = FIGURES_DIR / "table_mc_full.tex"
        path.write_text(table + "\n")
        print(f"Generated {path}")
    except FileNotFoundError as e:
        print(f"Skipping table_mc_full: {e}")

    # table_predictive
    try:
        table = generate_table_8()
        path = FIGURES_DIR / "table_predictive.tex"
        path.write_text(table + "\n")
        print(f"Generated {path}")
    except FileNotFoundError as e:
        print(f"Skipping table_predictive: {e}")

    # table_coverage
    try:
        table = generate_table_9()
        path = FIGURES_DIR / "table_coverage.tex"
        path.write_text(table + "\n")
        print(f"Generated {path}")
    except FileNotFoundError as e:
        print(f"Skipping table_coverage: {e}")

    # table_variance
    try:
        table = generate_table_variance()
        path = FIGURES_DIR / "table_variance.tex"
        path.write_text(table + "\n")
        print(f"Generated {path}")
    except FileNotFoundError as e:
        print(f"Skipping table_variance: {e}")

    # table_ebay_category
    try:
        table = generate_table_ebay_category()
        path = FIGURES_DIR / "table_ebay_category.tex"
        path.write_text(table + "\n")
        print(f"Generated {path}")
    except FileNotFoundError as e:
        print(f"Skipping table_ebay_category: {e}")

    # Car Wash tables (Appendix H)
    for name, generator in (
        ("table_collusion_auc", generate_table_collusion_auc),
        ("table_collusion_effects", generate_table_collusion_effects),
        ("table_carwash_robustness", generate_table_carwash_robustness),
    ):
        try:
            table = generator()
            path = FIGURES_DIR / f"{name}.tex"
            path.write_text(table + "\n")
            print(f"Generated {path}")
        except FileNotFoundError as e:
            print(f"Skipping {name}: {e}")

    # generated_stats (LaTeX macros)
    try:
        macros = generate_stats_macros()
        path = FIGURES_DIR / "generated_stats.tex"
        path.write_text(macros.rstrip("\n") + "\n")
        print(f"Generated {path}")
    except FileNotFoundError as e:
        print(f"Skipping generated_stats: {e}")


if __name__ == "__main__":
    main()
