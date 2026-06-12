#!/usr/bin/env python3
"""Load Operation Car Wash auction data and compute ACC metrics.

This script loads bid-level procurement auction data from the Signor et al. (2023)
Operation Car Wash dataset and computes ACC metrics for collusion detection analysis.

Source: "Collusion Detection in Infrastructure Procurement: A Modified Order
        Statistic Method for Uncapped Auctions" (IEEE Trans. Eng. Management)

Dataset: 101 Brazilian Oil Company procurement auctions (2002-2013)
         with confession-based ground truth labels.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from acc_metrics.core import acc_components

# =============================================================================
# Constants
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "empirical" / "carwash"
INPUT_PATH = DATA_DIR / "supp2-3049129.xlsx"
OUTPUT_PATH = DATA_DIR / "carwash_auctions.csv"


def load_excel_data() -> pd.DataFrame:
    """Load raw auction data from Excel file.

    The Excel has a wide format:
    - Row 1: Header row with column names
    - Row 2: Subheader with bid numbers (1, 2, 3, ...)
    - Rows 3+: Auction data

    Returns:
        DataFrame with raw auction data.
    """
    # Read Excel, skipping the subheader row
    df = pd.read_excel(INPUT_PATH, sheet_name="Auctions and results", header=0)

    # Drop the subheader row (first data row contains column indices)
    df = df.iloc[1:].reset_index(drop=True)

    return df


def parse_auctions(df: pd.DataFrame) -> list[dict]:
    """Parse auction data from wide format DataFrame.

    Args:
        df: Raw DataFrame from Excel.

    Returns:
        List of auction dictionaries with parsed data.
    """
    auctions = []

    for _, row in df.iterrows():
        # Skip rows without valid auction ID
        auction_id = row["Auction ID"]
        if pd.isna(auction_id):
            continue

        try:
            auction_id = int(float(auction_id))
        except ValueError, TypeError:
            continue

        # Parse PTE (Pre-Tender Estimate)
        pte = row["PTE (R$)"]
        if pd.isna(pte) or pte <= 0:
            continue

        # Parse number of bidders
        n_bidders = row["N Bidders"]
        if pd.isna(n_bidders):
            continue
        n_bidders = int(float(n_bidders))

        # Extract bids from columns (Bids (R$), Unnamed: 5, Unnamed: 6, ...)
        # The first bid column is "Bids (R$)", subsequent are "Unnamed: X"
        bids = []
        bid_col_start = df.columns.get_loc("Bids (R$)")

        for col_idx in range(bid_col_start, bid_col_start + n_bidders):
            col_name = df.columns[col_idx]
            bid_val = row[col_name]
            if pd.notna(bid_val) and bid_val > 0:
                bids.append(float(bid_val))

        if len(bids) < 2:
            continue

        # Parse ground truth from 'Reality' column
        reality = row.get("'Reality'", None)
        if pd.isna(reality) or reality == "":
            ground_truth = 0
            ground_truth_label = "Not confirmed"
        elif "Partly" in str(reality):
            ground_truth = -1  # Exclude these
            ground_truth_label = "Partly Collusive"
        elif "Collusion" in str(reality):
            ground_truth = 1
            ground_truth_label = "Collusive"
        else:
            ground_truth = 0
            ground_truth_label = "Not confirmed"

        # Parse observation notes
        obs = row.get("Obs.", None)
        has_confession = pd.notna(obs) and "Confession" in str(obs)

        auctions.append(
            {
                "auction_id": auction_id,
                "auction_code": row["Auction's Code"],
                "pte": pte,
                "n_bidders": n_bidders,
                "bids": bids,
                "winning_bid": min(bids),  # Low bid wins in procurement
                "ground_truth": ground_truth,
                "ground_truth_label": ground_truth_label,
                "has_confession": has_confession,
            }
        )

    return auctions


def compute_metrics(auctions: list[dict]) -> pd.DataFrame:
    """Compute ACC and simple screen metrics for each auction.

    Args:
        auctions: List of auction dictionaries from parse_auctions().

    Returns:
        DataFrame with auction-level metrics.
    """
    results = []

    for auction in auctions:
        bids = np.array(auction["bids"])
        n = len(bids)
        pte = auction["pte"]
        winning_bid = auction["winning_bid"]

        # Compute ACC components on the raw cost vector. With reverse=True
        # only rho flips to lowest/second-lowest; H and G* are computed on
        # the original costs. This is the transform documented in the Car Wash
        # section of the paper.
        h, rho, g_star, acc_val, evidence = acc_components(bids, reverse=True)

        # Compute ACE = ACC / log(n)
        ace = acc_val / np.log(n) if n > 1 else 0.0

        # Robustness transform: reciprocal scores x_i = 1/b_i (low cost ->
        # high score, standard auction orientation). Note PTE/b_i = PTE *
        # (1/b_i) differs from 1/b_i only by a within-auction scale factor,
        # so scale-invariant ACC coincides on the two; PTE - b_i is
        # non-positive whenever a bid reaches the estimate (uncapped format)
        # and is therefore recorded as infeasible rather than computed.
        recip = 1.0 / bids
        h_r, rho_r, g_star_r, acc_recip, _ = acc_components(recip)
        ace_recip = acc_recip / np.log(n) if n > 1 else 0.0
        n_bids_at_or_above_pte = int(np.sum(bids >= pte))

        # Simple screens
        cv = np.std(bids) / np.mean(bids) if np.mean(bids) > 0 else 0.0
        spread = (
            (np.max(bids) - np.min(bids)) / np.min(bids) if np.min(bids) > 0 else 0.0
        )
        win_pte_ratio = winning_bid / pte if pte > 0 else 0.0

        # Relative distance (Imhof screen): (b2 - b1) / std
        sorted_bids = np.sort(bids)
        rel_dist = (
            (sorted_bids[1] - sorted_bids[0]) / np.std(bids)
            if np.std(bids) > 0
            else 0.0
        )

        results.append(
            {
                "auction_id": auction["auction_id"],
                "auction_code": auction["auction_code"],
                "pte": pte,
                "n_bidders": n,
                "winning_bid": winning_bid,
                "win_pte_ratio": win_pte_ratio,
                "ground_truth": auction["ground_truth"],
                "ground_truth_label": auction["ground_truth_label"],
                "has_confession": auction["has_confession"],
                # ACC components
                "H": h,
                "rho": rho,
                "G_star": g_star,
                "ACC": acc_val,
                "ACE": ace,
                "evidence": evidence,
                # Reciprocal-transform robustness columns
                "H_recip": h_r,
                "rho_recip": rho_r,
                "G_star_recip": g_star_r,
                "ACC_recip": acc_recip,
                "ACE_recip": ace_recip,
                "n_bids_at_or_above_pte": n_bids_at_or_above_pte,
                # Simple screens
                "cv": cv,
                "spread": spread,
                "rel_dist": rel_dist,
                # Bid statistics
                "mean_bid": np.mean(bids),
                "std_bid": np.std(bids),
                "min_bid": np.min(bids),
                "max_bid": np.max(bids),
            }
        )

    return pd.DataFrame(results)


def main() -> pd.DataFrame:
    """Load Operation Car Wash data and compute all metrics.

    Returns:
        DataFrame with auction-level metrics.
    """
    print("Loading Operation Car Wash data...")
    print(f"  Input: {INPUT_PATH}")

    # Load and parse
    raw_df = load_excel_data()
    print(f"  Raw rows: {len(raw_df)}")

    auctions = parse_auctions(raw_df)
    print(f"  Parsed auctions: {len(auctions)}")

    # Compute metrics
    df = compute_metrics(auctions)

    # Summary statistics
    n_collusive = (df["ground_truth"] == 1).sum()
    n_honest = (df["ground_truth"] == 0).sum()
    n_partial = (df["ground_truth"] == -1).sum()

    print("\nGround truth breakdown:")
    print(f"  Collusive (confession):   {n_collusive}")
    print(f"  Honest (no collusion):    {n_honest}")
    print(f"  Partly collusive (excluded): {n_partial}")

    # Filter to binary classification sample (exclude partly collusive)
    df_binary = df[df["ground_truth"] >= 0].copy()
    print(f"\nBinary classification sample: {len(df_binary)} auctions")

    # Compare collusive vs honest
    collusive = df_binary[df_binary["ground_truth"] == 1]
    honest = df_binary[df_binary["ground_truth"] == 0]

    print("\n--- ACC Comparison ---")
    acc_coll_mean = collusive["ACC"].mean()
    acc_coll_std = collusive["ACC"].std()
    acc_hon_mean = honest["ACC"].mean()
    acc_hon_std = honest["ACC"].std()
    print(f"  Collusive: ACC mean={acc_coll_mean:.4f}, std={acc_coll_std:.4f}")
    print(f"  Honest:    ACC mean={acc_hon_mean:.4f}, std={acc_hon_std:.4f}")

    print("\n--- ACE Comparison ---")
    ace_coll_mean = collusive["ACE"].mean()
    ace_coll_std = collusive["ACE"].std()
    ace_hon_mean = honest["ACE"].mean()
    ace_hon_std = honest["ACE"].std()
    print(f"  Collusive: ACE mean={ace_coll_mean:.4f}, std={ace_coll_std:.4f}")
    print(f"  Honest:    ACE mean={ace_hon_mean:.4f}, std={ace_hon_std:.4f}")

    print("\n--- Win/PTE Ratio Comparison ---")
    print(f"  Collusive: ratio mean={collusive['win_pte_ratio'].mean():.4f}")
    print(f"  Honest:    ratio mean={honest['win_pte_ratio'].mean():.4f}")

    print("\n--- Component Comparison (Collusive vs Honest) ---")
    for comp in ["H", "rho", "G_star"]:
        print(f"  {comp}: {collusive[comp].mean():.4f} vs {honest[comp].mean():.4f}")

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")

    return df


if __name__ == "__main__":
    df = main()
