#!/usr/bin/env python3
"""Clean and consolidate eBay auction data into a single parquet file.

Processes Xbox, Cartier, and Palm auction data, computing ACC metrics for each auction.
"""

from pathlib import Path

import numpy as np
import polars as pl

import acc_metrics as acc

# eBay's minimum bid increment is price-dependent; $0.01 is the finest
# recorded resolution and the convention used in the paper (Appendix G).
EBAY_TICK = 0.01


def load_xbox_data(data_path: Path) -> pl.DataFrame:
    """Load Xbox auction data from all duration files."""
    xbox_path = data_path / "Xbox game consoles on eBay (thanks to Shanshan Wang)"

    dfs = []
    for csv_file in sorted(xbox_path.glob("*.csv")):
        duration = csv_file.stem.split()[1]  # "3-day", "5-day", "7-day"
        df = pl.read_csv(csv_file)
        df = df.with_columns(
            [
                pl.lit("Xbox").alias("category"),
                pl.lit(duration).alias("duration"),
            ]
        )
        dfs.append(df)

    return pl.concat(dfs)


def load_cartier_data(data_path: Path) -> pl.DataFrame:
    """Load Cartier auction data from all duration files."""
    cartier_path = data_path / "Cartier wristwatches on eBay (thanks to Sharad Borle)"

    dfs = []
    for csv_file in sorted(cartier_path.glob("*.csv")):
        # Skip the WinnersCurse file (different format/purpose)
        if "WinnersCurse" in csv_file.name:
            continue

        name = csv_file.stem.replace("+", " ")
        duration = name.split()[1] if "day" in name else "unknown"

        df = pl.read_csv(csv_file, schema_overrides={"openbid": pl.Float64})
        df = df.with_columns(
            [
                pl.lit("Cartier").alias("category"),
                pl.lit(duration).alias("duration"),
            ]
        )
        dfs.append(df)

    return pl.concat(dfs)


def load_palm_data(data_path: Path) -> pl.DataFrame:
    """Load Palm Pilot auction data from standard format files only.

    The Palm folder has files with different schemas. We use only the
    standard format files (3-day, 5-day, 7-day) which have consistent columns.
    """
    palm_path = data_path / "Palm Pilot M515 PDAs on eBay"

    # Only load files with standard format
    standard_files = [
        "Palm+3-day+auctions.csv",
        "Palm+5-day+auctions.csv",
        "Palm+7-day+auctions.csv",
    ]

    dfs = []
    for filename in standard_files:
        csv_file = palm_path / filename
        if not csv_file.exists():
            continue

        duration = filename.split("+")[1]  # "3-day", "5-day", "7-day"
        df = pl.read_csv(csv_file)
        df = df.with_columns(
            [
                pl.lit("Palm").alias("category"),
                pl.lit(duration).alias("duration"),
            ]
        )
        dfs.append(df)

    combined = pl.concat(dfs)
    return combined.unique(subset=["auctionid", "bid", "bidder"])


def aggregate_to_auctions(bids_df: pl.DataFrame) -> pl.DataFrame:
    """Aggregate bid-level data to auction-level with ACC metrics.

    For each auction, we take the maximum bid per bidder (their final bid),
    then compute ACC on the resulting bid vector.
    """
    # Get max bid per bidder per auction (include duration in grouping)
    max_bids = bids_df.group_by(["auctionid", "bidder", "category", "duration"]).agg(
        [
            pl.col("bid").max().alias("final_bid"),
            pl.col("openbid").first(),
            pl.col("price").first(),
        ]
    )

    # Now aggregate to auction level (include duration)
    auctions = max_bids.group_by(["auctionid", "category", "duration"]).agg(
        [
            pl.col("final_bid").sort(descending=True).alias("bids"),
            pl.col("bidder").n_unique().alias("n_bidders"),
            pl.col("openbid").first().alias("opening_bid"),
            pl.col("price").first().alias("final_price"),
        ]
    )

    # Compute ACC metrics, including the finite-resolution estimand
    # ACC_delta and the saturation flag at the recording tick (Section 8.1).
    acc_values = []
    h_values = []
    rho_values = []
    gstar_values = []
    acc_delta_values = []
    sat_values = []

    for row in auctions.iter_rows(named=True):
        bids = np.array(row["bids"])
        n = len(bids)

        if n >= 2:
            # Compute components
            h = acc.entropy(bids)
            rho = acc.competitive_ratio(bids)
            gstar = acc.normalized_gini(bids)
            acc_val = acc.acc(bids)
            acc_delta_val = acc.acc_delta(bids, delta=EBAY_TICK)
            sat = bool(acc.saturation_flag(bids, delta=EBAY_TICK))

            h_values.append(h)
            rho_values.append(rho)
            gstar_values.append(gstar)
            acc_values.append(acc_val)
            acc_delta_values.append(acc_delta_val)
            sat_values.append(sat)
        else:
            h_values.append(None)
            rho_values.append(None)
            gstar_values.append(None)
            acc_values.append(None)
            acc_delta_values.append(None)
            sat_values.append(None)

    auctions = auctions.with_columns(
        [
            pl.Series("H", h_values),
            pl.Series("rho", rho_values),
            pl.Series("G_star", gstar_values),
            pl.Series("ACC", acc_values),
            pl.Series("ACC_delta", acc_delta_values),
            pl.Series("SAT", sat_values),
        ]
    )

    # Add derived metrics
    auctions = auctions.with_columns(
        [
            (pl.col("n_bidders").log()).alias("log_n"),
            (pl.col("ACC") / pl.col("n_bidders").log()).alias("ACE"),
            (pl.col("ACC_delta") / pl.col("n_bidders").log()).alias("ACE_delta"),
            (pl.col("H") / pl.col("n_bidders").log()).alias("H_norm"),
            (pl.col("final_price") / pl.col("opening_bid")).alias("price_ratio"),
        ]
    )

    return auctions.drop_nulls("ACC")


def main():
    data_path = Path(__file__).parent.parent.parent / "data" / "empirical" / "ebay"
    output_path = data_path / "ebay_auctions.parquet"

    print("Loading eBay auction data...")

    # Load all datasets
    xbox = load_xbox_data(data_path)
    print(f"  Xbox: {xbox.height:,} bids")

    cartier = load_cartier_data(data_path)
    print(f"  Cartier: {cartier.height:,} bids")

    palm = load_palm_data(data_path)
    print(f"  Palm: {palm.height:,} bids")

    # Combine all bids
    all_bids = pl.concat([xbox, cartier, palm])
    print(f"\nTotal bids: {all_bids.height:,}")

    # Aggregate to auction level with ACC
    print("\nAggregating to auction level and computing ACC...")
    auctions = aggregate_to_auctions(all_bids)

    # Summary statistics
    print("\n=== Summary ===")
    print(f"Total auctions: {auctions.height:,}")

    by_category = auctions.group_by("category").agg(
        [
            pl.len().alias("n_auctions"),
            pl.col("n_bidders").mean().alias("mean_n"),
            pl.col("ACC").mean().alias("mean_ACC"),
            pl.col("ACE").mean().alias("mean_ACE"),
        ]
    )
    print("\nBy category:")
    print(by_category)

    by_duration = auctions.group_by("duration").agg(
        [
            pl.len().alias("n_auctions"),
            pl.col("n_bidders").mean().alias("mean_n"),
            pl.col("ACC").mean().alias("mean_ACC"),
            pl.col("ACE").mean().alias("mean_ACE"),
        ]
    )
    print("\nBy duration:")
    print(by_duration)

    # Save to parquet
    auctions.write_parquet(output_path, compression="zstd")
    print(f"\nSaved to {output_path}")

    return auctions


if __name__ == "__main__":
    main()
