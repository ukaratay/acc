#!/usr/bin/env python3
"""Generate all figures for ACC paper.

Run after simulations are complete:
    python scripts/simulations/run_evt_domains.py
    python scripts/simulations/run_information_rent.py
    python scripts/simulations/run_coverage.py

Then run this script:
    python scripts/figures/generate_all.py
"""

from generate_fig_acc_geometry import generate_figure as gen_fig_acc_geometry
from generate_fig_collusion import generate_collusion_figure as gen_fig_collusion
from generate_fig_depth import generate_figure as gen_fig_depth
from generate_fig_depth_boxplots import generate_figure as gen_fig_depth_boxplots
from generate_fig_ebay_overview import generate_figure as gen_fig_ebay_overview
from generate_fig_ebay_segments import generate_figure as gen_fig_ebay_segments
from generate_fig_ebay_stability import generate_figure as gen_fig_ebay_stability
from generate_fig_evt import generate_figure as gen_fig_evt
from generate_tables import main as gen_tables


def main() -> None:
    """Generate all figures and tables."""
    print("=" * 60)
    print(" GENERATING ALL FIGURES")
    print("=" * 60)
    print()

    errors: list[tuple[str, Exception]] = []

    # fig_acc_geometry: ACC geometry 3D visualization
    print("fig_acc_geometry: ACC geometry...")
    try:
        gen_fig_acc_geometry()
    except Exception as e:
        print(f"  Error: {e}")
        errors.append(("fig_acc_geometry", e))

    # fig_depth: Depth curves (analytical)
    print("\nfig_depth: Depth curves...")
    try:
        gen_fig_depth()
    except Exception as e:
        print(f"  Error: {e}")
        errors.append(("fig_depth", e))

    # fig_evt: EVT (requires simulation data)
    print("\nfig_evt: EVT panels...")
    try:
        gen_fig_evt()
    except FileNotFoundError as e:
        print(f"  Skipped: {e}")
    except Exception as e:
        print(f"  Error: {e}")
        errors.append(("fig_evt", e))

    # fig_depth_boxplots: Depth curve boxplots
    print("\nfig_depth_boxplots: Depth boxplots...")
    try:
        gen_fig_depth_boxplots()
    except FileNotFoundError as e:
        print(f"  Skipped: {e}")
    except Exception as e:
        print(f"  Error: {e}")
        errors.append(("fig_depth_boxplots", e))

    # fig_ebay_overview: eBay data overview
    print("\nfig_ebay_overview: eBay overview...")
    try:
        gen_fig_ebay_overview()
    except FileNotFoundError as e:
        print(f"  Skipped: {e}")
    except Exception as e:
        print(f"  Error: {e}")
        errors.append(("fig_ebay_overview", e))

    # fig_ebay_segments: eBay segments
    print("\nfig_ebay_segments: eBay segments...")
    try:
        gen_fig_ebay_segments()
    except FileNotFoundError as e:
        print(f"  Skipped: {e}")
    except Exception as e:
        print(f"  Error: {e}")
        errors.append(("fig_ebay_segments", e))

    # fig_ebay_stability: eBay stability
    print("\nfig_ebay_stability: eBay stability...")
    try:
        gen_fig_ebay_stability()
    except FileNotFoundError as e:
        print(f"  Skipped: {e}")
    except Exception as e:
        print(f"  Error: {e}")
        errors.append(("fig_ebay_stability", e))

    # fig_collusion: collusion detection (Operation Car Wash)
    print("\nfig_collusion: collusion detection...")
    try:
        gen_fig_collusion()
    except FileNotFoundError as e:
        print(f"  Skipped: {e}")
    except Exception as e:
        print(f"  Error: {e}")
        errors.append(("fig_collusion", e))

    # Tables
    print("\nTables...")
    try:
        gen_tables()
    except Exception as e:
        print(f"  Error: {e}")
        errors.append(("tables", e))

    print()
    print("=" * 60)
    if errors:
        print(f" DONE WITH {len(errors)} ERROR(S):")
        for name, err in errors:
            print(f"  - {name}: {err}")
        print("=" * 60)
        raise RuntimeError(f"{len(errors)} figure(s) failed to generate")
    else:
        print(" DONE")
        print("=" * 60)


if __name__ == "__main__":
    main()
