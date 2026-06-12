# Auction Competitive Capacity (ACC)

Replication code for *"Auction Competitive Capacity: Characterization and Inference"*
(Durmus Karatay). The paper axiomatizes a measure of competitive pressure in auctions,

> ACC = H(p) · log(1 + ρ/G*),

where H(p) is the Shannon entropy of bid shares, ρ is the top-two bid ratio, and
G* is the normalized Gini dispersion of bids. This repository contains the Python
package implementing the metrics, the Monte Carlo simulations, the empirical data
pipelines, and the figure/table generators behind every number in the paper.

## Layout

- `src/acc_metrics/` — Python package: ACC components, dispersion measures,
  equilibrium bid functions, samplers, bootstrap/cluster-robust inference
- `scripts/data/` — empirical data preparation (eBay, Operation Car Wash)
- `scripts/simulations/` — Monte Carlo simulations (fixed seed 12345)
- `scripts/figures/` — figure and table generation
- `scripts/verify_*.py` — symbolic (sympy) and numerical theorem verification
- `tests/` — unit and property-based tests
- `data/empirical/` — empirical datasets (see Data below)

## Setup

Requires Python ≥ 3.14 and [uv](https://docs.astral.sh/uv/). Some scripts use
syntax introduced in Python 3.14 (PEP 758), so older interpreters will fail to
parse them.

```bash
make sync   # install dependencies
make test   # run the test suite
```

## Usage

```python
from acc_metrics import acc, acc_components

bids = [100, 95, 80, 60, 40]
print(f"ACC: {acc(bids):.3f}")
h, rho, gini, acc_val, evidence = acc_components(bids)
```

## Reproducing the paper's results

All outputs are regenerated from scratch; nothing generated is committed.

```bash
make simulations  # Monte Carlo runs -> data/simulations/*.parquet
make figures      # figures + LaTeX tables -> figures/generated/
make verify       # symbolic and numerical theorem checks
```

Simulations use a fixed seed (12345), so regenerated figures and statistics
reproduce the paper exactly.

## Data

- **eBay auctions** (`data/empirical/ebay/`): bid-level data for Xbox consoles,
  Cartier wristwatches, and Palm Pilot M515 PDAs from the dataset accompanying
  Jank and Shmueli, *Modeling Online Auctions* (Wiley-Blackwell, 2010), with
  contributor credits (Shanshan Wang, Sharad Borle) preserved in the directory
  names. `scripts/data/clean_ebay_data.py` consolidates the raw CSVs into
  `ebay_auctions.parquet` (included).
- **Operation Car Wash** (`data/empirical/carwash/`): 101 Brazilian procurement
  auctions (2002–2013) with confession-based ground-truth labels, from Signor
  et al., "Collusion Detection in Infrastructure Procurement: A Modified Order
  Statistic Method for Uncapped Auctions" (*IEEE Transactions on Engineering
  Management*). The processed `carwash_auctions.csv` is included. The raw IEEE
  supplementary file (`supp2-3049129.xlsx`) is not redistributed here; to rerun
  `scripts/data/load_carwash_data.py` from scratch, download it from the IEEE
  article page (DOI: 10.1109/TEM.2021.3049129) and place it in
  `data/empirical/carwash/`.

## License

MIT. See `LICENSE`.
