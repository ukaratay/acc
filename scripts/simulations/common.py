"""Common utilities for ACC simulations.

All simulations follow reproducibility conventions:
- Fixed seeds for each experiment
- Results saved to data/simulations/ as parquet
- Use acc_metrics for Numba-optimized batch computations
"""

from pathlib import Path

import numpy as np
from numpy.random import Generator

# =============================================================================
# Constants
# =============================================================================

SEED = 12345
N_TRIALS = 10_000  # Default number of auctions per configuration
N_BOOTSTRAP = 1_000  # Default bootstrap samples

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "simulations"

# Distribution names used by acc_metrics.sample_auctions
DISTRIBUTIONS = ["uniform", "exponential", "lognormal", "pareto"]

EVT_DOMAINS = {
    "uniform": "weibull",
    "exponential": "gumbel",
    "lognormal": "gumbel",
    "pareto": "frechet",
}


# =============================================================================
# Helpers
# =============================================================================


def save_results(df, filename: str) -> Path:
    """Save DataFrame to parquet in data/simulations/."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / filename
    df.write_parquet(path, compression="zstd")
    print(f"Saved results to {path}")
    return path


def get_rng(seed: int = SEED) -> Generator:
    """Get a reproducible random number generator."""
    return np.random.default_rng(seed)
