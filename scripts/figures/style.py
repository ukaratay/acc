"""Figure styling for economics journals (Econometrica/ReStud style).

Design principles:
- Clean white background, minimal gridlines
- Serif fonts matching LaTeX (Palatino via newpxtext/newpxmath)
- Cubehelix palette: grayscale-friendly with monotonic luminance
- Professional, minimalist aesthetic
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# =============================================================================
# Configure matplotlib for academic style
# =============================================================================

# Seaborn paper context: base_context scaled by 0.8, fonts unchanged
# This gives cleaner, thinner lines suitable for journal figures
_PAPER_SCALE = 0.8

mpl.rcParams.update(
    {
        # Font settings (seaborn texts_base_context - unchanged for paper)
        "font.family": "serif",
        "font.serif": ["TeX Gyre Pagella", "Palatino", "Palatino Linotype", "Times"],
        # STIX Two is designed to harmonize with Times/Palatino-like fonts
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.title_fontsize": 9,
        # Figure settings
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        # Axes settings - only left and bottom spines
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Line widths (seaborn base_context * 0.8)
        "axes.linewidth": 1.25 * _PAPER_SCALE,  # 1.0
        "grid.linewidth": 1.0 * _PAPER_SCALE,  # 0.8
        "lines.linewidth": 1.5 * _PAPER_SCALE,  # 1.2
        "lines.markersize": 6 * _PAPER_SCALE,  # 4.8
        "patch.linewidth": 1.0 * _PAPER_SCALE,  # 0.8
        # Tick widths (seaborn base_context * 0.8)
        "xtick.major.width": 1.25 * _PAPER_SCALE,  # 1.0
        "ytick.major.width": 1.25 * _PAPER_SCALE,  # 1.0
        "xtick.minor.width": 1.0 * _PAPER_SCALE,  # 0.8
        "ytick.minor.width": 1.0 * _PAPER_SCALE,  # 0.8
        # Tick sizes (seaborn base_context * 0.8)
        "xtick.major.size": 6 * _PAPER_SCALE,  # 4.8
        "ytick.major.size": 6 * _PAPER_SCALE,  # 4.8
        "xtick.minor.size": 4 * _PAPER_SCALE,  # 3.2
        "ytick.minor.size": 4 * _PAPER_SCALE,  # 3.2
        # Tick direction
        "xtick.direction": "out",
        "ytick.direction": "out",
        # Grid - very subtle if used
        "axes.grid": False,
        "grid.alpha": 0.3,
        "grid.linestyle": ":",
        # Legend - no frame
        "legend.frameon": False,
        "legend.borderpad": 0.4,
    }
)

# =============================================================================
# Color palettes (cubehelix: grayscale-friendly with monotonic luminance)
# =============================================================================

# Generate cubehelix palette with good separation
# Parameters: start=0.5, rot=-0.75 gives a purple→blue→teal progression
# that prints well in grayscale (monotonically increasing luminance)
_CUBEHELIX_PALETTE = sns.cubehelix_palette(
    n_colors=8, start=0.5, rot=-0.75, light=0.85, dark=0.15, as_cmap=False
)


# Convert to hex for consistency
def _rgb_to_hex(rgb: tuple) -> str:
    """Convert RGB tuple (0-1 scale) to hex string."""
    return f"#{int(rgb[0] * 255):02x}{int(rgb[1] * 255):02x}{int(rgb[2] * 255):02x}"


CUBEHELIX = [_rgb_to_hex(c) for c in _CUBEHELIX_PALETTE]

# Primary grayscale palette (for reference lines, text, etc.)
GRAY_BLACK = "#000000"
GRAY_DARK = "#404040"
GRAY_MED = "#808080"
GRAY_LIGHT = "#B0B0B0"

# Named cubehelix colors (ordered by increasing luminance)
COLOR_1 = CUBEHELIX[0]  # Darkest (dark purple)
COLOR_2 = CUBEHELIX[1]  # Dark blue-purple
COLOR_3 = CUBEHELIX[2]  # Blue
COLOR_4 = CUBEHELIX[3]  # Teal-blue
COLOR_5 = CUBEHELIX[4]  # Teal
COLOR_6 = CUBEHELIX[5]  # Light teal
COLOR_7 = CUBEHELIX[6]  # Light
COLOR_8 = CUBEHELIX[7]  # Lightest

# Legacy color names (mapped to cubehelix for backwards compatibility)
COLOR_DARK_BLUE = COLOR_1
COLOR_MUTED_RED = COLOR_2
COLOR_FOREST = COLOR_3
COLOR_PURPLE = COLOR_4
COLOR_ORANGE = COLOR_5

# Distribution colors (cubehelix, grayscale-distinguishable)
# Standard order: uniform, lognormal, exponential (then pareto for extended)
DIST_COLORS: dict[str, str] = {
    "uniform": COLOR_1,
    "lognormal": COLOR_3,
    "exponential": COLOR_5,
    "pareto": COLOR_7,
}

# Distribution markers (for B&W distinction)
DIST_MARKERS: dict[str, str] = {
    "uniform": "o",  # circle
    "lognormal": "s",  # square
    "exponential": "^",  # triangle up
    "pareto": "D",  # diamond
}

# Standard distribution order for iteration
DIST_ORDER = ["uniform", "lognormal", "exponential"]
DIST_ORDER_EXTENDED = ["uniform", "lognormal", "exponential", "pareto"]

# N-value styling (cubehelix + markers)
N_COLORS: dict[int, str] = {
    3: COLOR_1,
    5: COLOR_2,
    10: COLOR_4,
    20: COLOR_6,
    50: COLOR_7,
}

N_MARKERS: dict[int, str] = {
    3: "o",
    5: "s",
    10: "^",
    20: "D",
    50: "v",
}

N_LINESTYLES: dict[int, str] = {
    3: "-",
    5: "--",
    10: "-.",
    20: ":",
    50: (0, (3, 1, 1, 1)),  # dash-dot-dot
}


# Centralized sequential colormap for heatmaps/contours
# Use this instead of creating ad-hoc colormaps in individual scripts
CMAP_SEQ = sns.cubehelix_palette(
    start=0.5, rot=-0.75, light=0.95, dark=0.25, as_cmap=True
)


def get_cubehelix_colors(n: int) -> list[str]:
    """Get n colors from cubehelix palette.

    Args:
        n: Number of colors needed.

    Returns:
        List of hex color strings with monotonically increasing luminance.
    """
    palette = sns.cubehelix_palette(
        n_colors=n, start=0.5, rot=-0.75, light=0.85, dark=0.15
    )
    return [_rgb_to_hex(c) for c in palette]


# =============================================================================
# Paths
# =============================================================================

FIGURES_DIR = Path(__file__).parent.parent.parent / "figures" / "generated"
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "simulations"

# =============================================================================
# Figure sizes (single column ~3.5in, double ~7in for economics journals)
# =============================================================================

_SINGLE_COL_WIDTH = 3.5
_DOUBLE_COL_WIDTH = 7.0

FIGSIZE_SINGLE = (_SINGLE_COL_WIDTH, _SINGLE_COL_WIDTH * 0.75)  # (3.5, 2.625)
FIGSIZE_WIDE = (_DOUBLE_COL_WIDTH, _DOUBLE_COL_WIDTH * 0.5)  # (7.0, 3.5)
FIGSIZE_TALL = (_DOUBLE_COL_WIDTH, _DOUBLE_COL_WIDTH * 0.85)  # (7.0, 5.95) — 2×2 panels

# DPI for publication
DPI = 300


# =============================================================================
# Helper functions
# =============================================================================


def log_transform_ci(
    mean: float | np.ndarray, se: float | np.ndarray, z: float = 1.96
) -> tuple[np.ndarray, np.ndarray]:
    """Compute asymmetric confidence intervals for positive-valued statistics.

    Uses log-transform to ensure lower bound is always positive.
    Equivalent to constructing a normal CI in log-space and exponentiating.

    Args:
        mean: Point estimate(s). Must be positive.
        se: Standard error(s).
        z: Critical value (default 1.96 for 95% CI).

    Returns:
        Tuple of (lower_error, upper_error) for use with matplotlib errorbar.
        These are *distances* from the mean, not absolute bounds.

    Example:
        >>> mean, se = 0.5, 0.1
        >>> lower_err, upper_err = log_transform_ci(mean, se)
        >>> ax.errorbar(x, mean, yerr=[lower_err, upper_err], ...)
    """
    mean = np.asarray(mean)
    se = np.asarray(se)

    # CI bounds in log-space, transformed back
    # lower = mean * exp(-z * se/mean), upper = mean * exp(+z * se/mean)
    cv = se / mean  # coefficient of variation
    lower_bound = mean * np.exp(-z * cv)
    upper_bound = mean * np.exp(+z * cv)

    # Return distances from mean (for errorbar yerr format)
    lower_error = mean - lower_bound
    upper_error = upper_bound - mean

    return lower_error, upper_error


def save_figure(fig: plt.Figure, filename: str) -> Path:
    """Save figure as both PDF (vector) and PNG (raster preview).

    PDF is the primary output for journal submission (vector line art).
    PNG is saved alongside for quick preview and README embedding.

    Args:
        fig: Matplotlib figure to save.
        filename: Output filename (e.g., "fig_example.png").

    Returns:
        Path to saved PDF file.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    base = Path(filename).stem
    _save_kwargs = dict(bbox_inches="tight", facecolor="white", edgecolor="none")
    # Primary: PDF (vector)
    pdf_path = FIGURES_DIR / f"{base}.pdf"
    fig.savefig(pdf_path, dpi=DPI, **_save_kwargs)
    # Secondary: PNG (raster preview)
    png_path = FIGURES_DIR / f"{base}.png"
    fig.savefig(png_path, dpi=DPI, **_save_kwargs)
    plt.close(fig)
    return pdf_path


def setup_ax(
    ax: plt.Axes,
    xlabel: str,
    ylabel: str,
    title: str | None = None,
    grid: bool = False,
) -> None:
    """Configure axis with consistent academic styling.

    Args:
        ax: Matplotlib axis to configure.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        title: Optional title (usually omitted for journal figures).
        grid: Whether to show subtle grid (default False).
    """
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if grid:
        ax.grid(True, alpha=0.3, linewidth=0.5, linestyle=":")


def add_panel_label(ax: plt.Axes, label: str, loc: str = "upper left") -> None:
    """Add panel label (a), (b), etc. for multi-panel figures.

    Args:
        ax: Matplotlib axis.
        label: Panel label like "(a)" or "A".
        loc: Location string.
    """
    if loc == "upper left":
        ax.text(
            -0.1,
            1.05,
            label,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="bottom",
        )
