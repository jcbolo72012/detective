"""Colours for "Earth seen from space": deep-space navy, ocean blue, aurora green. Used by every chart.

Importing this module also styles matplotlib to match (dark panels, light text), so charts sit on the app's
background instead of showing up as white boxes. No Streamlit here: the analysis code imports it too."""
import matplotlib as mpl
from cycler import cycler

SPACE = "#06142e"    # page background: deep space
PANEL = "#0b2142"    # chart and card background: night-side ocean
INK = "#e6f1ff"      # main text and observed data
MUTED = "#9fb7d6"    # secondary text, anomalies
GRID = "#2b4a74"     # axes, zero lines
OCEAN = "#4fa3ff"    # series A, the straight line, observed raw values
AURORA = "#3ddc97"   # series B, the LSTM, highlights
SUN = "#f5c451"      # "not significant" trend line, backtest windows
CORAL = "#ff7a6b"    # "significant" trend line, fitted lines

mpl.rcParams.update({
    "figure.facecolor": PANEL, "axes.facecolor": PANEL, "savefig.facecolor": PANEL,
    "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK, "axes.edgecolor": GRID,
    "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID,
    "legend.facecolor": PANEL, "legend.edgecolor": GRID, "legend.labelcolor": INK,
    "axes.prop_cycle": cycler(color=[OCEAN, AURORA, SUN, CORAL]),
})
