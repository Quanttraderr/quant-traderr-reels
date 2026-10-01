"""
BacktestIsland_Static_Pipeline.py
=================================
Hero still for "The Island That Sank": same DATA -> COMPUTE -> VALIDATE, the
finished in-sample island (every rule that beat buy-and-hold 1994 to 2009)
with the best backtest starred, one fixed camera. The readout underneath
carries what happened after 2010.

RUN
    python BacktestIsland_Static_Pipeline.py
    -> BacktestIsland_Static.png  (3240x5760; 10.8x19.2in at 300 DPI)
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from BacktestIsland_Reel_Pipeline import (CONFIG, THEME, BASE_DIR, LAYOUT, load,
                                          log, scene, style_axes, axis_labels,
                                          text_block)

STATIC = {"W": 3240, "H": 5760, "DPI": 300, "ELEV": 24, "AZIM": -48}


def render_static(out):
    d = load()                               # same asserts guard the still
    s = STATIC
    fig = plt.figure(figsize=(s["W"] / s["DPI"], s["H"] / s["DPI"]),
                     dpi=s["DPI"], facecolor=THEME["BG"])
    ax = fig.add_axes(LAYOUT["AXES"], projection="3d", facecolor=THEME["BG"])
    scene(ax, d, CONFIG["PHASE1"], CONFIG["PHASE1"])
    ax.view_init(elev=s["ELEV"], azim=s["AZIM"])
    ax.set_box_aspect((1.15, 1.15, 1.25), zoom=CONFIG["ZOOM"])
    style_axes(ax)
    axis_labels(ax, d)
    text_block(fig, d["figs"], phase=1)
    fig.savefig(out, dpi=s["DPI"], facecolor=THEME["BG"])
    plt.close(fig)
    log(f"static saved: {out}")


if __name__ == "__main__":
    render_static(os.path.join(BASE_DIR, "BacktestIsland_Static.png"))
