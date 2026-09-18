"""
Tomography_Static_Pipeline.py
=============================
Hero still for the same topic: same DATA -> COMPUTE -> VALIDATE, one frame,
fixed curated camera, the whole stack.

WHAT THE VISUAL IS NOT
    The same disclaimer as the reel: ten funds are not the investable universe,
    correlation is linear dependence only and says nothing about tails, the
    windows overlap, and the calm reading before a crash is one observation
    rather than a signal.

RUN
    python Tomography_Static_Pipeline.py
    -> Tomography_Static.png  (3240x5760; 10.8x19.2in at 300 DPI)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from Tomography_Reel_Pipeline import (CONFIG, THEME, CMAP, BASE_DIR, LAYOUT,
                                      fetch, compute, validate,
                                      prepare_geometry, log)

STATIC = {"W": 3240, "H": 5760, "DPI": 300, "ELEV": 13, "AZIM": -51,
          "ZOOM": 1.12}


def render_static(out):
    d = compute(fetch())
    figs = validate(d)                       # same asserts guard the still
    prepare_geometry(d)
    s, L = STATIC, LAYOUT

    fig = plt.figure(figsize=(s["W"] / s["DPI"], s["H"] / s["DPI"]),
                     dpi=s["DPI"], facecolor=THEME["BG"])
    fig.text(0.5, L["TITLE"], "WHEN DIVERSIFICATION DIES", ha="center",
             fontsize=25, fontweight="bold", color=THEME["TEXT"],
             family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             f"{figs['n_assets']} asset classes  |  {figs['n_windows']} rolling "
             f"{figs['window_days']}-day windows",
             ha="center", fontsize=12.5, color=THEME["ORANGE"],
             family=THEME["FONT"])
    fig.text(0.5, L["KEY"], "plate colour = how much the market moved as one",
             ha="center", fontsize=10, color=THEME["TEXT_DIM"],
             family=THEME["FONT"])
    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])

    ax = fig.add_axes(L["AXES"], projection="3d", facecolor=THEME["BG"])

    ax.add_collection3d(Poly3DCollection(d["quads"], facecolors=d["quad_cols"],
                                         linewidths=0, zorder=2))
    ax.add_collection3d(Line3DCollection(d["ring_segs"], colors=d["ring_cols"],
                                         linewidths=1.7, zorder=5))

    ax.set_xlim(*d["lim"][0]); ax.set_ylim(*d["lim"][1]); ax.set_zlim(*d["lim"][2])
    ax.view_init(elev=s["ELEV"], azim=s["AZIM"])
    ax.set_box_aspect(d["box_aspect"], zoom=s["ZOOM"])
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    fig.text(0.5, L["NOTE"], "each plate is one 60-day correlation matrix",
             ha="center", fontsize=10, color=THEME["TEXT_DIM"],
             family=THEME["MONO"], alpha=0.85)
    fig.text(0.5, L["BIG"],
             f"{figs['calm_corr']} CALM, {figs['crisis_corr']} IN A CRASH",
             ha="center", fontsize=21, color=THEME["RED"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             f"lowest reading of all: {figs['min_corr']}, {figs['min_date']}",
             ha="center", fontsize=12, color=THEME["TEXT_DIM"],
             family=THEME["MONO"])

    fig.savefig(out, dpi=s["DPI"], facecolor=THEME["BG"])
    plt.close(fig)
    log(f"static saved: {out}")


if __name__ == "__main__":
    render_static(os.path.join(BASE_DIR, "Tomography_Static.png"))
