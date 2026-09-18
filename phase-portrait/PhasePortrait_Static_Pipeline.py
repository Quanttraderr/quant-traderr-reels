"""
PhasePortrait_Static_Pipeline.py
================================
Hero still for the same topic: same DATA -> COMPUTE -> VALIDATE, one frame,
fixed curated camera, the whole path drawn at once.

WHAT THE VISUAL IS NOT
    The same disclaimer as the reel: not a forecast, three features are a
    choice out of many, a loop is not a tradeable cycle, and the overlapping
    windows make neighbouring states dependent.

RUN
    python PhasePortrait_Static_Pipeline.py
    -> PhasePortrait_Static.png  (3240x5760; 10.8x19.2in at 300 DPI)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from PhasePortrait_Reel_Pipeline import (CONFIG, THEME, CMAP, BASE_DIR, LAYOUT,
                                         fetch, compute, validate,
                                         prepare_geometry, log)

STATIC = {"W": 3240, "H": 5760, "DPI": 300, "ELEV": 27, "AZIM": -33,
          "ZOOM": 1.30}


def render_static(out):
    d = compute(fetch())
    figs = validate(d)                       # same asserts guard the still
    prepare_geometry(d)
    s = STATIC

    fig = plt.figure(figsize=(s["W"] / s["DPI"], s["H"] / s["DPI"]),
                     dpi=s["DPI"], facecolor=THEME["BG"])
    L = LAYOUT
    fig.text(0.5, L["TITLE"], "THE MARKET'S PHASE PORTRAIT", ha="center",
             fontsize=25, fontweight="bold", color=THEME["TEXT"],
             family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             f"SPY  |  {figs['n_sessions']} sessions  |  "
             f"{figs['window_days']}-day momentum, volatility, drawdown",
             ha="center", fontsize=12.5, color=THEME["ORANGE"],
             family=THEME["FONT"])
    fig.text(0.5, L["KEY"], "colour = how fast the state is moving",
             ha="center", fontsize=10, color=THEME["TEXT_DIM"],
             family=THEME["FONT"])
    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])

    ax = fig.add_axes(L["AXES"], projection="3d", facecolor=THEME["BG"])

    z, speed = d["z"], d["speed"]
    seg = np.stack([z[:-1], z[1:]], axis=1)
    w = np.clip(speed[1:] / max(d["speed_hi"], 1e-9), 0, 1)
    cols = CMAP(w)
    cols[:, 3] = 0.55                       # the still has no head to compete with
    ax.add_collection3d(Line3DCollection(seg, colors=cols, linewidths=1.1,
                                         capstyle="round", zorder=3))

    XX, YY, ZZ = d["plane"]
    ax.plot_wireframe(XX, YY, ZZ, color=THEME["TEXT"], alpha=0.20,
                      linewidth=0.5, zorder=1)

    last = z[-1]
    ax.scatter(*last, s=560, color=THEME["TEXT"], alpha=0.12, linewidths=0,
               zorder=4)
    ax.scatter(*last, s=90, color=THEME["TEXT"], alpha=1.0, linewidths=0,
               zorder=5)

    ax.set_xlim(*d["lim"][0]); ax.set_ylim(*d["lim"][1]); ax.set_zlim(*d["lim"][2])
    ax.view_init(elev=s["ELEV"], azim=s["AZIM"])
    ax.set_box_aspect(d["box_aspect"], zoom=s["ZOOM"])
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    fig.text(0.5, L["NOTE"], "the grid is the previous high", ha="center",
             fontsize=10, color=THEME["TEXT_DIM"], family=THEME["MONO"],
             alpha=0.8)

    fig.text(0.5, L["BIG"], f"{figs['pct_underwater']} OF DAYS UNDERWATER",
             ha="center", fontsize=19, color=THEME["RED"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             f"fastest {figs['fastest_date']}, {figs['speed_ratio']} a typical "
             f"day  ·  worst {figs['deepest_dd']}",
             ha="center", fontsize=12, color=THEME["TEXT_DIM"],
             family=THEME["MONO"])

    fig.savefig(out, dpi=s["DPI"], facecolor=THEME["BG"])
    plt.close(fig)
    log(f"static saved: {out}")


if __name__ == "__main__":
    render_static(os.path.join(BASE_DIR, "PhasePortrait_Static.png"))
