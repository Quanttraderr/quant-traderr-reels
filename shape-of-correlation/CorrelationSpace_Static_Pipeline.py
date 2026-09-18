"""
CorrelationSpace_Static_Pipeline.py
===================================
Hero still for the same topic: same DATA -> COMPUTE -> VALIDATE, one frame,
fixed curated camera, the finished tree. Shares nothing with the reel at runtime
beyond the maths module, so the still stands alone if the reel is re-cut.

WHAT THE VISUAL IS NOT
    The same disclaimer as the reel: not causation, not a forecast, not a sector
    map, and the three dimensions hold only part of the distance structure.

RUN
    python CorrelationSpace_Static_Pipeline.py
    -> CorrelationSpace_Static.png  (3240x5760; 10.8x19.2in at 300 DPI)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from CorrelationSpace_Reel_Pipeline import (CONFIG, THEME, CMAP, BASE_DIR, LAYOUT,
                                            fetch, compute, validate,
                                            prepare_geometry, log)

STATIC = {"W": 3240, "H": 5760, "DPI": 300, "ELEV": 20, "AZIM": -58,
          "ZOOM": 1.38}


def render_static(out):
    d = compute(fetch())
    figs = validate(d)                       # same asserts guard the still
    prepare_geometry(d)
    s = STATIC

    fig = plt.figure(figsize=(s["W"] / s["DPI"], s["H"] / s["DPI"]),
                     dpi=s["DPI"], facecolor=THEME["BG"])
    L = LAYOUT
    fig.text(0.5, L["TITLE"], "THE SHAPE OF CORRELATION", ha="center", fontsize=27,
             fontweight="bold", color=THEME["TEXT"], family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             f"{figs['n_names']} US large caps  |  {figs['n_sessions']} sessions"
             f"  |  {figs['n_links']} tree links",
             ha="center", fontsize=13, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.5, L["KEY"],
             "distance = sqrt(2(1 - rho)), colour = how much a name moves with the rest",
             ha="center", fontsize=9.5, color=THEME["TEXT_DIM"],
             family=THEME["FONT"])
    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])

    ax = fig.add_axes(L["AXES"], projection="3d", facecolor=THEME["BG"])

    pos, edges, avg = d["pos"], d["edges"], d["avg_corr"]
    lo, hi = float(avg.min()), float(avg.max())
    norm = (avg - lo) / max(hi - lo, 1e-9)
    cols = CMAP(norm)

    e_lo, e_hi = d["edge_range"]
    for a, b, rho in edges:                  # the still shows the whole tree
        w = (rho - e_lo) / max(e_hi - e_lo, 1e-9)
        ax.plot([pos[a, 0], pos[b, 0]], [pos[a, 1], pos[b, 1]],
                [pos[a, 2], pos[b, 2]], color=CMAP(np.clip(w, 0, 1)),
                linewidth=1.3, alpha=0.7, solid_capstyle="round", zorder=1)

    sizes = 58.0 + 360.0 * norm
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=sizes * 5.0, c=cols,
               alpha=0.10, linewidths=0, zorder=2)
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=sizes, c=cols,
               alpha=0.95, linewidths=0, zorder=3)

    for i in d["label_idx"]:
        dx, dy = d["label_off"][i]
        ax.text(pos[i, 0] + dx, pos[i, 1] + dy, pos[i, 2] + d["label_dz"],
                d["names"][i], color=THEME["TEXT"], fontsize=8.5,
                ha="center", va="bottom", family=THEME["MONO"], alpha=0.85,
                zorder=20)

    ax.set_xlim(*d["lim"][0]); ax.set_ylim(*d["lim"][1]); ax.set_zlim(*d["lim"][2])
    ax.view_init(elev=s["ELEV"], azim=s["AZIM"])
    ax.set_box_aspect(d["box_aspect"], zoom=s["ZOOM"])
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    fig.text(0.5, L["BIG"], f"MEAN CORRELATION {figs['mean_corr']}", ha="center",
             fontsize=19, color=THEME["RED"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             f"tightest {figs['top_pair']} {figs['top_pair_corr']}  ·  "
             f"hub {figs['hub_name']}  ·  rim {figs['rim_name']}",
             ha="center", fontsize=12, color=THEME["TEXT_DIM"],
             family=THEME["MONO"])

    fig.savefig(out, dpi=s["DPI"], facecolor=THEME["BG"])
    plt.close(fig)
    log(f"static saved: {out}")


if __name__ == "__main__":
    render_static(os.path.join(BASE_DIR, "CorrelationSpace_Static.png"))
