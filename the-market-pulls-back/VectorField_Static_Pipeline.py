"""
VectorField_Static_Pipeline.py
==============================
Hero still for the same topic: same DATA -> COMPUTE -> VALIDATE, one frame,
fixed curated camera, the full field with the tracers caught mid stream.

WHAT THE VISUAL IS NOT
    The same disclaimer as the reel: the tracers are not market paths, the
    arrows are averages with a far wider spread around them, and part of the
    pull toward the middle is mechanical in a standardised space.

RUN
    python VectorField_Static_Pipeline.py
    -> VectorField_Static.png  (3240x5760; 10.8x19.2in at 300 DPI)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from VectorField_Reel_Pipeline import (CONFIG, THEME, CMAP, BASE_DIR, LAYOUT,
                                       fetch, compute, validate,
                                       prepare_geometry, log)

STATIC = {"W": 3240, "H": 5760, "DPI": 300, "ELEV": 16, "AZIM": -50,
          "ZOOM": 1.12, "FRAME": 300}


def render_static(out):
    d = compute(fetch())
    figs = validate(d)                       # same asserts guard the still
    prepare_geometry(d)
    c, s, L = CONFIG, STATIC, LAYOUT

    fig = plt.figure(figsize=(s["W"] / s["DPI"], s["H"] / s["DPI"]),
                     dpi=s["DPI"], facecolor=THEME["BG"])
    fig.text(0.5, L["TITLE"], "THE MARKET PULLS BACK", ha="center", fontsize=26,
             fontweight="bold", color=THEME["TEXT"], family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             f"SPY  |  {figs['n_years']} years  |  {figs['n_states']} states",
             ha="center", fontsize=13, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.5, L["KEY"], "colour = how hard the field pulls",
             ha="center", fontsize=10, color=THEME["TEXT_DIM"],
             family=THEME["FONT"])
    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])

    ax = fig.add_axes(L["AXES"], projection="3d", facecolor=THEME["BG"])

    pts, vecs = d["pts"], d["vecs"]
    segs = [[tuple(p), tuple(p + v * d["arrow_scale"])] for p, v in zip(pts, vecs)]
    ax.add_collection3d(Line3DCollection(segs, colors=d["arrow_cols"],
                                         linewidths=1.5, alpha=0.6, zorder=2))
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=7, c=d["arrow_cols"],
               alpha=0.55, linewidths=0, zorder=3)

    j = s["FRAME"] + d["warm"]
    hist = d["hist"]
    base = CMAP(np.clip(d["pspeed"][j] / max(d["speed_hi"], 1e-9), 0, 1) ** c["GAMMA"])
    tsegs, tcols = [], []
    alive = np.ones(hist.shape[1], dtype=bool)
    for b in range(c["TRAIL"] - 1):
        alive &= ~d["reborn"][j - b]
        if not alive.any():
            break
        fade = 0.95 * (1.0 - b / max(c["TRAIL"] - 1, 1)) ** 1.5
        cols = base[alive].copy()
        cols[:, 3] = fade
        tsegs.extend(np.stack([hist[j - b][alive], hist[j - b - 1][alive]], axis=1))
        tcols.append(cols)
    if tsegs:
        ax.add_collection3d(Line3DCollection(tsegs, colors=np.concatenate(tcols),
                                             linewidths=1.6, zorder=5))
    ax.scatter(hist[j][:, 0], hist[j][:, 1], hist[j][:, 2], s=15, c=base,
               alpha=0.95, linewidths=0, zorder=6)

    ax.set_xlim(*d["lim"][0]); ax.set_ylim(*d["lim"][1]); ax.set_zlim(*d["lim"][2])
    ax.view_init(elev=s["ELEV"], azim=s["AZIM"])
    ax.set_box_aspect(d["box_aspect"], zoom=s["ZOOM"])
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    fig.text(0.5, L["NOTE"], "tracers, not market paths: they show the field",
             ha="center", fontsize=10, color=THEME["TEXT_DIM"],
             family=THEME["MONO"], alpha=0.85)
    fig.text(0.5, L["BIG"], f"{figs['share_inward']} OF THE FIELD POINTS HOME",
             ha="center", fontsize=19, color=THEME["RED"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             f"{figs['n_inward']} of {figs['n_cells']} cells  ·  "
             f"{figs['horizon']}-session horizon",
             ha="center", fontsize=12, color=THEME["TEXT_DIM"],
             family=THEME["MONO"])

    fig.savefig(out, dpi=s["DPI"], facecolor=THEME["BG"])
    plt.close(fig)
    log(f"static saved: {out}")


if __name__ == "__main__":
    render_static(os.path.join(BASE_DIR, "VectorField_Static.png"))
