"""
TenDays_Static_Pipeline.py
==========================
Hero still for the same topic: same DATA -> COMPUTE -> VALIDATE, one frame,
fixed curated camera, both coils finished.

WHAT THE VISUAL IS NOT
    The same disclaimer as the reel: not a strategy and not advice. Nobody can
    remove only the best days, the mirror case of dodging the worst ones is not
    shown, and the angle is a display wrap rather than a calendar cycle.

RUN
    python TenDays_Static_Pipeline.py
    -> TenDays_Static.png  (3240x5760; 10.8x19.2in at 300 DPI)
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from TenDays_Reel_Pipeline import (CONFIG, THEME, CMAP, BASE_DIR, LAYOUT,
                                   fetch, compute, validate,
                                   prepare_geometry, log)

STATIC = {"W": 3240, "H": 5760, "DPI": 300, "ELEV": 15, "AZIM": -72,
          "ZOOM": 1.25}


def render_static(out):
    d = compute(fetch())
    figs = validate(d)                       # same asserts guard the still
    prepare_geometry(d)
    s, L = STATIC, LAYOUT

    fig = plt.figure(figsize=(s["W"] / s["DPI"], s["H"] / s["DPI"]),
                     dpi=s["DPI"], facecolor=THEME["BG"])
    fig.text(0.5, L["TITLE"], "THE TEN DAYS", ha="center", fontsize=30,
             fontweight="bold", color=THEME["TEXT"], family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             f"SPY  |  {figs['n_sessions']} sessions  |  height = wealth",
             ha="center", fontsize=13, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.5, L["KEY"], "colour = size of that session's move",
             ha="center", fontsize=10, color=THEME["TEXT_DIM"],
             family=THEME["FONT"])
    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])

    ax = fig.add_axes(L["AXES"], projection="3d", facecolor=THEME["BG"])

    xs, ys = d["xy"]
    xe, ye = d["xy_ex"]
    w, w_ex = d["wealth"], d["wealth_ex"]
    n = d["n"]

    seg_ex = np.stack([np.column_stack([xe[:-1], ye[:-1], w_ex[:-1]]),
                       np.column_stack([xe[1:], ye[1:], w_ex[1:]])], axis=1)
    ax.add_collection3d(Line3DCollection(seg_ex, colors=(0.55, 0.60, 0.70, 0.60),
                                         linewidths=1.0, capstyle="round",
                                         zorder=2))

    step = max(n // 38, 1)
    ax.add_collection3d(Line3DCollection(
        [[(xs[i], ys[i], w[i]), (xe[i], ye[i], w_ex[i])] for i in range(0, n, step)],
        colors=(0.62, 0.68, 0.80, 0.20), linewidths=0.7, zorder=3))

    seg = np.stack([np.column_stack([xs[:-1], ys[:-1], w[:-1]]),
                    np.column_stack([xs[1:], ys[1:], w[1:]])], axis=1)
    mv = np.clip(np.abs(d["ret"].values[1:]) / max(d["move_hi"], 1e-9), 0, 1)
    ax.add_collection3d(Line3DCollection(seg, colors=CMAP(mv), linewidths=1.7,
                                         capstyle="round", zorder=4))

    ax.plot([xs[-1], xe[-1]], [ys[-1], ye[-1]], [w[-1], w_ex[-1]],
            color=THEME["TEXT"], alpha=0.55, linewidth=1.2, zorder=5)

    b = d["best_pos"]
    ax.scatter(xs[b], ys[b], w[b], s=150, color=THEME["RED"], alpha=0.18,
               linewidths=0, zorder=6)
    ax.scatter(xs[b], ys[b], w[b], s=34, color=THEME["TEXT"], alpha=0.95,
               linewidths=0, zorder=7)

    for px_, py_, pz_ in ((xs[-1], ys[-1], w[-1]), (xe[-1], ye[-1], w_ex[-1])):
        ax.scatter([px_], [py_], [pz_], s=420, color=THEME["ORANGE"],
                   alpha=0.13, linewidths=0, zorder=8)
        ax.scatter([px_], [py_], [pz_], s=70, color=THEME["ORANGE"],
                   alpha=1.0, linewidths=0, zorder=9)

    ax.set_xlim(*d["lim"][0]); ax.set_ylim(*d["lim"][1]); ax.set_zlim(*d["lim"][2])
    ax.view_init(elev=s["ELEV"], azim=s["AZIM"])
    ax.set_box_aspect(d["box_aspect"], zoom=s["ZOOM"])
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    fig.text(0.5, L["NOTE"], "left: every session.  right: the same decade minus ten",
             ha="center", fontsize=10, color=THEME["TEXT_DIM"],
             family=THEME["MONO"], alpha=0.85)
    fig.text(0.5, L["BIG"], f"{figs['gain_all']} BECOMES {figs['gain_ex']}",
             ha="center", fontsize=22, color=THEME["RED"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             f"{figs['n_missed']} missed sessions  ·  {figs['in_drawdown']} of "
             f"them inside a drawdown",
             ha="center", fontsize=12, color=THEME["TEXT_DIM"],
             family=THEME["MONO"])

    fig.savefig(out, dpi=s["DPI"], facecolor=THEME["BG"])
    plt.close(fig)
    log(f"static saved: {out}")


if __name__ == "__main__":
    render_static(os.path.join(BASE_DIR, "TenDays_Static.png"))
