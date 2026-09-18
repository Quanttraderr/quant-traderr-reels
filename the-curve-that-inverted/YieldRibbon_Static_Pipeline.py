"""
YieldRibbon_Static_Pipeline.py
==============================
Hero still for the same topic: same DATA -> COMPUTE -> VALIDATE, one frame,
fixed curated camera, the whole ribbon.

WHAT THE VISUAL IS NOT
    The same disclaimer as the reel: not a timing signal, four quoted
    maturities are not the curve, and the strand between them is drawn rather
    than measured.

RUN
    python YieldRibbon_Static_Pipeline.py
    -> YieldRibbon_Static.png  (3240x5760; 10.8x19.2in at 300 DPI)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from YieldRibbon_Reel_Pipeline import (CONFIG, THEME, BASE_DIR, LAYOUT,
                                       fetch, compute, validate,
                                       prepare_geometry, log)

STATIC = {"W": 3240, "H": 5760, "DPI": 300, "ELEV": 14, "AZIM": -55,
          "ZOOM": 1.08}


def render_static(out):
    d = compute(fetch())
    figs = validate(d)                       # same asserts guard the still
    prepare_geometry(d)
    s, L = STATIC, LAYOUT

    fig = plt.figure(figsize=(s["W"] / s["DPI"], s["H"] / s["DPI"]),
                     dpi=s["DPI"], facecolor=THEME["BG"])
    fig.text(0.5, L["TITLE"], "THE CURVE THAT INVERTED", ha="center",
             fontsize=26, fontweight="bold", color=THEME["TEXT"],
             family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             f"US Treasuries  |  {figs['n_years']} years  |  "
             f"{figs['n_sessions']} sessions",
             ha="center", fontsize=13, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.5, L["KEY"], "one strand per session, red where inverted",
             ha="center", fontsize=10, color=THEME["TEXT_DIM"],
             family=THEME["FONT"])
    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])

    ax = fig.add_axes(L["AXES"], projection="3d", facecolor=THEME["BG"])
    ax.add_collection3d(Line3DCollection(d["strands"], colors=d["strand_cols"],
                                         linewidths=1.25, zorder=2))
    ax.add_collection3d(Line3DCollection([d["strands"][-1]],
                                         colors=[(1, 1, 1, 0.9)],
                                         linewidths=2.2, zorder=5))

    ax.set_xlim(*d["lim"][0]); ax.set_ylim(*d["lim"][1]); ax.set_zlim(*d["lim"][2])
    ax.view_init(elev=s["ELEV"], azim=s["AZIM"])
    ax.set_box_aspect(d["box_aspect"], zoom=s["ZOOM"])
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    fig.text(0.5, L["NOTE"], "short rates on the left, 30 years on the right",
             ha="center", fontsize=10, color=THEME["TEXT_DIM"],
             family=THEME["MONO"], alpha=0.85)
    fig.text(0.5, L["BIG"], f"INVERTED FOR {figs['longest']} SESSIONS",
             ha="center", fontsize=20, color=THEME["RED"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             f"{figs['inv_start']} to {figs['inv_end']}  ·  deepest "
             f"{figs['deepest']} points, {figs['deepest_date']}",
             ha="center", fontsize=11, color=THEME["TEXT_DIM"],
             family=THEME["MONO"])

    fig.savefig(out, dpi=s["DPI"], facecolor=THEME["BG"])
    plt.close(fig)
    log(f"static saved: {out}")


if __name__ == "__main__":
    render_static(os.path.join(BASE_DIR, "YieldRibbon_Static.png"))
