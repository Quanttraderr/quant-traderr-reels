"""
Gaps_Static_Pipeline.py
=======================
Hero still for "Gaps Always Fill?": same DATA -> COMPUTE -> VALIDATE, the scene
finished, one fixed camera at the end of the reel's sweep.

RUN
    python Gaps_Static_Pipeline.py
    -> Gaps_Static.png  (3240x5760; 10.8x19.2in at 300 DPI)
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Gaps_Reel_Pipeline import (CONFIG, THEME, BASE_DIR, LAYOUT, load,
                                log, scene, style_axes, scene_labels, text_block)

STATIC = {"W": 3240, "H": 5760, "DPI": 300,
          "ELEV": CONFIG["ELEV_BASE"], "AZIM": CONFIG["AZIM_START"] + CONFIG["AZIM_SWEEP"]}


def render_static(out):
    d = load()                               # same asserts guard the still
    s = STATIC
    fig = plt.figure(figsize=(s["W"] / s["DPI"], s["H"] / s["DPI"]),
                     dpi=s["DPI"], facecolor=THEME["BG"])
    text_block(fig, d["figs"])
    ax = fig.add_axes(LAYOUT["AXES"], projection="3d", facecolor=THEME["BG"])
    scene(ax, d, 1.0, lw=0.7)
    ax.view_init(elev=s["ELEV"], azim=s["AZIM"])
    ax.set_box_aspect((1.7, 1.0, 0.75), zoom=CONFIG["ZOOM"])
    style_axes(ax)
    scene_labels(ax, d, 1.0)
    fig.savefig(out, dpi=s["DPI"], facecolor=THEME["BG"])
    plt.close(fig)
    log(f"static saved: {out}")


if __name__ == "__main__":
    render_static(os.path.join(BASE_DIR, "Gaps_Static.png"))
