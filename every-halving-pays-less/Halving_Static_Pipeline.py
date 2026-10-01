"""
Halving_Static_Pipeline.py
==========================
Hero still for "Every Halving Pays Less": same DATA -> COMPUTE -> VALIDATE,
the three finished chalices, one fixed camera at the end of the reel's sweep.

RUN
    python Halving_Static_Pipeline.py
    -> Halving_Static.png  (3240x5760; 10.8x19.2in at 300 DPI)
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from Halving_Reel_Pipeline import (CONFIG, THEME, BASE_DIR, LAYOUT, load, log,
                                   scene, style_axes, scene_labels, text_block)

STATIC = {"W": 3240, "H": 5760, "DPI": 300,
          "ELEV": CONFIG["ELEV_BASE"], "AZIM": CONFIG["AZIM_START"] + CONFIG["AZIM_SWEEP"]}


def render_static(out):
    d = load()                               # same asserts guard the still
    s, c = STATIC, CONFIG
    fig = plt.figure(figsize=(s["W"] / s["DPI"], s["H"] / s["DPI"]),
                     dpi=s["DPI"], facecolor=THEME["BG"])
    text_block(fig, d["figs"])
    ax = fig.add_axes(LAYOUT["AXES"], projection="3d", facecolor=THEME["BG"])
    zscale = scene(ax, d, 1.0, rows=220)
    ax.view_init(elev=s["ELEV"], azim=s["AZIM"])
    ax.set_box_aspect((2 * (c["SPACING"] + 1.35) / 2.8, 1.0, 2.0), zoom=c["ZOOM"])
    style_axes(ax)
    scene_labels(ax, d, 1.0, zscale)
    fig.savefig(out, dpi=s["DPI"], facecolor=THEME["BG"])
    plt.close(fig)
    log(f"static saved: {out}")


if __name__ == "__main__":
    render_static(os.path.join(BASE_DIR, "Halving_Static.png"))
