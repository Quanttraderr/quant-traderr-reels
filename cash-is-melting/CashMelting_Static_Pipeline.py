"""
CashMelting_Static_Pipeline.py
==============================
Hero still for "Cash Is Melting": same DATA -> COMPUTE -> VALIDATE, the
hourglass at the last CPI print, one fixed camera at the end of the reel's
sweep, the closing readout from figures.json.

RUN
    python CashMelting_Static_Pipeline.py
    -> CashMelting_Static.png  (3240x5760; 10.8x19.2in at 300 DPI)
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from CashMelting_Reel_Pipeline import (CONFIG, THEME, BASE_DIR, LAYOUT, load, log,
                                       scene, style_axes, scene_labels, text_block,
                                       camera, frame_plan)

STATIC = {"W": 3240, "H": 5760, "DPI": 300}


def render_static(out):
    d = load()                               # same asserts guard the still
    s = STATIC
    total = frame_plan()[2]
    fr = d["frames"][-1]                     # the held end state
    elev, azim = camera(total - 1, total)
    fig = plt.figure(figsize=(s["W"] / s["DPI"], s["H"] / s["DPI"]),
                     dpi=s["DPI"], facecolor=THEME["BG"])
    text_block(fig, d["figs"], fr)
    ax = fig.add_axes(LAYOUT["AXES"], projection="3d", facecolor=THEME["BG"])
    scene(ax, d, fr, azim, total - 1, s=0.85)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1.0, 1.0, 1.55), zoom=CONFIG["ZOOM"])
    style_axes(ax)
    scene_labels(ax, d, fr)
    fig.savefig(out, dpi=s["DPI"], facecolor=THEME["BG"])
    plt.close(fig)
    log(f"static saved: {out}")


if __name__ == "__main__":
    render_static(os.path.join(BASE_DIR, "CashMelting_Static.png"))
