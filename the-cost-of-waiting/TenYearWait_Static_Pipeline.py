"""
TenYearWait_Static_Pipeline.py
===============================
Hero still for "The Cost of Waiting": same DATA -> COMPUTE -> VALIDATE, the finished
waterfall, one fixed camera.

RUN
    python TenYearWait_Static_Pipeline.py
    -> TenYearWait_Static.png  (3240x5760; 10.8x19.2in at 300 DPI)
"""
import os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from TenYearWait_Reel_Pipeline import (CONFIG, THEME, BASE_DIR, LAYOUT, fetch,
                                compute, validate, log, scene, note,
                                style_axes, scene_labels, text_block)

STATIC = {"W": 3240, "H": 5760, "DPI": 300, "ELEV": 15, "AZIM": -94}


def render_static(out):
    d = compute(fetch())
    figs = validate(d)                       # same asserts guard the still
    s = STATIC
    fig = plt.figure(figsize=(s["W"] / s["DPI"], s["H"] / s["DPI"]),
                     dpi=s["DPI"], facecolor=THEME["BG"])
    text_block(fig, figs)
    ax = fig.add_axes(LAYOUT["AXES"], projection="3d", facecolor=THEME["BG"])
    last = scene(ax, d, 1.0)
    note(fig, d, last)
    ax.view_init(elev=s["ELEV"], azim=s["AZIM"])
    ax.set_box_aspect((1.2, 0.7, 1.35), zoom=CONFIG["ZOOM"])
    style_axes(ax)
    scene_labels(ax, d, last)
    fig.savefig(out, dpi=s["DPI"], facecolor=THEME["BG"])
    plt.close(fig)
    log(f"static saved: {out}")


if __name__ == "__main__":
    render_static(os.path.join(BASE_DIR, "TenYearWait_Static.png"))
