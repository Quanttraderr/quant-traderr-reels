"""
VolLandscape_Static_Pipeline.py
===============================
Hero still for the same topic: same DATA -> COMPUTE -> VALIDATE, one frame, a
fixed curated camera, three times the resolution.

Rather than keeping a second copy of the drawing code, this drives the reel's
own render_frame with the figure enlarged to 3240x5760 and the camera frozen.
Every font size and line width is in points, and the figure stays 10.8x19.2
inches, so raising the DPI scales the whole picture instead of shrinking the
type inside a bigger canvas. That also means the still cannot drift away from
the reel: there is only one renderer, and the asserts guard both.

RUN
    python VolLandscape_Static_Pipeline.py
    -> VolLandscape_Static.png  (3240x5760; 10.8x19.2in at 300 DPI)
"""
import os

from VolLandscape_Reel_Pipeline import (CONFIG, BASE_DIR, fetch, compute,
                                        validate, prepare_geometry,
                                        render_frame, log)

STATIC = {"W": 3240, "H": 5760, "DPI": 300, "ELEV": 22, "AZIM": -58}


def render_static(out):
    d = compute(fetch())
    validate(d)                       # same asserts guard the still
    prepare_geometry(d)

    s = STATIC
    CONFIG["W"], CONFIG["H"], CONFIG["DPI"] = s["W"], s["H"], s["DPI"]
    # One curated angle instead of wherever the sweep happens to end. The
    # elevation ramp and the sine both vanish at the last frame, so setting the
    # base values fixes the camera exactly.
    CONFIG["ELEV_BASE"], CONFIG["AZIM_START"], CONFIG["AZIM_SWEEP"] = \
        s["ELEV"], s["AZIM"], 0

    total = int(CONFIG["FPS"] * (CONFIG["HOOK_SEC"] + CONFIG["BUILD_SEC"]
                                 + CONFIG["HOLD_SEC"]))
    render_frame(total - 1, total, d, out)
    log(f"static saved: {out}")


if __name__ == "__main__":
    render_static(os.path.join(BASE_DIR, "VolLandscape_Static.png"))
