"""
YieldRibbon_Reel_Pipeline.py
============================
Reel 6 for @quant.traderr. Twenty years of the US Treasury curve as a ribbon of
strands climbing through time, twisting where the curve inverts.

THE MATHS
    Four quoted maturities from Yahoo Finance: 13 weeks, 5 years, 10 years and
    30 years. One strand per session, running from short to long along the
    maturity axis, at its own height on the time axis. The maturity axis is
    spaced by the logarithm of years, because the distance from 3 months to 5
    years is not the same kind of step as 10 years to 30. Between the four
    quoted points the strand is linearly interpolated, purely so the ribbon
    reads as a surface; nothing is inferred about maturities that were not
    quoted.

    Colour is the 10 year yield minus the 13 week yield, the spread the phrase
    "inverted curve" refers to, mapped so that inversion is hot. A session
    counts as inverted when that spread is below zero.

THE CLAIM THE VISUAL MAKES
    An inversion is not a moment, it is a season. The longest one in this sample
    ran for 503 sessions, and the curve spent that time twisted the wrong way
    round while the calendar kept moving.

WHAT THE VISUAL IS NOT
    **Not a timing signal.** Twenty years hold a handful of inversion episodes,
    which is far too few to conclude anything about what follows one, and this
    reel deliberately shows no market return next to the curve. Four quoted
    maturities are not the curve; the strand between them is drawn, not
    measured. Yields here are index quotes rather than a fitted par curve, and
    nothing here forecasts rates.

RUN
    python YieldRibbon_Reel_Pipeline.py --smoke   # 3 frames -> temp_frames_smoke/
    python YieldRibbon_Reel_Pipeline.py           # full render -> topic.mp4

OUTPUT
    1080x1920 @ 30 FPS, 1.4s hook + 8.5s build + 1.8s hold = 11.7s

STAGES
    DATA -> COMPUTE -> VALIDATE (asserts, writes figures.json) -> RENDER -> COMPILE
"""
import argparse, json, os, shutil, subprocess, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Line3DCollection

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "_cache")

CONFIG = {
    # >>> REPLACE: data + maths knobs for your topic
    "TICKERS": ["^IRX", "^FVX", "^TNX", "^TYX"],
    "MATURITIES": [0.25, 5.0, 10.0, 30.0],     # years, matching TICKERS
    "PERIOD": "20y",
    "STRIDE": 6,              # one strand every 6 sessions
    "N_INTERP": 26,           # points along the drawn strand
    "Z_SPAN": 9.0,            # length of the time axis in plot units

    # render
    "W": 1080, "H": 1920, "DPI": 100,
    "FPS": 30,
    "HOOK_SEC": 1.4, "BUILD_SEC": 8.5, "HOLD_SEC": 1.8,
    "ELEV_BASE": 14, "AZIM_START": -72, "AZIM_SWEEP": 34,
    "ZOOM": 1.08,
}

THEME = {
    "BG": "#000000", "TEXT": "#ffffff", "TEXT_DIM": "#8a8a8a",
    "ORANGE": "#ff9500", "CYAN": "#00f2ff", "YELLOW": "#ffd400",
    "RED": "#ff3050", "GREEN": "#00ff8c", "BLUE": "#0066ff",
    "FONT": "DejaVu Sans", "MONO": "DejaVu Sans Mono",
}
CMAP = LinearSegmentedColormap.from_list(
    "house", [THEME["BLUE"], THEME["CYAN"], THEME["GREEN"],
              THEME["YELLOW"], THEME["ORANGE"], THEME["RED"]], N=256)

SAFE = {"TOP": 0.12, "BOTTOM": 0.20, "RIGHT": 0.14}

LAYOUT = {
    "TITLE": 0.862, "SUBTITLE": 0.836, "KEY": 0.814, "HANDLE": 0.792,
    "AXES": [-0.08, 0.30, 1.16, 0.485],
    "NOTE": 0.272, "BIG": 0.242, "SUB": 0.215,
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------- DATA
def fetch():
    """yfinance once, then a versioned CSV cache. The cache is what makes a
    re-render months from now draw the same bars the caption was asserted
    against. Delete _cache/ to refetch."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "yields.csv")
    if os.path.exists(path):
        log(f"[Data] cache {path}")
        return pd.read_csv(path, index_col=0, parse_dates=True)
    import yfinance as yf
    log(f"[Data] fetching {CONFIG['TICKERS']} {CONFIG['PERIOD']}")
    y = yf.download(CONFIG["TICKERS"], period=CONFIG["PERIOD"],
                    interval="1d", progress=False)["Close"]
    y = y[CONFIG["TICKERS"]].dropna()
    if y.empty:
        raise SystemExit("no data returned; refusing to draw a synthetic curve")
    y.to_csv(path)
    return y


# ------------------------------------------------------------- COMPUTE
def compute(y):
    # >>> REPLACE: the maths for your topic. Return whatever render_frame needs.
    c = CONFIG
    vals = y[c["TICKERS"]].values
    spread = vals[:, 2] - vals[:, 0]          # 10 year minus 13 week
    inverted = spread < 0.0

    # Longest unbroken run of inverted sessions.
    best_len, best_i, run, start = 0, 0, 0, 0
    for i, f in enumerate(inverted):
        if f:
            if run == 0:
                start = i
            run += 1
            if run > best_len:
                best_len, best_i = run, start
        else:
            run = 0

    deepest = int(np.argmin(spread))
    keep = list(range(0, len(vals), c["STRIDE"]))
    if keep[-1] != len(vals) - 1:
        keep.append(len(vals) - 1)            # the present is the last strand

    return {"y": y, "vals": vals, "spread": spread, "inverted": inverted,
            "keep": keep, "n_sessions": len(vals),
            "share_inverted": float(inverted.mean()),
            "longest": best_len,
            "inv_start": y.index[best_i], "inv_end": y.index[best_i + best_len - 1],
            "deepest": float(spread[deepest]), "deepest_date": y.index[deepest],
            "years": (y.index[-1] - y.index[0]).days / 365.25}


# ------------------------------------------------------------ VALIDATE
def validate(d):
    """Every number that reaches a frame is asserted here, before any render.

    The band on the longest run is deliberately generous at the top: an
    inversion lasting years is unusual but real, and a gate tight enough to
    exclude it would be a snapshot of this sample rather than a check."""
    c = CONFIG
    assert 0.0 < d["share_inverted"] < 0.60, f"inverted share {d['share_inverted']:.2f} outside the 0-60% band"
    assert 20 <= d["longest"] <= 1200, f"longest inversion {d['longest']} sessions outside the 20-1200 band"
    assert -5.0 <= d["deepest"] < 0.0, f"deepest spread {d['deepest']:.2f} outside the -5 to 0 band"
    assert d["n_sessions"] > 3000, f"only {d['n_sessions']} sessions, not two decades"
    assert d["vals"].min() >= -1.0, "a negative quoted yield: check the join"
    assert d["vals"].max() <= 25.0, "a yield above 25 percent: check the join"

    figs = {
        "longest": f"{d['longest']:,}",
        "inv_start": d["inv_start"].strftime("%b %Y"),
        "inv_end": d["inv_end"].strftime("%b %Y"),
        "share_inverted": f"{d['share_inverted'] * 100:.0f}%",
        "deepest": f"{abs(d['deepest']):.2f}",
        "deepest_date": d["deepest_date"].strftime("%b %Y"),
        "n_sessions": f"{d['n_sessions']:,}",
        "n_maturities": str(len(c["TICKERS"])),
        "n_years": f"{d['years']:.0f}",
        # Die Laufzeiten stehen als Zeichenketten hier, damit das Caption-Gate
        # auch "13 weeks" oder "30 years" im Text gegen etwas pruefen kann und
        # nicht jede Zahl im Fliesstext ungeprueft durchrutscht.
        "mat_short": f"{c['MATURITIES'][0] * 52:.0f} weeks",
        "mat_bench": f"{c['MATURITIES'][2]:.0f} year",
        "mat_long": f"{c['MATURITIES'][-1]:.0f} years",
    }
    with open(os.path.join(BASE_DIR, "figures.json"), "w") as fh:
        json.dump(figs, fh, indent=1)
    log(f"[validate] {figs}")
    d["figs"] = figs
    return figs


def prepare_geometry(d):
    """Every strand built once: a polyline per drawn session, plus its colour."""
    c = CONFIG
    mats = np.log10(np.array(c["MATURITIES"]))
    mats = (mats - mats.min()) / (mats.max() - mats.min()) * 3.0
    xs = np.linspace(mats[0], mats[-1], c["N_INTERP"])
    keep = d["keep"]
    zs = np.linspace(0.0, c["Z_SPAN"], len(keep))

    strands = []
    for mi, z in zip(keep, zs):
        ys = np.interp(xs, mats, d["vals"][mi])
        strands.append(np.column_stack([xs, ys, np.full_like(xs, z)]))

    sp = d["spread"][keep]
    # Inversion is the hot end of the ramp. Mapping it to the cold end would be
    # just as defensible and would hide the thing the reel is about.
    lo, hi = np.percentile(sp, 1), np.percentile(sp, 99)
    norm = np.clip(1.0 - (sp - lo) / max(hi - lo, 1e-9), 0, 1)
    cols = CMAP(norm)
    cols[:, 3] = 0.72

    d["strands"] = strands
    d["strand_cols"] = cols
    d["strand_inverted"] = d["inverted"][keep]

    allpts = np.concatenate(strands)
    lo3, hi3 = allpts.min(axis=0), allpts.max(axis=0)
    pad = 0.04 * float((hi3 - lo3).max())
    d["lim"] = [(float(lo3[j]) - pad, float(hi3[j]) + pad) for j in range(3)]
    span = np.array([b - a for a, b in d["lim"]])
    d["box_aspect"] = tuple(span / span.max())
    return d


# -------------------------------------------------------------- RENDER
def render_frame(idx, total, d, out_path):
    c, figs, L = CONFIG, d["figs"], LAYOUT
    n_hook = int(c["FPS"] * c["HOOK_SEC"])
    n_build = int(c["FPS"] * c["BUILD_SEC"])
    if idx < n_hook:
        frac = 0.0
    elif idx < n_hook + n_build:
        frac = (idx - n_hook) / max(n_build - 1, 1)
    else:
        frac = 1.0
    t = idx / max(total - 1, 1)

    fig = plt.figure(figsize=(c["W"] / c["DPI"], c["H"] / c["DPI"]),
                     dpi=c["DPI"], facecolor=THEME["BG"])
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

    # >>> REPLACE: the geometry. The ribbon grows one strand at a time.
    n = max(int(round(frac * len(d["strands"]))), 1)
    ax.add_collection3d(Line3DCollection(d["strands"][:n],
                                         colors=d["strand_cols"][:n],
                                         linewidths=1.25, zorder=2))

    # The strand that just landed, drawn bright, so the eye has a leading edge
    # to follow up the ribbon.
    lead = d["strands"][n - 1]
    ax.add_collection3d(Line3DCollection([lead], colors=[(1, 1, 1, 0.9)],
                                         linewidths=2.2, zorder=5))

    ax.set_xlim(*d["lim"][0]); ax.set_ylim(*d["lim"][1]); ax.set_zlim(*d["lim"][2])
    ax.view_init(elev=c["ELEV_BASE"] + 5 * np.sin(t * np.pi),
                 azim=c["AZIM_START"] + c["AZIM_SWEEP"] * t)
    ax.set_box_aspect(d["box_aspect"], zoom=c["ZOOM"])
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    fig.text(0.5, L["NOTE"], "short rates on the left, 30 years on the right",
             ha="center", fontsize=10, color=THEME["TEXT_DIM"],
             family=THEME["MONO"], alpha=0.85)

    # Readout: strings only, straight from figures.json. Never re-derive here.
    fig.text(0.5, L["BIG"], f"INVERTED FOR {figs['longest']} SESSIONS",
             ha="center", fontsize=20, color=THEME["RED"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             f"{figs['inv_start']} to {figs['inv_end']}  ·  deepest "
             f"{figs['deepest']} points, {figs['deepest_date']}",
             ha="center", fontsize=11, color=THEME["TEXT_DIM"],
             family=THEME["MONO"])

    fig.savefig(out_path, dpi=c["DPI"], facecolor=THEME["BG"])
    plt.close(fig)


# ------------------------------------------------------------- COMPILE
def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def main(smoke=False):
    c = CONFIG
    d = compute(fetch())
    validate(d)                       # <- before a single frame is written
    prepare_geometry(d)

    total = int(c["FPS"] * (c["HOOK_SEC"] + c["BUILD_SEC"] + c["HOLD_SEC"]))
    if smoke:
        out = os.path.join(BASE_DIR, "temp_frames_smoke")
        shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
        for k, idx in enumerate((0, total // 2, total - 1)):
            render_frame(idx, total, d, os.path.join(out, f"smoke_{k}.png"))
        log(f"smoke: 3 frames -> {out}")
        return

    frames = os.path.join(BASE_DIR, "temp_frames")
    shutil.rmtree(frames, ignore_errors=True); os.makedirs(frames)
    t0 = time.time()
    for i in range(total):
        render_frame(i, total, d, os.path.join(frames, f"f_{i:05d}.png"))
        if i % 30 == 0:
            log(f"  frame {i}/{total}  ({time.time() - t0:.0f}s)")

    mp4 = os.path.join(BASE_DIR, "topic.mp4")
    log("encoding...")
    r = subprocess.run([ffmpeg_exe(), "-y", "-framerate", str(c["FPS"]),
                        "-i", os.path.join(frames, "f_%05d.png"),
                        "-sws_flags", "lanczos+accurate_rnd+full_chroma_int",
                        "-c:v", "libx264", "-preset", "slow", "-crf", "14",
                        "-pix_fmt", "yuv420p",
                        "-color_primaries", "bt709", "-color_trc", "bt709",
                        "-colorspace", "bt709", "-movflags", "+faststart", mp4],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(r.stderr[-2000:])
    shutil.rmtree(frames, ignore_errors=True)
    log(f"OK {mp4}  ({total} frames, {total / c['FPS']:.1f}s, {c['W']}x{c['H']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    main(**vars(ap.parse_args()))
