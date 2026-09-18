"""
Tomography_Reel_Pipeline.py
===========================
Reel 4 for @quant.traderr. Fifteen years of cross asset correlation as a stack
of slices, one slice per rolling window, read from the bottom of the decade up.

THE MATHS
    Daily log returns of N_ASSETS exchange traded funds spanning US large cap,
    US small cap, developed and emerging equity, long and short government
    bonds, investment grade and high yield credit, gold and real estate. For
    every session a correlation matrix is computed over the trailing WINDOW
    sessions. Every SLICE_STRIDE windows one of those matrices is drawn as a
    grid of cells at its own height, so height is time and each plate is one
    picture of how the market was moving together at that moment. The diagonal
    is left empty because a correlation of one with itself carries nothing. A
    session counts as a crisis when SPY sits more than DD_THRESHOLD below its
    running high.

THE CLAIM THE VISUAL MAKES
    Average pairwise correlation is not a constant. It climbs when the market
    falls, which is the one moment a diversified portfolio is supposed to
    protect you, and it was at its calmest reading of the whole period a few
    weeks before the fastest crash in the sample.

WHAT THE VISUAL IS NOT
    Not causation and not a forecast. Ten funds are not the investable
    universe, correlation is linear dependence only and says nothing about tail
    behaviour, and the windows overlap so neighbouring slices share most of
    their data. The calm reading before a crash is one observation in this
    sample, not a signal, and nothing here says a low reading predicts anything.

RUN
    python Tomography_Reel_Pipeline.py --smoke   # 3 frames -> temp_frames_smoke/
    python Tomography_Reel_Pipeline.py           # full render -> topic.mp4

OUTPUT
    1440x2560 (2K vertical) @ 30 FPS, 1.4s hook + 8.5s build + 1.8s hold = 11.7s

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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "_cache")

CONFIG = {
    # >>> REPLACE: data + maths knobs for your topic
    # Ten funds across asset classes, not ten flavours of the same thing. A
    # basket of US equity sectors would answer a different and easier question.
    "TICKERS": ["SPY", "QQQ", "IWM", "EFA", "EEM",
                "TLT", "LQD", "HYG", "GLD", "VNQ"],
    "PERIOD": "15y",
    "WINDOW": 60,
    "SLICE_STRIDE": 110,      # one drawn slice every 110 windows
    "DD_THRESHOLD": 0.15,     # what counts as a crisis
    "CORR_LO": -0.10,         # colour ramp ends
    "CORR_HI": 0.90,
    "Z_SPAN": 14.0,           # height of the stack in grid units

    # render (leave these alone unless the topic needs landscape)
    "W": 1440, "H": 2560, "DPI": 400 / 3,   # 2K master. The figure stays
    # 10.8x19.2in, so every fontsize and linewidth lands where it would at
    # 1080x1920; only the sampling density goes up.
    "FPS": 30,
    "HOOK_SEC": 1.4, "BUILD_SEC": 8.5, "HOLD_SEC": 1.8,
    "ELEV_BASE": 13, "AZIM_START": -66, "AZIM_SWEEP": 30,
    "ZOOM": 1.12,
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

# Instagram lays its own controls over a reel: the Reels header on top, the
# name, caption and audio line at the bottom, the like and comment buttons on
# the right. Nothing readable may sit in those bands. The form itself may bleed
# into them, text never.
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
    path = os.path.join(CACHE_DIR, "closes.csv")
    if os.path.exists(path):
        log(f"[Data] cache {path}")
        return pd.read_csv(path, index_col=0, parse_dates=True)
    import yfinance as yf
    log(f"[Data] fetching {len(CONFIG['TICKERS'])} funds, {CONFIG['PERIOD']}")
    px = yf.download(CONFIG["TICKERS"], period=CONFIG["PERIOD"],
                     interval="1d", progress=False, auto_adjust=True)["Close"]
    px = px[CONFIG["TICKERS"]].dropna()
    if px.empty:
        raise SystemExit("no data returned; refusing to draw a synthetic tape")
    px.to_csv(path)
    return px


# ------------------------------------------------------------- COMPUTE
def compute(px):
    # >>> REPLACE: the maths for your topic. Return whatever render_frame needs.
    c = CONFIG
    names = list(px.columns)
    n = len(names)
    rets = np.log(px / px.shift(1)).dropna()
    vals = rets.values
    w = c["WINDOW"]
    iu = np.triu_indices(n, 1)

    mats, mean_corr, dates = [], [], []
    for i in range(w, len(vals) + 1):
        m = np.corrcoef(vals[i - w:i].T)
        mats.append(m)
        mean_corr.append(float(m[iu].mean()))
        dates.append(rets.index[i - 1])
    mean_corr = pd.Series(mean_corr, index=dates)

    # A crisis is a real drawdown in the reference asset, not a volatility
    # threshold: the claim is about what happens when portfolios are losing
    # money, which is what people actually experience.
    spy = px[c["TICKERS"][0]]
    dd = (spy / spy.cummax() - 1.0).reindex(mean_corr.index)
    crisis = (dd < -c["DD_THRESHOLD"]).values

    keep = list(range(0, len(mats), c["SLICE_STRIDE"]))
    if keep[-1] != len(mats) - 1:
        keep.append(len(mats) - 1)      # the present is always the top slice

    return {"names": names, "n": n, "mats": mats, "mean_corr": mean_corr,
            "dates": dates, "crisis": crisis, "keep": keep,
            "calm": float(mean_corr.values[~crisis].mean()),
            "crisis_corr": float(mean_corr.values[crisis].mean()),
            "share_crisis": float(crisis.mean()),
            "min_corr": float(mean_corr.min()), "min_date": mean_corr.idxmin(),
            "max_corr": float(mean_corr.max()), "max_date": mean_corr.idxmax(),
            "n_windows": len(mats)}


# ------------------------------------------------------------ VALIDATE
def validate(d):
    """Every number that reaches a frame is asserted here, before any render.

    The gate on crisis_corr > calm is not decoration. It is the claim of the
    reel, so a data refresh that reverses it must stop the build rather than
    ship a video whose picture contradicts its own caption."""
    c = CONFIG
    assert 0.05 <= d["calm"] <= 0.80, f"calm correlation {d['calm']:.2f} outside the 0.05-0.80 band"
    assert 0.10 <= d["crisis_corr"] <= 0.95, f"crisis correlation {d['crisis_corr']:.2f} outside the 0.10-0.95 band"
    assert d["crisis_corr"] > d["calm"], "correlation did not rise in drawdowns; the reel's claim fails"
    assert 0.0 <= d["min_corr"] <= 0.55, f"lowest reading {d['min_corr']:.2f} outside the 0-0.55 band"
    assert 0.30 <= d["max_corr"] <= 0.99, f"highest reading {d['max_corr']:.2f} outside the 0.30-0.99 band"
    assert 0.01 <= d["share_crisis"] <= 0.50, f"crisis share {d['share_crisis']:.2f} outside the 1-50% band"
    assert d["n_windows"] > 2000, f"only {d['n_windows']} windows, too few for a decade and a half"
    assert d["n"] == len(c["TICKERS"]), "an asset dropped out of the join"

    figs = {
        "calm_corr": f"{d['calm']:.2f}",
        "crisis_corr": f"{d['crisis_corr']:.2f}",
        "min_corr": f"{d['min_corr']:.2f}",
        "min_date": d["min_date"].strftime("%b %Y"),
        "max_corr": f"{d['max_corr']:.2f}",
        "max_date": d["max_date"].strftime("%b %Y"),
        "n_assets": str(d["n"]),
        "n_windows": f"{d['n_windows']:,}",
        "window_days": str(c["WINDOW"]),
        "dd_threshold": f"{c['DD_THRESHOLD'] * 100:.0f}%",
    }
    with open(os.path.join(BASE_DIR, "figures.json"), "w") as fh:
        json.dump(figs, fh, indent=1)
    log(f"[validate] {figs}")
    d["figs"] = figs
    return figs


def prepare_geometry(d):
    """Build every quad once. A slice is 90 cells, the stack is about forty
    slices, and rebuilding 3600 polygons inside render_frame would dominate the
    render. The diagonal is skipped: a correlation of one with itself is not
    information, and leaving it out stops a red spine running up the stack."""
    c = CONFIG
    n, keep = d["n"], d["keep"]
    zs = np.linspace(0.0, c["Z_SPAN"], len(keep))

    quads, cols, plate = [], [], []
    for s, (mi, z) in enumerate(zip(keep, zs)):
        m = d["mats"][mi]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                quads.append([(i, j, z), (i + 1, j, z),
                              (i + 1, j + 1, z), (i, j + 1, z)])
                cols.append((m[i, j] - c["CORR_LO"]) /
                            max(c["CORR_HI"] - c["CORR_LO"], 1e-9))
        plate.append(z)

    # Each plate also gets a frame coloured by its own mean off-diagonal
    # correlation. Without it the stack is a rainbow and the one thing the reel
    # is about, the level moving with the market, is invisible: the cells carry
    # the detail, the frames carry the signal.
    iu = np.triu_indices(n, 1)
    ring_segs, ring_cols = [], []
    for mi, z in zip(keep, zs):
        mean = float(d["mats"][mi][iu].mean())
        col = CMAP(np.clip((mean - c["CORR_LO"]) /
                           max(c["CORR_HI"] - c["CORR_LO"], 1e-9), 0, 1))
        for a, b in (((0, 0), (n, 0)), ((n, 0), (n, n)),
                     ((n, n), (0, n)), ((0, n), (0, 0))):
            ring_segs.append([(a[0], a[1], z), (b[0], b[1], z)])
            ring_cols.append(col)
    d["ring_segs"] = np.array(ring_segs, dtype=float)
    d["ring_cols"] = np.array(ring_cols)
    d["ring_cols"][:, 3] = 0.95

    d["quads"] = np.array(quads, dtype=float)
    d["quad_cols"] = CMAP(np.clip(np.array(cols), 0, 1))
    d["quad_cols"][:, 3] = 0.62
    d["per_slice"] = n * n - n
    d["plate_z"] = np.array(plate)
    d["slice_idx"] = keep

    pad = 0.5
    d["lim"] = [(-pad, n + pad), (-pad, n + pad), (-1.0, c["Z_SPAN"] + 1.0)]
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
    fig.text(0.5, L["TITLE"], "WHEN DIVERSIFICATION DIES", ha="center",
             fontsize=25, fontweight="bold", color=THEME["TEXT"],
             family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             f"{figs['n_assets']} asset classes  |  {figs['n_windows']} rolling "
             f"{figs['window_days']}-day windows",
             ha="center", fontsize=12.5, color=THEME["ORANGE"],
             family=THEME["FONT"])
    fig.text(0.5, L["KEY"], "plate colour = how much the market moved as one",
             ha="center", fontsize=10, color=THEME["TEXT_DIM"],
             family=THEME["FONT"])
    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])

    ax = fig.add_axes(L["AXES"], projection="3d", facecolor=THEME["BG"])

    # >>> REPLACE: the geometry. The stack fills upward, one plate per window.
    n_slice = max(int(round(frac * len(d["plate_z"]))), 1)
    k = n_slice * d["per_slice"]
    ax.add_collection3d(Poly3DCollection(d["quads"][:k],
                                         facecolors=d["quad_cols"][:k],
                                         linewidths=0, zorder=2))
    ax.add_collection3d(Line3DCollection(d["ring_segs"][:n_slice * 4],
                                         colors=d["ring_cols"][:n_slice * 4],
                                         linewidths=1.7, zorder=5))

    # A thin frame around the plate that just landed, so the eye has somewhere
    # to be while the stack builds.
    z = d["plate_z"][n_slice - 1]
    nn = d["n"]
    ring = [[(0, 0, z), (nn, 0, z)], [(nn, 0, z), (nn, nn, z)],
            [(nn, nn, z), (0, nn, z)], [(0, nn, z), (0, 0, z)]]
    ax.add_collection3d(Line3DCollection(ring, colors=(1, 1, 1, 0.55),
                                         linewidths=1.2, zorder=6))

    ax.set_xlim(*d["lim"][0]); ax.set_ylim(*d["lim"][1]); ax.set_zlim(*d["lim"][2])
    ax.view_init(elev=c["ELEV_BASE"] + 5 * np.sin(t * np.pi),
                 azim=c["AZIM_START"] + c["AZIM_SWEEP"] * t)
    ax.set_box_aspect(d["box_aspect"], zoom=c["ZOOM"])
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    fig.text(0.5, L["NOTE"], "each plate is one 60-day correlation matrix",
             ha="center", fontsize=10, color=THEME["TEXT_DIM"],
             family=THEME["MONO"], alpha=0.85)

    # Readout: strings only, straight from figures.json. Never re-derive here.
    fig.text(0.5, L["BIG"],
             f"{figs['calm_corr']} CALM, {figs['crisis_corr']} IN A CRASH",
             ha="center", fontsize=21, color=THEME["RED"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             f"lowest reading of all: {figs['min_corr']}, {figs['min_date']}",
             ha="center", fontsize=12, color=THEME["TEXT_DIM"],
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
                        "-c:v", "libx264", "-preset", "slow", "-crf", "18",
                        "-pix_fmt", "yuv420p", mp4],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(r.stderr[-2000:])
    shutil.rmtree(frames, ignore_errors=True)   # the cache is 6 GB if you don't
    log(f"OK {mp4}  ({total} frames, {total / c['FPS']:.1f}s, {c['W']}x{c['H']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    main(**vars(ap.parse_args()))
