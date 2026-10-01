"""
BuyTheFear_Reel_Pipeline.py
===========================
Reel 20 for @quant.traderr, "Buy When It Hurts". What SPY did in the twelve
months after a day, sorted by how scared the market was on that day (VIX).

THE DATA
    SPY daily closes (auto-adjusted, dividends reinvested) and the CBOE VIX
    close from Yahoo, Jan 1993 to Sep 2026. Real data throughout. Nothing here
    is simulated.

THE MATHS
    fwd_t = SPY_{t+252} / SPY_t - 1, the total return over the next 252
    sessions. Days without a full year ahead are dropped. Every figure in the
    readout is computed on DAILY observations, bucketed by the VIX close on
    the day of purchase.

    In 3D: a Voronoi tessellation of month-ends. Each month-end with a full
    year ahead is a seed at (time, log VIX). Its cell is extruded to a height
    equal to that month-end's forward 12-month return; cells below the floor
    lost money. Colour carries the same return. The calm months spread along
    the front, the panic months sit high on the VIX axis at the back, and they
    are the tall cells. White lines mark VIX 40 and VIX 15. The build raises the
    cells in calendar order.

THE CLAIM THE VISUAL MAKES
    The scariest days were the best days to buy. VIX at 40 or above was
    followed by a far higher average 12-month return than an ordinary day.

WHAT THE VISUAL IS NOT
    Not a timing rule. The VIX-40 days cluster into a handful of crises
    (1998, 2001 to 2002, 2008 to 2011, 2015, 2020, 2025), so the sample is a
    few episodes, not hundreds of independent trades, and consecutive days
    share most of their forward year. Sep 2001 bought at VIX 40 was still down
    a year later. Hindsight picks the crises that ended; a crisis that does
    not end looks different. Cell areas come from how month-ends spread over
    time and VIX and carry no weight in the numbers.

RUN
    python BuyTheFear_Reel_Pipeline.py --smoke   # 3 frames
    python BuyTheFear_Reel_Pipeline.py           # full render -> topic.mp4

OUTPUT
    1080x1920 @ 30 FPS, 1.4s hook + 8.5s build + 1.8s hold = 11.7s

STAGES
    DATA -> COMPUTE -> VALIDATE (asserts, writes figures.json) -> RENDER -> COMPILE
"""
import argparse, json, os, shutil, subprocess, time, warnings
from multiprocessing import Pool
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "_cache")

CONFIG = {
    "START": "1993-01-29",
    "END": "2026-09-30",
    "HORIZON": 252,
    "FEAR": 40.0,
    "CALM": 15.0,
    "Z_SCALE": 0.03,
    "CELL_REACH": 0.55,

    # render
    "W": 1080, "H": 1920, "DPI": 100,
    "FPS": 30,
    "HOOK_SEC": 1.4, "BUILD_SEC": 8.5, "HOLD_SEC": 1.8,
    "ELEV_BASE": 20, "AZIM_START": 100, "AZIM_SWEEP": 22,
    "ZOOM": 1.05,
    "WORKERS": 7,
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

RET_LO, RET_HI = -40.0, 70.0          # colour scale, forward return in %


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------- DATA
def fetch():
    """yfinance once, then a CSV cache. Delete _cache/ to refetch."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "spy_vix.csv")
    if not os.path.exists(path):
        import yfinance as yf
        log("[Data] fetching SPY and ^VIX")
        spy = yf.download("SPY", start=CONFIG["START"], progress=False,
                          auto_adjust=True)["Close"].squeeze()
        vix = yf.download("^VIX", start=CONFIG["START"], progress=False,
                          auto_adjust=True)["Close"].squeeze()
        df = pd.DataFrame({"SPY": spy, "VIX": vix}).dropna()
        if df.empty:
            raise SystemExit("no data returned; refusing to draw a synthetic tape")
        df.to_csv(path)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.loc[CONFIG["START"]:CONFIG["END"]]


# ------------------------------------------------------------- COMPUTE
def _voronoi_cells(pts, box):
    """Bounded Voronoi cells: mirror the seeds across the four box edges so
    every original cell is finite, then keep the first len(pts) regions."""
    from scipy.spatial import Voronoi
    x0, x1, y0, y1 = box
    mirrored = np.vstack([pts,
                          np.c_[2 * x0 - pts[:, 0], pts[:, 1]],
                          np.c_[2 * x1 - pts[:, 0], pts[:, 1]],
                          np.c_[pts[:, 0], 2 * y0 - pts[:, 1]],
                          np.c_[pts[:, 0], 2 * y1 - pts[:, 1]]])
    vor = Voronoi(mirrored)
    cells = []
    for i in range(len(pts)):
        reg = vor.regions[vor.point_region[i]]
        poly = vor.vertices[reg]
        c = poly.mean(axis=0)
        ang = np.arctan2(poly[:, 1] - c[1], poly[:, 0] - c[0])
        cells.append(poly[np.argsort(ang)])
    return cells


def compute(df):
    h = CONFIG["HORIZON"]
    df = df.copy()
    df["fwd"] = df["SPY"].shift(-h) / df["SPY"] - 1
    daily = df.dropna()

    # geometry: one seed per month-end with a full year ahead
    me = daily.groupby(daily.index.to_period("M")).tail(1)
    t = (me.index - me.index[0]).days.values.astype(float)
    x = t / t.max() * 10.0
    lv = np.log(me["VIX"].values)
    y = (lv - np.log(9.0)) / (np.log(90.0) - np.log(9.0)) * 8.0
    pts = np.c_[x, y]
    cells = _voronoi_cells(pts, (0.0, 10.0, 0.0, 8.0))
    # Display only: pull every cell 10% toward its seed so the cells read as
    # separate crystals, and cap the reach of the few lonely edge cells.
    R = CONFIG["CELL_REACH"]
    out = []
    for seed, poly in zip(pts, cells):
        v = (poly - seed) * 0.9
        dist = np.linalg.norm(v, axis=1, keepdims=True)
        v = np.where(dist > R, v / dist * R, v)
        out.append(seed + v)
    cells = out
    ly = lambda v: (np.log(v) - np.log(9.0)) / (np.log(90.0) - np.log(9.0)) * 8.0
    return {"calm_y": ly(CONFIG["CALM"]),"df": df, "daily": daily, "me": me, "pts": pts, "cells": cells,
            "fear_y": (np.log(CONFIG["FEAR"]) - np.log(9.0))
                      / (np.log(90.0) - np.log(9.0)) * 8.0}


# ------------------------------------------------------------ VALIDATE
def validate(d):
    """Every number that reaches a frame is asserted here, before any render."""
    dl, c = d["daily"], CONFIG
    fear = dl[dl["VIX"] >= c["FEAR"]]
    calm = dl[dl["VIX"] < c["CALM"]]
    m_all, m_fear, m_calm = dl["fwd"].mean(), fear["fwd"].mean(), calm["fwd"].mean()
    p_all, p_fear = (dl["fwd"] > 0).mean(), (fear["fwd"] > 0).mean()
    years = sorted(set(fear.index.year))

    assert len(dl) > 7000, f"only {len(dl)} days with a forward year"
    assert 30 <= len(fear) <= 1500, f"{len(fear)} fear days, VIX series looks off"
    assert -0.5 < m_all < 0.5 and -0.8 < m_fear < 2.0 and -0.5 < m_calm < 0.5
    assert m_fear > m_all, "fear days did not beat the average day; claim gone"
    assert 0 <= p_fear <= 1 and 0 <= p_all <= 1
    assert len(d["cells"]) == len(d["me"])
    assert all(np.isfinite(cell).all() for cell in d["cells"])

    figs = {
        "fear_level": f"{c['FEAR']:.0f}",
        "calm_level": f"{c['CALM']:.0f}",
        "fear_mean": f"+{m_fear * 100:.0f}%",
        "all_mean": f"+{m_all * 100:.0f}%",
        "calm_mean": f"+{m_calm * 100:.0f}%",
        "fear_pos": f"{p_fear * 100:.0f}%",
        "all_pos": f"{p_all * 100:.0f}%",
        "fear_worst": f"{fear['fwd'].min() * 100:.0f}%",
        "fear_worst_date": fear["fwd"].idxmin().strftime("%b %Y"),
        "n_fear": f"{len(fear):,}",
        "n_days": f"{len(dl):,}",
        "n_fear_years": f"{len(years)}",
        "n_months": f"{len(d['me'])}",
        "vix_peak": f"{d['df']['VIX'].max():.0f}",
        "vix_peak_date": d["df"]["VIX"].idxmax().strftime("%b %Y"),
        "start": dl.index[0].strftime("%Y"),
        "last_seed": d["me"].index[-1].strftime("%Y"),
        "end": dl.index[-1].strftime("%b %Y"),
    }
    with open(os.path.join(BASE_DIR, "figures.json"), "w") as fh:
        json.dump(figs, fh, indent=1)
    log(f"[validate] {figs}")
    d["figs"] = figs
    return figs


# -------------------------------------------------------------- RENDER
def style_axes(ax):
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
        pane.line.set_color((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def _prism(poly, z):
    """Top face and side walls of one extruded cell, floor at 0."""
    n = len(poly)
    top = [(px, py, z) for px, py in poly]
    walls = []
    for i in range(n):
        (ax_, ay), (bx, by) = poly[i], poly[(i + 1) % n]
        walls.append([(ax_, ay, 0), (bx, by, 0), (bx, by, z), (ax_, ay, z)])
    return top, walls


def scene(ax, d, frac, lw=0.25):
    """One Poly3DCollection per cell, so matplotlib depth-sorts whole prisms
    and a wall never paints over the lid of the cell in front of it."""
    me, cells = d["me"], d["cells"]
    zs = CONFIG["Z_SCALE"]
    shown = frac * len(cells)
    floor = []
    for i, poly in enumerate(cells):
        grow = np.clip(shown - i, 0, 1)            # each cell rises over one step
        if grow <= 0:
            floor.append([(px, py, 0) for px, py in poly])
            continue
        r = float(me["fwd"].iloc[i]) * 100
        z = r * zs * grow
        col = CMAP(np.clip((r - RET_LO) / (RET_HI - RET_LO), 0, 1))
        top, walls = _prism(poly, z)
        dark = tuple(np.array(col[:3]) * 0.5) + (1.0,)
        fear = float(me["VIX"].iloc[i]) >= CONFIG["FEAR"]
        ax.add_collection3d(Poly3DCollection(
            walls + [top], facecolors=[dark] * len(walls) + [col],
            edgecolor=(1, 1, 1, 0.95) if fear else (0, 0, 0, 0.6),
            linewidth=1.1 if fear else lw), autolim=False)
    if floor:
        ax.add_collection3d(Poly3DCollection(
            floor, facecolors=(0.22, 0.22, 0.22, 0.5), edgecolor=(0, 0, 0, 0.8),
            linewidth=lw), autolim=False)
    for yv in (d["fear_y"], d["calm_y"]):
        ax.plot([0, 10], [yv, yv], [0, 0], color=THEME["TEXT"], lw=0.9, alpha=0.7)
    zs_ = CONFIG["Z_SCALE"]
    ax.set_xlim(0, 10); ax.set_ylim(0, 8)
    ax.set_zlim(RET_LO * zs_, RET_HI * zs_ * 1.25)


def scene_labels(ax, d, fs=11):
    """Projected 2D text drawn above every 3D collection."""
    M = ax.get_proj()
    figs = d["figs"]

    def put(x, y, z, s, **kw):
        x2, y2, _ = proj3d.proj_transform(x, y, z, M)
        ax.text2D(x2, y2, s, transform=ax.transData, zorder=1e6, ha="center",
                  va="center", family=THEME["MONO"], **kw)

    put(10.7, d["fear_y"], 0, f"VIX {figs['fear_level']}", color=THEME["TEXT"],
        fontsize=fs, fontweight="bold")


def text_block(fig, figs, s=1.0):
    L = LAYOUT
    fig.text(0.5, L["TITLE"], "BUY WHEN IT HURTS", ha="center", fontsize=29 * s,
             fontweight="bold", color=THEME["TEXT"], family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             f"SPY one year later  |  {figs['n_days']} days  |  "
             f"sorted by the VIX on the day you bought",
             ha="center", fontsize=12.5 * s, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.5, L["KEY"],
             "one cell per month  ·  height = return next year  ·  white edge = VIX 40+",
             ha="center", fontsize=10 * s, color=THEME["TEXT_DIM"], family=THEME["FONT"])
    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10 * s,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])
    fig.text(0.5, L["BIG"],
             f"VIX {figs['fear_level']}+  →  {figs['fear_mean']} next year",
             ha="center", fontsize=22 * s, color=THEME["RED"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             f"up in {figs['fear_pos']} of cases  ·  average day {figs['all_mean']}",
             ha="center", fontsize=12 * s, color=THEME["TEXT_DIM"], family=THEME["MONO"])


def render_frame(idx, total, d, out_path):
    c, figs = CONFIG, d["figs"]
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
    text_block(fig, figs)
    ax = fig.add_axes(LAYOUT["AXES"], projection="3d", facecolor=THEME["BG"])
    scene(ax, d, frac)
    ax.view_init(elev=c["ELEV_BASE"] + 5 * np.sin(t * np.pi),
                 azim=c["AZIM_START"] + c["AZIM_SWEEP"] * t)
    ax.set_box_aspect((1.5, 1.2, 1.05), zoom=c["ZOOM"])
    style_axes(ax)
    scene_labels(ax, d)
    fig.savefig(out_path, dpi=c["DPI"], facecolor=THEME["BG"])
    plt.close(fig)


# ------------------------------------------------------------- COMPILE
def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


_D = None


def _init(d):
    global _D
    _D = d


def _job(args):
    i, total, path = args
    render_frame(i, total, _D, path)
    return i


def load():
    d = compute(fetch())
    validate(d)                       # <- before a single frame is written
    return d


def main(smoke=False):
    c = CONFIG
    d = load()
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
    jobs = [(i, total, os.path.join(frames, f"f_{i:05d}.png")) for i in range(total)]
    with Pool(c["WORKERS"], initializer=_init, initargs=(d,)) as pool:
        for n, _ in enumerate(pool.imap_unordered(_job, jobs)):
            if n % 30 == 0:
                log(f"  frame {n}/{total}  ({time.time() - t0:.0f}s)")
    assert len(os.listdir(frames)) == total, "frame count mismatch"

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
