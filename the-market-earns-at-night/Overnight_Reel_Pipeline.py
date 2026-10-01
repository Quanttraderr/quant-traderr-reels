"""
Overnight_Reel_Pipeline.py
==========================
Reel 18 for @quant.traderr, "The Market Earns at Night". SPY split into the
two halves of every trading day: the gap from yesterday's close to today's
open (night) and the move from today's open to today's close (day).

THE DATA
    SPY daily OHLC from Yahoo, auto-adjusted, 29 Jan 1993 to 30 Sep 2026.
    Open and close are scaled by the same adjustment factor on any given day,
    so dividends and splits do not leak into one half and not the other.
    Real data throughout. Nothing here is simulated.

THE MATHS
    night_t = Open_t / Close_{t-1} - 1
    day_t   = Close_t / Open_t     - 1
    (1 + night_t) * (1 + day_t) = Close_t / Close_{t-1}, so the two halves
    multiply back to buy-and-hold exactly, which VALIDATE asserts.
    Two hypothetical dollars: one held only overnight, one held only during
    market hours, compounded with no costs.

    In 3D: a voxel grid. One column per calendar month, years along the long
    axis, months across. Two blocks side by side, NIGHT and DAY. The column
    height is the value of that block's dollar at the end of the month on a
    log scale, one cube per factor of 1.25 (rounded to whole cubes). Cubes
    below the floor mean the dollar is below 1. Colour carries the same
    height, so colour and geometry say the same thing. The build fills the
    months in calendar order.

THE CLAIM THE VISUAL MAKES
    Almost all of SPY's gain since 1993 arrived while the exchange was closed.
    The dollar held only during trading hours barely moved.

WHAT THE VISUAL IS NOT
    Not a strategy. Trading in at the close and out at the open every day
    means two trades a day, roughly 17,000 round trips, and spreads plus
    commissions on that eat the edge; the dollars here pay nothing. Not a
    claim about why (overnight risk premium, news timing and the opening
    auction are all candidates; this shows none of them). Rounding to whole
    cubes makes heights approximate; the numbers in the readout are exact.
    One ETF, one country, one 33-year history.

RUN
    python Overnight_Reel_Pipeline.py --smoke   # 3 frames -> temp_frames_smoke/
    python Overnight_Reel_Pipeline.py           # full render -> topic.mp4

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

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "_cache")

CONFIG = {
    "TICKER": "SPY",
    "START": "1993-01-29",
    "END": "2026-09-30",
    "CUBE": 1.25,
    "GAP": 3,
    "DAY_LABEL_Z": -5.2,

    # render (leave these alone unless the topic needs landscape)
    "W": 1080, "H": 1920, "DPI": 100,
    "FPS": 30,
    "HOOK_SEC": 1.4, "BUILD_SEC": 8.5, "HOLD_SEC": 1.8,
    "ELEV_BASE": 24, "AZIM_START": -76, "AZIM_SWEEP": 18,
    "ZOOM": 1.12,
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


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------- DATA
def fetch():
    """yfinance once, then a CSV cache. Delete _cache/ to refetch."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "spy_ohlc.csv")
    if os.path.exists(path):
        log(f"[Data] cache {path}")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        import yfinance as yf
        log(f"[Data] fetching {CONFIG['TICKER']}")
        df = yf.download(CONFIG["TICKER"], start=CONFIG["START"], interval="1d",
                         progress=False, auto_adjust=True)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        if df.empty:
            raise SystemExit("no data returned; refusing to draw a synthetic tape")
        df.to_csv(path)
    return df.loc[CONFIG["START"]:CONFIG["END"]]


# ------------------------------------------------------------- COMPUTE
def compute(df):
    night = (df["Open"] / df["Close"].shift(1) - 1).iloc[1:]
    day = (df["Close"] / df["Open"] - 1).iloc[1:]
    hold = (df["Close"] / df["Close"].shift(1) - 1).iloc[1:]
    w_night = (1 + night).cumprod()
    w_day = (1 + day).cumprod()
    # value of each dollar at the last session of every month
    m_night = w_night.groupby(w_night.index.to_period("M")).last()
    m_day = w_day.groupby(w_day.index.to_period("M")).last()

    years = list(range(df.index[0].year, df.index[-1].year + 1))
    ny = len(years)
    lc = np.log(CONFIG["CUBE"])
    cols = []                        # (order, block, xi, yi, height) in calendar order
    for k, (p, vn) in enumerate(m_night.items()):
        xi, yi = years.index(p.year), p.month - 1
        cols.append((k, 0, xi, yi, int(np.round(np.log(vn) / lc))))
        cols.append((k, 1, xi, yi, int(np.round(np.log(m_day[p]) / lc))))
    return {"df": df, "night": night, "day": day, "hold": hold,
            "w_night": w_night, "w_day": w_day, "years": years, "ny": ny,
            "cols": cols, "n_months": len(m_night),
            "y_night": pd.DataFrame({"n": (1 + night).groupby(night.index.year).prod(),
                                     "d": (1 + day).groupby(day.index.year).prod()})}


# ------------------------------------------------------------ VALIDATE
def validate(d):
    """Every number that reaches a frame is asserted here, before any render."""
    on = float(d["w_night"].iloc[-1] - 1)
    dy = float(d["w_day"].iloc[-1] - 1)
    bh = float((1 + d["hold"]).prod() - 1)
    recon = (1 + on) * (1 + dy) - 1
    yn = d["y_night"]
    n_win = int((yn["n"] > yn["d"]).sum())

    assert abs(recon - bh) < 1e-6 * (1 + bh), f"halves {recon} do not rebuild hold {bh}"
    assert 1.0 <= on <= 200.0, f"night return {on:.2f} outside the +100%..+20000% band"
    assert -0.9 <= dy <= 5.0, f"day return {dy:.2f} outside the -90%..+500% band"
    assert on > dy, "night does not beat day; the reel's claim is gone"
    assert len(d["night"]) > 8000, f"only {len(d['night'])} sessions"
    assert d["night"].abs().max() < 0.2, "an overnight gap above 20%; check the data"
    assert 0 < n_win <= len(yn)

    figs = {
        "night_ret": f"+{on * 100:,.0f}%",
        "day_ret": f"{dy * 100:+,.0f}%",
        "hold_ret": f"+{bh * 100:,.0f}%",
        "night_usd": f"${1 + on:,.2f}",
        "day_usd": f"${1 + dy:,.2f}",
        "n_sessions": f"{len(d['night']):,}",
        "start": d["night"].index[0].strftime("%b %Y"),
        "end": d["night"].index[-1].strftime("%b %Y"),
        "years_night_won": f"{n_win} of {len(yn)}",
        "cube": f"x{CONFIG['CUBE']}",
    }
    with open(os.path.join(BASE_DIR, "figures.json"), "w") as fh:
        json.dump(figs, fh, indent=1)
    log(f"[validate] {figs}")
    d["figs"] = figs
    return figs


# -------------------------------------------------------------- RENDER
def voxel_scene(ax, d, n_cols_shown, edge_alpha=0.55, lw=0.25):
    """Fill a voxel array with the first n_cols_shown month-columns."""
    c, ny, gap = CONFIG, d["ny"], CONFIG["GAP"]
    hs = [col[4] for col in d["cols"]]
    zmin, zmax = min(min(hs), 0), max(hs)
    nz = zmax - zmin
    nyy = 12 + gap + 12
    filled = np.zeros((ny, nyy, nz), dtype=bool)
    fc = np.zeros((ny, nyy, nz, 4))
    for (_, blk, xi, yi, h) in d["cols"][:n_cols_shown]:
        yy = yi + (12 + gap if blk == 0 else 0)   # day in front, night behind
        lo, hi = (0, h) if h >= 0 else (h, 0)
        for lev in range(lo, hi):
            k = lev - zmin
            filled[xi, yy, k] = True
            if lev >= 0:
                fc[xi, yy, k] = CMAP(np.clip((lev + 0.5) / max(zmax, 1), 0, 1))
            else:
                fc[xi, yy, k] = matplotlib.colors.to_rgba(THEME["RED"], 0.85)
    X, Y, Z = np.indices((ny + 1, nyy + 1, nz + 1)).astype(float)
    Z = Z + zmin
    ec = (0, 0, 0, edge_alpha)
    if filled.any():
        ax.voxels(X, Y, Z, filled, facecolors=fc, edgecolors=ec,
                  linewidth=lw, shade=True)
    # floor outlines of both blocks, so the hook frame already shows the board
    for y0 in (0, 12 + gap):
        xs = [0, ny, ny, 0, 0]; ys = [y0, y0, y0 + 12, y0 + 12, y0]
        ax.plot(xs, ys, [0] * 5, color=THEME["TEXT_DIM"], lw=0.8, alpha=0.7)
    ax.set_xlim(0, ny); ax.set_ylim(0, nyy); ax.set_zlim(zmin, zmax)
    return zmin, zmax, nyy


def style_axes(ax):
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
        pane.line.set_color((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def scene_labels(ax, d, zmax, fs=11):
    """Labels are projected to 2D and drawn on top. A 3D text inside a voxel
    scene gets depth-sorted behind the cubes and vanishes. Call after
    view_init so the projection matches the frame."""
    from mpl_toolkits.mplot3d import proj3d
    gap, ny = CONFIG["GAP"], d["ny"]
    M = ax.get_proj()

    def put(x, y, z, s, **kw):
        x2, y2, _ = proj3d.proj_transform(x, y, z, M)
        ax.text2D(x2, y2, s, transform=ax.transData, zorder=1e6, ha="center",
                  va="center", **kw)

    put(ny * 0.40, 12 + gap + 6, zmax * 0.95, "NIGHT", color=THEME["ORANGE"],
        fontsize=fs + 4, fontweight="bold", family=THEME["FONT"])
    put(ny * 0.50, -1.0, CONFIG["DAY_LABEL_Z"], "DAY", color=THEME["CYAN"],
        fontsize=fs + 4, fontweight="bold", family=THEME["FONT"])
    for yr in (d["years"][0], d["years"][-1]):
        xi = d["years"].index(yr) + 0.5
        put(xi, -2.5, 0, str(yr), color=THEME["TEXT_DIM"], fontsize=fs,
            family=THEME["MONO"])


def text_block(fig, figs, scale=1.0):
    L, s = LAYOUT, scale
    fig.text(0.5, L["TITLE"], "THE MARKET EARNS AT NIGHT", ha="center",
             fontsize=27 * s, fontweight="bold", color=THEME["TEXT"],
             family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             f"$1 in SPY  |  {figs['n_sessions']} sessions  |  "
             f"{figs['start']} to {figs['end']}",
             ha="center", fontsize=13 * s, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.5, L["KEY"],
             f"one column per month  ·  one cube = {figs['cube']} growth  ·  red = below $1",
             ha="center", fontsize=10 * s, color=THEME["TEXT_DIM"], family=THEME["FONT"])
    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10 * s,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])
    fig.text(0.30, L["BIG"], f"NIGHT {figs['night_ret']}", ha="center",
             fontsize=21 * s, color=THEME["ORANGE"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.70, L["BIG"], f"DAY {figs['day_ret']}", ha="center",
             fontsize=21 * s, color=THEME["CYAN"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             f"held only overnight vs only in market hours  ·  no costs",
             ha="center", fontsize=11.5 * s, color=THEME["TEXT_DIM"],
             family=THEME["MONO"])


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

    n_show = int(round(frac * d["n_months"])) * 2      # night + day per month
    zmin, zmax, _ = voxel_scene(ax, d, n_show)
    ax.view_init(elev=c["ELEV_BASE"] + 5 * np.sin(t * np.pi),
                 azim=c["AZIM_START"] + c["AZIM_SWEEP"] * t)
    ax.set_box_aspect((1.35, 1.1, 1.35), zoom=c["ZOOM"])
    style_axes(ax)
    scene_labels(ax, d, zmax)

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


def main(smoke=False):
    c = CONFIG
    d = compute(fetch())
    validate(d)                       # <- before a single frame is written

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
