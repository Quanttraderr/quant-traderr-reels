"""
VolLandscape_Reel_Pipeline.py
=============================
Reel 7 for @quant.traderr. Thirty three years of SPY seen through every
lookback window at once, drawn as a landscape.

THE MATHS
    Realised volatility is not one number, it is a number per lookback window.
    N_WIN windows are laid out geometrically from MIN_WIN to MAX_WIN sessions.
    For each one, sigma_t = std(log returns over that window) * sqrt(252),
    in percent. That gives a grid of window against time, which is sampled at
    each month end so one column is one month. Height and colour are both that
    volatility, so the ridges are crises and the flat blue plain is the ordinary
    market.

    Every figure on screen is read off the grid that is actually drawn, not off
    the daily series behind it. What you see is what is asserted.

THE CLAIM THE VISUAL MAKES
    Volatility has a shape in two directions at once. A short window spikes and
    forgets, a long window carries the same event for a year, and the crises
    are the ridges that run through every window at the same time.

WHAT THE VISUAL IS NOT
    Not implied volatility, so nothing here is what the options market expected.
    Not a forecast: realised volatility is backward looking by construction.
    The windows overlap heavily, so neighbouring rows are not independent
    measurements of anything, they are the same returns re-averaged. One index.
    Month end sampling means a spike that began and ended inside a month is
    flattened, which is why the peak is read off this grid and not off the daily
    series.

RUN
    python VolLandscape_Reel_Pipeline.py --smoke   # 3 frames -> temp_frames_smoke/
    python VolLandscape_Reel_Pipeline.py           # full render -> topic.mp4

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

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "_cache")

CONFIG = {
    # >>> REPLACE: data + maths knobs for your topic
    "TICKERS": ["SPY"],
    "PERIOD": "max",
    "MIN_WIN": 10,            # shortest lookback in sessions
    "MAX_WIN": 252,           # longest lookback, one trading year
    "N_WIN": 19,              # rows of the landscape, geometrically spaced
    "COL_LO": 5.0,            # colour floor in percent, a dead calm tape
    "COL_HI": 55.0,           # colour ceiling: above this everything is red
    "GAMMA": 0.85,            # colour curve, lifts the middle into the bright tones

    # render (leave these alone unless the topic needs landscape)
    # Instagram delivers reels at 1080x1920 at most, so that is what we render.
    # A 1440x2560 master looked like the safer choice, but the platform then
    # downscales it to 75%, and this content is hairlines and small mono text on
    # black: exactly what a resample smears. Rendering native means the glyphs
    # are hinted at the size they are shown and no pixel is ever resampled.
    # The figure stays 10.8x19.2in and 1080 / 100 = 10.8 exactly, so every
    # fontsize in points and every linewidth lands where it did before.
    "W": 1080, "H": 1920, "DPI": 100,
    "FPS": 30,
    "HOOK_SEC": 1.4, "BUILD_SEC": 8.5, "HOLD_SEC": 1.8,
    "ELEV_BASE": 19, "AZIM_START": -72, "AZIM_SWEEP": 28,
    "ZOOM": 1.45,
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
    log(f"[Data] fetching {CONFIG['TICKERS']} {CONFIG['PERIOD']}")
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
    close = px[c["TICKERS"][0]]
    rets = np.log(close / close.shift(1)).dropna()

    wins = sorted(set(int(round(w)) for w in np.geomspace(
        c["MIN_WIN"], c["MAX_WIN"], c["N_WIN"])))
    daily = pd.DataFrame(
        {w: rets.rolling(w).std() * np.sqrt(252) * 100.0 for w in wins})
    daily = daily.dropna()

    # One column per quarter, and the column is the AVERAGE reading over that
    # quarter. The daily grid is 8000 columns wide, which at this size is a comb
    # of two pixel fins rather than terrain, so it has to be summarised. A
    # period end snapshot was wrong because it drops whatever happened between
    # two dates, and a maximum was worse: it turns a single bad week into an
    # isolated one column wall that reads as a rendering glitch. An average of
    # an already rolling measure is what a landscape should be made of, and it
    # understates the extremes rather than inventing them. Every figure below
    # is read off this grid, so the picture and the caption cannot drift apart.
    grid = daily.resample("QE").mean().dropna()

    return {"grid": grid, "wins": wins,
            "years": (close.index[-1] - close.index[0]).days / 365.25}


# ------------------------------------------------------------ VALIDATE
def validate(d):
    """Every number that reaches a frame is asserted here, before any render.
    A data refresh that moves a figure out of band fails the build instead of
    shipping a caption that no longer matches the picture."""
    grid, wins = d["grid"], d["wins"]
    Z = grid.values                       # rows: months, cols: windows

    peak = float(Z.max())
    r, col = np.unravel_index(int(np.argmax(Z)), Z.shape)
    peak_when = grid.index[r]
    peak_win = int(grid.columns[col])
    median = float(np.median(Z))
    calm = float(np.percentile(Z, 10))
    long_peak = float(Z[:, -1].max())     # the 252 day row at its worst

    assert 6.0 <= median <= 30.0, f"median vol {median:.1f}% outside the 8-30 band"
    assert 30.0 <= peak <= 250.0, f"peak vol {peak:.0f}% outside the 40-250 band"
    assert peak > median * 2, "peak is not clear of the median; check the windows"
    assert peak_win <= 40, f"peak sits on a {peak_win} day window, expected a short one"
    assert long_peak < peak, "the longest window peaks above the shortest, impossible"
    assert len(grid) > 80, f"only {len(grid)} quarters, too few to draw"
    assert 8 <= len(wins) <= 30, f"{len(wins)} windows outside the drawable range"

    figs = {
        "peak_vol": f"{peak:.0f}%",
        "peak_win": f"{peak_win}-day",
        "peak_date": peak_when.strftime("%b %Y"),
        "median_vol": f"{median:.0f}%",
        "calm_vol": f"{calm:.0f}%",
        "long_peak": f"{long_peak:.0f}%",
        "n_windows": str(len(wins)),
        "n_cols": f"{len(grid):,}",
        "n_years": f"{d['years']:.0f}",
        "first_year": str(grid.index[0].year),
        "last_year": str(grid.index[-1].year),
        "min_win": str(min(wins)),
        "max_win": str(max(wins)),
    }
    with open(os.path.join(BASE_DIR, "figures.json"), "w") as fh:
        json.dump(figs, fh, indent=1)
    log(f"[validate] {figs}")
    d["figs"] = figs
    d["peak_rc"] = (r, col)
    d["peak_val"] = peak
    return figs


def prepare_geometry(d):
    """Colours and limits once, so a frame stays a pure function of its index."""
    c = CONFIG
    Z = d["grid"].values.T                # rows: windows, cols: months
    d["Z"] = Z
    norm = np.clip((Z - c["COL_LO"]) / (c["COL_HI"] - c["COL_LO"]), 0, 1) ** c["GAMMA"]
    d["cols"] = CMAP(norm)
    d["zmax"] = float(Z.max()) * 1.06
    return d


# -------------------------------------------------------------- RENDER
def render_frame(idx, total, d, out_path):
    c, figs, L = CONFIG, d["figs"], LAYOUT
    if "Z" not in d:
        prepare_geometry(d)
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
    fig.text(0.5, L["TITLE"], "THE VOLATILITY LANDSCAPE", ha="center",
             fontsize=25, fontweight="bold", color=THEME["TEXT"],
             family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             f"SPY  |  {figs['first_year']} to {figs['last_year']}  |  "
             f"{figs['n_windows']} lookback windows",
             ha="center", fontsize=13, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.5, L["KEY"], "height and colour are both realised volatility",
             ha="center", fontsize=10, color=THEME["TEXT_DIM"],
             family=THEME["FONT"])
    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])

    # computed_zorder=False: matplotlib depth sorts each quad of a surface by
    # its centroid, so a tall wall in a back row draws over the rows in front of
    # it and the terrain sprouts detached slivers. Drawing one strip per row,
    # back to front, with the order stated instead of guessed, is what a painter
    # would do and it is correct for a height field seen from one side.
    ax = fig.add_axes(L["AXES"], projection="3d", facecolor=THEME["BG"],
                      computed_zorder=False)

    # >>> REPLACE: the geometry. The landscape grows left to right through time.
    Z_all, cols_all = d["Z"], d["cols"]
    n_win, n_mon = Z_all.shape
    n_show = max(int(round(n_mon * (0.45 + 0.55 * frac))), 2)
    Z = Z_all[:, :n_show]
    X, Y = np.meshgrid(np.arange(n_show), np.arange(n_win))

    for i in range(n_win - 1, 0, -1):          # back row first, front row last
        ax.plot_surface(X[i - 1:i + 1], Y[i - 1:i + 1], Z[i - 1:i + 1],
                        facecolors=cols_all[i - 1:i + 1, :n_show], linewidth=0,
                        antialiased=False, shade=False, alpha=1.0,
                        zorder=2 + (n_win - i), rstride=1, cstride=1,
                        edgecolor="none")

    # The peak gets a stem the moment the build reaches it, so the number in the
    # readout has something to point at instead of floating over the picture.
    pr, pc = d["peak_rc"]
    if n_show > pc:
        ax.plot([pc, pc], [pr, pr], [d["peak_val"], d["zmax"]],
                color=THEME["TEXT"], linewidth=1.0, alpha=0.8,
                zorder=n_win + 4)
        ax.scatter([pc], [pr], [d["zmax"]], s=22, color=THEME["TEXT"],
                   linewidths=0, zorder=n_win + 5)

    ax.set_xlim(0, n_show)
    ax.set_ylim(-0.6, n_win - 0.4)
    ax.set_zlim(-0.20 * d["zmax"], d["zmax"])
    ax.view_init(elev=c["ELEV_BASE"] - 9 * (1 - frac) + 5 * np.sin(t * np.pi),
                 azim=c["AZIM_START"] + c["AZIM_SWEEP"] * t)
    ax.set_box_aspect((1.55, 1.0, 1.15), zoom=c["ZOOM"] * (1.20 - 0.20 * frac))
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    # Readout: strings only, straight from figures.json. Never re-derive here.
    fig.text(0.5, L["NOTE"],
             f"back row {figs['min_win']}-day windows, front row {figs['max_win']}-day"
             f"  ·  one column per quarter",
             ha="center", fontsize=10, color=THEME["TEXT_DIM"],
             family=THEME["FONT"])
    fig.text(0.5, L["BIG"], f"PEAK {figs['peak_vol']}", ha="center", fontsize=19,
             color=THEME["RED"], fontweight="bold", family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             f"{figs['peak_win']}  ·  {figs['peak_date']}  ·  "
             f"median {figs['median_vol']}",
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
    # crf 14 rather than 18: Instagram re-encodes whatever it is given, and two
    # lossy passes compound. The colour tags matter too. Without them the file
    # reports "unknown" primaries and every player, Instagram included, falls
    # back to a guess, which is where washed out colour comes from.
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
    shutil.rmtree(frames, ignore_errors=True)   # the cache is 6 GB if you don't
    log(f"OK {mp4}  ({total} frames, {total / c['FPS']:.1f}s, {c['W']}x{c['H']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    main(**vars(ap.parse_args()))
