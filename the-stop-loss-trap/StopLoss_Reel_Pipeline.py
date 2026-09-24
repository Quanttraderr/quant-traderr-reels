"""
StopLoss_Reel_Pipeline.py
=========================
Reel 13 for @quant.traderr. Every one-year holding window of SPY since 1993,
started on every trading day, run through a stop-loss barrier at -10%.

THE MATHS
    For each start day s, the path is price(s + k) / price(s) - 1 for
    k = 0 .. 252 trading days, dividends included (adjusted closes). A window
    "touched the barrier" if its path reached -10% at any point inside the
    year. For every window that touched it, we then ask where it finished on
    day 252: above or below where it started.

    In 3D: the trading day inside the window runs along the floor, the start
    date runs into depth, the return goes up. The red glass sheet is the
    barrier at -10%. All paths grow together, one trading day per step, so the
    viewer watches the barrier fill up and then sees which of those paths
    climbed back.

THE CLAIM THE VISUAL MAKES
    A -10% stop on the S&P fires in a large share of years, and about half the
    time it fires in a year that finished in profit anyway. The stop sold the
    dip and missed the recovery.

WHAT THE VISUAL IS NOT
    Not an argument against every stop. On the S&P the stop did cut the worst
    years (2008) short, and on a single stock that can go to zero a stop means
    something else entirely. The windows overlap heavily, day-by-day starts
    share almost the whole year with their neighbours, so the 8,000 windows
    are not 8,000 independent trials: this is about 33 years of one index.
    The counterfactual ignores taxes, costs and whatever the stopped-out money
    did next. Only every 14th window is drawn; every figure uses all of them.

RUN
    python StopLoss_Reel_Pipeline.py --smoke   # 3 frames -> temp_frames_smoke/
    python StopLoss_Reel_Pipeline.py           # full render -> topic.mp4

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
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from mpl_toolkits.mplot3d.art3d import Line3DCollection

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "_cache")

CONFIG = {
    # >>> REPLACE: data + maths knobs for your topic
    "TICKERS": ["SPY"],
    "PERIOD": "max",
    "WINDOW": 252,            # trading days in one holding year
    "BARRIER": -10.0,         # the stop, percent below the entry
    "DRAW_EVERY": 14,         # draw every n-th window, figures use all
    "Z_LO": -50.0,            # display range of the return axis, percent
    "Z_HI": 60.0,

    # render
    "W": 1080, "H": 1920, "DPI": 100,
    "FPS": 30,
    "HOOK_SEC": 1.4, "BUILD_SEC": 8.5, "HOLD_SEC": 1.8,
    "ELEV_BASE": 24, "AZIM_START": -64, "AZIM_SWEEP": 14,
    "ZOOM": 1.1,
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
    "TITLE": 0.862, "SUBTITLE": 0.838, "KEY": 0.816, "ROW_A": 0.799,
    "ROW_B": 0.783, "HANDLE": 0.764,
    "AXES": [-0.08, 0.29, 1.16, 0.47],
    "NOTE": 0.272, "BIG": 0.242, "SUB": 0.215,
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------- DATA
def fetch():
    """yfinance once, then a versioned CSV cache. Delete _cache/ to refetch."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "closes.csv")
    if os.path.exists(path):
        log(f"[Data] cache {path}")
        return pd.read_csv(path, index_col=0, parse_dates=True)
    import yfinance as yf
    log(f"[Data] fetching {CONFIG['TICKERS']} {CONFIG['PERIOD']}")
    px = yf.download(CONFIG["TICKERS"], period=CONFIG["PERIOD"],
                     interval="1d", progress=False, auto_adjust=True)["Close"]
    px = px[CONFIG["TICKERS"]]
    if px.empty:
        raise SystemExit("no data returned; refusing to draw a synthetic tape")
    px.to_csv(path)
    return px


# ------------------------------------------------------------- COMPUTE
def compute(px):
    # >>> REPLACE: the maths for your topic. Return whatever render_frame needs.
    s = px["SPY"].dropna()
    v = s.values
    N = CONFIG["WINDOW"]
    starts = np.arange(len(v) - N)
    idx = starts[:, None] + np.arange(N + 1)[None, :]
    paths = (v[idx] / v[starts][:, None] - 1.0) * 100.0      # (windows, N+1)
    runmin = np.minimum.accumulate(paths, axis=1)
    hit = runmin[:, -1] <= CONFIG["BARRIER"]
    end = paths[:, -1]
    return {"paths": paths, "runmin": runmin, "hit": hit, "end": end,
            "dates": s.index[starts]}


# ------------------------------------------------------------ VALIDATE
def validate(d):
    """Every number that reaches a frame is asserted here, before any render."""
    hit, end, B = d["hit"], d["end"], CONFIG["BARRIER"]
    n = len(hit)
    p_hit = float(hit.mean() * 100)
    p_up = float((end > 0).mean() * 100)
    p_saved = float((hit & (end > 0)).sum() / hit.sum() * 100)   # stopped, ended up
    p_green_dip = float((hit & (end > 0)).sum() / (end > 0).sum() * 100)
    p_hit_down = float((hit & (end <= B)).sum() / hit.sum() * 100)
    med_end_hit_up = float(np.median(end[hit & (end > 0)]))

    assert n > 6000, f"only {n} windows"
    assert 10.0 <= p_hit <= 60.0, f"{p_hit:.0f}% of windows touched the stop"
    assert 60.0 <= p_up <= 95.0, f"{p_up:.0f}% of windows ended up"
    assert 20.0 <= p_saved <= 80.0, f"{p_saved:.0f}% of stop-outs ended green"
    assert 5.0 <= p_green_dip <= 50.0, f"{p_green_dip:.0f}% of green years dipped"
    assert 0.0 < med_end_hit_up < 60.0, f"median end {med_end_hit_up:.1f}"
    # the stop must also have done its job somewhere, or the docstring lies
    assert d["end"][d["dates"].year == 2008].min() < -30.0, "2008 missing"

    figs = {
        "windows": f"{n:,}",
        "hit": f"{p_hit:.0f}%",
        "up": f"{p_up:.0f}%",
        "saved": f"{p_saved:.0f}%",
        "green_dip": f"{p_green_dip:.0f}%",
        "hit_down": f"{p_hit_down:.0f}%",
        "med_end": f"+{med_end_hit_up:.0f}%",
        "barrier": f"{B:.0f}%",
        "first_year": str(d["dates"][0].year),
        "last_year": str(d["dates"][-1].year + 1),
        "window": str(CONFIG["WINDOW"]),
        "crash": "2008",                           # asserted above: the stop's good year
    }
    with open(os.path.join(BASE_DIR, "figures.json"), "w") as fh:
        json.dump(figs, fh, indent=1)
    log(f"[validate] {figs}")

    # running counter for the build: share of windows that had touched the
    # barrier by trading day k, from the same running minimum
    run = (d["runmin"] <= B).mean(axis=0) * 100.0
    d["run"] = [f"{x:.0f}%" for x in run]
    assert d["run"][-1] == figs["hit"], "running counter disagrees with figures"
    d["figs"] = figs
    return figs


def prepare_geometry(d):
    c = CONFIG
    sel = np.arange(0, len(d["hit"]), c["DRAW_EVERY"])
    P = np.clip(d["paths"][sel], c["Z_LO"], c["Z_HI"])
    m, N1 = P.shape
    xs = np.linspace(0, 1, N1)
    ys = np.linspace(0, 1, m)
    d["P"], d["xs"], d["ys"] = P, xs, ys
    hit, end = d["hit"][sel], d["end"][sel]
    # colour is the fate of the window: never touched, touched and recovered,
    # touched and finished down
    cols = np.empty((m, 4))
    cols[:] = to_rgba(THEME["CYAN"], 0.16)
    cols[hit & (end > 0)] = to_rgba(THEME["YELLOW"], 0.85)
    cols[hit & (end <= 0)] = to_rgba(THEME["RED"], 0.85)
    d["cols"] = cols
    d["order"] = np.argsort(hit.astype(int))          # touched paths drawn last
    d["lw"] = np.where(hit, 0.7, 0.45)
    return d


# -------------------------------------------------------------- RENDER
def render_frame(idx, total, d, out_path):
    c, figs, L = CONFIG, d["figs"], LAYOUT
    if "P" not in d:
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
    N = c["WINDOW"]
    k = max(int(round(N * (0.12 + 0.88 * frac))), 2)          # trading day reached

    fig = plt.figure(figsize=(c["W"] / c["DPI"], c["H"] / c["DPI"]),
                     dpi=c["DPI"], facecolor=THEME["BG"])
    fig.text(0.5, L["TITLE"], "THE STOP-LOSS TRAP", ha="center", fontsize=28,
             fontweight="bold", color=THEME["TEXT"], family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             f"SPY  |  {figs['windows']} one-year windows  |  "
             f"{figs['first_year']} to {figs['last_year']}  |  stop at {figs['barrier']}",
             ha="center", fontsize=13, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.5, L["KEY"],
             "cyan: never touched   yellow: stopped, then ended up   red: stopped, ended down",
             ha="center", fontsize=10, color=THEME["TEXT_DIM"], family=THEME["FONT"])
    fig.text(0.5, L["ROW_A"],
             f"windows ending up {figs['up']}   touched the stop {figs['hit']}",
             ha="center", va="center", fontsize=9, color=THEME["TEXT"],
             family=THEME["MONO"])
    fig.text(0.5, L["ROW_B"],
             f"green years that hit {figs['barrier']} first: {figs['green_dip']}",
             ha="center", va="center", fontsize=9, color=THEME["YELLOW"],
             family=THEME["MONO"])
    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])

    ax = fig.add_axes(L["AXES"], projection="3d", facecolor=THEME["BG"],
                      computed_zorder=False)

    # >>> REPLACE: the geometry.
    zb = c["BARRIER"]
    bx, by = np.meshgrid([0, 1], [0, 1])
    ax.plot_surface(bx, by, np.full_like(bx, zb, dtype=float), color=THEME["RED"],
                    alpha=0.22, linewidth=0, shade=False, zorder=1)
    for g in np.linspace(0, 1, 9):
        ax.plot([0, 1], [g, g], [zb, zb], color=THEME["RED"], lw=0.6, alpha=0.55,
                zorder=1)
        ax.plot([g, g], [0, 1], [zb, zb], color=THEME["RED"], lw=0.6, alpha=0.55,
                zorder=1)
    ax.plot([0, 0], [0, 1], [0, 0], color=THEME["TEXT_DIM"], lw=0.6, alpha=0.6,
            zorder=1)                                          # entry line

    P, xs, ys = d["P"], d["xs"], d["ys"]
    o = d["order"]
    segs = [np.column_stack([xs[:k + 1], np.full(k + 1, ys[i]), P[i, :k + 1]])
            for i in o]
    ax.add_collection3d(Line3DCollection(segs, colors=d["cols"][o],
                                         linewidths=d["lw"][o], zorder=3))
    if frac < 1.0:                                             # the moving front
        ax.scatter(np.full(len(ys), xs[k]), ys, P[:, k], s=1.5,
                   color=THEME["TEXT"], alpha=0.5, depthshade=False, zorder=4)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(c["Z_LO"], c["Z_HI"])
    ax.view_init(elev=c["ELEV_BASE"] + 5 * np.sin(t * np.pi),
                 azim=c["AZIM_START"] + c["AZIM_SWEEP"] * t)
    ax.set_box_aspect((1.5, 1.25, 1.55), zoom=c["ZOOM"])
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    # Readout: strings only, precomputed and asserted in validate().
    if frac < 1.0:
        fig.text(0.5, L["NOTE"],
                 f"day {k:>3} of {figs['window']}   touched {figs['barrier']} so far: "
                 f"{d['run'][k]}",
                 ha="center", fontsize=11, color=THEME["TEXT"], family=THEME["MONO"])
    else:
        fig.text(0.5, L["NOTE"],
                 "every line is one year held: day along, start date deep, return up",
                 ha="center", fontsize=10, color=THEME["TEXT_DIM"],
                 family=THEME["FONT"])
    fig.text(0.5, L["BIG"], f"{figs['saved']} OF STOP-OUTS ENDED GREEN",
             ha="center", fontsize=19, color=THEME["YELLOW"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             f"the stop sold the dip, the year still closed up",
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
