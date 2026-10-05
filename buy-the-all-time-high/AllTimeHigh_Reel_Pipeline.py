"""
AllTimeHigh_Reel_Pipeline.py
============================
Reel 24 for @quant.traderr, "Buy the All-Time High". Is a record close a bad
day to buy? Every S&P 500 all-time high since 1950, and where the index stood
one year later.

THE DATA
    S&P 500 index daily closes (^GSPC) from Yahoo, Dec 1927 to Sep 2026.
    This is the PRICE index: dividends are not included, and nothing is
    adjusted for inflation. Real data throughout. Nothing here is simulated.

THE MATHS
    A day t is an all-time high when its close is above every earlier close
    since Dec 1927 (so the 1929 peak counts, and the first new high after it
    only comes in 1954). For every day from 1950 with a full year ahead:
        r_t = close_{t+252} / close_t - 1
    The share of all-time-high days with r_t > 0 is compared with the same
    share over ALL days in the window.

    In 3D: a forest. One pillar per all-time high, x = year, y = trading day
    inside that year, height = the return over the next 252 sessions. Gains
    run blue to orange, red pillars hang below the floor: those highs were
    lower a year later. Years without a new high are clearings.

THE CLAIM THE VISUAL MAKES
    Buying at a record was about as good as buying on any day: 73% of
    all-time highs were higher a year later, against 75% of all days.

WHAT THE VISUAL IS NOT
    Not proof that highs are a buy signal: they are not better than a random
    day, only not worse. Highs cluster in runs, so the 1,310 pillars are far
    fewer independent bets than they look, and the worst cluster (2007) sits
    right before a crash. Price only, one index, one country, 75 years that
    went well.

RUN
    python AllTimeHigh_Reel_Pipeline.py --smoke   # 3 frames
    python AllTimeHigh_Reel_Pipeline.py           # full render -> topic.mp4

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
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from mpl_toolkits.mplot3d import proj3d

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "_cache")

CONFIG = {
    "HISTORY": "1927-12-30",
    "START": "1950-01-01",
    "END": "2026-09-30",
    "H": 252,                        # one year of sessions ahead
    "ZCAP": 0.55,                    # pillar height cap (+-55%), labels stay true

    # render
    "W": 1080, "H_PX": 1920, "DPI": 100,
    "FPS": 30,
    "HOOK_SEC": 1.4, "BUILD_SEC": 8.5, "HOLD_SEC": 1.8,
    "ELEV_BASE": 20, "AZIM_START": -62, "AZIM_SWEEP": 24,
    "ZOOM": 1.26,
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
# gains only: blue -> orange, red is reserved for losses
GAIN = LinearSegmentedColormap.from_list(
    "gain", [THEME["BLUE"], THEME["CYAN"], THEME["GREEN"],
             THEME["YELLOW"], THEME["ORANGE"]], N=256)

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
    path = os.path.join(CACHE_DIR, "gspc.csv")
    if not os.path.exists(path):
        import yfinance as yf
        log("[Data] fetching ^GSPC")
        s = yf.download("^GSPC", start=CONFIG["HISTORY"], progress=False,
                        auto_adjust=True)["Close"].squeeze()
        if s.empty:
            raise SystemExit("no data returned; refusing to draw a synthetic tape")
        s.to_frame("GSPC").to_csv(path)
    s = pd.read_csv(path, index_col=0, parse_dates=True)["GSPC"]
    return s.loc[CONFIG["HISTORY"]:CONFIG["END"]].dropna()


# ------------------------------------------------------------- COMPUTE
def compute(px):
    prev_max = px.shift(1).cummax()
    is_ath = px > prev_max
    fwd = (px.shift(-CONFIG["H"]) / px - 1).loc[CONFIG["START"]:].dropna()
    ath = is_ath.reindex(fwd.index)
    r_ath = fwd[ath]
    idx = r_ath.index
    years = idx.year.values
    # trading-day position inside its own year, 0..~252
    day_in_year = px.loc[CONFIG["START"]:].groupby(px.loc[CONFIG["START"]:].index.year).cumcount()
    yd = day_in_year.reindex(idx).values.astype(float)
    return {"px": px, "fwd": fwd, "r_ath": r_ath, "x": years.astype(float),
            "y": yd, "z": np.clip(r_ath.values, -CONFIG["ZCAP"], CONFIG["ZCAP"]),
            "x0": float(years.min()), "x1": float(years.max())}


# ------------------------------------------------------------ VALIDATE
def validate(d):
    """Every number that reaches a frame is asserted here, before any render."""
    px, fwd, r = d["px"], d["fwd"], d["r_ath"]
    assert len(px) > 20000, f"only {len(px)} closes, ^GSPC history looks cut"
    assert px.index[0].year <= 1928, "history must reach back to the 1920s"
    assert 500 < len(r) < 3000, f"{len(r)} all-time highs with a year ahead, implausible"
    p_ath = float((r > 0).mean())
    p_all = float((fwd > 0).mean())
    assert 0.5 < p_ath < 0.95, f"share up after a high {p_ath:.3f} implausible"
    assert 0.5 < p_all < 0.95, f"share up after any day {p_all:.3f} implausible"
    assert abs(p_ath - p_all) < 0.06, \
        f"highs {p_ath:.3f} vs all {p_all:.3f} no longer 'about the same'; claim gone"
    worst_d = r.idxmin()
    assert worst_d.year == 2007, f"worst high now {worst_d:%b %Y}; caption names 2007"
    assert r.min() < -0.2, "worst high no longer a deep loss"
    assert d["x0"] == 1954, f"first new high after 1929 is {d['x0']}, caption says 1954"
    assert np.isfinite(d["z"]).all() and np.isfinite(d["y"]).all()

    yrs = pd.Series(r.index.year)
    no_high = sorted(set(range(1954, int(d["x1"]) + 1)) - set(yrs))
    figs = {
        "share_up_ath": f"{p_ath * 100:.0f}%",
        "share_up_all": f"{p_all * 100:.0f}%",
        "share_down_ath": f"{(1 - p_ath) * 100:.0f}%",
        "n_ath": f"{len(r):,}",
        "n_days": f"{len(fwd):,}",
        "median_ath": f"+{r.median() * 100:.0f}%",
        "median_all": f"+{fwd.median() * 100:.0f}%",
        "worst": f"{r.min() * 100:.0f}%",
        "worst_date": worst_d.strftime("%b %Y"),
        "first_ath": "1954",
        "start": "1950",
        "last_ath": r.index[-1].strftime("%b %Y"),
        "end": px.index[-1].strftime("%b %Y"),
        "years_no_high": f"{len(no_high)}",
        "horizon": "1 year, 12 months, 252 sessions",
    }
    assert figs["median_ath"].startswith("+") and r.median() > 0
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


def _cut(d, frac):
    """The forest grows left to right: the year up to which pillars stand."""
    return d["x0"] - 1 + frac * (d["x1"] - d["x0"] + 2.5)


def scene(ax, d, frac, lw=1.4):
    x0, x1 = d["x0"] - 4, d["x1"] + 1
    zc = CONFIG["ZCAP"]
    # floor and decade lines
    ax.add_collection3d(Poly3DCollection(
        [[(x0, 0, 0), (x1, 0, 0), (x1, 253, 0), (x0, 253, 0)]],
        facecolors=(1, 1, 1, 0.05), edgecolor=(1, 1, 1, 0.2), linewidth=0.6),
        autolim=False)
    for yr in range(1960, int(x1) + 1, 10):
        ax.plot([yr, yr], [0, 253], [0, 0], color=(1, 1, 1, 0.13), lw=0.6)

    cut = _cut(d, frac)
    grow = np.clip((cut - d["x"]) / 2.0, 0, 1)           # each year rises in ~2 years of sweep
    m = grow > 0
    if m.any():
        x, y, z, g = d["x"][m], d["y"][m], d["z"][m], grow[m]
        zt = z * g
        cols = np.array([GAIN(min(v / zc, 1.0)) if v >= 0 else
                         matplotlib.colors.to_rgba(THEME["RED"]) for v in z])
        segs = [[(a, b, 0), (a, b, c)] for a, b, c in zip(x, y, zt)]
        glow = cols.copy(); glow[:, 3] = 0.18
        ax.add_collection3d(Line3DCollection(segs, colors=glow, linewidths=lw * 3.2),
                            autolim=False)
        ax.add_collection3d(Line3DCollection(segs, colors=cols, linewidths=lw),
                            autolim=False)
        ax.scatter(x, y, zt, s=5, c=cols, depthshade=False, linewidths=0)
    ax.set_xlim(x0, x1)
    ax.set_ylim(0, 253)
    ax.set_zlim(-zc, zc)


def scene_labels(ax, d, frac, fs=12):
    """Projected 2D text drawn above every 3D collection."""
    M = ax.get_proj()
    figs = d["figs"]

    def put(x, y, z, s, **kw):
        x2, y2, _ = proj3d.proj_transform(x, y, z, M)
        ax.text2D(x2, y2, s, transform=ax.transData, zorder=1e6,
                  va="center", family=THEME["MONO"], **kw)

    cut = _cut(d, frac)
    for yr in range(1960, int(d["x1"]) + 1, 10):
        if yr <= cut:
            put(yr, -8, 0, str(yr), ha="center", color=THEME["TEXT_DIM"], fontsize=fs * 0.85)
    r = d["r_ath"]
    wd = r.idxmin()
    if cut >= wd.year + 2:
        i = int(np.argmin(r.values))
        put(d["x"][i], d["y"][i], d["z"][i] - 0.07,
            f"{figs['worst_date']}: {figs['worst']}", ha="center", color=THEME["TEXT"],
            fontsize=fs, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc=THEME["RED"], ec="none", alpha=0.9))


def text_block(fig, figs, s=1.0):
    L = LAYOUT
    fig.text(0.5, L["TITLE"], "BUY THE ALL-TIME HIGH?", ha="center", fontsize=29 * s,
             fontweight="bold", color=THEME["TEXT"], family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             f"S&P 500 since {figs['start']}  |  {figs['n_ath']} record closes  |  "
             f"one year later",
             ha="center", fontsize=12.5 * s, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.5, L["KEY"],
             "one pillar per all-time high  ·  height = next 12 months  ·  red = lower",
             ha="center", fontsize=10 * s, color=THEME["TEXT_DIM"], family=THEME["FONT"])
    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10 * s,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])
    fig.text(0.5, L["BIG"], f"{figs['share_up_ath']} higher a year later",
             ha="center", fontsize=24 * s, color=THEME["GREEN"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             f"any random day: {figs['share_up_all']}  ·  price only",
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
    frac = 0.12 + 0.88 * frac                     # the hook already shows the first grove
    t = idx / max(total - 1, 1)

    fig = plt.figure(figsize=(c["W"] / c["DPI"], c["H_PX"] / c["DPI"]),
                     dpi=c["DPI"], facecolor=THEME["BG"])
    text_block(fig, figs)
    ax = fig.add_axes(LAYOUT["AXES"], projection="3d", facecolor=THEME["BG"])
    scene(ax, d, frac)
    ax.view_init(elev=c["ELEV_BASE"] + 5 * np.sin(t * np.pi),
                 azim=c["AZIM_START"] + c["AZIM_SWEEP"] * t)
    ax.set_box_aspect((1.7, 0.9, 1.55), zoom=c["ZOOM"])
    style_axes(ax)
    scene_labels(ax, d, frac)
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
    log(f"OK {mp4}  ({total} frames, {total / c['FPS']:.1f}s, {c['W']}x{c['H_PX']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    main(**vars(ap.parse_args()))
