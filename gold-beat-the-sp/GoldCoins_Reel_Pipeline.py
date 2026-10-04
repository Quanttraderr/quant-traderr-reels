"""
GoldCoins_Reel_Pipeline.py
==========================
Reel 23 for @quant.traderr, "Gold Beat the S&P". One dollar in gold and one
dollar in SPY (dividends reinvested) from Aug 2000, stacked as coins.

THE DATA
    Gold: Yahoo GC=F, the front-month COMEX gold future, daily close, as a
    proxy for the gold price. It starts on 30 Aug 2000, so that is day one.
    SPY: Yahoo daily close, auto-adjusted, so dividends are reinvested.
    Both to Sep 2026, on the days both traded. Real data throughout. Nothing
    here is simulated.

THE MATHS
    value_t = close_t / close_start for each asset, the worth of $1.
    Max drawdown = the deepest fall of value_t below its running peak.
    The same ratio is also computed from the first trading day of 2010.

    In 3D: two stacks of coins, one coin = $0.50 of value, gold on the left,
    SPY on the right. The build walks month-end by month-end; a stack shows
    as many coins as the dollar is worth that month, the top coin partly
    thick, so the stacks shrink in a crash and grow back. A coin's colour is
    the year its level was first reached (blue = 2000s, red = 2020s), so the
    colour says WHEN the wealth arrived, not just how much.

THE CLAIM THE VISUAL MAKES
    From Aug 2000, a dollar in gold ended worth more than a dollar in SPY
    with every dividend reinvested.

WHAT THE VISUAL IS NOT
    Not proof that gold is the better asset. The result hangs on the start
    date: Aug 2000 was near the top of the dot-com bubble and near a 20-year
    low in gold. Start on the first day of 2010 and SPY wins by a wide margin.
    Gold paid nothing for long stretches (it fell by almost half from 2011
    and took until 2020 to recover). Futures prices skip storage, ETF fees
    and roll costs; no taxes, no inflation adjustment.

RUN
    python GoldCoins_Reel_Pipeline.py --smoke   # 3 frames
    python GoldCoins_Reel_Pipeline.py           # full render -> topic.mp4

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
    "START": "2000-08-30",
    "END": "2026-09-30",
    "SPLIT": "2010-01-01",
    "COIN": 0.5,                      # dollars of value per coin
    "PITCH": 0.21, "THICK": 0.17,     # coin spacing and thickness
    "RADIUS": 1.0, "SEG": 30,
    "XPOS": {"GOLD": -1.45, "SPY": 1.45},

    # render
    "W": 1080, "H": 1920, "DPI": 100,
    "FPS": 30,
    "HOOK_SEC": 1.4, "BUILD_SEC": 8.5, "HOLD_SEC": 1.8,
    "ELEV_BASE": 16, "AZIM_START": -72, "AZIM_SWEEP": 24,
    "ZOOM": 1.22,
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
    path = os.path.join(CACHE_DIR, "gold_spy.csv")
    if not os.path.exists(path):
        import yfinance as yf
        log("[Data] fetching GC=F and SPY")
        g = yf.download("GC=F", start=CONFIG["START"], progress=False,
                        auto_adjust=True)["Close"].squeeze()
        s = yf.download("SPY", start=CONFIG["START"], progress=False,
                        auto_adjust=True)["Close"].squeeze()
        df = pd.concat([g, s], axis=1, keys=["GOLD", "SPY"], sort=True).dropna()
        if df.empty:
            raise SystemExit("no data returned; refusing to draw a synthetic tape")
        df.to_csv(path)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.loc[CONFIG["START"]:CONFIG["END"]]


# ------------------------------------------------------------- COMPUTE
def _maxdd(v):
    peak = v.cummax()
    dd = v / peak - 1
    trough = dd.idxmin()
    top = v.loc[:trough].idxmax()
    after = v.loc[trough:]
    back = after[after >= v.loc[top]]
    return float(dd.min()), top, trough, (back.index[0] if len(back) else None)


def compute(df):
    val = df / df.iloc[0]                       # worth of $1, daily
    me = val.groupby(val.index.to_period("M")).tail(1)
    me = pd.concat([val.iloc[[0]], me])          # day one leads the build
    coin = CONFIG["COIN"]
    t0, t1 = me.index[0].year, me.index[-1].year + 1
    # colour of coin k = year its level (k+1)*coin was first reached
    first = {}
    for a in ("GOLD", "SPY"):
        n = int(np.ceil(val[a].max() / coin))
        yrs = []
        for k in range(n):
            hit = val.index[val[a].values >= (k + 1) * coin]
            yrs.append(hit[0].year + hit[0].dayofyear / 366 if len(hit) else t1)
        first[a] = [(y - t0) / (t1 - t0) for y in yrs]
    s10 = df.loc[CONFIG["SPLIT"]:]
    return {"df": df, "val": val, "me": me, "first": first,
            "x10": s10.iloc[-1] / s10.iloc[0], "d10": s10.index[0],
            "dd": {a: _maxdd(val[a]) for a in ("GOLD", "SPY")},
            "zmax": max(val.max()) / coin * CONFIG["PITCH"]}


# ------------------------------------------------------------ VALIDATE
def validate(d):
    """Every number that reaches a frame is asserted here, before any render."""
    val, me = d["val"], d["me"]
    g, s = float(val["GOLD"].iloc[-1]), float(val["SPY"].iloc[-1])
    g10, s10 = float(d["x10"]["GOLD"]), float(d["x10"]["SPY"])
    assert len(val) > 5000, f"only {len(val)} common days"
    assert val.index[0].year == 2000 and val.index[-1].year >= 2026
    assert 1 < g < 100 and 1 < s < 100, f"implausible multiples {g:.2f} {s:.2f}"
    assert g > s, "gold no longer beats SPY from 2000; claim gone"
    assert s10 > g10, "SPY no longer wins from 2010; the catch changed"
    gdd, gtop, gtr, gback = d["dd"]["GOLD"]
    sdd, stop, str_, sback = d["dd"]["SPY"]
    assert -0.8 < gdd < -0.2 and -0.8 < sdd < -0.2
    assert gback is not None and sback is not None

    figs = {
        "start": val.index[0].strftime("%b %Y"),
        "start_year": val.index[0].strftime("%Y"),
        "end": val.index[-1].strftime("%b %Y"),
        "coin": f"${CONFIG['COIN']:.2f}",
        "gold_final": f"${g:.2f}",
        "spy_final": f"${s:.2f}",
        "gold_x": f"{g:.1f}x",
        "spy_x": f"{s:.1f}x",
        "split": d["d10"].strftime("%b %Y"),
        "split_year": d["d10"].strftime("%Y"),
        "gold_x10": f"{g10:.1f}x",
        "spy_x10": f"{s10:.1f}x",
        "gold_dd": f"{gdd * 100:.0f}%",
        "gold_dd_top": gtop.strftime("%b %Y"),
        "gold_dd_low": gtr.strftime("%b %Y"),
        "gold_dd_back": gback.strftime("%b %Y"),
        "spy_dd": f"{sdd * 100:.0f}%",
        "spy_dd_low": str_.strftime("%b %Y"),
        "spy_dd_back": sback.strftime("%b %Y"),
        "n_days": f"{len(val):,}",
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


_ANG = None


def _coin(ax, cx, z0, h, col, azim):
    """One coin: shaded side wall plus a lid, as a single collection so the
    stack depth-sorts coin by coin."""
    c = CONFIG
    th = np.linspace(0, 2 * np.pi, c["SEG"] + 1)
    R = c["RADIUS"]
    xs, ys = cx + R * np.cos(th), R * np.sin(th)
    light = np.deg2rad(azim - 50)
    polys, cols = [], []
    base = np.array(col[:3])
    for i in range(c["SEG"]):
        mid = 0.5 * (th[i] + th[i + 1])
        f = 0.35 + 0.65 * max(0.0, np.cos(mid - light))
        polys.append([(xs[i], ys[i], z0), (xs[i + 1], ys[i + 1], z0),
                      (xs[i + 1], ys[i + 1], z0 + h), (xs[i], ys[i], z0 + h)])
        cols.append(tuple(base * f) + (1.0,))
    polys.append(list(zip(xs, ys, [z0 + h] * len(xs))))
    cols.append(tuple(np.minimum(base * 1.15, 1)) + (1.0,))
    ax.add_collection3d(Poly3DCollection(polys, facecolors=cols,
                                         edgecolor="none"), autolim=False)
    ax.plot(xs, ys, [z0 + h] * len(xs), color=(0, 0, 0, 0.55), lw=0.6)


def scene(ax, d, frac, azim):
    c = CONFIG
    me = d["me"]
    i = int(round(frac * (len(me) - 1)))
    row = me.iloc[i]
    tops = {}
    for a, cx in c["XPOS"].items():
        v = float(row[a]) / c["COIN"]
        full, part = int(v), v - int(v)
        firsts = d["first"][a]
        for k in range(full + (1 if part > 0.02 else 0)):
            h = c["THICK"] if k < full else c["THICK"] * part
            _coin(ax, cx, k * c["PITCH"], h, CMAP(firsts[k]), azim)
        tops[a] = (full + part) * c["PITCH"]
    th = np.linspace(0, 2 * np.pi, 61)
    for cx in c["XPOS"].values():                 # floor rings
        ax.plot(cx + 1.25 * np.cos(th), 1.25 * np.sin(th), 0, color=(1, 1, 1, 0.25), lw=0.6)
    zt = d["zmax"]
    ax.set_xlim(-3.2, 3.2); ax.set_ylim(-1.6, 1.6); ax.set_zlim(-0.9, zt * 1.05)
    return me.index[i], tops


def scene_labels(ax, d, date, tops, hold, fs=14):
    M = ax.get_proj()
    figs = d["figs"]
    xp = CONFIG["XPOS"]

    def put(x, y, z, s, **kw):
        x2, y2, _ = proj3d.proj_transform(x, y, z, M)
        ax.text2D(x2, y2, s, transform=ax.transData, zorder=1e6, ha="center",
                  va="bottom", family=THEME["MONO"], **kw)

    gold = "GOLD " + figs["gold_final"] if hold else "GOLD"
    spy = "SPY " + figs["spy_final"] if hold else "SPY"
    put(xp["GOLD"], 0, tops["GOLD"] + 0.35, gold, color=THEME["YELLOW"],
        fontsize=fs, fontweight="bold")
    put(xp["SPY"], 0, tops["SPY"] + 0.35, spy, color=THEME["CYAN"],
        fontsize=fs, fontweight="bold")


def text_block(fig, figs, s=1.0):
    L = LAYOUT
    fig.text(0.5, L["TITLE"], "GOLD BEAT THE S&P", ha="center", fontsize=29 * s,
             fontweight="bold", color=THEME["TEXT"], family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             f"$1 from {figs['start']} to {figs['end']}  |  SPY with every dividend reinvested",
             ha="center", fontsize=12.5 * s, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.5, L["KEY"],
             f"one coin = {figs['coin']}  ·  colour = year the coin was first reached",
             ha="center", fontsize=10 * s, color=THEME["TEXT_DIM"], family=THEME["FONT"])
    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10 * s,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])
    fig.text(0.5, L["BIG"],
             f"GOLD {figs['gold_x']}  vs  SPY {figs['spy_x']}",
             ha="center", fontsize=22 * s, color=THEME["YELLOW"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             f"but start in {figs['split_year']} and SPY wins: "
             f"{figs['spy_x10']} vs {figs['gold_x10']}",
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
    azim = c["AZIM_START"] + c["AZIM_SWEEP"] * t

    fig = plt.figure(figsize=(c["W"] / c["DPI"], c["H"] / c["DPI"]),
                     dpi=c["DPI"], facecolor=THEME["BG"])
    text_block(fig, figs)
    ax = fig.add_axes(LAYOUT["AXES"], projection="3d", facecolor=THEME["BG"])
    date, tops = scene(ax, d, frac, azim)
    ax.view_init(elev=c["ELEV_BASE"] + 5 * np.sin(t * np.pi), azim=azim)
    ax.set_box_aspect((6.4, 3.2, d["zmax"] * 1.05 + 0.9), zoom=c["ZOOM"])
    style_axes(ax)
    scene_labels(ax, d, date, tops, frac >= 1.0)
    fig.text(0.5, LAYOUT["NOTE"], date.strftime("%Y"), ha="center", fontsize=13,
             color=THEME["TEXT_DIM"], family=THEME["MONO"])
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
