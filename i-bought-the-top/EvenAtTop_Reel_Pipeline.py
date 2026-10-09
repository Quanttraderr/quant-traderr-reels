"""
EvenAtTop_Reel_Pipeline.py
==========================
Reel 33 for @quant.traderr, "I Bought the Top". $10 of Bitcoin every single day,
starting on the worst possible day: the record close of December 2017.

THE DATA
    BTC-USD daily closes from Yahoo, 17 Sep 2014 to Sep 2026. Real data
    throughout.

THE MATHS
    Start day = the highest close of 2017 (and, for the caption, of 2021).
    From that day on, buy $10 of BTC at every daily close:
        coins_t    = sum(10 / price_s)
        value_t    = coins_t * price_t
        deposits_t = 10 * days so far
    Underwater = value below deposits.

    In 3D: a mirrored area graph. Time runs left to right. The front wall is
    the plan's value, cut at the deposit line: green above what was paid in,
    red below it. Behind it stands a pale glass wall, the money paid in. The
    floor is a mirror, so the whole thing is reflected underneath. Heights
    are the square root of dollars, so the 2018 to 2020 stretch, when the
    sums were still small, does not vanish against the 2025 peak.

THE CLAIM THE VISUAL MAKES
    Even the buyer who started on the 2017 record close and kept buying
    through two crashes ended with several times the money they put in.

WHAT THE VISUAL IS NOT
    Hindsight on the one coin that survived. Thousands of coins also had a
    record close and then went to zero, and a daily plan into one of those
    ends at zero. Bitcoin's past cycles are no promise for the next one. No
    fees, no taxes, no spread; buying at the daily close is an idealisation.

RUN
    python EvenAtTop_Reel_Pipeline.py --smoke   # 3 frames -> temp_frames_smoke/
    python EvenAtTop_Reel_Pipeline.py           # full render -> topic.mp4

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

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "_cache")

CONFIG = {
    "START": "2014-09-17", "END": "2026-09-30",
    "TOP_YEAR": 2017, "TOP_YEAR_2": 2021,
    "DAILY": 10,
    "STEP": 7,                      # one wall segment per week of the daily plan
    "DEPTH": 0.16,                  # gap between value wall and deposit glass

    # render (leave these alone unless the topic needs landscape)
    "W": 1080, "H": 1920, "DPI": 100,
    "FPS": 30,
    "HOOK_SEC": 1.4, "BUILD_SEC": 8.5, "HOLD_SEC": 1.8,
    "ELEV_BASE": 9, "AZIM_START": -104, "AZIM_SWEEP": 26,
    "ZOOM": 1.06,
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
# profit side only: no red in it, red is kept for "below what you paid in"
GAIN = LinearSegmentedColormap.from_list(
    "gain", [THEME["CYAN"], THEME["GREEN"], THEME["YELLOW"], THEME["ORANGE"]], N=256)
GLASS = "#c9d3dc"

SAFE = {"TOP": 0.12, "BOTTOM": 0.20, "RIGHT": 0.14}

LAYOUT = {
    "TITLE": 0.862, "SUBTITLE": 0.836, "KEY": 0.814, "HANDLE": 0.792,
    "AXES": [-0.08, 0.30, 1.16, 0.485],
    "NOTE": 0.272, "BIG": 0.242, "SUB": 0.215,
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def usd(s):
    """matplotlib reads a pair of $ as maths; escape them for the frame."""
    return s.replace("$", r"\$")


def money(x):
    return f"${x / 1e3:.0f}K"


# ---------------------------------------------------------------- DATA
def fetch():
    """yfinance once, then a CSV cache. Delete _cache/ to refetch."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "btc.csv")
    if not os.path.exists(path):
        import yfinance as yf
        log("[Data] fetching BTC-USD")
        btc = yf.download("BTC-USD", start=CONFIG["START"], progress=False,
                          auto_adjust=True)["Close"].squeeze()
        if btc.empty:
            raise SystemExit("no data returned; refusing to draw a synthetic tape")
        btc.to_frame("BTC").to_csv(path)
    log(f"[Data] cache {path}")
    s = pd.read_csv(path, index_col=0, parse_dates=True)["BTC"]
    s = s.loc[CONFIG["START"]:CONFIG["END"]].dropna()
    if s.empty:
        raise SystemExit("BTC came back empty; refusing to continue")
    return s


# ------------------------------------------------------------- COMPUTE
def _plan(btc, year):
    seg = btc.loc[str(year)]
    t0 = seg.idxmax()                              # the record close of that year
    p = btc.loc[t0:]
    dep = CONFIG["DAILY"] * np.arange(1, len(p) + 1, dtype=float)
    val = np.cumsum(CONFIG["DAILY"] / p.values) * p.values
    return {"t0": t0, "px0": float(p.iloc[0]), "dates": p.index,
            "dep": dep, "val": val}


def compute(btc):
    c = CONFIG
    a = _plan(btc, c["TOP_YEAR"])
    b = _plan(btc, c["TOP_YEAR_2"])
    return {"btc": btc, "plan": a, "plan2": b}


# ------------------------------------------------------------ VALIDATE
def validate(d):
    """Every number that reaches a frame is asserted here, before any render."""
    a, b, btc = d["plan"], d["plan2"], d["btc"]
    dep, val = a["dep"][-1], a["val"][-1]
    under = a["val"] < a["dep"]
    worst = float((a["val"] / a["dep"]).min())
    last_under = a["dates"][under][-1]
    yrs = (a["dates"][-1] - a["t0"]).days / 365.25

    # the start really is the top: nothing in the window before it closed higher
    assert a["px0"] == float(btc.loc[:a["t0"]].max()), "2017 start is not a record close"
    assert b["px0"] == float(btc.loc[:b["t0"]].max()), "2021 start is not a record close"
    assert 10_000 <= a["px0"] <= 25_000, f"2017 top ${a['px0']:,.0f} outside band"
    assert 2 * dep <= val <= 30 * dep, f"value ${val:,.0f} outside band"
    assert 0.2 <= worst < 1.0, f"worst value/deposits {worst:.2f} outside band"
    assert 0.0 < under.mean() < 0.5
    assert b["val"][-1] > b["dep"][-1], "the 2021 buyer is underwater; caption claim gone"
    assert 7 <= yrs <= 10

    figs = {
        "daily": f"${CONFIG['DAILY']}",
        "top_date": a["t0"].strftime("%d %b %Y").lstrip("0"),
        "top_px": f"${a['px0']:,.0f}",
        "dep": f"${dep:,.0f}", "val": f"${val:,.0f}",
        "dep_short": money(dep), "val_short": money(val),
        "mult": f"{val / dep:.1f}x",
        "worst": f"{(worst - 1) * 100:.0f}%",
        "under_share": f"{under.mean() * 100:.0f}%",
        "last_under": last_under.strftime("%b %Y"),
        "years": f"{yrs:.1f}",
        "top2_date": b["t0"].strftime("%d %b %Y").lstrip("0"),
        "top2_px": f"${b['px0']:,.0f}",
        "dep2": f"${b['dep'][-1]:,.0f}", "val2": f"${b['val'][-1]:,.0f}",
        "mult2": f"{b['val'][-1] / b['dep'][-1]:.1f}x",
        "end": a["dates"][-1].strftime("%b %Y"),
    }
    with open(os.path.join(BASE_DIR, "figures.json"), "w") as fh:
        json.dump(figs, fh, indent=1)
    log(f"[validate] {figs}")
    d["figs"] = figs
    return figs


# -------------------------------------------------------------- RENDER
def _series(d):
    """Weekly samples of the daily plan, plus the very last day."""
    a = d["plan"]
    idx = np.r_[np.arange(0, len(a["dep"]), CONFIG["STEP"]), len(a["dep"]) - 1]
    idx = np.unique(idx)
    top = a["val"].max()
    x = np.linspace(0, 1, len(idx))
    # square root of dollars: 2018 to 2020, when the sums were small, stays visible
    return idx, x, np.sqrt(a["val"][idx] / top), np.sqrt(a["dep"][idx] / top)


def _wall(y, x, lo, hi, cols, alpha, z0, sign=1):
    quads = [[(x[k], y, sign * lo[k]), (x[k + 1], y, sign * lo[k + 1]),
              (x[k + 1], y, sign * hi[k + 1]), (x[k], y, sign * hi[k])]
             for k in range(len(x) - 1)]
    return Poly3DCollection(quads, facecolors=cols, edgecolors=cols,
                            linewidths=0.4, alpha=alpha, zorder=z0)


def scene(ax, d, frac):
    """Order by hand: mirror, glass, value wall, lines."""
    c = CONFIG
    idx, x, v, dp = _series(d)
    K = len(idx)
    n = int(round(frac * (K - 1))) + 1 if frac > 0 else 1
    ax.computed_zorder = False
    ax.plot([0, 1], [0, 0], [0, 0], color="#3a4048", lw=1.0, zorder=0)
    ax.plot([0, 1], [c["DEPTH"], c["DEPTH"]], [0, 0], color="#3a4048", lw=1.0, zorder=0)
    if n >= 2:
        xs, vv, dd = x[:n], v[:n], dp[:n]
        cols = [THEME["RED"] if vv[k] < dd[k] else GAIN(min(1.0, vv[k] ** 0.7))
                for k in range(n - 1)]
        zero = np.zeros(n)
        # mirror first, so everything above covers it
        ax.add_collection3d(_wall(0, xs, zero, 0.4 * vv, cols, 0.2, 1, sign=-1))
        ax.add_collection3d(_wall(c["DEPTH"], xs, zero, 0.4 * dd, GLASS, 0.08, 1, sign=-1))
        # deposits: pale glass behind
        ax.add_collection3d(_wall(c["DEPTH"], xs, zero, dd, GLASS, 0.30, 2))
        ax.plot(xs, np.full(n, c["DEPTH"]), dd, color="white", lw=1.4, zorder=3)
        # value: the wall in front, cut at the deposit line
        ax.add_collection3d(_wall(0, xs, zero, vv, cols, 0.97, 4))
        ax.plot(xs, np.zeros(n), vv, color="white", lw=1.6, zorder=5)
        ax.plot(xs, np.zeros(n), dd, color="black", lw=1.0, ls=(0, (2, 2)),
                alpha=0.8, zorder=5)
    ax.set_xlim(0, 1); ax.set_ylim(-0.3, 0.45); ax.set_zlim(-0.3, 1.02)
    return idx[n - 1]


def style_axes(ax):
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
        pane.line.set_color((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def scene_labels(ax, d, last, fs=11):
    """Projected to 2D so the walls cannot depth-sort the labels away."""
    from mpl_toolkits.mplot3d import proj3d
    M = ax.get_proj()
    c, figs = CONFIG, d["figs"]
    idx, x, v, dp = _series(d)

    def put(x_, y, z, s, **kw):
        x2, y2, _ = proj3d.proj_transform(x_, y, z, M)
        ax.text2D(x2, y2, s, transform=ax.transData, zorder=1e6, ha="center",
                  va="center", bbox=dict(boxstyle="round,pad=0.25", fc="black",
                                         ec="none", alpha=0.65), **kw)

    put(0.02, 0, 0.12, usd(f"START\n{figs['top_date']}\n{figs['top_px']}"),
        color=THEME["RED"], fontsize=fs, fontweight="bold", family=THEME["MONO"])
    if last == len(d["plan"]["dep"]) - 1:
        put(0.86, 0, v[-1] + 0.1, usd(f"VALUE\n{figs['val']}"), color=THEME["GREEN"],
            fontsize=fs + 4, fontweight="bold", family=THEME["MONO"])
        put(0.86, c["DEPTH"], dp[-1] + 0.08, usd(f"PAID IN {figs['dep']}"),
            color="white", fontsize=fs + 1, fontweight="bold", family=THEME["MONO"])


def text_block(fig, figs, scale=1.0):
    L, s = LAYOUT, scale
    fig.text(0.5, L["TITLE"], "I BOUGHT THE TOP", ha="center",
             fontsize=29 * s, fontweight="bold", color=THEME["TEXT"],
             family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             usd(f"{figs['daily']} of Bitcoin every day  |  from the record close of "
                 f"{figs['top_date']}"),
             ha="center", fontsize=12 * s, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.5, L["KEY"],
             "glass = money paid in  ·  wall = value  ·  red = below what you paid  ·  √ scale",
             ha="center", fontsize=10 * s, color=THEME["TEXT_DIM"], family=THEME["FONT"])
    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10 * s,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])
    fig.text(0.5, L["BIG"], usd(f"{figs['dep_short']} IN, {figs['val_short']} OUT"),
             ha="center", fontsize=22 * s, color=THEME["GREEN"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             usd(f"{figs['mult']} after buying the top  ·  worst point {figs['worst']}"),
             ha="center", fontsize=11.5 * s, color=THEME["TEXT_DIM"],
             family=THEME["MONO"])


def note(fig, d, last, scale=1.0):
    fig.text(0.5, LAYOUT["NOTE"], d["plan"]["dates"][last].strftime("%b %Y"),
             ha="center", fontsize=12 * scale, color=THEME["TEXT"],
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
    last = scene(ax, d, frac)
    note(fig, d, last)
    ax.view_init(elev=c["ELEV_BASE"] + 4 * np.sin(t * np.pi),
                 azim=c["AZIM_START"] + c["AZIM_SWEEP"] * t)
    ax.set_box_aspect((1.25, 0.5, 1.45), zoom=c["ZOOM"])
    style_axes(ax)
    scene_labels(ax, d, last)

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
