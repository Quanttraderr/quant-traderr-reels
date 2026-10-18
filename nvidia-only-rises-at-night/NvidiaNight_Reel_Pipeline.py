"""
NvidiaNight_Reel_Pipeline.py
============================
Reel 36 for @quant.traderr, "Nvidia Only Rises at Night". The habit it takes
apart: watching the chart all day, as if the money were made while the market
is open. This reel splits every NVDA trading day since the IPO into two
sessions and lets $1 ride each one on its own.

THE DATA
    NVDA daily Open and Close from Yahoo, auto-adjusted for splits and
    dividends, 22 Jan 1999 (first trading day) to the latest close. Fetched
    once and kept in _cache/. Real data throughout. Nothing is simulated.

THE MATHS
    day_t    = prod over days s <= t of  Close_s / Open_s          (9:30 to 4:00)
    night_t  = prod over days s <= t of  Open_s / Close_(s-1)      (4:00 to 9:30)
    hold_t   = day_t * night_t  =  Close_t / Open_first
    $1 in each. The two sessions multiply back to buy and hold exactly.

    In 3D: two pipes running through time, their height the log10 of the
    dollar value. The day pipe is red, the losing side; the night pipe is
    coloured by its own height. Faint shelves on the back wall mark $0.10,
    $1, $10, $100, $1k and $10k.

THE CLAIM THE VISUAL MAKES
    For Nvidia since 1999, the market-hours session on its own lost money and
    the whole gain of the stock arrived in the overnight gaps.

WHAT THE VISUAL IS NOT
    Not a strategy. Buying every close and selling every open costs a spread
    and a commission twice a day, about 7,000 times; that eats the edge. The
    night return also carries the earnings gaps and the news risk that day
    traders avoid. It is one stock, the best performer of its era, so the
    sizes are extreme; the same split on SPY is far milder. The opening print
    from Yahoo is the official open, not a price anyone was guaranteed to get.

RUN
    python NvidiaNight_Reel_Pipeline.py --smoke   # 3 frames -> temp_frames_smoke/
    python NvidiaNight_Reel_Pipeline.py           # full render -> topic.mp4

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
matplotlib.rcParams["text.parse_math"] = False   # "$12,338" is money, not maths
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "_cache")

CONFIG = {
    "TICKER": "NVDA",
    "STEP": 8,                 # trading days per pipe ring
    "SIDES": 14,
    "ROW_Y": {"DAY": -0.75, "NIGHT": 0.75},
    "R_Y": 0.34, "R_Z": 0.17,  # pipe radius in y and in log10 units
    "ZMIN": -1.6, "ZMAX": 4.7,

    # render (leave these alone unless the topic needs landscape)
    "W": 1080, "H": 1920, "DPI": 100,
    "FPS": 30,
    "HOOK_SEC": 1.4, "BUILD_SEC": 8.5, "HOLD_SEC": 1.8,
    "ELEV_BASE": 14, "AZIM_START": -64, "AZIM_SWEEP": 16,
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
NIGHT_CMAP = LinearSegmentedColormap.from_list(
    "night", [THEME["BLUE"], THEME["CYAN"], THEME["GREEN"], THEME["YELLOW"]], N=256)

SAFE = {"TOP": 0.12, "BOTTOM": 0.20, "RIGHT": 0.14}

LAYOUT = {
    "TITLE": 0.862, "SUBTITLE": 0.836, "KEY": 0.814, "HANDLE": 0.792,
    "AXES": [-0.08, 0.29, 1.16, 0.49],
    "NOTE": 0.272, "BIG": 0.242, "SUB": 0.215,
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------- DATA
def fetch():
    """Yahoo once, then the CSV cache. Delete _cache/ to refetch.
    A failed fetch stops the run; nothing synthetic is ever drawn."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    p = os.path.join(CACHE_DIR, "nvda.csv")
    if not os.path.exists(p):
        import yfinance as yf
        log(f"[Data] fetching {CONFIG['TICKER']}")
        df = yf.download(CONFIG["TICKER"], period="max", interval="1d",
                         progress=False, auto_adjust=True)
        if df.empty:
            raise SystemExit("no data returned; refusing to draw a synthetic tape")
        out = pd.DataFrame({"Open": df["Open"].squeeze(), "Close": df["Close"].squeeze()})
        out.to_csv(p)
    log(f"[Data] cache {p}")
    df = pd.read_csv(p, index_col=0, parse_dates=True).dropna()
    assert (df["Open"] > 0).all() and (df["Close"] > 0).all()
    assert (df["Open"] == df["Close"]).mean() < 0.05, "opens look like copied closes"
    return df


# ------------------------------------------------------------- COMPUTE
def compute(df):
    o, c = df["Open"].to_numpy(), df["Close"].to_numpy()
    day = np.cumprod(c / o)
    night = np.cumprod(np.r_[1.0, o[1:] / c[:-1]])
    hold = c / o[0]
    assert np.allclose(day * night, hold, rtol=1e-9)
    idx = df.index
    yrs = idx.year + (idx.dayofyear - 1) / 365.25
    keep = np.r_[np.arange(0, len(df), CONFIG["STEP"]), len(df) - 1]
    keep = np.unique(keep)
    return {"dates": idx, "x": np.asarray(yrs)[keep], "dates_k": idx[keep],
            "day": day[keep], "night": night[keep],
            "day_end": float(day[-1]), "night_end": float(night[-1]),
            "hold_end": float(hold[-1]), "day_min": float(day.min()),
            "day_max": float(day.max()), "n_days": len(df)}


# ------------------------------------------------------------ VALIDATE
def validate(d):
    """Every number that reaches a frame is asserted here, before any render."""
    assert d["n_days"] > 6000
    assert 0.05 <= d["day_end"] <= 3, f"day ${d['day_end']:.2f} outside 0.05..3"
    assert 500 <= d["night_end"] <= 30000, f"night ${d['night_end']:,.0f} outside band"
    assert d["night_end"] > 100 * d["day_end"], "the split is not dramatic any more"
    assert 0.01 <= d["day_min"] < 1
    lo, hi = np.log10(min(d["day"].min(), d["night"].min())), np.log10(d["night"].max())
    assert lo > CONFIG["ZMIN"] + 0.25 and hi < CONFIG["ZMAX"] - 0.25, "pipes leave the box"

    figs = {
        "stake": "$1",
        "ticker": "NVDA",
        "day": f"${d['day_end']:.2f}",
        "night": f"${d['night_end']:,.0f}",
        "hold": f"${d['hold_end']:,.0f}",
        "day_loss": f"{(d['day_end'] - 1) * 100:.0f}%",
        "day_min": f"${d['day_min']:.2f}",
        "n_days": f"{d['n_days']:,}",
        "hours": "9:30 to 4:00",
        "start": d["dates"][0].strftime("%b %Y"),
        "end": d["dates"][-1].strftime("%b %Y"),
    }
    with open(os.path.join(BASE_DIR, "figures.json"), "w") as fh:
        json.dump(figs, fh, indent=1)
    log(f"[validate] {figs}")
    d["figs"] = figs
    return figs


# -------------------------------------------------------------- RENDER
def _shade(c, k):
    c = np.array(to_rgba(c)); c[:3] = np.clip(c[:3] * k, 0, 1); return c


def _pipe(polys, cols, x, z, y0, col_fn):
    """A tube along x around the centre line (x, y0, z); quads between rings."""
    c = CONFIG
    th = np.linspace(0, 2 * np.pi, c["SIDES"] + 1)
    cy, sz = c["R_Y"] * np.cos(th), c["R_Z"] * np.sin(th)
    # light from the camera side (front, slightly above)
    light = 0.55 + 0.6 * np.clip(-np.cos(th - 0.35 * np.pi) * 0 + np.sin(th + 0.25 * np.pi), -1, 1) * 0.5
    for i in range(len(x) - 1):
        base = col_fn(i)
        for j in range(c["SIDES"]):
            polys.append([(x[i], y0 + cy[j], z[i] + sz[j]),
                          (x[i + 1], y0 + cy[j], z[i + 1] + sz[j]),
                          (x[i + 1], y0 + cy[j + 1], z[i + 1] + sz[j + 1]),
                          (x[i], y0 + cy[j + 1], z[i] + sz[j + 1])])
            cols.append(_shade(base, light[j] + 0.35))


def scene(ax, d, frac):
    c = CONFIG
    n_all = len(d["x"])
    n = n_all if frac >= 1 else max(2, int(np.floor(frac * n_all)))
    x = d["x"][:n]
    zd, zn = np.log10(d["day"][:n]), np.log10(d["night"][:n])
    ztop = np.log10(d["night"].max())
    polys, cols = [], []
    _pipe(polys, cols, x, zn, c["ROW_Y"]["NIGHT"],
          lambda i: NIGHT_CMAP(0.08 + 0.92 * max(zn[i], 0) / ztop))
    _pipe(polys, cols, x, zd, c["ROW_Y"]["DAY"], lambda i: THEME["RED"])
    ax.add_collection3d(Poly3DCollection(polys, facecolors=cols, edgecolors="none",
                                         linewidths=0), autolim=False)
    x0, x1 = d["x"][0] - 0.5, d["x"][-1] + 0.5
    # shelves on the back wall: $0.10 ... $10k
    for e in range(-1, 5):
        ax.plot([x0, x1], [1.6, 1.6], [e, e], color=THEME["TEXT_DIM"],
                lw=0.9 if e == 0 else 0.5, alpha=0.6 if e == 0 else 0.3)
    # floor shadows under the pipes
    for y in c["ROW_Y"].values():
        ax.plot([x0, x1], [y, y], [c["ZMIN"], c["ZMIN"]], color=THEME["TEXT_DIM"],
                lw=0.6, alpha=0.4)
    ax.set_xlim(x0, x1); ax.set_ylim(-1.6, 1.6); ax.set_zlim(c["ZMIN"], c["ZMAX"])
    return n - 1


def style_axes(ax):
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
        pane.line.set_color((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


SHELF = {-1: "$0.10", 0: "$1", 1: "$10", 2: "$100", 3: "$1k", 4: "$10k"}


def scene_labels(ax, d, last, fs=11):
    """Projected to 2D so the pipes cannot depth-sort the labels away."""
    from mpl_toolkits.mplot3d import proj3d
    M = ax.get_proj()
    c, figs = CONFIG, d["figs"]

    def put(x, y, z, s, **kw):
        x2, y2, _ = proj3d.proj_transform(x, y, z, M)
        ax.text2D(x2, y2, s, transform=ax.transData, zorder=1e6, ha="center",
                  va="center", **kw)

    x0 = d["x"][0]
    for e, s in SHELF.items():
        put(x0 - 3.2, 1.6, e, s, color=THEME["TEXT_DIM"], fontsize=fs - 1,
            family=THEME["MONO"], bbox=dict(facecolor="black", alpha=0.85,
                                            edgecolor="none", pad=1.5))
    for yr in (2000, 2010, 2020):
        put(yr, c["ROW_Y"]["DAY"] - 0.9, c["ZMIN"], str(yr),
            color=THEME["TEXT_DIM"], fontsize=fs - 1, family=THEME["MONO"])
    done = last == len(d["x"]) - 1
    xl = d["x"][last]
    zn, zd = np.log10(d["night"][last]), np.log10(d["day"][last])
    put(xl - 2.2, c["ROW_Y"]["NIGHT"], zn + 0.55,
        f"NIGHT\n{figs['night']}" if done else "NIGHT",
        color=THEME["YELLOW"], fontsize=fs + 4, fontweight="bold", family=THEME["MONO"])
    put(xl - 2.2, c["ROW_Y"]["DAY"], zd + 0.65,
        f"DAY\n{figs['day']}" if done else "DAY",
        color=THEME["RED"], fontsize=fs + 4, fontweight="bold", family=THEME["MONO"],
        bbox=dict(facecolor="black", alpha=0.8, edgecolor=THEME["RED"], lw=1.0,
                  boxstyle="round,pad=0.35"))


def text_block(fig, figs, scale=1.0):
    L, s = LAYOUT, scale
    fig.text(0.5, L["TITLE"], "NVIDIA ONLY RISES AT NIGHT", ha="center",
             fontsize=27 * s, fontweight="bold", color=THEME["TEXT"],
             family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             f"{figs['stake']} in {figs['ticker']}, held only {figs['hours']} or only overnight",
             ha="center", fontsize=12.5 * s, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.5, L["KEY"],
             f"{figs['start']} to {figs['end']}  ·  log scale  ·  red = market hours",
             ha="center", fontsize=10 * s, color=THEME["TEXT_DIM"], family=THEME["FONT"])
    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10 * s,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])
    fig.text(0.5, L["BIG"], f"NIGHT {figs['night']}  ·  DAY {figs['day']}",
             ha="center", fontsize=23 * s, color=THEME["YELLOW"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             f"buy and hold {figs['hold']}  ·  {figs['n_days']} trading days  ·  no costs",
             ha="center", fontsize=11.5 * s, color=THEME["TEXT_DIM"],
             family=THEME["MONO"])


def note(fig, d, last, scale=1.0):
    if last is None:
        return
    fig.text(0.5, LAYOUT["NOTE"], d["dates_k"][last].strftime("%b %Y").upper(),
             ha="center", fontsize=12 * scale, color=THEME["TEXT"],
             family=THEME["MONO"])


def render_frame(idx, total, d, out_path):
    c, figs = CONFIG, d["figs"]
    n_hook = int(c["FPS"] * c["HOOK_SEC"])
    n_build = int(c["FPS"] * c["BUILD_SEC"])
    if idx < n_hook:
        frac = 0.0
    elif idx < n_hook + n_build:
        frac = (idx - n_hook + 1) / n_build
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
    ax.set_box_aspect((1.6, 0.5, 1.5), zoom=c["ZOOM"])
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
