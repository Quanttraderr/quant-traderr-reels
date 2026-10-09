"""
TenYearWait_Reel_Pipeline.py
============================
Reel 32 for @quant.traderr, "The Cost of Waiting". The same $500 a month into
SPY, started in different years. Every year of waiting looks cheap on the
deposit slip and costs a fortune at the end.

THE DATA
    SPY daily closes from Yahoo, auto-adjusted (dividends reinvested),
    29 Jan 1993 to Sep 2026. Real data throughout.

THE MATHS
    For each start year Y (February 1993, 1995, ... 2017):
        on the first trading day of every month from Feb Y on, buy $500 of SPY
        shares_t   = sum(500 / price_buy)
        value_t    = shares_t * price_t      (month-end close, last = last close)
        deposits_t = 500 * number of buys so far
    The headline compares the Feb 1993 start with the Feb 2003 start: the same
    plan, ten years later.

    In 3D: a waterfall of staggered area graphs, one slice per start year.
    Calendar time runs left to right, the earliest start sits at the back,
    the latest at the front. Each slice is cut in two: the grey base is the
    money paid in, the coloured cap on top is what the market added. The
    2003 slice is red, the ten-year wait. Dark red means a plan sat below
    its own deposits.

THE CLAIM THE VISUAL MAKES
    Waiting ten years saved about $60,000 of deposits and cost about a million
    dollars of ending value.

WHAT THE VISUAL IS NOT
    Not a forecast: 1993 to 2026 was a strong stretch for US stocks, and a
    different window gives different numbers. No taxes, no fees, no
    inflation adjustment, and $500 in 1993 was a bigger bite of a paycheck
    than $500 in 2003. One index, one country.

RUN
    python TenYearWait_Reel_Pipeline.py --smoke   # 3 frames -> temp_frames_smoke/
    python TenYearWait_Reel_Pipeline.py           # full render -> topic.mp4

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
    "TICKERS": ["SPY", "^IRX"],
    "PERIOD": "max",
    "START_YEARS": list(range(1993, 2018, 2)),
    "EARLY": 1993, "LATE": 2003,
    "MONTHLY": 500,

    # render (leave these alone unless the topic needs landscape)
    "W": 1080, "H": 1920, "DPI": 100,
    "FPS": 30,
    "HOOK_SEC": 1.4, "BUILD_SEC": 8.5, "HOLD_SEC": 1.8,
    "ELEV_BASE": 13, "AZIM_START": -106, "AZIM_SWEEP": 24,
    "ZOOM": 1.14,
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
# growth only: no red in it, red is kept for "below your own deposits"
GROW = LinearSegmentedColormap.from_list(
    "grow", [THEME["BLUE"], THEME["CYAN"], THEME["GREEN"], THEME["YELLOW"],
             THEME["ORANGE"]], N=256)
DEPOSIT = "#5d6670"
UNDER = "#8b1028"      # below the money paid in

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
    """$1.81M / $795K, short enough for a thumbnail."""
    return f"${x / 1e6:.2f}M" if x >= 1e6 else f"${x / 1e3:.0f}K"


# ---------------------------------------------------------------- DATA
def fetch():
    """yfinance once, then a CSV cache. Delete _cache/ to refetch."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "closes.csv")
    if os.path.exists(path):
        log(f"[Data] cache {path}")
        px = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        import yfinance as yf
        log(f"[Data] fetching {CONFIG['TICKERS']}")
        px = yf.download(CONFIG["TICKERS"], period=CONFIG["PERIOD"],
                         interval="1d", progress=False, auto_adjust=True)["Close"]
        if px.empty or "SPY" not in px.columns:
            raise SystemExit("no data returned; refusing to draw a synthetic tape")
        px = px[CONFIG["TICKERS"]]
        px.to_csv(path)
    s = px["SPY"].dropna()
    if s.empty:
        raise SystemExit("SPY came back empty; refusing to continue")
    return s


# ------------------------------------------------------------- COMPUTE
def compute(s):
    c = CONFIG
    per = s.index.to_period("M")
    buy = s.groupby(per).first()                   # first trading day of the month
    mend = s.groupby(per).last()                   # month-end close for the path
    months = mend.index[1:]                        # Feb 1993 on: Jan 1993 is one day
    plans = {}
    for Y in c["START_YEARS"]:
        p0 = pd.Period(f"{Y}-02", "M")
        on = months >= p0
        sh = np.cumsum(np.where(on, c["MONTHLY"] / buy.reindex(months).values, 0.0))
        dep = np.cumsum(np.where(on, c["MONTHLY"], 0.0))
        val = sh * mend.reindex(months).values
        plans[Y] = {"val": val, "dep": dep, "on": on}
    return {"s": s, "months": months, "plans": plans}


# ------------------------------------------------------------ VALIDATE
def validate(d):
    """Every number that reaches a frame is asserted here, before any render."""
    c = CONFIG
    e, l = d["plans"][c["EARLY"]], d["plans"][c["LATE"]]
    ev, ed, lv, ld = e["val"][-1], e["dep"][-1], l["val"][-1], l["dep"][-1]
    cost = ev - lv
    yrs = (d["months"][-1] - pd.Period(f"{c['EARLY']}-02", "M")).n / 12

    assert 150_000 <= ed <= 260_000, f"early deposits ${ed:,.0f} outside band"
    assert ld < ed and ed - ld == 120 * c["MONTHLY"], "deposit gap is not 10 years"
    assert 3 * ed <= ev <= 30 * ed, f"early value ${ev:,.0f} outside band"
    assert ld <= lv <= 20 * ld, f"late value ${lv:,.0f} outside band"
    assert cost > 5 * (ed - ld), "waiting did not cost more than it saved; claim gone"
    finals = [d["plans"][Y]["val"][-1] for Y in c["START_YEARS"]]
    assert all(a > b for a, b in zip(finals, finals[1:])), "later start ended higher"
    assert 30 <= yrs <= 40

    figs = {
        "monthly": f"${c['MONTHLY']}",
        "early": str(c["EARLY"]), "late": str(c["LATE"]),
        "early_dep": f"${ed:,.0f}", "late_dep": f"${ld:,.0f}",
        "early_val": f"${ev:,.0f}", "late_val": f"${lv:,.0f}",
        "early_short": money(ev), "late_short": money(lv),
        "dep_gap": f"${ed - ld:,.0f}",
        "cost": f"${cost:,.0f}", "cost_short": money(cost),
        "ratio": f"{ev / lv:.1f}x",
        "early_mult": f"{ev / ed:.1f}x", "late_mult": f"{lv / ld:.1f}x",
        "per_year": f"${cost / 10:,.0f}",
        "wait_years": str(c["LATE"] - c["EARLY"]),
        "n_starts": str(len(c["START_YEARS"])),
        "first_start": f"Feb {c['START_YEARS'][0]}",
        "last_start": f"Feb {c['START_YEARS'][-1]}",
        "end": d["s"].index[-1].strftime("%b %Y"),
    }
    with open(os.path.join(BASE_DIR, "figures.json"), "w") as fh:
        json.dump(figs, fh, indent=1)
    log(f"[validate] {figs}")
    d["figs"] = figs
    return figs


# -------------------------------------------------------------- RENDER
def _ypos(Y):
    """Earliest start at the back (y = 1), latest at the front (y = 0)."""
    ys = CONFIG["START_YEARS"]
    return 1 - (Y - ys[0]) / (ys[-1] - ys[0])


def _top(d):
    return d["plans"][CONFIG["START_YEARS"][0]]["val"].max()


def scene(ax, d, frac):
    """Draw slices back to front by hand: mplot3d cannot sort them."""
    c = CONFIG
    K = len(d["months"])
    n = int(round(frac * (K - 1))) + 1 if frac > 0 else 1
    x = np.linspace(0, 1, K)
    top = _top(d)
    ax.computed_zorder = False
    for rank, Y in enumerate(c["START_YEARS"]):
        p, y = d["plans"][Y], _ypos(Y)
        z0 = 10 * rank
        on = np.where(p["on"][:n])[0]
        hl = Y in (c["EARLY"], c["LATE"])
        if len(on) < 2:
            ax.plot([0, 1], [y, y], [0, 0], color="#2a2f35", lw=0.8, zorder=z0)
            continue
        xs, v, dp = x[on], p["val"][on] / top, p["dep"][on] / top
        # grey base: the deposits
        base = [(xs[0], y, 0)] + list(zip(xs, np.full_like(xs, y), dp)) + [(xs[-1], y, 0)]
        ax.add_collection3d(Poly3DCollection(
            [base], facecolors=DEPOSIT, edgecolors="none", alpha=0.97, zorder=z0 + 1))
        # cap: one quad per month, coloured by how far above the deposits it sits
        quads, cols = [], []
        for k in range(len(xs) - 1):
            quads.append([(xs[k], y, dp[k]), (xs[k + 1], y, dp[k + 1]),
                          (xs[k + 1], y, v[k + 1]), (xs[k], y, v[k])])
            if v[k] < dp[k]:
                cols.append(UNDER)
            elif Y == c["LATE"]:
                cols.append(THEME["RED"])
            else:
                cols.append(GROW(min(1.0, 0.12 + 0.88 * (v[k] ** 0.6))))
        ax.add_collection3d(Poly3DCollection(
            quads, facecolors=cols, edgecolors=cols, linewidths=0.4,
            alpha=0.97, zorder=z0 + 2))
        lc = "white" if Y == c["EARLY"] else (THEME["RED"] if Y == c["LATE"] else "#d7dde3")
        ax.plot(xs, np.full_like(xs, y), v, color=lc, lw=2.2 if hl else 0.9,
                zorder=z0 + 3)
        ax.plot(xs, np.full_like(xs, y), dp, color="#9aa3ad", lw=0.6, zorder=z0 + 3)
        ax.plot([0, 1], [y, y], [0, 0], color="#2a2f35", lw=0.8, zorder=z0)
    ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1.05); ax.set_zlim(0, 1.0)
    return n - 1


def style_axes(ax):
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
        pane.line.set_color((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def scene_labels(ax, d, last, fs=11):
    """Projected to 2D so the slices cannot depth-sort the labels away."""
    from mpl_toolkits.mplot3d import proj3d
    M = ax.get_proj()
    c, figs = CONFIG, d["figs"]
    top = _top(d)

    def put(x, y, z, s, **kw):
        x2, y2, _ = proj3d.proj_transform(x, y, z, M)
        ax.text2D(x2, y2, s, transform=ax.transData, zorder=1e6, ha="center",
                  va="center", bbox=dict(boxstyle="round,pad=0.25", fc="black",
                                         ec="none", alpha=0.65), **kw)

    # start-year tags on the left edge of the two highlighted slices
    for Y, col in ((c["EARLY"], "white"), (c["LATE"], THEME["RED"])):
        x0 = (pd.Period(f"{Y}-02", "M") - d["months"][0]).n / (len(d["months"]) - 1)
        put(x0 - 0.07, _ypos(Y), 0.02, f"START\n{Y}", color=col, fontsize=fs,
            fontweight="bold", family=THEME["MONO"])
    if last == len(d["months"]) - 1:
        e = d["plans"][c["EARLY"]]["val"][-1] / top
        l = d["plans"][c["LATE"]]["val"][-1] / top
        put(0.84, _ypos(c["EARLY"]), e + 0.02, usd(figs["early_short"]),
            color="white", fontsize=fs + 5, fontweight="bold", family=THEME["MONO"])
        put(0.84, _ypos(c["LATE"]), l + 0.05, usd(figs["late_short"]),
            color=THEME["RED"], fontsize=fs + 5, fontweight="bold",
            family=THEME["MONO"])


def text_block(fig, figs, scale=1.0):
    L, s = LAYOUT, scale
    fig.text(0.5, L["TITLE"], "THE COST OF WAITING", ha="center",
             fontsize=29 * s, fontweight="bold", color=THEME["TEXT"],
             family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             usd(f"{figs['monthly']} a month into SPY  |  one slice per start year  |  "
                 f"{figs['first_start']} to {figs['last_start']}"),
             ha="center", fontsize=12 * s, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.5, L["KEY"],
             "grey = money paid in  ·  colour = what the market added  ·  red = the 2003 start",
             ha="center", fontsize=10 * s, color=THEME["TEXT_DIM"], family=THEME["FONT"])
    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10 * s,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])
    fig.text(0.5, L["BIG"], usd(f"WAITING 10 YEARS COST {figs['cost_short']}"),
             ha="center", fontsize=22 * s, color=THEME["RED"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             usd(f"paid in only {figs['dep_gap']} less  ·  {figs['early']} start "
                 f"{figs['early_short']}  ·  {figs['late']} start {figs['late_short']}"),
             ha="center", fontsize=11.5 * s, color=THEME["TEXT_DIM"],
             family=THEME["MONO"])


def note(fig, d, last, scale=1.0):
    fig.text(0.5, LAYOUT["NOTE"], f"{d['months'][last].year}",
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
    ax.view_init(elev=c["ELEV_BASE"] + 5 * np.sin(t * np.pi),
                 azim=c["AZIM_START"] + c["AZIM_SWEEP"] * t)
    ax.set_box_aspect((1.2, 0.7, 1.35), zoom=c["ZOOM"])
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
