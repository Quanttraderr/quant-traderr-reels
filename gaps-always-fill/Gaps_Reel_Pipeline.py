"""
Gaps_Reel_Pipeline.py
=====================
Reel 25 for @quant.traderr, "Gaps Always Fill?". Every opening gap of more
than 1% in SPY since 1993, and how long it took the price to come back to
the previous close.

THE DATA
    SPY daily open, high, low and close from Yahoo, Jan 1993 to Sep 2026,
    dividend-adjusted (auto_adjust), so ex-dividend days do not show up as
    fake gaps. Real data throughout. Nothing here is simulated.

THE MATHS
    gap_t = open_t / close_{t-1} - 1. A gap counts when |gap_t| > 1%.
    A gap up is filled on the first day k >= t with low_k <= close_{t-1},
    a gap down on the first day with high_k >= close_{t-1}. Days to fill
    = k - t, so a fill inside the gap day itself is 0. Only gaps with a full
    252 sessions ahead are counted, so "still open after a year" is defined
    for every gap in the sample.

    In 3D: a field of arches. x = date of the gap. Each arch springs from
    the centre line and lands at the distance it took to fill, on a log
    scale, gap ups to one side and gap downs to the other. Arch height is
    the size of the gap. Colour runs blue (filled fast) to orange (filled
    slow). Gaps still open after a year are red half arches that never come
    down.

THE CLAIM THE VISUAL MAKES
    Big gaps mostly do not fill the same day: 36% of SPY gaps over 1% did,
    and 8% were still open a year later.

WHAT THE VISUAL IS NOT
    Not a trading rule in either direction. Small gaps do fill most of the
    time (count every gap and about two in three close the same day), which
    is where the saying comes from. "Filled" ignores what happened in
    between: a gap that fills after a 20% drawdown counts the same as one
    that fills by lunch. One ETF, daily bars, no intraday path.

RUN
    python Gaps_Reel_Pipeline.py --smoke   # 3 frames
    python Gaps_Reel_Pipeline.py           # full render -> topic.mp4

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
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from mpl_toolkits.mplot3d import proj3d

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "_cache")

CONFIG = {
    "START": "1993-01-01",
    "END": "2026-09-30",
    "THRESH": 0.01,                  # gaps above 1% either way
    "HORIZON": 252,                  # one year of sessions to fill
    "GAPCAP": 0.05,                  # arch height cap at 5%, labels stay true
    "NPTS": 24,                      # points per arch

    # render
    "W": 1080, "H": 1920, "DPI": 100,
    "FPS": 30,
    "HOOK_SEC": 1.4, "BUILD_SEC": 8.5, "HOLD_SEC": 1.8,
    "ELEV_BASE": 24, "AZIM_START": -50, "AZIM_SWEEP": 18,
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
# filled gaps only: blue (fast) -> orange (slow), red is reserved for never
FILL = LinearSegmentedColormap.from_list(
    "fill", [THEME["BLUE"], THEME["CYAN"], THEME["GREEN"],
             THEME["YELLOW"], THEME["ORANGE"]], N=256)

SAFE = {"TOP": 0.12, "BOTTOM": 0.20, "RIGHT": 0.14}

LAYOUT = {
    "TITLE": 0.862, "SUBTITLE": 0.836, "KEY": 0.814, "HANDLE": 0.792,
    "AXES": [-0.08, 0.30, 1.16, 0.485],
    "NOTE": 0.272, "BIG": 0.242, "SUB": 0.215,
}

# span ticks on the floor: sessions -> label
TICKS = [(0, "same day"), (5, "1 week"), (21, "1 month"), (252, "1 year")]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------- DATA
def fetch():
    """yfinance once, then a CSV cache. Delete _cache/ to refetch."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "spy_ohlc.csv")
    if not os.path.exists(path):
        import yfinance as yf
        log("[Data] fetching SPY")
        df = yf.download("SPY", start=CONFIG["START"], end="2026-10-01",
                         auto_adjust=True, progress=False)
        if df.empty:
            raise SystemExit("no data returned; refusing to draw a synthetic tape")
        df.columns = df.columns.get_level_values(0)
        df.to_csv(path)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.loc[CONFIG["START"]:CONFIG["END"], ["Open", "High", "Low", "Close"]].dropna()


# ------------------------------------------------------------- COMPUTE
def span(days):
    """Sessions to fill -> arch span on the floor, log scale, 0..1."""
    return 0.06 + 0.94 * np.log1p(days) / np.log1p(CONFIG["HORIZON"])


def compute(df):
    o, h, l, c = (df[k].values for k in ("Open", "High", "Low", "Close"))
    pc = np.r_[np.nan, c[:-1]]
    gap = o / pc - 1
    H = CONFIG["HORIZON"]
    n = len(df)
    rows = []
    for i in range(1, n - H):
        g = gap[i]
        if not abs(g) > CONFIG["THRESH"]:
            continue
        if g > 0:
            hit = np.nonzero(l[i:i + H + 1] <= pc[i])[0]
        else:
            hit = np.nonzero(h[i:i + H + 1] >= pc[i])[0]
        days = int(hit[0]) if len(hit) else -1          # -1 = still open after a year
        rows.append((df.index[i], g, days))
    gaps = pd.DataFrame(rows, columns=["date", "gap", "days"]).set_index("date")

    # every gap, however small, for the honest comparison in the caption
    all_g = gap[1:n - H]
    all_up = all_g > 0
    fill0 = np.where(all_up, l[1:n - H] <= pc[1:n - H], h[1:n - H] >= pc[1:n - H])
    nz = all_g != 0
    share_all_same = float(fill0[nz].mean())

    d = gaps.index
    x = d.year + (d.dayofyear - 1) / 365.25
    days = gaps["days"].values
    sgn = np.sign(gaps["gap"].values)
    L = np.where(days >= 0, span(np.maximum(days, 0)), 1.0)
    return {"df": df, "gaps": gaps, "x": np.asarray(x, float), "L": L, "sgn": sgn,
            "z": np.minimum(np.abs(gaps["gap"].values), CONFIG["GAPCAP"]) / CONFIG["GAPCAP"],
            "days": days, "share_all_same": share_all_same,
            "x0": float(np.floor(x.min())), "x1": float(np.ceil(x.max()))}


# ------------------------------------------------------------ VALIDATE
def _pct(p):
    return f"{p * 100:.0f}%"


def validate(d):
    """Every number that reaches a frame is asserted here, before any render."""
    df, gp = d["df"], d["gaps"]
    days = gp["days"].values
    up = gp["gap"].values > 0
    assert len(df) > 8000, f"only {len(df)} SPY sessions, history looks cut"
    assert df.index[0].year == 1993
    assert 300 < len(gp) < 2000, f"{len(gp)} gaps over 1%, implausible"
    same = float((days == 0).mean())
    week = float(((days >= 0) & (days <= 5)).mean())
    month = float(((days >= 0) & (days <= 21)).mean())
    never = float((days < 0).mean())
    assert 0.15 < same < 0.5, f"same-day fill {same:.3f}: claim 'most do not' gone"
    assert 0.0 < never < 0.25, f"open-after-a-year share {never:.3f} implausible"
    assert same < week < month < 1 - never + 1e-9
    assert 0.5 < d["share_all_same"] < 0.9, \
        f"all-gap same-day fill {d['share_all_same']:.3f}; caption says about two in three"
    assert d["share_all_same"] > same + 0.15, "small gaps no longer fill much more often"
    big = gp["gap"].abs().idxmax()
    assert np.isfinite(d["L"]).all() and np.isfinite(d["z"]).all()

    figs = {
        "share_same": _pct(same),
        "share_week": _pct(week),
        "share_month": _pct(month),
        "share_open": _pct(never),
        "share_same_up": _pct(float((days[up] == 0).mean())),
        "share_same_down": _pct(float((days[~up] == 0).mean())),
        "share_all_same": _pct(d["share_all_same"]),
        "n_gaps": f"{len(gp):,}",
        "n_up": f"{int(up.sum()):,}",
        "n_down": f"{int((~up).sum()):,}",
        "threshold": "1%",
        "biggest": f"{gp.loc[big, 'gap'] * 100:+.0f}%",
        "biggest_date": big.strftime("%b %Y"),
        "start": "1993",
        "end": df.index[-1].strftime("%b %Y"),
        "last_gap": gp.index[-1].strftime("%b %Y"),
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


def _cut(d, frac):
    """Arches rise left to right in time: the date up to which they stand."""
    return d["x0"] - 0.5 + frac * (d["x1"] - d["x0"] + 2.0)


def _color(days):
    if days < 0:
        return to_rgba(THEME["RED"])
    return FILL(np.log1p(days) / np.log1p(CONFIG["HORIZON"]))


def scene(ax, d, frac, lw=0.9):
    x0, x1 = d["x0"] - 0.5, d["x1"] + 0.5
    # floor, centre line and span ticks on both sides
    ax.add_collection3d(Poly3DCollection(
        [[(x0, -1.05, 0), (x1, -1.05, 0), (x1, 1.05, 0), (x0, 1.05, 0)]],
        facecolors=(1, 1, 1, 0.05), edgecolor=(1, 1, 1, 0.2), linewidth=0.6),
        autolim=False)
    ax.plot([x0, x1], [0, 0], [0, 0], color=(1, 1, 1, 0.35), lw=0.8)
    for s, _ in TICKS[1:]:
        for side in (-1, 1):
            ax.plot([x0, x1], [side * span(s)] * 2, [0, 0], color=(1, 1, 1, 0.1), lw=0.5)

    cut = _cut(d, frac)
    grow = np.clip((cut - d["x"]) / 1.5, 0, 1)
    idx = np.nonzero(grow > 0)[0]
    if len(idx) == 0:
        return _limits(ax, x0, x1)
    u = np.linspace(0, 1, CONFIG["NPTS"])
    segs, cols, glows, ends = [], [], [], []
    for i in idx:
        g = grow[i]
        L, z, sg, dy = d["L"][i], d["z"][i], d["sgn"][i], d["days"][i]
        if dy < 0:
            uu = u[u <= 0.5 + 1e-9] * 1.0      # half arch: rises and never lands
        else:
            uu = u
        uu = uu[uu <= g * uu.max() + 1e-9]
        if len(uu) < 2:
            continue
        ys = sg * L * uu
        zs = z * np.sin(np.pi * uu)
        xs = np.full_like(uu, d["x"][i])
        c = _color(dy)
        segs.append(np.column_stack([xs, ys, zs]))
        cols.append(c)
        glows.append((*c[:3], 0.16))
        if dy < 0 and g >= 1:
            ends.append((xs[-1], ys[-1], zs[-1]))
    ax.add_collection3d(Line3DCollection(segs, colors=glows, linewidths=lw * 3.5),
                        autolim=False)
    ax.add_collection3d(Line3DCollection(segs, colors=cols, linewidths=lw),
                        autolim=False)
    if ends:
        e = np.array(ends)
        ax.scatter(e[:, 0], e[:, 1], e[:, 2], s=14, c=THEME["RED"],
                   depthshade=False, linewidths=0)
    _limits(ax, x0, x1)


def _limits(ax, x0, x1):
    ax.set_xlim(x0, x1)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(0, 1.0)


def scene_labels(ax, d, frac, fs=12):
    """Projected 2D text drawn above every 3D collection."""
    M = ax.get_proj()

    def put(x, y, z, s, **kw):
        x2, y2, _ = proj3d.proj_transform(x, y, z, M)
        ax.text2D(x2, y2, s, transform=ax.transData, zorder=1e6,
                  va="center", family=THEME["MONO"], **kw)

    xl = d["x0"] - 0.9
    cut = _cut(d, frac)
    for s, name in TICKS[2:]:                       # "1 week" would sit on GAP UP
        put(xl, span(s), 0, f"{name} ", ha="right", color=THEME["TEXT"], fontsize=fs * 0.8,
            bbox=dict(boxstyle="round,pad=0.15", fc=THEME["BG"], ec="none", alpha=0.75))
    put(xl, 0.12, 0, "GAP UP ", ha="right", color=THEME["TEXT"], fontsize=fs * 0.85,
        fontweight="bold")
    put(xl, -0.55, 0, "GAP DOWN ", ha="right", color=THEME["TEXT"], fontsize=fs * 0.85,
        fontweight="bold")
    for yr in range(2000, int(d["x1"]) + 1, 10):
        if yr <= cut:
            put(yr, -1.12, 0, str(yr), ha="center", color=THEME["TEXT_DIM"], fontsize=fs * 0.85)


def text_block(fig, figs, s=1.0):
    L = LAYOUT
    fig.text(0.5, L["TITLE"], "GAPS ALWAYS FILL?", ha="center", fontsize=29 * s,
             fontweight="bold", color=THEME["TEXT"], family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             f"SPY since {figs['start']}  |  {figs['n_gaps']} opening gaps over "
             f"{figs['threshold']}",
             ha="center", fontsize=12.5 * s, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.5, L["KEY"],
             "one arch per gap  ·  span = days to fill  ·  red = still open after a year",
             ha="center", fontsize=10 * s, color=THEME["TEXT_DIM"], family=THEME["FONT"])
    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10 * s,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])
    fig.text(0.5, L["BIG"], f"only {figs['share_same']} filled the same day",
             ha="center", fontsize=23 * s, color=THEME["ORANGE"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             f"{figs['share_open']} were still open a year later",
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
    frac = 0.35 + 0.65 * frac                     # the hook already shows 1993 to 2005
    t = idx / max(total - 1, 1)

    fig = plt.figure(figsize=(c["W"] / c["DPI"], c["H"] / c["DPI"]),
                     dpi=c["DPI"], facecolor=THEME["BG"])
    text_block(fig, figs)
    ax = fig.add_axes(LAYOUT["AXES"], projection="3d", facecolor=THEME["BG"])
    scene(ax, d, frac)
    ax.view_init(elev=c["ELEV_BASE"] + 5 * np.sin(t * np.pi),
                 azim=c["AZIM_START"] + c["AZIM_SWEEP"] * t)
    ax.set_box_aspect((1.7, 1.0, 0.75), zoom=c["ZOOM"])
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
    log(f"OK {mp4}  ({total} frames, {total / c['FPS']:.1f}s, {c['W']}x{c['H']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    main(**vars(ap.parse_args()))
