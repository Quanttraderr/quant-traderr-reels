"""
WinnersCrashed_Reel_Pipeline.py
===============================
Reel for @quant.traderr, "Every Winner Crashed Hard". Seven of the biggest
winners on the US market, each from its first trading day, and the crash that
every one of them sat through on the way up.

THE DATA
    Daily closes from Yahoo (yfinance, auto_adjust, so splits and dividends are
    folded in) for AAPL, MSFT, AMZN, NVDA, NFLX, TSLA and META, each from its
    IPO, which is also the oldest date Yahoo carries for all seven, up to the
    last closed session in CONFIG["END"]. Real data throughout. Nothing here is
    simulated. A failed download aborts the run, it never falls back to noise.

THE MATHS
    multiple_t  = close_t / close_first            (growth since the IPO)
    drawdown_t  = close_t / max(close_<=t) - 1     (distance below the running high)
    Per stock: total multiple (first to last close), the deepest drawdown and
    the date of its trough, and the multiple from that trough close to the last
    close.

    In 3D, a cave ("Tropfsteinhoehle"): one track per stock in depth (y),
    time along x. Above a dark floor a glowing wall rises with log10 of the
    multiple, coloured blue to yellow by that multiple. Below the floor the
    drawdown hangs down as stalactites, monthly deepest drawdown, coloured
    yellow to red by depth, so the redder and longer the icicle, the worse the
    crash. Time runs left to right and the cave grows with it. Below the IPO
    price the wall has zero height (META 2012), it is clipped, not hidden.

THE CLAIM THE VISUAL MAKES
    Every one of these winners fell at least 69% from a high at some point,
    AMZN by 94% into Sep 2001, and the big multiples came after the crash.

WHAT THE VISUAL IS NOT
    Not evidence that crashes lead to big winners. The seven were picked
    because they won, after the fact: survivorship bias. Many stocks that fell
    90% never came back and are not in this picture. Not a strategy, not a
    forecast, prices only, no dividends paid out beyond what auto_adjust folds
    in, no fees, no taxes.

RUN
    python WinnersCrashed_Reel_Pipeline.py --smoke     # 3 frames -> temp_frames_smoke/
    python WinnersCrashed_Reel_Pipeline.py             # full render -> topic.mp4

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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from mpl_toolkits.mplot3d import proj3d

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "_cache")

CONFIG = {
    # data + maths
    # back to front in depth: the oldest listing at the back, so the younger
    # walls in front start late and leave the view to the old ones open
    "TICKERS": ["AAPL", "MSFT", "AMZN", "NVDA", "NFLX", "TSLA", "META"],
    "HERO": "AMZN",                  # the running counter and the big number
    "END": "2026-10-06",             # last fully closed session
    "LOGTOP": 4.0,                   # wall height: log10(multiple) / LOGTOP
    "DDSCALE": 1.0,                  # icicle length: drawdown * DDSCALE
    "CUT0": 2002.85,                 # the hook opens on the dot-com lows

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
    "ELEV_BASE": 40, "AZIM_START": -98, "AZIM_SWEEP": 18,
    "BOX": (1.1, 2.2, 0.9),
    "ZOOM": 1.28,
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
# the wall: growth only, blue (1x) -> yellow (10,000x); red stays for pain
GROW = LinearSegmentedColormap.from_list(
    "grow", [THEME["BLUE"], THEME["CYAN"], THEME["GREEN"], THEME["YELLOW"]], N=256)
# the icicles: shallow dip yellow -> deep crash red
PAIN = LinearSegmentedColormap.from_list(
    "pain", [THEME["YELLOW"], THEME["ORANGE"], THEME["RED"]], N=256)

# Instagram lays its own controls over a reel: the Reels header on top, the
# name, caption and audio line at the bottom, the like and comment buttons on
# the right. Nothing readable may sit in those bands. The form itself may bleed
# into them, text never.
SAFE = {"TOP": 0.12, "BOTTOM": 0.20, "RIGHT": 0.14}

# Every text row in figure coordinates, measured from the bottom. All of them
# live between SAFE["BOTTOM"] and 1 - SAFE["TOP"], which pruefe_schutzzonen.py
# checks. Counts that used to sit in a third readout line now ride in the
# subtitle, because the bottom band no longer has room for three rows.
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
        px = pd.read_csv(path, index_col=0, parse_dates=True)
        return px.loc[:CONFIG["END"], CONFIG["TICKERS"]]
    import yfinance as yf
    cols = {}
    for t in CONFIG["TICKERS"]:
        log(f"[Data] fetching {t} (max history)")
        df = yf.download(t, period="max", interval="1d", progress=False,
                         auto_adjust=True)
        if df is None or df.empty:
            raise SystemExit(f"no data returned for {t}; refusing to draw a synthetic tape")
        cols[t] = df["Close"].squeeze().dropna()
    # no dropna across columns: each stock keeps its own listing date
    px = pd.DataFrame(cols)[CONFIG["TICKERS"]]
    px.to_csv(path)
    return px.loc[:CONFIG["END"]]


# ------------------------------------------------------------- COMPUTE
def _yr(idx):
    return np.asarray(idx.year + (idx.dayofyear - 1) / 365.25, float)


def frame_frac(idx, total):
    """Shared by VALIDATE (counter strings) and RENDER: build progress 0..1."""
    c = CONFIG
    n_hook = int(c["FPS"] * c["HOOK_SEC"])
    n_build = int(c["FPS"] * c["BUILD_SEC"])
    if idx < n_hook:
        return 0.0
    if idx < n_hook + n_build:
        return (idx - n_hook) / max(n_build - 1, 1)
    return 1.0


def compute(px):
    c = CONFIG
    stocks = {}
    for t in c["TICKERS"]:
        s = px[t].dropna()
        if len(s) < 2000:
            raise SystemExit(f"{t}: only {len(s)} closes, history looks cut")
        mult = s / s.iloc[0]
        dd = s / s.cummax() - 1
        trough = dd.idxmin()
        # monthly geometry: last close for the wall, deepest drawdown for the
        # icicle, so the trough month hangs exactly as deep as the figure says
        m_mult = mult.resample("ME").last()
        m_dd = dd.resample("ME").min()
        mx = _yr(m_mult.index)
        stocks[t] = {
            "s": s, "mult": mult, "dd": dd, "trough": trough,
            "total": float(mult.iloc[-1]),
            "maxdd": float(dd.min()),
            "rebound": float(s.iloc[-1] / s.loc[trough]),
            "first": s.index[0],
            "mx": mx,
            "zu": np.clip(np.log10(m_mult.values), 0, None) / c["LOGTOP"],
            "cu": np.clip(np.log10(np.maximum(m_mult.values, 1)) / c["LOGTOP"], 0, 1),
            "zd": m_dd.values * c["DDSCALE"],
            "cd": np.clip(-m_dd.values / 0.95, 0, 1),
            "x_trough": float(_yr(pd.DatetimeIndex([trough]))[0]),
        }
    last = max(st["s"].index[-1] for st in stocks.values())
    x1 = float(_yr(pd.DatetimeIndex([last]))[0])
    x0 = float(min(st["mx"][0] for st in stocks.values()))
    return {"px": px, "stocks": stocks, "x0": x0, "x1": x1, "last": last}


def cut_of(d, frac):
    return CONFIG["CUT0"] + frac * (d["x1"] + 0.05 - CONFIG["CUT0"])


# ------------------------------------------------------------ VALIDATE
def _mult(m):
    return f"{m:.1f}x" if m < 10 else f"{m:,.0f}x"


def _pctv(p):
    """Drawdowns are <= 0; a flat 0 prints as 0%, never -0%."""
    v = round(p * 100)
    return f"{v:.0f}%" if v != 0 else "0%"


def validate(d):
    """Every number that reaches a frame is asserted here, before any render.
    A data refresh that moves a figure out of band fails the build instead of
    shipping a caption that no longer matches the picture."""
    c, S = CONFIG, d["stocks"]
    ipo = {"AAPL": "1980-12-12", "MSFT": "1986-03-13", "AMZN": "1997-05-15",
           "NVDA": "1999-01-22", "NFLX": "2002-05-23", "TSLA": "2010-06-29",
           "META": "2012-05-18"}
    for t, st in S.items():
        if t in ipo:
            gap = abs((st["first"] - pd.Timestamp(ipo[t])).days)
            assert gap <= 10, f"{t} starts {st['first'].date()}, not at its IPO"
        assert -1.0 < st["maxdd"] < 0.0, f"{t} drawdown {st['maxdd']} impossible"
        assert st["total"] > 1.0, f"{t} is not a winner over its history"
        assert st["rebound"] >= 1.0, f"{t} below its own trough today"
        assert np.isfinite(st["zu"]).all() and np.isfinite(st["zd"]).all()
        assert st["zu"].max() <= 1.05, f"{t} wall above LOGTOP, raise it"
    worst_min = max(st["maxdd"] for st in S.values())   # the mildest of the worst
    assert worst_min <= -0.5, \
        f"one stock never fell 50% ({worst_min:.2f}); 'every winner crashed' gone"
    assert (d["last"] - pd.Timestamp(c["END"])).days <= 0
    assert (pd.Timestamp(c["END"]) - d["last"]).days <= 7, "data ends early"

    H = S[c["HERO"]]
    figs = {
        "n_stocks": str(len(S)),
        "mildest_crash": _pctv(worst_min),
        "end": d["last"].strftime("%b %Y"),
        "hero": c["HERO"],
    }
    for t, st in S.items():
        k = t.lower()
        figs[f"{k}_total"] = _mult(st["total"])
        figs[f"{k}_dd"] = _pctv(st["maxdd"])
        figs[f"{k}_trough"] = st["trough"].strftime("%b %Y")
        figs[f"{k}_rebound"] = _mult(st["rebound"])
        figs[f"{k}_ipo"] = st["first"].strftime("%Y")

    # The running counter: one string per frame, made here, never in render.
    total = int(c["FPS"] * (c["HOOK_SEC"] + c["BUILD_SEC"] + c["HOLD_SEC"]))
    s, mult, dd = H["s"], H["mult"], H["dd"]
    xs = _yr(s.index)
    counter = []
    for i in range(total):
        cut = cut_of(d, frame_frac(i, total))
        j = int(np.searchsorted(xs, cut, side="right")) - 1
        j = min(max(j, 0), len(s) - 1)
        # frames step about a month, so the running worst is taken over every
        # session up to the cut, not sampled: the trough day is never skipped
        counter.append(f"{s.index[j].strftime('%b %Y')}  {c['HERO']} "
                       f"{_pctv(float(dd.iloc[j])):>4}  worst "
                       f"{_pctv(float(dd.iloc[:j + 1].min())):>4}  "
                       f"{_mult(float(mult.iloc[j])):>6} since IPO")
    assert figs[f"{c['HERO'].lower()}_total"] in counter[-1], \
        "last counter frame does not land on the asserted total"
    assert any(figs[f"{c['HERO'].lower()}_dd"] in x for x in counter), \
        "the counter never shows the asserted trough"
    d["counter"] = counter

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


def _limits(ax, d):
    ax.set_xlim(d["x0"] - 0.6, d["x1"] + 0.4)
    ax.set_ylim(-0.6, len(CONFIG["TICKERS"]) - 0.4)
    ax.set_zlim(-1.0, 1.05)


def _curtain(x, y, z, cols, alpha):
    """Vertical quads from the floor (z=0) to z, one per month."""
    n = len(x)
    if n < 2:
        return None, None, None
    v = np.zeros((n - 1, 4, 3))
    v[:, 0] = np.column_stack([x[:-1], np.full(n - 1, y), np.zeros(n - 1)])
    v[:, 1] = np.column_stack([x[1:], np.full(n - 1, y), np.zeros(n - 1)])
    v[:, 2] = np.column_stack([x[1:], np.full(n - 1, y), z[1:]])
    v[:, 3] = np.column_stack([x[:-1], np.full(n - 1, y), z[:-1]])
    fc = cols[:-1].copy()
    fc[:, 3] = alpha
    edge = np.stack([np.column_stack([x[:-1], np.full(n - 1, y), z[:-1]]),
                     np.column_stack([x[1:], np.full(n - 1, y), z[1:]])], axis=1)
    return v, fc, edge


def scene(ax, d, frac, lw=1.0):
    x0, x1 = d["x0"] - 0.6, d["x1"] + 0.4
    ny = len(CONFIG["TICKERS"])
    # the floor of the cave and one faint lane per stock
    ax.add_collection3d(Poly3DCollection(
        [[(x0, -0.6, 0), (x1, -0.6, 0), (x1, ny - 0.4, 0), (x0, ny - 0.4, 0)]],
        facecolors=(1, 1, 1, 0.035), edgecolor=(1, 1, 1, 0.22), linewidth=0.6),
        autolim=False)
    for k in range(ny):
        ax.plot([x0, x1], [k, k], [0, 0], color=(1, 1, 1, 0.08), lw=0.5)

    cut = cut_of(d, frac)
    polys, fcs, eu, ecu, ed, ecd = [], [], [], [], [], []
    for k, t in enumerate(CONFIG["TICKERS"]):
        st = d["stocks"][t]
        m = st["mx"] <= cut
        if m.sum() < 2:
            continue
        x = st["mx"][m]
        yk = ny - 1 - k                      # first ticker sits at the back
        cu = GROW(st["cu"][m]); cd = PAIN(st["cd"][m])
        v, fc, e = _curtain(x, yk, st["zu"][m], cu, 0.55)
        polys.append(v); fcs.append(fc); eu.append(e); ecu.append(cu[1:])
        v, fc, e = _curtain(x, yk, st["zd"][m], cd, 0.62)
        polys.append(v); fcs.append(fc); ed.append(e); ecd.append(cd[1:])
    if polys:
        ax.add_collection3d(Poly3DCollection(np.concatenate(polys),
                                             facecolors=np.concatenate(fcs),
                                             edgecolors="none", linewidths=0),
                            autolim=False)
        for segs, cols in ((eu, ecu), (ed, ecd)):
            S = np.concatenate(segs); C = np.concatenate(cols)
            glow = C.copy(); glow[:, 3] = 0.22
            ax.add_collection3d(Line3DCollection(S, colors=glow, linewidths=lw * 4),
                                autolim=False)
            ax.add_collection3d(Line3DCollection(S, colors=C, linewidths=lw),
                                autolim=False)
    _limits(ax, d)


def scene_labels(ax, d, frac, fs=12):
    """Projected 2D text drawn above every 3D collection, all inside the limits."""
    M = ax.get_proj()
    ny = len(CONFIG["TICKERS"])
    figs = d["figs"]

    def put(x, y, z, s, **kw):
        x2, y2, _ = proj3d.proj_transform(x, y, z, M)
        kw.setdefault("va", "center")
        ax.text2D(x2, y2, s, transform=ax.transData, zorder=1e6,
                  family=THEME["MONO"], **kw)

    cut = cut_of(d, frac)
    for k, t in enumerate(CONFIG["TICKERS"]):
        st = d["stocks"][t]
        yk = ny - 1 - k
        if st["mx"][0] > cut:
            continue
        # ticker on the floor where its lane begins
        put(st["mx"][0] - 0.3, yk, 0, f"{t} ", ha="right", color=THEME["TEXT"],
            fontsize=fs * 0.95, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.12", fc=THEME["BG"], ec="none", alpha=0.7))
        # the worst crash, pinned to the root of its deepest icicle in its own
        # lane (a tip label would project into the next lane and misread)
        if st["x_trough"] <= cut:
            hero = t == CONFIG["HERO"]
            put(st["x_trough"], yk, -0.03,
                figs[f"{t.lower()}_dd"], ha="center", va="top",
                color=THEME["RED"], fontsize=fs * (1.25 if hero else 0.95),
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc=THEME["BG"], ec="none", alpha=0.85))
    for yr in (1990, 2000, 2010, 2020):
        if yr <= cut:
            put(yr, -0.95, 0, str(yr), ha="center", color=THEME["TEXT_DIM"],
                fontsize=fs * 0.8)


def text_block(fig, figs, counter=None, s=1.0):
    L = LAYOUT
    fig.text(0.5, L["TITLE"], "EVERY WINNER CRASHED HARD", ha="center",
             fontsize=27 * s, fontweight="bold", color=THEME["TEXT"],
             family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             f"{figs['n_stocks']} mega winners from their IPO  |  every one fell "
             f"at least {figs['mildest_crash'].lstrip('-')}",
             ha="center", fontsize=12.5 * s, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.5, L["KEY"],
             "wall up = growth since IPO (log)  ·  icicles down = drawdown from high",
             ha="center", fontsize=10 * s, color=THEME["TEXT_DIM"], family=THEME["FONT"])
    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10 * s,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])
    if counter:
        fig.text(0.5, L["NOTE"], counter, ha="center", fontsize=12.5 * s,
                 color=THEME["CYAN"], family=THEME["MONO"])
    h = figs["hero"].lower()
    fig.text(0.5, L["BIG"], f"{figs['hero']} {figs[h + '_dd']}, then {figs[h + '_rebound']}",
             ha="center", fontsize=24 * s, color=THEME["RED"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             f"crash to {figs[h + '_trough']}  ·  {figs[h + '_total']} since IPO",
             ha="center", fontsize=12 * s, color=THEME["TEXT_DIM"], family=THEME["MONO"])


def render_frame(idx, total, d, out_path):
    c, figs = CONFIG, d["figs"]
    frac = frame_frac(idx, total)
    t = idx / max(total - 1, 1)

    fig = plt.figure(figsize=(c["W"] / c["DPI"], c["H"] / c["DPI"]),
                     dpi=c["DPI"], facecolor=THEME["BG"])
    # Readout: strings only, straight from figures.json and the counter list.
    text_block(fig, figs, d["counter"][idx])
    ax = fig.add_axes(LAYOUT["AXES"], projection="3d", facecolor=THEME["BG"])
    scene(ax, d, frac)
    ax.view_init(elev=c["ELEV_BASE"] + 5 * np.sin(t * np.pi),
                 azim=c["AZIM_START"] + c["AZIM_SWEEP"] * t)
    ax.set_box_aspect(c["BOX"], zoom=c["ZOOM"])
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
