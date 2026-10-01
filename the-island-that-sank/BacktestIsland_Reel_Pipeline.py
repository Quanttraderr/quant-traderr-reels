"""
BacktestIsland_Reel_Pipeline.py
===============================
Reel 19 for @quant.traderr, "The Island That Sank". One classic trend rule,
6,000 settings of it, scored on SPY twice: once on 1994 to 2009, once on
2010 to 2026.

THE DATA
    SPY daily closes from Yahoo, auto-adjusted (dividends reinvested), and the
    13-week T-bill yield ^IRX as the return on cash. Real data throughout.
    Nothing here is simulated.

THE MATHS
    Rule: long SPY while MA_fast > MA_slow * (1 + band), in T-bills while
    MA_fast < MA_slow * (1 - band), otherwise keep the previous position
    (the band is hysteresis). The signal is lagged one day, so a close is
    never traded on itself. No costs, no taxes.
    Grid: 20 fast windows (5 to 100 days, log spaced), 20 slow windows (50 to
    300 days, log spaced), 16 bands (0% to 3%). Pairs with fast >= slow are
    dropped, leaving the settings counted in figures.json.
    Score: CAGR of the rule minus CAGR of buy-and-hold over the same days.
    In sample: first day with a full 300-day slow MA (Apr 1994) to Dec 2009.
    Out of sample: Jan 2010 to Sep 2026.

    In 3D: the grid is a box, fast x slow x band. Build part one draws the
    isosurface score = L while L sweeps from the best in-sample score down to
    0: the island of settings that beat buy-and-hold rises out of the box,
    coloured by L. Part two keeps that island as a ghost and colours every
    setting by its out-of-sample score. The white marker is the single best
    in-sample setting.

THE CLAIM THE VISUAL MAKES
    A backtest that most settings pass can still tell you nothing about the
    next 16 years. 79% beat buy-and-hold in sample; none did out of sample.

WHAT THE VISUAL IS NOT
    Not proof that trend following is useless: 2010 to 2026 was a long bull
    market with short crashes, which is the worst case for a slow trend filter,
    and the rule still cut drawdowns, which CAGR does not reward. Not a
    risk-adjusted comparison. One asset, one rule family, one split date; a
    different split moves the numbers. No trading costs, which would only make
    the rule look worse. The smooth surface is interpolated between grid
    points for display; the shares in the readout are counted on the raw grid.

RUN
    python BacktestIsland_Reel_Pipeline.py --smoke   # 3 frames
    python BacktestIsland_Reel_Pipeline.py           # full render -> topic.mp4

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
    "START": "1993-01-29",
    "END": "2026-09-30",
    "SPLIT": "2010-01-01",
    "FAST": (5, 100, 20),
    "SLOW": (50, 300, 20),
    "BAND": (0.0, 0.03, 16),
    "UPSAMPLE": 3,
    "PHASE1": 0.55,

    # render
    "W": 1080, "H": 1920, "DPI": 100,
    "FPS": 30,
    "HOOK_SEC": 1.4, "BUILD_SEC": 8.5, "HOLD_SEC": 1.8,
    "ELEV_BASE": 22, "AZIM_START": -58, "AZIM_SWEEP": 28,
    "ZOOM": 1.0,
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

# score colour scale, percentage points of CAGR versus buy-and-hold
SCORE_LO, SCORE_HI = -8.0, 6.0


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------- DATA
def fetch():
    """yfinance once, then CSV caches. Delete _cache/ to refetch."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    p_spy = os.path.join(CACHE_DIR, "spy_ohlc.csv")
    p_irx = os.path.join(CACHE_DIR, "irx.csv")
    import yfinance as yf
    if not os.path.exists(p_spy):
        log("[Data] fetching SPY")
        df = yf.download("SPY", start=CONFIG["START"], progress=False, auto_adjust=True)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        if df.empty:
            raise SystemExit("no SPY data; refusing to draw a synthetic tape")
        df.to_csv(p_spy)
    if not os.path.exists(p_irx):
        log("[Data] fetching ^IRX")
        irx = yf.download("^IRX", start=CONFIG["START"], progress=False,
                          auto_adjust=True)["Close"].squeeze()
        if irx.empty:
            raise SystemExit("no ^IRX data; refusing to invent a cash rate")
        irx.to_frame("IRX").to_csv(p_irx)
    close = pd.read_csv(p_spy, index_col=0, parse_dates=True)["Close"]
    close = close.loc[CONFIG["START"]:CONFIG["END"]]
    irx = pd.read_csv(p_irx, index_col=0, parse_dates=True)["IRX"]
    return close, irx


# ------------------------------------------------------------- COMPUTE
def _cagr(x):
    return (1 + x).prod() ** (252 / len(x)) - 1


def compute(data):
    close, irx = data
    c = CONFIG
    r = close.pct_change().fillna(0)
    rf = (irx.reindex(close.index).ffill() / 100 / 252).fillna(0)
    fasts = np.unique(np.geomspace(*c["FAST"]).round().astype(int))
    slows = np.unique(np.geomspace(*c["SLOW"]).round().astype(int))
    bands = np.linspace(*c["BAND"])
    ma = {w: close.rolling(w).mean() for w in set(fasts) | set(slows)}
    first = close.index[max(slows)]
    is_sl = slice(first, pd.Timestamp(c["SPLIT"]) - pd.Timedelta(days=1))
    os_sl = slice(pd.Timestamp(c["SPLIT"]), None)
    bh_is, bh_os = _cagr(r[is_sl]), _cagr(r[os_sl])

    shape = (len(fasts), len(slows), len(bands))
    g_is, g_os = np.full(shape, np.nan), np.full(shape, np.nan)
    os_idx = r[os_sl].index
    lr_os = {}                       # (i, j, k) -> daily log returns after the split
    for i, f in enumerate(fasts):
        for j, s in enumerate(slows):
            if f >= s:
                continue
            ratio = ma[f] / ma[s] - 1
            for k, b in enumerate(bands):
                sig = pd.Series(np.where(ratio > b, 1.0,
                                         np.where(ratio < -b, 0.0, np.nan)),
                                index=close.index).ffill().fillna(0).shift(1).fillna(0)
                pr = sig * r + (1 - sig) * rf
                g_is[i, j, k] = _cagr(pr[is_sl])
                g_os[i, j, k] = _cagr(pr[os_sl])
                lr_os[(i, j, k)] = np.log1p(pr[os_sl].values)
    log(f"[compute] {np.isfinite(g_is).sum()} settings scored")

    # The sinking, measured: score every rule on an out-of-sample window that
    # starts Jan 2010 and grows month by month. Annualised log excess over
    # buy-and-hold, in percentage points. Real data at every checkpoint.
    keys = list(lr_os)
    cum = np.cumsum(np.array([lr_os[k] for k in keys]), axis=1)
    cum_bh = np.cumsum(np.log1p(r[os_sl].values))
    month_ends = os_idx.to_series().groupby(os_idx.to_period("M")).last()
    pos = os_idx.get_indexer(month_ends.values)
    windows, shares = [], []
    for p_ in pos:
        yrs = (p_ + 1) / 252
        ex = (cum[:, p_] - cum_bh[p_]) / yrs * 100
        field = np.full(shape, np.nan)
        for (i, j, k), v in zip(keys, ex):
            field[i, j, k] = v
        windows.append(field)
        shares.append(float((ex > 1e-9).mean()))
    return {"fasts": fasts, "slows": slows, "bands": bands,
            "s_is": (g_is - bh_is) * 100, "s_os": (g_os - bh_os) * 100,
            "g_is": g_is, "g_os": g_os, "bh_is": bh_is, "bh_os": bh_os,
            "is_start": r[is_sl].index[0], "is_end": r[is_sl].index[-1],
            "os_start": r[os_sl].index[0], "os_end": r[os_sl].index[-1],
            "windows": windows, "win_shares": shares,
            "win_dates": list(month_ends.values)}


def upsample(field):
    """Pad and upsample a score field for a smooth, closed isosurface.
    Display only: every share in the readout is counted on the raw grid."""
    from scipy.ndimage import zoom
    fill = np.where(np.isfinite(field), field, -15.0)
    fill = np.pad(fill, 1, constant_values=-15.0)
    return zoom(fill, CONFIG["UPSAMPLE"], order=1)


def prepare_geometry(d):
    d["field"] = upsample(d["s_is"])
    n0 = d["s_is"].shape[0] + 2
    d["field_scale"] = [(n0 - 1) / (d["field"].shape[0] - 1)] * 3


def isosurface(d, level, field=None):
    from skimage.measure import marching_cubes
    f = d["field"] if field is None else field
    if level >= f.max() or level <= f.min():
        return None
    verts, faces, _, _ = marching_cubes(f, level=level)
    verts = verts * np.array(d["field_scale"]) - 1.0     # undo the pad
    return verts[faces]


# ------------------------------------------------------------ VALIDATE
def validate(d):
    """Every number that reaches a frame is asserted here, before any render."""
    ok = np.isfinite(d["s_is"])
    n = int(ok.sum())
    sh_is = float((d["s_is"][ok] > 0).mean())
    sh_os = float((d["s_os"][ok] > 0).mean())
    best = np.unravel_index(np.nanargmax(d["s_is"]), d["s_is"].shape)
    d["best"] = best
    b_is, b_os = float(d["g_is"][best]), float(d["g_os"][best])

    assert 4000 <= n <= 6400, f"{n} settings, grid changed"
    assert 0.0 <= sh_os <= 1.0 and 0.0 <= sh_is <= 1.0
    assert sh_is > sh_os, "in sample does not beat out of sample; claim gone"
    assert -0.05 < d["bh_is"] < 0.25 and -0.05 < d["bh_os"] < 0.30
    assert b_is > d["bh_is"], "best in-sample setting does not beat buy-and-hold"
    assert -0.10 < b_os < 0.40
    assert d["is_end"] < d["os_start"]
    n_tie = int((np.abs(d["s_os"][ok]) < 1e-9).sum())
    assert abs(d["win_shares"][-1] - sh_os) < 0.005, "window series disagrees with OS share"
    i_2010 = [k for k, t in enumerate(d["win_dates"]) if pd.Timestamp(t).year == 2010][-1]
    sh_2010 = d["win_shares"][i_2010]
    assert 0.0 <= sh_2010 <= 1.0

    f, s, b = (d["fasts"][best[0]], d["slows"][best[1]], d["bands"][best[2]])
    figs = {
        "n_settings": f"{n:,}",
        "share_is": f"{sh_is * 100:.0f}%",
        "share_os": f"{sh_os * 100:.0f}%",
        "is_period": f"{d['is_start'].year} to {d['is_end'].year}",
        "os_period": f"{d['os_start'].year} to {d['os_end'].year}",
        "bh_is": f"{d['bh_is'] * 100:.1f}%",
        "bh_os": f"{d['bh_os'] * 100:.1f}%",
        "best_is": f"{b_is * 100:.1f}%",
        "best_os": f"{b_os * 100:.1f}%",
        "best_rule": f"{f}/{s}-day MA, {b * 100:.1f}% band",
        "grid": f"{len(d['fasts'])} x {len(d['slows'])} x {len(d['bands'])}",
        "n_beat_os": f"{int((d['s_os'][ok] > 1e-9).sum())}",
        "n_tied_os": f"{n_tie}",
        "share_end_2010": f"{sh_2010 * 100:.0f}%",
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


def box_edges(ax, shape):
    nx, ny, nz = [s - 1 for s in shape]
    pts = np.array([[x, y, z] for x in (0, nx) for y in (0, ny) for z in (0, nz)])
    for i in range(8):
        for j in range(i + 1, 8):
            if np.sum(pts[i] != pts[j]) == 1:
                ax.plot(*zip(pts[i], pts[j]), color=THEME["TEXT_DIM"], lw=0.6, alpha=0.5)


def _shaded(tri, color, alpha, edge=(1, 1, 1, 0.06)):
    from matplotlib.colors import LightSource
    return Poly3DCollection(tri, facecolors=color, edgecolor=edge, linewidth=0.12,
                            alpha=alpha, shade=True,
                            lightsource=LightSource(azdeg=300, altdeg=40))


def scene(ax, d, frac, phase_frac, fs=11):
    """Shared by reel and still. frac 0..1 over the build.

    Part one lowers the iso level on the in-sample field until the island of
    winners stands at level 0. Part two keeps that island as a white ghost and
    draws the level-0 island of the out-of-sample field on a window that
    starts Jan 2010 and grows one month per step, so what sinks is measured,
    not morphed. Late in part two every rule becomes a red dot, sized by how far it
    trailed buy-and-hold out of sample."""
    s_is, s_os = d["s_is"], d["s_os"]
    shape = s_is.shape
    ok = np.isfinite(s_is)
    I, J, K = np.nonzero(ok)
    top = float(np.nanmax(s_is))
    p1 = phase_frac

    def colour(level):
        return CMAP(np.clip((level - SCORE_LO) / (SCORE_HI - SCORE_LO), 0, 1))

    if frac <= p1:
        u = frac / p1
        level = top * 0.6 * (1 - u) ** 1.3 + 1e-3
        ax.scatter(I, J, K, s=1.0, c=THEME["TEXT_DIM"], alpha=0.18,
                   depthshade=False, linewidths=0)
        tri = isosurface(d, level)
        if tri is not None:
            ax.add_collection3d(_shaded(tri, colour(level), 0.9))
        phase = 1
    else:
        u = (frac - p1) / (1 - p1)
        tri = isosurface(d, 0.0)
        if tri is not None:
            ax.add_collection3d(Poly3DCollection(
                tri, facecolor=(1, 1, 1, 0.02), edgecolor=(1, 1, 1, 0.045),
                linewidth=0.15))
        # month of the growing window: slow at first, where the sinking happens
        n_w = len(d["windows"])
        w = min(int(round((n_w - 1) * min(1.0, u / 0.8) ** 2.2)), n_w - 1)
        tri = isosurface(d, 0.0, upsample(d["windows"][w]))
        if tri is not None:
            ax.add_collection3d(_shaded(tri, colour(0.0), 0.9))
        a = np.clip((u - 0.35) / 0.4, 0, 1)
        if a > 0:
            # every rule lost, so colour is binary (red = behind buy & hold)
            # and the dot size carries how far behind, in points per year
            v = s_os[ok]
            size = 2.0 + 14.0 * np.clip(-v / 8.0, 0, 1)
            ax.scatter(I, J, K, s=size, linewidths=0, depthshade=True,
                       c=THEME["RED"], alpha=0.85 * a)
        phase = 2

    b = d["best"]
    if phase == 2 or frac > p1 * 0.85:
        ax.scatter([b[0]], [b[1]], [b[2]], s=260, c="white", marker="*",
                   depthshade=False, linewidths=0, zorder=10)
    box_edges(ax, shape)
    ax.set_xlim(0, shape[0] - 1); ax.set_ylim(0, shape[1] - 1)
    ax.set_zlim(0, shape[2] - 1)
    return phase


def axis_labels(ax, d, fs=10):
    """Projected 2D text drawn above every 3D collection."""
    M = ax.get_proj()
    nx, ny, nz = [s - 1 for s in d["s_is"].shape]
    f, s, b = d["fasts"], d["slows"], d["bands"]

    def put(x, y, z, txt, **kw):
        x2, y2, _ = proj3d.proj_transform(x, y, z, M)
        ax.text2D(x2, y2, txt, transform=ax.transData, zorder=1e6, ha="center",
                  va="center", color=THEME["TEXT_DIM"], family=THEME["MONO"],
                  fontsize=fs, **kw)

    put(nx / 2, -2.2, 0, f"fast MA {f[0]} to {f[-1]}d")
    put(nx + 2.2, ny / 2, 0, f"slow MA {s[0]} to {s[-1]}d")
    put(-1.5, -1.5, nz / 2, f"band\n0 to {b[-1] * 100:.0f}%")


def text_block(fig, figs, phase, s=1.0):
    L = LAYOUT
    fig.text(0.5, L["TITLE"], "THE ISLAND THAT SANK", ha="center",
             fontsize=28 * s, fontweight="bold", color=THEME["TEXT"],
             family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             f"SPY  |  {figs['n_settings']} moving-average rules  |  "
             f"grid {figs['grid']}",
             ha="center", fontsize=13 * s, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.5, L["KEY"],
             "surface = rules beating buy & hold  ·  red dot = lost, bigger = worse  ·  ★ best backtest",
             ha="center", fontsize=10 * s, color=THEME["TEXT_DIM"], family=THEME["FONT"])
    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10 * s,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])
    if phase == 1:
        note, col = f"BACKTEST  {figs['is_period']}", THEME["GREEN"]
    else:
        note, col = f"WHAT CAME NEXT  {figs['os_period']}", THEME["RED"]
    fig.text(0.5, L["NOTE"], note, ha="center", fontsize=12 * s, color=col,
             fontweight="bold", family=THEME["MONO"])
    fig.text(0.36, L["BIG"], f"{figs['share_is']} beat it", ha="center",
             fontsize=21 * s, color=THEME["GREEN"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["BIG"], "→", ha="center", fontsize=21 * s,
             color=THEME["TEXT_DIM"], fontweight="bold", family=THEME["FONT"])
    fig.text(0.64, L["BIG"], f"{figs['share_os']} beat it", ha="center",
             fontsize=21 * s, color=THEME["RED"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             f"best backtest {figs['best_is']}/yr  →  {figs['best_os']}/yr "
             f"vs buy & hold {figs['bh_os']}",
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
    ax = fig.add_axes(LAYOUT["AXES"], projection="3d", facecolor=THEME["BG"])
    phase = scene(ax, d, frac, c["PHASE1"])
    ax.view_init(elev=c["ELEV_BASE"] + 6 * np.sin(t * np.pi),
                 azim=c["AZIM_START"] + c["AZIM_SWEEP"] * t)
    ax.set_box_aspect((1.15, 1.15, 1.25), zoom=c["ZOOM"])
    style_axes(ax)
    axis_labels(ax, d)
    text_block(fig, figs, phase)

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
    prepare_geometry(d)
    return d


def main(smoke=False):
    c = CONFIG
    d = load()
    total = int(c["FPS"] * (c["HOOK_SEC"] + c["BUILD_SEC"] + c["HOLD_SEC"]))
    if smoke:
        out = os.path.join(BASE_DIR, "temp_frames_smoke")
        shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
        n_hook = int(c["FPS"] * c["HOOK_SEC"])
        n_build = int(c["FPS"] * c["BUILD_SEC"])
        picks = (0, n_hook + int(n_build * c["PHASE1"] * 0.98), total - 1)
        for k, idx in enumerate(picks):
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
