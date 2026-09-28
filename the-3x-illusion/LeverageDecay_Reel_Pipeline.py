"""
LeverageDecay_Reel_Pipeline.py
==============================
Reel 16 for @quant.traderr, "The 3x Illusion". Every one-year window of SPY
since 1993, started on every trading day, run through a 3x product that is
rebalanced every day, and held against what "3x" seems to promise: three times
the year's return.

THE MATHS
    r_t is the daily simple return of SPY from adjusted closes (dividends
    included). For each start day s and the next 252 trading days:

        SPY year        R  = prod(1 + r_t) - 1
        3x promised     3R
        3x daily        L  = prod(1 + 3 r_t) - 1
        gap             L - 3R, in percentage points
        realized vol    std of daily log returns * sqrt(252)

    The 3x daily line is a calculation on real SPY returns, NOT a real ETF:
    no management fee, no financing cost of the borrowed exposure, no
    tracking error. Real 3x funds carry those on top, so they would do worse
    than this line, not better.

    The glass lens is the continuous-time approximation of the same gap,
    (1 + R)^3 * exp(-3 sigma^2) - 1 - 3R. Its flat face is the zero-gap plane,
    "exactly what 3x promises". Its curved face hangs below that plane
    wherever volatility eats more than compounding adds: that pocket is
    volatility decay. The formula only draws the glass; every number on screen
    comes from the daily data, and validate() asserts that the data and the
    formula agree (correlation above 0.95) before a frame is drawn.

    In 3D: SPY's one-year return runs across, realized volatility runs into
    depth, the gap goes up. Every dot is one window, coloured by its gap, and
    the windows appear in date order.

THE CLAIM THE VISUAL MAKES
    In most one-year windows the daily 3x product delivered less than three
    times SPY's year, and the share rises sharply with the volatility of the
    year: calm years mostly beat the promise, wild years mostly miss it.

WHAT THE VISUAL IS NOT
    Not a backtest of any fund you can buy: fees and financing are left out,
    and they only make it worse. Not an argument that 3x never pays: in calm
    trending years compounding beats the promise, and the dots above the
    glass are exactly those years. The windows overlap heavily, day-by-day
    starts share almost the whole year with their neighbours, so the 8,000
    windows are not 8,000 independent trials: this is about 33 years of one
    index. The glass is a model, drawn for orientation; the counts are not
    read from it. Only windows inside the display box are drawn (the most
    extreme crash and rebound years fall outside it); every figure uses all
    of them.

RUN
    python LeverageDecay_Reel_Pipeline.py --smoke   # 3 frames -> temp_frames_smoke/
    python LeverageDecay_Reel_Pipeline.py           # full render -> topic.mp4

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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "_cache")

CONFIG = {
    "TICKERS": ["SPY"],
    "PERIOD": "max",
    "WINDOW": 252,            # trading days in one holding year
    "LEV": 3.0,               # daily leverage factor
    "X_LO": -35.0,            # display box: SPY one-year return, percent
    "X_HI": 60.0,
    "Y_LO": 0.0,              # display box: realized vol, percent
    "Y_HI": 34.0,
    "Z_LO": -45.0,            # display box: gap, percentage points
    "Z_HI": 60.0,
    "C_SPAN": 40.0,           # colour saturates at +-40 points of gap

    # render
    "W": 1080, "H": 1920, "DPI": 100,
    "FPS": 30,
    "HOOK_SEC": 1.4, "BUILD_SEC": 8.5, "HOLD_SEC": 1.8,
    "ELEV_BASE": 10, "AZIM_START": -102, "AZIM_SWEEP": 22,
    "ZOOM": 1.4,
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
    """yfinance once, then a versioned CSV cache. Delete _cache/ to refetch.

    A bar dated today is dropped: during market hours it is a moving price,
    not a close, and the cache would freeze it."""
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
    px = px[px.index.normalize() < pd.Timestamp.today().normalize()]
    px.to_csv(path)
    return px


# ------------------------------------------------------------- COMPUTE
def lens_gap(R, sig, lev=3.0):
    """Continuous-time gap of a daily-rebalanced lev-x product, as a fraction.
    1 + L = (1 + R)^lev * exp(-lev (lev - 1) / 2 * sigma^2)."""
    return (1 + R) ** lev * np.exp(-lev * (lev - 1) / 2 * sig ** 2) - 1 - lev * R


def compute(px):
    s = px["SPY"].dropna()
    r = s.pct_change().dropna()
    v = r.values
    N, k = CONFIG["WINDOW"], CONFIG["LEV"]
    n = len(v) - N + 1
    idx = np.arange(n)[:, None] + np.arange(N)[None, :]
    D = v[idx]                                            # (windows, 252)
    spy = np.prod(1 + D, axis=1) - 1
    lev = np.prod(1 + k * D, axis=1) - 1
    prom = k * spy
    gap = lev - prom
    vol = np.std(np.log1p(D), axis=1, ddof=1) * np.sqrt(252)
    return {"spy": spy, "lev": lev, "prom": prom, "gap": gap, "vol": vol,
            "th": lens_gap(spy, vol, k),
            "start": r.index[:n], "end_date": r.index[N - 1:]}


# ------------------------------------------------------------ VALIDATE
def validate(d):
    """Every number that reaches a frame is asserted here, before any render."""
    gap, vol, spy, lev = d["gap"], d["vol"], d["spy"], d["lev"]
    n = len(gap)
    below = gap < 0
    p_below = float(below.mean() * 100)
    v_med = float(np.median(vol))
    calm, wild = vol < v_med, vol >= v_med
    p_calm = float(below[calm].mean() * 100)
    p_wild = float(below[wild].mean() * 100)
    p_lost_up = float(((lev < 0) & (spy > 0)).mean() * 100)
    med_prom = float(np.median(d["prom"]) * 100)
    med_lev = float(np.median(lev) * 100)
    worst = float(gap.min() * 100)
    worst_year = int(d["start"][int(np.argmin(gap))].year)
    corr = float(np.corrcoef(d["th"], gap)[0, 1])

    assert n > 6000, f"only {n} windows"
    assert 30.0 <= p_below <= 85.0, f"{p_below:.0f}% below promise"
    assert 0.0 <= p_calm < p_wild <= 100.0, \
        f"calm {p_calm:.0f}% vs wild {p_wild:.0f}%: the vol link is the claim"
    assert 5.0 <= v_med * 100 <= 30.0, f"median vol {v_med:.2%}"
    assert 0.0 < p_lost_up < 25.0, f"{p_lost_up:.1f}% lost while SPY rose"
    assert med_lev < med_prom, "median 3x daily should trail median promise"
    assert -100.0 < worst < -10.0, f"worst gap {worst:.0f}"
    assert corr > 0.95, f"lens formula disagrees with the data (corr {corr:.3f})"
    assert d["start"][0].year == 1993, "history should start in 1993"

    figs = {
        "windows": f"{n:,}",
        "below": f"{p_below:.0f}%",
        "calm": f"{p_calm:.0f}%",
        "wild": f"{p_wild:.0f}%",
        "vol_split": f"{v_med * 100:.0f}%",
        "lost_up": f"{p_lost_up:.0f}%",
        "med_prom": f"+{med_prom:.0f}%",
        "med_lev": f"+{med_lev:.0f}%",
        "worst": f"{worst:.0f} points",
        "worst_year": str(worst_year),
        "first_year": str(d["start"][0].year),
        "last_year": str(d["end_date"][-1].year),
        "window": str(CONFIG["WINDOW"]),
        "lev": f"{CONFIG['LEV']:.0f}x",
    }
    with open(os.path.join(BASE_DIR, "figures.json"), "w") as fh:
        json.dump(figs, fh, indent=1)
    log(f"[validate] {figs}  (lens corr {corr:.4f})")

    # running counter for the build: windows appear in date order, share of
    # those shown so far that fell short of the promise
    cum = np.cumsum(below) / np.arange(1, n + 1) * 100.0
    d["run"] = [f"{x:.0f}%" for x in cum]
    d["run_n"] = [f"{i + 1:,}" for i in range(n)]
    assert d["run"][-1] == figs["below"], "running counter disagrees with figures"
    assert d["run_n"][-1] == figs["windows"]
    d["figs"] = figs
    return figs


def _roots(sig, lev):
    """Both zero-gap returns at volatility sig (fractions), on a fine grid."""
    R = np.linspace(-0.33, 1.2, 6001)
    g = lens_gap(R, sig, lev)
    sc = np.nonzero(np.diff(np.sign(g)) != 0)[0]
    if len(sc) < 2:
        return 0.0, 0.0
    a, b = sc[0], sc[-1]
    r1 = R[a] - g[a] * (R[a + 1] - R[a]) / (g[a + 1] - g[a])
    r2 = R[b] - g[b] * (R[b + 1] - R[b]) / (g[b + 1] - g[b])
    return r1, r2


def prepare_geometry(d):
    c = CONFIG
    x, y, z = d["spy"] * 100, d["vol"] * 100, d["gap"] * 100
    inside = ((x >= c["X_LO"]) & (x <= c["X_HI"]) & (y >= c["Y_LO"])
              & (y <= c["Y_HI"]) & (z >= c["Z_LO"]) & (z <= c["Z_HI"]))
    d["inside"] = inside
    d["x"], d["y"], d["z"] = x, y, z
    cn = np.clip(0.5 - z / (2 * c["C_SPAN"]), 0, 1)      # negative gap -> red
    d["pcol"] = CMAP(cn)

    # the lens: flat face on the zero-gap plane, curved face = decay pocket
    sig = np.linspace(0.0, c["Y_HI"] / 100, 46)
    u = np.linspace(0, 1, 41)
    RR = np.zeros((len(sig), len(u)))
    for i, s in enumerate(sig):
        r1, r2 = _roots(s, c["LEV"])
        RR[i] = r1 + u * (r2 - r1)
    SS = np.repeat(sig[:, None], len(u), axis=1)
    ZZ = np.minimum(lens_gap(RR, SS, c["LEV"]), 0.0)
    d["LX"], d["LY"], d["LZ"] = RR * 100, SS * 100, ZZ * 100
    lc = np.clip(0.5 - d["LZ"] / (2 * c["C_SPAN"]), 0, 1)
    fc = CMAP(lc)
    fc[..., 3] = 0.30
    d["lens_fc"] = fc
    return d


# -------------------------------------------------------------- RENDER
def render_frame(idx, total, d, out_path):
    c, figs, L = CONFIG, d["figs"], LAYOUT
    if "LX" not in d:
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
    n = len(d["gap"])
    m = max(int(round(n * (0.06 + 0.94 * frac))), 1)          # windows shown

    fig = plt.figure(figsize=(c["W"] / c["DPI"], c["H"] / c["DPI"]),
                     dpi=c["DPI"], facecolor=THEME["BG"])
    fig.text(0.5, L["TITLE"], "THE 3x ILLUSION", ha="center", fontsize=28,
             fontweight="bold", color=THEME["TEXT"], family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             f"SPY  |  {figs['windows']} one-year windows  |  "
             f"{figs['first_year']} to {figs['last_year']}  |  {figs['lev']} rebalanced daily",
             ha="center", fontsize=13, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.5, L["KEY"],
             "glass: what 3x promises   lens below it: volatility decay   dot: one year",
             ha="center", fontsize=10, color=THEME["TEXT_DIM"], family=THEME["FONT"])
    fig.text(0.5, L["ROW_A"],
             f"median year   SPY x3 {figs['med_prom']}   3x daily {figs['med_lev']}",
             ha="center", va="center", fontsize=9, color=THEME["TEXT"],
             family=THEME["MONO"])
    fig.text(0.5, L["ROW_B"],
             f"fell short   calm years {figs['calm']}   wild years {figs['wild']}",
             ha="center", va="center", fontsize=9, color=THEME["YELLOW"],
             family=THEME["MONO"])
    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])

    ax = fig.add_axes(L["AXES"], projection="3d", facecolor=THEME["BG"],
                      computed_zorder=False)

    # the zero-gap plane over the whole box, faint: "3x exactly"
    X0, X1, Y0, Y1 = c["X_LO"], c["X_HI"], c["Y_LO"], c["Y_HI"]
    for g in np.linspace(X0, X1, 11):
        ax.plot([g, g], [Y0, Y1], [0, 0], color=THEME["CYAN"], lw=0.4,
                alpha=0.22, zorder=1)
    for g in np.linspace(Y0, Y1, 9):
        ax.plot([X0, X1], [g, g], [0, 0], color=THEME["CYAN"], lw=0.4,
                alpha=0.22, zorder=1)

    # the lens: curved decay face, flat glass face, bright rim
    LX, LY, LZ = d["LX"], d["LY"], d["LZ"]
    ax.plot_surface(LX, LY, LZ, facecolors=d["lens_fc"], linewidth=0.25,
                    edgecolor=to_rgba(THEME["TEXT"], 0.10), shade=False,
                    antialiased=True, zorder=2)
    ax.plot_surface(LX, LY, np.zeros_like(LZ), color=THEME["CYAN"], alpha=0.16,
                    linewidth=0, shade=False, zorder=2)
    ax.plot(LX[:, 0], LY[:, 0], 0 * LY[:, 0], color=THEME["TEXT"], lw=1.2,
            alpha=0.9, zorder=5)
    ax.plot(LX[:, -1], LY[:, -1], 0 * LY[:, -1], color=THEME["TEXT"], lw=1.2,
            alpha=0.9, zorder=5)
    ax.plot(LX[-1], LY[-1], LZ[-1], color=THEME["TEXT"], lw=0.9, alpha=0.7,
            zorder=5)
    cap = [list(zip(LX[-1], LY[-1], LZ[-1]))]
    ax.add_collection3d(Poly3DCollection(cap, facecolor=to_rgba(THEME["RED"], 0.18),
                                         edgecolor="none", zorder=2))

    # the windows, in date order
    sel = np.nonzero(d["inside"][:m])[0]
    below = d["z"][sel] < 0
    for mask, zo in ((~below, 3), (below, 4)):
        ii = sel[mask]
        ax.scatter(d["x"][ii], d["y"][ii], d["z"][ii], s=2.2, c=d["pcol"][ii],
                   alpha=0.75, linewidths=0, depthshade=False, zorder=zo)
    if frac < 1.0:                                           # the newest windows
        new = sel[sel >= m - 90]
        ax.scatter(d["x"][new], d["y"][new], d["z"][new], s=9,
                   color=THEME["TEXT"], alpha=0.9, linewidths=0,
                   depthshade=False, zorder=6)

    ax.set_xlim(X0, X1); ax.set_ylim(Y0, Y1); ax.set_zlim(c["Z_LO"], c["Z_HI"])
    ax.view_init(elev=c["ELEV_BASE"] + 5 * np.sin(t * np.pi),
                 azim=c["AZIM_START"] + c["AZIM_SWEEP"] * t)
    ax.set_box_aspect((1.5, 1.2, 1.25), zoom=c["ZOOM"])
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    # Readout: strings only, precomputed and asserted in validate().
    if frac < 1.0:
        yr = d["end_date"][m - 1].year
        fig.text(0.5, L["NOTE"],
                 f"year to {yr}   windows {d['run_n'][m - 1]:>5}   "
                 f"short of 3x so far {d['run'][m - 1]}",
                 ha="center", fontsize=11, color=THEME["TEXT"], family=THEME["MONO"])
    else:
        fig.text(0.5, L["NOTE"],
                 "across: SPY year   deep: volatility   up: 3x daily minus 3x SPY",
                 ha="center", fontsize=10, color=THEME["TEXT_DIM"],
                 family=THEME["FONT"])
    fig.text(0.5, L["BIG"], f"{figs['below']} OF YEARS: 3x DAILY FELL SHORT",
             ha="center", fontsize=19, color=THEME["YELLOW"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             "of three times the year SPY actually had",
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
