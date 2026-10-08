"""
SavingLoses_Reel_Pipeline.py
============================
Reel 30 for @quant.traderr, "Saving Is Losing". $10,000 parked in 1993, once
in cash that earns the 13-week T-bill rate, once in SPY with dividends
reinvested, both measured in 1993 dollars, so what the money can still buy.

THE DATA
    SPY daily closes from Yahoo, auto-adjusted (dividends reinvested), and the
    13-week T-bill yield ^IRX, 29 Jan 1993 to Sep 2026. US CPI (CPIAUCNS,
    monthly, not seasonally adjusted) from FRED, carried forward to each
    trading day; the last CPI print is Aug 2026. Real data throughout.

THE MATHS
    cash_t      = 10,000 * prod(1 + IRX_{s-1} / 100 / 252)
    invested_t  = 10,000 * SPY_t / SPY_0
    real_x_t    = x_t * CPI_0 / CPI_t
    Both pots are deflated by the same CPI, so the gap between them is not
    an artefact of the deflator.

    In 3D: a nautilus. Time runs clockwise round the circle, one spoke per
    month, starting at two o'clock and ending at twelve. Spoke length is the square root of the
    real value, so the AREA of each fan grows with the money. INVESTED fans
    out behind, coloured by its value. SAVED sits in front, pale grey while it is
    above its starting buying power, red while it is below. The dashed white
    circle is the starting $10,000.

THE CLAIM THE VISUAL MAKES
    Thirty-three years of interest left the saver with less buying power than
    they started with; the same money invested bought about thirteen times
    more.

WHAT THE VISUAL IS NOT
    Not a real savings account: T-bills paid more than most bank accounts did,
    so a typical saver did worse than this line. No taxes on either pot, and
    tax on interest would push the cash lower still. Not a claim that cash is
    useless: it did not fall 50% in 2008. One country, one index, one stretch.

RUN
    python SavingLoses_Reel_Pipeline.py --smoke   # 3 frames -> temp_frames_smoke/
    python SavingLoses_Reel_Pipeline.py           # full render -> topic.mp4

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
    "CPI_URL": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCNS",
    "START": "1993-01-29",
    "AMOUNT": 10_000,
    "SPAN_DEG": 330,
    "START_DEG": 60,
    "DEPTH": 0.18,

    # render (leave these alone unless the topic needs landscape)
    "W": 1080, "H": 1920, "DPI": 100,
    "FPS": 30,
    "HOOK_SEC": 1.4, "BUILD_SEC": 8.5, "HOLD_SEC": 1.8,
    "ELEV_BASE": 12, "AZIM_START": -112, "AZIM_SWEEP": 36,
    "ZOOM": 1.42,
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


def usd(s):
    """matplotlib reads a pair of $ as maths; escape them for the frame."""
    return s.replace("$", r"\$")


# ---------------------------------------------------------------- DATA
def fetch():
    """yfinance and FRED once, then CSV caches. Delete _cache/ to refetch."""
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
        if px.empty or any(t not in px.columns for t in CONFIG["TICKERS"]):
            raise SystemExit("no data returned; refusing to draw a synthetic tape")
        px = px[CONFIG["TICKERS"]]
        px.to_csv(path)
    cpath = os.path.join(CACHE_DIR, "CPIAUCNS.csv")
    if not os.path.exists(cpath):
        import urllib.request
        log("[Data] fetching CPIAUCNS from FRED")
        try:
            with urllib.request.urlopen(CONFIG["CPI_URL"], timeout=60) as r:
                raw = r.read()
        except Exception as e:
            raise SystemExit(f"FRED fetch failed ({e}); refusing to draw a synthetic series")
        if b"CPIAUCNS" not in raw[:200]:
            raise SystemExit("FRED returned something that is not the CPI csv")
        with open(cpath, "wb") as fh:
            fh.write(raw)
    cpi = pd.read_csv(cpath, index_col=0, parse_dates=True, na_values=["."]).iloc[:, 0]
    cpi = cpi.dropna().astype(float)
    if px["SPY"].dropna().empty or px["^IRX"].dropna().empty or cpi.empty:
        raise SystemExit("a series came back empty; refusing to continue")
    return px, cpi


# ------------------------------------------------------------- COMPUTE
def compute(data):
    px, cpi = data
    c = CONFIG
    s = px["SPY"].dropna().loc[c["START"]:]
    irx = px["^IRX"].reindex(s.index).ffill()
    if irx.isna().any():
        raise SystemExit("T-bill series has holes before its first value")
    # CPI is monthly; each trading day uses the latest print at or before it
    cpi_d = cpi.reindex(cpi.index.union(s.index)).ffill().reindex(s.index)
    if cpi_d.isna().any():
        raise SystemExit("CPI does not cover the start of the window")
    defl = cpi_d.values[0] / cpi_d.values
    cash = c["AMOUNT"] * np.cumprod(np.r_[1.0, 1 + irx.values[:-1] / 100 / 252])
    inv = c["AMOUNT"] * s.values / s.values[0]
    r_cash, r_inv = cash * defl, inv * defl

    # one spoke per month: the last trading day of each month
    me = pd.Series(np.arange(len(s)), s.index).groupby(s.index.to_period("M")).last().values
    return {"s": s, "dates": s.index, "cash": cash, "inv": inv,
            "r_cash": r_cash, "r_inv": r_inv, "me": me,
            "infl": float(cpi_d.values[-1] / cpi_d.values[0]),
            "cpi_last": cpi.index[-1]}


# ------------------------------------------------------------ VALIDATE
def validate(d):
    """Every number that reaches a frame is asserted here, before any render."""
    A = CONFIG["AMOUNT"]
    rc, ri = float(d["r_cash"][-1]), float(d["r_inv"][-1])
    yrs = (d["dates"][-1] - d["dates"][0]).days / 365.25
    below = float((d["r_cash"] < A).mean())

    assert 1.5 <= d["infl"] <= 4.0, f"price level x{d['infl']:.2f} outside band"
    assert 0.5 * A <= rc <= 3 * A, f"real cash ${rc:,.0f} outside band"
    assert 2 * A <= ri <= 100 * A, f"real invested ${ri:,.0f} outside band"
    assert rc < A, "the saver kept their buying power; the reel's claim is gone"
    assert float(d["cash"][-1]) > A, "nominal cash below the start; check ^IRX"
    assert 30 <= yrs <= 40
    assert len(d["s"]) > 8000

    figs = {
        "amount": f"${A:,.0f}",
        "cash_nom": f"${d['cash'][-1]:,.0f}",
        "inv_nom": f"${d['inv'][-1]:,.0f}",
        "cash_real": f"${rc:,.0f}",
        "inv_real": f"${ri:,.0f}",
        "cash_chg": f"{(rc / A - 1) * 100:+.0f}%",
        "ratio": f"{ri / rc:.0f}x",
        "infl": f"x{d['infl']:.2f}",
        "years": f"{yrs:.1f}",
        "below_share": f"{below * 100:.0f}%",
        "start": d["dates"][0].strftime("%b %Y"),
        "end": d["dates"][-1].strftime("%b %Y"),
        "cpi_last": d["cpi_last"].strftime("%b %Y"),
    }
    with open(os.path.join(BASE_DIR, "figures.json"), "w") as fh:
        json.dump(figs, fh, indent=1)
    log(f"[validate] {figs}")
    d["figs"] = figs
    return figs


# -------------------------------------------------------------- RENDER
def _geom(d):
    """Spoke angles and lengths. Length = sqrt(real value / largest value),
    so fan area tracks money."""
    c = CONFIG
    me = d["me"]
    K = len(me)
    th = np.deg2rad(c["START_DEG"] - c["SPAN_DEG"] * np.arange(K) / (K - 1))
    top = d["r_inv"][me].max()
    ri = np.sqrt(d["r_inv"][me] / top)
    rc = np.sqrt(d["r_cash"][me] / top)
    r0 = np.sqrt(c["AMOUNT"] / top)
    return th, ri, rc, r0, K


def _fan(ax, th, r, y, cols, edge):
    tris = []
    for k in range(len(th) - 1):
        tris.append([(0, y, 0),
                     (r[k] * np.cos(th[k]), y, r[k] * np.sin(th[k])),
                     (r[k + 1] * np.cos(th[k + 1]), y, r[k + 1] * np.sin(th[k + 1]))])
    if tris:
        pc = Poly3DCollection(tris, facecolors=cols[:-1], edgecolors=cols[:-1],
                              linewidths=0.3, alpha=0.95)
        ax.add_collection3d(pc)
        ax.plot(r * np.cos(th), np.full_like(th, y), r * np.sin(th),
                color=edge, lw=1.6)


def scene(ax, d, frac):
    c = CONFIG
    th, ri, rc, r0, K = _geom(d)
    n = int(round(frac * (K - 1))) + 1 if frac > 0 else 0
    A = c["AMOUNT"]
    me = d["me"]
    if n >= 2:
        ci = CMAP(0.05 + 0.72 * ri[:n] ** 2)
        _fan(ax, th[:n], ri[:n], c["DEPTH"], ci, "white")
        cc = np.tile(matplotlib.colors.to_rgba("#cfd6dc"), (n, 1))
        cc[d["r_cash"][me[:n]] < A] = matplotlib.colors.to_rgba(THEME["RED"])
        _fan(ax, th[:n], rc[:n], -c["DEPTH"], cc, "white")
    # the starting $10,000: dashed ring, there from the hook frame on
    t = np.linspace(0, 2 * np.pi, 120)
    ax.plot(r0 * np.cos(t), np.full_like(t, -c["DEPTH"] - 0.01), r0 * np.sin(t),
            color="white", lw=1.2, ls=(0, (3, 3)), alpha=0.9)
    ax.set_xlim(-0.9, 0.9); ax.set_ylim(-1, 1); ax.set_zlim(-0.72, 1.07)
    return me[n - 1] if n >= 1 else 0


def style_axes(ax):
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
        pane.line.set_color((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def scene_labels(ax, d, last, fs=11):
    """Projected to 2D so the fans cannot depth-sort the labels away."""
    from mpl_toolkits.mplot3d import proj3d
    M = ax.get_proj()
    c, figs = CONFIG, d["figs"]
    th, ri, rc, r0, K = _geom(d)

    def put(x, y, z, s, **kw):
        x2, y2, _ = proj3d.proj_transform(x, y, z, M)
        ax.text2D(x2, y2, s, transform=ax.transData, zorder=1e6, ha="center",
                  va="center", **kw)

    put(0, -c["DEPTH"], 0.04, "SAVED", color="black", fontsize=fs + 2,
        fontweight="bold", family=THEME["FONT"])
    if last == len(d["s"]) - 1:
        a = th[-1] - 0.3
        put(0.62 * np.cos(a), c["DEPTH"], 0.62 * np.sin(a),
            usd(f"INVESTED\n{figs['inv_real']}"), color=THEME["ORANGE"],
            fontsize=fs + 3, fontweight="bold", family=THEME["MONO"])
        put(0, -c["DEPTH"], -0.06, usd(figs["cash_real"]), color=THEME["RED"],
            fontsize=fs + 3, fontweight="bold", family=THEME["MONO"])


def text_block(fig, figs, scale=1.0):
    L, s = LAYOUT, scale
    fig.text(0.5, L["TITLE"], "SAVING IS LOSING", ha="center",
             fontsize=29 * s, fontweight="bold", color=THEME["TEXT"],
             family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             usd(f"{figs['amount']} in 1993  |  savings at the T-bill rate vs SPY  |  "
                 f"in 1993 dollars"),
             ha="center", fontsize=12 * s, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.5, L["KEY"],
             "one spoke per month, clockwise  ·  fan area = buying power  ·  red = below start",
             ha="center", fontsize=10 * s, color=THEME["TEXT_DIM"], family=THEME["FONT"])
    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10 * s,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])
    fig.text(0.5, L["BIG"], f"{figs['years']} YEARS OF SAVING: {figs['cash_chg']}",
             ha="center", fontsize=22 * s, color=THEME["RED"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             usd(f"saved {figs['cash_real']}  ·  invested {figs['inv_real']}  ·  "
                 f"after inflation"),
             ha="center", fontsize=11.5 * s, color=THEME["TEXT_DIM"],
             family=THEME["MONO"])


def note(fig, d, last, scale=1.0):
    fig.text(0.5, LAYOUT["NOTE"], f"{d['dates'][last].year}",
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
    ax.set_box_aspect((0.9, 0.6, 0.87), zoom=c["ZOOM"])
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
