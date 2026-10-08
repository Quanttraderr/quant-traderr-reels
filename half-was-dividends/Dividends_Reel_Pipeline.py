"""
Dividends_Reel_Pipeline.py
==========================
Reel 31 for @quant.traderr, "Half Was Dividends". $10,000 in SPY, measured
two ways: the price chart everybody looks at, and the same holding with every
dividend reinvested. The gap between them is money the price chart never
shows.

THE DATA
    SPY daily from Yahoo with auto_adjust OFF: "Close" is the traded price
    (SPY has never split), "Adj Close" folds every dividend back in.
    29 Jan 1993 to 30 Sep 2026. Real data throughout.

THE MATHS
    price_t = 10,000 * Close_t     / Close_0
    total_t = 10,000 * AdjClose_t  / AdjClose_0
    dividend share = (total - price) / total at the end. VALIDATE also checks
    that total never falls below price, which would mean a broken series.

    In 3D: a sediment block. Time runs along the long axis, the height is the
    dollar value on a linear scale. The bottom stratum, blue to cyan, is the
    price-only value. The top stratum, orange to red, is what reinvested
    dividends added on top. The block grows month by month and its end face
    shows both layers in section.

THE CLAIM THE VISUAL MAKES
    Close to half of what a SPY holder has today came from dividends, money
    that never shows up on the price chart.

WHAT THE VISUAL IS NOT
    Not a claim about dividend stocks beating other stocks: it is the same
    index, counted with and without its own payouts. No taxes on dividends,
    which a taxable account pays every year. The dividend layer includes the
    growth on reinvested dividends, not just the cash paid out. Linear height
    makes the early years look flat; the readout numbers are exact.

RUN
    python Dividends_Reel_Pipeline.py --smoke   # 3 frames -> temp_frames_smoke/
    python Dividends_Reel_Pipeline.py           # full render -> topic.mp4

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

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "_cache")

CONFIG = {
    "START": "1993-01-29",
    "END": "2026-09-30",
    "AMOUNT": 10_000,
    "DEPTH": 0.3,

    # render (leave these alone unless the topic needs landscape)
    "W": 1080, "H": 1920, "DPI": 100,
    "FPS": 30,
    "HOOK_SEC": 1.4, "BUILD_SEC": 8.5, "HOLD_SEC": 1.8,
    "ELEV_BASE": 16, "AZIM_START": -58, "AZIM_SWEEP": 18,
    "ZOOM": 1.1,
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
    """yfinance once (unadjusted close AND adjusted close), then a CSV cache.
    Delete _cache/ to refetch."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "spy_raw.csv")
    if os.path.exists(path):
        log(f"[Data] cache {path}")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        import yfinance as yf
        log("[Data] fetching SPY, unadjusted")
        df = yf.download("SPY", start=CONFIG["START"], interval="1d",
                         progress=False, auto_adjust=False)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df[["Close", "Adj Close"]].dropna()
        if df.empty:
            raise SystemExit("no data returned; refusing to draw a synthetic tape")
        df.to_csv(path)
    return df.loc[CONFIG["START"]:CONFIG["END"]]


# ------------------------------------------------------------- COMPUTE
def compute(df):
    c = CONFIG
    price = c["AMOUNT"] * df["Close"].values / df["Close"].values[0]
    total = c["AMOUNT"] * df["Adj Close"].values / df["Adj Close"].values[0]
    me = pd.Series(np.arange(len(df)), df.index).groupby(df.index.to_period("M")).last().values
    return {"df": df, "dates": df.index, "price": price, "total": total, "me": me}


# ------------------------------------------------------------ VALIDATE
def validate(d):
    """Every number that reaches a frame is asserted here, before any render."""
    A = CONFIG["AMOUNT"]
    p, t = float(d["price"][-1]), float(d["total"][-1])
    share = (t - p) / t
    yrs = (d["dates"][-1] - d["dates"][0]).days / 365.25

    assert (d["total"] >= d["price"] * 0.999).all(), "total return below price somewhere"
    assert 2 * A <= p <= 100 * A, f"price-only ${p:,.0f} outside band"
    assert p < t <= 200 * A, f"total ${t:,.0f} outside band"
    assert 0.15 <= share <= 0.70, f"dividend share {share:.2f} outside 15%..70%"
    assert 30 <= yrs <= 40
    assert len(d["df"]) > 8000

    figs = {
        "amount": f"${A:,.0f}",
        "price_usd": f"${p:,.0f}",
        "total_usd": f"${t:,.0f}",
        "div_usd": f"${t - p:,.0f}",
        "share": f"{share * 100:.0f}%",
        "price_x": f"{p / A:.1f}x",
        "total_x": f"{t / A:.1f}x",
        "years": f"{yrs:.1f}",
        "start": d["dates"][0].strftime("%b %Y"),
        "end": d["dates"][-1].strftime("%b %Y"),
    }
    with open(os.path.join(BASE_DIR, "figures.json"), "w") as fh:
        json.dump(figs, fh, indent=1)
    log(f"[validate] {figs}")
    d["figs"] = figs
    return figs


# -------------------------------------------------------------- RENDER
def _face(ax, x, y, z0, z1, cols, zo=3):
    """A vertical face at depth y between z0 and z1, one colour per column."""
    X = np.vstack([x, x]); Y = np.full_like(X, y)
    Z = np.vstack([z0, z1])
    ax.plot_surface(X, Y, Z, facecolors=np.asarray(cols)[None, :-1, :],
                    rstride=1, cstride=1, linewidth=0, antialiased=False,
                    shade=False, zorder=zo)


def scene(ax, d, frac):
    c = CONFIG
    me = d["me"]
    K = len(me)
    n = int(round(frac * (K - 1))) + 1 if frac > 0 else 0
    top = d["total"].max()
    D = c["DEPTH"]
    if n >= 2:
        sel = me[:n]
        x = np.arange(n) / (K - 1)
        zp = d["price"][sel] / top
        zt = d["total"][sel] / top
        tt = np.arange(n) / (K - 1)
        cp = CMAP(0.02 + 0.30 * tt)                      # blue -> cyan
        cd = CMAP(0.70 + 0.30 * tt)                      # orange -> red
        _face(ax, x, -D, np.zeros(n), zp, cp)            # PRICE stratum, front
        _face(ax, x, -D, zp, zt, cd)                     # DIVIDEND stratum, front
        # lid: the top of the block across its depth
        X = np.vstack([x, x]); Y = np.vstack([np.full(n, -D), np.full(n, D)])
        Z = np.vstack([zt, zt])
        lid = np.asarray(cd).copy(); lid[:, :3] *= 0.75
        ax.plot_surface(X, Y, Z, facecolors=lid[None, :-1, :], rstride=1,
                        cstride=1, linewidth=0, antialiased=False, shade=False,
                        zorder=1)
        # end cap at the current month, showing both strata in section
        for (z0, z1, col) in ((0, zp[-1], cp[-1]), (zp[-1], zt[-1], cd[-1])):
            ax.plot_surface(np.full((2, 2), x[-1]), np.array([[-D, D], [-D, D]]),
                            np.array([[z0, z0], [z1, z1]]),
                            color=tuple(np.clip(np.array(col[:3]) * 0.85, 0, 1)),
                            linewidth=0, shade=False, zorder=2)
        ax.plot(x, np.full(n, -D), zp, color="white", lw=1.2, alpha=0.9, zorder=4)
        ax.plot(x, np.full(n, -D), zt, color="white", lw=1.6, zorder=4)
    ax.plot([0, 1, 1, 0, 0], [-D, -D, D, D, -D], [0] * 5,
            color=THEME["TEXT_DIM"], lw=0.8, alpha=0.6, zorder=0)
    ax.set_xlim(0, 1); ax.set_ylim(-0.6, 0.6); ax.set_zlim(0, 1.05)
    return me[n - 1] if n >= 1 else 0


def style_axes(ax):
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
        pane.line.set_color((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def scene_labels(ax, d, last, fs=11):
    """Projected to 2D so the block cannot depth-sort the labels away."""
    from mpl_toolkits.mplot3d import proj3d
    M = ax.get_proj()
    c, figs = CONFIG, d["figs"]
    top = d["total"].max()

    def put(x, y, z, s, **kw):
        x2, y2, _ = proj3d.proj_transform(x, y, z, M)
        ax.text2D(x2, y2, s, transform=ax.transData, zorder=1e6, ha="center",
                  va="center", **kw)

    put(0.04, -c["DEPTH"], -0.05, str(d["dates"][0].year), color=THEME["TEXT_DIM"],
        fontsize=fs, family=THEME["MONO"])
    if last == len(d["df"]) - 1:
        zp, zt = d["price"][-1] / top, d["total"][-1] / top
        put(0.62, -c["DEPTH"], (zp + zt) / 2 + 0.12,
            usd(f"DIVIDENDS\n+{figs['div_usd']}"), color=THEME["ORANGE"],
            fontsize=fs + 3, fontweight="bold", family=THEME["MONO"])
        put(0.74, -c["DEPTH"], 0.14, usd(f"PRICE\n{figs['price_usd']}"),
            color="black", fontsize=fs + 3, fontweight="bold", family=THEME["MONO"])


def text_block(fig, figs, scale=1.0):
    L, s = LAYOUT, scale
    fig.text(0.5, L["TITLE"], "HALF WAS DIVIDENDS", ha="center",
             fontsize=28 * s, fontweight="bold", color=THEME["TEXT"],
             family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             f"{figs['amount']} in SPY  |  {figs['start']} to {figs['end']}".replace("$", r"\$"),
             ha="center", fontsize=13 * s, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.5, L["KEY"],
             "bottom layer = price chart only  ·  top layer = dividends reinvested",
             ha="center", fontsize=10 * s, color=THEME["TEXT_DIM"], family=THEME["FONT"])
    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10 * s,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])
    fig.text(0.5, L["BIG"], f"DIVIDENDS: {figs['share']} OF THE MONEY",
             ha="center", fontsize=22 * s, color=THEME["ORANGE"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             usd(f"price only {figs['price_usd']}  ·  with dividends {figs['total_usd']}"),
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
    ax.computed_zorder = False     # the block is drawn back to front by hand
    last = scene(ax, d, frac)
    note(fig, d, last)
    ax.view_init(elev=c["ELEV_BASE"] + 5 * np.sin(t * np.pi),
                 azim=c["AZIM_START"] + c["AZIM_SWEEP"] * t)
    ax.set_box_aspect((1.6, 0.6, 1.5), zoom=c["ZOOM"])
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
