"""
HoldingPeriod_Reel_Pipeline.py
==============================
Reel 22 for @quant.traderr, "Time Kills Risk". How often the S&P 500 was
below where you bought it, depending only on how long you held.

THE DATA
    S&P 500 index daily closes (^GSPC) from Yahoo, Dec 1927 to Sep 2026.
    This is the PRICE index: dividends are not included, and nothing is
    adjusted for inflation. Real data throughout. Nothing here is simulated.

THE MATHS
    For every start day t and holding period h (1, 21, 252, 1260, 2520 and
    5040 sessions, read as 1 day, 1 month, 1 year, 5, 10 and 20 years):
        r_t(h) = close_{t+h} / close_t - 1
    Start days without h sessions ahead are dropped. The loss share is the
    fraction of start days with r_t(h) < 0.

    In 3D: one curtain per holding period, the 1-day curtain at the back and
    the 20-year curtain in front. Along each curtain the start days are
    sorted from worst to best outcome (percentile 0 to 100), and the curtain
    height is that outcome. The part below the floor is red: those are the
    start days that lost money, so the red stretch is exactly the loss share,
    and a white post marks where the curtain crosses zero. Each curtain is
    scaled by its own 95th percentile of |r|, because a 20-year return and a
    1-day return do not fit one axis; heights compare shapes, not sizes.

THE CLAIM THE VISUAL MAKES
    The chance of being down shrinks as the holding period grows: 46% of
    single days lost money, 3.5% of 20-year holds did.

WHAT THE VISUAL IS NOT
    Not a promise that 20 years is safe. Overlapping windows share most of
    their days, so the 20-year sample is a handful of independent periods,
    and every 20-year loss started in or around the 1929 crash. Price only:
    with dividends the long-run loss share would be lower, after inflation it
    would be higher. One index, one country, a century that went well.

RUN
    python HoldingPeriod_Reel_Pipeline.py --smoke   # 3 frames
    python HoldingPeriod_Reel_Pipeline.py           # full render -> topic.mp4

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
from matplotlib.colors import LinearSegmentedColormap, LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "_cache")

CONFIG = {
    "START": "1927-12-30",
    "END": "2026-09-30",
    "HORIZONS": [(1, "1 DAY"), (21, "1 MONTH"), (252, "1 YEAR"),
                 (1260, "5 YEARS"), (2520, "10 YEARS"), (5040, "20 YEARS")],
    "GAP": 1.7,                      # depth between curtains
    "CLIP": 1.25,                    # curtain height cap, in own-scale units

    # render
    "W": 1080, "H": 1920, "DPI": 100,
    "FPS": 30,
    "HOOK_SEC": 1.4, "BUILD_SEC": 8.5, "HOLD_SEC": 1.8,
    "ELEV_BASE": 34, "AZIM_START": -106, "AZIM_SWEEP": 22,
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
# gains only: blue -> orange, red is reserved for losses
GAIN = LinearSegmentedColormap.from_list(
    "gain", [THEME["BLUE"], THEME["CYAN"], THEME["GREEN"],
             THEME["YELLOW"], THEME["ORANGE"]], N=256)

SAFE = {"TOP": 0.12, "BOTTOM": 0.20, "RIGHT": 0.14}

LAYOUT = {
    "TITLE": 0.862, "SUBTITLE": 0.836, "KEY": 0.814, "HANDLE": 0.792,
    "AXES": [-0.08, 0.30, 1.16, 0.485],
    "NOTE": 0.272, "BIG": 0.242, "SUB": 0.215,
}

LS = LightSource(azdeg=225, altdeg=40)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------- DATA
def fetch():
    """yfinance once, then a CSV cache. Delete _cache/ to refetch."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "gspc.csv")
    if not os.path.exists(path):
        import yfinance as yf
        log("[Data] fetching ^GSPC")
        s = yf.download("^GSPC", start=CONFIG["START"], progress=False,
                        auto_adjust=True)["Close"].squeeze()
        if s.empty:
            raise SystemExit("no data returned; refusing to draw a synthetic tape")
        s.to_frame("GSPC").to_csv(path)
    s = pd.read_csv(path, index_col=0, parse_dates=True)["GSPC"]
    return s.loc[CONFIG["START"]:CONFIG["END"]].dropna()


# ------------------------------------------------------------- COMPUTE
def compute(px):
    curt = []
    pct = np.linspace(0, 100, 201)
    for k, (h, name) in enumerate(CONFIG["HORIZONS"]):
        r = (px.shift(-h) / px - 1).dropna()
        q = np.percentile(r.values, pct)
        scale = np.percentile(np.abs(r.values), 95)
        z = np.clip(q / scale, -CONFIG["CLIP"], CONFIG["CLIP"])
        loss = float((r < 0).mean())
        curt.append({"h": h, "name": name, "r": r, "q": q, "z": z,
                     "loss": loss, "cross": 10.0 * loss})
    n = len(curt)
    for k, c in enumerate(curt):
        c["y"] = (n - 1 - k) * CONFIG["GAP"]          # 1 day at the back
    return {"px": px, "curt": curt, "pct": pct}


# ------------------------------------------------------------ VALIDATE
def _pct(p):
    return f"{p * 100:.1f}%" if p < 0.1 else f"{p * 100:.0f}%"


def validate(d):
    """Every number that reaches a frame is asserted here, before any render."""
    px, curt = d["px"], d["curt"]
    loss = [c["loss"] for c in curt]
    assert len(px) > 20000, f"only {len(px)} closes, ^GSPC history looks cut"
    assert px.index[0].year <= 1928, "history must reach back to the 1920s"
    assert 0.35 < loss[0] < 0.6, f"1-day loss share {loss[0]:.3f} implausible"
    assert all(0 <= p <= 1 for p in loss)
    assert all(a > b for a, b in zip(loss, loss[1:])), \
        f"loss share does not fall with holding period: {loss}; claim gone"
    assert loss[-1] < 0.15, "20-year loss share no longer small; claim gone"
    for c in curt:
        assert np.isfinite(c["z"]).all()

    r20 = curt[-1]["r"]
    lost20 = r20[r20 < 0]
    assert len(lost20) > 0, "no 20-year loss left; rewrite the claim"
    keys = ["1d", "1m", "1y", "5y", "10y", "20y"]
    figs = {f"loss_{k}": _pct(c["loss"]) for k, c in zip(keys, curt)}
    figs.update({
        "n_days": f"{len(curt[0]['r']):,}",
        "n_20y": f"{len(r20):,}",
        "worst_20y": f"{r20.min() * 100:.0f}%",
        "worst_20y_start": r20.idxmin().strftime("%b %Y"),
        "first_20y_loss": lost20.index[0].strftime("%b %Y"),
        "last_20y_loss": lost20.index[-1].strftime("%b %Y"),
        "start": px.index[0].strftime("%Y"),
        "end": px.index[-1].strftime("%b %Y"),
        "years": "20",
        "horizons": ", ".join(name.lower() for _, name in CONFIG["HORIZONS"]),
    })
    assert lost20.index[-1].year < 1960, "a 20-year loss after 1960; caption says 1929 era"
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


def _reveal(frac, k, n):
    """Curtain k sweeps open left to right in its own slot of the build."""
    slot = 1.0 / n
    return float(np.clip((frac - k * slot) / (slot * 0.85), 0, 1))


def scene(ax, d, frac, lw=0.9):
    curt, pct = d["curt"], d["pct"]
    n = len(curt)
    x = pct / 10.0
    ymax = (n - 1) * CONFIG["GAP"]
    ax.add_collection3d(Poly3DCollection(
        [[(0, -0.4, 0), (10, -0.4, 0), (10, ymax + 0.4, 0), (0, ymax + 0.4, 0)]],
        facecolors=(1, 1, 1, 0.06), edgecolor=(1, 1, 1, 0.18), linewidth=0.6),
        autolim=False)
    for k, c in enumerate(curt):
        y, z = c["y"], c["z"]
        ax.plot([0, 10], [y, y], [0, 0], color=(1, 1, 1, 0.25), lw=0.6)
        g = _reveal(frac, k, n)
        if g <= 0:
            continue
        m = int(np.ceil(g * (len(x) - 1)))
        quads, cols = [], []
        gain_col = GAIN(k / (n - 1))
        for i in range(m):
            z0, z1 = z[i], z[i + 1]
            quads.append([(x[i], y, 0), (x[i + 1], y, 0),
                          (x[i + 1], y, z1), (x[i], y, z0)])
            cols.append(THEME["RED"] if (z0 + z1) < 0 else gain_col)
        ax.add_collection3d(Poly3DCollection(
            quads, facecolors=cols, edgecolor="none", alpha=0.88), autolim=False)
        ax.plot(x[:m + 1], [y] * (m + 1), z[:m + 1], color=THEME["TEXT"], lw=lw)
        if g >= 1:
            ax.plot([c["cross"]] * 2, [y, y], [-0.35, 0.55], color=THEME["TEXT"], lw=2.0)
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, (n - 1) * CONFIG["GAP"] + 0.5)
    ax.set_zlim(-CONFIG["CLIP"], CONFIG["CLIP"])


def scene_labels(ax, d, frac, fs=12):
    """Projected 2D text drawn above every 3D collection."""
    M = ax.get_proj()
    figs, curt = d["figs"], d["curt"]
    keys = ["1d", "1m", "1y", "5y", "10y", "20y"]

    def put(x, y, z, s, **kw):
        x2, y2, _ = proj3d.proj_transform(x, y, z, M)
        ax.text2D(x2, y2, s, transform=ax.transData, zorder=1e6,
                  va="center", family=THEME["MONO"], **kw)

    for k, (c, key) in enumerate(zip(curt, keys)):
        if _reveal(frac, k, len(curt)) <= 0:
            continue
        put(-0.3, c["y"], 0, c["name"], ha="right", color=THEME["TEXT_DIM"],
            fontsize=fs * 0.85)
        if _reveal(frac, k, len(curt)) >= 1:
            put(c["cross"], c["y"], 0.62, figs[f"loss_{key}"], ha="center",
                color=THEME["TEXT"], fontsize=fs, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", fc=THEME["RED"], ec="none",
                          alpha=0.9))


def text_block(fig, figs, s=1.0):
    L = LAYOUT
    fig.text(0.5, L["TITLE"], "TIME KILLS RISK", ha="center", fontsize=29 * s,
             fontweight="bold", color=THEME["TEXT"], family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             f"S&P 500 since {figs['start']}  |  {figs['n_days']} start days  |  "
             f"how often you were down",
             ha="center", fontsize=12.5 * s, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.5, L["KEY"],
             "one curtain per holding period  ·  worst to best  ·  red = lost money",
             ha="center", fontsize=10 * s, color=THEME["TEXT_DIM"], family=THEME["FONT"])
    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10 * s,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])
    fig.text(0.5, L["BIG"],
             f"1 day: {figs['loss_1d']}  →  20 years: {figs['loss_20y']}",
             ha="center", fontsize=22 * s, color=THEME["RED"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             "chance you ended below what you paid  ·  price only",
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
    n = len(d["curt"])
    frac = 1.0 / n + (1.0 - 1.0 / n) * frac      # the hook already shows 1 day
    t = idx / max(total - 1, 1)

    fig = plt.figure(figsize=(c["W"] / c["DPI"], c["H"] / c["DPI"]),
                     dpi=c["DPI"], facecolor=THEME["BG"])
    text_block(fig, figs)
    ax = fig.add_axes(LAYOUT["AXES"], projection="3d", facecolor=THEME["BG"])
    scene(ax, d, frac)
    ax.view_init(elev=c["ELEV_BASE"] + 5 * np.sin(t * np.pi),
                 azim=c["AZIM_START"] + c["AZIM_SWEEP"] * t)
    ax.set_box_aspect((1.5, 1.25, 1.15), zoom=c["ZOOM"])
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
