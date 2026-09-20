"""
BellCurve_Reel_Pipeline.py
==========================
Reel 8 for @quant.traderr. Twenty two years of three assets as one cloud of
points, with the shell a normal distribution says they should live inside.

THE MATHS
    One point per session: the day's log return in SPY, in TLT and in GLD, each
    standardised over the whole sample, so one unit means one standard
    deviation of that asset and the three axes are comparable.

    Fit the multivariate normal that a plain risk model assumes: the covariance
    matrix of those standardised returns. Its Mahalanobis distance
    d = sqrt(x' inv(S) x) is how many standard deviations a day sits from the
    middle once the correlations between the three are accounted for. The shell
    drawn in the picture is the surface d = SHELL_K, which under that normal
    distribution is the boundary essentially nothing should ever cross: the
    expected number of days outside it over this sample is a fraction of one.

    Colour is d, so a point's colour and its distance from the middle say the
    same thing.

THE CLAIM THE VISUAL MAKES
    The bell curve is not slightly wrong in the tails, it is wrong by orders of
    magnitude. The shell is where a normal distribution stops, and the market
    put dozens of days far outside it.

WHAT THE VISUAL IS NOT
    Not a claim that the market has no distribution, only that this one does
    not fit it. A normal distribution with one fixed covariance matrix is a
    straw man on purpose, because it is the assumption sitting inside ordinary
    risk numbers: volatility clusters, so a model that lets the covariance move
    explains much of this excess without any magic.

    The covariance is estimated in sample, on the same days it is then tested
    against, which flatters the fit rather than the tails. Three assets are not
    a market. The odds quoted for the worst day are what the fitted normal
    distribution implies, not a physical probability of anything, and they are
    quoted to show how badly the model breaks, not to predict.

RUN
    python BellCurve_Reel_Pipeline.py --smoke   # 3 frames -> temp_frames_smoke/
    python BellCurve_Reel_Pipeline.py           # full render -> topic.mp4

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
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "_cache")

CONFIG = {
    # >>> REPLACE: data + maths knobs for your topic
    "TICKERS": ["SPY", "TLT", "GLD"],
    "PERIOD": "max",
    "SHELL_K": 5,             # the shell drawn, in Mahalanobis sigma
    "COL_HI": 5.0,            # colour ceiling: saturates exactly at the shell
    "GAMMA": 0.80,
    "N_U": 60, "N_V": 30,     # mesh of the shell

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
    "ELEV_BASE": 14, "AZIM_START": -60, "AZIM_SWEEP": 34,
    "ZOOM": 1.95,
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

# Instagram lays its own controls over a reel: the Reels header on top, the
# name, caption and audio line at the bottom, the like and comment buttons on
# the right. Nothing readable may sit in those bands. The form itself may bleed
# into them, text never.
SAFE = {"TOP": 0.12, "BOTTOM": 0.20, "RIGHT": 0.14}

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
        return pd.read_csv(path, index_col=0, parse_dates=True)
    import yfinance as yf
    log(f"[Data] fetching {CONFIG['TICKERS']} {CONFIG['PERIOD']}")
    px = yf.download(CONFIG["TICKERS"], period=CONFIG["PERIOD"],
                     interval="1d", progress=False, auto_adjust=True)["Close"]
    px = px[CONFIG["TICKERS"]].dropna()
    if px.empty:
        raise SystemExit("no data returned; refusing to draw a synthetic tape")
    px.to_csv(path)
    return px


# ------------------------------------------------------------- COMPUTE
def compute(px):
    # >>> REPLACE: the maths for your topic. Return whatever render_frame needs.
    rets = np.log(px / px.shift(1)).dropna()
    z = ((rets - rets.mean()) / rets.std()).values
    cov = np.cov(z.T)
    inv = np.linalg.inv(cov)
    dist = np.sqrt(np.einsum("ij,jk,ik->i", z, inv, z))
    return {"z": z, "dist": dist, "cov": cov, "dates": rets.index,
            "tickers": list(px.columns),
            "years": (rets.index[-1] - rets.index[0]).days / 365.25}


# ------------------------------------------------------------ VALIDATE
def validate(d):
    """Every number that reaches a frame is asserted here, before any render.
    A data refresh that moves a figure out of band fails the build instead of
    shipping a caption that no longer matches the picture."""
    from scipy.stats import chi2
    c = CONFIG
    dist, n = d["dist"], len(d["dist"])
    k = c["SHELL_K"]

    outside = int((dist > k).sum())
    expected = float(n * chi2.sf(k * k, 3))       # 3 assets, 3 degrees of freedom
    worst = float(dist.max())
    worst_when = d["dates"][int(np.argmax(dist))]
    # logsf, not sf: at 11.8 sigma the survival probability underflows to zero
    # in double precision and the odds would silently come out as infinity.
    log10_years = float(-np.log10(252.0) - chi2.logsf(worst ** 2, 3) / np.log(10.0))

    assert n > 3000, f"only {n} sessions, too few for a tail claim"
    assert 6.0 <= worst <= 25.0, f"worst day {worst:.1f} sigma outside the 6-25 band"
    assert outside > expected * 5, "the tail is not fat in this sample; check the fit"
    assert 10 <= outside <= 400, f"{outside} days outside the shell, outside the band"
    assert 15.0 <= log10_years <= 80.0, f"odds exponent {log10_years:.0f} implausible"
    assert len(d["tickers"]) == 3, "the cloud needs exactly three axes"

    figs = {
        "shell_k": str(k),
        "n_outside": str(outside),
        "exp_outside": f"{expected:.1f}",
        "worst_sigma": f"{worst:.1f}",
        "worst_date": worst_when.strftime("%b %Y"),
        "odds_exp": f"{log10_years:.0f}",
        "odds_years": f"10^{log10_years:.0f}",
        "n_days": f"{n:,}",
        "n_years": f"{d['years']:.0f}",
        "assets": ", ".join(d["tickers"]),
    }
    with open(os.path.join(BASE_DIR, "figures.json"), "w") as fh:
        json.dump(figs, fh, indent=1)
    log(f"[validate] {figs}")
    d["figs"] = figs
    return figs


def prepare_geometry(d):
    """The shell, the colours and the limits once, so a frame stays a pure
    function of its index."""
    c = CONFIG
    z, dist = d["z"], d["dist"]

    # x = L u maps the unit sphere onto the surface where the Mahalanobis
    # distance is exactly SHELL_K, because x' inv(S) x = u'u for S = L L'.
    u = np.linspace(0, 2 * np.pi, c["N_U"])
    v = np.linspace(0, np.pi, c["N_V"])
    sphere = np.stack([np.outer(np.cos(u), np.sin(v)),
                       np.outer(np.sin(u), np.sin(v)),
                       np.outer(np.ones_like(u), np.cos(v))])
    L = np.linalg.cholesky(d["cov"])
    shell = c["SHELL_K"] * np.einsum("ij,jkl->ikl", L, sphere)
    d["shell"] = shell

    d["cols"] = CMAP(np.clip(dist / c["COL_HI"], 0, 1) ** c["GAMMA"])
    d["out"] = dist > c["SHELL_K"]
    lim = float(np.abs(z).max()) * 1.06
    d["lim"] = lim
    return d


# -------------------------------------------------------------- RENDER
def render_frame(idx, total, d, out_path):
    c, figs, L = CONFIG, d["figs"], LAYOUT
    if "shell" not in d:
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

    fig = plt.figure(figsize=(c["W"] / c["DPI"], c["H"] / c["DPI"]),
                     dpi=c["DPI"], facecolor=THEME["BG"])
    fig.text(0.5, L["TITLE"], "THE BELL CURVE LIE", ha="center", fontsize=28,
             fontweight="bold", color=THEME["TEXT"], family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             f"{figs['assets']}  |  {figs['n_years']} years  |  "
             f"{figs['n_days']} sessions",
             ha="center", fontsize=13, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.5, L["KEY"], "one dot is one day, colour is how far out it sits",
             ha="center", fontsize=10, color=THEME["TEXT_DIM"],
             family=THEME["FONT"])
    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])

    # computed_zorder=False: the cloud, the shell and the outliers have to be
    # layered deliberately. Left to matplotlib, the translucent shell is depth
    # sorted against 5000 individual points and the picture flickers between
    # frames as the sort order changes.
    ax = fig.add_axes(L["AXES"], projection="3d", facecolor=THEME["BG"],
                      computed_zorder=False)

    # >>> REPLACE: the geometry. Days arrive in order, the shell stands still.
    z, cols, out = d["z"], d["cols"], d["out"]
    n = len(z)
    n_show = max(int(round(n * (0.14 + 0.86 * frac))), 2)
    sel = np.arange(n_show)
    inside = sel[~out[:n_show]]
    beyond = sel[out[:n_show]]

    ax.scatter(z[inside, 0], z[inside, 1], z[inside, 2], s=5.5,
               c=cols[inside], alpha=0.70, linewidths=0, zorder=2)

    sh = d["shell"]
    ax.plot_wireframe(sh[0], sh[1], sh[2], color=THEME["CYAN"], alpha=0.30,
                      linewidth=0.5, rstride=3, cstride=2, zorder=3)

    # The days that broke the model go on top, and they are drawn last on
    # purpose: they are the whole point of the picture.
    ax.scatter(z[beyond, 0], z[beyond, 1], z[beyond, 2], s=44,
               c=cols[beyond], alpha=0.95, linewidths=0.5,
               edgecolors=THEME["TEXT"], zorder=4)

    lim = d["lim"]
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.view_init(elev=c["ELEV_BASE"] + 7 * np.sin(t * np.pi),
                 azim=c["AZIM_START"] + c["AZIM_SWEEP"] * t)
    ax.set_box_aspect((1.0, 1.0, 1.0), zoom=c["ZOOM"])
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    # Axis labels sit at a real data coordinate just inside the limit. Anything
    # placed outside set_xlim is clipped by the canvas without a warning, which
    # is the most common way a 3D frame ships with text cut off.
    for j, name in enumerate(d["tickers"]):
        pos = [0.0, 0.0, 0.0]
        pos[j] = lim * 0.78
        ax.text(*pos, name, color=THEME["TEXT_DIM"], fontsize=10,
                ha="center", va="center", family=THEME["MONO"], zorder=6)

    # Readout: strings only, straight from figures.json. Never re-derive here.
    fig.text(0.5, L["NOTE"],
             f"the shell is the {figs['shell_k']} sigma surface a bell curve draws",
             ha="center", fontsize=10, color=THEME["TEXT_DIM"],
             family=THEME["FONT"])
    fig.text(0.5, L["BIG"], f"{figs['n_outside']} DAYS OUTSIDE", ha="center",
             fontsize=19, color=THEME["RED"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             f"a bell curve expects {figs['exp_outside']}  ·  worst day "
             f"{figs['worst_sigma']} sigma",
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
