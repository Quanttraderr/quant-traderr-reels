"""
PhasePortrait_Reel_Pipeline.py
==============================
Reel 2 for @quant.traderr. Five years of SPY drawn as a single path through a
three dimensional state space, with a head that races along it.

THE MATHS
    Three features describe SPY on any session: WINDOW day momentum
    (px / px.shift(W) - 1), WINDOW day realised volatility (std of log returns
    times sqrt(252)), and drawdown from the running high (px / px.cummax() - 1).
    Each feature is standardised to zero mean and unit variance, so one step in
    any direction means the same thing, and the path is the sequence of those
    standardised triples. State speed is the length of the step from one session
    to the next in that standardised space.

THE CLAIM THE VISUAL MAKES
    The market does not move through its own states at a constant rate. It
    crawls for months and then covers more ground in a day than it usually does
    in a fortnight, and the path spends most of its life below the old high.

WHAT THE VISUAL IS NOT
    Not a forecast: nothing here says where the path goes next. Not the market
    either, because three features are a choice out of many, and a different
    three would draw a different shape. A loop in a phase portrait is not a
    cycle that can be traded. The windows overlap, so consecutive states are not
    independent, and the standardisation is over this sample only.

RUN
    python PhasePortrait_Reel_Pipeline.py --smoke   # 3 frames -> temp_frames_smoke/
    python PhasePortrait_Reel_Pipeline.py           # full render -> topic.mp4

OUTPUT
    1440x2560 (2K vertical) @ 30 FPS, 1.4s hook + 8.5s build + 1.8s hold = 11.7s

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
from mpl_toolkits.mplot3d.art3d import Line3DCollection

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "_cache")

CONFIG = {
    # >>> REPLACE: data + maths knobs for your topic
    "TICKERS": ["SPY"],
    "PERIOD": "5y",
    "WINDOW": 20,
    "HEAD_TAIL": 45,          # sessions kept bright behind the head

    # render (leave these alone unless the topic needs landscape)
    "W": 1440, "H": 2560, "DPI": 400 / 3,   # 2K master. The figure stays
    # 10.8x19.2in, so every fontsize and linewidth lands where it would at
    # 1080x1920; only the sampling density goes up.
    "FPS": 30,
    "HOOK_SEC": 1.4, "BUILD_SEC": 8.5, "HOLD_SEC": 1.8,
    "ELEV_BASE": 27, "AZIM_START": -62, "AZIM_SWEEP": 58,
    # The playbook says do not speed the orbit up. This one runs at 58 degrees
    # instead of the usual 30 because Jannis asked for more motion, and it is
    # still a slow sweep next to the head, which covers about 145 sessions per
    # second. The subject carries the speed; the camera only supports it.
    "ZOOM": 1.44,
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
    log(f"[Data] fetching {CONFIG['TICKERS']}")
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
    s = px[CONFIG["TICKERS"][0]].dropna()
    w = CONFIG["WINDOW"]

    rets = np.log(s / s.shift(1))
    mom = (s / s.shift(w) - 1.0) * 100.0
    vol = rets.rolling(w).std() * np.sqrt(252) * 100.0
    dd = (s / s.cummax() - 1.0) * 100.0
    feat = pd.concat({"mom": mom, "vol": vol, "dd": dd}, axis=1).dropna()

    # Standardise every feature. Momentum in percent, volatility in annualised
    # percent and drawdown in percent are three different rulers; without this
    # the "distance" the speed measures would be dominated by whichever feature
    # happens to have the largest raw spread.
    mu, sd = feat.mean(), feat.std()
    z = ((feat - mu) / sd).values

    # State speed: how far the market travelled between two sessions.
    step = np.linalg.norm(np.diff(z, axis=0), axis=1)
    speed = np.concatenate([[step[0]], step])       # align to the state index
    typical = float(np.median(step))
    fastest_i = int(np.argmax(step)) + 1

    # Where drawdown equals zero, standardised: the plane of the old high.
    high_plane = float((0.0 - mu["dd"]) / sd["dd"])

    return {"z": z, "speed": speed, "index": feat.index,
            "feat": feat, "typical": typical, "fastest_i": fastest_i,
            "speed_ratio": float(step.max() / typical),
            "pct_underwater": float((feat["dd"].values < -1e-4).mean()),
            "deepest_dd": float(feat["dd"].min()),
            "peak_vol": float(feat["vol"].max()),
            "calm_vol": float(feat["vol"].min()),
            "high_plane": high_plane, "n_states": len(feat)}


# ------------------------------------------------------------ VALIDATE
def validate(d):
    """Every number that reaches a frame is asserted here, before any render.

    The bands are what these quantities can be in any regime, not what they are
    today. A market is below its previous high most of the time by construction,
    but never all of it and never almost none of it, so 0.30 to 0.995 is a real
    gate and not a snapshot."""
    pct = d["pct_underwater"]
    ratio = d["speed_ratio"]
    deepest = abs(d["deepest_dd"])

    assert 0.30 <= pct <= 0.995, f"underwater share {pct:.2f} outside the 0.30-0.995 band"
    assert 3.0 <= ratio <= 60.0, f"speed ratio {ratio:.1f}x outside the 3-60 band"
    assert 3.0 <= deepest <= 90.0, f"deepest drawdown {deepest:.1f}% outside the 3-90 band"
    assert 8.0 <= d["peak_vol"] <= 150.0, f"peak vol {d['peak_vol']:.1f}% outside the 8-150 band"
    assert 2.0 <= d["calm_vol"] <= 30.0, f"calm vol {d['calm_vol']:.1f}% outside the 2-30 band"
    assert d["peak_vol"] > d["calm_vol"], "peak vol is not above the calm reading"
    assert d["n_states"] > 500, f"only {d['n_states']} states, too few to draw a path"

    figs = {
        "pct_underwater": f"{pct * 100:.0f}%",
        "speed_ratio": f"{ratio:.0f}x",
        "fastest_date": d["index"][d["fastest_i"]].strftime("%b %Y"),
        "deepest_dd": f"{deepest:.1f}%",
        "peak_vol": f"{d['peak_vol']:.0f}%",
        "calm_vol": f"{d['calm_vol']:.0f}%",
        "n_sessions": f"{d['n_states']:,}",
        "window_days": str(CONFIG["WINDOW"]),
    }
    with open(os.path.join(BASE_DIR, "figures.json"), "w") as fh:
        json.dump(figs, fh, indent=1)
    log(f"[validate] {figs}")
    d["figs"] = figs
    return figs


def prepare_geometry(d):
    """Limits, aspect and the colour normalisation, computed once so every frame
    shares them. Per-frame limits would make the path breathe as it grows."""
    z = d["z"]
    lo, hi = z.min(axis=0), z.max(axis=0)
    pad = 0.06 * float((hi - lo).max())
    d["lim"] = [(float(lo[j]) - pad, float(hi[j]) + pad) for j in range(3)]
    span = np.array([b - a for a, b in d["lim"]])
    d["box_aspect"] = tuple(span / span.max())
    # Speed is long tailed: one crash day would flatten every other colour, so
    # the ramp is normalised to the 95th percentile and the rest is clipped.
    d["speed_hi"] = float(np.percentile(d["speed"], 95))

    # The grid stops just inside the path, so the extremes of the frame are
    # always data, never scenery.
    gx = np.linspace(z[:, 0].min() * 0.60, z[:, 0].max() * 0.60, 4)
    gy = np.linspace(z[:, 1].min() * 0.60, z[:, 1].max() * 0.60, 4)
    XX, YY = np.meshgrid(gx, gy)
    d["plane"] = (XX, YY, np.full_like(XX, d["high_plane"], dtype=float))
    return d


# -------------------------------------------------------------- RENDER
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
    L = LAYOUT
    fig.text(0.5, L["TITLE"], "THE MARKET'S PHASE PORTRAIT", ha="center",
             fontsize=25, fontweight="bold", color=THEME["TEXT"],
             family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             f"SPY  |  {figs['n_sessions']} sessions  |  "
             f"{figs['window_days']}-day momentum, volatility, drawdown",
             ha="center", fontsize=12.5, color=THEME["ORANGE"],
             family=THEME["FONT"])
    fig.text(0.5, L["KEY"], "colour = how fast the state is moving",
             ha="center", fontsize=10, color=THEME["TEXT_DIM"],
             family=THEME["FONT"])
    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])

    ax = fig.add_axes(L["AXES"], projection="3d", facecolor=THEME["BG"])

    # >>> REPLACE: the geometry. The head races along the path, the trail stays.
    z, speed = d["z"], d["speed"]
    k = max(int(round(frac * (len(z) - 1))), 1)

    seg = np.stack([z[:k], z[1:k + 1]], axis=1)          # (k, 2, 3)
    w = np.clip(speed[1:k + 1] / max(d["speed_hi"], 1e-9), 0, 1)
    cols = CMAP(w)
    cols[:, 3] = 0.34                                    # the settled trail
    if k > c["HEAD_TAIL"]:                               # the recent stretch
        cols[-c["HEAD_TAIL"]:, 3] = np.linspace(0.35, 1.0, c["HEAD_TAIL"])
    else:
        cols[:, 3] = np.linspace(0.35, 1.0, k)
    lw = np.full(k, 1.0)
    lw[-min(k, c["HEAD_TAIL"]):] = np.linspace(1.0, 2.8, min(k, c["HEAD_TAIL"]))
    ax.add_collection3d(Line3DCollection(seg, colors=cols, linewidths=lw,
                                         capstyle="round", zorder=3))

    # The plane of the old high, drawn as a grid rather than a filled slab: at
    # any alpha a filled surface reads as a grey rectangle against pure black,
    # and it ran past the canvas edge. The grid spans the path, not the axis
    # limits, so it cannot be the thing that gets clipped.
    XX, YY, ZZ = d["plane"]
    ax.plot_wireframe(XX, YY, ZZ, color=THEME["TEXT"], alpha=0.20,
                      linewidth=0.5, zorder=1)

    head = z[k]
    ax.scatter(*head, s=560, color=THEME["TEXT"], alpha=0.12, linewidths=0,
               zorder=4)
    ax.scatter(*head, s=90, color=CMAP(np.clip(speed[k] / max(d["speed_hi"], 1e-9), 0, 1)),
               alpha=1.0, linewidths=0, zorder=5)

    ax.set_xlim(*d["lim"][0]); ax.set_ylim(*d["lim"][1]); ax.set_zlim(*d["lim"][2])
    ax.view_init(elev=c["ELEV_BASE"] + 8 * np.sin(t * np.pi),
                 azim=c["AZIM_START"] + c["AZIM_SWEEP"] * t)
    ax.set_box_aspect(d["box_aspect"], zoom=c["ZOOM"])
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    # The axis key sits in figure coordinates, not on the axes. A 3D axes clips
    # anything placed outside its limits without a word of warning.
    fig.text(0.5, L["NOTE"], "the grid is the previous high", ha="center",
             fontsize=10, color=THEME["TEXT_DIM"], family=THEME["MONO"],
             alpha=0.8)

    # Readout: strings only, straight from figures.json. Never re-derive here.
    fig.text(0.5, L["BIG"], f"{figs['pct_underwater']} OF DAYS UNDERWATER",
             ha="center", fontsize=19, color=THEME["RED"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             f"fastest {figs['fastest_date']}, {figs['speed_ratio']} a typical "
             f"day  ·  worst {figs['deepest_dd']}",
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
                        "-c:v", "libx264", "-preset", "slow", "-crf", "18",
                        "-pix_fmt", "yuv420p", mp4],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(r.stderr[-2000:])
    shutil.rmtree(frames, ignore_errors=True)   # the cache is 6 GB if you don't
    log(f"OK {mp4}  ({total} frames, {total / c['FPS']:.1f}s, {c['W']}x{c['H']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    main(**vars(ap.parse_args()))
