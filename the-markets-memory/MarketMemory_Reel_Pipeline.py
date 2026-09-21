"""
MarketMemory_Reel_Pipeline.py
=============================
Reel 9 for @quant.traderr. Thirty three years of SPY reduced to five states and
the chords between them, spun fast on purpose.

THE MATHS
    Every session is put in one of five buckets by its own return: below
    -BIG, -BIG to -SMALL, the flat middle, SMALL to BIG, and above BIG, with
    BIG and SMALL fixed in CONFIG rather than fitted. Counting every pair of
    consecutive sessions gives the transition matrix P, where P[i][j] is the
    share of days in state i that were followed by a day in state j. Each row
    sums to 100 percent by construction.

    The ring is those five states. A chord from i to j carries P[i][j] in both
    its width and its colour, and it bows above the ring going one way around
    and below going the other, so the direction of a transition is visible
    rather than implied. The loop outside a node is that state repeating.

    The headline compares three numbers from the same matrix: the chance of a
    big move after a big down day, after a flat day, and on an average day.

THE CLAIM THE VISUAL MAKES
    The market has a short memory and it is about size, not direction. A big
    day makes another big day much more likely, and it says almost nothing
    about which way that day goes.

WHAT THE VISUAL IS NOT
    Not a forecast and not a strategy. These are unconditional counts over the
    whole sample, so they mix 2008 and 2017 into one number, and the clustering
    is concentrated in crises rather than spread evenly.

    **The bucket edges are a choice.** Moving them moves every figure here. A
    one day lag is also the shortest possible memory: nothing in this picture
    says anything about what happens over a week. And a transition matrix
    assumes the next day depends only on today, which is a convenient fiction,
    not a property of markets.

RUN
    python MarketMemory_Reel_Pipeline.py --smoke   # 3 frames -> temp_frames_smoke/
    python MarketMemory_Reel_Pipeline.py           # full render -> topic.mp4

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
from mpl_toolkits.mplot3d.art3d import Line3DCollection

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "_cache")

CONFIG = {
    # >>> REPLACE: data + maths knobs for your topic
    "TICKERS": ["SPY"],
    "PERIOD": "max",
    "BIG": 1.5,               # percent, the edge of a big day
    "SMALL": 0.5,             # percent, the edge of the flat middle
    "R": 1.0,                 # ring radius
    "BOW": 2.30,              # how far a chord bows out of the ring plane
    "LOOP_R": 0.12,           # radius of a self transition loop
    "GAMMA": 0.75,            # colour curve on the transition probability

    # render
    "W": 1080, "H": 1920, "DPI": 100,
    "FPS": 30,
    "HOOK_SEC": 1.4, "BUILD_SEC": 8.5, "HOLD_SEC": 1.8,
    # Two full turns rather than the house quarter sweep. Jannis asked for a
    # fast reel, and a ring is the one form that survives it: it is symmetric
    # about the spin axis, so a fast rotation reads as motion rather than as
    # the picture falling over. The labels ride the nodes and move about seven
    # pixels per frame at this rate, which is still comfortably readable.
    "ELEV_BASE": 26, "AZIM_START": -60, "AZIM_SWEEP": 720,
    "ZOOM": 1.34,
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

STATES = ["BIG DOWN", "DOWN", "FLAT", "UP", "BIG UP"]


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
    c = CONFIG
    close = px[c["TICKERS"][0]]
    rets = (close / close.shift(1) - 1).dropna() * 100.0

    edges = [-np.inf, -c["BIG"], -c["SMALL"], c["SMALL"], c["BIG"], np.inf]
    state = pd.cut(rets, edges, labels=STATES)

    share = (state.value_counts(normalize=True).reindex(STATES) * 100.0).values
    P = (pd.crosstab(state[:-1], state.shift(-1)[:-1], normalize="index")
         .reindex(index=STATES, columns=STATES).fillna(0.0).values * 100.0)

    return {"P": P, "share": share, "n": len(rets),
            "years": (close.index[-1] - close.index[0]).days / 365.25}


# ------------------------------------------------------------ VALIDATE
def validate(d):
    """Every number that reaches a frame is asserted here, before any render.
    A data refresh that moves a figure out of band fails the build instead of
    shipping a caption that no longer matches the picture."""
    c, P, share = CONFIG, d["P"], d["share"]
    big = [STATES.index("BIG DOWN"), STATES.index("BIG UP")]

    uncond = float(share[big].sum())
    after_bd = float(P[STATES.index("BIG DOWN")][big].sum())
    after_flat = float(P[STATES.index("FLAT")][big].sum())
    factor = after_bd / uncond
    r, col = np.unravel_index(int(np.argmax(P)), P.shape)

    assert d["n"] > 5000, f"only {d['n']} sessions, too few for a matrix"
    assert abs(share.sum() - 100.0) < 0.5, "state shares do not sum to 100"
    assert all(abs(row.sum() - 100.0) < 0.5 for row in P), "a matrix row is not 100"
    assert 30.0 <= share[STATES.index("FLAT")] <= 65.0, "the flat bucket is mis-sized"
    assert 5.0 <= uncond <= 25.0, f"big days are {uncond:.0f}% of the sample"
    assert 15.0 <= after_bd <= 55.0, f"after a big down day {after_bd:.0f}%, out of band"
    assert factor > 1.3, "no clustering in this sample; check the buckets"
    assert after_flat < uncond, "a flat day is not the quiet state; check the buckets"

    figs = {
        "after_bigdown": f"{after_bd:.0f}%",
        "uncond_big": f"{uncond:.0f}%",
        "after_flat": f"{after_flat:.0f}%",
        "factor": f"{factor:.1f}",
        "big_edge": f"{c['BIG']}%",
        "flat_edge": f"{c['SMALL']}%",
        "top_from": STATES[r], "top_to": STATES[col],
        "top_p": f"{P[r][col]:.0f}%",
        "n_days": f"{d['n']:,}",
        "n_years": f"{d['years']:.0f}",
        "n_states": str(len(STATES)),
    }
    for name, v in zip(STATES, share):
        figs["share_" + name.lower().replace(" ", "")] = f"{v:.0f}%"
    with open(os.path.join(BASE_DIR, "figures.json"), "w") as fh:
        json.dump(figs, fh, indent=1)
    log(f"[validate] {figs}")
    d["figs"] = figs
    return figs


def _bezier(p0, c0, p1, n=48):
    t = np.linspace(0, 1, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * t * (1 - t) * c0 + t ** 2 * p1


def prepare_geometry(d):
    """Ring, chords and loops once, so a frame stays a pure function of its
    index."""
    c, P = CONFIG, d["P"]
    k = len(STATES)
    # One colour per state, and the colour says two things at once: warm is a
    # down day and cool is an up day, brighter is a bigger day. A chord then
    # takes the colour of the day it leads to.
    d["state_cols"] = [THEME["RED"], THEME["ORANGE"], THEME["BLUE"],
                       THEME["GREEN"], THEME["YELLOW"]]
    ang = np.linspace(0, 2 * np.pi, k, endpoint=False) + np.pi / 2
    nodes = np.stack([c["R"] * np.cos(ang), c["R"] * np.sin(ang),
                      np.zeros(k)], axis=1)
    d["nodes"], d["ang"] = nodes, ang

    hi = float(P.max())
    segs, cols, widths, order = [], [], [], []
    for i in range(k):
        for j in range(k):
            p = float(P[i][j])
            if p < 0.5:                     # below half a percent it is a hairline
                continue
            shade = d["state_cols"][j]
            w = 0.6 + 6.5 * (p / hi) ** c["GAMMA"]
            if i == j:
                # A self transition is a loop sitting outside its own node, in
                # the plane spanned by the radial direction and the spin axis,
                # so it stays visible from every angle of the turn.
                a = ang[i]
                radial = np.array([np.cos(a), np.sin(a), 0.0])
                centre = nodes[i] + radial * c["LOOP_R"] * 1.25
                th = np.linspace(0, 2 * np.pi, 40)
                pts = (centre[None, :]
                       + np.outer(np.cos(th), radial * c["LOOP_R"])
                       + np.outer(np.sin(th), np.array([0, 0, c["LOOP_R"]])))
            else:
                # Above the ring going one way round, below going the other.
                # Without that a pair of opposite transitions draws as one line
                # and the direction of the market's memory disappears.
                fwd = ((j - i) % k) <= k // 2
                ctrl = np.array([0.0, 0.0, c["BOW"] * (1 if fwd else -1)])
                pts = _bezier(nodes[i], ctrl, nodes[j])
            segs.append(np.stack([pts[:-1], pts[1:]], axis=1))
            cols.append(shade)
            widths.append(w)
            order.append(p)

    d["chords"] = segs
    d["chord_cols"] = cols
    d["chord_w"] = widths
    d["chord_order"] = np.argsort(-np.array(order))   # strongest drawn first
    d["node_cols"] = d["state_cols"]
    d["lim"] = c["R"] * 1.30
    return d


# -------------------------------------------------------------- RENDER
def render_frame(idx, total, d, out_path):
    c, figs, L = CONFIG, d["figs"], LAYOUT
    if "chords" not in d:
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
    fig.text(0.5, L["TITLE"], "THE MARKET'S MEMORY", ha="center", fontsize=28,
             fontweight="bold", color=THEME["TEXT"], family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             f"SPY  |  {figs['n_years']} years  |  {figs['n_days']} sessions",
             ha="center", fontsize=13, color=THEME["ORANGE"], family=THEME["FONT"])
    for i, name in enumerate(STATES):
        share = figs["share_" + name.lower().replace(" ", "")]
        fig.text(0.17 + i * 0.165, L["KEY"], f"{name}\n{share}", ha="center",
                 va="center", fontsize=8, color=d["state_cols"][i],
                 family=THEME["MONO"], linespacing=1.6)
    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])

    ax = fig.add_axes(L["AXES"], projection="3d", facecolor=THEME["BG"],
                      computed_zorder=False)

    # >>> REPLACE: the geometry. Chords arrive strongest first, ring stands.
    nodes = d["nodes"]
    ring_th = np.linspace(0, 2 * np.pi, 160)
    ax.plot(c["R"] * np.cos(ring_th), c["R"] * np.sin(ring_th),
            np.zeros_like(ring_th), color=THEME["TEXT"], alpha=0.18,
            linewidth=0.8, zorder=2)

    n_ch = len(d["chords"])
    show = max(int(round(n_ch * (0.20 + 0.80 * frac))), 1)
    for rank in d["chord_order"][:show]:
        ax.add_collection3d(Line3DCollection(
            d["chords"][rank], colors=[d["chord_cols"][rank]],
            linewidths=d["chord_w"][rank], alpha=0.8, zorder=3))

    ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], s=90,
               c=d["node_cols"], edgecolors=THEME["TEXT"], linewidths=0.6,
               zorder=6)

    lim = d["lim"]
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim * 1.05, lim * 1.05)
    ax.view_init(elev=c["ELEV_BASE"] + 8 * np.sin(t * 2 * np.pi),
                 azim=c["AZIM_START"] + c["AZIM_SWEEP"] * t)
    ax.set_box_aspect((1.0, 1.0, 1.05), zoom=c["ZOOM"])
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    # Readout: strings only, straight from figures.json. Never re-derive here.
    fig.text(0.5, L["NOTE"],
             f"big means the day moved more than {figs['big_edge']}"
             f"  ·  warm is a down day, cool is an up day",
             ha="center", fontsize=10, color=THEME["TEXT_DIM"],
             family=THEME["FONT"])
    fig.text(0.5, L["BIG"],
             f"{figs['after_bigdown']} AFTER A BIG DOWN DAY", ha="center",
             fontsize=19, color=THEME["RED"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             f"{figs['uncond_big']} on an average day  ·  "
             f"{figs['after_flat']} after a quiet one",
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
