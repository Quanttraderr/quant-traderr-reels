"""
VectorField_Reel_Pipeline.py
============================
Reel 5 for @quant.traderr. Thirty three years of SPY turned into a drift field
over its own state space, with particles released into it.

THE MATHS
    Three features describe SPY on any session: WINDOW day momentum, WINDOW day
    realised volatility and drawdown from the running high. Each is standardised
    over the whole sample, so one step means the same thing in every direction.
    The cube of state space is cut into GRID bins per axis. For every bin that
    holds at least MIN_OBS sessions, the drift is the average change of the
    state vector over the following HORIZON sessions. That average is the arrow.

    The share reported on screen is the fraction of those bins whose arrow has a
    negative radial component, that is, whose average next move points back
    toward the middle of the distribution rather than further out.

THE CLAIM THE VISUAL MAKES
    The market's state does not wander off in whatever direction it is already
    going. Wherever you stand in this space, the average next twenty sessions
    pull you back toward the ordinary.

WHAT THE VISUAL IS NOT
    **The particles are not market paths.** They are tracers released into an
    average field and re-seeded when they reach the middle, which is how a flow
    is drawn. No particle is a simulation of anything that happened.

    The arrows are averages, and an average is not a rule: the spread around
    each arrow is wide, most of it is noise, and the drift is small next to the
    variation it sits in. Mean reversion in a standardised space is also partly
    mechanical, because a standardised variable cannot run away from its own
    mean forever. Not a forecast, not a strategy, one index, overlapping windows
    so neighbouring states are not independent.

RUN
    python VectorField_Reel_Pipeline.py --smoke   # 3 frames -> temp_frames_smoke/
    python VectorField_Reel_Pipeline.py           # full render -> topic.mp4

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
    "WINDOW": 20,
    "HORIZON": 20,
    "GRID": 7,
    "MIN_OBS": 30,

    # particles
    "N_PART": 650,
    "TRAIL": 14,
    "DT": 0.05,
    "IDW_K": 4,               # Zellen, über die das Feld geglättet wird
    "SEED": 20260918,
    "GAMMA": 0.85,            # Farbkurve: hebt die Mitte in die bunten Töne

    # render
    "W": 1080, "H": 1920, "DPI": 100,
    "FPS": 30,
    "HOOK_SEC": 1.4, "BUILD_SEC": 8.5, "HOLD_SEC": 1.8,
    "ELEV_BASE": 16, "AZIM_START": -64, "AZIM_SWEEP": 40,
    "ZOOM": 1.12,
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
    s = px[c["TICKERS"][0]].dropna()
    w, h, g = c["WINDOW"], c["HORIZON"], c["GRID"]

    rets = np.log(s / s.shift(1))
    feat = pd.concat({"mom": (s / s.shift(w) - 1.0) * 100.0,
                      "vol": rets.rolling(w).std() * np.sqrt(252) * 100.0,
                      "dd": (s / s.cummax() - 1.0) * 100.0}, axis=1).dropna()

    # Standardise, or the "distance" the drift measures would be dominated by
    # whichever feature happens to have the widest raw spread.
    z = ((feat - feat.mean()) / feat.std())
    future = (z.shift(-h) - z).dropna()
    zz = z.loc[future.index]

    # Bin edges from the 1st to the 99th percentile: the tails hold a handful of
    # sessions each and would produce arrows built on almost nothing.
    cols = ("mom", "vol", "dd")
    edges = [np.linspace(zz[cc].quantile(0.01), zz[cc].quantile(0.99), g + 1)
             for cc in cols]
    idx = [np.clip(np.digitize(zz[cc], e) - 1, 0, g - 1)
           for cc, e in zip(cols, edges)]
    cent = [(e[:-1] + e[1:]) / 2.0 for e in edges]

    pts, vecs, counts = [], [], []
    for i in range(g):
        for j in range(g):
            for k in range(g):
                m = (idx[0] == i) & (idx[1] == j) & (idx[2] == k)
                n = int(m.sum())
                if n < c["MIN_OBS"]:
                    continue
                pts.append([cent[0][i], cent[1][j], cent[2][k]])
                vecs.append(future[m].mean().values)
                counts.append(n)

    pts = np.array(pts)
    vecs = np.array(vecs)
    inward = int(sum(np.dot(v, -p) > 0 for p, v in zip(pts, vecs)))

    return {"pts": pts, "vecs": vecs, "counts": np.array(counts),
            "n_cells": len(pts), "inward": inward,
            "share_inward": inward / len(pts),
            "max_drift": float(np.linalg.norm(vecs, axis=1).max()),
            "n_states": len(future), "years": len(s) / 252.0}


# ------------------------------------------------------------ VALIDATE
def validate(d):
    """Every number that reaches a frame is asserted here, before any render.

    The band on share_inward is wide on purpose. A standardised variable cannot
    run away from its own mean forever, so some pull toward the middle is
    mechanical and a reading near a half would mean the binning broke rather
    than that the market changed."""
    c = CONFIG
    assert 0.50 <= d["share_inward"] <= 1.0, f"inward share {d['share_inward']:.2f} outside the 0.50-1.0 band"
    assert 12 <= d["n_cells"] <= 343, f"{d['n_cells']} filled cells is not a usable field"
    assert 0.05 <= d["max_drift"] <= 6.0, f"strongest drift {d['max_drift']:.2f} outside the 0.05-6 band"
    assert d["n_states"] > 2000, f"only {d['n_states']} states, too few to bin"
    assert int(d["counts"].min()) >= c["MIN_OBS"], "a cell slipped through under the minimum"

    figs = {
        "share_inward": f"{d['share_inward'] * 100:.0f}%",
        "n_cells": str(d["n_cells"]),
        "n_inward": str(d["inward"]),
        "n_states": f"{d['n_states']:,}",
        "horizon": str(c["HORIZON"]),
        "window_days": str(c["WINDOW"]),
        "n_years": f"{d['years']:.0f}",
        "min_obs": str(c["MIN_OBS"]),
    }
    with open(os.path.join(BASE_DIR, "figures.json"), "w") as fh:
        json.dump(figs, fh, indent=1)
    log(f"[validate] {figs}")
    d["figs"] = figs
    return figs


def prepare_geometry(d):
    """Limits, arrow colours, and the whole particle run, computed once.

    The particle positions for all 351 frames are worked out here rather than
    inside render_frame, so a frame stays a pure function of its index: render
    frame 200 twice and you get the same picture, which is what makes a re-cut
    or a partial re-render safe."""
    from scipy.spatial import cKDTree
    c = CONFIG
    pts, vecs = d["pts"], d["vecs"]

    lo, hi = pts.min(axis=0), pts.max(axis=0)
    pad = 0.12 * float((hi - lo).max())
    d["lim"] = [(float(lo[j]) - pad, float(hi[j]) + pad) for j in range(3)]
    span = np.array([b - a for a, b in d["lim"]])
    d["box_aspect"] = tuple(span / span.max())

    mag = np.linalg.norm(vecs, axis=1)
    d["arrow_cols"] = CMAP(np.clip(mag / np.percentile(mag, 95), 0, 1) ** c["GAMMA"])
    d["arrow_scale"] = 0.30 * float(span.max()) / max(mag.max(), 1e-9)
    order = np.argsort(-d["counts"])          # busiest cells appear first
    d["arrow_order"] = order

    tree = cKDTree(pts)
    step = float(np.mean([np.diff(sorted(set(pts[:, j]))).mean()
                          for j in range(3)]))
    rng = np.random.default_rng(c["SEED"])
    total = int(c["FPS"] * (c["HOOK_SEC"] + c["BUILD_SEC"] + c["HOLD_SEC"]))
    n = c["N_PART"]

    kk = min(c["IDW_K"], len(pts))

    def field(p):
        """Inverse distance weighting over the nearest cells.

        Snapping each tracer to its single nearest cell gave every tracer in a
        cell the identical arrow, so they marched in rows and the picture read
        as a few streaks instead of a flow. Weighting the nearest few cells
        smooths the field between the samples, which is how a sampled vector
        field is normally drawn."""
        dist, near = tree.query(p, k=kk)
        w = 1.0 / np.maximum(dist, 1e-6)
        w /= w.sum(axis=1, keepdims=True)
        return (vecs[near] * w[..., None]).sum(axis=1), dist[:, 0]

    def seed(k):
        """Release tracers anywhere in the field, not just at the rim: the point
        is that every part of the space drifts, not only the extremes."""
        base = pts[rng.integers(0, len(pts), size=k)]
        return base + rng.normal(0, step * 0.5, size=(k, 3))

    # Vorlauf über die Spurlänge hinaus: ohne ihn hat das Hook-Bild noch keine
    # Spuren und das Reel beginnt mit einem fast leeren Bild, also genau dort,
    # wo entschieden wird, ob jemand weiterscrollt.
    warm = c["TRAIL"] + 2
    steps = total + warm
    pos = seed(n)
    hist = np.empty((steps, n, 3), dtype=np.float32)
    speed = np.empty((steps, n), dtype=np.float32)
    # A tracer that reaches the middle is released again at the rim. Without
    # remembering that, the trail joins the old position to the new one and
    # draws a straight line across the whole frame; the first smoke frames were
    # full of them.
    reborn = np.zeros((steps, n), dtype=bool)
    for t in range(steps):
        v, dist = field(pos)
        lost = (dist > step * 1.6) | (np.linalg.norm(pos, axis=1) < 0.25)
        if lost.any():
            pos[lost] = seed(int(lost.sum()))
            v, dist = field(pos)
        reborn[t] = lost
        hist[t] = pos
        speed[t] = np.linalg.norm(v, axis=1)
        pos = pos + v * c["DT"]

    d["hist"] = hist
    d["reborn"] = reborn
    d["warm"] = warm
    d["speed_hi"] = float(np.percentile(speed, 97))
    d["pspeed"] = speed
    return d


# -------------------------------------------------------------- RENDER
def render_frame(idx, total, d, out_path):
    c, figs, L = CONFIG, d["figs"], LAYOUT
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
    fig.text(0.5, L["TITLE"], "THE MARKET PULLS BACK", ha="center", fontsize=26,
             fontweight="bold", color=THEME["TEXT"], family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             f"SPY  |  {figs['n_years']} years  |  {figs['n_states']} states",
             ha="center", fontsize=13, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.5, L["KEY"], "colour = how hard the field pulls",
             ha="center", fontsize=10, color=THEME["TEXT_DIM"],
             family=THEME["FONT"])
    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])

    ax = fig.add_axes(L["AXES"], projection="3d", facecolor=THEME["BG"])

    # >>> REPLACE: the geometry. Arrows build, tracers flow the whole time.
    pts, vecs = d["pts"], d["vecs"]
    k = max(int(round((0.25 + 0.75 * frac) * len(pts))), 1)
    sel = d["arrow_order"][:k]
    segs = [[tuple(pts[i]), tuple(pts[i] + vecs[i] * d["arrow_scale"])]
            for i in sel]
    ax.add_collection3d(Line3DCollection(segs, colors=d["arrow_cols"][sel],
                                         linewidths=1.5, alpha=0.55, zorder=2))
    ax.scatter(pts[sel, 0], pts[sel, 1], pts[sel, 2], s=7,
               c=d["arrow_cols"][sel], alpha=0.5, linewidths=0, zorder=3)

    # Tracers. The trail is drawn as one collection per frame: a segment per
    # particle per step, coloured by the pull it is feeling and fading back.
    j = idx + d["warm"]                   # Versatz um den Vorlauf
    trail = c["TRAIL"]
    hist = d["hist"]
    tsegs, tcols = [], []
    base = CMAP(np.clip(d["pspeed"][j] / max(d["speed_hi"], 1e-9), 0, 1) ** c["GAMMA"])
    alive = np.ones(hist.shape[1], dtype=bool)
    for b in range(trail - 1):
        alive &= ~d["reborn"][j - b]        # a rebirth ends that trail
        if not alive.any():
            break
        a0 = hist[j - b][alive]
        a1 = hist[j - b - 1][alive]
        fade = 0.95 * (1.0 - b / max(trail - 1, 1)) ** 1.5
        cols = base[alive].copy()
        cols[:, 3] = fade
        tsegs.extend(np.stack([a0, a1], axis=1))
        tcols.append(cols)
    if tsegs:
        ax.add_collection3d(Line3DCollection(
            tsegs, colors=np.concatenate(tcols), linewidths=1.6, zorder=5))
    ax.scatter(hist[j][:, 0], hist[j][:, 1], hist[j][:, 2], s=15,
               c=base, alpha=0.95, linewidths=0, zorder=6)

    ax.set_xlim(*d["lim"][0]); ax.set_ylim(*d["lim"][1]); ax.set_zlim(*d["lim"][2])
    ax.view_init(elev=c["ELEV_BASE"] + 6 * np.sin(t * np.pi),
                 azim=c["AZIM_START"] + c["AZIM_SWEEP"] * t)
    ax.set_box_aspect(d["box_aspect"], zoom=c["ZOOM"])
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    fig.text(0.5, L["NOTE"], "tracers, not market paths: they show the field",
             ha="center", fontsize=10, color=THEME["TEXT_DIM"],
             family=THEME["MONO"], alpha=0.85)

    # Readout: strings only, straight from figures.json. Never re-derive here.
    fig.text(0.5, L["BIG"], f"{figs['share_inward']} OF THE FIELD POINTS HOME",
             ha="center", fontsize=19, color=THEME["RED"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             f"{figs['n_inward']} of {figs['n_cells']} cells  ·  "
             f"{figs['horizon']}-session horizon",
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
