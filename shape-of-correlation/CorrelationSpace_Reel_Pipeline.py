"""
CorrelationSpace_Reel_Pipeline.py
=================================
Reel 1 for @quant.traderr. 44 US large caps placed in a metric space built from
their return correlations, wired by the minimum spanning tree.

THE MATHS
    Daily log returns over PERIOD. Pearson correlation for every pair, turned
    into the Mantegna distance d_ij = sqrt(2 * (1 - rho_ij)), which is a proper
    metric. Classical MDS (double centre the squared distance matrix, take the
    top three eigenvectors scaled by the square root of their eigenvalues)
    places every name in 3D. The edges are the minimum spanning tree of the full
    distance matrix, so exactly n - 1 links, the cheapest wiring that still
    reaches every name.

THE CLAIM THE VISUAL MAKES
    A watchlist is a list. The same names under a distance are a shape, and the
    shape has a middle, an edge, and clusters nobody labelled.

WHAT THE VISUAL IS NOT
    Not causation. Not a forecast. Not a sector map: no sector label enters the
    maths at any point. Not a faithful picture of the distance matrix either,
    because three dimensions hold only part of its structure and that share is
    asserted and shown. Distances between two points that share no tree edge are
    approximations, not measurements.

RUN
    python CorrelationSpace_Reel_Pipeline.py --smoke   # 3 frames -> temp_frames_smoke/
    python CorrelationSpace_Reel_Pipeline.py           # full render -> topic.mp4

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
    # 44 US large caps spread across sectors. The spread is deliberate: a basket
    # from one sector would manufacture the clustering the reel claims to find.
    "TICKERS": ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "AVGO", "ORCL",
                "CRM", "AMD",
                "JPM", "BAC", "WFC", "GS", "MS", "BLK", "SPGI",
                "JNJ", "UNH", "LLY", "PFE", "ABBV", "MRK", "TMO",
                "XOM", "CVX", "COP", "SLB", "EOG",
                "PG", "KO", "PEP", "WMT", "COST", "MCD", "NKE", "HD",
                "CAT", "HON", "UNP", "GE",
                "NEE", "DUK", "SO"],
    "PERIOD": "3y",
    "N_LABELS": 6,

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
    "ELEV_BASE": 18, "AZIM_START": -68, "AZIM_SWEEP": 34,
    "ZOOM": 1.52,
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
    log(f"[Data] fetching {len(CONFIG['TICKERS'])} names")
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
    from scipy.sparse.csgraph import minimum_spanning_tree

    names = list(px.columns)
    n = len(names)
    rets = np.log(px / px.shift(1)).dropna()
    corr = np.array(rets.corr().values, dtype=float)   # .values is read-only
    np.fill_diagonal(corr, 1.0)

    # Mantegna distance: a real metric, unlike 1 - rho.
    dist = np.sqrt(np.clip(2.0 * (1.0 - corr), 0.0, None))

    # Classical MDS. Double centre the squared distances, then the top three
    # eigenvectors scaled by sqrt(eigenvalue) are the coordinates.
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (dist ** 2) @ J
    evals, evecs = np.linalg.eigh(B)
    order = np.argsort(evals)[::-1]
    top3 = order[:3]
    pos = evecs[:, top3] * np.sqrt(np.clip(evals[top3], 0.0, None))

    # eigh returns eigenvectors up to a sign, which would mirror the cloud
    # between runs. Pin each axis so the most extreme name on it is positive.
    for j in range(3):
        if pos[np.argmax(np.abs(pos[:, j])), j] < 0:
            pos[:, j] *= -1.0

    # The frame is 9:16, so the cloud's widest direction is put on the vertical
    # axis. This is a camera choice, not a data choice: the three coordinates
    # keep one shared scale, so no distance is stretched by the swap.
    pos = pos[:, [1, 2, 0]]

    pos_total = float(np.clip(evals, 0.0, None).sum())
    mds_share = float(np.clip(evals[top3], 0.0, None).sum() / pos_total)

    # Minimum spanning tree over the full distance matrix: n - 1 links.
    mst = minimum_spanning_tree(dist).tocoo()
    edges = [(int(a), int(b), float(corr[a, b]))
             for a, b in zip(mst.row, mst.col)]
    edges.sort(key=lambda e: -e[2])          # strongest link appears first

    # How much each name moves with everything else. This drives the colour, so
    # the ramp says the same thing as the position: hot means central.
    avg_corr = (corr.sum(axis=1) - 1.0) / (n - 1)

    iu = np.triu_indices(n, 1)
    k = int(np.argmax(corr[iu]))
    top_pair = (names[iu[0][k]], names[iu[1][k]], float(corr[iu][k]))

    return {"names": names, "corr": corr, "pos": pos, "edges": edges,
            "avg_corr": avg_corr, "mean_corr": float(corr[iu].mean()),
            "mds_share": mds_share, "top_pair": top_pair,
            "n_sessions": len(rets)}


# ------------------------------------------------------------ VALIDATE
def validate(d):
    """Every number that reaches a frame is asserted here, before any render.
    A data refresh that moves a figure out of band fails the build instead of
    shipping a caption that no longer matches the picture.

    The bands are what these quantities can be across any regime, not what they
    happen to be today. Average pairwise correlation of a broad US large cap
    basket has run near 0.1 in a dispersed tape and above 0.8 in a panic, so the
    band is wide on purpose and still catches a broken join or a bad fetch."""
    names, n = d["names"], len(d["names"])
    mean_corr = d["mean_corr"]
    hub = names[int(np.argmax(d["avg_corr"]))]
    rim = names[int(np.argmin(d["avg_corr"]))]
    a, b, pair_corr = d["top_pair"]

    assert 0.05 <= mean_corr <= 0.90, f"mean correlation {mean_corr:.2f} outside the 0.05-0.90 band"
    assert 0.40 <= pair_corr <= 0.999, f"top pair correlation {pair_corr:.2f} outside the 0.40-0.999 band"
    assert pair_corr > mean_corr, "the tightest pair is not above the average; check the join"
    assert len(d["edges"]) == n - 1, f"{len(d['edges'])} tree links, expected {n - 1}"
    assert d["n_sessions"] > 400, f"only {d['n_sessions']} sessions, too few to correlate"
    assert 0.10 <= d["mds_share"] <= 0.99, f"3D share {d['mds_share']:.2f} outside the 0.10-0.99 band"
    assert hub != rim, "hub and rim are the same name; the basket collapsed"

    figs = {
        "mean_corr": f"{mean_corr:.2f}",
        "top_pair": f"{a}/{b}",
        "top_pair_corr": f"{pair_corr:.2f}",
        "hub_name": hub,
        "rim_name": rim,
        "n_names": str(n),
        "n_sessions": f"{d['n_sessions']}",
        "n_links": str(n - 1),
        "mds_share": f"{d['mds_share'] * 100:.0f}%",
    }
    with open(os.path.join(BASE_DIR, "figures.json"), "w") as fh:
        json.dump(figs, fh, indent=1)
    log(f"[validate] {figs}")
    d["figs"] = figs
    return figs


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
    fig.text(0.5, L["TITLE"], "THE SHAPE OF CORRELATION", ha="center", fontsize=27,
             fontweight="bold", color=THEME["TEXT"], family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             f"{figs['n_names']} US large caps  |  {figs['n_sessions']} sessions"
             f"  |  {figs['n_links']} tree links",
             ha="center", fontsize=13, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.5, L["KEY"],
             "distance = sqrt(2(1 - rho)), colour = how much a name moves with the rest",
             ha="center", fontsize=9.5, color=THEME["TEXT_DIM"],
             family=THEME["FONT"])
    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])

    ax = fig.add_axes(L["AXES"], projection="3d", facecolor=THEME["BG"])

    # >>> REPLACE: the geometry. The cloud stands from the first frame, the
    # minimum spanning tree wires itself in strongest link first.
    pos, edges, avg = d["pos"], d["edges"], d["avg_corr"]
    lo, hi = float(avg.min()), float(avg.max())
    norm = (avg - lo) / max(hi - lo, 1e-9)
    cols = CMAP(norm)

    # Edges carry pair correlation, nodes carry average correlation. Two
    # different quantities, so each gets its own normalisation: reusing the node
    # range here would push every link to the top of the ramp.
    e_lo, e_hi = d["edge_range"]
    n_edge = int(round(frac * len(edges)))
    for a, b, rho in edges[:n_edge]:
        w = (rho - e_lo) / max(e_hi - e_lo, 1e-9)
        ax.plot([pos[a, 0], pos[b, 0]], [pos[a, 1], pos[b, 1]],
                [pos[a, 2], pos[b, 2]], color=CMAP(np.clip(w, 0, 1)),
                linewidth=1.3, alpha=0.7, solid_capstyle="round", zorder=1)

    # glow, then core: two scatters at the same points is the cheapest way to
    # make a node read as a light source on black.
    sizes = 58.0 + 360.0 * norm
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=sizes * 5.0, c=cols,
               alpha=0.10, linewidths=0, zorder=2)
    ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=sizes, c=cols,
               alpha=0.95, linewidths=0, zorder=3)

    # Labels sit on their own node, so they are inside the limits by
    # construction. A label anchored to an axis end gets clipped by the canvas.
    for i in d["label_idx"]:
        dx, dy = d["label_off"][i]
        ax.text(pos[i, 0] + dx, pos[i, 1] + dy, pos[i, 2] + d["label_dz"],
                d["names"][i], color=THEME["TEXT"], fontsize=8.5,
                ha="center", va="bottom", family=THEME["MONO"], alpha=0.85,
                zorder=20)

    ax.set_xlim(*d["lim"][0]); ax.set_ylim(*d["lim"][1]); ax.set_zlim(*d["lim"][2])
    ax.view_init(elev=c["ELEV_BASE"] + 6 * np.sin(t * np.pi),
                 azim=c["AZIM_START"] + c["AZIM_SWEEP"] * t)
    ax.set_box_aspect(d["box_aspect"], zoom=c["ZOOM"])
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    # Readout: strings only, straight from figures.json. Never re-derive here.
    fig.text(0.5, L["BIG"], f"MEAN CORRELATION {figs['mean_corr']}", ha="center",
             fontsize=19, color=THEME["RED"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             f"tightest {figs['top_pair']} {figs['top_pair_corr']}  ·  "
             f"hub {figs['hub_name']}  ·  rim {figs['rim_name']}",
             ha="center", fontsize=12, color=THEME["TEXT_DIM"],
             family=THEME["MONO"])

    fig.savefig(out_path, dpi=c["DPI"], facecolor=THEME["BG"])
    plt.close(fig)


def prepare_geometry(d):
    """Limits and label picks, computed once so every frame shares them. A
    per-frame limit would make the cloud breathe as the tree grows."""
    pos = d["pos"]
    # One shared symmetric range for all three axes. Per-axis limits would
    # stretch the weaker MDS directions to fill the box, and in a picture whose
    # whole subject is distance, that is not a cosmetic choice: it would make
    # two names look closer or further apart than the maths says.
    # The box wraps the cloud tightly, and box_aspect is set to the same
    # proportions as the limits. That combination is what keeps one shared scale
    # on all three axes: a cube around a non-cubic cloud wastes most of the
    # frame, but a tight box with the default aspect would stretch the weaker
    # directions and lie about distance.
    lo, hi = pos.min(axis=0), pos.max(axis=0)
    pad = 0.06 * float((hi - lo).max())
    d["lim"] = [(float(lo[j]) - pad, float(hi[j]) + pad) for j in range(3)]
    span = np.array([b - a for a, b in d["lim"]])
    d["box_aspect"] = tuple(span / span.max())
    d["label_dz"] = 0.030 * span[2]

    # Every label is nudged back towards the middle of the cloud. A name on the
    # far left otherwise sets its text half outside the canvas, and a 3D axes
    # clips that without a word of warning.
    ctr = (np.array([a for a, _ in d["lim"]]) + np.array([b for _, b in d["lim"]])) / 2.0
    d["label_off"] = {
        i: (-np.sign(pos[i, 0] - ctr[0]) * 0.075 * span[0],
            -np.sign(pos[i, 1] - ctr[1]) * 0.075 * span[1])
        for i in range(len(pos))
    }

    rho = [e[2] for e in d["edges"]]
    d["edge_range"] = (float(min(rho)), float(max(rho)))

    # Hub and rim first, then the extreme name on each axis so the labels span
    # the cloud. The tightest pair is deliberately not labelled: GS and MS sit
    # almost on top of each other, their two labels collided, and the readout
    # already names them.
    avg = d["avg_corr"]
    picks = [int(np.argmax(avg)), int(np.argmin(avg))]
    for j in range(3):
        picks.append(int(np.argmax(np.abs(pos[:, j]))))
        picks.append(int(np.argmin(pos[:, j])))
    seen, out = set(), []
    for i in picks:
        if i not in seen:
            seen.add(i); out.append(i)
    d["label_idx"] = out[:CONFIG["N_LABELS"]]
    return d


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
