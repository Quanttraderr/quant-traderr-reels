"""
TenDays_Reel_Pipeline.py
========================
Reel 3 for @quant.traderr. Ten years of SPY as a climbing spiral, next to the
same decade with its ten best sessions removed.

THE MATHS
    Daily total return of SPY over PERIOD. The bright spiral is the cumulative
    growth of one unit, compounded day by day. The dim spiral is the identical
    series with the N_MISSED largest daily returns dropped and the rest
    recompounded in place, so both curves see exactly the same sessions bar
    those few. Height is wealth, the angle is elapsed time wrapped into TURNS
    turns for display, and the radius opens with time so the coil does not sit
    on top of itself. The drawdown on any session is the distance from the
    running high of the price series.

THE CLAIM THE VISUAL MAKES
    A handful of sessions carries most of a decade, and those sessions are not
    scattered through the calm stretches. They happen while the market is deep
    below its high, which is exactly when sitting out feels most sensible.

WHAT THE VISUAL IS NOT
    Not a strategy and not advice. Nobody can remove only the best days, and the
    mirror case of dodging the worst days is just as extreme and is not shown
    here, so this is an argument about timing being hard, not proof that holding
    wins. The angle is a display wrap, not a calendar cycle. One index, one
    decade, dividends reinvested, no costs and no tax.

RUN
    python TenDays_Reel_Pipeline.py --smoke   # 3 frames -> temp_frames_smoke/
    python TenDays_Reel_Pipeline.py           # full render -> topic.mp4

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
    "PERIOD": "10y",
    "N_MISSED": 10,
    "DD_THRESHOLD": 0.10,     # what counts as "inside a drawdown"
    "TURNS": 3.0,             # display wrap of the time axis
    "R_INNER": 0.22,
    "R_OUTER": 0.52,
    "SEP": 1.05,              # half the distance between the two coils

    # render (leave these alone unless the topic needs landscape)
    "W": 1440, "H": 2560, "DPI": 400 / 3,   # 2K master. The figure stays
    # 10.8x19.2in, so every fontsize and linewidth lands where it would at
    # 1080x1920; only the sampling density goes up.
    "FPS": 30,
    "HOOK_SEC": 1.4, "BUILD_SEC": 8.5, "HOLD_SEC": 1.8,
    "ELEV_BASE": 15, "AZIM_START": -80, "AZIM_SWEEP": 24,
    "ZOOM": 1.25,
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
    c = CONFIG
    s = px[c["TICKERS"][0]].dropna()
    ret = s.pct_change().dropna()
    n = len(ret)

    # The N best sessions, and the same decade with exactly those removed. The
    # returns are set to zero rather than deleted, so both curves stay aligned
    # to the same calendar and the spirals can be compared point for point.
    best = ret.nlargest(c["N_MISSED"]).index
    ret_ex = ret.copy()
    ret_ex.loc[best] = 0.0

    wealth = (1.0 + ret).cumprod().values
    wealth_ex = (1.0 + ret_ex).cumprod().values

    drawdown = (s / s.cummax() - 1.0).reindex(ret.index).values
    best_pos = np.array([ret.index.get_loc(b) for b in best])
    in_dd = float((drawdown[best_pos] < -c["DD_THRESHOLD"]).mean())

    # Two coils side by side rather than one on top of the other. Stacked, the
    # bright coil's early turns sat at the same height as the dim coil's later
    # ones and the eye could not tell them apart; separated, the shorter tower
    # is the message and it reads at thumbnail size.
    t = np.arange(n) / max(n - 1, 1)
    angle = 2.0 * np.pi * c["TURNS"] * t
    radius = c["R_INNER"] + (c["R_OUTER"] - c["R_INNER"]) * t
    cx, cy = radius * np.cos(angle), radius * np.sin(angle)
    xs, ys = cx - c["SEP"], cy
    xs_ex, ys_ex = cx + c["SEP"], cy

    return {"ret": ret, "index": ret.index, "wealth": wealth,
            "wealth_ex": wealth_ex, "xy": (xs, ys), "xy_ex": (xs_ex, ys_ex),
            "drawdown": drawdown,
            "best_pos": best_pos, "in_dd": in_dd, "n": n,
            "best_move": float(ret.loc[best[0]]),
            "best_date": best[0], "best_dd": float(drawdown[best_pos[0]])}


# ------------------------------------------------------------ VALIDATE
def validate(d):
    """Every number that reaches a frame is asserted here, before any render.

    The bands are what these quantities can be over any decade of a broad equity
    index, not what they happen to be today. A decade that compounded to less
    than 1.2x or more than 20x, or a single session outside 3 to 25 percent,
    means the data is wrong, not that the market did something interesting."""
    c = CONFIG
    full, ex = float(d["wealth"][-1]), float(d["wealth_ex"][-1])
    share_lost = 1.0 - (ex - 1.0) / (full - 1.0)

    assert 1.2 <= full <= 20.0, f"decade growth {full:.2f}x outside the 1.2-20 band"
    assert ex < full, "removing the best days did not lower the result; check the join"
    assert 0.15 <= share_lost <= 0.95, f"share lost {share_lost:.2f} outside the 0.15-0.95 band"
    assert 0.03 <= d["best_move"] <= 0.25, f"best session {d['best_move']:.3f} outside the 3-25% band"
    assert 0.30 <= d["in_dd"] <= 1.0, f"only {d['in_dd']:.2f} of the best days sat in a drawdown"
    assert 2000 <= d["n"] <= 2700, f"{d['n']} sessions is not a decade of trading days"
    assert len(d["best_pos"]) == c["N_MISSED"], "wrong number of removed sessions"

    figs = {
        "gain_all": f"{(full - 1) * 100:.0f}%",
        "gain_ex": f"{(ex - 1) * 100:.0f}%",
        "share_lost": f"{share_lost * 100:.0f}%",
        "n_missed": str(c["N_MISSED"]),
        "in_drawdown": f"{d['in_dd'] * 100:.0f}%",
        "dd_threshold": f"{c['DD_THRESHOLD'] * 100:.0f}%",
        "best_date": d["best_date"].strftime("%b %Y"),
        "best_move": f"{d['best_move'] * 100:.1f}%",
        "n_sessions": f"{d['n']:,}",
    }
    with open(os.path.join(BASE_DIR, "figures.json"), "w") as fh:
        json.dump(figs, fh, indent=1)
    log(f"[validate] {figs}")
    d["figs"] = figs
    return figs


def prepare_geometry(d):
    """Limits, aspect and the colour normalisation, computed once so every frame
    shares them. Per-frame limits would make the coil breathe as it grows."""
    c = CONFIG
    zmax = float(d["wealth"].max())
    pad = 0.06 * zmax
    half = c["SEP"] + c["R_OUTER"]
    d["lim"] = [(-half - pad, half + pad),
                (-c["R_OUTER"] - pad, c["R_OUTER"] + pad),
                (0.80, zmax + pad)]
    span = np.array([b - a for a, b in d["lim"]])
    d["box_aspect"] = tuple(span / span.max())

    # Daily moves are long tailed. Normalising the ramp to the 97th percentile
    # of the absolute move keeps ordinary sessions readable instead of flattening
    # everything below one crash day.
    d["move_hi"] = float(np.percentile(np.abs(d["ret"].values), 97))
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
    fig.text(0.5, L["TITLE"], "THE TEN DAYS", ha="center", fontsize=30,
             fontweight="bold", color=THEME["TEXT"], family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             f"SPY  |  {figs['n_sessions']} sessions  |  height = wealth",
             ha="center", fontsize=13, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.5, L["KEY"], "colour = size of that session's move",
             ha="center", fontsize=10, color=THEME["TEXT_DIM"],
             family=THEME["FONT"])
    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])

    ax = fig.add_axes(L["AXES"], projection="3d", facecolor=THEME["BG"])

    # >>> REPLACE: the geometry. Two coils climb together and split apart.
    xs, ys = d["xy"]
    xe, ye = d["xy_ex"]
    w, w_ex = d["wealth"], d["wealth_ex"]
    k = max(int(round(frac * (d["n"] - 1))), 1)

    # The dim coil first, so the bright one always wins an overlap.
    seg_ex = np.stack([np.column_stack([xe[:k], ye[:k], w_ex[:k]]),
                       np.column_stack([xe[1:k + 1], ye[1:k + 1], w_ex[1:k + 1]])],
                      axis=1)
    ax.add_collection3d(Line3DCollection(seg_ex, colors=(0.55, 0.60, 0.70, 0.55),
                                         linewidths=1.0, capstyle="round",
                                         zorder=2))

    seg = np.stack([np.column_stack([xs[:k], ys[:k], w[:k]]),
                    np.column_stack([xs[1:k + 1], ys[1:k + 1], w[1:k + 1]])],
                   axis=1)
    mv = np.clip(np.abs(d["ret"].values[1:k + 1]) / max(d["move_hi"], 1e-9), 0, 1)
    ax.add_collection3d(Line3DCollection(seg, colors=CMAP(mv), linewidths=1.7,
                                         capstyle="round", zorder=4))

    # A rung between the two towers every so often, so the eye pairs them up in
    # time instead of reading two unrelated objects, and a brighter rung at the
    # point both have reached. The rungs fanning apart are the whole argument.
    step = max(d["n"] // 38, 1)
    ax.add_collection3d(Line3DCollection(
        [[(xs[i], ys[i], w[i]), (xe[i], ye[i], w_ex[i])] for i in range(0, k, step)],
        colors=(0.62, 0.68, 0.80, 0.20), linewidths=0.7, zorder=3))
    ax.plot([xs[k], xe[k]], [ys[k], ye[k]], [w[k], w_ex[k]],
            color=THEME["TEXT"], alpha=0.55, linewidth=1.2, zorder=5)

    # The removed sessions, as beads on the bright coil.
    shown = d["best_pos"][d["best_pos"] <= k]
    if len(shown):
        ax.scatter(xs[shown], ys[shown], w[shown], s=150, color=THEME["RED"],
                   alpha=0.18, linewidths=0, zorder=6)
        ax.scatter(xs[shown], ys[shown], w[shown], s=34, color=THEME["TEXT"],
                   alpha=0.95, linewidths=0, zorder=7)

    for px_, py_, pz_ in ((xs[k], ys[k], w[k]), (xe[k], ye[k], w_ex[k])):
        ax.scatter([px_], [py_], [pz_], s=420, color=THEME["ORANGE"],
                   alpha=0.13, linewidths=0, zorder=8)
        ax.scatter([px_], [py_], [pz_], s=70, color=THEME["ORANGE"],
                   alpha=1.0, linewidths=0, zorder=9)

    ax.set_xlim(*d["lim"][0]); ax.set_ylim(*d["lim"][1]); ax.set_zlim(*d["lim"][2])
    ax.view_init(elev=c["ELEV_BASE"] + 6 * np.sin(t * np.pi),
                 azim=c["AZIM_START"] + c["AZIM_SWEEP"] * t)
    ax.set_box_aspect(d["box_aspect"], zoom=c["ZOOM"])
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    fig.text(0.5, L["NOTE"], "left: every session.  right: the same decade minus ten",
             ha="center", fontsize=10, color=THEME["TEXT_DIM"],
             family=THEME["MONO"], alpha=0.85)

    # Readout: strings only, straight from figures.json. Never re-derive here.
    fig.text(0.5, L["BIG"], f"{figs['gain_all']} BECOMES {figs['gain_ex']}",
             ha="center", fontsize=22, color=THEME["RED"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             f"{figs['n_missed']} missed sessions  ·  {figs['in_drawdown']} of "
             f"them inside a drawdown",
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
