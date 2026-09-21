"""
Drawdown_Reel_Pipeline.py
=========================
Reel 11 for @quant.traderr. Bitcoin and the S&P measured in time spent
underwater, drawn as two funnels on the same depth axis.

THE MATHS
    Drawdown is how far below its own running maximum a price sits, in
    percent, on every single day. That gives one number per day per asset, and
    the whole reel is one question asked of it: how much of its life does an
    asset spend at least this far down?

    For a depth d, the share of days with drawdown at or below d is the width
    of the funnel at that depth. At the rim every day qualifies, so both
    funnels start at the same radius; going down, each one narrows at its own
    pace. A funnel that stays wide is an asset that spends its life far from
    its highs.

    Both assets are measured over the same calendar span, the one Bitcoin has,
    and each against its own running maximum.

THE CLAIM THE VISUAL MAKES
    The difference between these two assets is not only how far they fall, it
    is how long they stay down. One funnel is a bowl, the other is a needle.

WHAT THE VISUAL IS NOT
    Not a return comparison, and deliberately so: over this span Bitcoin
    returned far more than the S&P, and none of that is in this picture. This
    is the cost side alone.

    **The two assets do not trade on the same days.** Bitcoin trades every day
    of the year and the S&P about 252 days, so these are shares of each
    asset's own days rather than of a shared calendar. A drawdown series also
    depends on where the sample starts, and this one starts where Bitcoin's
    usable history does. One sample, one cycle era, no forecast.

RUN
    python Drawdown_Reel_Pipeline.py --smoke   # 3 frames -> temp_frames_smoke/
    python Drawdown_Reel_Pipeline.py           # full render -> topic.mp4

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
    "TICKERS": ["BTC-USD", "SPY"],
    "PERIOD": "max",
    "DEPTH_STEPS": 140,       # rows of the funnel wall
    "N_THETA": 64,            # facets around it
    "RIM_R": 0.62,            # radius where every day still qualifies
    "GAP": 0.78,              # half the distance between the two funnels
    "DEEP_Z": 1.75,           # how tall the deepest drawdown is drawn
    "MARKS": [20, 50],        # depths called out in the fixed rows, percent

    # render
    "W": 1080, "H": 1920, "DPI": 100,
    "FPS": 30,
    "HOOK_SEC": 1.4, "BUILD_SEC": 8.5, "HOLD_SEC": 1.8,
    "ELEV_BASE": 12, "AZIM_START": -68, "AZIM_SWEEP": 300,
    "ZOOM": 1.58,
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

# The comparison is the point, so the two called out depths get their own
# fixed rows. Nothing readable sits inside the rotating axes: which funnel is
# on the left changes as the camera goes round, so the legend names colours,
# never sides.
LAYOUT = {
    "TITLE": 0.862, "SUBTITLE": 0.838, "KEY": 0.816, "ROW_A": 0.799,
    "ROW_B": 0.783, "HANDLE": 0.764,
    "AXES": [-0.08, 0.30, 1.16, 0.455],
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
    px = px[CONFIG["TICKERS"]]
    if px.empty:
        raise SystemExit("no data returned; refusing to draw a synthetic tape")
    px.to_csv(path)
    return px


# ------------------------------------------------------------- COMPUTE
def compute(px):
    # >>> REPLACE: the maths for your topic. Return whatever render_frame needs.
    a, b = CONFIG["TICKERS"]
    first = px[a].dropna().index[0]          # start where Bitcoin's history does
    out = {}
    for name in (a, b):
        s = px[name].dropna()
        s = s[s.index >= first]
        dd = (s / s.cummax() - 1.0) * 100.0
        out[name] = dd
    return {"dd": out, "tickers": [a, b],
            "span": (first, px[a].dropna().index[-1])}


# ------------------------------------------------------------ VALIDATE
def validate(d):
    """Every number that reaches a frame is asserted here, before any render.
    A data refresh that moves a figure out of band fails the build instead of
    shipping a caption that no longer matches the picture."""
    a, b = d["tickers"]
    da, db = d["dd"][a], d["dd"][b]

    def anteil(dd, k):                        # share of days at least k% down
        return float((dd <= -k).mean() * 100.0)

    worst_a, worst_b = float(da.min()), float(db.min())
    deep_a, deep_b = anteil(da, 50), anteil(db, 50)
    mid_a, mid_b = anteil(da, 20), anteil(db, 20)
    high_a = float((da > -0.01).mean() * 100.0)
    high_b = float((db > -0.01).mean() * 100.0)

    assert len(da) > 2000, f"only {len(da)} days of {a}"
    assert len(db) > 1500, f"only {len(db)} days of {b}"
    assert -95.0 <= worst_a <= -40.0, f"{a} worst drawdown {worst_a:.0f}% out of band"
    assert -70.0 <= worst_b <= -10.0, f"{b} worst drawdown {worst_b:.0f}% out of band"
    assert deep_a > deep_b, "the comparison collapsed: check the tickers"
    assert 5.0 <= deep_a <= 70.0, f"{a} spends {deep_a:.0f}% of days below -50%"
    assert deep_b < 5.0, f"{b} spends {deep_b:.0f}% of days below -50%, unexpected"

    figs = {
        "a_name": a.replace("-USD", ""), "b_name": b,
        "a_worst": f"{worst_a:.0f}%", "b_worst": f"{worst_b:.0f}%",
        "a_deep": f"{deep_a:.0f}%", "b_deep": f"{deep_b:.0f}%",
        "a_mid": f"{mid_a:.0f}%", "b_mid": f"{mid_b:.0f}%",
        "a_high": f"{high_a:.0f}%", "b_high": f"{high_b:.0f}%",
        "a_days": f"{len(da):,}", "b_days": f"{len(db):,}",
        "mark_mid": f"{CONFIG['MARKS'][0]}%", "mark_deep": f"{CONFIG['MARKS'][1]}%",
        "first_year": str(d["span"][0].year), "last_year": str(d["span"][1].year),
    }
    with open(os.path.join(BASE_DIR, "figures.json"), "w") as fh:
        json.dump(figs, fh, indent=1)
    log(f"[validate] {figs}")
    d["figs"] = figs
    return figs


def _funnel(dd, depths, c, x0):
    """Surface of revolution: the width at a depth is the share of days that
    were at least that far down."""
    share = np.array([(dd <= -z).mean() for z in depths])
    r = c["RIM_R"] * share
    th = np.linspace(0, 2 * np.pi, c["N_THETA"])
    R, TH = np.meshgrid(r, th, indexing="ij")          # both (depth, theta)
    Z = np.repeat(depths[:, None], c["N_THETA"], axis=1)
    X = x0 + R * np.cos(TH)
    Y = R * np.sin(TH)
    return X, Y, Z, share


def prepare_geometry(d):
    """Both funnels, on one shared depth axis, computed once."""
    c = CONFIG
    a, b = d["tickers"]
    tiefe = min(float(d["dd"][a].min()), float(d["dd"][b].min()))
    depths = np.linspace(0.0, -tiefe, c["DEPTH_STEPS"])       # 0 .. 83 positive
    # Depth in data units is mapped to height so the deepest point of the
    # deeper asset sits at DEEP_Z. Both funnels share the mapping, otherwise
    # the comparison would be a lie told with axes.
    skala = c["DEEP_Z"] / max(depths[-1], 1e-9)

    Xa, Ya, Za, sa = _funnel(d["dd"][a], depths, c, -c["GAP"])
    Xb, Yb, Zb, sb = _funnel(d["dd"][b], depths, c, +c["GAP"])
    d["A"] = (Xa, Ya, -Za * skala)
    d["B"] = (Xb, Yb, -Zb * skala)

    # Colour is depth, the same ramp for both funnels, so a colour means the
    # same thing wherever it appears.
    norm = depths / max(depths[-1], 1e-9)
    cols = CMAP(norm)[:-1]
    d["cols"] = np.repeat(cols[:, None, :], c["N_THETA"] - 1, axis=1)
    d["grey"] = np.full((len(depths) - 1, c["N_THETA"] - 1, 4), 0.0)
    d["grey"][..., :3] = 0.62
    d["grey"][..., 3] = 1.0
    d["depths"], d["skala"] = depths, skala
    d["lim_xy"] = c["GAP"] + c["RIM_R"] * 1.35
    return d


# -------------------------------------------------------------- RENDER
def render_frame(idx, total, d, out_path):
    c, figs, L = CONFIG, d["figs"], LAYOUT
    if "A" not in d:
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
    fig.text(0.5, L["TITLE"], "THE SHAPE OF PAIN", ha="center", fontsize=28,
             fontweight="bold", color=THEME["TEXT"], family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             f"{figs['a_name']} and {figs['b_name']}  |  {figs['first_year']} to "
             f"{figs['last_year']}  |  drawdown every day",
             ha="center", fontsize=13, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.5, L["KEY"],
             f"{figs['a_name']} is the coloured funnel, {figs['b_name']} the grey one",
             ha="center", fontsize=10, color=THEME["TEXT_DIM"],
             family=THEME["FONT"])
    fig.text(0.5, L["ROW_A"],
             f"deeper than -{figs['mark_mid']}   {figs['a_name']} {figs['a_mid']}"
             f"   {figs['b_name']} {figs['b_mid']}",
             ha="center", va="center", fontsize=9, color=THEME["TEXT"],
             family=THEME["MONO"])
    fig.text(0.5, L["ROW_B"],
             f"deeper than -{figs['mark_deep']}   {figs['a_name']} {figs['a_deep']}"
             f"   {figs['b_name']} {figs['b_deep']}",
             ha="center", va="center", fontsize=9, color=THEME["RED"],
             family=THEME["MONO"])
    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])

    ax = fig.add_axes(L["AXES"], projection="3d", facecolor=THEME["BG"],
                      computed_zorder=False)

    # >>> REPLACE: the geometry. Both funnels sink at the same rate.
    n = len(d["depths"])
    k = max(int(round(n * (0.18 + 0.82 * frac))), 2)
    for (X, Y, Z), fc, zo in ((d["A"], d["cols"], 3), (d["B"], d["grey"], 2)):
        ax.plot_surface(X[:k], Y[:k], Z[:k], facecolors=fc[:k - 1], linewidth=0,
                        antialiased=False, shade=False, alpha=1.0,
                        rstride=1, cstride=1, edgecolor="none", zorder=zo)

    lim = d["lim_xy"]
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_zlim(-c["DEEP_Z"] * 1.04, c["DEEP_Z"] * 0.18)
    ax.view_init(elev=c["ELEV_BASE"] + 7 * np.sin(t * np.pi),
                 azim=c["AZIM_START"] + c["AZIM_SWEEP"] * t)
    ax.set_box_aspect((1.0, 1.0, 0.95), zoom=c["ZOOM"])
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    # Readout: strings only, straight from figures.json. Never re-derive here.
    fig.text(0.5, L["NOTE"],
             "the width at a depth is the share of days at least that far down",
             ha="center", fontsize=10, color=THEME["TEXT_DIM"],
             family=THEME["FONT"])
    fig.text(0.5, L["BIG"],
             f"{figs['a_deep']} OF DAYS OVER {figs['mark_deep']} DOWN",
             ha="center", fontsize=19, color=THEME["RED"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             f"{figs['b_name']} {figs['b_deep']}  ·  worst {figs['a_worst']} "
             f"against {figs['b_worst']}",
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
