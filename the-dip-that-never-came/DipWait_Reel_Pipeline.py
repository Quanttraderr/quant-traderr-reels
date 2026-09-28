"""
DipWait_Reel_Pipeline.py
========================
Reel 14 for @quant.traderr, "The Dip That Never Came". Every one-year window
of SPY since 1993, started on every trading day, asked one question: if you
said "I'll buy when it drops 10%", did that drop ever come inside the year?

THE MATHS
    For each start day s, the path is price(s + k) / price(s) - 1 for
    k = 0 .. 252 trading days, dividends included (adjusted closes). The dip
    "came" if the path reached -10% at any point inside the year. For the
    windows where it never came we record where the price stood on day 252,
    which is the gain the waiter watched from cash. The waiter's strategy:
    hold cash (at zero return) until the path first touches -10%, buy there,
    hold to day 252. No dip, no trade.

    In 3D the geometry is cylindrical. Height is the trading day inside the
    window (day 0 at the bottom, day 252 at the top). The start date runs
    around the circle, 1993 to 2025 once round. The return is the distance
    from the central axis: every path starts on the white entry ring at
    radius 1 and moves outward when it gains, inward when it loses. The red
    tube is the -10% level. A path that pierces the tube got its dip; the
    white dot marks the fill. A path that stays outside the tube for the
    whole year is a year the dip never came, coloured by the gain it made.
    All paths grow upward together, one trading day per step.

THE CLAIM THE VISUAL MAKES
    In most one-year windows since 1993 the -10% dip never arrived, and in
    those years SPY finished well up. Waiting for the dip mostly means
    watching from cash.

WHAT THE VISUAL IS NOT
    Not a forecast and not a case against ever holding cash. The windows
    overlap heavily, day-by-day starts share almost the whole year with their
    neighbours, so the 8,000 windows are not 8,000 independent trials: this is
    about 33 years of one index in a period that went mostly up. The waiter's
    cash earns zero here; T-bills paid real money in parts of this history and
    would narrow the gap. In the crash years (2008) waiting did pay. The radius
    is a linear map of the return, clipped at -45% and +60% for display only;
    every figure uses the unclipped paths. Only every 10th window is drawn;
    every figure uses all of them.

RUN
    python DipWait_Reel_Pipeline.py --smoke   # 3 frames -> temp_frames_smoke/
    python DipWait_Reel_Pipeline.py           # full render -> topic.mp4

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
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from mpl_toolkits.mplot3d.art3d import Line3DCollection

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "_cache")

CONFIG = {
    "TICKERS": ["SPY"],
    "PERIOD": "max",
    "WINDOW": 252,            # trading days in one holding year
    "DIP": -10.0,             # the dip the waiter wants, percent below the entry
    "DRAW_EVERY": 10,         # draw every n-th window, figures use all
    "R_SCALE": 2.0,           # radius = 1 + R_SCALE * return
    "RET_LO": -45.0,          # display clip of the return, percent
    "RET_HI": 60.0,
    "GAIN_TOP": 40.0,         # end return that maps to the top of the colour scale

    # render
    "W": 1080, "H": 1920, "DPI": 100,
    "FPS": 30,
    "HOOK_SEC": 1.4, "BUILD_SEC": 8.5, "HOLD_SEC": 1.8,
    "ELEV_BASE": 36, "AZIM_START": -60, "AZIM_SWEEP": 40,
    "ZOOM": 1.1,
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
# the no-dip years use the warm half of the house scale: green = small gain
# missed, red = big gain missed
GAIN_CMAP = LinearSegmentedColormap.from_list(
    "gain", [THEME["GREEN"], THEME["YELLOW"], THEME["ORANGE"], THEME["RED"]], N=256)

SAFE = {"TOP": 0.12, "BOTTOM": 0.20, "RIGHT": 0.14}

LAYOUT = {
    "TITLE": 0.862, "SUBTITLE": 0.838, "KEY": 0.816, "ROW_A": 0.799,
    "ROW_B": 0.783, "HANDLE": 0.764,
    "AXES": [-0.08, 0.285, 1.16, 0.475],
    "NOTE": 0.272, "BIG": 0.242, "SUB": 0.215,
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------- DATA
def fetch():
    """yfinance once, then a versioned CSV cache. Delete _cache/ to refetch."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "closes.csv")
    if os.path.exists(path):
        log(f"[Data] cache {path}")
        return pd.read_csv(path, index_col=0, parse_dates=True)
    import yfinance as yf
    log(f"[Data] fetching {CONFIG['TICKERS']} {CONFIG['PERIOD']}")
    px = yf.download(CONFIG["TICKERS"], period=CONFIG["PERIOD"],
                     interval="1d", progress=False, auto_adjust=True)["Close"]
    if px.empty:
        raise SystemExit("no data returned; refusing to draw a synthetic tape")
    px = px[CONFIG["TICKERS"]]
    # today's bar may still be trading; only finished sessions go in
    px = px[px.index < pd.Timestamp.today().normalize()]
    px.to_csv(path)
    return px


# ------------------------------------------------------------- COMPUTE
def compute(px):
    s = px["SPY"].dropna()
    v = s.values
    N = CONFIG["WINDOW"]
    starts = np.arange(len(v) - N)
    idx = starts[:, None] + np.arange(N + 1)[None, :]
    paths = (v[idx] / v[starts][:, None] - 1.0) * 100.0      # (windows, N+1)
    runmin = np.minimum.accumulate(paths, axis=1)
    dip = runmin[:, -1] <= CONFIG["DIP"]
    touch = np.where(dip, np.argmax(paths <= CONFIG["DIP"], axis=1), -1)
    end = paths[:, -1]
    # the waiter: cash until the first touch of -10%, then long to day 252
    buy = 1.0 + CONFIG["DIP"] / 100.0
    waiter = np.where(dip, ((1.0 + end / 100.0) / buy - 1.0) * 100.0, 0.0)
    return {"paths": paths, "runmin": runmin, "dip": dip, "touch": touch,
            "end": end, "waiter": waiter, "dates": s.index[starts]}


# ------------------------------------------------------------ VALIDATE
def validate(d):
    """Every number that reaches a frame is asserted here, before any render."""
    dip, end, waiter, D = d["dip"], d["end"], d["waiter"], CONFIG["DIP"]
    n = len(dip)
    nod = ~dip
    p_came = float(dip.mean() * 100)
    p_never = float(nod.mean() * 100)
    med_missed = float(np.median(end[nod]))
    # a fall of |D| from the year-end price still sits above the entry iff
    # end >= 1 / (1 + D) - 1, i.e. +11.1% for a 10% dip
    undo = (1.0 / (1.0 + D / 100.0) - 1.0) * 100.0
    p_out_of_reach = float((end[nod] >= undo).mean() * 100)
    p_never_up = float((end[nod] > 0).mean() * 100)
    mean_buy = float(end.mean())
    mean_wait = float(waiter.mean())
    p_wait_wins = float((waiter > end).mean() * 100)
    med_days = float(np.median(d["touch"][dip]))

    assert n > 6000, f"only {n} windows"
    assert 40.0 <= p_never <= 90.0, f"{p_never:.0f}% of windows never dipped"
    assert abs(p_never + p_came - 100.0) < 1e-9
    assert 3.0 <= med_missed <= 40.0, f"median missed {med_missed:.1f}"
    assert 20.0 <= p_out_of_reach <= 95.0, f"{p_out_of_reach:.0f}% out of reach"
    assert 70.0 <= p_never_up <= 100.0, f"{p_never_up:.0f}% of no-dip ended up"
    assert 3.0 <= mean_buy <= 20.0, f"mean buy-now {mean_buy:.1f}"
    assert -10.0 <= mean_wait < mean_buy, f"mean waiter {mean_wait:.1f}"
    assert 10.0 <= p_wait_wins <= 60.0, f"waiter wins {p_wait_wins:.0f}%"
    assert 5.0 <= med_days <= 252.0, f"median days to dip {med_days}"
    # waiting must have paid somewhere, or the docstring's 2008 line lies
    y08 = d["dates"].year == 2008
    assert (waiter[y08] > end[y08]).mean() > 0.9, "2008 windows missing"
    assert abs(undo - 11.1) < 0.05

    figs = {
        "windows": f"{n:,}",
        "never": f"{p_never:.0f}%",
        "came": f"{p_came:.0f}%",
        "missed": f"+{med_missed:.0f}%",
        "out_of_reach": f"{p_out_of_reach:.0f}%",
        "undo": f"+{undo:.0f}%",
        "never_up": f"{p_never_up:.0f}%",
        "buy_now": f"+{mean_buy:.0f}%",
        "waiter": f"+{mean_wait:.0f}%",
        "wait_wins": f"{p_wait_wins:.0f}%",
        "days": f"{med_days:.0f}",
        "dip": f"{D:.0f}%",
        "drop": f"{abs(D):.0f}%",
        "first_year": str(d["dates"][0].year),
        "last_year": str(d["dates"][-1].year + 1),
        "window": str(CONFIG["WINDOW"]),
        "crash": "2008",                           # asserted above: waiting paid
    }
    with open(os.path.join(BASE_DIR, "figures.json"), "w") as fh:
        json.dump(figs, fh, indent=1)
    log(f"[validate] {figs}")

    # running counter for the build: share of windows still waiting for their
    # dip on trading day k, from the same running minimum
    run = (d["runmin"] > D).mean(axis=0) * 100.0
    d["run"] = [f"{x:.0f}%" for x in run]
    assert d["run"][-1] == figs["never"], "running counter disagrees with figures"
    d["figs"] = figs
    return figs


def prepare_geometry(d):
    c = CONFIG
    sel = np.arange(0, len(d["dip"]), c["DRAW_EVERY"])
    P = np.clip(d["paths"][sel], c["RET_LO"], c["RET_HI"])
    m, N1 = P.shape
    R = 1.0 + c["R_SCALE"] * P / 100.0
    theta = np.linspace(0, 2 * np.pi, m, endpoint=False)[:, None]
    d["X"], d["Y"] = R * np.cos(theta), R * np.sin(theta)
    d["Z"] = np.linspace(0, 1, N1)
    d["theta"] = theta[:, 0]
    dip, end, touch = d["dip"][sel], d["end"][sel], d["touch"][sel]
    cols = np.empty((m, 4))
    g = np.clip(end / c["GAIN_TOP"], 0, 1)
    for i in range(m):
        if dip[i]:
            cols[i] = to_rgba(THEME["BLUE"], 0.6)
        else:
            cols[i] = GAIN_CMAP(g[i])[:3] + (0.8,)
    d["cols"] = cols
    d["order"] = np.argsort(~dip)                   # no-dip paths drawn last
    d["lw"] = np.where(dip, 0.5, 0.75)
    d["sel_dip"], d["sel_touch"] = dip, touch
    d["r_dip"] = 1.0 + c["R_SCALE"] * c["DIP"] / 100.0
    d["r_max"] = 1.0 + c["R_SCALE"] * c["RET_HI"] / 100.0
    return d


# -------------------------------------------------------------- RENDER
def render_frame(idx, total, d, out_path):
    c, figs, L = CONFIG, d["figs"], LAYOUT
    if "X" not in d:
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
    N = c["WINDOW"]
    k = max(int(round(N * (0.12 + 0.88 * frac))), 2)          # trading day reached

    fig = plt.figure(figsize=(c["W"] / c["DPI"], c["H"] / c["DPI"]),
                     dpi=c["DPI"], facecolor=THEME["BG"])
    fig.text(0.5, L["TITLE"], "THE DIP THAT NEVER CAME", ha="center", fontsize=27,
             fontweight="bold", color=THEME["TEXT"], family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             f"SPY  |  {figs['windows']} one-year windows  |  "
             f"{figs['first_year']} to {figs['last_year']}  |  buy at {figs['dip']}",
             ha="center", fontsize=13, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.5, L["KEY"],
             "blue: the dip came   green to red: it never came, colour = gain missed",
             ha="center", fontsize=10, color=THEME["TEXT_DIM"], family=THEME["FONT"])
    fig.text(0.5, L["ROW_A"],
             f"dip came {figs['came']}   waiter avg {figs['waiter']}   "
             f"buy-now avg {figs['buy_now']}",
             ha="center", va="center", fontsize=9, color=THEME["TEXT"],
             family=THEME["MONO"])
    fig.text(0.5, L["ROW_B"],
             f"no-dip years a later {figs['dip']} still couldn't undo: "
             f"{figs['out_of_reach']}",
             ha="center", va="center", fontsize=9, color=THEME["YELLOW"],
             family=THEME["MONO"])
    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])

    ax = fig.add_axes(L["AXES"], projection="3d", facecolor=THEME["BG"],
                      computed_zorder=False)

    # the barrier tube at -10%: translucent wall, rings and ribs
    rd = d["r_dip"]
    th = np.linspace(0, 2 * np.pi, 97)
    zz = np.linspace(0, 1, 2)
    TH, ZZ = np.meshgrid(th, zz)
    ax.plot_surface(rd * np.cos(TH), rd * np.sin(TH), ZZ, color=THEME["RED"],
                    alpha=0.07, linewidth=0, shade=False, zorder=5)
    for z in np.linspace(0, 1, 11):
        ax.plot(rd * np.cos(th), rd * np.sin(th), np.full_like(th, z),
                color=THEME["RED"], lw=0.8, alpha=0.6, zorder=5)
    for a in np.linspace(0, 2 * np.pi, 16, endpoint=False):
        ax.plot([rd * np.cos(a)] * 2, [rd * np.sin(a)] * 2, [0, 1],
                color=THEME["RED"], lw=0.5, alpha=0.35, zorder=5)
    ax.text(0, 0, 0.985, f"{figs['dip']} tube", ha="center", va="bottom",
            fontsize=12, color=THEME["RED"], fontweight="bold",
            family=THEME["FONT"], zorder=6,
            path_effects=[pe.withStroke(linewidth=3, foreground=THEME["BG"])])
    # entry ring: where every year starts
    ax.plot(np.cos(th), np.sin(th), np.zeros_like(th), color=THEME["TEXT"],
            lw=0.9, alpha=0.7, zorder=2)

    X, Y, Z = d["X"], d["Y"], d["Z"]
    o = d["order"]
    segs = [np.column_stack([X[i, :k + 1], Y[i, :k + 1], Z[:k + 1]]) for i in o]
    ax.add_collection3d(Line3DCollection(segs, colors=d["cols"][o],
                                         linewidths=d["lw"][o], zorder=3))
    # the fills: where a dip path first pierced the tube
    tch = d["sel_touch"]
    got = np.nonzero(d["sel_dip"] & (tch <= k))[0]
    if len(got):
        ax.scatter(X[got, tch[got]], Y[got, tch[got]], Z[tch[got]], s=3,
                   color=THEME["TEXT"], alpha=0.85, depthshade=False, zorder=4)

    lim = d["r_max"] * 0.92
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(0, 1)
    ax.view_init(elev=c["ELEV_BASE"] + 5 * np.sin(t * np.pi),
                 azim=c["AZIM_START"] + c["AZIM_SWEEP"] * t)
    ax.set_box_aspect((1.0, 1.0, 1.2), zoom=c["ZOOM"])
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_axis_off()

    # Readout: strings only, precomputed and asserted in validate().
    if frac < 1.0:
        fig.text(0.5, L["NOTE"],
                 f"day {k:>3} of {figs['window']}   still waiting for {figs['dip']}: "
                 f"{d['run'][k]}",
                 ha="center", fontsize=11, color=THEME["TEXT"], family=THEME["MONO"])
    else:
        fig.text(0.5, L["NOTE"],
                 "one line per year held: day up, start date around, return outward",
                 ha="center", fontsize=10, color=THEME["TEXT_DIM"],
                 family=THEME["FONT"])
    fig.text(0.5, L["BIG"], f"{figs['never']} OF YEARS: NO DIP",
             ha="center", fontsize=21, color=THEME["ORANGE"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             f"median gain missed while waiting in cash: {figs['missed']}",
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
