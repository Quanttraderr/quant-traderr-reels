"""
LumpSum_Reel_Pipeline.py
========================
Reel 15 for @quant.traderr. Every start day of SPY since 1993: put a sum of
money in all at once, or drip it in as 12 equal monthly buys. Who is ahead at
the end of month 12?

THE MATHS
    Month = 21 trading days, year = 12 months = 252 trading days, prices are
    adjusted closes (dividends included). For each start day s:

        lump(s) = P(s + 252) / P(s)
        dca(s)  = mean over j = 0..11 of  P(s + 252) / P(s + 21 j)

    Both start with the same 1.0 of capital and are compared at s + 252, when
    the last drip buy (day 231) has been in the market for one month. Money
    still waiting to be dripped earns nothing in the headline number. A second
    figure pays that waiting cash the 13-week T-bill rate (^IRX, accrued
    daily as rate / 252), so the headline cannot hide behind a zero cash rate.

    In 3D: trading day inside the year runs along the floor, start date runs
    into depth, value in percent of the starting sum goes up. Each start date
    carries a glass wall, the lump sum, fully invested from day one. Inside it
    stands the staircase of the drip: the market value of what has been bought
    so far, rising one step every 21 days. The empty glass above the stairs is
    money sitting out of the market. The colour of the stairs is the verdict
    at month 12: orange where the lump sum finished ahead, green where the
    drip did.

THE CLAIM THE VISUAL MAKES
    Over one-year horizons on the S&P 500 the lump sum finished ahead on most
    start days, because the market rose in most months and the drip spends
    most of the year partly in cash.

WHAT THE VISUAL IS NOT
    Not a forecast and not advice for any single start date: the drip wins
    exactly in the crash years, by a lot. The start dates overlap heavily, so
    the 8,000 or so comparisons are not independent trials: this is about 33
    years of one index that went up. No taxes, no costs, no behaviour. Only
    every n-th start date is drawn; every figure uses all of them.

RUN
    python LumpSum_Reel_Pipeline.py --smoke   # 3 frames -> temp_frames_smoke/
    python LumpSum_Reel_Pipeline.py           # full render -> topic.mp4

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
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "_cache")

CONFIG = {
    "TICKERS": ["SPY", "^IRX"],
    "PERIOD": "max",
    "MONTHS": 12,             # number of drip buys
    "STEP": 21,               # trading days between drip buys
    "DRAW_EVERY": 21,         # draw every n-th start date, figures use all
    "LEAD_CLIP": 10.0,        # colour saturates at this lead, points of the sum

    # render
    "W": 1080, "H": 1920, "DPI": 100,
    "FPS": 30,
    "HOOK_SEC": 1.4, "BUILD_SEC": 8.5, "HOLD_SEC": 1.8,
    "ELEV_BASE": 22, "AZIM_START": -122, "AZIM_SWEEP": 14,
    "ZOOM": 0.97,
}

THEME = {
    "BG": "#000000", "TEXT": "#ffffff", "TEXT_DIM": "#8a8a8a",
    "ORANGE": "#ff9500", "CYAN": "#00f2ff", "YELLOW": "#ffd400",
    "RED": "#ff3050", "GREEN": "#00ff8c", "BLUE": "#0066ff",
    "FONT": "DejaVu Sans", "MONO": "DejaVu Sans Mono",
}
DIV = LinearSegmentedColormap.from_list(
    "verdict", ["#00ff8c", "#1c2b40", "#ff9500"], N=256)
CMAP = LinearSegmentedColormap.from_list(
    "house", [THEME["BLUE"], THEME["CYAN"], THEME["GREEN"],
              THEME["YELLOW"], THEME["ORANGE"], THEME["RED"]], N=256)

SAFE = {"TOP": 0.12, "BOTTOM": 0.20, "RIGHT": 0.14}

LAYOUT = {
    "TITLE": 0.862, "SUBTITLE": 0.838, "KEY": 0.816, "ROW_A": 0.799,
    "ROW_B": 0.783, "HANDLE": 0.764,
    "AXES": [-0.08, 0.29, 1.16, 0.47],
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
    if px.empty or any(t not in px.columns for t in CONFIG["TICKERS"]):
        raise SystemExit("no data returned; refusing to draw a synthetic tape")
    px = px[CONFIG["TICKERS"]]
    if px["SPY"].dropna().empty or px["^IRX"].dropna().empty:
        raise SystemExit("SPY or ^IRX came back empty; refusing to continue")
    px.to_csv(path)
    return px


# ------------------------------------------------------------- COMPUTE
def compute(px):
    c = CONFIG
    s = px["SPY"].dropna()
    v = s.values
    irx = px["^IRX"].reindex(s.index).ffill().values / 100.0
    if np.isnan(irx).any():
        raise SystemExit("T-bill series has holes before its first value")
    M, st = c["MONTHS"], c["STEP"]
    N = M * st                                              # 252
    starts = np.arange(len(v) - N)
    days = np.arange(N + 1)
    idx = starts[:, None] + days[None, :]                   # (S, N+1)
    rel = v[idx] / v[starts][:, None]                       # P(s+k)/P(s)
    buys = starts[:, None] + st * np.arange(M)[None, :]     # (S, M)

    # drip: invested market value at day k = sum over bought tranches of
    # (1/M) * P(s+k) / P(s + 21 j)
    inv = np.zeros_like(rel)
    for j in range(M):
        k0 = st * j
        inv[:, k0:] += (v[idx[:, k0:]] / v[buys[:, j]][:, None]) / M
    n_bought = np.minimum(days // st + 1, M)                # tranches in by day k
    cash = (M - n_bought) / M
    dca_wealth = inv + cash[None, :]                        # zero cash rate

    lump = rel[:, -1]
    dca = dca_wealth[:, -1]
    diff = (lump - dca) * 100.0                             # points of capital

    # the same comparison with the waiting cash earning the 13-week T-bill
    cum = np.concatenate([[0.0], np.cumsum(np.log1p(irx / 252.0))])
    grow = np.exp(cum[buys] - cum[starts][:, None])         # (S, M)
    end = v[starts + N]
    dca_tb = (grow * end[:, None] / v[buys]).mean(axis=1)
    diff_tb = (lump - dca_tb) * 100.0

    return {"rel": rel * 100.0, "inv": inv * 100.0, "lead": (rel - dca_wealth),
            "lump": lump, "dca": dca, "diff": diff, "diff_tb": diff_tb,
            "dates": s.index[starts], "N": N}


# ------------------------------------------------------------ VALIDATE
def validate(d):
    """Every number that reaches a frame is asserted here, before any render."""
    diff, diff_tb, dates = d["diff"], d["diff_tb"], d["dates"]
    n = len(diff)
    p_lump = float((diff > 0).mean() * 100)
    p_lump_tb = float((diff_tb > 0).mean() * 100)
    med = float(np.median(diff))
    med_lose = float(np.median(-diff[diff < 0]))            # drip's typical win
    iw = int(np.argmin(diff))
    worst = float(-diff[iw])                                # drip's best start
    worst_date = dates[iw]

    # sanity of the maths itself: the last drip buy is day 231, so on day 252
    # all cash must be invested and the zero-rate drip must equal the direct sum
    assert d["N"] == 252, d["N"]
    assert np.allclose(d["inv"][:, -1] / 100.0, d["dca"]), "staircase top != dca"

    assert n > 6000, f"only {n} start days"
    assert 50.0 <= p_lump <= 95.0, f"lump sum won {p_lump:.0f}%"
    assert 50.0 <= p_lump_tb <= p_lump, f"with T-bills {p_lump_tb:.0f}%"
    assert 0.0 < med < 20.0, f"median lead {med:.1f}"
    assert 0.0 < med_lose < 20.0, f"median drip win {med_lose:.1f}"
    assert 10.0 < worst < 60.0, f"drip's best case {worst:.1f}"
    assert worst_date.year in (2007, 2008), f"drip's best start {worst_date}"

    figs = {
        "starts": f"{n:,}",
        "lump_win": f"{p_lump:.0f}%",
        "dca_win": f"{100 - p_lump:.0f}%",
        "lump_win_tb": f"{p_lump_tb:.0f}%",
        "median_lead": f"+{med:.1f}%",
        "dca_median_win": f"{med_lose:.1f}%",
        "dca_best": f"{worst:.0f}%",
        "dca_best_start": worst_date.strftime("%b %Y"),
        "crash": str(worst_date.year),
        "first_year": str(dates[0].year),
        "last_year": str(dates[-1].year + 1),
        "months": str(CONFIG["MONTHS"]),
        "step": str(CONFIG["STEP"]),
        "horizon": str(d["N"]),
    }
    with open(os.path.join(BASE_DIR, "figures.json"), "w") as fh:
        json.dump(figs, fh, indent=1)
    log(f"[validate] {figs}")

    # running counter for the build: share of start dates where the lump sum is
    # ahead of the drip (zero cash rate) on trading day k of the year
    run = (d["lead"] > 0).mean(axis=0) * 100.0
    d["run"] = [f"{x:.0f}%" for x in run]
    assert d["run"][-1] == figs["lump_win"], "running counter disagrees with figures"
    d["figs"] = figs
    return figs


def prepare_geometry(d):
    c = CONFIG
    M, st = c["MONTHS"], c["STEP"]
    sel = np.arange(0, len(d["diff"]), c["DRAW_EVERY"])
    m = len(sel)
    d["lead_sel"] = d["lead"][sel] * 100.0            # points of the sum, (m, N+1)
    d["y0"] = np.arange(m) / m
    d["dy"] = 1.0 / m
    d["xs"] = np.linspace(0, 1, d["N"] + 1)
    d["h"] = 100.0 * np.arange(M + 1) / M              # invested share after j buys
    # where a few years sit along the depth axis, for the labels
    ds = d["dates"][sel]
    labs = [d["figs"]["first_year"], "2000", d["figs"]["crash"], "2020"]
    d["year_y"] = [(y, float(d["y0"][np.searchsorted(ds.year, int(y))]) + d["dy"] / 2)
                   for y in labs]
    return d


def lead_color(x):
    """Diverging: green where the drip is ahead, orange where the lump sum is."""
    return DIV(np.clip(0.5 + np.asarray(x) / (2 * CONFIG["LEAD_CLIP"]), 0, 1))


# -------------------------------------------------------------- RENDER
def render_frame(idx, total, d, out_path):
    c, figs, L = CONFIG, d["figs"], LAYOUT
    if "xs" not in d:
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
    N = d["N"]
    k = max(int(round(N * (0.10 + 0.90 * frac))), 2)          # trading day reached

    fig = plt.figure(figsize=(c["W"] / c["DPI"], c["H"] / c["DPI"]),
                     dpi=c["DPI"], facecolor=THEME["BG"])
    fig.text(0.5, L["TITLE"], "LUMP SUM BEATS THE DRIP", ha="center", fontsize=28,
             fontweight="bold", color=THEME["TEXT"], family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             f"SPY  |  {figs['starts']} start days  |  "
             f"{figs['first_year']} to {figs['last_year']}  |  "
             f"all at once vs {figs['months']} monthly buys",
             ha="center", fontsize=12, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.5, L["KEY"],
             "glass block: lump sum   stairs: 12 buys   orange: lump ahead   green: drip ahead",
             ha="center", fontsize=10, color=THEME["TEXT_DIM"], family=THEME["FONT"])
    fig.text(0.5, L["ROW_A"],
             f"waiting cash paid T-bill interest: lump still wins {figs['lump_win_tb']}",
             ha="center", va="center", fontsize=9, color=THEME["TEXT"],
             family=THEME["MONO"])
    fig.text(0.5, L["ROW_B"],
             f"drip's best start: {figs['dca_best_start']}, ahead by {figs['dca_best']}",
             ha="center", va="center", fontsize=9, color=THEME["GREEN"],
             family=THEME["MONO"])
    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])

    ax = fig.add_axes(L["AXES"], projection="3d", facecolor=THEME["BG"],
                      computed_zorder=False)

    M, st, xs, y0, dy, h = c["MONTHS"], c["STEP"], d["xs"], d["y0"], d["dy"], d["h"]
    L_ = d["lead_sel"]
    # the drip: one staircase per start date, extruded into depth. A tread is
    # a month; its height is the share invested, its colour is who is ahead at
    # the end of that month (or today, while the month is still running).
    polys, cols = [], []
    for j in range(M):
        a0 = st * j
        if k <= a0:
            break
        a1 = min(st * (j + 1), k)
        xa, xb = xs[a0], xs[a1]
        col = lead_color(L_[:, a1])
        z0, z1 = h[j], h[j + 1]
        for i in range(len(y0)):
            ya, yb = y0[i], y0[i] + dy
            polys.append([(xa, ya, z1), (xb, ya, z1), (xb, yb, z1), (xa, yb, z1)])
            polys.append([(xa, ya, z0), (xa, ya, z1), (xa, yb, z1), (xa, yb, z0)])
            cols += [col[i], col[i] * np.array([0.55, 0.55, 0.55, 1.0])]
    ax.add_collection3d(Poly3DCollection(polys, facecolors=cols, edgecolors="none",
                                         zorder=2))
    # the lump sum: one block, fully invested from day one
    X, Y, Z = (0, 1), (0, 1), (0, 100)
    faces = [[(X[0], Y[0], Z[1]), (X[1], Y[0], Z[1]), (X[1], Y[1], Z[1]), (X[0], Y[1], Z[1])],
             [(X[0], Y[0], Z[0]), (X[1], Y[0], Z[0]), (X[1], Y[0], Z[1]), (X[0], Y[0], Z[1])],
             [(X[0], Y[1], Z[0]), (X[1], Y[1], Z[0]), (X[1], Y[1], Z[1]), (X[0], Y[1], Z[1])],
             [(X[0], Y[0], Z[0]), (X[0], Y[1], Z[0]), (X[0], Y[1], Z[1]), (X[0], Y[0], Z[1])],
             [(X[1], Y[0], Z[0]), (X[1], Y[1], Z[0]), (X[1], Y[1], Z[1]), (X[1], Y[0], Z[1])]]
    ax.add_collection3d(Poly3DCollection(faces, facecolors=to_rgba(THEME["CYAN"], 0.035),
                                         edgecolors=to_rgba(THEME["CYAN"], 0.55),
                                         linewidths=0.8, zorder=4))
    for y, yy in d["year_y"]:
        ax.text(0.0, yy, 0, f"{y}  ", color=THEME["TEXT_DIM"], fontsize=9,
                family=THEME["MONO"], ha="right", va="center", zorder=5)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 104)
    ax.view_init(elev=c["ELEV_BASE"] + 5 * np.sin(t * np.pi),
                 azim=c["AZIM_START"] + c["AZIM_SWEEP"] * t)
    ax.set_box_aspect((1.5, 1.35, 1.0), zoom=c["ZOOM"])
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    # Readout: strings only, precomputed and asserted in validate().
    if frac < 1.0:
        fig.text(0.5, L["NOTE"],
                 f"day {k:>3} of {figs['horizon']}   lump sum ahead so far: "
                 f"{d['run'][k]:>4}",
                 ha="center", fontsize=11, color=THEME["TEXT"], family=THEME["MONO"])
    else:
        fig.text(0.5, L["NOTE"],
                 "one staircase per start date: month along, start date deep",
                 ha="center", fontsize=10, color=THEME["TEXT_DIM"],
                 family=THEME["FONT"])
    fig.text(0.5, L["BIG"], f"LUMP SUM WON {figs['lump_win']} OF START DAYS",
             ha="center", fontsize=19, color=THEME["ORANGE"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             f"median lead after {figs['months']} months: {figs['median_lead']} of the sum",
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
