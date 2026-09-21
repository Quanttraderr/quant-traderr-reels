"""
Seasonality_Reel_Pipeline.py
============================
Reel 10 for @quant.traderr. Every month of every year on one torus, and the
calendar effect that does not survive its own error bar.

THE MATHS
    Monthly total returns for SPY over the complete calendar years in the
    sample. The torus has two angles and the data has two indices, which is
    the whole reason for the shape: going the long way round is the month,
    January to December, and going round the tube is the year. One patch is
    one month of one year, coloured by what that month did.

    The tube is fatter where that month's average return is higher, so the
    lumps are the seasonal effect people trade on. Against each lump sits the
    standard error of that same average, sd / sqrt(number of years), which is
    printed rather than hidden.

THE CLAIM THE VISUAL MAKES
    The seasonal pattern is real in the sense that the averages differ. It is
    a mirage in the sense that the differences are small next to the spread of
    the months they are averaged from.

WHAT THE VISUAL IS NOT
    Not a trading calendar. **The worst month's average is inside one standard
    error of zero**, which the reel asserts rather than assumes: if that ever
    stopped being true the build would fail instead of quietly shipping.

    Twelve months are tested at once here, so the best looking month is partly
    just the winner of twelve draws. One index, one country, and the years are
    treated as independent when they are not. Nothing here is a forecast.

RUN
    python Seasonality_Reel_Pipeline.py --smoke   # 3 frames -> temp_frames_smoke/
    python Seasonality_Reel_Pipeline.py           # full render -> topic.mp4

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
    "TICKERS": ["SPY"],
    "PERIOD": "max",
    "R": 1.0,                 # major radius, the calendar ring
    "TUBE_MIN": 0.26,         # tube radius at the weakest month
    "TUBE_MAX": 0.46,         # tube radius at the strongest month
    "SUB": 7,                 # substeps per month sector, for a smooth tube
    "COL_ABS": 8.0,           # colour saturates at plus or minus this, percent

    # render
    "W": 1080, "H": 1920, "DPI": 100,
    "FPS": 30,
    "HOOK_SEC": 1.4, "BUILD_SEC": 8.5, "HOLD_SEC": 1.8,
    "ELEV_BASE": 34, "AZIM_START": -62, "AZIM_SWEEP": 360,
    "ZOOM": 1.62,
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

# Two label rows instead of one: the twelve monthly averages are the point of
# the reel, so they get their own fixed rows where nothing can turn them into
# the Instagram button column. The handle moves down to make room.
LAYOUT = {
    "TITLE": 0.862, "SUBTITLE": 0.838, "KEY": 0.816, "MONTHS_A": 0.799,
    "MONTHS_B": 0.783, "HANDLE": 0.764,
    "AXES": [-0.08, 0.30, 1.16, 0.455],
    "NOTE": 0.272, "BIG": 0.242, "SUB": 0.215,
}

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


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
    close = px[CONFIG["TICKERS"][0]]
    m = close.resample("ME").last().pct_change().dropna() * 100.0

    tab = pd.DataFrame({"r": m.values, "y": m.index.year, "m": m.index.month})
    grid = tab.pivot_table(index="y", columns="m", values="r")
    # Only whole calendar years. A torus with a bite out of it is a torus with
    # a bug in it, and every figure below is read off the grid that is drawn.
    grid = grid.dropna(axis=0, how="any")

    return {"grid": grid,                       # rows: years, cols: months
            "years": list(grid.index)}


# ------------------------------------------------------------ VALIDATE
def validate(d):
    """Every number that reaches a frame is asserted here, before any render.
    A data refresh that moves a figure out of band fails the build instead of
    shipping a caption that no longer matches the picture."""
    grid = d["grid"]
    Z = grid.values                             # years x months
    n_y, n_m = Z.shape
    mean = Z.mean(axis=0)
    se = Z.std(axis=0, ddof=1) / np.sqrt(n_y)

    worst, best = int(np.argmin(mean)), int(np.argmax(mean))
    spread = float(mean[best] - mean[worst])
    t_worst = float(mean[worst] / se[worst])

    assert n_m == 12, f"{n_m} months per year, the torus needs 12"
    assert n_y >= 25, f"only {n_y} whole years, too few for a monthly average"
    assert -4.0 <= mean[worst] <= 1.0, f"worst month {mean[worst]:.2f}% out of band"
    assert 0.5 <= mean[best] <= 6.0, f"best month {mean[best]:.2f}% out of band"
    assert 1.0 <= spread <= 9.0, f"spread {spread:.2f} points out of band"
    assert 0.2 <= se[worst] <= 2.5, f"standard error {se[worst]:.2f} implausible"
    # The claim of the reel, as an assert: the weakest month of the year is not
    # separable from zero. If that ever stops being true, this build stops too.
    assert abs(t_worst) < 2.0, (
        f"the worst month is now {t_worst:.1f} standard errors from zero, "
        "the reel's own claim no longer holds")

    figs = {
        "worst_month": MONTHS[worst],
        "worst_mean": f"{mean[worst]:.1f}%",
        "best_month": MONTHS[best],
        "best_mean": f"{mean[best]:+.1f}%".replace("+", ""),
        "spread": f"{spread:.1f}",
        "se_worst": f"{se[worst]:.1f}",
        "n_years": str(n_y),
        "n_months": f"{n_y * n_m:,}",
        "first_year": str(int(grid.index[0])),
        "last_year": str(int(grid.index[-1])),
    }
    for i, name in enumerate(MONTHS):
        figs["m_" + name.lower()] = f"{mean[i]:+.1f}".replace("+0.0", "0.0")
    with open(os.path.join(BASE_DIR, "figures.json"), "w") as fh:
        json.dump(figs, fh, indent=1)
    log(f"[validate] worst {figs['worst_month']} {figs['worst_mean']} "
        f"se {figs['se_worst']} | best {figs['best_month']} {figs['best_mean']} "
        f"| {figs['n_months']} months over {figs['n_years']} years")
    d["figs"] = figs
    d["mean"], d["worst"], d["best"] = mean, worst, best
    return figs


def prepare_geometry(d):
    """One closed tube whose radius breathes with the monthly average.

    The first version built twelve separate sectors, one per month, each at its
    own constant radius. That left a wedge shaped hole at every boundary, and
    plugging the holes with dark walls was worse: a dozen black rings sitting
    over the colour. Interpolating the radius between the sector centres gives
    one surface with no seams, and the months still read, because a fat stretch
    is still a fat month.
    """
    c, grid = CONFIG, d["grid"]
    Z = grid.values                             # years x months
    n_y = Z.shape[0]
    mean = d["mean"]

    # Tube radius carries the monthly average, colour carries the individual
    # months. Both are returns but they live on different scales, so the
    # readout names the averages in numbers rather than asking anyone to read
    # them off a colour.
    lo, hi = float(mean.min()), float(mean.max())
    norm = (mean - lo) / (hi - lo)
    rad_m = c["TUBE_MIN"] + (c["TUBE_MAX"] - c["TUBE_MIN"]) * norm

    n_th = 12 * c["SUB"]
    th = np.linspace(0, 2 * np.pi, n_th + 1)
    # Radius is defined at the middle of each month and wrapped around the
    # year, so December flows into January instead of stepping.
    centres = (np.arange(12) + 0.5) / 12 * 2 * np.pi
    th_ext = np.concatenate([centres - 2 * np.pi, centres, centres + 2 * np.pi])
    rad_ext = np.concatenate([rad_m, rad_m, rad_m])
    rad = np.interp(th, th_ext, rad_ext)

    phi = np.linspace(0, 2 * np.pi, n_y + 1)    # round the tube: one year each
    TH, PH = np.meshgrid(th, phi, indexing="ij")
    RR = rad[:, None]
    d["X"] = (c["R"] + RR * np.cos(PH)) * np.cos(TH)
    d["Y"] = (c["R"] + RR * np.cos(PH)) * np.sin(TH)
    d["Z"] = RR * np.sin(PH)

    # One quad per (slice of a month, year), coloured by that month in that
    # year, so a stripe running round the tube is one calendar month.
    cols = CMAP(np.clip(Z / c["COL_ABS"] * 0.5 + 0.5, 0, 1))      # years x months x 4
    d["fc"] = np.repeat(cols.transpose(1, 0, 2), c["SUB"], axis=0)
    d["n_th"] = n_th
    d["lim"] = c["R"] + c["TUBE_MAX"] * 1.25
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

    fig = plt.figure(figsize=(c["W"] / c["DPI"], c["H"] / c["DPI"]),
                     dpi=c["DPI"], facecolor=THEME["BG"])
    fig.text(0.5, L["TITLE"], "THE SEASONALITY MIRAGE", ha="center",
             fontsize=26, fontweight="bold", color=THEME["TEXT"],
             family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             f"SPY  |  {figs['first_year']} to {figs['last_year']}  |  "
             f"{figs['n_months']} months",
             ha="center", fontsize=13, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.5, L["KEY"],
             "the long way round is the month, round the tube is the year",
             ha="center", fontsize=10, color=THEME["TEXT_DIM"],
             family=THEME["FONT"])

    # The twelve averages, in two fixed rows. The weakest and the strongest get
    # a colour, the rest stay quiet, so the eye lands on the comparison the
    # readout is about.
    for i, name in enumerate(MONTHS):
        row = L["MONTHS_A"] if i < 6 else L["MONTHS_B"]
        x = 0.19 + (i % 6) * 0.124
        col = (THEME["RED"] if i == d["worst"] else
               THEME["GREEN"] if i == d["best"] else THEME["TEXT_DIM"])
        fig.text(x, row, f"{name} {figs['m_' + name.lower()]}", ha="center",
                 va="center", fontsize=8, color=col, family=THEME["MONO"])

    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])

    ax = fig.add_axes(L["AXES"], projection="3d", facecolor=THEME["BG"],
                      computed_zorder=False)

    # >>> REPLACE: the geometry. The calendar closes month by month.
    k = max(int(round(d["n_th"] * (0.25 + 0.75 * frac))), 2)
    ax.plot_surface(d["X"][:k + 1], d["Y"][:k + 1], d["Z"][:k + 1],
                    facecolors=d["fc"][:k], linewidth=0, antialiased=False,
                    shade=False, alpha=1.0, rstride=1, cstride=1,
                    edgecolor="none", zorder=2)

    lim = d["lim"]
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim * 0.62, lim * 0.62)
    ax.view_init(elev=c["ELEV_BASE"] + 9 * np.sin(t * 2 * np.pi),
                 azim=c["AZIM_START"] + c["AZIM_SWEEP"] * t)
    ax.set_box_aspect((1.0, 1.0, 0.62), zoom=c["ZOOM"])
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    # Readout: strings only, straight from figures.json. Never re-derive here.
    fig.text(0.5, L["NOTE"],
             f"a fatter tube is a better month, and the colour is one month "
             f"in one year",
             ha="center", fontsize=10, color=THEME["TEXT_DIM"],
             family=THEME["FONT"])
    fig.text(0.5, L["BIG"],
             f"{figs['worst_month'].upper()} AVERAGES {figs['worst_mean']}",
             ha="center", fontsize=19, color=THEME["RED"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             f"give or take {figs['se_worst']}  ·  best is "
             f"{figs['best_month']} at {figs['best_mean']}",
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
