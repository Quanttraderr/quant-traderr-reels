"""
VolPremium_Reel_Pipeline.py
===========================
Reel 12 for @quant.traderr. Fear priced against fear delivered: every trading
day since 1993, the VIX set against the volatility the S&P actually produced
over the month that followed.

THE MATHS
    The VIX is the market's price for the next 30 calendar days of S&P
    volatility, annualised, in volatility points. Realised volatility is what
    then happened: the standard deviation of SPY's daily log returns over the
    next 21 trading days, times sqrt(252), times 100, so both sit in the same
    unit. The spread VIX minus realised is the volatility risk premium: what
    buyers of protection paid above what the market delivered.

    Each day is one point in 3D: time along the floor, the gap VIX minus
    realised in depth, the VIX level up. The glass wall stands at gap zero,
    VIX equal to realised. Points in front of it are days the insurance was
    overpriced (green), points behind it days it was too cheap (red).

THE CLAIM THE VISUAL MAKES
    Most of the cloud sits on one side of the wall, and has done so in every
    decade, not only in calm years.

WHAT THE VISUAL IS NOT
    Not a trading strategy. Selling volatility collects the green days slowly
    and gives it back on the red ones in a handful of sessions, and the red
    days are exactly the crashes. The VIX is also not an exact SPY forecast: it
    is priced off S&P 500 index options over 30 calendar days, and we compare
    it with close to close SPY volatility over 21 trading days, which is the
    standard proxy, not an identity. Points are capped at 70 on the display
    axes only (VIX at 55, the gap at 30 points either way, a few days in 2008
    and 2020); every figure uses the raw values.

RUN
    python VolPremium_Reel_Pipeline.py --smoke   # 3 frames -> temp_frames_smoke/
    python VolPremium_Reel_Pipeline.py           # full render -> topic.mp4

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
    "TICKERS": ["SPY", "^VIX"],
    "PERIOD": "max",
    "HORIZON": 21,            # trading days of realised vol after each VIX print
    "CAP": 55.0,              # display cap on the VIX axis, figures use raw data
    "GAP_CAP": 30.0,          # display cap on the gap axis, vol points
    "TRAIL": 45,              # days in the bright comet head

    # render
    "W": 1080, "H": 1920, "DPI": 100,
    "FPS": 30,
    "HOOK_SEC": 1.4, "BUILD_SEC": 8.5, "HOLD_SEC": 1.8,
    "ELEV_BASE": 20, "AZIM_START": -32, "AZIM_SWEEP": 18,
    "ZOOM": 1.24,
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
    px = px[CONFIG["TICKERS"]]
    if px.empty:
        raise SystemExit("no data returned; refusing to draw a synthetic tape")
    px.to_csv(path)
    return px


# ------------------------------------------------------------- COMPUTE
def compute(px):
    # >>> REPLACE: the maths for your topic. Return whatever render_frame needs.
    px = px.dropna()
    h = CONFIG["HORIZON"]
    r = np.log(px["SPY"]).diff()
    # realised vol of the NEXT h sessions, aligned to the day the VIX printed
    rv = r.rolling(h).std().shift(-h) * np.sqrt(252) * 100.0
    df = pd.DataFrame({"vix": px["^VIX"], "rv": rv}).dropna()
    df["spread"] = df["vix"] - df["rv"]
    return {"df": df}


# ------------------------------------------------------------ VALIDATE
def validate(d):
    """Every number that reaches a frame is asserted here, before any render."""
    df = d["df"]
    over = df["spread"] > 0
    share = float(over.mean() * 100.0)
    med = float(df["spread"].median())
    vix_mean, rv_mean = float(df["vix"].mean()), float(df["rv"].mean())
    worst_i = df["spread"].idxmin()
    worst = float(df["spread"].min())

    # share per decade, to back the "every decade" claim in the docstring
    dec = over.groupby((df.index.year // 10) * 10).mean() * 100.0

    assert len(df) > 6000, f"only {len(df)} days"
    assert 55.0 <= share <= 97.0, f"VIX above realised on {share:.0f}% of days"
    assert 0.0 < med < 15.0, f"median premium {med:.1f} out of band"
    assert 10.0 <= vix_mean <= 35.0, f"mean VIX {vix_mean:.1f}"
    assert 5.0 <= rv_mean <= 35.0, f"mean realised {rv_mean:.1f}"
    assert vix_mean > rv_mean, "the premium collapsed: check alignment"
    assert worst < -10.0, f"worst miss {worst:.1f}, expected a crash-sized miss"
    assert (dec > 50.0).all(), f"a decade below half: {dec.round(0).to_dict()}"
    # "the other share held the crashes": the run-ups to the two deepest
    # selloffs of the sample must be mostly red, or the line comes off
    for a, b in (("2008-09-01", "2008-10-31"), ("2020-02-15", "2020-03-15")):
        red = float((~over[a:b]).mean())
        assert red > 0.5, f"{a}..{b} only {red:.0%} red; crash claim fails"

    figs = {
        "share": f"{share:.0f}%",
        "miss": f"{100 - share:.0f}%",
        "median": f"{med:.1f}",
        "vix_mean": f"{vix_mean:.1f}",
        "rv_mean": f"{rv_mean:.1f}",
        "worst": f"{worst:.0f}",
        "worst_when": worst_i.strftime("%b %Y"),
        "days": f"{len(df):,}",
        "first_year": str(df.index[0].year),
        "last_year": str(df.index[-1].year),
        "dec_min": f"{int(np.floor(dec.min()))}%",   # floor: caption says "at least"
        "crash_a": "2008", "crash_b": "2020",      # the two windows asserted red above
        "horizon": str(CONFIG["HORIZON"]),
    }
    with open(os.path.join(BASE_DIR, "figures.json"), "w") as fh:
        json.dump(figs, fh, indent=1)
    log(f"[validate] {figs}")

    # running counter shown during the build, from the same boolean series;
    # its last value must be the asserted headline or the build stops
    run = (np.cumsum(over.values) / np.arange(1, len(over) + 1)) * 100.0
    d["run"] = [f"{v:.0f}%" for v in run]
    assert d["run"][-1] == figs["share"], "running counter disagrees with figures"
    d["years"] = [str(y) for y in df.index.year]
    d["figs"] = figs
    return figs


def prepare_geometry(d):
    df, c = d["df"], CONFIG
    n = len(df)
    d["x"] = np.linspace(0.0, 1.0, n)                     # time along the floor
    # depth is the gap itself, so the fair line becomes a wall at zero and
    # the question "which side" is answered by position, not by squinting
    d["y"] = np.clip(df["spread"].values, -c["GAP_CAP"], c["GAP_CAP"]) / c["GAP_CAP"]
    d["z"] = np.clip(df["vix"].values, 0, c["CAP"]) / c["CAP"]
    sp = df["spread"].values
    # colour is the spread itself: green where insurance was overpriced,
    # red where it was too cheap, brighter the bigger the gap
    mag = np.clip(np.abs(sp) / 15.0, 0.25, 1.0)
    base = np.where(sp[:, None] > 0,
                    np.array(matplotlib.colors.to_rgb(THEME["GREEN"]))[None],
                    np.array(matplotlib.colors.to_rgb(THEME["RED"]))[None])
    d["rgba"] = np.column_stack([base, 0.18 + 0.62 * mag])
    d["size"] = np.where(sp > 0, 2.2, 5.0)                # misses drawn bigger
    return d


# -------------------------------------------------------------- RENDER
def render_frame(idx, total, d, out_path):
    c, figs, L = CONFIG, d["figs"], LAYOUT
    if "x" not in d:
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
    n = len(d["x"])
    k = max(int(round(n * (0.10 + 0.90 * frac))), 2)

    fig = plt.figure(figsize=(c["W"] / c["DPI"], c["H"] / c["DPI"]),
                     dpi=c["DPI"], facecolor=THEME["BG"])
    fig.text(0.5, L["TITLE"], "FEAR IS OVERPRICED", ha="center", fontsize=28,
             fontweight="bold", color=THEME["TEXT"], family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             f"VIX vs the next {figs['horizon']} days of SPY  |  "
             f"{figs['first_year']} to {figs['last_year']}  |  {figs['days']} days",
             ha="center", fontsize=13, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.5, L["KEY"],
             "green: VIX above what SPY then did   red: below   glass: fair",
             ha="center", fontsize=10, color=THEME["TEXT_DIM"], family=THEME["FONT"])
    fig.text(0.5, L["ROW_A"],
             f"avg VIX {figs['vix_mean']}   avg realised {figs['rv_mean']}"
             f"   median gap +{figs['median']}",
             ha="center", va="center", fontsize=9, color=THEME["TEXT"],
             family=THEME["MONO"])
    fig.text(0.5, L["ROW_B"],
             f"worst miss {figs['worst']} vol pts  ({figs['worst_when']})",
             ha="center", va="center", fontsize=9, color=THEME["RED"],
             family=THEME["MONO"])
    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])

    ax = fig.add_axes(L["AXES"], projection="3d", facecolor=THEME["BG"],
                      computed_zorder=False)

    # >>> REPLACE: the geometry.
    # the fair wall: VIX == realised, a glass sheet at gap zero
    xs, zs = np.meshgrid(np.linspace(0, 1, 2), np.linspace(0, 1, 2))
    ax.plot_surface(xs, np.zeros_like(xs), zs, color=THEME["CYAN"], alpha=0.08,
                    linewidth=0, shade=False, zorder=1)
    for zg in np.linspace(0, 1, 8):
        ax.plot([0, 1], [0, 0], [zg, zg], color=THEME["CYAN"], lw=0.4,
                alpha=0.28, zorder=1)
    for xg in np.linspace(0, 1, 14):
        ax.plot([xg, xg], [0, 0], [0, 1], color=THEME["CYAN"], lw=0.4,
                alpha=0.28, zorder=1)
    x, y, z = d["x"][:k], d["y"][:k], d["z"][:k]
    ax.scatter(x, y, z, c=d["rgba"][:k], s=d["size"][:k], depthshade=False,
               linewidths=0, zorder=3)
    # comet head: the last TRAIL days as a bright line tracing VIX
    h0 = max(k - c["TRAIL"], 0)
    if frac < 1.0:
        ax.plot(x[h0:], y[h0:], z[h0:], color=THEME["YELLOW"], lw=1.3,
                alpha=0.9, zorder=4)
        ax.scatter([x[-1]], [y[-1]], [z[-1]], color=THEME["YELLOW"], s=60,
                   depthshade=False, zorder=5)

    ax.set_xlim(0, 1); ax.set_ylim(-1, 1); ax.set_zlim(0, 1)
    ax.view_init(elev=c["ELEV_BASE"] + 6 * np.sin(t * np.pi),
                 azim=c["AZIM_START"] + c["AZIM_SWEEP"] * t)
    ax.set_box_aspect((1.7, 1.2, 1.15), zoom=c["ZOOM"])
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    # Readout: strings only, precomputed and asserted in validate().
    if frac < 1.0:
        fig.text(0.5, L["NOTE"],
                 f"{d['years'][k - 1]}   VIX above realised so far: {d['run'][k - 1]}",
                 ha="center", fontsize=11, color=THEME["YELLOW"],
                 family=THEME["MONO"])
    else:
        fig.text(0.5, L["NOTE"],
                 "each dot is one day: height is the VIX, depth is the gap",
                 ha="center", fontsize=10, color=THEME["TEXT_DIM"],
                 family=THEME["FONT"])
    big = figs["share"]                   # the headline is the asserted figure only
    fig.text(0.5, L["BIG"], f"{big} OF DAYS FEAR WAS OVERPRICED",
             ha="center", fontsize=19, color=THEME["GREEN"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             f"the other {figs['miss']} held {figs['crash_a']} and {figs['crash_b']}",
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
