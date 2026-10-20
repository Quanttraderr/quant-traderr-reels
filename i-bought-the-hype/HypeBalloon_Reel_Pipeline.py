"""
HypeBalloon_Reel_Pipeline.py
============================
Reel 38 for @quant.traderr, "I Bought the Hype". The habit it takes apart:
piling into the fund everyone is talking about, right when everyone is
talking about it. $10,000 goes into ARK Innovation (ARKK) at its closing high
on 12 Feb 2021, and $10,000 into the S&P 500 (SPY) the same day.

THE DATA
    ARKK and SPY daily closes from Yahoo, auto-adjusted for splits and
    distributions, 12 Feb 2021 to the latest close, kept in _cache/. Real data
    throughout. Nothing is simulated.

THE MATHS
    value_t = 10,000 * Close_t / Close_12Feb2021      for each fund

    In 3D: two balloons released together at $10,000. Each drifts to the
    right with time and floats at the height of its dollar value (linear),
    trailing its string along the exact path it took. The ARKK balloon is
    red, the losing side; the SPY balloon is coloured by its height.

THE CLAIM THE VISUAL MAKES
    Buying the hype fund at its peak lost about half the money more than five
    years later, while the plain index bought the same day roughly doubled.

WHAT THE VISUAL IS NOT
    Not a verdict on ARKK over its whole life: it was the best fund in the
    country in 2020, and someone who bought in 2016 is still ahead. The start
    date is picked at the top on purpose, because that is when the inflows
    peaked, but a single entry day is still a single entry day. No taxes or
    fees beyond the funds' own.

RUN
    python HypeBalloon_Reel_Pipeline.py --smoke   # 3 frames -> temp_frames_smoke/
    python HypeBalloon_Reel_Pipeline.py           # full render -> topic.mp4

OUTPUT
    1080x1920 @ 30 FPS, 1.4s hook + 8.5s build + 1.8s hold = 11.7s

STAGES
    DATA -> COMPUTE -> VALIDATE (asserts, writes figures.json) -> RENDER -> COMPILE
"""
import argparse, json, os, shutil, subprocess, time, warnings
from multiprocessing import Pool
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["text.parse_math"] = False   # "$5,416" is money, not maths
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "_cache")

CONFIG = {
    "TICKERS": ["ARKK", "SPY"],
    "START": "2021-02-12",
    "STAKE": 10_000,
    "ROW_Y": {"ARKK": -0.6, "SPY": 0.6},
    "ZMAX": 26.0,              # $k
    "BOX": (1.3, 0.55, 1.6),
    "R": 0.075,                # balloon radius as a share of the box height
    "STRING_LW": 3.2,

    # render (leave these alone unless the topic needs landscape)
    "W": 1080, "H": 1920, "DPI": 100,
    "FPS": 30,
    "HOOK_SEC": 1.4, "BUILD_SEC": 8.5, "HOLD_SEC": 1.8,
    "ELEV_BASE": 10, "AZIM_START": -70, "AZIM_SWEEP": 14,
    "ZOOM": 1.08,
    "WORKERS": 7,
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
SPY_CMAP = LinearSegmentedColormap.from_list(
    "spy", [THEME["BLUE"], THEME["CYAN"], THEME["GREEN"], THEME["YELLOW"]], N=256)

SAFE = {"TOP": 0.12, "BOTTOM": 0.20, "RIGHT": 0.14}

LAYOUT = {
    "TITLE": 0.862, "SUBTITLE": 0.836, "KEY": 0.814, "HANDLE": 0.792,
    "AXES": [-0.08, 0.29, 1.16, 0.49],
    "NOTE": 0.272, "BIG": 0.242, "SUB": 0.215,
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------- DATA
def fetch():
    """Yahoo once, then the CSV cache. Delete _cache/ to refetch.
    A failed fetch stops the run; nothing synthetic is ever drawn."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    p = os.path.join(CACHE_DIR, "arkk_spy.csv")
    if not os.path.exists(p):
        import yfinance as yf
        log(f"[Data] fetching {CONFIG['TICKERS']}")
        px = yf.download(CONFIG["TICKERS"], start="2020-01-01", interval="1d",
                         progress=False, auto_adjust=True)["Close"]
        if px.empty or px.dropna().empty:
            raise SystemExit("no data returned; refusing to draw a synthetic tape")
        px.to_csv(p)
    log(f"[Data] cache {p}")
    return pd.read_csv(p, index_col=0, parse_dates=True).dropna()


# ------------------------------------------------------------- COMPUTE
def compute(px):
    check_peak(px)
    y20 = px["ARKK"].loc["2020"]
    run_2020 = float(y20.iloc[-1] / y20.iloc[0])     # the run-up that made the hype
    px = px.loc[CONFIG["START"]:]
    assert px.index[0] == pd.Timestamp(CONFIG["START"])
    k = CONFIG["STAKE"]
    v = {t: (k * px[t] / px[t].iloc[0]).to_numpy() for t in CONFIG["TICKERS"]}
    w = px.resample("W-FRI").last().index           # weekly knots for the strings
    pos = np.unique(np.r_[px.index.get_indexer(w, method="pad"), len(px) - 1])
    pos = pos[pos >= 0]
    days = (px.index - px.index[0]).days.to_numpy() / 365.25
    return {"dates": px.index, "pos": pos, "x": days[pos],
            "arkk": v["ARKK"][pos], "spy": v["SPY"][pos],
            "arkk_end": float(v["ARKK"][-1]), "spy_end": float(v["SPY"][-1]),
            "arkk_min": float(v["ARKK"].min()),
            "arkk_min_d": px.index[int(v["ARKK"].argmin())], "run_2020": run_2020}


def check_peak(px):
    """ARKK's all-time closing high is the start date, or the story is wrong."""
    a = px["ARKK"]
    assert a.idxmax() == pd.Timestamp(CONFIG["START"]), f"ARKK peak is {a.idxmax()}"


# ------------------------------------------------------------ VALIDATE
def validate(d):
    """Every number that reaches a frame is asserted here, before any render."""
    k = CONFIG["STAKE"]
    assert 0.2 * k <= d["arkk_end"] <= 0.9 * k, f"ARKK ${d['arkk_end']:,.0f} outside band"
    assert 1.2 * k <= d["spy_end"] <= 3.0 * k, f"SPY ${d['spy_end']:,.0f} outside band"
    assert 0.05 * k <= d["arkk_min"] <= 0.5 * k
    assert 1.5 <= d["run_2020"] <= 4, f"2020 run {d['run_2020']:.2f}x"
    assert max(d["spy"].max(), d["arkk"].max()) / 1e3 < CONFIG["ZMAX"] * 0.85

    figs = {
        "stake": f"${k:,.0f}",
        "arkk": f"${d['arkk_end']:,.0f}",
        "spy": f"${d['spy_end']:,.0f}",
        "arkk_k": f"${d['arkk_end'] / 1e3:.1f}k",
        "spy_k": f"${d['spy_end'] / 1e3:.1f}k",
        "arkk_pct": f"{(d['arkk_end'] / k - 1) * 100:.0f}%",
        "spy_pct": f"+{(d['spy_end'] / k - 1) * 100:.0f}%",
        "arkk_min": f"${d['arkk_min']:,.0f}",
        "arkk_dd": f"{(d['arkk_min'] / k - 1) * 100:.0f}%",
        "arkk_min_d": d["arkk_min_d"].strftime("%b %Y"),
        "ratio": f"{d['spy_end'] / d['arkk_end']:.1f}x",
        "start": d["dates"][0].strftime("%b %Y"),
        "start_day": d["dates"][0].strftime("%d %b %Y").lstrip("0"),
        "end": d["dates"][-1].strftime("%b %Y"),
        "index": "S&P 500",
        "run_2020": f"+{(d['run_2020'] - 1) * 100:.0f}%",
        "year_2020": "2020",
        "years": f"{(d['dates'][-1] - d['dates'][0]).days / 365.25:.1f}",
    }
    with open(os.path.join(BASE_DIR, "figures.json"), "w") as fh:
        json.dump(figs, fh, indent=1)
    log(f"[validate] {figs}")
    d["figs"] = figs
    return figs


# -------------------------------------------------------------- RENDER
def _shade(c, k):
    c = np.array(to_rgba(c)); c[:3] = np.clip(c[:3] * k, 0, 1); return c


def _radii(d):
    c = CONFIG
    bx, by, bz = c["BOX"]
    xr = d["x"][-1] + 1.2
    r = c["R"] * bz
    return r * xr / bx, r * 3.0 / by, r * c["ZMAX"] / bz


def _balloon(polys, cols, x, y, z, rad, col):
    """A lit sphere with a small knot underneath, centre above the string end."""
    rx, ry, rz = rad
    u = np.linspace(0, 2 * np.pi, 34)
    v = np.linspace(0, np.pi, 20)
    cz = z + rz * 1.05
    L = np.array([-0.45, -0.6, 0.65]); L /= np.linalg.norm(L)
    for i in range(len(u) - 1):
        for j in range(len(v) - 1):
            uu, vv = (u[i] + u[i + 1]) / 2, (v[j] + v[j + 1]) / 2
            n = np.array([np.cos(uu) * np.sin(vv), np.sin(uu) * np.sin(vv), np.cos(vv)])
            k = 0.45 + 0.75 * max(n @ L, 0) + 0.35 * max(n @ L, 0) ** 12
            q = [(x + rx * np.cos(a) * np.sin(b), y + ry * np.sin(a) * np.sin(b),
                  cz + rz * 1.12 * np.cos(b))
                 for a, b in ((u[i], v[j]), (u[i + 1], v[j]), (u[i + 1], v[j + 1]),
                              (u[i], v[j + 1]))]
            polys.append(q); cols.append(_shade(col, k))
    kz = cz - rz * 1.12
    polys.append([(x - rx * 0.18, y, kz - rz * 0.12), (x + rx * 0.18, y, kz - rz * 0.12),
                  (x, y, kz + rz * 0.05)])
    cols.append(_shade(col, 0.7))


def scene(ax, d, frac):
    c = CONFIG
    n_all = len(d["x"])
    n = n_all if frac >= 1 else max(1, int(np.ceil(frac * n_all)))
    x = d["x"][:n]
    za, zs = d["arkk"][:n] / 1e3, d["spy"][:n] / 1e3
    ya, ys = c["ROW_Y"]["ARKK"], c["ROW_Y"]["SPY"]
    rad = _radii(d)
    colS = SPY_CMAP(0.15 + 0.85 * min(zs[-1] / 22, 1))
    # strings: the exact path each balloon took
    if n > 1:
        ax.plot(x, [ys] * n, zs, color=_shade(colS, 0.9), lw=c["STRING_LW"], alpha=0.95)
        ax.plot(x, [ya] * n, za, color=THEME["RED"], lw=c["STRING_LW"], alpha=0.95)
    # floor guide at the stake
    x1 = d["x"][-1] + 1.2
    for y in (ya, ys):
        ax.plot([-0.3, x1], [y, y], [10, 10], color=THEME["TEXT_DIM"], lw=0.6,
                alpha=0.45, ls=(0, (4, 4)))
        ax.plot([-0.3, x1], [y, y], [0, 0], color=THEME["TEXT_DIM"], lw=0.6, alpha=0.35)
    polys, cols = [], []
    # gain and loss against the $10k stake, filled as a translucent sail
    for i in range(n - 1):
        for y, z, cf in ((ys, zs, lambda q: SPY_CMAP(0.15 + 0.85 * min(q / 22, 1))),
                         (ya, za, lambda q: THEME["RED"])):
            polys.append([(x[i], y, 10), (x[i + 1], y, 10), (x[i + 1], y, z[i + 1]),
                          (x[i], y, z[i])])
            cc = np.array(to_rgba(cf(z[i]))); cc[3] = 0.42
            cols.append(cc)
    _balloon(polys, cols, x[-1], ys, zs[-1], rad, colS)
    _balloon(polys, cols, x[-1], ya, za[-1], rad, THEME["RED"])
    ax.add_collection3d(Poly3DCollection(polys, facecolors=cols, edgecolors="none"),
                        autolim=False)
    ax.set_xlim(-0.3, x1); ax.set_ylim(-1.5, 1.5); ax.set_zlim(0, c["ZMAX"])
    return n - 1


def style_axes(ax):
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
        pane.line.set_color((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def scene_labels(ax, d, last, fs=11):
    """Projected to 2D so the balloons cannot depth-sort the labels away."""
    from mpl_toolkits.mplot3d import proj3d
    M = ax.get_proj()
    c, figs = CONFIG, d["figs"]
    rad = _radii(d)

    def put(x, y, z, s, **kw):
        x2, y2, _ = proj3d.proj_transform(x, y, z, M)
        ax.text2D(x2, y2, s, transform=ax.transData, zorder=1e6, ha="center",
                  va="center", **kw)

    put(-0.55, c["ROW_Y"]["ARKK"], 10, "$10k", color=THEME["TEXT_DIM"],
        fontsize=fs - 1, family=THEME["MONO"])
    for yr in (2022, 2024, 2026):
        xx = (pd.Timestamp(f"{yr}-01-01") - d["dates"][0]).days / 365.25
        put(xx, c["ROW_Y"]["ARKK"] - 0.9, 0, str(yr), color=THEME["TEXT_DIM"],
            fontsize=fs - 1, family=THEME["MONO"])
    done = last == len(d["x"]) - 1
    xl = d["x"][last]
    zs, za = d["spy"][last] / 1e3, d["arkk"][last] / 1e3
    put(xl - rad[0] * 4.2, c["ROW_Y"]["SPY"], zs + rad[2] * 1.2,
        f"S&P 500\n{figs['spy']}" if done else "S&P 500",
        color=THEME["YELLOW"], fontsize=fs + 4, fontweight="bold", family=THEME["MONO"])
    put(xl + 0.05, c["ROW_Y"]["ARKK"], za - rad[2] * 1.6,
        f"ARKK\n{figs['arkk']}" if done else "ARKK",
        color=THEME["RED"], fontsize=fs + 4, fontweight="bold", family=THEME["MONO"],
        bbox=dict(facecolor="black", alpha=0.8, edgecolor=THEME["RED"], lw=1.0,
                  boxstyle="round,pad=0.35"))


def text_block(fig, figs, scale=1.0):
    L, s = LAYOUT, scale
    fig.text(0.5, L["TITLE"], "I BOUGHT THE HYPE", ha="center",
             fontsize=29 * s, fontweight="bold", color=THEME["TEXT"],
             family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             f"{figs['stake']} into ARKK at its peak, {figs['start_day']}, or into the S&P 500",
             ha="center", fontsize=12 * s, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.5, L["KEY"],
             f"{figs['start']} to {figs['end']}  ·  height = dollars  ·  red = ARKK",
             ha="center", fontsize=10 * s, color=THEME["TEXT_DIM"], family=THEME["FONT"])
    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10 * s,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])
    fig.text(0.5, L["BIG"], f"ARKK {figs['arkk']}  ·  S&P {figs['spy']}",
             ha="center", fontsize=23 * s, color=THEME["YELLOW"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             f"ARKK low {figs['arkk_min']} ({figs['arkk_dd']})  ·  dividends included",
             ha="center", fontsize=11.5 * s, color=THEME["TEXT_DIM"],
             family=THEME["MONO"])


def note(fig, d, last, scale=1.0):
    if last is None:
        return
    fig.text(0.5, LAYOUT["NOTE"], d["dates"][d["pos"][last]].strftime("%b %Y").upper(),
             ha="center", fontsize=12 * scale, color=THEME["TEXT"],
             family=THEME["MONO"])


def render_frame(idx, total, d, out_path):
    c, figs = CONFIG, d["figs"]
    n_hook = int(c["FPS"] * c["HOOK_SEC"])
    n_build = int(c["FPS"] * c["BUILD_SEC"])
    if idx < n_hook:
        frac = 0.0
    elif idx < n_hook + n_build:
        frac = (idx - n_hook + 1) / n_build
    else:
        frac = 1.0
    t = idx / max(total - 1, 1)

    fig = plt.figure(figsize=(c["W"] / c["DPI"], c["H"] / c["DPI"]),
                     dpi=c["DPI"], facecolor=THEME["BG"])
    text_block(fig, figs)
    ax = fig.add_axes(LAYOUT["AXES"], projection="3d", facecolor=THEME["BG"])
    last = scene(ax, d, frac)
    note(fig, d, last)
    ax.view_init(elev=c["ELEV_BASE"] + 4 * np.sin(t * np.pi),
                 azim=c["AZIM_START"] + c["AZIM_SWEEP"] * t)
    ax.set_box_aspect(c["BOX"], zoom=c["ZOOM"])
    style_axes(ax)
    scene_labels(ax, d, last)

    fig.savefig(out_path, dpi=c["DPI"], facecolor=THEME["BG"])
    plt.close(fig)
# ------------------------------------------------------------- COMPILE
def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


_D = None


def _init(d):
    global _D
    _D = d


def _job(args):
    i, total, path = args
    render_frame(i, total, _D, path)
    return i


def main(smoke=False):
    c = CONFIG
    d = compute(fetch())
    validate(d)                       # <- before a single frame is written

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
    jobs = [(i, total, os.path.join(frames, f"f_{i:05d}.png")) for i in range(total)]
    with Pool(c["WORKERS"], initializer=_init, initargs=(d,)) as pool:
        for n, _ in enumerate(pool.imap_unordered(_job, jobs)):
            if n % 30 == 0:
                log(f"  frame {n}/{total}  ({time.time() - t0:.0f}s)")
    assert len(os.listdir(frames)) == total, "frame count mismatch"

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
