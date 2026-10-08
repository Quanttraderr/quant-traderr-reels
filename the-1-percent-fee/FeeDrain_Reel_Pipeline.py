"""
FeeDrain_Reel_Pipeline.py
=========================
Reel 29 for @quant.traderr, "The 1% Fee". $10,000 in SPY with dividends
reinvested, once as is and once with a 1% annual fee taken out, the kind an
advisor or an active fund charges. Every dollar the fee removes, plus
everything that dollar would have grown into, is peeled off as a red stream.

THE DATA
    SPY daily closes from Yahoo, auto-adjusted (dividends reinvested),
    29 Jan 1993 to Sep 2026. Real data throughout. The fee is a hypothetical
    overlay on that real path, not a real product's history: it is charged
    daily at (1 - fee)^(1/252), which compounds to exactly 1% a year.

THE MATHS
    no_fee_t  = 10,000 * SPY_t / SPY_0
    fee_t     = no_fee_t * (1 - f)^(years_t)
    drained_t = no_fee_t - fee_t
    The share lost, 1 - (1 - f)^years, depends only on time, not on the path,
    which VALIDATE checks against the end values.

    In 3D: particle streams rising like a trunk. Time runs up the vertical axis. Each time slice
    holds dots in proportion to the no-fee balance (linear dollars, so the
    early years are a thin trickle). The share of dots the fee has claimed by
    then is red and flows into a second stream that splits off, FEES. The rest
    stay in YOU, coloured by time. A dim ring traces how wide YOU would be
    without the fee.

THE CLAIM THE VISUAL MAKES
    Over 33 years a fee that sounds small took close to a third of the money.

WHAT THE VISUAL IS NOT
    Not a claim that every fee is wasted: a manager who beats the index by
    more than the fee earns it, most do not, and this reel does not test that.
    SPY's own 0.09% expense ratio is already inside the no-fee line. No taxes,
    no inflation. Dot positions inside a stream are random scatter for texture;
    only the dot count per slice and the red share carry data.

RUN
    python FeeDrain_Reel_Pipeline.py --smoke   # 3 frames -> temp_frames_smoke/
    python FeeDrain_Reel_Pipeline.py           # full render -> topic.mp4

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
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "_cache")

CONFIG = {
    "TICKERS": ["SPY"],
    "PERIOD": "max",
    "START": "1993-01-29",
    "AMOUNT": 10_000,
    "FEE": 0.01,
    "OTHER_FEES": (0.005, 0.02),
    "SLICES": 260,
    "NMAX": 260,
    "R": 0.55,
    "LANE": 0.32,
    "SPLIT": 0.55,
    "DOT": 4.5,

    # render (leave these alone unless the topic needs landscape)
    "W": 1080, "H": 1920, "DPI": 100,
    "FPS": 30,
    "HOOK_SEC": 1.4, "BUILD_SEC": 8.5, "HOLD_SEC": 1.8,
    "ELEV_BASE": 12, "AZIM_START": -100, "AZIM_SWEEP": 30,
    "ZOOM": 1.12,
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
    """yfinance once, then a CSV cache. Delete _cache/ to refetch."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "closes.csv")
    if os.path.exists(path):
        log(f"[Data] cache {path}")
        return pd.read_csv(path, index_col=0, parse_dates=True)
    import yfinance as yf
    log(f"[Data] fetching {CONFIG['TICKERS']}")
    px = yf.download(CONFIG["TICKERS"], period=CONFIG["PERIOD"],
                     interval="1d", progress=False, auto_adjust=True)["Close"]
    if px.empty or any(t not in px.columns for t in CONFIG["TICKERS"]):
        raise SystemExit("no data returned; refusing to draw a synthetic tape")
    px = px[CONFIG["TICKERS"]]
    if px["SPY"].dropna().empty:
        raise SystemExit("SPY came back empty; refusing to continue")
    px.to_csv(path)
    return px


# ------------------------------------------------------------- COMPUTE
def compute(px):
    c = CONFIG
    s = px["SPY"].dropna().loc[c["START"]:]
    yrs = (s.index - s.index[0]).days.values / 365.25
    no_fee = c["AMOUNT"] * s.values / s.values[0]
    fee = no_fee * (1 - c["FEE"]) ** yrs
    other = {f: no_fee[-1] * (1 - f) ** yrs[-1] for f in c["OTHER_FEES"]}

    # particle cloud, fixed seed: positions are texture, counts are data
    rng = np.random.default_rng(7)
    K = c["SLICES"]
    idx = np.linspace(0, len(s) - 1, K).astype(int)
    wmax = no_fee.max()
    pts = []                                     # (slice, u, v, x_jit, red)
    for k, i in enumerate(idx):
        n = max(3, int(round(c["NMAX"] * no_fee[i] / wmax)))
        share = 1 - fee[i] / no_fee[i]
        red = np.zeros(n, dtype=bool)
        red[:int(round(share * n))] = True
        th = rng.uniform(0, 2 * np.pi, n); rr = np.sqrt(rng.uniform(0, 1, n))
        pts.append(np.c_[np.full(n, k), rr * np.cos(th), rr * np.sin(th),
                         rng.uniform(0, 1, n), red])
    pts = np.vstack(pts)
    return {"s": s, "dates": s.index, "yrs": yrs, "no_fee": no_fee, "fee": fee,
            "other": other, "idx": idx, "pts": pts, "wmax": wmax}


# ------------------------------------------------------------ VALIDATE
def validate(d):
    """Every number that reaches a frame is asserted here, before any render."""
    c = CONFIG
    nf, f1 = float(d["no_fee"][-1]), float(d["fee"][-1])
    lost = 1 - f1 / nf
    theory = 1 - (1 - c["FEE"]) ** d["yrs"][-1]
    red_share = d["pts"][d["pts"][:, 0] == c["SLICES"] - 1][:, 4].mean()

    assert abs(lost - theory) < 1e-9, "fee drag does not match (1-f)^years"
    assert 50_000 <= nf <= 2_000_000, f"no-fee end ${nf:,.0f} outside band"
    assert 0.10 <= lost <= 0.60, f"share lost {lost:.2f} outside 10%..60%"
    assert abs(red_share - lost) < 0.02, "red dots do not match the lost share"
    assert 30 <= d["yrs"][-1] <= 40
    assert c["OTHER_FEES"] == (0.005, 0.02), "fee_half / fee_two labels are fixed"
    assert len(d["s"]) > 8000

    o = d["other"]
    figs = {
        "amount": f"${c['AMOUNT']:,.0f}",
        "fee": f"{c['FEE'] * 100:.0f}%",
        "no_fee_usd": f"${nf:,.0f}",
        "fee_usd": f"${f1:,.0f}",
        "drained_usd": f"${nf - f1:,.0f}",
        "lost": f"{lost * 100:.0f}%",
        "years": f"{d['yrs'][-1]:.1f}",
        "half_fee": f"{(1 - o[0.005] / nf) * 100:.0f}%",
        "two_fee": f"{(1 - o[0.02] / nf) * 100:.0f}%",
        "spy_er": "0.09%",
        "fee_half": "0.5%",
        "fee_two": "2%",
        "start": d["dates"][0].strftime("%b %Y"),
        "end": d["dates"][-1].strftime("%b %Y"),
    }
    with open(os.path.join(BASE_DIR, "figures.json"), "w") as fh:
        json.dump(figs, fh, indent=1)
    log(f"[validate] {figs}")
    d["figs"] = figs
    return figs


# -------------------------------------------------------------- RENDER
def scene(ax, d, frac, t):
    """Time runs up the z axis, so the streams rise like a trunk with a red
    branch splitting off to the left."""
    c = CONFIG
    K = c["SLICES"]
    kmax = max(1, int(round(frac * (K - 1))))
    P = d["pts"][d["pts"][:, 0] <= kmax] if frac > 0 else d["pts"][:0]
    k = P[:, 0]
    i = d["idx"][k.astype(int)]
    z = (k + (P[:, 3] + t * 6) % 1.0) / (K - 1)        # dots drift upward
    R = c["R"]
    nf, fe = d["no_fee"][i], d["fee"][i]
    red = P[:, 4] > 0.5
    share_end = 1 - d["fee"][-1] / d["no_fee"][-1]
    r = np.where(red, R * np.sqrt((nf - fe) / d["wmax"]), R * np.sqrt(fe / d["wmax"]))
    cx = np.where(red, c["LANE"] - (c["LANE"] + c["SPLIT"]) * (1 - fe / nf) / share_end,
                  c["LANE"])
    x = cx + r * P[:, 1]
    y = r * P[:, 2]
    cols = CMAP(0.05 + 0.72 * k / (K - 1))
    cols[red] = matplotlib.colors.to_rgba(THEME["RED"])
    if len(P):
        ax.scatter(x, y, z, s=c["DOT"], c=cols, depthshade=False, linewidths=0,
                   alpha=0.9)
    # ghost: how wide YOU would be without the fee
    th = np.linspace(0, 2 * np.pi, 48)
    for kk in (np.arange(0, kmax + 1, 8) if frac > 0 else []):
        rg = R * np.sqrt(d["no_fee"][d["idx"][kk]] / d["wmax"])
        ax.plot(c["LANE"] + rg * np.cos(th), rg * np.sin(th),
                np.full_like(th, kk / (K - 1)), color=THEME["TEXT_DIM"],
                lw=0.6, alpha=0.5)
    ax.plot([c["LANE"], c["LANE"]], [0, 0], [0, 1], color=THEME["TEXT_DIM"],
            lw=0.6, alpha=0.4)
    ax.set_xlim(-0.95, 0.95); ax.set_ylim(-0.95, 0.95); ax.set_zlim(-0.02, 1.16)
    return d["idx"][kmax] if frac > 0 else 0


def style_axes(ax):
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
        pane.line.set_color((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def scene_labels(ax, d, last, fs=11):
    """Projected to 2D so the dots cannot depth-sort the labels away."""
    from mpl_toolkits.mplot3d import proj3d
    M = ax.get_proj()
    c, figs = CONFIG, d["figs"]
    n = len(d["s"]) - 1

    def put(x, y, z, s, **kw):
        x2, y2, _ = proj3d.proj_transform(x, y, z, M)
        ax.text2D(x2, y2, s, transform=ax.transData, zorder=1e6, ha="center",
                  va="center", **kw)

    R = c["R"]
    if last == n:
        put(c["LANE"], 0, 1.08, f"YOU {figs['fee_usd']}".replace("$", r"\$"),
            color=THEME["ORANGE"], fontsize=fs + 3, fontweight="bold",
            family=THEME["MONO"])
        put(-c["SPLIT"], 0, 1.08, f"FEES {figs['drained_usd']}".replace("$", r"\$"),
            color=THEME["RED"], fontsize=fs + 3, fontweight="bold",
            family=THEME["MONO"])
    put(c["LANE"] - 0.42, 0, 0.03, figs["amount"].replace("$", r"\$"), color=THEME["CYAN"],
        fontsize=fs + 1, fontweight="bold", family=THEME["MONO"])


def text_block(fig, figs, scale=1.0):
    L, s = LAYOUT, scale
    fig.text(0.5, L["TITLE"], "THE 1% FEE", ha="center",
             fontsize=30 * s, fontweight="bold", color=THEME["TEXT"],
             family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             f"{figs['amount']} in SPY, dividends reinvested  |  "
             f"{figs['start']} to {figs['end']}",
             ha="center", fontsize=13 * s, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.5, L["KEY"],
             "dots = dollars  ·  red = taken by the fee, plus what it would have grown into",
             ha="center", fontsize=10 * s, color=THEME["TEXT_DIM"], family=THEME["FONT"])
    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10 * s,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])
    fig.text(0.5, L["BIG"], f"{figs['fee']} A YEAR TOOK {figs['lost']} OF IT",
             ha="center", fontsize=22 * s, color=THEME["RED"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             f"no fee {figs['no_fee_usd']}  ·  with fee {figs['fee_usd']}".replace("$", r"\$"),
             ha="center", fontsize=11.5 * s, color=THEME["TEXT_DIM"],
             family=THEME["MONO"])


def note(fig, d, last, scale=1.0):
    fig.text(0.5, LAYOUT["NOTE"], f"{d['dates'][last].year}",
             ha="center", fontsize=12 * scale, color=THEME["TEXT"],
             family=THEME["MONO"])


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
    text_block(fig, figs)
    ax = fig.add_axes(LAYOUT["AXES"], projection="3d", facecolor=THEME["BG"])
    last = scene(ax, d, frac, t)
    note(fig, d, last)
    ax.view_init(elev=c["ELEV_BASE"] + 5 * np.sin(t * np.pi),
                 azim=c["AZIM_START"] + c["AZIM_SWEEP"] * t)
    ax.set_box_aspect((1.0, 1.0, 1.9), zoom=c["ZOOM"])
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
