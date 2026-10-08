"""
PanicSell_Reel_Pipeline.py
==========================
Reel 28 for @quant.traderr, "The Panic Sell". One dollar in SPY that is never
touched, against one dollar owned by a trader who sells every time SPY closes
20% below its last high and only buys back once the old high is regained,
the moment it "feels safe again".

THE DATA
    SPY daily closes from Yahoo, auto-adjusted (dividends reinvested), and the
    13-week T-bill yield ^IRX for the cash leg, 29 Jan 1993 to Sep 2026.
    Real data throughout. Nothing here is simulated.

THE MATHS
    hold_t  = SPY_t / SPY_0
    panic:  in the market, track the running peak P of SPY. Close <= 0.8 * P
            -> sell at that close, sit in T-bills (^IRX / 252 per day).
            Close >= P -> buy back at that close, reset the peak.
    The seller always sells at -20% or worse and always buys back at the old
    peak, so every round trip locks in at least the 25% gap between the two
    prices, minus what the T-bills paid while waiting.

    In 3D: a double wall. Time runs along the long axis, each wall rises from
    the floor to its dollar on a log scale, HOLD behind, PANIC in front. Both
    are coloured by their own height; PANIC turns red for every stretch it sat
    in cash. A white post marks each panic sale.

THE CLAIM THE VISUAL MAKES
    Selling after a 20% drop and waiting for the all clear left the trader with
    well under half of what doing nothing would have.

WHAT THE VISUAL IS NOT
    Not a claim that every exit is wrong: a seller who buys back near the
    bottom does fine, this rule simply never does. One rule, one threshold, one
    index over one 33-year stretch with only a handful of bear markets, so the
    count of round trips is small. No taxes or trading costs, which would make
    the seller look worse. The walls are filled to the floor for readability;
    only their top edge is data.

RUN
    python PanicSell_Reel_Pipeline.py --smoke   # 3 frames -> temp_frames_smoke/
    python PanicSell_Reel_Pipeline.py           # full render -> topic.mp4

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
    "TICKERS": ["SPY", "^IRX"],
    "PERIOD": "max",
    "START": "1993-01-29",
    "DROP": 0.20,
    "STEP": 12,
    "LANE": 0.5,

    # render (leave these alone unless the topic needs landscape)
    "W": 1080, "H": 1920, "DPI": 100,
    "FPS": 30,
    "HOOK_SEC": 1.4, "BUILD_SEC": 8.5, "HOLD_SEC": 1.8,
    "ELEV_BASE": 14, "AZIM_START": -46, "AZIM_SWEEP": 14,
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
    if px["SPY"].dropna().empty or px["^IRX"].dropna().empty:
        raise SystemExit("SPY or ^IRX came back empty; refusing to continue")
    px.to_csv(path)
    return px


# ------------------------------------------------------------- COMPUTE
def compute(px):
    c = CONFIG
    s = px["SPY"].dropna().loc[c["START"]:]
    irx = px["^IRX"].reindex(s.index).ffill()
    if irx.isna().any():
        raise SystemExit("T-bill series has holes before its first value")
    v = s.values
    cash = np.r_[0.0, irx.values[:-1] / 100.0 / 252.0]
    hold = v / v[0]

    w = np.empty(len(v)); w[0] = 1.0
    inm = np.ones(len(v), dtype=bool)
    peak, invested, sells, cash_days = v[0], True, [], 0
    sell_px, buy_px = [], []
    for i in range(1, len(v)):
        w[i] = w[i - 1] * (v[i] / v[i - 1] if invested else 1 + cash[i])
        if not invested:
            cash_days += 1
        if invested:
            peak = max(peak, v[i])
            if v[i] <= peak * (1 - c["DROP"]):
                invested = False; sells.append(i); sell_px.append(v[i])
        elif v[i] >= peak:
            invested = True; peak = v[i]; buy_px.append(v[i])
        inm[i] = invested
    return {"s": s, "hold": hold, "panic": w, "inm": inm, "sells": sells,
            "sell_px": sell_px, "buy_px": buy_px, "cash_days": cash_days,
            "dates": s.index}


# ------------------------------------------------------------ VALIDATE
def validate(d):
    """Every number that reaches a frame is asserted here, before any render."""
    h, p = float(d["hold"][-1]), float(d["panic"][-1])
    n = len(d["sells"])
    keep = p / h
    yrs_cash = d["cash_days"] / 252.0
    gaps = [b / s - 1 for s, b in zip(d["sell_px"], d["buy_px"])]

    assert 5 <= h <= 200, f"hold ${h:.2f} outside the $5..$200 band"
    assert 0.5 <= p <= 200, f"panic ${p:.2f} outside the $0.50..$200 band"
    assert p < h, "the panic seller beat holding; the reel's claim is gone"
    assert 1 <= n <= 20, f"{n} panic sales"
    assert all(g >= 0.25 - 1e-9 for g in gaps), "a buy-back below the 25% gap"
    assert 0 < yrs_cash < 34
    assert len(d["s"]) > 8000

    figs = {
        "hold_usd": f"${h:,.2f}",
        "panic_usd": f"${p:,.2f}",
        "keep": f"{keep * 100:.0f}%",
        "lost": f"{(1 - keep) * 100:.0f}%",
        "n_sells": f"{n}",
        "sell_years": ", ".join(str(d["dates"][i].year) for i in d["sells"]),
        "years_cash": f"{yrs_cash:.1f}",
        "drop": f"-{CONFIG['DROP'] * 100:.0f}%",
        "gap": f"+{min(gaps) * 100:.0f}%",
        "start": d["dates"][0].strftime("%b %Y"),
        "end": d["dates"][-1].strftime("%b %Y"),
    }
    with open(os.path.join(BASE_DIR, "figures.json"), "w") as fh:
        json.dump(figs, fh, indent=1)
    log(f"[validate] {figs}")
    d["figs"] = figs
    return figs


# -------------------------------------------------------------- RENDER
def _wall(ax, xs, y, zs, cols):
    """A vertical wall from the floor up to the dollar, one colour per column.
    The bright top edge is the data line."""
    X = np.vstack([xs, xs]); Y = np.full_like(X, y)
    Z = np.vstack([np.zeros_like(zs), zs])
    fc = np.asarray(cols)[None, :-1, :]
    ax.plot_surface(X, Y, Z, facecolors=fc, rstride=1, cstride=1,
                    linewidth=0,
                    antialiased=False, shade=False)
    top = np.asarray(cols).copy(); top[:, :3] = np.clip(top[:, :3] * 1.25 + 0.1, 0, 1)
    for k in range(len(xs) - 1):
        ax.plot(xs[k:k + 2], [y, y], zs[k:k + 2], color=top[k], lw=1.6)


def scene(ax, d, frac):
    c = CONFIG
    idx = np.arange(0, len(d["hold"]), c["STEP"])
    if idx[-1] != len(d["hold"]) - 1:
        idx = np.r_[idx, len(d["hold"]) - 1]
    n = max(2, int(round(frac * len(idx))))
    sel = idx[:n]
    N = len(d["hold"]) - 1
    x = sel / N
    zh = np.log10(d["hold"][sel]); zp = np.log10(d["panic"][sel])
    zmax = np.log10(d["hold"].max())
    norm = lambda z: CMAP(0.05 + 0.72 * np.clip(z / zmax, 0, 1))
    ch, cp = norm(zh), norm(zp)
    cp[~d["inm"][sel]] = matplotlib.colors.to_rgba(THEME["RED"])
    L = c["LANE"]
    if frac > 0:
        _wall(ax, x, L, zh, ch)        # HOLD behind
        _wall(ax, x, -L, zp, cp)       # PANIC in front
    for y in (L, -L):
        ax.plot([0, 1], [y, y], [0, 0], color=THEME["TEXT_DIM"], lw=0.8, alpha=0.6)
    for i in d["sells"]:
        if frac > 0 and i <= sel[-1]:
            xi, zi = i / N, np.log10(d["panic"][i])
            ax.plot([xi, xi], [-L, -L], [zi, zi + 0.28], color="white", lw=1.2)
            ax.scatter([xi], [-L], [zi + 0.28], s=55, color="white",
                       depthshade=False)
    ax.set_xlim(0, 1); ax.set_ylim(-0.75, 0.75); ax.set_zlim(-0.05, zmax + 0.2)
    return sel[-1]


def style_axes(ax):
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
        pane.line.set_color((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def scene_labels(ax, d, last, fs=11):
    """Projected to 2D so the tubes cannot depth-sort the labels away."""
    from mpl_toolkits.mplot3d import proj3d
    M = ax.get_proj()
    L, n = CONFIG["LANE"], len(d["hold"]) - 1

    def put(x, y, z, s, **kw):
        x2, y2, _ = proj3d.proj_transform(x, y, z, M)
        ax.text2D(x2, y2, s, transform=ax.transData, zorder=1e6, ha="center",
                  va="center", **kw)

    put(0.04, -L, -0.2, "PANIC", color=THEME["CYAN"], fontsize=fs + 4,
        fontweight="bold", family=THEME["FONT"])
    xe = last / n
    put(xe - 0.16, L, np.log10(d["hold"][last]) + 0.1,
        f"HOLD ${d['hold'][last]:,.2f}" if last == n else "", color=THEME["ORANGE"],
        fontsize=fs + 2, fontweight="bold", family=THEME["MONO"])
    put(xe - 0.05, -L, np.log10(d["panic"][last]) + 0.42,
        f"PANIC ${d['panic'][last]:,.2f}" if last == n else "", color=THEME["CYAN"],
        fontsize=fs + 2, fontweight="bold", family=THEME["MONO"])


def text_block(fig, figs, scale=1.0):
    L, s = LAYOUT, scale
    fig.text(0.5, L["TITLE"], "THE PANIC SELL", ha="center",
             fontsize=29 * s, fontweight="bold", color=THEME["TEXT"],
             family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             f"$1 in SPY  |  sell at {figs['drop']}, buy back at the old high  |  "
             f"{figs['start']} to {figs['end']}",
             ha="center", fontsize=12 * s, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.5, L["KEY"],
             f"height = dollar (log)  ·  red = sitting in cash  ·  white post = panic sale",
             ha="center", fontsize=10 * s, color=THEME["TEXT_DIM"], family=THEME["FONT"])
    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10 * s,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])
    fig.text(0.5, L["BIG"], f"PANIC KEPT {figs['keep']} OF THE MONEY",
             ha="center", fontsize=21 * s, color=THEME["RED"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             f"{figs['n_sells']} panic sales: {figs['sell_years']}  ·  cash earned T-bills",
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
    last = scene(ax, d, frac)
    note(fig, d, last)
    ax.view_init(elev=c["ELEV_BASE"] + 5 * np.sin(t * np.pi),
                 azim=c["AZIM_START"] + c["AZIM_SWEEP"] * t)
    ax.set_box_aspect((1.7, 0.75, 1.5), zoom=c["ZOOM"])
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
