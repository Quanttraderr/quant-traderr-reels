"""
IphoneVsApple_Reel_Pipeline.py
==============================
Reel 37 for @quant.traderr, "Every iPhone vs Apple Stock". The habit it takes
apart: buying the new iPhone on launch day. This reel takes the launch price
of every standard iPhone from 2007 to 2025 and puts the same dollars into
Apple shares on the same day instead.

THE DATA
    AAPL daily closes from Yahoo, auto-adjusted for splits and dividends (so
    dividends are reinvested), Jun 2007 to the latest close, kept in _cache/.
    The launch dates and US launch prices of the standard model are typed in
    from Apple's own announcements (LAUNCHES below). Up to the iPhone 6 (2014)
    Apple quoted the price on a two-year carrier contract, $199; from the 6s
    on it quoted the full price. The table uses what Apple quoted. Real data
    throughout. Nothing is simulated.

THE MATHS
    worth_i = price_i * Close_today / Close_(first trading day >= launch_i)
    spent   = sum price_i ;  worth = sum worth_i

    In 3D: a row of phones. Each launch stands as a phone-shaped slab, its
    height the value of that phone's money in AAPL today, coloured by that
    value. In front of it the phone you actually bought lies flat on the
    floor, red, its thickness the price paid, on the same linear dollar
    scale. The red phones are flat because next to their shares they are.

THE CLAIM THE VISUAL MAKES
    The money spent on one standard iPhone per launch since 2007 would be
    worth more than ten times as much today in the stock of the company that
    sold them.

WHAT THE VISUAL IS NOT
    Not advice to skip the phone: a phone is a tool, and the comparison is
    picked with hindsight on one of the best stocks of the century. The pre
    2015 prices are the subsidised contract prices, so the true cost of those
    phones was higher and the stock side would be larger still. One phone per
    launch is a heavier upgrade habit than most people have. No taxes or
    trading costs.

RUN
    python IphoneVsApple_Reel_Pipeline.py --smoke   # 3 frames -> temp_frames_smoke/
    python IphoneVsApple_Reel_Pipeline.py           # full render -> topic.mp4

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
matplotlib.rcParams["text.parse_math"] = False   # "$145,310" is money, not maths
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "_cache")

# (on sale in the US, model, launch price of the standard model in USD)
LAUNCHES = [
    ("2007-06-29", "iPhone", 499), ("2008-07-11", "iPhone 3G", 199),
    ("2009-06-19", "iPhone 3GS", 199), ("2010-06-24", "iPhone 4", 199),
    ("2011-10-14", "iPhone 4S", 199), ("2012-09-21", "iPhone 5", 199),
    ("2013-09-20", "iPhone 5s", 199), ("2014-09-19", "iPhone 6", 199),
    ("2015-09-25", "iPhone 6s", 649), ("2016-09-16", "iPhone 7", 649),
    ("2017-09-22", "iPhone 8", 699), ("2018-10-26", "iPhone XR", 749),
    ("2019-09-20", "iPhone 11", 699), ("2020-10-23", "iPhone 12", 799),
    ("2021-09-24", "iPhone 13", 799), ("2022-09-16", "iPhone 14", 799),
    ("2023-09-22", "iPhone 15", 799), ("2024-09-20", "iPhone 16", 799),
    ("2025-09-19", "iPhone 17", 799),
]

CONFIG = {
    "TICKER": "AAPL",
    "PHONE_W": 0.7, "PHONE_T": 0.16, "PHONE_L": 1.25,
    "ROW_Y": {"STOCK": 0.7, "PHONE": -0.9},

    # render (leave these alone unless the topic needs landscape)
    "W": 1080, "H": 1920, "DPI": 100,
    "FPS": 30,
    "HOOK_SEC": 1.4, "BUILD_SEC": 8.5, "HOLD_SEC": 1.8,
    "ELEV_BASE": 18, "AZIM_START": -74, "AZIM_SWEEP": 12,
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
STOCK_CMAP = LinearSegmentedColormap.from_list(
    "stock", [THEME["BLUE"], THEME["CYAN"], THEME["GREEN"], THEME["YELLOW"]], N=256)

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
    p = os.path.join(CACHE_DIR, "aapl.csv")
    if not os.path.exists(p):
        import yfinance as yf
        log(f"[Data] fetching {CONFIG['TICKER']}")
        px = yf.download([CONFIG["TICKER"]], start="2007-01-01", interval="1d",
                         progress=False, auto_adjust=True)["Close"]
        if px.empty or px.iloc[:, 0].dropna().empty:
            raise SystemExit("no data returned; refusing to draw a synthetic tape")
        px.to_csv(p)
    log(f"[Data] cache {p}")
    return pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0].dropna()


# ------------------------------------------------------------- COMPUTE
def compute(px):
    last = float(px.iloc[-1])
    rows = []
    for day, model, price in LAUNCHES:
        buy = px[px.index >= day]
        assert len(buy) and (buy.index[0] - pd.Timestamp(day)).days <= 4, day
        rows.append({"date": pd.Timestamp(day), "model": model, "price": price,
                     "worth": price * last / float(buy.iloc[0])})
    return {"rows": rows, "price": np.array([r["price"] for r in rows], float),
            "worth": np.array([r["worth"] for r in rows]), "end": px.index[-1]}


# ------------------------------------------------------------ VALIDATE
def validate(d):
    """Every number that reaches a frame is asserted here, before any render."""
    spent, worth = d["price"].sum(), d["worth"].sum()
    first = d["rows"][0]
    assert len(d["rows"]) == 19
    assert 5_000 <= spent <= 15_000
    assert 4 * spent <= worth <= 40 * spent, f"worth ${worth:,.0f} outside 4x..40x"
    assert 20 * first["price"] <= first["worth"] <= 200 * first["price"]
    assert (d["worth"] > d["price"]).all(), "some phone beat its shares; the claim changes"

    figs = {
        "spent": f"${spent:,.0f}",
        "worth": f"${worth:,.0f}",
        "ratio": f"{worth / spent:.1f}x",
        "n": f"{len(d['rows'])}",
        "first_paid": f"${first['price']:,.0f}",
        "first_worth": f"${first['worth']:,.0f}",
        "first_x": f"{first['worth'] / first['price']:.0f}x",
        "last_model": d["rows"][-1]["model"],
        "last_worth": f"${d['rows'][-1]['worth']:,.0f}",
        "contract": "$199",
        "switch": "2015",
        "ticker": "AAPL",
        "start": first["date"].strftime("%b %Y"),
        "end": d["end"].strftime("%b %Y"),
    }
    with open(os.path.join(BASE_DIR, "figures.json"), "w") as fh:
        json.dump(figs, fh, indent=1)
    log(f"[validate] {figs}")
    d["figs"] = figs
    return figs


# -------------------------------------------------------------- RENDER
def _shade(c, k):
    c = np.array(to_rgba(c)); c[:3] = np.clip(c[:3] * k, 0, 1); return c


def _slab(polys, cols, x, y, w, dep, z0, z1, col, screen=None, top_screen=False):
    """Box from z0 to z1 with an optional dark screen inset on the front face
    (standing phone) or on the top face (phone lying flat)."""
    x0, x1, y0, y1 = x - w / 2, x + w / 2, y - dep / 2, y + dep / 2
    faces = [([(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)], 1.0),
             ([(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)], 0.7),
             ([(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)], 0.55),
             ([(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)], 0.5),
             ([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], 1.3)]
    for f, k in faces:
        polys.append(f); cols.append(_shade(col, k))
    if screen is not None:
        mx, mz = 0.07 * w, 0.035 * (z1 - z0)
        e = y0 - 0.002
        polys.append([(x0 + mx, e, z0 + mz), (x1 - mx, e, z0 + mz),
                      (x1 - mx, e, z1 - mz), (x0 + mx, e, z1 - mz)])
        cols.append(screen)
    if top_screen:
        mx, my = 0.07 * w, 0.05 * dep
        e = z1 + 0.0005
        polys.append([(x0 + mx, y0 + my, e), (x1 - mx, y0 + my, e),
                      (x1 - mx, y1 - my, e), (x0 + mx, y1 - my, e)])
        cols.append((0.05, 0.05, 0.07, 1))


def scene(ax, d, frac):
    c = CONFIG
    n_all = len(d["rows"])
    n = n_all if frac >= 1 else int(np.floor(frac * n_all))
    zmax = d["worth"].max() / 1e3
    polys, cols = [], []
    for i in range(n):
        x = float(i)
        zs, zp = d["worth"][i] / 1e3, d["price"][i] / 1e3
        col = STOCK_CMAP(0.1 + 0.9 * (zs / zmax) ** 0.6)
        _slab(polys, cols, x, c["ROW_Y"]["STOCK"], c["PHONE_W"], c["PHONE_T"], 0, zs,
              col, screen=_shade(col, 0.35))
        _slab(polys, cols, x, c["ROW_Y"]["PHONE"], c["PHONE_W"], c["PHONE_L"], 0,
              max(zp, 0.12), THEME["RED"], top_screen=True)
    if polys:
        ax.add_collection3d(Poly3DCollection(polys, facecolors=cols,
                                             edgecolors=(0, 0, 0, 0.45), linewidths=0.35),
                             autolim=False)
    ax.plot([-0.6, n_all - 0.4], [c["ROW_Y"]["STOCK"]] * 2, [0, 0],
            color=THEME["TEXT_DIM"], lw=0.6, alpha=0.5)
    ax.set_xlim(-0.6, n_all - 0.4); ax.set_ylim(-1.8, 1.4); ax.set_zlim(0, zmax * 1.04)
    return n - 1 if n else None


def style_axes(ax):
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
        pane.line.set_color((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def scene_labels(ax, d, last, fs=11):
    """Projected to 2D so the slabs cannot depth-sort the labels away."""
    from mpl_toolkits.mplot3d import proj3d
    M = ax.get_proj()
    c, figs = CONFIG, d["figs"]
    n_all = len(d["rows"])

    def put(x, y, z, s, **kw):
        x2, y2, _ = proj3d.proj_transform(x, y, z, M)
        ax.text2D(x2, y2, s, transform=ax.transData, zorder=1e6, ha="center",
                  va="center", **kw)

    for i in (0, 8, 18):
        put(i, c["ROW_Y"]["PHONE"] - 1.05, 0, f"{d['rows'][i]['date'].year}",
            color=THEME["TEXT_DIM"], fontsize=fs - 1, family=THEME["MONO"])
    if last is None:
        return
    done = last == n_all - 1
    z0 = d["worth"][0] / 1e3
    put(2.4, c["ROW_Y"]["STOCK"], z0 * 0.9,
        f"IN AAPL\n{figs['first_worth']}",
        color=THEME["YELLOW"], fontsize=fs + 3, fontweight="bold", family=THEME["MONO"])
    if done:
        put(n_all - 3, c["ROW_Y"]["PHONE"], z0 * 0.24,
            f"PHONES\n{figs['spent']}",
            color=THEME["RED"], fontsize=fs + 3, fontweight="bold", family=THEME["MONO"],
            bbox=dict(facecolor="black", alpha=0.8, edgecolor=THEME["RED"], lw=1.0,
                      boxstyle="round,pad=0.35"))


def text_block(fig, figs, scale=1.0):
    L, s = LAYOUT, scale
    fig.text(0.5, L["TITLE"], "EVERY iPHONE VS APPLE STOCK", ha="center",
             fontsize=27 * s, fontweight="bold", color=THEME["TEXT"],
             family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             f"the launch price of every new iPhone, put into {figs['ticker']} on launch day",
             ha="center", fontsize=12 * s, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.5, L["KEY"],
             f"{figs['n']} launches  ·  {figs['start']} to {figs['end']}  ·  red = the phone",
             ha="center", fontsize=10 * s, color=THEME["TEXT_DIM"], family=THEME["FONT"])
    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10 * s,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])
    fig.text(0.5, L["BIG"], f"PAID {figs['spent']}  ·  NOW {figs['worth']}",
             ha="center", fontsize=23 * s, color=THEME["YELLOW"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             f"first iPhone {figs['first_paid']} → {figs['first_worth']}  ·  dividends reinvested",
             ha="center", fontsize=11.5 * s, color=THEME["TEXT_DIM"],
             family=THEME["MONO"])


def note(fig, d, last, scale=1.0):
    if last is None:
        return
    r = d["rows"][last]
    fig.text(0.5, LAYOUT["NOTE"], f"{r['model']}  ·  {r['date'].year}",
             ha="center", fontsize=12 * scale, color=THEME["TEXT"],
             family=THEME["MONO"])


def render_frame(idx, total, d, out_path):
    c, figs = CONFIG, d["figs"]
    n_hook = int(c["FPS"] * c["HOOK_SEC"])
    n_build = int(c["FPS"] * c["BUILD_SEC"])
    if idx < n_hook:
        frac = 1 / len(d["rows"])          # the first iPhone already stands
    elif idx < n_hook + n_build:
        frac = (idx - n_hook + 1) / n_build
    else:
        frac = 1.0
    frac = max(frac, 1 / len(d["rows"]))
    t = idx / max(total - 1, 1)

    fig = plt.figure(figsize=(c["W"] / c["DPI"], c["H"] / c["DPI"]),
                     dpi=c["DPI"], facecolor=THEME["BG"])
    text_block(fig, figs)
    ax = fig.add_axes(LAYOUT["AXES"], projection="3d", facecolor=THEME["BG"])
    last = scene(ax, d, frac)
    note(fig, d, last)
    ax.view_init(elev=c["ELEV_BASE"] + 4 * np.sin(t * np.pi),
                 azim=c["AZIM_START"] + c["AZIM_SWEEP"] * t)
    ax.set_box_aspect((1.6, 0.5, 1.5), zoom=c["ZOOM"])
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
