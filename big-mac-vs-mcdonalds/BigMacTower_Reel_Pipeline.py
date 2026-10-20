"""
BigMacTower_Reel_Pipeline.py
============================
Reel 39 for @quant.traderr, "Big Mac vs McDonald's". The habit it takes apart:
the small weekly treat that "doesn't count". One Big Mac every week since
April 2000, at the real US price of the day, against the same dollars put
into McDonald's stock (MCD) every week instead.

THE DATA
    US Big Mac price from The Economist's Big Mac index (public on GitHub,
    output-data/big-mac-full-index.csv, column local_price for USA), one
    reading or two per year since Apr 2000, carried forward between readings.
    MCD weekly closes (Friday) from Yahoo, auto-adjusted for splits and
    dividends, so dividends are reinvested. Both kept in _cache/. Real data
    throughout. Nothing is simulated.

THE MATHS
    each week w:  price_w  = latest Big Mac price on or before w
                  shares_w = price_w / MCD_w
    paid  = sum price_w           worth = sum shares_w * MCD_today
    Grouped by calendar year for the layers.

    In 3D: two towers on the same linear dollar scale. On the left a stack of
    burgers, one burger per five years, its thickness the money spent on Big
    Macs in those years; bun, patty and cheese are just the burger's colours, the
    patty in the losing red. On the right a stack of discs, one per year, its
    thickness what that year's burger money is worth today in MCD, coloured by
    year.

THE CLAIM THE VISUAL MAKES
    The money for one Big Mac a week since 2000 would be worth about five
    times as much today in the stock of the company that sold the burgers.

WHAT THE VISUAL IS NOT
    Not advice to stop eating: food is not an investment, and hindsight picks
    a stock that did well. The Economist samples the price once or twice a year
    in one market, so the real price you paid varied by city and by month.
    No taxes or trading costs; weekly buying of fractional shares is assumed.

RUN
    python BigMacTower_Reel_Pipeline.py --smoke   # 3 frames -> temp_frames_smoke/
    python BigMacTower_Reel_Pipeline.py           # full render -> topic.mp4

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
matplotlib.rcParams["text.parse_math"] = False   # "$26,742" is money, not maths
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "_cache")

CONFIG = {
    "TICKER": "MCD",
    "BIGMAC_URL": "https://raw.githubusercontent.com/TheEconomist/big-mac-data/"
                  "master/output-data/big-mac-full-index.csv",
    "START": "2000-04-01",
    "X": {"BURGER": -1.0, "STOCK": 1.0},
    "R": 0.72, "SIDES": 40,
    "BOX": (1.0, 0.75, 1.75),

    # render (leave these alone unless the topic needs landscape)
    "W": 1080, "H": 1920, "DPI": 100,
    "FPS": 30,
    "HOOK_SEC": 1.4, "BUILD_SEC": 8.5, "HOLD_SEC": 1.8,
    "ELEV_BASE": 16, "AZIM_START": -78, "AZIM_SWEEP": 20,
    "ZOOM": 1.02,
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
    "stock", [THEME["YELLOW"], THEME["GREEN"], THEME["CYAN"], THEME["BLUE"]], N=256)
# bottom bun, patty, cheese, lettuce, top bun: (share of the layer, colour)
BURGER = [(0.26, "#d9822b"), (0.34, THEME["RED"]), (0.08, THEME["YELLOW"]),
          (0.07, "#2fbf4a"), (0.25, "#e8953a")]

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
    """GitHub and Yahoo once, then the CSV cache. Delete _cache/ to refetch.
    A failed fetch stops the run; nothing synthetic is ever drawn."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    bp = os.path.join(CACHE_DIR, "big-mac-full-index.csv")
    if not os.path.exists(bp):
        import urllib.request
        log("[Data] fetching the Big Mac index")
        try:
            with urllib.request.urlopen(CONFIG["BIGMAC_URL"], timeout=60) as r:
                raw = r.read()
        except Exception as e:
            raise SystemExit(f"Big Mac fetch failed ({e}); refusing to draw a synthetic series")
        if not raw.startswith(b"date,iso_a3"):
            raise SystemExit("GitHub returned something that is not the Big Mac csv")
        with open(bp, "wb") as fh:
            fh.write(raw)
    mp = os.path.join(CACHE_DIR, "mcd.csv")
    if not os.path.exists(mp):
        import yfinance as yf
        log(f"[Data] fetching {CONFIG['TICKER']}")
        px = yf.download([CONFIG["TICKER"]], start="2000-01-01", interval="1d",
                         progress=False, auto_adjust=True)["Close"]
        if px.empty or px.iloc[:, 0].dropna().empty:
            raise SystemExit("no data returned; refusing to draw a synthetic tape")
        px.to_csv(mp)
    log(f"[Data] cache {bp}, {mp}")
    bm = pd.read_csv(bp, parse_dates=["date"])
    us = bm[bm["iso_a3"] == "USA"].set_index("date")["local_price"].astype(float).sort_index()
    mcd = pd.read_csv(mp, index_col=0, parse_dates=True).iloc[:, 0].dropna()
    return us, mcd


# ------------------------------------------------------------- COMPUTE
def compute(data):
    us, mcd = data
    wk = mcd.resample("W-FRI").last().dropna().loc[CONFIG["START"]:]
    price = us.reindex(wk.index, method="ffill")
    assert price.isna().sum() == 0, "a week before the first Big Mac reading"
    shares = price / wk
    last = float(mcd.iloc[-1])
    df = pd.DataFrame({"paid": price, "worth": shares * last})
    yr = df.groupby(df.index.year).sum()
    return {"years": yr.index.to_numpy(), "paid_y": yr["paid"].to_numpy(),
            "worth_y": yr["worth"].to_numpy(), "paid": float(df["paid"].sum()),
            "worth": float(df["worth"].sum()), "weeks": len(df),
            "p0": float(price.iloc[0]), "p1": float(us.iloc[-1]),
            "start": wk.index[0], "end": mcd.index[-1]}


# ------------------------------------------------------------ VALIDATE
def validate(d):
    """Every number that reaches a frame is asserted here, before any render."""
    assert 1300 <= d["weeks"] <= 1500
    assert 3_000 <= d["paid"] <= 9_000, f"paid ${d['paid']:,.0f} outside band"
    assert 2 * d["paid"] <= d["worth"] <= 15 * d["paid"], f"worth ${d['worth']:,.0f}"
    assert 1.5 <= d["p0"] <= 3.5 and 4 <= d["p1"] <= 9
    assert len(d["years"]) == d["end"].year - 2000 + 1

    figs = {
        "paid": f"${d['paid']:,.0f}",
        "worth": f"${d['worth']:,.0f}",
        "ratio": f"{d['worth'] / d['paid']:.1f}x",
        "weeks": f"{d['weeks']:,}",
        "p0": f"${d['p0']:.2f}",
        "p1": f"${d['p1']:.2f}",
        "price_x": f"{d['p1'] / d['p0']:.1f}x",
        "ticker": "MCD",
        "start": d["start"].strftime("%b %Y"),
        "end": d["end"].strftime("%b %Y"),
        "first_year": str(d["years"][0]),
        "first_year_paid": f"${d['paid_y'][0]:,.0f}",
        "first_year_worth": f"${d['worth_y'][0]:,.0f}",
    }
    with open(os.path.join(BASE_DIR, "figures.json"), "w") as fh:
        json.dump(figs, fh, indent=1)
    log(f"[validate] {figs}")
    d["figs"] = figs
    return figs


# -------------------------------------------------------------- RENDER
def _shade(c, k):
    c = np.array(to_rgba(c)); c[:3] = np.clip(c[:3] * k, 0, 1); return c


def _disc(polys, cols, x0, z0, z1, col, top=True, r0=1.0, r1=1.0):
    """Cylinder (or cone) slice from z0 to z1 around (x0, 0), radius r0*R at
    the bottom and r1*R at the top, lit from the front left."""
    c = CONFIG
    th = np.linspace(0, 2 * np.pi, c["SIDES"] + 1)
    R = c["R"]
    bx, by = x0 + r0 * R * np.cos(th), r0 * R * np.sin(th)
    tx, ty = x0 + r1 * R * np.cos(th), r1 * R * np.sin(th)
    for j in range(c["SIDES"]):
        mid = (th[j] + th[j + 1]) / 2
        k = 0.5 + 0.65 * max(np.cos(mid - np.deg2rad(215)), 0) + 0.12
        polys.append([(bx[j], by[j], z0), (bx[j + 1], by[j + 1], z0),
                      (tx[j + 1], ty[j + 1], z1), (tx[j], ty[j], z1)])
        cols.append(_shade(col, k))
    if top:
        polys.append([(tx[j], ty[j], z1) for j in range(c["SIDES"])])
        cols.append(_shade(col, 1.25))


def _burger(polys, cols, x0, z0, h, top_open):
    """One burger of height h: bun, patty, cheese, lettuce, domed top bun."""
    z = z0
    for share, col in BURGER[:-1]:
        _disc(polys, cols, x0, z, z + share * h, col, top=False, r0=0.97, r1=0.97)
        z += share * h
    share, col = BURGER[-1]
    dome = [(0.0, 0.45, 1.0, 1.0), (0.45, 0.75, 1.0, 0.9), (0.75, 1.0, 0.9, 0.6)]
    for a, b, ra, rb in dome:
        _disc(polys, cols, x0, z + a * share * h, z + b * share * h, col,
              top=(b == 1.0), r0=ra, r1=rb)


def scene(ax, d, frac):
    c = CONFIG
    n_all = len(d["years"])
    n = n_all if frac >= 1 else max(1, int(np.ceil(frac * n_all)))
    zmax = d["worth"] / 1e3
    polys, cols = [], []
    # one burger per five calendar years, so a burger is big enough to read
    zb = 0.0
    yrs = d["years"][:n]
    for g in sorted(set((yrs // 5) * 5)):
        h = d["paid_y"][:n][(yrs // 5) * 5 == g].sum() / 1e3
        _burger(polys, cols, c["X"]["BURGER"], zb, h, True)
        zb += h
    zs = 0.0
    for i in range(n):
        h = d["worth_y"][i] / 1e3
        _disc(polys, cols, c["X"]["STOCK"], zs, zs + h,
              STOCK_CMAP(i / (n_all - 1)), top=(i == n - 1))
        zs += h
    # painter's order: far side faces first, so draw tallest stack last
    ax.add_collection3d(Poly3DCollection(polys, facecolors=cols,
                                         edgecolors=(0, 0, 0, 0.18), linewidths=0.2),
                        autolim=False)
    ax.set_xlim(-2.1, 2.1); ax.set_ylim(-1.6, 1.6); ax.set_zlim(0, zmax * 1.03)
    return n - 1, zb, zs


def style_axes(ax):
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
        pane.line.set_color((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def scene_labels(ax, d, state, fs=11):
    """Projected to 2D so the stacks cannot depth-sort the labels away."""
    from mpl_toolkits.mplot3d import proj3d
    M = ax.get_proj()
    c, figs = CONFIG, d["figs"]
    last, zb, zs = state

    def put(x, y, z, s, **kw):
        x2, y2, _ = proj3d.proj_transform(x, y, z, M)
        ax.text2D(x2, y2, s, transform=ax.transData, zorder=1e6, ha="center",
                  va="center", **kw)

    done = last == len(d["years"]) - 1
    zmax = d["worth"] / 1e3
    put(c["X"]["BURGER"], 0, zb + zmax * 0.09,
        f"BIG MACS\n{figs['paid']}" if done else "BIG MACS",
        color=THEME["RED"], fontsize=fs + 4, fontweight="bold", family=THEME["MONO"],
        bbox=dict(facecolor="black", alpha=0.8, edgecolor=THEME["RED"], lw=1.0,
                  boxstyle="round,pad=0.35"))
    put(c["X"]["STOCK"], 0, zs + zmax * 0.07,
        f"MCD STOCK\n{figs['worth']}" if done else "MCD STOCK",
        color=THEME["YELLOW"], fontsize=fs + 4, fontweight="bold", family=THEME["MONO"])


def text_block(fig, figs, scale=1.0):
    L, s = LAYOUT, scale
    fig.text(0.5, L["TITLE"], "BIG MAC VS McDONALD'S", ha="center",
             fontsize=28 * s, fontweight="bold", color=THEME["TEXT"],
             family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             f"one Big Mac a week since {figs['start']}, or the same money in {figs['ticker']} stock",
             ha="center", fontsize=12 * s, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.5, L["KEY"],
             f"one burger per 5 years  ·  same dollar scale  ·  {figs['weeks']} weeks",
             ha="center", fontsize=10 * s, color=THEME["TEXT_DIM"], family=THEME["FONT"])
    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10 * s,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])
    fig.text(0.5, L["BIG"], f"EATEN {figs['paid']}  ·  STOCK {figs['worth']}",
             ha="center", fontsize=23 * s, color=THEME["YELLOW"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             f"Big Mac {figs['p0']} → {figs['p1']}  ·  dividends reinvested",
             ha="center", fontsize=11.5 * s, color=THEME["TEXT_DIM"],
             family=THEME["MONO"])


def note(fig, d, state, scale=1.0):
    last = state[0]
    fig.text(0.5, LAYOUT["NOTE"], str(d["years"][last]),
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
    state = scene(ax, d, frac)
    note(fig, d, state)
    ax.view_init(elev=c["ELEV_BASE"] + 4 * np.sin(t * np.pi),
                 azim=c["AZIM_START"] + c["AZIM_SWEEP"] * t)
    ax.set_box_aspect(c["BOX"], zoom=c["ZOOM"])
    style_axes(ax)
    scene_labels(ax, d, state)

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
