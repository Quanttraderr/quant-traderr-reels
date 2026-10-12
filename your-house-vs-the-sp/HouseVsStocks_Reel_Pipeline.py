"""
HouseVsStocks_Reel_Pipeline.py
==============================
Reel 35 for @quant.traderr, "Your House vs the S&P". The most common belief in
personal finance: a house is the best investment you will ever make. This reel
puts $100,000 into the average US home in January 1987 and $100,000 into the
S&P 500 the same month, and lets both run to July 2026.

THE DATA
    S&P/Case-Shiller U.S. National Home Price Index, not seasonally adjusted
    (FRED series CSUSHPINSA), monthly, Jan 1987 to Jul 2026, fetched from
    fred.stlouisfed.org without an API key and kept in _cache/.
    ^GSPC daily closes from Yahoo (price index, dividends NOT included), last
    close of each matching month. Real data throughout. Nothing is simulated.

THE MATHS
    house_t  = 100,000 * HPI_t / HPI_Jan1987
    stocks_t = 100,000 * SPX_t / SPX_Jan1987      (month-end close)
    One bar per January from 1987 to 2026, plus Jul 2026, the latest HPI month.

    In 3D: a street. The front row is houses, one per year, a box with a gable
    roof, its eaves at the house value. The back row is towers, one per year,
    their roofs at the stock value. Both on the SAME linear dollar scale, so
    the gap is the real gap and not a log trick. Houses are red, the losing
    side; towers are coloured by their own height.

THE CLAIM THE VISUAL MAKES
    As a pure price investment, the same $100,000 grew several times more in
    the index than in the average house, even without counting dividends.

WHAT THE VISUAL IS NOT
    Not a verdict on owning a home. A house is also a place to live, so it
    saves rent, and it is almost always bought with a mortgage, so a 20% down
    payment carries the whole price gain; neither is in these bars. Nor are
    property tax, upkeep, insurance and transaction costs, which cut the house
    the other way, or dividends, which lift the stocks. The index is a
    national average, not anyone's actual house or city.

RUN
    python HouseVsStocks_Reel_Pipeline.py --smoke   # 3 frames -> temp_frames_smoke/
    python HouseVsStocks_Reel_Pipeline.py           # full render -> topic.mp4

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
matplotlib.rcParams["text.parse_math"] = False   # "$529k ... $2.73M" is money, not maths
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "_cache")

CONFIG = {
    "TICKER": "^GSPC",
    "HPI_URL": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CSUSHPINSA",
    "START": "1987-01",
    "STAKE": 100_000,
    "BAR_W": 0.62,
    "ROW_Y": {"HOUSE": -0.9, "STOCK": 0.9},
    "DEPTH": {"HOUSE": 1.0, "STOCK": 0.9},
    "ROOF": 0.11,

    # render (leave these alone unless the topic needs landscape)
    "W": 1080, "H": 1920, "DPI": 100,
    "FPS": 30,
    "HOOK_SEC": 1.4, "BUILD_SEC": 8.5, "HOLD_SEC": 1.8,
    "ELEV_BASE": 12, "AZIM_START": -62, "AZIM_SWEEP": 16,
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
TOWER_CMAP = LinearSegmentedColormap.from_list(
    "tower", [THEME["BLUE"], THEME["CYAN"], THEME["GREEN"], THEME["YELLOW"]], N=256)

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
    """FRED and Yahoo once, then the CSV cache. Delete _cache/ to refetch.
    A failed fetch stops the run; nothing synthetic is ever drawn."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    hp = os.path.join(CACHE_DIR, "CSUSHPINSA.csv")
    if not os.path.exists(hp):
        import urllib.request
        log("[Data] fetching CSUSHPINSA from FRED")
        try:
            with urllib.request.urlopen(CONFIG["HPI_URL"], timeout=60) as r:
                raw = r.read()
        except Exception as e:
            raise SystemExit(f"FRED fetch failed ({e}); refusing to draw a synthetic series")
        if b"CSUSHPINSA" not in raw[:200]:
            raise SystemExit("FRED returned something that is not the HPI csv")
        with open(hp, "wb") as fh:
            fh.write(raw)
    sp = os.path.join(CACHE_DIR, "gspc.csv")
    if not os.path.exists(sp):
        import yfinance as yf
        log(f"[Data] fetching {CONFIG['TICKER']}")
        px = yf.download([CONFIG["TICKER"]], period="max", interval="1d",
                         progress=False, auto_adjust=True)["Close"]
        if px.empty or px.iloc[:, 0].dropna().empty:
            raise SystemExit("no data returned; refusing to draw a synthetic tape")
        px.to_csv(sp)
    log(f"[Data] cache {hp}, {sp}")
    hpi = pd.read_csv(hp, index_col=0, parse_dates=True, na_values=["."]).iloc[:, 0]
    assert hpi.isna().sum() == 0, "HPI has missing months"
    spx = pd.read_csv(sp, index_col=0, parse_dates=True).iloc[:, 0].dropna()
    return hpi.astype(float), spx


# ------------------------------------------------------------- COMPUTE
def compute(data):
    hpi, spx = data
    hpi = hpi.loc[CONFIG["START"]:]
    last = hpi.index[-1]
    months = [pd.Timestamp(f"{y}-01-01") for y in range(hpi.index[0].year, last.year + 1)]
    if months[-1] != last:
        months.append(last)
    spm = spx.resample("ME").last()                      # month-end close
    sp_at = lambda m: float(spm.loc[m + pd.offsets.MonthEnd(0)])
    k = CONFIG["STAKE"]
    house = np.array([k * hpi.loc[m] / hpi.loc[months[0]] for m in months])
    stock = np.array([k * sp_at(m) / sp_at(months[0]) for m in months])
    # the monthly path, for the drawdown line in the caption
    hm = hpi / hpi.iloc[0]
    house_dd = float((hm / hm.cummax() - 1).min())
    sm = spm.loc[months[0] + pd.offsets.MonthEnd(0):last + pd.offsets.MonthEnd(0)]
    stock_dd = float((sm / sm.cummax() - 1).min())
    return {"months": months, "house": house, "stock": stock,
            "house_dd": house_dd, "stock_dd": stock_dd}


# ------------------------------------------------------------ VALIDATE
def validate(d):
    """Every number that reaches a frame is asserted here, before any render."""
    h, s = float(d["house"][-1]), float(d["stock"][-1])
    k = CONFIG["STAKE"]
    assert len(d["months"]) >= 40
    assert 2 * k <= h <= 15 * k, f"house ${h:,.0f} outside 2x..15x"
    assert 5 * k <= s <= 80 * k, f"stocks ${s:,.0f} outside 5x..80x"
    assert s > h, "the house beat the index; the reel's claim is gone"
    assert -0.5 <= d["house_dd"] < 0 and -0.7 <= d["stock_dd"] < 0

    figs = {
        "stake": f"${k:,.0f}",
        "house_usd": f"${h:,.0f}",
        "stock_usd": f"${s:,.0f}",
        "house_k": f"${h / 1e3:,.0f}k",
        "stock_m": f"${s / 1e6:,.2f}M",
        "house_x": f"{h / k:.1f}x",
        "stock_x": f"{s / k:.1f}x",
        "ratio": f"{s / h:.1f}x",
        "house_dd": f"{d['house_dd'] * 100:.0f}%",
        "stock_dd": f"{d['stock_dd'] * 100:.0f}%",
        "years": f"{(d['months'][-1] - d['months'][0]).days / 365.25:.0f}",
        "index": "S&P 500",
        "down": "20%",
        "start": d["months"][0].strftime("%b %Y"),
        "end": d["months"][-1].strftime("%b %Y"),
    }
    with open(os.path.join(BASE_DIR, "figures.json"), "w") as fh:
        json.dump(figs, fh, indent=1)
    log(f"[validate] {figs}")
    d["figs"] = figs
    return figs


# -------------------------------------------------------------- RENDER
def _shade(c, k):
    c = np.array(to_rgba(c)); c[:3] = np.clip(c[:3] * k, 0, 1); return c


def _box(polys, cols, x, y, w, dep, z, col):
    """Front, side, back and top of a box from the floor to z; the camera sits
    front-left, so the right side is the shaded one."""
    x0, x1, y0, y1 = x - w / 2, x + w / 2, y - dep / 2, y + dep / 2
    faces = [([(x0, y0, 0), (x1, y0, 0), (x1, y0, z), (x0, y0, z)], 1.0),
             ([(x0, y0, 0), (x0, y1, 0), (x0, y1, z), (x0, y0, z)], 0.7),
             ([(x1, y0, 0), (x1, y1, 0), (x1, y1, z), (x1, y0, z)], 0.55),
             ([(x0, y1, 0), (x1, y1, 0), (x1, y1, z), (x0, y1, z)], 0.5),
             ([(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)], 1.3)]
    for f, k in faces:
        polys.append(f); cols.append(_shade(col, k))


def _roof(polys, cols, x, y, w, dep, z, rh, col):
    """Gable roof, ridge along y, sitting on the eaves at z."""
    x0, x1, y0, y1 = x - w / 2, x + w / 2, y - dep / 2, y + dep / 2
    polys += [[(x0, y0, z), (x, y0, z + rh), (x, y1, z + rh), (x0, y1, z)],
              [(x1, y0, z), (x, y0, z + rh), (x, y1, z + rh), (x1, y1, z)],
              [(x0, y0, z), (x1, y0, z), (x, y0, z + rh)],
              [(x0, y1, z), (x1, y1, z), (x, y1, z + rh)]]
    cols += [_shade(col, 1.35), _shade(col, 0.75), _shade(col, 1.1), _shade(col, 0.6)]


def scene(ax, d, frac):
    c = CONFIG
    n_all = len(d["months"])
    n = n_all if frac >= 1 else max(1, int(np.floor(frac * n_all)))
    zmax = d["stock"].max() / 1e6
    polys, cols = [], []
    for i in range(n):
        x = float(i)
        zs, zh = d["stock"][i] / 1e6, d["house"][i] / 1e6
        _box(polys, cols, x, c["ROW_Y"]["STOCK"], c["BAR_W"], c["DEPTH"]["STOCK"], zs,
             TOWER_CMAP(0.08 + 0.92 * zs / zmax))
        _box(polys, cols, x, c["ROW_Y"]["HOUSE"], c["BAR_W"], c["DEPTH"]["HOUSE"], zh,
             THEME["RED"])
        _roof(polys, cols, x, c["ROW_Y"]["HOUSE"], c["BAR_W"] + 0.12,
              c["DEPTH"]["HOUSE"] + 0.1, zh, c["ROOF"], THEME["RED"])
    if polys:
        ax.add_collection3d(Poly3DCollection(polys, facecolors=cols,
                                             edgecolors=(0, 0, 0, 0.45), linewidths=0.35),
                             autolim=False)
    for y in c["ROW_Y"].values():
        ax.plot([-0.6, n_all - 0.4], [y, y], [0, 0], color=THEME["TEXT_DIM"], lw=0.6, alpha=0.5)
    ax.set_xlim(-0.6, n_all - 0.4); ax.set_ylim(-1.6, 1.6); ax.set_zlim(0, zmax * 1.04)
    return max(n - 1, 0) if n else None


def style_axes(ax):
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
        pane.line.set_color((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def scene_labels(ax, d, last, fs=11):
    """Projected to 2D so the boxes cannot depth-sort the labels away."""
    from mpl_toolkits.mplot3d import proj3d
    M = ax.get_proj()
    c, figs = CONFIG, d["figs"]
    n_all = len(d["months"])

    def put(x, y, z, s, **kw):
        x2, y2, _ = proj3d.proj_transform(x, y, z, M)
        ax.text2D(x2, y2, s, transform=ax.transData, zorder=1e6, ha="center",
                  va="center", **kw)

    for yr_i in range(0, n_all - 1, 10):
        put(yr_i, c["ROW_Y"]["HOUSE"] - 0.95, 0, f"{d['months'][yr_i].year}",
            color=THEME["TEXT_DIM"], fontsize=fs - 1, family=THEME["MONO"])
    if last is None:
        return
    done = last == n_all - 1
    zs = d["stock"][last] / 1e6
    zh = d["house"][last] / 1e6
    put(last - 2.5, c["ROW_Y"]["STOCK"], zs + 0.17,
        f"S&P 500\n{figs['stock_m']}" if done else "S&P 500",
        color=THEME["YELLOW"], fontsize=fs + 3, fontweight="bold", family=THEME["MONO"])
    put(last - 3, c["ROW_Y"]["HOUSE"] - 0.6, zh + c["ROOF"] + 0.32,
        f"HOUSE\n{figs['house_k']}" if done else "HOUSE",
        color=THEME["RED"], fontsize=fs + 3, fontweight="bold", family=THEME["MONO"],
        bbox=dict(facecolor="black", alpha=0.8, edgecolor=THEME["RED"], lw=1.0,
                  boxstyle="round,pad=0.35"))


def text_block(fig, figs, scale=1.0):
    L, s = LAYOUT, scale
    fig.text(0.5, L["TITLE"], "YOUR HOUSE VS THE S&P", ha="center",
             fontsize=27 * s, fontweight="bold", color=THEME["TEXT"],
             family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             f"{figs['stake']} in the average US home or the S&P 500  |  "
             f"{figs['start']} to {figs['end']}",
             ha="center", fontsize=12 * s, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.5, L["KEY"],
             "one bar per year  ·  same linear dollar scale  ·  red = house",
             ha="center", fontsize=10 * s, color=THEME["TEXT_DIM"], family=THEME["FONT"])
    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10 * s,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])
    fig.text(0.5, L["BIG"], f"HOUSE {figs['house_x']}  ·  STOCKS {figs['stock_x']}",
             ha="center", fontsize=23 * s, color=THEME["YELLOW"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             f"Case-Shiller national index  ·  S&P price only, no dividends",
             ha="center", fontsize=11.5 * s, color=THEME["TEXT_DIM"],
             family=THEME["MONO"])


def note(fig, d, last, scale=1.0):
    if last is None:
        return
    fig.text(0.5, LAYOUT["NOTE"], d["months"][last].strftime("%b %Y").upper(),
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
    ax.set_box_aspect((1.6, 0.45, 1.5), zoom=c["ZOOM"])
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
