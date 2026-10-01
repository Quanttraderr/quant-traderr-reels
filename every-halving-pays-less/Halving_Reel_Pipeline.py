"""
Halving_Reel_Pipeline.py
========================
Reel 21 for @quant.traderr, "Every Halving Pays Less". Bitcoin from each
halving day to the highest close before the next halving, three cycles side
by side.

THE DATA
    BTC-USD daily closes from Yahoo, Sep 2014 to Sep 2026. Halving dates are
    the block-height dates: 9 Jul 2016, 11 May 2020, 19 Apr 2024. The Nov 2012
    halving is NOT included because Yahoo's BTC series starts in Sep 2014.
    Real data throughout. Nothing here is simulated.

THE MATHS
    For each cycle, m(t) = close_t / close_halving for every day from the
    halving to the day before the next halving (the open 2024 cycle runs to
    the last close in the data). The cycle peak is max m(t). The readout
    numbers are exact daily closes.

    In 3D: one surface of revolution per cycle, a goblet. The vertical axis is
    days since the halving, the same scale for all three. The radius at each
    height is 0.12 + 0.70 * log10(m), so a goblet bulges where the price
    multiple was high and pinches where it fell back. Colour carries the same
    log multiple. For display only the multiple is smoothed with a 15-day
    centred mean, so the rim is not a saw blade; the white ring marks the
    exact peak day. The build grows all three goblets on one clock, so at
    every frame they are compared at the same day after their halving.

THE CLAIM THE VISUAL MAKES
    Each halving cycle so far has paid a smaller peak multiple than the one
    before: +2,895%, then +750%, then +95% (so far).

WHAT THE VISUAL IS NOT
    Not a forecast and not a model of the halving. Three cycles are three data
    points, the peaks are picked with hindsight, and the 2024 cycle is still
    open, so its peak can still rise. The shrinking multiple also fits a
    simpler story: a larger asset needs more money to move by the same
    percentage. Correlation with the halving date is not proof that the
    halving drives the price.

RUN
    python Halving_Reel_Pipeline.py --smoke   # 3 frames
    python Halving_Reel_Pipeline.py           # full render -> topic.mp4

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
from mpl_toolkits.mplot3d import proj3d

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "_cache")

CONFIG = {
    "START": "2014-09-17",
    "END": "2026-09-30",
    "HALVINGS": ["2016-07-09", "2020-05-11", "2024-04-19"],
    "SMOOTH": 15,
    "R0": 0.12,
    "RK": 0.70,
    "SPACING": 2.75,
    "N_THETA": 48,

    # render
    "W": 1080, "H": 1920, "DPI": 100,
    "FPS": 30,
    "HOOK_SEC": 1.4, "BUILD_SEC": 8.5, "HOLD_SEC": 1.8,
    "ELEV_BASE": 16, "AZIM_START": -100, "AZIM_SWEEP": 40,
    "ZOOM": 1.22,
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

LOG_HI = 1.5                     # colour scale top, log10 multiple (about 32x)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------- DATA
def fetch():
    """yfinance once, then a CSV cache. Delete _cache/ to refetch."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "btc.csv")
    if not os.path.exists(path):
        import yfinance as yf
        log("[Data] fetching BTC-USD")
        btc = yf.download("BTC-USD", start=CONFIG["START"], progress=False,
                          auto_adjust=True)["Close"].squeeze()
        if btc.empty:
            raise SystemExit("no data returned; refusing to draw a synthetic tape")
        btc.to_frame("BTC").to_csv(path)
    s = pd.read_csv(path, index_col=0, parse_dates=True)["BTC"]
    return s.loc[CONFIG["START"]:CONFIG["END"]].dropna()


# ------------------------------------------------------------- COMPUTE
def compute(btc):
    hs = [pd.Timestamp(h) for h in CONFIG["HALVINGS"]]
    cycles = []
    for k, h in enumerate(hs):
        end = hs[k + 1] - pd.Timedelta(days=1) if k + 1 < len(hs) else btc.index[-1]
        seg = btc.loc[h:end]
        m = seg / seg.iloc[0]
        days = (seg.index - seg.index[0]).days.values
        sm = m.rolling(CONFIG["SMOOTH"], center=True, min_periods=1).mean()
        cycles.append({"halving": h, "seg": seg, "m": m, "days": days,
                       "m_smooth": sm.values,
                       "peak": float(m.max()), "peak_date": m.idxmax(),
                       "peak_day": int((m.idxmax() - seg.index[0]).days),
                       "open": k + 1 == len(hs)})
    max_days = max(c["days"][-1] for c in cycles)
    return {"cycles": cycles, "max_days": max_days, "btc": btc}


# ------------------------------------------------------------ VALIDATE
def validate(d):
    """Every number that reaches a frame is asserted here, before any render."""
    cyc = d["cycles"]
    peaks = [c["peak"] - 1 for c in cyc]
    assert len(cyc) == 3
    assert all(0.2 < p < 200 for p in peaks), f"peak multiples out of band: {peaks}"
    assert peaks[0] > peaks[1] > peaks[2], "peaks do not shrink; the claim is gone"
    for c in cyc:
        assert c["m"].iloc[0] == 1.0
        assert c["seg"].index[0] - c["halving"] < pd.Timedelta(days=2), \
            "no close on the halving day"
        assert len(c["seg"]) > 300
    now = cyc[-1]["m"].iloc[-1] - 1
    assert -0.9 < now < 50

    figs = {}
    for k, c in enumerate(cyc, 1):
        figs[f"year_{k}"] = c["halving"].strftime("%Y")
        figs[f"peak_{k}"] = f"+{(c['peak'] - 1) * 100:,.0f}%"
        figs[f"peak_date_{k}"] = c["peak_date"].strftime("%b %Y")
        figs[f"peak_days_{k}"] = f"{c['peak_day']:,} days"
        figs[f"start_px_{k}"] = f"${c['seg'].iloc[0]:,.0f}"
        figs[f"peak_px_{k}"] = f"${c['seg'].max():,.0f}"
    figs["now_3"] = f"{now * 100:+.0f}%"
    figs["last_close"] = f"${cyc[-1]['seg'].iloc[-1]:,.0f}"
    figs["end"] = cyc[-1]["seg"].index[-1].strftime("%b %Y")
    figs["n_cycles"] = str(len(cyc))
    with open(os.path.join(BASE_DIR, "figures.json"), "w") as fh:
        json.dump(figs, fh, indent=1)
    log(f"[validate] {figs}")
    d["figs"] = figs
    return figs


# -------------------------------------------------------------- RENDER
def style_axes(ax):
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
        pane.line.set_color((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def _radius(m):
    c = CONFIG
    return np.maximum(c["R0"] + c["RK"] * np.log10(np.maximum(m, 1e-3)), 0.03)


def scene(ax, d, frac, rows=140):
    """Grow all goblets on one clock: day = frac * longest cycle."""
    c = CONFIG
    day_now = max(frac * d["max_days"], 8)
    zscale = 6.0 / d["max_days"]                  # whole height = 6 units
    th = np.linspace(0, 2 * np.pi, c["N_THETA"])
    for k, cy in enumerate(d["cycles"]):
        x0 = (k - 1) * c["SPACING"]
        # ghost of the finished goblet, so the hook already shows the race
        st = max(len(cy["days"]) // 40, 1)
        gR = _radius(cy["m_smooth"][::st])
        gT, gZ = np.meshgrid(th[::3], cy["days"][::st] * zscale)
        ax.plot_wireframe(x0 + gR[:, None] * np.cos(gT), gR[:, None] * np.sin(gT),
                          gZ, color=THEME["TEXT"], alpha=0.06, linewidth=0.4)
        keep = cy["days"] <= day_now
        if keep.sum() < 2:
            continue
        days = cy["days"][keep]
        ms = cy["m_smooth"][keep]
        step = max(len(days) // rows, 1)
        idx = np.r_[np.arange(0, len(days), step), len(days) - 1]
        dz, mz = days[idx] * zscale, ms[idx]
        R = _radius(mz)
        TH, Z = np.meshgrid(th, dz)
        X = x0 + R[:, None] * np.cos(TH)
        Y = R[:, None] * np.sin(TH)
        L = np.clip(np.log10(np.maximum(mz, 1e-3)) / LOG_HI, 0, 1)
        fc = CMAP(np.repeat(L[:, None], len(th), axis=1))
        fc[..., 3] = 0.88
        ax.plot_surface(X, Y, Z, facecolors=fc, linewidth=0, antialiased=True,
                        shade=True, rstride=1, cstride=1)
        ax.plot_wireframe(X, Y, Z, color=THEME["TEXT"], alpha=0.07, linewidth=0.3,
                          rstride=max(len(idx) // 25, 1), cstride=4)
        # base disc outline at the halving
        ax.plot(x0 + 1.3 * np.cos(th), 1.3 * np.sin(th), 0 * th,
                color=THEME["TEXT_DIM"], lw=0.6, alpha=0.5)
        if cy["peak_day"] <= day_now:
            rp = _radius(np.array([cy["peak"]]))[0]
            ax.plot(x0 + rp * np.cos(th), rp * np.sin(th),
                    np.full_like(th, cy["peak_day"] * zscale),
                    color="white", lw=1.8, alpha=0.95)
    span = c["SPACING"] + 1.35
    ax.set_xlim(-span, span); ax.set_ylim(-1.4, 1.4); ax.set_zlim(0, 6.3)
    return zscale


def scene_labels(ax, d, frac, zscale, fs=12):
    M = ax.get_proj()
    figs, c = d["figs"], CONFIG
    day_now = max(frac * d["max_days"], 8)

    def put(x, y, z, s, **kw):
        x2, y2, _ = proj3d.proj_transform(x, y, z, M)
        ax.text2D(x2, y2, s, transform=ax.transData, zorder=1e6, ha="center",
                  va="center", **kw)

    for k, cy in enumerate(d["cycles"], 1):
        x0 = (k - 2) * c["SPACING"]
        put(x0, 0, -0.55, figs[f"year_{k}"], color=THEME["TEXT_DIM"],
            fontsize=fs, family=THEME["MONO"], fontweight="bold")
        if cy["peak_day"] <= day_now:
            put(x0, 0, cy["peak_day"] * zscale + 0.45, figs[f"peak_{k}"],
                color=THEME["TEXT"], fontsize=fs + 2, family=THEME["FONT"],
                fontweight="bold")


def text_block(fig, figs, s=1.0):
    L = LAYOUT
    fig.text(0.5, L["TITLE"], "EVERY HALVING PAYS LESS", ha="center",
             fontsize=27 * s, fontweight="bold", color=THEME["TEXT"],
             family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             f"Bitcoin  |  halving day to cycle peak  |  "
             f"{figs['n_cycles']} cycles to {figs['end']}",
             ha="center", fontsize=13 * s, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.5, L["KEY"],
             "height = days since halving  ·  width = price multiple (log)  ·  ring = peak",
             ha="center", fontsize=10 * s, color=THEME["TEXT_DIM"], family=THEME["FONT"])
    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10 * s,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])
    fig.text(0.5, L["BIG"],
             f"{figs['peak_1']}  →  {figs['peak_2']}  →  {figs['peak_3']}",
             ha="center", fontsize=22 * s, color=THEME["ORANGE"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             f"peak gain after the {figs['year_1']}, {figs['year_2']} and "
             f"{figs['year_3']} halvings",
             ha="center", fontsize=12 * s, color=THEME["TEXT_DIM"], family=THEME["MONO"])


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
    zscale = scene(ax, d, frac)
    ax.view_init(elev=c["ELEV_BASE"] + 5 * np.sin(t * np.pi),
                 azim=c["AZIM_START"] + c["AZIM_SWEEP"] * t)
    ax.set_box_aspect((2 * (c["SPACING"] + 1.35) / 2.8, 1.0, 2.0), zoom=c["ZOOM"])
    style_axes(ax)
    scene_labels(ax, d, frac, zscale)
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


def load():
    d = compute(fetch())
    validate(d)                       # <- before a single frame is written
    return d


def main(smoke=False):
    c = CONFIG
    d = load()
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
