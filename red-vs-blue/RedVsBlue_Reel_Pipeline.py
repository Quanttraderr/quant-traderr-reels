"""
RedVsBlue_Reel_Pipeline.py
==========================
Reel 34 for @quant.traderr, "Red vs Blue". Many investors sell when the other
party wins the White House. This reel holds $1 in the S&P 500 only while one
party is in office and sits in cash the rest of the time, against $1 that
simply stays invested, from Hoover's inauguration in March 1929 to Sep 2026.

THE DATA
    ^GSPC daily closes from Yahoo (price index, dividends NOT included),
    Mar 1929 to 30 Sep 2026, and the 17 presidencies with their party and the
    date each took office. Real data throughout. Nothing here is simulated.

THE MATHS
    For presidency k from s_k to e_k:  m_k = P(e_k) / P(s_k)
    always      = P(end) / P(Mar 1929)              = product of every m_k
    blue only   = product of m_k over Democratic presidencies
    red only    = product of m_k over Republican presidencies
    Out of office, the party-only dollar sits in cash at 0%.

    In 3D: an arena. Every presidency is a curved block on a ring, its arc
    length is its time in office, its height is log10(m_k), so a block that
    rises gained and a block that sinks below the floor lost. Democrats fill
    the left half, Republicans the right, both in time order from the front of
    the ring to the back. Three columns in the middle compound the dollars:
    always invested, blue only, red only, also on log10.

THE CLAIM THE VISUAL MAKES
    Whichever party you back, sitting out the other side's years lost to
    staying invested.

WHAT THE VISUAL IS NOT
    Not a claim that presidents move markets. Red only is dragged down mostly
    by one crash, 1929 to 1933 under Hoover; start the clock in 1933 and the
    gap between the parties shrinks a lot. 17 presidencies is a tiny sample,
    Congress, the Fed and wars are ignored, dividends are left out (they would
    lift every line), and cash earns 0%, which is harsh on the party-only
    dollars before 1980 and harsher after.

RUN
    python RedVsBlue_Reel_Pipeline.py --smoke   # 3 frames -> temp_frames_smoke/
    python RedVsBlue_Reel_Pipeline.py           # full render -> topic.mp4

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
matplotlib.rcParams["text.parse_math"] = False   # "$132 ... $2.27" is money, not maths
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "_cache")

PRESIDENTS = [
    ("HOOVER", "R", "1929-03-04"), ("ROOSEVELT", "D", "1933-03-04"),
    ("TRUMAN", "D", "1945-04-12"), ("EISENHOWER", "R", "1953-01-20"),
    ("KENNEDY", "D", "1961-01-20"), ("JOHNSON", "D", "1963-11-22"),
    ("NIXON", "R", "1969-01-20"), ("FORD", "R", "1974-08-09"),
    ("CARTER", "D", "1977-01-20"), ("REAGAN", "R", "1981-01-20"),
    ("BUSH", "R", "1989-01-20"), ("CLINTON", "D", "1993-01-20"),
    ("BUSH JR", "R", "2001-01-20"), ("OBAMA", "D", "2009-01-20"),
    ("TRUMP", "R", "2017-01-20"), ("BIDEN", "D", "2021-01-20"),
    ("TRUMP II", "R", "2025-01-20"),
]

CONFIG = {
    "TICKER": "^GSPC",
    "END": "2026-09-30",
    "R_IN": 1.55,
    "R_OUT": 2.35,
    "GAP_DEG": 1.6,
    "SEG": 6,
    "WEDGE_Z": 1.6,
    "COL_R": 0.22,

    # render (leave these alone unless the topic needs landscape)
    "W": 1080, "H": 1920, "DPI": 100,
    "FPS": 30,
    "HOOK_SEC": 1.4, "BUILD_SEC": 8.5, "HOLD_SEC": 1.8,
    "ELEV_BASE": 22, "AZIM_START": -70, "AZIM_SWEEP": 22,
    "ZOOM": 1.16,
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
PARTY = {"D": "#1f6bff", "R": THEME["RED"]}

SAFE = {"TOP": 0.12, "BOTTOM": 0.20, "RIGHT": 0.14}

LAYOUT = {
    "TITLE": 0.862, "SUBTITLE": 0.836, "KEY": 0.814, "HANDLE": 0.792,
    "AXES": [-0.13, 0.28, 1.16, 0.50],
    "NOTE": 0.272, "BIG": 0.242, "SUB": 0.215,
}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------- DATA
def fetch():
    """yfinance once, then a CSV cache. Delete _cache/ to refetch."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "gspc.csv")
    if not os.path.exists(path):
        import yfinance as yf
        log(f"[Data] fetching {CONFIG['TICKER']}")
        px = yf.download([CONFIG["TICKER"]], period="max", interval="1d",
                         progress=False, auto_adjust=True)["Close"]
        if px.empty or px.iloc[:, 0].dropna().empty:
            raise SystemExit("no data returned; refusing to draw a synthetic tape")
        px.to_csv(path)
    log(f"[Data] cache {path}")
    s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0].dropna()
    return s.loc[:CONFIG["END"]]


# ------------------------------------------------------------- COMPUTE
def compute(s):
    def at(d):
        return float(s.loc[:d].iloc[-1])

    starts = [pd.Timestamp(p[2]) for p in PRESIDENTS]
    ends = starts[1:] + [s.index[-1]]
    pres = []
    for (name, party, _), a, b in zip(PRESIDENTS, starts, ends):
        pres.append({"name": name, "party": party, "start": a, "end": b,
                     "m": at(b) / at(a), "p0": at(a)})
    t0, t1 = starts[0], s.index[-1]
    span = (t1 - t0).days

    # arc geometry: both parties start at the front of the ring (270 deg, the
    # side facing the camera) and run in time order towards the back,
    # Democrats round the left, Republicans round the right; arc length = days
    deg_per_day = 360.0 / span
    pos = {"D": 270.0, "R": 270.0}
    for p in pres:
        w = (p["end"] - p["start"]).days * deg_per_day
        if p["party"] == "D":
            p["a0"], p["a1"] = pos["D"] - w, pos["D"]; pos["D"] -= w
        else:
            p["a0"], p["a1"] = pos["R"], pos["R"] + w; pos["R"] += w

    days = s.loc[t0:].index
    return {"s": s.loc[t0:], "pres": pres, "t0": t0, "t1": t1, "days": days,
            "always": at(t1) / at(t0),
            "blue": float(np.prod([p["m"] for p in pres if p["party"] == "D"])),
            "red": float(np.prod([p["m"] for p in pres if p["party"] == "R"]))}


def state_at(d, T):
    """Heights at time T: each block's log multiple so far, and the three
    compounded dollars."""
    s = d["s"]
    pT = float(s.loc[:T].iloc[-1])
    h, acc = [], {"D": 1.0, "R": 1.0}
    for p in d["pres"]:
        if T <= p["start"]:
            h.append(None); continue
        m = p["m"] if T >= p["end"] else pT / p["p0"]
        h.append(np.log10(m)); acc[p["party"]] *= m
    return h, pT / float(s.iloc[0]), acc["D"], acc["R"]


# ------------------------------------------------------------ VALIDATE
def validate(d):
    """Every number that reaches a frame is asserted here, before any render."""
    a, b, r = d["always"], d["blue"], d["red"]
    pres = d["pres"]
    assert len(pres) == 17
    assert abs(a - b * r) / a < 1e-9, "the party products do not multiply to always"
    assert 50 <= a <= 2000, f"always ${a:.0f} outside $50..$2000"
    assert 1 <= b < a and 0.2 <= r < a, "a party-only dollar beat staying invested"
    hoover = pres[0]["m"]
    assert 0.1 <= hoover <= 0.5, f"Hoover multiple {hoover:.2f}"
    ex_hoover = r / hoover
    yrs = {k: sum((p["end"] - p["start"]).days for p in pres if p["party"] == k) / 365.25
           for k in "DR"}
    assert 90 <= yrs["D"] + yrs["R"] <= 100
    losers = [p["name"] for p in pres if p["m"] < 1]

    figs = {
        "always": f"${a:,.0f}",
        "blue": f"${b:,.0f}",
        "red": f"${r:,.2f}",
        "red_ex_hoover": f"${ex_hoover:,.2f}",
        "hoover": f"{(hoover - 1) * 100:.0f}%",
        "n_pres": f"{len(pres)}",
        "yrs_blue": f"{yrs['D']:.0f}",
        "yrs_red": f"{yrs['R']:.0f}",
        "losers": ", ".join(losers),
        "stake": "$1",
        "index": "S&P 500",
        "hoover_span": f"{pres[0]['start'].year} to {pres[0]['end'].year}",
        "cash": "0%",
        "start": d["t0"].strftime("%b %Y"),
        "end": d["t1"].strftime("%b %Y"),
    }
    with open(os.path.join(BASE_DIR, "figures.json"), "w") as fh:
        json.dump(figs, fh, indent=1)
    log(f"[validate] {figs}")
    d["figs"] = figs
    return figs


# -------------------------------------------------------------- RENDER
def _shade(c, k):
    c = np.array(to_rgba(c)); c[:3] = np.clip(c[:3] * k, 0, 1); return c


def _wedge(polys, cols, a0, a1, h, col):
    c = CONFIG
    th = np.radians(np.linspace(a0 + c["GAP_DEG"] / 2, a1 - c["GAP_DEG"] / 2,
                                c["SEG"] + 1))
    z = h * c["WEDGE_Z"]
    ri, ro = c["R_IN"], c["R_OUT"]
    xi, yi, xo, yo = ri * np.cos(th), ri * np.sin(th), ro * np.cos(th), ro * np.sin(th)
    top = list(zip(xo, yo, [z] * len(th))) + list(zip(xi[::-1], yi[::-1], [z] * len(th)))
    polys.append(top); cols.append(_shade(col, 1.25 if z >= 0 else 0.9))
    for k in range(len(th) - 1):
        polys.append([(xo[k], yo[k], 0), (xo[k + 1], yo[k + 1], 0),
                      (xo[k + 1], yo[k + 1], z), (xo[k], yo[k], z)])
        cols.append(_shade(col, 0.8))
        polys.append([(xi[k], yi[k], 0), (xi[k + 1], yi[k + 1], 0),
                      (xi[k + 1], yi[k + 1], z), (xi[k], yi[k], z)])
        cols.append(_shade(col, 0.55))
    for j in (0, -1):
        polys.append([(xi[j], yi[j], 0), (xo[j], yo[j], 0),
                      (xo[j], yo[j], z), (xi[j], yi[j], z)])
        cols.append(_shade(col, 0.65))


def _column(polys, cols, x, y, h, col):
    r = CONFIG["COL_R"]
    th = np.linspace(0, 2 * np.pi, 25)
    cx, cy = x + r * np.cos(th), y + r * np.sin(th)
    for k in range(len(th) - 1):
        lit = 0.55 + 0.6 * max(0.0, np.cos(th[k] - np.radians(225)))
        polys.append([(cx[k], cy[k], 0), (cx[k + 1], cy[k + 1], 0),
                      (cx[k + 1], cy[k + 1], h), (cx[k], cy[k], h)])
        cols.append(_shade(col, lit))
    polys.append(list(zip(cx, cy, [h] * len(th))))
    cols.append(_shade(col, 1.35))


COLS = [("ALWAYS", 0.0, 0.0, "YELLOW"), ("BLUE", -0.7, -0.25, None),
        ("RED", 0.78, -0.3, None)]


def scene(ax, d, frac):
    c = CONFIG
    T = d["t0"] + (d["t1"] - d["t0"]) * frac
    h, al, bl, rd = state_at(d, T)
    polys, cols = [], []
    for p, hk in zip(d["pres"], h):
        if hk is None:
            continue
        _wedge(polys, cols, p["a0"], p["a1"], hk, PARTY[p["party"]])
        if hk < 0:                     # a loss: outline the sunk top in white
            th = np.radians(np.linspace(p["a0"], p["a1"], 20))
            ri, ro, z = c["R_IN"], c["R_OUT"], hk * c["WEDGE_Z"]
            xs = np.r_[ro * np.cos(th), ri * np.cos(th[::-1]), ro * np.cos(th[:1])]
            ys = np.r_[ro * np.sin(th), ri * np.sin(th[::-1]), ro * np.sin(th[:1])]
            ax.plot(xs, ys, np.full_like(xs, z), color="white", lw=1.1)
    for (name, x, y, _), val, col in zip(
            COLS, (al, bl, rd), (THEME["YELLOW"], PARTY["D"], PARTY["R"])):
        _column(polys, cols, x, y, max(np.log10(val), 0.004) * c["WEDGE_Z"], col)
    if polys:
        pc = Poly3DCollection(polys, facecolors=cols, edgecolors=(0, 0, 0, 0.35),
                              linewidths=0.3)
        ax.add_collection3d(pc, autolim=False)
    # floor rings, thin, so the sunk blocks read as below ground
    th = np.linspace(0, 2 * np.pi, 200)
    for r in (c["R_IN"], c["R_OUT"]):
        ax.plot(r * np.cos(th), r * np.sin(th), 0 * th, color=THEME["TEXT_DIM"],
                lw=0.6, alpha=0.5)
    R = c["R_OUT"] + 0.1
    ax.set_xlim(-R, R); ax.set_ylim(-R, R)
    ax.set_zlim(-1.1, np.log10(d["always"]) * c["WEDGE_Z"] + 0.2)
    return T, h, (al, bl, rd)


def style_axes(ax):
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
        pane.line.set_color((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def scene_labels(ax, d, st, fs=10):
    """Projected to 2D so the blocks cannot depth-sort the labels away."""
    from mpl_toolkits.mplot3d import proj3d
    M = ax.get_proj()
    c, figs = CONFIG, d["figs"]
    T, h, vals = st

    def put(x, y, z, s, **kw):
        x2, y2, _ = proj3d.proj_transform(x, y, z, M)
        ax.text2D(x2, y2, s, transform=ax.transData, zorder=1e6, ha="center",
                  va="center", **kw)

    rl = c["R_OUT"] + 0.32
    for p, hk in zip(d["pres"], h):
        if hk is None or p["a1"] - p["a0"] < 7:   # Trump II sits under TRUMP
            continue
        a = np.radians((p["a0"] + p["a1"]) / 2)
        put(rl * np.cos(a), rl * np.sin(a), max(hk, 0) * c["WEDGE_Z"] + 0.05,
            p["name"] + ("" if p["m"] >= 1 else " ▼"), color=THEME["TEXT"],
            fontsize=fs - 2.5, family=THEME["MONO"], alpha=0.9)
    done = T >= d["t1"]
    txt = {"ALWAYS": figs["always"], "BLUE": figs["blue"], "RED": figs["red"]}
    colr = {"ALWAYS": THEME["YELLOW"], "BLUE": "#5c9bff", "RED": THEME["RED"]}
    for (name, x, y, _), v in zip(COLS, vals):
        z = max(np.log10(v), 0) * c["WEDGE_Z"] + 0.28
        put(x, y, z, f"{name}\n{txt[name]}" if done else name, color=colr[name],
            fontsize=fs + 2, fontweight="bold", family=THEME["MONO"])


def text_block(fig, figs, scale=1.0):
    L, s = LAYOUT, scale
    fig.text(0.5, L["TITLE"], "RED VS BLUE", ha="center",
             fontsize=30 * s, fontweight="bold", color=THEME["TEXT"],
             family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             f"$1 in the S&P 500, held only under one party  |  "
             f"{figs['start']} to {figs['end']}",
             ha="center", fontsize=12 * s, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.5, L["KEY"],
             "block = one presidency, height = gain (log), below floor = loss",
             ha="center", fontsize=10 * s, color=THEME["TEXT_DIM"], family=THEME["FONT"])
    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10 * s,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])
    fig.text(0.5, L["BIG"], f"STAYING IN: {figs['always']}. BEAT BOTH.",
             ha="center", fontsize=22 * s, color=THEME["YELLOW"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             f"blue only {figs['blue']}  ·  red only {figs['red']}  ·  "
             f"price only, cash at 0%",
             ha="center", fontsize=11.5 * s, color=THEME["TEXT_DIM"],
             family=THEME["MONO"])


def note(fig, d, st, scale=1.0):
    T = st[0]
    who = [p["name"] for p in d["pres"] if p["start"] <= T][-1]
    fig.text(0.5, LAYOUT["NOTE"], f"{T.year}  ·  {who}",
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
    st = scene(ax, d, frac)
    note(fig, d, st)
    ax.view_init(elev=c["ELEV_BASE"] + 5 * np.sin(t * np.pi),
                 azim=c["AZIM_START"] + c["AZIM_SWEEP"] * t)
    ax.set_box_aspect((1.0, 1.0, 0.95), zoom=c["ZOOM"])
    style_axes(ax)
    scene_labels(ax, d, st)

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
