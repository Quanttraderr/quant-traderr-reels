"""
CashMelting_Reel_Pipeline.py
============================
Reel for @quant.traderr, "Cash Is Melting". What $100 of US cash from
January 1913 still buys today, drawn as a glass hourglass running out.

THE DATA
    US CPI-U, all items, not seasonally adjusted (FRED series CPIAUCNS),
    monthly from Jan 1913 to the latest print. Fetched once from
    fred.stlouisfed.org without an API key and kept in _cache/CPIAUCNS.csv.
    Real data throughout. Nothing here is simulated. One month, Oct 2025,
    has no value in the series (the BLS never published it during the
    government shutdown); that month is dropped, not filled in.
    T-bills as a counterweight were left out on purpose: Yahoo's ^IRX only
    starts around 1960, and splicing a second series onto half the window
    would make the picture look more complete than the data is.

THE MATHS
    Purchasing power of $100 held as cash: P_t = 100 * CPI_Jan1913 / CPI_t.
    Loss since a date d: 1 - CPI_d / CPI_last. Worst inflation: the highest
    12-month change CPI_t / CPI_{t-12} - 1. Average inflation: the compound
    annual rate from Jan 1913 to the last month.

    In 3D: a glass hourglass, a surface of revolution drawn as a wire mesh.
    Sand volume, not sand height, is the money: the top chamber holds
    P_t / 100 of the sand, the bottom chamber holds the rest. Every grain in
    the bottom pile stands for a slice of the original $100 and is coloured
    by the decade in which that slice was lost for good (the last time the
    loss rose through that level), blue for the 1910s up to red for the
    2020s. When prices fell, as in the early 1930s, sand really does climb
    back into the top chamber.

THE CLAIM THE VISUAL MAKES
    $100 kept as cash since Jan 1913 buys what $2.93 bought then: 97% of
    its purchasing power is gone, 88% of it since the gold window closed in
    Aug 1971.

WHAT THE VISUAL IS NOT
    Not a return on anything. Nobody held one banknote for 113 years, and
    wages and interest rates rose alongside prices, so this is not "what
    savers lost". CPI is an average basket that has itself changed over the
    century (and is measured differently today than in 1913), so the number
    is an index ratio, not the price of one fixed set of goods. The sand
    levels are volume, not height: an hourglass is narrow at the neck, so a
    small drop in the top chamber can be a lot of money.

RUN
    python CashMelting_Reel_Pipeline.py --smoke   # 3 frames
    python CashMelting_Reel_Pipeline.py           # full render -> topic.mp4

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
from matplotlib.colors import LinearSegmentedColormap, to_rgba
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from mpl_toolkits.mplot3d import proj3d

warnings.filterwarnings("ignore")
plt.rcParams["text.parse_math"] = False      # "$100 -> $2.93" is money, not mathtext
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "_cache")

CONFIG = {
    "URL": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCNS",
    "START": "1913-01-01",
    "N_GRAINS": 6000,                # grains for the full $100
    "SAND_FILL": 0.80,               # share of one bulb the sand fills
    "R_NECK": 0.07,                  # hourglass neck radius
    "R_BULB": 0.62,                  # widest bulb radius
    "BULB_K": 0.70,                  # profile r = neck + (R - neck) * sin(pi*K*u)

    # render
    "W": 1080, "H": 1920, "DPI": 100,
    "FPS": 30,
    "HOOK_SEC": 1.4, "BUILD_SEC": 8.5, "HOLD_SEC": 1.8,
    "ELEV_BASE": 14, "AZIM_START": -70, "AZIM_SWEEP": 30,
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
    "AXES": [-0.08, 0.265, 1.16, 0.52],     # taller than the house rect: the hourglass is a tall form
    "NOTE": 0.272, "BIG": 0.242, "SUB": 0.215,
}

DECADES = list(range(1910, 2030, 10))          # 1910s .. 2020s, colour index
RULER = [100, 50, 25, 10]                      # $ marks on the top chamber


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------------------------------------------------------------- DATA
def fetch():
    """FRED once, then the CSV cache. The cache is what makes a re-render
    months from now draw the same sand the caption was asserted against.
    Delete _cache/ to refetch. A failed fetch stops the run."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, "CPIAUCNS.csv")
    if not os.path.exists(path):
        import urllib.request
        log("[Data] fetching CPIAUCNS from FRED")
        try:
            with urllib.request.urlopen(CONFIG["URL"], timeout=60) as r:
                raw = r.read()
        except Exception as e:
            raise SystemExit(f"FRED fetch failed ({e}); refusing to draw a synthetic series")
        if b"CPIAUCNS" not in raw[:200]:
            raise SystemExit("FRED returned something that is not the CPI csv")
        with open(path, "wb") as fh:
            fh.write(raw)
    log(f"[Data] cache {path}")
    s = pd.read_csv(path, index_col=0, parse_dates=True, na_values=["."]).iloc[:, 0]
    missing = s.index[s.isna()]
    assert len(missing) <= 2, f"{len(missing)} CPI months missing: {list(missing)}"
    s = s.dropna().astype(float)
    s = s.loc[CONFIG["START"]:]
    if s.empty:
        raise SystemExit("CPI series empty; refusing to draw a synthetic series")
    s.attrs["missing"] = [m.strftime("%b %Y") for m in missing]
    return s


# ------------------------------------------------------------- COMPUTE
def radius(u):
    """Hourglass profile, u = |z| from neck (0) to cap (1)."""
    c = CONFIG
    return c["R_NECK"] + (c["R_BULB"] - c["R_NECK"]) * np.sin(np.pi * c["BULB_K"] * u)


def _vdc(n, base=2):
    """Van der Corput sequence: deterministic, evenly spread radii, no RNG."""
    out = np.zeros(n)
    for i in range(n):
        q, denom, k = 0.0, 1.0, i + 1
        while k:
            k, rem = divmod(k, base)
            denom *= base
            q += rem / denom
        out[i] = q
    return out


def frame_plan():
    c = CONFIG
    n_hook = int(c["FPS"] * c["HOOK_SEC"])
    n_build = int(c["FPS"] * c["BUILD_SEC"])
    total = int(c["FPS"] * (c["HOOK_SEC"] + c["BUILD_SEC"] + c["HOLD_SEC"]))
    return n_hook, n_build, total


def compute(cpi):
    c = CONFIG
    P = 100.0 * cpi.iloc[0] / cpi.values               # purchasing power, $
    L = 100.0 - P                                      # lost, $ of the original 100
    dates = cpi.index
    yoy = (cpi / cpi.shift(12, freq="MS").reindex(cpi.index) - 1)

    # volume of one bulb as a function of u
    ug = np.linspace(0, 1, 4001)
    cumv = np.concatenate([[0], np.cumsum(np.pi * radius(0.5 * (ug[1:] + ug[:-1])) ** 2
                                          * np.diff(ug))])
    Vb = cumv[-1]
    Vs = c["SAND_FILL"] * Vb

    def u_top(dollars):        # top sand from the neck up to this u
        return np.interp(np.asarray(dollars) / 100.0 * Vs, cumv, ug)

    def u_bot(dollars):        # bottom sand from the cap (u=1) inwards to this u
        return np.interp(Vb - np.clip(np.asarray(dollars), 0, None) / 100.0 * Vs, cumv, ug)

    # grains: fixed positions, revealed by count
    N = c["N_GRAINS"]
    nmax = int(np.ceil(N * max(P.max(), 100) / 100)) + 2
    j = np.arange(nmax)
    q = np.sqrt(_vdc(nmax)) * 0.94
    ang = 2 * np.pi * _vdc(nmax, 3)                    # Halton (2,3): no spiral streaks
    lev = (j + 0.5) / N * 100                          # $ level of grain j
    ut = u_top(np.minimum(lev, 100 / c["SAND_FILL"] * 0.999))
    rt = radius(ut) * q
    top_xyz = np.column_stack([rt * np.cos(ang), rt * np.sin(ang), ut])
    ub = u_bot(np.minimum(lev, 100 / c["SAND_FILL"] * 0.999))
    rb = radius(ub) * q
    ang2 = 2 * np.pi * _vdc(nmax, 5)
    bot_xyz = np.column_stack([rb * np.cos(ang2), rb * np.sin(ang2), -ub])

    # decade in which each $ level was lost for good, as of every month:
    # last month s <= m where L rose through that level
    up = (L[None, 1:] >= lev[:, None]) & (L[None, :-1] < lev[:, None])
    idx = np.where(up, np.arange(1, len(L))[None, :], -1)
    last = np.maximum.accumulate(idx, axis=1)          # shape (grains, months-1)
    dec_of_month = np.clip((dates.year // 10 * 10 - DECADES[0]) // 10, 0, len(DECADES) - 1)

    # frame -> month
    n_hook, n_build, total = frame_plan()
    months = np.empty(total, int)
    for f in range(total):
        if f < n_hook:
            months[f] = 0
        elif f < n_hook + n_build:
            months[f] = int(round((f - n_hook) / (n_build - 1) * (len(P) - 1)))
        else:
            months[f] = len(P) - 1

    frames = []
    for f in range(total):
        m = months[f]
        ntop = int(round(N * P[m] / 100))
        nbot = int(round(N * max(L[m], 0) / 100))
        if m == 0 or nbot == 0:
            dec = np.zeros(nbot, int)
        else:
            dec = dec_of_month[last[:nbot, m - 1]]
        mprev = months[f - 1] if f > 0 else m
        flowing = (n_hook <= f < n_hook + n_build) and L[m] > L[mprev] + 1e-9
        # decade bands in the pile: label at the middle of each band
        bands = []
        if nbot:
            for k in np.unique(dec):
                sel = np.nonzero(dec == k)[0]
                share = len(sel) / N
                if share >= 0.005:                 # spread() keeps thin bands legible
                    mid = 100 * (sel[0] + sel[-1] + 1) / 2 / N
                    bands.append((f"{DECADES[k]}s", float(-u_bot(mid)), int(k)))
        frames.append({"m": int(m), "ntop": ntop, "nbot": nbot, "dec": dec.astype(np.int16),
                       "flow": bool(flowing), "z_pile": float(-u_bot(max(L[m], 0))),
                       "z_top": float(u_top(max(P[m], 0))),
                       "stream_dec": int(dec_of_month[m]), "bands": bands})

    ruler = [(f"${v}", float(u_top(v))) for v in RULER]
    return {"cpi": cpi, "P": P, "L": L, "yoy": yoy, "dates": dates,
            "top_xyz": top_xyz, "bot_xyz": bot_xyz, "frames": frames,
            "ruler": ruler, "months": months}


# ------------------------------------------------------------ VALIDATE
def _pct0(x):
    return f"{x * 100:.0f}%"


def validate(d):
    """Every number that reaches a frame is asserted here, before any render.
    The per-frame counter strings are built here too, so render_frame only
    ever draws text it was handed."""
    cpi, P, yoy, dates = d["cpi"], d["P"], d["yoy"], d["dates"]
    last = cpi.iloc[-1]
    assert dates[0] == pd.Timestamp("1913-01-01"), "series no longer starts Jan 1913"
    assert len(cpi) > 1300, f"only {len(cpi)} months, history looks cut"
    assert dates[-1] >= pd.Timestamp("2025-01-01"), f"last print {dates[-1]:%b %Y} is stale"
    assert (cpi > 0).all() and np.isfinite(cpi).all()

    final = float(P[-1])
    assert 0.5 < final < 20, f"$100 -> ${final:.2f}, outside any plausible band"

    def loss_since(day):
        v = float(cpi.loc[day])
        return 1 - v / last, 100 * cpi.iloc[0] / v

    l71, v71 = loss_since("1971-08-01")                 # Nixon closes the gold window
    l00, _ = loss_since("2000-01-01")
    l20, _ = loss_since("2020-01-01")
    assert 0.5 < l71 < 0.99 and 0.2 < l00 < 0.8 and 0.05 < l20 < 0.6
    assert l71 > l00 > l20, "losses since earlier dates must be larger"
    assert 5 < v71 < 60

    y = yoy.dropna()
    worst_at = y.idxmax()
    worst = float(y.max())
    assert 0.08 < worst < 0.5, f"worst 12-month inflation {worst:.3f} implausible"
    y20 = y.loc["2020-01-01":]
    p20_at, p20 = y20.idxmax(), float(y20.max())
    assert 0.02 < p20 < 0.2
    dep = pd.Series(P, index=dates).loc["1929-01-01":"1940-12-01"]
    dep_at, dep_v = dep.idxmax(), float(dep.max())
    assert dep_v > float(pd.Series(P, index=dates).loc["1929-10-01"]), "no 1930s deflation?"
    years = (dates[-1] - dates[0]).days / 365.25
    cagr = (last / cpi.iloc[0]) ** (1 / years) - 1
    assert 0.01 < cagr < 0.08

    figs = {
        "base": "$100",
        "final_value": f"${final:.2f}",
        "loss_total": _pct0(1 - final / 100),
        "start": "Jan 1913",
        "end": dates[-1].strftime("%b %Y"),
        "years": f"{int(years)}",          # whole years, Jan 1913 to now
        "n_months": f"{len(cpi):,}",
        "value_1971": f"${v71:.2f}",
        "loss_since_1971": _pct0(l71),
        "gold_window": "Aug 1971",
        "since_2000": "Jan 2000",
        "loss_since_2000": _pct0(l00),
        "since_2020": "Jan 2020",
        "loss_since_2020": _pct0(l20),
        "yoy_window": "12 months",
        "worst_yoy": f"+{worst * 100:.1f}%",
        "worst_date": worst_at.strftime("%b %Y"),
        "peak_2020s": f"+{p20 * 100:.1f}%",
        "peak_2020s_date": p20_at.strftime("%b %Y"),
        "depression_value": f"${dep_v:.2f}",
        "depression_date": dep_at.strftime("%b %Y"),
        "cagr": f"{cagr * 100:.1f}%",
        "missing_month": ", ".join(cpi.attrs.get("missing", [])) or "none",
    }
    with open(os.path.join(BASE_DIR, "figures.json"), "w") as fh:
        json.dump(figs, fh, indent=1)
    log(f"[validate] {figs}")

    # per-frame strings: the counter and the line under it
    n_hook, n_build, total = frame_plan()
    # two short lines each: a one-line tag ran into the left edge at 1933
    events = [(worst_at, f"{figs['worst_date']}\n{figs['worst_yoy']} in 12 months"),
              (dep_at, f"{figs['depression_date']}: deflation\n$100 buys "
                       f"{figs['depression_value']} again"),
              (pd.Timestamp("1971-08-01"), f"{figs['gold_window']}\ngold window closes"),
              (p20_at, f"{figs['peak_2020s_date']}\n{figs['peak_2020s']} in 12 months")]
    for fr in d["frames"]:
        m = fr["m"]
        fr["counter"] = f"{dates[m].year}   $100 → ${P[m]:.2f}"
        yv = yoy.iloc[m]
        fr["sub"] = (f"starting point  {figs['start']}" if not np.isfinite(yv)
                     else f"prices over 12 months  {yv * 100:+5.1f}%")
        fr["event"] = ""
        for at, txt in events:
            age = (dates[m] - at).days / 365.25
            if 0 <= age < 9:
                fr["event"] = txt
    endf = d["frames"][-1]
    endf_counter = f"$100 → {figs['final_value']}"
    assert endf["counter"].endswith(figs["final_value"]), "counter end != figures.json"
    for k in range(n_hook + n_build, total):
        d["frames"][k]["counter"] = endf_counter
        d["frames"][k]["sub"] = (f"-{figs['loss_total']} since {figs['start'][-4:]}  ·  "
                                 f"-{figs['loss_since_1971']} since {figs['gold_window'][-4:]}  ·  "
                                 f"-{figs['loss_since_2020']} since {figs['since_2020'][-4:]}")
        d["frames"][k]["event"] = ""
    d["figs"] = figs
    return figs


# -------------------------------------------------------------- RENDER
def style_axes(ax):
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
        pane.line.set_color((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


def glass(azim_deg, n_mer=20, n_par=9):
    """Wire mesh of the hourglass, split into the half behind and in front."""
    zs = np.linspace(-1, 1, 161)
    rs = radius(np.abs(zs))
    back, front = [], []
    a0 = np.deg2rad(azim_deg)
    for th in np.linspace(0, 2 * np.pi, n_mer, endpoint=False):
        seg = np.column_stack([rs * np.cos(th), rs * np.sin(th), zs])
        (front if np.cos(th - a0) > 0 else back).append(seg)
    th = np.linspace(0, 2 * np.pi, 121)
    for side in (-1, 1):
        for u in np.linspace(0.12, 1.0, n_par):
            r = radius(u)
            ring = np.column_stack([r * np.cos(th), r * np.sin(th), np.full_like(th, side * u)])
            fr = np.cos(th - a0) > 0
            back.append(ring[~fr]); front.append(ring[fr])
    return back, front


def frame_wood():
    """Caps and three posts of the stand."""
    th = np.linspace(0, 2 * np.pi, 121)
    R = CONFIG["R_BULB"] + 0.08
    segs = []
    for z in (-1.0, 1.0):
        for rr in (R, R * 0.55):
            segs.append(np.column_stack([rr * np.cos(th), rr * np.sin(th), np.full_like(th, z)]))
    for a in (0.3, 0.3 + 2 * np.pi / 3, 0.3 + 4 * np.pi / 3):
        segs.append(np.array([[R * np.cos(a), R * np.sin(a), -1.0],
                              [R * np.cos(a), R * np.sin(a), 1.0]]))
    return segs


DEC_RGBA = np.array([CMAP(k / (len(DECADES) - 1)) for k in range(len(DECADES))])


def scene(ax, d, fr, azim, idx, s=1.0):
    ax.computed_zorder = False
    back, front = glass(azim)
    cy = to_rgba(THEME["CYAN"])
    ax.add_collection3d(Line3DCollection(frame_wood(), colors=[(1, 1, 1, 0.22)],
                                         linewidths=0.8 * s, zorder=0), autolim=False)
    ax.add_collection3d(Line3DCollection(back, colors=[(*cy[:3], 0.05)],
                                         linewidths=2.6 * s, zorder=1), autolim=False)
    ax.add_collection3d(Line3DCollection(back, colors=[(*cy[:3], 0.20)],
                                         linewidths=0.55 * s, zorder=1), autolim=False)

    nt, nb = fr["ntop"], fr["nbot"]
    if nt:
        t = d["top_xyz"][:nt]
        ax.scatter(t[:, 0], t[:, 1], t[:, 2], s=3.2 * s ** 2, c=[(1, 0.97, 0.88, 0.85)],
                   depthshade=False, linewidths=0, zorder=2)
    if nb:
        b = d["bot_xyz"][:nb]
        cols = DEC_RGBA[fr["dec"]]
        ax.scatter(b[:, 0], b[:, 1], b[:, 2], s=9 * s ** 2, c=cols[:, :3], alpha=0.12,
                   depthshade=False, linewidths=0, zorder=2)
        ax.scatter(b[:, 0], b[:, 1], b[:, 2], s=3.2 * s ** 2, c=cols,
                   depthshade=False, linewidths=0, zorder=3)
    if fr["flow"]:
        z0, z1 = 0.02, fr["z_pile"]
        k = np.arange(26) / 26.0
        ph = (k + idx * 0.083) % 1.0
        zz = z0 + (z1 - z0) * ph
        col = DEC_RGBA[fr["stream_dec"]]
        ax.scatter(np.zeros_like(zz), np.zeros_like(zz), zz, s=6 * s ** 2, c=[col],
                   depthshade=False, linewidths=0, zorder=4)
        ax.plot([0, 0], [0, 0], [z0, z1], color=(*col[:3], 0.35), lw=1.2 * s, zorder=4)

    ax.add_collection3d(Line3DCollection(front, colors=[(*cy[:3], 0.06)],
                                         linewidths=2.6 * s, zorder=5), autolim=False)
    ax.add_collection3d(Line3DCollection(front, colors=[(*cy[:3], 0.30)],
                                         linewidths=0.6 * s, zorder=5), autolim=False)
    lim = CONFIG["R_BULB"] + 0.1
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-1.04, 1.04)


def scene_labels(ax, d, fr, s=1.0):
    """Projected 2D labels at the hourglass silhouette, drawn above the 3D."""
    M = ax.get_proj()
    th = np.linspace(0, 2 * np.pi, 72, endpoint=False)

    def edge(z, side):
        r = radius(abs(z))
        x2, y2, _ = proj3d.proj_transform(r * np.cos(th), r * np.sin(th),
                                          np.full_like(th, z), M)
        i = np.argmin(x2) if side < 0 else np.argmax(x2)
        return x2[i], y2[i]

    def put(x, y, txt, **kw):
        ax.text2D(x, y, txt, transform=ax.transData, zorder=1e6, va="center",
                  family=THEME["MONO"], **kw)

    xr = ax.get_xlim()
    pad = (xr[1] - xr[0]) * 0.012
    gap = abs(edge(1.0, -1)[1] - edge(-1.0, -1)[1]) * 0.036   # min label spacing

    def spread(items):
        """Push labels apart vertically so none overlap; keeps their order."""
        items = sorted(items, key=lambda it: it[1])
        for i in range(1, len(items)):
            if items[i][1] - items[i - 1][1] < gap:
                items[i] = (items[i][0], items[i - 1][1] + gap, *items[i][2:])
        return items

    # $ ruler on the left of the top chamber
    rows = []
    for txt, z in d["ruler"]:
        x2, y2 = edge(z, -1)
        rows.append((x2, y2, txt, fr["z_top"] >= z - 1e-6))
    for x2, y2, txt, live in spread([(r[0], r[1], r[2], r[3]) for r in rows]):
        put(x2 - pad, y2, f"{txt} ──", ha="right", fontsize=10.5 * s,
            color=THEME["TEXT"] if live else THEME["TEXT_DIM"], alpha=1 if live else 0.6)
    # decade bands on the left of the pile
    rows = []
    for name, z, k in fr["bands"]:
        x2, y2 = edge(z, -1)
        rows.append((x2, y2, name, k))
    for x2, y2, name, k in spread(rows):
        put(x2 - pad, y2, f"{name} ──", ha="right", fontsize=10.5 * s,
            color=DEC_RGBA[k], fontweight="bold")
    if fr["event"]:
        x2, y2 = edge(0.0, -1)                 # left: the right edge is Instagram's
        put(x2 - pad * 2, y2, fr["event"], ha="right", multialignment="right", fontsize=10.5 * s,
            color=THEME["YELLOW"],
            bbox=dict(boxstyle="round,pad=0.25", fc=THEME["BG"], ec="none", alpha=0.8))


def text_block(fig, figs, fr, s=1.0):
    L = LAYOUT
    fig.text(0.5, L["TITLE"], "CASH IS MELTING", ha="center", fontsize=29 * s,
             fontweight="bold", color=THEME["TEXT"], family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             f"$100 held as cash  |  US CPI, {figs['start']} to {figs['end']}",
             ha="center", fontsize=12.5 * s, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.5, L["KEY"],
             "sand on top = what it still buys  ·  below = lost  ·  colour = decade lost",
             ha="center", fontsize=10 * s, color=THEME["TEXT_DIM"], family=THEME["FONT"])
    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10 * s,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])
    fig.text(0.5, L["BIG"], fr["counter"], ha="center", fontsize=27 * s,
             color=THEME["ORANGE"], fontweight="bold", family=THEME["MONO"])
    fig.text(0.5, L["SUB"], fr["sub"], ha="center", fontsize=12 * s,
             color=THEME["TEXT_DIM"], family=THEME["MONO"])


def camera(idx, total):
    c = CONFIG
    t = idx / max(total - 1, 1)
    return c["ELEV_BASE"] + 5 * np.sin(t * np.pi), c["AZIM_START"] + c["AZIM_SWEEP"] * t


def render_frame(idx, total, d, out_path):
    c, figs = CONFIG, d["figs"]
    fr = d["frames"][idx]
    elev, azim = camera(idx, total)

    fig = plt.figure(figsize=(c["W"] / c["DPI"], c["H"] / c["DPI"]),
                     dpi=c["DPI"], facecolor=THEME["BG"])
    text_block(fig, figs, fr)
    ax = fig.add_axes(LAYOUT["AXES"], projection="3d", facecolor=THEME["BG"])
    scene(ax, d, fr, azim, idx)
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect((1.0, 1.0, 1.55), zoom=c["ZOOM"])
    style_axes(ax)
    scene_labels(ax, d, fr)
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
    total = frame_plan()[2]
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
    # crf 14 rather than 18: Instagram re-encodes whatever it is given, and two
    # lossy passes compound. The colour tags matter too. Without them the file
    # reports "unknown" primaries and every player, Instagram included, falls
    # back to a guess, which is where washed out colour comes from.
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
    shutil.rmtree(frames, ignore_errors=True)   # the cache is large if you don't
    log(f"OK {mp4}  ({total} frames, {total / c['FPS']:.1f}s, {c['W']}x{c['H']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    main(**vars(ap.parse_args()))
