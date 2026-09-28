"""
Recovery_Reel_Pipeline.py
=========================
Reel 17 for @quant.traderr, "It Always Comes Back". Every fall of the S&P 500
of 5% or more since 1928, measured not by how deep it went but by how long you
had to wait until the old high was back.

THE DATA
    ^GSPC from Yahoo, daily closes since Dec 1927. This is the PRICE index,
    dividends are NOT included, and before March 1957 the series is the
    S&P 90, not the 500. It is the only daily series that reaches back to 1929,
    which is the whole point. As a check with dividends we also pull SPY
    (adjusted closes, dividends reinvested) since 1993 and report its longest
    wait next to it.

THE MATHS
    An episode starts at a closing high, runs through its lowest close and ends
    on the first close at or above that old high. Episodes whose trough is at
    least 5% below the high are kept. The wait is the calendar time from the old
    high to the new one. An episode still open on the last day of the data is
    not counted (and asserted to be shallower than 5%, so nothing is hidden).

    In 3D: one sphere per episode. Sphere VOLUME is proportional to the wait,
    so the radius is the cube root of the years. Colour is the depth of the
    fall, blue at -5% to red at the deepest. The spheres are packed greedily in
    order of their wait, shortest first: each new sphere is placed touching the
    pack, as close to its centre as it fits. The build grows the pack in that
    same order, so the viewer sees dozens of small quick recoveries and then
    the few long waits, the last being 1929.

THE CLAIM THE VISUAL MAKES
    "It always comes back" is true for every closed episode, and most come back
    fast. But the longest wait on the S&P price index was 25 years.

WHAT THE VISUAL IS NOT
    Not a total-return picture: with dividends reinvested every wait is
    shorter (SPY vs ^GSPC after 2000 shows how much), and nothing here is
    adjusted for inflation, which would make the 1970s wait longer. Positions
    inside the pack carry no meaning beyond the order of the waits. Episodes
    are not independent trials, and one index over one century is one history,
    not a law. It says nothing about single stocks, many of which never came
    back at all.

RUN
    python Recovery_Reel_Pipeline.py --smoke   # 3 frames -> temp_frames_smoke/
    python Recovery_Reel_Pipeline.py           # full render -> topic.mp4

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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, "_cache")

CONFIG = {
    "TICKERS": ["^GSPC", "SPY"],
    "INDEX": "^GSPC",
    "CHECK": "SPY",
    "PERIOD": "max",
    "MIN_DEPTH": 5.0,          # percent, smallest fall that counts as an episode
    "FAST_MONTHS": 12,         # "came back within" threshold
    "SLOW_YEARS": 2,           # second threshold for the key row
    "N_DIRS": 600,             # candidate directions per sphere when packing

    # render
    "W": 1080, "H": 1920, "DPI": 100,
    "FPS": 30,
    "HOOK_SEC": 1.4, "BUILD_SEC": 8.5, "HOLD_SEC": 1.8,
    "ELEV_BASE": 18, "AZIM_START": -70, "AZIM_SWEEP": 40,
    "ZOOM": 1.45,
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
    for t in CONFIG["TICKERS"]:
        if px[t].dropna().empty:
            raise SystemExit(f"no data for {t}; refusing to draw a synthetic tape")
    px.to_csv(path)
    return px


# ------------------------------------------------------------- COMPUTE
def episodes(s, min_depth):
    """Closed peak -> trough -> new-high episodes, plus the open one if any."""
    v, d = s.values, s.index
    out, peak, trough, under = [], 0, 0, False
    for i in range(1, len(v)):
        if v[i] >= v[peak]:
            if under:
                depth = (v[trough] / v[peak] - 1.0) * 100.0
                if depth <= -min_depth:
                    out.append({"peak": d[peak], "trough": d[trough],
                                "back": d[i], "depth": depth,
                                "years": (d[i] - d[peak]).days / 365.25})
                under = False
            peak, trough = i, i
        else:
            under = True
            if v[i] < v[trough]:
                trough = i
    open_depth = (v[trough] / v[peak] - 1.0) * 100.0 if under else 0.0
    return out, open_depth, d[peak]


def pack(radii, n_dirs):
    """Greedy sphere packing in the given order: each sphere touches one that is
    already placed, never overlaps, and sits as close to the pack's centre as
    it can. Deterministic: fibonacci directions, no randomness."""
    k = np.arange(n_dirs) + 0.5
    phi = np.arccos(1 - 2 * k / n_dirs)
    th = np.pi * (1 + 5 ** 0.5) * k
    dirs = np.column_stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi),
                            np.cos(phi)])
    C = np.zeros((len(radii), 3))
    for i in range(1, len(radii)):
        placed, rp = C[:i], radii[:i]
        centre = (placed * rp[:, None] ** 3).sum(0) / (rp ** 3).sum()
        cand = (placed[:, None, :] + (rp[:, None, None] + radii[i]) * dirs[None]
                ).reshape(-1, 3)
        dist = np.linalg.norm(cand[:, None, :] - placed[None], axis=2)
        ok = (dist >= (rp[None] + radii[i]) * 0.999).all(axis=1)
        cand = cand[ok]
        C[i] = cand[np.argmin(np.linalg.norm(cand - centre, axis=1))]
    return C


def compute(px):
    c = CONFIG
    s = px[c["INDEX"]].dropna()
    eps, open_depth, open_peak = episodes(s, c["MIN_DEPTH"])
    chk = px[c["CHECK"]].dropna()
    eps_chk, _, _ = episodes(chk, c["MIN_DEPTH"])

    years = np.array([e["years"] for e in eps])
    depth = np.array([e["depth"] for e in eps])
    order = np.argsort(years, kind="stable")          # shortest wait first
    radii = np.cbrt(years[order])                      # volume ~ wait
    centres = pack(radii, c["N_DIRS"])
    return {"eps": [eps[i] for i in order], "years": years[order],
            "depth": depth[order], "radii": radii, "centres": centres,
            "open_depth": open_depth, "open_peak": open_peak,
            "eps_chk": eps_chk, "first": s.index[0], "last": s.index[-1],
            "chk_first": chk.index[0]}


# ------------------------------------------------------------ VALIDATE
def validate(d):
    """Every number that reaches a frame is asserted here, before any render."""
    c = CONFIG
    yrs, dep = d["years"], d["depth"]
    n = len(yrs)
    fast = float((yrs * 12 <= c["FAST_MONTHS"]).mean() * 100)
    slow = float((yrs <= c["SLOW_YEARS"]).mean() * 100)
    med_months = float(np.median(yrs) * 12)
    top = d["eps"][-1]
    chk_top = max(d["eps_chk"], key=lambda e: e["years"])
    same_ep = [e for e in d["eps"] if e["peak"].year == chk_top["peak"].year]

    assert n >= 40, f"only {n} episodes"
    assert d["first"].year <= 1929, "index history does not reach 1929"
    assert abs(d["open_depth"]) < c["MIN_DEPTH"], \
        f"an open episode at {d['open_depth']:.1f}% would be hidden"
    assert 40.0 <= fast <= 95.0, f"{fast:.0f}% back within {c['FAST_MONTHS']}m"
    assert fast <= slow <= 100.0, "thresholds out of order"
    assert 0.5 <= med_months <= 24, f"median wait {med_months:.1f} months"
    assert 10.0 <= top["years"] <= 40.0, f"longest wait {top['years']:.1f}y"
    assert top["peak"].year == 1929, "the longest wait is no longer 1929"
    assert -95 <= dep.min() <= -50, f"deepest fall {dep.min():.0f}%"
    assert 2.0 <= chk_top["years"] <= 15.0, f"SPY longest {chk_top['years']:.1f}y"
    assert len(same_ep) == 1 and same_ep[0]["years"] > chk_top["years"], \
        "dividends should shorten the same episode"
    # packing sanity: no two spheres overlap
    C, r = d["centres"], d["radii"]
    D = np.linalg.norm(C[:, None] - C[None], axis=2) + np.eye(n) * 1e9
    assert (D >= (r[:, None] + r[None]) * 0.998).all(), "spheres overlap"

    big3 = d["eps"][-3:][::-1]
    figs = {
        "index_name": "S&P 500",               # label only; history is ^GSPC
        "episodes": str(n),
        "min_depth": f"{c['MIN_DEPTH']:.0f}%",
        "fast": f"{fast:.0f}%",
        "fast_months": str(c["FAST_MONTHS"]),
        "slow": f"{slow:.0f}%",
        "slow_years": str(c["SLOW_YEARS"]),
        "median_months": f"{med_months:.1f}",
        "longest": f"{top['years']:.0f}",
        "longest_exact": f"{top['years']:.1f}",
        "longest_peak": top["peak"].strftime("%b %Y"),
        "longest_back": top["back"].strftime("%b %Y"),
        "longest_depth": f"{top['depth']:.0f}%",
        "deepest": f"{dep.min():.0f}%",
        "first_year": str(d["first"].year),
        "last_year": str(d["last"].year),
        "chk_first": str(d["chk_first"].year),
        "chk_longest": f"{chk_top['years']:.1f}",
        "chk_peak": str(chk_top["peak"].year),
        "chk_back": str(chk_top["back"].year),
        "idx_same": f"{same_ep[0]['years']:.1f}",
        "big_years": " ".join(str(e["peak"].year) for e in big3),
        "big_waits": " ".join(f"{e['years']:.1f}" for e in big3),
    }
    with open(os.path.join(BASE_DIR, "figures.json"), "w") as fh:
        json.dump(figs, fh, indent=1)
    log(f"[validate] {figs}")

    # running readout for the build, from the same sorted waits
    d["run_n"] = [str(i + 1) for i in range(n)]
    d["run_y"] = [f"{y:4.1f}" for y in yrs]
    assert d["run_y"][-1].strip() == figs["longest_exact"], "readout disagrees"
    d["figs"] = figs
    return figs


# ------------------------------------------------------------ GEOMETRY
LIGHT = np.array([-0.45, -0.55, 0.70]) / np.linalg.norm([-0.45, -0.55, 0.70])


def prepare_geometry(d):
    """Unit-sphere meshes per sphere, resolution by size, and the colours."""
    r = d["radii"]
    lo, hi = np.log(CONFIG["MIN_DEPTH"]), np.log(-d["depth"].min())
    d["cnorm"] = (np.log(-d["depth"]) - lo) / (hi - lo)
    meshes = []
    for ri in r:
        nu = int(np.clip(12 + 9 * ri, 16, 40))
        nv = max(nu // 2, 6)
        u = np.linspace(0, 2 * np.pi, nu + 1)
        v = np.linspace(0, np.pi, nv + 1)
        X = np.outer(np.cos(u), np.sin(v))
        Y = np.outer(np.sin(u), np.sin(v))
        Z = np.outer(np.ones_like(u), np.cos(v))
        P = np.stack([X, Y, Z], -1)
        quads = np.stack([P[:-1, :-1], P[1:, :-1], P[1:, 1:], P[:-1, 1:]], 2)
        quads = quads.reshape(-1, 4, 3)
        nrm = quads.mean(1)
        nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
        meshes.append((quads, nrm))
    d["meshes"] = meshes
    C = d["centres"]
    n = len(r)
    b0, b1 = 0.02, 0.80                    # appear times for all but the last
    a = np.linspace(b0, b1, n - 1)
    d["appear"] = np.append(a, 0.83)
    d["ramp"] = np.append(np.full(n - 1, 0.07), 0.17)

    # The camera frames the pack as it is at each moment, so the first small
    # spheres fill the picture too. Bounding box per frame, then a centred
    # moving average so the framing drifts instead of jumping.
    c = CONFIG
    total = int(c["FPS"] * (c["HOOK_SEC"] + c["BUILD_SEC"] + c["HOLD_SEC"]))
    boxes = []
    for i in range(total):
        s = scales(i, total, d)
        m = s > 0.01
        lo3 = (C[m] - (r * s)[m, None]).min(0)
        hi3 = (C[m] + (r * s)[m, None]).max(0)
        boxes.append(np.concatenate([(lo3 + hi3) / 2, [(hi3 - lo3).max() / 2]]))
    B = np.array(boxes)
    B[:, 3] = np.maximum(B[:, 3], 0.4 * B[-1, 3])   # never blow the first few up
    w = int(c["FPS"] * 0.8)
    pad = np.pad(B, ((w, w), (0, 0)), mode="edge")
    ker = np.ones(2 * w + 1) / (2 * w + 1)
    Bs = np.column_stack([np.convolve(pad[:, j], ker, mode="valid")
                          for j in range(4)])
    Bs[-int(c["FPS"] * c["HOLD_SEC"]):] = B[-1]      # hold sits exactly on the end
    d["frame_box"] = Bs
    return d


def ease(x):
    x = np.clip(x, 0, 1)
    return 1 - (1 - x) ** 3


def build_frac(idx, total):
    c = CONFIG
    n_hook = int(c["FPS"] * c["HOOK_SEC"])
    n_build = int(c["FPS"] * c["BUILD_SEC"])
    if idx < n_hook:
        return 0.0
    if idx < n_hook + n_build:
        return (idx - n_hook) / max(n_build - 1, 1)
    return 1.0


def scales(idx, total, d):
    """Growth of every sphere at this frame; the hook shows the first handful."""
    f = max(build_frac(idx, total), 0.10)
    return ease((f - d["appear"]) / d["ramp"])


# -------------------------------------------------------------- RENDER
def render_frame(idx, total, d, out_path):
    c, figs, L = CONFIG, d["figs"], LAYOUT
    if "meshes" not in d:
        prepare_geometry(d)
    frac = build_frac(idx, total)
    t = idx / max(total - 1, 1)

    fig = plt.figure(figsize=(c["W"] / c["DPI"], c["H"] / c["DPI"]),
                     dpi=c["DPI"], facecolor=THEME["BG"])
    fig.text(0.5, L["TITLE"], "IT ALWAYS COMES BACK", ha="center", fontsize=28,
             fontweight="bold", color=THEME["TEXT"], family=THEME["FONT"])
    fig.text(0.5, L["SUBTITLE"],
             f"S&P 500 price index  |  {figs['episodes']} falls of {figs['min_depth']}+  |  "
             f"{figs['first_year']} to {figs['last_year']}",
             ha="center", fontsize=13, color=THEME["ORANGE"], family=THEME["FONT"])
    fig.text(0.5, L["KEY"],
             f"sphere volume: years to a new high   colour: depth, blue -{figs['min_depth']} to red {figs['deepest']}",
             ha="center", fontsize=10, color=THEME["TEXT_DIM"], family=THEME["FONT"])
    fig.text(0.5, L["ROW_A"],
             f"back within {figs['fast_months']} months: {figs['fast']}   "
             f"within {figs['slow_years']} years: {figs['slow']}",
             ha="center", va="center", fontsize=9, color=THEME["TEXT"],
             family=THEME["MONO"])
    fig.text(0.5, L["ROW_B"],
             f"SPY with dividends since {figs['chk_first']}, longest wait: "
             f"{figs['chk_longest']} years ({figs['chk_peak']} to {figs['chk_back']})",
             ha="center", va="center", fontsize=9, color=THEME["CYAN"],
             family=THEME["MONO"])
    fig.text(0.5, L["HANDLE"], "@quant.traderr", ha="center", fontsize=10,
             color=THEME["TEXT_DIM"], alpha=0.75, family=THEME["FONT"])

    ax = fig.add_axes(L["AXES"], projection="3d", facecolor=THEME["BG"])

    # geometry: one merged collection so the painter sorts every face
    C, r = d["centres"], d["radii"]
    scale = scales(idx, total, d)
    polys, fcol, ecol = [], [], []
    for i in np.nonzero(scale > 0.01)[0]:
        quads, nrm = d["meshes"][i]
        polys.append(C[i] + quads * (r[i] * scale[i]))
        base = np.array(CMAP(d["cnorm"][i]))
        lam = np.clip(nrm @ LIGHT, 0, 1)
        shade = 0.28 + 0.72 * lam
        fc = np.empty((len(quads), 4))
        fc[:, :3] = base[:3] * shade[:, None]
        fc[:, 3] = 0.62
        ec = np.empty((len(quads), 4))
        ec[:, :3] = np.clip(base[:3] * 0.6 + 0.4 * shade[:, None] * base[:3] + 0.1, 0, 1)
        ec[:, 3] = 0.55
        fcol.append(fc); ecol.append(ec)
    if polys:
        pc = Poly3DCollection(np.concatenate(polys), facecolors=np.concatenate(fcol),
                              edgecolors=np.concatenate(ecol), linewidths=0.25)
        ax.add_collection3d(pc)

    # labels on the three longest waits, once they are fully grown
    for j, (yr, w) in enumerate(zip(figs["big_years"].split()[:3],
                                    figs["big_waits"].split()[:3])):
        i = len(r) - 1 - j
        if scale[i] > 0.98:
            ax.text(C[i, 0], C[i, 1], C[i, 2] + r[i] * 1.05,
                    f"{yr}: {w} yrs", ha="center", va="bottom",
                    fontsize=12 if j == 0 else 9, fontweight="bold",
                    color=THEME["TEXT"], family=THEME["MONO"], zorder=10)

    box = d["frame_box"][min(idx, len(d["frame_box"]) - 1)]
    mid, half = box[:3], box[3] * 1.02
    ax.set_xlim(mid[0] - half, mid[0] + half)
    ax.set_ylim(mid[1] - half, mid[1] + half)
    ax.set_zlim(mid[2] - half, mid[2] + half)
    ax.view_init(elev=c["ELEV_BASE"] + 6 * np.sin(t * np.pi),
                 azim=c["AZIM_START"] + c["AZIM_SWEEP"] * t)
    ax.set_box_aspect((1, 1, 1), zoom=c["ZOOM"])
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_alpha(0); pane.pane.set_edgecolor((0, 0, 0, 0))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_axis_off()

    # Readout: strings only, precomputed and asserted in validate().
    shown = int((scale > 0.01).sum())
    if frac < 1.0:
        k = max(shown - 1, 0)
        fig.text(0.5, L["NOTE"],
                 f"falls {d['run_n'][k]:>2} of {figs['episodes']}   "
                 f"longest so far {d['run_y'][k]} yrs",
                 ha="center", fontsize=11, color=THEME["TEXT"], family=THEME["MONO"])
    else:
        fig.text(0.5, L["NOTE"],
                 "one sphere per fall: old high, low, new high",
                 ha="center", fontsize=10, color=THEME["TEXT_DIM"],
                 family=THEME["FONT"])
    fig.text(0.5, L["BIG"], f"LONGEST WAIT: {figs['longest']} YEARS",
             ha="center", fontsize=21, color=THEME["RED"], fontweight="bold",
             family=THEME["FONT"])
    fig.text(0.5, L["SUB"],
             f"old high {figs['longest_peak']}, back {figs['longest_back']}",
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
