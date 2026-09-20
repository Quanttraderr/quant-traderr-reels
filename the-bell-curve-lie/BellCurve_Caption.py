"""
BellCurve_Caption.py
====================
Caption, hashtags and alt text for one reel, plus the gate that decides whether
they are allowed to reach disk.

Two rules make this more than a text file:

  1. Every number in CAPTION must appear in figures.json, the file the pipeline
     wrote from asserted computation. A number retyped from memory fails.
  2. The gate runs BEFORE anything is written. A validator that writes first and
     complains second still ships the broken caption.

RUN
    python BellCurve_Caption.py
    -> caption.txt, hashtags.txt, alt_text.txt, metadata.json (this folder)
       exit 0 only if every check passes
"""
import json, os, re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(BASE_DIR, "figures.json")
OUT_DIR = BASE_DIR                      # the caption ships next to its reel
HANDLE = "@quant.traderr"
KEYWORD = "TAILS"                       # the comment-keyword CTA

CAPTION = """
Retail hears that markets are random and pictures a bell curve. A quant fits the bell curve, draws the boundary it says nothing should ever cross, and then plots the days that crossed it anyway.

One dot is one session: the returns of SPY, TLT and GLD as three axes, 5,491 sessions across 22 years. The shell is the 5 sigma surface of the normal distribution fitted to those very days. Over a sample this size that model expects 0.1 days outside it. There are 63.

The worst sits at 11.8 sigma, Oct 2008. Under the same model that is a one in 10^27 years event, and the universe has been around for rather less than that. This is not a distribution that is slightly wrong in the tail. It is wrong by orders of magnitude.

The catch: a normal distribution with one frozen covariance matrix is a straw man, and volatility clustering explains a lot of this without anything exotic. The fit is in sample, and three assets are not a market.

Comment "TAILS" and I'll send the code that builds it.

Follow @quant.traderr for daily breakdowns of the math behind the markets.
"""

HASHTAGS = ("#quant #trading #finance #markets #riskmanagement #quantfinance "
            "#algotrading #datascience #python #investing")

ALT = (
    "Slide 1: A 3D scatter on a black background. Each dot is one trading day, "
    "positioned by the returns of SPY, TLT and GLD, coloured blue in the "
    "middle through red at the edge. A faint cyan wireframe ellipsoid marks "
    "the 5 sigma surface of a fitted normal distribution, and dozens of red "
    "dots sit well outside it."
)

NUM = re.compile(r"[+-]?\d(?:[\d,]*\d)?(?:\.\d+)?%?")   # trailing comma is punctuation


def _checks(text, tags_line, figs):
    """Everything that can make a caption unpostable, in one place."""
    problems = []
    tags = tags_line.split()

    if "—" in text or "–" in text:
        problems.append("contains an em or en dash")
    if "--" in text:
        problems.append("contains a double hyphen standing in for an em dash")
    if KEYWORD not in text:
        problems.append(f"missing the comment keyword {KEYWORD!r}")
    if HANDLE not in text:
        problems.append(f"missing the follow handle {HANDLE!r}")
    if not 800 <= len(text) <= 1200:
        problems.append(f"length {len(text)} outside the house 800-1200 band")
    if not 3 <= len(tags) <= 15:
        problems.append(f"{len(tags)} hashtags, want 3 to 15")
    if len(set(tags)) != len(tags):
        problems.append("duplicate hashtags")
    if [t for t in tags if not t.startswith("#")]:
        problems.append(f"hashtags missing '#': {[t for t in tags if not t.startswith('#')]}")
    if any(t != t.lower() for t in tags):
        problems.append("hashtags must be lowercase")
    if not ALT.strip().startswith("Slide 1:"):
        problems.append("alt text must start with 'Slide 1:'")

    # Figure gate, both directions. Forward: every numeric token in the caption
    # must appear inside some figures.json value, so "Oct 2008" in the file
    # covers a bare "2008" in the prose and "10^27" covers both of its halves.
    # Backward: every headline figure must survive an edit, so a trim cannot
    # quietly drop the evidence.
    pool = " | ".join(figs.values())
    for tok in NUM.findall(text):
        if tok not in pool:
            problems.append(f"quoted figure {tok!r} is not in figures.json")
    for key in ("n_outside", "exp_outside", "worst_sigma", "worst_date",
                "shell_k", "n_days", "n_years", "odds_years"):
        if figs[key] not in text:
            problems.append(f"{key} = {figs[key]!r} lost in the trim")
    return problems


def main():
    text = CAPTION.strip()
    if not os.path.exists(FIGURES):
        raise SystemExit("figures.json missing; run the reel pipeline first")
    with open(FIGURES) as fh:
        figs = json.load(fh)

    problems = _checks(text, HASHTAGS, figs)

    if problems:                       # bail BEFORE writing, never after
        print("PROBLEMS:")
        for p in problems:
            print("  -", p)
        raise SystemExit(1)

    os.makedirs(OUT_DIR, exist_ok=True)
    for name, body in (("caption.txt", text), ("hashtags.txt", HASHTAGS),
                       ("alt_text.txt", ALT.strip())):
        with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8") as f:
            f.write(body + "\n")
        print("  wrote", name, len(body), "chars")
    with open(os.path.join(OUT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"handle": HANDLE, "keyword": KEYWORD,
                   "title": "The Bell Curve Lie",
                   "caption_chars": len(text),
                   "n_hashtags": len(HASHTAGS.split()),
                   "figures": figs}, f, indent=2)
    print(f"caption ok  ({len(text)} chars, {len(HASHTAGS.split())} hashtags)")


if __name__ == "__main__":
    main()
