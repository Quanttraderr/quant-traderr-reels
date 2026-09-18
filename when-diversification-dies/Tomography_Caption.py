"""
Tomography_Caption.py
=====================
Caption, hashtags and alt text for one reel, plus the gate that decides whether
they are allowed to reach disk.

Two rules make this more than a text file:

  1. Every number in CAPTION must appear in figures.json, the file the pipeline
     wrote from asserted computation. A number retyped from memory fails.
  2. The gate runs BEFORE anything is written. A validator that writes first and
     complains second still ships the broken caption.

RUN
    python Tomography_Caption.py
    -> caption.txt, hashtags.txt, alt_text.txt, metadata.json (this folder)
       exit 0 only if every check passes
"""
import json, os, re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(BASE_DIR, "figures.json")
OUT_DIR = BASE_DIR                      # the caption ships next to its reel
HANDLE = "@quant.traderr"
KEYWORD = "MATRIX"                      # the comment-keyword CTA

CAPTION = """
Retail buys ten different things and calls it diversified. A quant asks what those ten things do on the days that actually matter.

Ten funds covering US and foreign equity, government bonds, credit, gold and property. 3,711 rolling 60-day windows, each one a correlation matrix, stacked here as plates with time running upward.

Average pairwise correlation is 0.37 in a calm tape. When the index sits more than 15% below its high it is 0.56. The protection thins out at exactly the moment you are paying for it. The calmest reading of the whole period, 0.14, landed in Feb 2020, weeks before the fastest crash in the sample. The hottest, 0.70, came in May 2026.

The catch: 10 funds are not the market, correlation is linear dependence and says nothing about tails, and one calm reading before one crash is an observation, not a signal.

Comment "MATRIX" and I'll send the code that builds it.

Follow @quant.traderr for daily breakdowns of the math behind the markets.
"""

HASHTAGS = ("#quant #trading #finance #markets #correlation #portfolio "
            "#quantfinance #algotrading #datascience #investing")

ALT = (
    "Slide 1: A tall stack of translucent square plates glowing in a black 3D "
    "space. Each plate is a correlation matrix of ten funds over a 60-day "
    "window, drawn as a grid of coloured cells from blue for low correlation "
    "to red for high, with the diagonal left empty. The plates are stacked by "
    "date so height is time, and each plate has a frame coloured by its own "
    "average correlation, so the stack shows warm bands during market "
    "drawdowns and cooler ones in calm periods."
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
    if "The catch:" not in text:
        problems.append("missing the 'The catch:' line")
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
    # must appear inside some figures.json value, so "Apr 2025" in the file
    # covers a bare "2025" in the prose. Backward: every headline figure must
    # survive an edit, so a trim cannot quietly drop the evidence.
    pool = " | ".join(figs.values())
    for tok in NUM.findall(text):
        if tok not in pool:
            problems.append(f"quoted figure {tok!r} is not in figures.json")
    for key in ("calm_corr", "crisis_corr", "min_corr", "min_date", "max_corr",
                "max_date", "n_assets", "n_windows", "window_days",
                "dd_threshold"):
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
                   "caption_chars": len(text),
                   "n_hashtags": len(HASHTAGS.split()),
                   "figures": figs}, f, indent=2)
    print(f"caption ok  ({len(text)} chars, {len(HASHTAGS.split())} hashtags)")


if __name__ == "__main__":
    main()
