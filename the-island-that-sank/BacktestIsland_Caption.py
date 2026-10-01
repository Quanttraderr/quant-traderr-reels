"""
BacktestIsland_Caption.py
=========================
Caption, hashtags and alt text for "The Island That Sank", plus the gate
that decides whether they are allowed to reach disk.

Two rules make this more than a text file:

  1. Every number in CAPTION must appear in figures.json, the file the pipeline
     wrote from asserted computation. A number retyped from memory fails.
  2. The gate runs BEFORE anything is written. A validator that writes first and
     complains second still ships the broken caption.

RUN
    python Topic_Caption.py
    -> caption.txt, hashtags.txt, alt_text.txt, metadata.json (this folder)
       exit 0 only if every check passes
"""
import json, os, re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(BASE_DIR, "figures.json")
OUT_DIR = BASE_DIR                      # the caption ships next to its reel
HANDLE = "@quant.traderr"
KEYWORD = "ISLAND"                   # the comment-keyword CTA

CAPTION = """
Retail finds a moving average crossover that crushed the backtest and calls it an edge. A quant asks one thing first: what happened after the backtest ended?

I tested 6,032 versions of the classic trend rule on SPY. Fast average, slow average, a buffer band, T-bills when out.

From 1994 to 2009, 79% of them beat buy and hold. The best, a 33/226-day MA with a 0.4% band, made 13.6% a year against 7.8%. That green solid is the island.

From 2010 to 2026, 0 of 6,032 beat buy and hold. The best backtest made 9.2% a year while simply holding made 14.2%. 12 rules only tied, because they never sold. This is what overfitting to a regime looks like.

The catch: 2010 to 2026 was a long bull market with short crashes, the worst case for a slow trend filter, and the rule still cut drawdowns, which this score ignores. One asset, one split date, no costs.

Comment "ISLAND" and I'll send you the code.

Follow @quant.traderr for the math behind the markets.
"""

HASHTAGS = ("#quant #trading #backtesting #algotrading #quantfinance "
            "#stockmarket #investing #sp500 #finance #python #datascience")

ALT = (
    "Slide 1: A 3D box on a black background. Its axes are the fast moving "
    "average, the slow moving average and a buffer band of a trend rule. A "
    "large solid fills most of the box: the settings that beat buy and hold "
    "from 1994 to 2009. The solid then breaks apart and vanishes as the test "
    "moves past 2010, and every setting turns into a red dot. Text reads 79% "
    "beat it, then 0% beat it."
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
    # must appear inside some figures.json value, so "Mar 2026" in the file
    # covers a bare "2026" in the prose. Backward: every headline figure must
    # survive an edit, so a trim cannot quietly drop the evidence.
    # Whole tokens, not substrings: a substring test let "27" pass because
    # "1927" is in the file. Every value is cut into the same number tokens.
    pool = set(NUM.findall(" | ".join(figs.values())))
    for tok in NUM.findall(text):
        if tok not in pool:
            problems.append(f"quoted figure {tok!r} is not in figures.json")
    for key in ("n_settings", "share_is", "best_is", "best_os", "bh_os", "bh_is", "n_beat_os"):
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
