"""
Drawdown_Caption.py
===================
Caption, hashtags and alt text for one reel, plus the gate that decides whether
they are allowed to reach disk.

Two rules make this more than a text file:

  1. Every number in CAPTION must appear in figures.json, the file the pipeline
     wrote from asserted computation. A number retyped from memory fails.
  2. The gate runs BEFORE anything is written. A validator that writes first and
     complains second still ships the broken caption.

RUN
    python Drawdown_Caption.py
    -> caption.txt, hashtags.txt, alt_text.txt, metadata.json (this folder)
       exit 0 only if every check passes
"""
import json, os, re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(BASE_DIR, "figures.json")
OUT_DIR = BASE_DIR                      # the caption ships next to its reel
HANDLE = "@quant.traderr"
KEYWORD = "UNDERWATER"                  # the comment-keyword CTA

CAPTION = """
Retail measures Bitcoin in returns. A quant measures it in time spent underwater, because the return is what you get at the end and the drawdown is what you have to sit through to get there.

Every day since 2014 for BTC and SPY, each measured against its own running high. The width of a funnel at any depth is the share of days spent at least that far down, so the two start at the same rim and then part company.

BTC spent 32% of its days more than 50% below its high. SPY spent 0% of the same span down there. At the 20% mark it is 66% against 2%. BTC was sitting at a new high on 4% of days, SPY on 15%. Worst points: -83% against -34%.

The catch: this is the cost side on its own. Over these same years BTC returned far more than SPY and none of that is in the picture. The two also trade on different calendars, 4,388 days against 3,020, so these are shares of each asset's own days.

Comment "UNDERWATER" and I'll send the code that builds it.

Follow @quant.traderr for daily breakdowns of the math behind the markets.
"""

HASHTAGS = ("#bitcoin #btc #quant #trading #finance #markets #crypto "
            "#quantfinance #datascience #python #investing")

ALT = (
    "Slide 1: Two funnels hanging in a black 3D space, seen from the side. "
    "Each one is an asset and depth is how far below its own high it is. The "
    "width at a depth is the share of days spent at least that far down. The "
    "Bitcoin funnel is coloured blue at the rim through red at the bottom and "
    "reaches far down, while the grey S&P funnel narrows to a thin needle "
    "near the top."
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
    # must appear inside some figures.json value, and the sign is part of the
    # token, so "-20%" would fail where the file holds "20%". Backward: every
    # headline figure must survive an edit.
    pool = " | ".join(figs.values())
    for tok in NUM.findall(text):
        if tok not in pool:
            problems.append(f"quoted figure {tok!r} is not in figures.json")
    for key in ("a_deep", "b_deep", "a_mid", "b_mid", "a_worst", "b_worst",
                "a_high", "b_high", "a_days", "b_days", "first_year"):
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
                   "title": "The Shape of Pain",
                   "caption_chars": len(text),
                   "n_hashtags": len(HASHTAGS.split()),
                   "figures": figs}, f, indent=2)
    print(f"caption ok  ({len(text)} chars, {len(HASHTAGS.split())} hashtags)")


if __name__ == "__main__":
    main()
