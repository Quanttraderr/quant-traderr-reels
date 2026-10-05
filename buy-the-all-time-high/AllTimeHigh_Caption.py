"""
AllTimeHigh_Caption.py
======================
Caption, hashtags and alt text for "Buy the All-Time High?", plus the gate
that decides whether they are allowed to reach disk.

Two rules make this more than a text file:

  1. Every number in CAPTION must appear in figures.json, the file the pipeline
     wrote from asserted computation. A number retyped from memory fails.
  2. The gate runs BEFORE anything is written. A validator that writes first and
     complains second still ships the broken caption.

RUN
    python AllTimeHigh_Caption.py
    -> caption.txt, hashtags.txt, alt_text.txt, metadata.json (this folder)
       exit 0 only if every check passes
"""
import json, os, re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(BASE_DIR, "figures.json")
OUT_DIR = BASE_DIR                      # the caption ships next to its reel
HANDLE = "@quant.traderr"
KEYWORD = "HIGH"                    # the comment-keyword CTA

CAPTION = """
Most traders see a new all-time high and think one thing: too late, it has to come down. Quants don't guess. They count.

I took every S&P record close since 1950, 1,310 of them, and checked where the index stood exactly one year later.

73% of all-time highs were higher a year later. On any random day since 1950 the figure is 75%. The median year after a record: +10%, against +11% for all days. Quants call this the base rate: a record close tells you almost nothing about the next 12 months.

The catch: highs are not a buy signal either, they are simply no worse than any other day. They come in clusters, so 1,310 pillars are far fewer independent bets than they look. The worst, Oct 2007, was -37% a year later. And this is the price index, without dividends and before inflation.

Comment "HIGH" and I'll send you the code.

Follow @quant.traderr for the math behind the markets.
"""

HASHTAGS = ('#investing #stockmarket #sp500 #quant #alltimehigh #finance #trading #quantfinance #python #personalfinance #wealth')

ALT = (
    "Slide 1: A forest of thin glowing pillars stands on a dark floor, one pillar for every S&P 500 all-time high since 1950, laid out by year from left to right with decade marks from 1960 to 2020. Each pillar's height is the index return over the following twelve months: blue to orange pillars rise above the floor, red pillars hang below it for highs that were lower a year later. Empty clearings mark years without a new high. A red tag marks the deepest one, Oct 2007 at -37%. Text reads 73% higher a year later, any random day 75%."
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
    for key in ('share_up_ath', 'share_up_all', 'n_ath', 'median_ath', 'median_all', 'worst', 'worst_date', 'start'):
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
