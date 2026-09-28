"""
LeverageDecay_Caption.py
===================
Caption, hashtags and alt text for one reel, plus the gate that decides whether
they are allowed to reach disk.

Two rules make this more than a text file:

  1. Every number in CAPTION must appear in figures.json, the file the pipeline
     wrote from asserted computation. A number retyped from memory fails.
  2. The gate runs BEFORE anything is written. A validator that writes first and
     complains second still ships the broken caption.

RUN
    python LeverageDecay_Caption.py
    -> caption.txt, hashtags.txt, alt_text.txt, metadata.json (this folder)
       exit 0 only if every check passes
"""
import json, os, re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(BASE_DIR, "figures.json")
OUT_DIR = BASE_DIR                      # the caption ships next to its reel
HANDLE = "@quant.traderr"
KEYWORD = "DECAY"                 # the comment-keyword CTA

CAPTION = """
Retail buys a 3x ETF and expects three times the market's year. A quant asks what resetting the leverage every day does to that promise.

This is volatility decay: a daily 3x product compounds three times each day's move, not three times the year.

Every one-year window of SPY since 1993, started on every trading day: 8,220 windows. For each one I rebuilt a 3x daily product from real SPY returns and held it against three times the year SPY actually had.

In 55% of windows, 3x daily fell short. Medians across all windows: SPY x3 said +43%, 3x daily delivered +38%. The swing decides it: in the calmer half of years, under about 15% volatility, 35% fell short. In the wilder half, 74% did. In 5% of windows SPY rose and the 3x product still lost money. Worst gap: -60 points, in the window starting 2008.

The catch: this is a calculation on real data, not a real fund. No fees, no financing costs, and both only make it worse. The windows overlap, so this is one index, not 8,220 independent trials. And in calm trending years 3x daily beat the promise.

Comment "DECAY" and I'll send the code that builds it.

Follow @quant.traderr for daily breakdowns of the math behind the markets.
"""

HASHTAGS = ('#leveragedetf #volatility #trading #quant #spy #sp500 #quantfinance #investing #etf #markets #python #datascience')

ALT = (
    'Slide 1: A black 3D space. A faint cyan grid marks zero, the return a 3x product would have if it simply delivered three times the S&P 500 year. Hanging below it is a glass lens with a flat top and a curved belly, deepening toward the high volatility side and glowing red at its lowest point: the pocket where daily rebalancing loses to the promise. Thousands of small dots, one per one-year window since 1993, fill in by date and trace a smile shape: the dots settle into the lens around small and medium years, while strong calm rallies rise above the glass in cyan and blue.'
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
    # Whole tokens, not substrings: a substring test let "27" pass because
    # "1927" is in the file. Every value is cut into the same number tokens.
    pool = set(NUM.findall(" | ".join(figs.values())))
    for tok in NUM.findall(text):
        if tok not in pool:
            problems.append(f"quoted figure {tok!r} is not in figures.json")
    for key in ('windows', 'below', 'calm', 'wild', 'vol_split', 'lost_up', 'med_prom', 'med_lev', 'worst', 'worst_year', 'first_year'):
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
                   "title": "The 3x Illusion",
                   "caption_chars": len(text),
                   "n_hashtags": len(HASHTAGS.split()),
                   "figures": figs}, f, indent=2)
    print(f"caption ok  ({len(text)} chars, {len(HASHTAGS.split())} hashtags)")


if __name__ == "__main__":
    main()
