"""
LumpSum_Caption.py
==================
Caption, hashtags and alt text for one reel, plus the gate that decides whether
they are allowed to reach disk.

Two rules make this more than a text file:

  1. Every number in CAPTION must appear in figures.json, the file the pipeline
     wrote from asserted computation. A number retyped from memory fails.
  2. The gate runs BEFORE anything is written. A validator that writes first and
     complains second still ships the broken caption.

RUN
    python LumpSum_Caption.py
    -> caption.txt, hashtags.txt, alt_text.txt, metadata.json (this folder)
       exit 0 only if every check passes
"""
import json, os, re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(BASE_DIR, "figures.json")
OUT_DIR = BASE_DIR                      # the caption ships next to its reel
HANDLE = "@quant.traderr"
KEYWORD = "DCA"                   # the comment-keyword CTA

CAPTION = """
Retail drips a lump sum into the market over 12 months so it never buys the top. A quant asks what the waiting cost.

Every start day of SPY since 1993: invest it all at once, or split it into 12 equal buys, one every 21 trading days. 8,221 start days, compared at the end of month 12. Each staircase is one start date, the glass block is the lump sum.

The lump sum finished ahead on 79% of start days, with a median lead of +6.3% of the starting money. The market rose in most months, and the drip spends most of the year partly in cash. Pay the waiting cash the T-bill rate and the lump sum still wins 75% of the time.

The catch: the drip won in the crashes, and by a lot. Start in Aug 2008 and it finished 34% ahead. The start days overlap, so this is one index over one stretch of history, not 8,221 independent trials. And if a lump sum would make you panic sell, the drip is the better plan.

Comment "DCA" and I'll send the code that builds it.

Follow @quant.traderr for daily breakdowns of the math behind the markets.
"""

HASHTAGS = ('#dca #dollarcostaveraging #lumpsum #investing #quant #spy #sp500 #quantfinance #personalfinance #markets #python #datascience')

ALT = (
    'Slide 1: A black 3D space with a translucent cyan glass block standing for a lump sum invested on day one. Inside it rises a staircase of 12 steps, one step per monthly buy, extruded into depth so that every start date of the S&P 500 since 1993 is one thin slice. Each step is coloured by who is ahead at the end of that month: orange where the lump sum leads, green where the drip leads. Orange dominates; green bands cluster around the 2000 and 2008 crashes.'
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
    for key in ('starts', 'lump_win', 'lump_win_tb', 'median_lead', 'dca_best', 'dca_best_start', 'first_year', 'months', 'step'):
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
                   "title": "Lump Sum Beats the Drip",
                   "caption_chars": len(text),
                   "n_hashtags": len(HASHTAGS.split()),
                   "figures": figs}, f, indent=2)
    print(f"caption ok  ({len(text)} chars, {len(HASHTAGS.split())} hashtags)")


if __name__ == "__main__":
    main()
