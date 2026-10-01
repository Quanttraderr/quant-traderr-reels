"""
BuyTheFear_Caption.py
=====================
Caption, hashtags and alt text for "Buy When It Hurts", plus the gate
that decides whether they are allowed to reach disk.

Two rules make this more than a text file:

  1. Every number in CAPTION must appear in figures.json, the file the pipeline
     wrote from asserted computation. A number retyped from memory fails.
  2. The gate runs BEFORE anything is written. A validator that writes first and
     complains second still ships the broken caption.

RUN
    python BuyTheFear_Caption.py
    -> caption.txt, hashtags.txt, alt_text.txt, metadata.json (this folder)
       exit 0 only if every check passes
"""
import json, os, re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(BASE_DIR, "figures.json")
OUT_DIR = BASE_DIR                      # the caption ships next to its reel
HANDLE = "@quant.traderr"
KEYWORD = "FEAR"                    # the comment-keyword CTA

CAPTION = """
Most traders sell when the VIX spikes and the headlines scream crash. Quants check what happened next, and the tape says those were some of the best days to buy.

Take all 8,223 SPY trading days from 1993 to Sep 2025, sort them by where the VIX closed, and ask one question: where was SPY one year later?

On an average day the answer is +12%, up in 82% of cases. On the 208 days the VIX closed at 40 or higher, the next year returned +36% and was up in 97% of them. Calm days with the VIX under 15 returned +13%, barely better than average. Quants call it the fear premium: you get paid for holding what nobody else wants to hold.

The catch: those 208 days are not 208 independent bets. They cluster in just 10 crisis years, so the real sample is small. The worst entry, Sep 2001, still returned -16% over the next year. Buying fear worked on average, not every time.

Comment "FEAR" and I'll send you the code.

Follow @quant.traderr for the math behind the markets.
"""

HASHTAGS = ('#quant #trading #stockmarket #investing #sp500 #vix #quantfinance #algotrading #finance #python #marketcrash')

ALT = (
    "Slide 1: A 3D field of Voronoi crystals on a black background, one cell per month of SPY from 1993 to 2025. Cells further back sat on a higher VIX, and each cell's height and colour show SPY's return over the following year, blue for losses and orange to red for big gains. The tallest orange towers have white edges and sit beyond a white line marked VIX 40. Text reads VIX 40+ to +36% next year, up in 97% of cases, average day +12%."
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
    for key in ('fear_mean', 'all_mean', 'fear_pos', 'n_fear', 'fear_worst', 'start', 'end'):
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
