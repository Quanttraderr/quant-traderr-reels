"""
Gaps_Caption.py
===============
Caption, hashtags and alt text for "Gaps Always Fill?", plus the gate
that decides whether they are allowed to reach disk.

Two rules make this more than a text file:

  1. Every number in CAPTION must appear in figures.json, the file the pipeline
     wrote from asserted computation. A number retyped from memory fails.
  2. The gate runs BEFORE anything is written. A validator that writes first and
     complains second still ships the broken caption.

RUN
    python Gaps_Caption.py
    -> caption.txt, hashtags.txt, alt_text.txt, metadata.json (this folder)
       exit 0 only if every check passes
"""
import json, os, re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(BASE_DIR, "figures.json")
OUT_DIR = BASE_DIR                      # the caption ships next to its reel
HANDLE = "@quant.traderr"
KEYWORD = "GAP"                    # the comment-keyword CTA

CAPTION = """
Retail traders have one rule for a gap at the open: gaps always fill, so fade it. Quants check the tape first.

I took every SPY session since 1993, pulled out each opening gap bigger than 1%, 710 of them, and counted the days until price came back to the prior close.

Only 36% filled the same day. 66% within a week, 81% within a month, and 8% were still open a full year later. Gap downs filled the same day 41% of the time, gap ups only 31%. The biggest, -10% in Mar 2020, is exactly the gap you did not want to fade.

The catch: count every tiny gap and 67% fill the same day. That is where the saying comes from, and it breaks down on the big gaps that feel like a trade. "Filled" also ignores the path: a gap that fills after a deep drawdown counts the same as one that fills by lunch.

Comment "GAP" and I'll send you the code.

Follow @quant.traderr for the math behind the markets.
"""

HASHTAGS = ('#daytrading #trading #stockmarket #spy #quant #gapfill #technicalanalysis #quantfinance #python #investing #finance')

ALT = (
    'Slide 1: A field of glowing arches stands on a dark floor, one arch for every SPY opening gap bigger than 1% since 1993, laid out by date from 2000 to 2020. Each arch springs from a centre line and lands where the gap was filled: gap ups to one side, gap downs to the other, the span growing with the days it took, marked at one month and one year. Tall narrow spikes are gaps filled the same day, wide orange arches took months, and red half arches with a dot never came down within a year. Text reads only 36% filled the same day, 8% were still open a year later.'
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
    for key in ('share_same', 'share_week', 'share_month', 'share_open', 'share_all_same', 'n_gaps', 'biggest', 'start'):
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
