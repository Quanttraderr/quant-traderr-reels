"""
TenDays_Caption.py
==================
Caption, hashtags and alt text for one reel, plus the gate that decides whether
they are allowed to reach disk.

Two rules make this more than a text file:

  1. Every number in CAPTION must appear in figures.json, the file the pipeline
     wrote from asserted computation. A number retyped from memory fails.
  2. The gate runs BEFORE anything is written. A validator that writes first and
     complains second still ships the broken caption.

RUN
    python TenDays_Caption.py
    -> caption.txt, hashtags.txt, alt_text.txt, metadata.json (this folder)
       exit 0 only if every check passes
"""
import json, os, re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(BASE_DIR, "figures.json")
OUT_DIR = BASE_DIR                      # the caption ships next to its reel
HANDLE = "@quant.traderr"
KEYWORD = "TEN"                         # the comment-keyword CTA

CAPTION = """
Retail plans to step aside before the crash and step back in once it is over. The sessions that carried this decade were almost all inside the crash.

SPY, 2,511 sessions, dividends reinvested. The bright coil compounds every one of them. The dim coil is the same decade with the 10 best sessions set to zero and every other day left exactly where it was.

313% becomes 118%. A handful of days carried 62% of the gain, and they were not waiting in the calm stretches: 90% of them landed while the index was more than 10% below its previous high. The best one, Apr 2025, paid 10.5% in a single session.

The catch: nobody can remove only the best days, and dodging the worst ones is just as extreme and is not shown here. This is an argument about how hard timing is, not proof that holding always wins.

Comment "TEN" and I'll send the code that builds it.

Follow @quant.traderr for daily breakdowns of the math behind the markets.
"""

HASHTAGS = ("#quant #trading #finance #markets #investing #quantfinance "
            "#algotrading #datascience #python #stockmarket")

ALT = (
    "Slide 1: Two glowing coils climbing through a black 3D space, side by "
    "side. Height is the growth of one dollar in the S&P 500 ETF over ten "
    "years. The left coil is bright and coloured blue through red by the size "
    "of each session's move; the right coil is dim and is the same decade with "
    "its ten best sessions removed. Faint rungs link the two coils in time and "
    "fan apart as the gap between them grows, and the left coil ends far above "
    "the right one."
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
    for key in ("gain_all", "gain_ex", "share_lost", "n_missed", "in_drawdown",
                "dd_threshold", "best_date", "best_move", "n_sessions"):
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
