"""
WinnersCrashed_Caption.py
=========================
Caption, hashtags and alt text for "Every Winner Crashed Hard", plus the
gate that decides whether they are allowed to reach disk.

Two rules make this more than a text file:

  1. Every number in CAPTION must appear in figures.json, the file the pipeline
     wrote from asserted computation. A number retyped from memory fails.
  2. The gate runs BEFORE anything is written. A validator that writes first and
     complains second still ships the broken caption.

RUN
    python WinnersCrashed_Caption.py
    -> caption.txt, hashtags.txt, alt_text.txt, metadata.json (this folder)
       exit 0 only if every check passes
"""
import json, os, re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(BASE_DIR, "figures.json")
OUT_DIR = BASE_DIR                      # the caption ships next to its reel
HANDLE = "@quant.traderr"
KEYWORD = "CRASH"                       # the comment-keyword CTA

CAPTION = """
Retail sees a stock cut in half and calls it dead money. Quants ask a different question: what did the biggest winners look like before they won?

I pulled every daily close for 7 mega winners from their IPO, AAPL, MSFT, AMZN, NVDA, NFLX, TSLA and META, and measured the deepest drawdown from a prior high. Every single one hit -69% or worse.

AMZN fell -94% into Sep 2001, then went 859x from that low, 2,617x since IPO. NVDA dropped -90% by Oct 2002 and is up 4,259x from that trough. AAPL sat at -82% in Apr 2003. Even MSFT, the calm one, was down -69% in Mar 2009.

That is drawdown tolerance: the return only went to whoever could sit through the icicles.

The catch: survivorship bias. I picked these 7 because they won, after the fact. Plenty of stocks that crashed just as hard never came back, and they are not in this picture. A crash is not a buy signal. It is a price of admission you only see in hindsight.

Comment "CRASH" and I'll send you the code.

Follow @quant.traderr for the math behind the markets.
"""

HASHTAGS = ("#quant #stockmarket #investing #drawdown #amazon #nvidia "
            "#survivorshipbias #quantfinance #python #trading #longterminvesting")

ALT = (
    "Slide 1: A glowing 3D cave on a black background, seven lanes in depth, one "
    "per stock: AAPL, MSFT, AMZN, NVDA, NFLX, TSLA and META, with time running "
    "left to right from 1980 to 2026. Above a dark floor each lane has a "
    "translucent wall whose height is the log of growth since the IPO, coloured "
    "blue to yellow. Below the floor hang orange to red icicles showing the "
    "drawdown from each stock's prior high. Red labels mark the deepest crash in "
    "every lane, from -69% for MSFT to -94% for AMZN. Text reads AMZN -94%, then "
    "859x, crash to Sep 2001, 2,617x since IPO."
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
    for key in ("amzn_dd", "amzn_trough", "amzn_rebound", "amzn_total", "mildest_crash", "n_stocks"):
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
