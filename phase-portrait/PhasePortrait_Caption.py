"""
PhasePortrait_Caption.py
========================
Caption, hashtags and alt text for one reel, plus the gate that decides whether
they are allowed to reach disk.

Two rules make this more than a text file:

  1. Every number in CAPTION must appear in figures.json, the file the pipeline
     wrote from asserted computation. A number retyped from memory fails.
  2. The gate runs BEFORE anything is written. A validator that writes first and
     complains second still ships the broken caption.

RUN
    python PhasePortrait_Caption.py
    -> caption.txt, hashtags.txt, alt_text.txt, metadata.json (this folder)
       exit 0 only if every check passes
"""
import json, os, re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(BASE_DIR, "figures.json")
OUT_DIR = BASE_DIR                      # the caption ships next to its reel
HANDLE = "@quant.traderr"
KEYWORD = "STATE"                       # the comment-keyword CTA

CAPTION = """
Most traders watch one line go up and down. A quant watches the state the market is in, because the same price can sit in a calm tape or a violent one, and those are not the same trade.

Three numbers describe SPY on any session: its 20 day momentum, its 20 day realised volatility, and how far it sits below its previous high. Read as coordinates, 1,234 sessions stop being a chart and become one path through a space.

That path spends 87% of its days below the old high, and at its worst it sat 24.5% under. It also refuses to move at a steady rate. It crawls for months, then bolts: on its fastest day, Apr 2025, it covered 14x the ground of a typical session, with volatility along the way running from 5% to 54%.

The catch: three features are a choice, not the market, and a loop here is not a cycle you can trade. The windows overlap, so neighbouring days are not independent.

Comment "STATE" and I'll send the code that builds it.

Follow @quant.traderr for daily breakdowns of the math behind the markets.
"""

HASHTAGS = ("#quant #trading #finance #markets #volatility #quantfinance "
            "#algotrading #datascience #python #investing")

ALT = (
    "Slide 1: A glowing path winding through a black 3D space. The three axes "
    "are 20 day momentum, 20 day realised volatility and drawdown from the "
    "previous high, so each point on the path is one trading session of the "
    "S&P 500 ETF. The path is coloured blue through red by how far the market "
    "moved between one session and the next, and a faint grid marks the level "
    "of the previous high, with almost all of the path hanging below it."
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
    for key in ("pct_underwater", "speed_ratio", "fastest_date", "deepest_dd",
                "peak_vol", "calm_vol", "n_sessions", "window_days"):
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
