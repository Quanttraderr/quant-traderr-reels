"""
CorrelationSpace_Caption.py
===========================
Caption, hashtags and alt text for one reel, plus the gate that decides whether
they are allowed to reach disk.

Two rules make this more than a text file:

  1. Every number in CAPTION must appear in figures.json, the file the pipeline
     wrote from asserted computation. A number retyped from memory fails.
  2. The gate runs BEFORE anything is written. A validator that writes first and
     complains second still ships the broken caption.

RUN
    python CorrelationSpace_Caption.py
    -> caption.txt, hashtags.txt, alt_text.txt, metadata.json (this folder)
       exit 0 only if every check passes
"""
import json, os, re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(BASE_DIR, "figures.json")
OUT_DIR = BASE_DIR                      # the caption ships next to its reel
HANDLE = "@quant.traderr"
KEYWORD = "SPACE"                       # the comment-keyword CTA

CAPTION = """
Most traders keep a watchlist. A quant keeps a metric space, because once every pair of names carries a distance, the market stops being a list and starts having a shape.

44 US large caps, 751 sessions of daily returns. Every pair gets a correlation, every correlation becomes a distance, and the 43 links are the minimum spanning tree: the cheapest wiring that still reaches every name.

Average pairwise correlation is 0.18, so most of these names sit further apart than a sector label suggests. The tightest link in the whole basket is GS/MS at 0.86. BLK lands nearest the middle of everything, UNH sits furthest out, and nobody drew either of those positions.

The catch: three dimensions hold only 35% of the distance structure, so nearness on screen is an approximation, not a measurement. Correlation is not causation, and none of this forecasts anything.

Comment "SPACE" and I'll send the code that builds it.

Follow @quant.traderr for daily breakdowns of the math behind the markets.
"""

HASHTAGS = ("#quant #trading #finance #markets #correlation #quantfinance "
            "#algotrading #datascience #python #investing")

ALT = (
    "Slide 1: A 3D point cloud on a black background. Each dot is one US large "
    "cap stock, placed so that the distance between any two dots reflects how "
    "little their daily returns move together. Dots are coloured blue through "
    "red by how much a name moves with the rest of the basket. Thin coloured "
    "lines wire the dots into a minimum spanning tree, and clusters of "
    "technology, banking, energy and defensive names separate without any "
    "sector label being used."
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
    # must appear inside some figures.json value. Backward: every headline
    # figure must survive an edit, so a trim cannot quietly drop the evidence.
    pool = " | ".join(figs.values())
    for tok in NUM.findall(text):
        if tok not in pool:
            problems.append(f"quoted figure {tok!r} is not in figures.json")
    for key in ("mean_corr", "top_pair", "top_pair_corr", "hub_name",
                "rim_name", "n_names", "n_sessions", "n_links", "mds_share"):
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
