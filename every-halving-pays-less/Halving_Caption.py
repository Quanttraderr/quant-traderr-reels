"""
Halving_Caption.py
==================
Caption, hashtags and alt text for "Every Halving Pays Less", plus the gate
that decides whether they are allowed to reach disk.

Two rules make this more than a text file:

  1. Every number in CAPTION must appear in figures.json, the file the pipeline
     wrote from asserted computation. A number retyped from memory fails.
  2. The gate runs BEFORE anything is written. A validator that writes first and
     complains second still ships the broken caption.

RUN
    python Halving_Caption.py
    -> caption.txt, hashtags.txt, alt_text.txt, metadata.json (this folder)
       exit 0 only if every check passes
"""
import json, os, re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(BASE_DIR, "figures.json")
OUT_DIR = BASE_DIR                      # the caption ships next to its reel
HANDLE = "@quant.traderr"
KEYWORD = "HALVING"                    # the comment-keyword CTA

CAPTION = """
Most crypto traders treat the halving like a launch button: supply gets cut, price goes up, repeat forever. Quants measure each launch, and every one has been weaker than the last.

Each chalice is one cycle starting on halving day. Height is days since the halving, width is how far the price had multiplied.

The 2016 halving caught Bitcoin at $651. It topped at $19,497 in Dec 2017, +2,895%. The 2020 halving started at $8,602 and peaked at $73,084 in Mar 2024, +750%. The 2024 halving started at $63,844 and so far topped at $124,753 in Oct 2025, +95%. Today it sits at $83,554, +31% from halving day. Quants call it diminishing returns: the bigger the market, the more money it takes to move it.

The catch: 3 cycles is a tiny sample, and peaks are only obvious in hindsight. The 2024 cycle is still open, and the very first halving is missing because the price data starts later.

Comment "HALVING" and I'll send you the code.

Follow @quant.traderr for the math behind the markets.
"""

HASHTAGS = ('#bitcoin #btc #crypto #halving #quant #trading #investing #cryptocurrency #quantfinance #python #finance')

ALT = (
    "Slide 1: Three glowing 3D chalices on a black background, one per Bitcoin halving cycle, labelled 2016, 2020 and 2024. Height is days since the halving and width is how far the price had multiplied, coloured from blue to orange. A white ring marks each cycle's peak. The 2016 chalice is the widest, the 2024 one is a thin stem. Text reads +2,895% to +750% to +95%, peak gain after the 2016, 2020 and 2024 halvings."
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
    for key in ('peak_1', 'peak_2', 'peak_3', 'year_1', 'year_2', 'year_3', 'now_3'):
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
