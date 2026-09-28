"""
Recovery_Caption.py
===================
Caption, hashtags and alt text for one reel, plus the gate that decides whether
they are allowed to reach disk.

Two rules make this more than a text file:

  1. Every number in CAPTION must appear in figures.json, the file the pipeline
     wrote from asserted computation. A number retyped from memory fails.
  2. The gate runs BEFORE anything is written. A validator that writes first and
     complains second still ships the broken caption.

RUN
    python Recovery_Caption.py
    -> caption.txt, hashtags.txt, alt_text.txt, metadata.json (this folder)
       exit 0 only if every check passes
"""
import json, os, re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(BASE_DIR, "figures.json")
OUT_DIR = BASE_DIR                      # the caption ships next to its reel
HANDLE = "@quant.traderr"
KEYWORD = "WAIT"                  # the comment-keyword CTA

CAPTION = """
Retail says the market always comes back, so just hold. A quant asks the only question that matters: how long did you have to wait?

Every fall of 5% or more in the S&P 500 since 1927, from old high to low to new high: 73 falls, each one a sphere. Volume is the wait, colour is the depth.

Most falls are quick. 79% were back at the old high within 12 months, 90% within 2 years, and the median wait was 2.8 months. Then there is Sep 1929. The index fell -86% and did not see that high again until Sep 1954: a 25 year wait. 1973 took 7.5 years, 2000 took 7.2.

The catch: this is the price index without dividends, so every wait is too long. SPY with dividends reinvested needed 6.6 years after 2000, not 7.2. Inflation is not in here either, and that makes the 1973 wait worse. It did come back, but "just hold" quietly assumes you have 25 years.

Comment "WAIT" and I'll send the code that builds it.

Follow @quant.traderr for daily breakdowns of the math behind the markets.
"""

HASHTAGS = ('#sp500 #buyandhold #drawdown #stockmarket #investing #quant #quantfinance #trading #markets #marketcrash #python #datascience')

ALT = (
    'Slide 1: A black 3D space with a tight pack of glowing wireframe spheres, one for every fall of 5 percent or more in the S&P 500 price index since 1927. Sphere volume is how long it took to get back to the old high, colour is how deep the fall went, from blue for shallow to red for the deepest. Dozens of small blue and cyan spheres sit together in the middle, three large orange spheres for 1973, 2000 and 2007 hang on the side, and one huge red sphere on top is 1929, labelled 25.0 years.'
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
    for key in ('episodes', 'min_depth', 'fast', 'slow', 'median_months', 'longest', 'longest_peak', 'longest_back', 'longest_depth', 'chk_longest', 'idx_same', 'first_year'):
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
                   "title": "It Always Comes Back",
                   "caption_chars": len(text),
                   "n_hashtags": len(HASHTAGS.split()),
                   "figures": figs}, f, indent=2)
    print(f"caption ok  ({len(text)} chars, {len(HASHTAGS.split())} hashtags)")


if __name__ == "__main__":
    main()
