"""
DipWait_Caption.py
==================
Caption, hashtags and alt text for one reel, plus the gate that decides whether
they are allowed to reach disk.

Two rules make this more than a text file:

  1. Every number in CAPTION must appear in figures.json, the file the pipeline
     wrote from asserted computation. A number retyped from memory fails.
  2. The gate runs BEFORE anything is written. A validator that writes first and
     complains second still ships the broken caption.

RUN
    python DipWait_Caption.py
    -> caption.txt, hashtags.txt, alt_text.txt, metadata.json (this folder)
       exit 0 only if every check passes
"""
import json, os, re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(BASE_DIR, "figures.json")
OUT_DIR = BASE_DIR                      # the caption ships next to its reel
HANDLE = "@quant.traderr"
KEYWORD = "DIP"                  # the comment-keyword CTA

CAPTION = """
Retail says "I'll buy when it drops 10%." A quant checks how often that dip ever showed up.

Every one-year window of SPY since 1993, started on every single trading day: 8,220 windows, each one a line. The red tube sits at -10% below the entry.

In 69% of those windows the dip never came. Not once in 252 trading days, while SPY closed the year at a median +17% and the waiter sat in cash. 74% of those no-dip years ended above +11%, so even a later -10% drop would still have been above the price they passed on.

When the dip did come (31% of windows), it took a median 72 trading days. Averaged over every window, waiting made +3% a year, buying on day one made +12%. The waiter came out ahead in only 34% of windows.

The catch: the windows overlap heavily, so this is one index over one mostly rising stretch of history, not 8,220 independent trials. Cash earns zero here, T-bills would narrow the gap, and in 2008 waiting was the right call.

Comment "DIP" and I'll send the code that builds it.

Follow @quant.traderr for daily breakdowns of the math behind the markets.
"""

HASHTAGS = ('#buythedip #investing #timeinthemarket #trading #quant #spy #sp500 #quantfinance #markets #python #datascience')

ALT = (
    'Slide 1: A black 3D space with a translucent red tube standing upright in the middle, marking minus 10 percent. Hundreds of thin lines rise from a white ring at its base, one for each year of holding the S&P 500 from a different start date since 1993. Distance from the centre is the return. Blue lines bend inward and pierce the tube, each with a white dot where the 10 percent dip was bought. Most lines stay outside the tube all year and flare outward, coloured green to red by how much they gained.'
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
    for key in ('windows', 'never', 'came', 'missed', 'out_of_reach', 'undo', 'buy_now', 'waiter', 'wait_wins', 'days', 'dip', 'first_year', 'window', 'crash'):
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
                   "title": "The Dip That Never Came",
                   "caption_chars": len(text),
                   "n_hashtags": len(HASHTAGS.split()),
                   "figures": figs}, f, indent=2)
    print(f"caption ok  ({len(text)} chars, {len(HASHTAGS.split())} hashtags)")


if __name__ == "__main__":
    main()
