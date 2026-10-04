"""
GoldCoins_Caption.py
====================
Caption, hashtags and alt text for "Gold Beat the S&P", plus the gate
that decides whether they are allowed to reach disk.

Two rules make this more than a text file:

  1. Every number in CAPTION must appear in figures.json, the file the pipeline
     wrote from asserted computation. A number retyped from memory fails.
  2. The gate runs BEFORE anything is written. A validator that writes first and
     complains second still ships the broken caption.

RUN
    python GoldCoins_Caption.py
    -> caption.txt, hashtags.txt, alt_text.txt, metadata.json (this folder)
       exit 0 only if every check passes
"""
import json, os, re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(BASE_DIR, "figures.json")
OUT_DIR = BASE_DIR                      # the caption ships next to its reel
HANDLE = "@quant.traderr"
KEYWORD = "GOLD"                    # the comment-keyword CTA

CAPTION = """
Most investors call gold a dead rock and stocks the only way to build wealth. The tape since Aug 2000 says otherwise, even with every SPY dividend reinvested.

A dollar put into gold on day one is worth $15.29 today. The same dollar in SPY, dividends included, is worth $8.08. That is 15.3x against 8.1x. And look at the colours: most of the gold coins were stacked in the 2020s.

The catch: this is a start date story. Aug 2000 was the top of the dot com bubble and the tail end of a two decade slump in gold. Start in Jan 2010 instead and SPY wins easily, 9.0x against 3.7x. Gold also had a -44% drawdown from Aug 2011 to Dec 2015 and only got back to its old high in Jul 2020, while paying nothing. Quants call this start date bias, and it hides behind almost every chart that says one asset beat another.

Comment "GOLD" and I'll send you the code.

Follow @quant.traderr for the math behind the markets.
"""

HASHTAGS = ('#gold #investing #stockmarket #sp500 #quant #finance #trading #quantfinance #wealth #python #preciousmetals')

ALT = (
    'Slide 1: Two stacks of glowing coins on a black background, one coin per 50 cents of value. The left stack is a dollar put into gold in Aug 2000, the right stack the same dollar in SPY with dividends reinvested. Both grow month by month to Sep 2026; each coin is coloured by the year it was first reached, blue for the 2000s up to red for the 2020s. The gold stack ends almost twice as tall, and its upper half is red. Labels read GOLD $15.29 and SPY $8.08. Text reads gold 15.3x vs SPY 8.1x, but start in 2010 and SPY wins, 9.0x vs 3.7x.'
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
    for key in ('gold_final', 'spy_final', 'gold_x', 'spy_x', 'gold_x10', 'spy_x10', 'start', 'split'):
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
