"""
TenYearWait_Caption.py
======================
Caption, hashtags and alt text for "The Cost of Waiting", plus the gate
that decides whether they are allowed to reach disk.

Two rules make this more than a text file:

  1. Every number in CAPTION must appear in figures.json, the file the pipeline
     wrote from asserted computation. A number retyped from memory fails.
  2. The gate runs BEFORE anything is written. A validator that writes first and
     complains second still ships the broken caption.

RUN
    python Topic_Caption.py
    -> caption.txt, hashtags.txt, alt_text.txt, metadata.json (this folder)
       exit 0 only if every check passes
"""
import json, os, re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(BASE_DIR, "figures.json")
OUT_DIR = BASE_DIR                      # the caption ships next to its reel
HANDLE = "@quant.traderr"
KEYWORD = "WAIT"                    # the comment-keyword CTA

CAPTION = """
Most people wait to invest until they earn more. The math says every year you wait costs more than waiting ever saves.

Same plan, two start dates: $500 a month into SPY with dividends reinvested, bought on the first trading day of every month.

Start in 1993 and you paid in $202,000. By Sep 2026 it was worth $1,810,343. Start in 2003 instead and you paid in $142,000 for $794,841. Waiting 10 years saved you $60,000 of deposits and cost you $1,015,502 at the end, about $101,550 for every year of waiting. The early plan ends 2.3x bigger. That is compounding: the last decade does most of the work, and only the early starter is still around for it.

The catch: 1993 to 2026 was a strong stretch for US stocks, so the exact dollars are hindsight. No taxes, no fees, no inflation adjustment, and $500 was a bigger bite of a paycheck in 1993. The shape is the lesson, not the number.

Comment "WAIT" and I'll send you the code.

Follow @quant.traderr for the math behind the markets.
"""

HASHTAGS = ("#investing #compounding #stockmarket #sp500 #personalfinance "
            "#quant #financialfreedom #wealth #indexfunds #python #finance")

ALT = (
    "Slide 1: A 3D waterfall of thirteen area graphs on a black background, "
    "one per start year of a 500 dollar monthly plan into SPY, from 1993 at "
    "the back to 2017 at the front. Each graph has a grey base for the money "
    "paid in and a coloured cap for market gains. The 1993 graph towers to "
    "1.81 million dollars; the red 2003 graph ends at 795 thousand. Text reads "
    "WAITING 10 YEARS COST 1.02 million dollars."
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
    for key in ("early_val", "late_val", "cost", "early_dep", "late_dep", "end"):
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
