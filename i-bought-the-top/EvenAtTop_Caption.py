"""
EvenAtTop_Caption.py
====================
Caption, hashtags and alt text for "I Bought the Top", plus the gate
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
KEYWORD = "BTC"                    # the comment-keyword CTA

CAPTION = """
Everyone's biggest fear with Bitcoin is buying the top. So we tested the worst case on purpose.

Start on 16 Dec 2017, the record close at $19,497. Then buy $10 of Bitcoin every single day, no matter what.

Within days the plan was underwater, at its worst point -56% against what had been paid in. It spent 16% of all days in the red, the last time in Apr 2020. Then it never looked back: by Sep 2026, $32,110 paid in over 8.8 years had become $165,383, that is 5.2x. Buying the next top on 8 Nov 2021 at $67,567 still turned $17,880 into $35,157, 2.0x. Quants call it dollar cost averaging: the crash days buy the most coins.

The catch: this is hindsight on the one coin that survived. Thousands of coins also had a record close and then went to zero, and a daily plan into one of those ends at zero. Past cycles promise nothing. No fees, no taxes.

Comment "BTC" and I'll send you the code.

Follow @quant.traderr for the math behind the markets.
"""

HASHTAGS = ("#bitcoin #btc #crypto #dca #investing #quant "
            "#cryptocurrency #bitcoininvestment #finance #python #datascience")

ALT = (
    "Slide 1: A 3D area graph on a black background, mirrored in a glossy "
    "floor. A coloured wall shows the value of buying 10 dollars of Bitcoin "
    "every day from the record close of 16 December 2017; it starts red, "
    "below a pale glass wall of the money paid in, then turns green, yellow "
    "and orange as it climbs far above it. Labels read VALUE 165,383 dollars "
    "and PAID IN 32,110 dollars. Text reads 32K IN, 165K OUT."
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
    for key in ("val", "dep", "mult", "worst", "top_px", "end"):
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
