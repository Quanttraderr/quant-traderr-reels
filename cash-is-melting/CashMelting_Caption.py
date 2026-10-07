"""
CashMelting_Caption.py
======================
Caption, hashtags and alt text for "Cash Is Melting", plus the gate that
decides whether they are allowed to reach disk.

Two rules make this more than a text file:

  1. Every number in CAPTION must appear in figures.json, the file the pipeline
     wrote from asserted computation. A number retyped from memory fails.
  2. The gate runs BEFORE anything is written. A validator that writes first and
     complains second still ships the broken caption.

RUN
    python CashMelting_Caption.py
    -> caption.txt, hashtags.txt, alt_text.txt, metadata.json (this folder)
       exit 0 only if every check passes
"""
import json, os, re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(BASE_DIR, "figures.json")
OUT_DIR = BASE_DIR                      # the caption ships next to its reel
HANDLE = "@quant.traderr"
KEYWORD = "MELT"                        # the comment-keyword CTA

CAPTION = """
Retail thinks cash is safe. Quants count what it buys, and watch it melt.

This is purchasing power decay: $100 of US cash from Jan 1913, run through every monthly CPI print up to Aug 2026. The sand on top is what it still buys.

Result: $2.93. 97% gone in 113 years, at an average inflation of only 3.2% a year, because the loss compounds. By Aug 1971, when the gold window closed, it was already down to $24.02, and it has lost 88% of its value since then. 50% since Jan 2000. 23% since Jan 2020 alone. Worst stretch: +23.7% in 12 months, Jun 1920. Only the Depression ran it backwards: by Mar 1933 the $100 bought $77.78 again.

The catch: CPI is an average basket that keeps changing, so this is an index ratio, not the price of one fixed shopping cart. Nobody keeps one banknote under the mattress for 113 years, and wages and interest rates rose along with prices. The point is not that savers lost 97%. It is that idle cash pays a quiet tax every single year.

Comment "MELT" and I'll send you the code.

Follow @quant.traderr for the math behind the markets.
"""

HASHTAGS = ("#inflation #cash #purchasingpower #quant #quantfinance #finance "
            "#investing #money #economics #cpi #python #dataviz")

ALT = (
    "Slide 1: A glass hourglass drawn as a thin cyan wire mesh on a black background, "
    "turning slowly. The sand stands for $100 of US cash from January 1913. The pale "
    "sand left in the top chamber is what that money still buys; the sand in the "
    "bottom chamber is purchasing power lost to inflation, layered and coloured by the "
    "decade it was lost in, blue for the 1910s at the bottom through green, yellow and "
    "orange to red for the 2020s on top. Dollar marks at $100, $50, $25 and $10 sit "
    "beside the top chamber. A counter runs from 1913 at $100 to the end readout: "
    "$100 becomes $2.93, minus 97% since 1913, 88% since 1971, 23% since 2020."
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
    for key in ("base", "final_value", "loss_total", "start", "end", "years",
                "value_1971", "loss_since_1971", "loss_since_2000",
                "loss_since_2020", "worst_yoy", "worst_date", "cagr"):
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
