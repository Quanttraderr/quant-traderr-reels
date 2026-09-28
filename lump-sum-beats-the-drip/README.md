# Lump Sum Beats the Drip

Every start day of SPY since 1993: invest a sum all at once, or drip it in as
12 equal monthly buys.

![hero](hero.png)

## What it measures

A month is 21 trading days, a year 252, prices are adjusted closes. For each
start day `s`:

```
lump(s) = P(s + 252) / P(s)
dca(s)  = mean over j = 0..11 of  P(s + 252) / P(s + 21 j)
```

Both are compared on day 252. Money still waiting to be dripped earns nothing
in the headline number. A second figure pays that waiting cash the 13-week
T-bill rate (`^IRX`, accrued daily), so the headline cannot hide behind a zero
cash rate.

In 3D: the trading day runs along the floor, the start date into depth, value
goes up. The glass block is the lump sum, fully invested from day one. Inside
it stands the staircase of the drip, one step every 21 days. The empty glass
above the stairs is money sitting out of the market. Orange stairs: the lump
sum finished ahead. Green: the drip did.

## What came out

| Figure | Value |
|---|---|
| Start days | 8,221 |
| Lump sum finished ahead | 79% |
| Drip finished ahead | 21% |
| Lump sum ahead with T-bill interest on the waiting cash | 75% |
| Median lead of the lump sum | +6.3% of the sum |
| Median lead when the drip won | 5.3% |
| Best start for the drip | Aug 2008, ahead by 34% |

## What this is not

**Not advice for any single start date.** The drip wins exactly in the crash
years, and by a lot.

The start dates overlap heavily, so these are not 8,221 independent trials,
this is about 33 years of one index that went up. No taxes, no costs, no
behaviour: if a lump sum would make you panic sell, the drip is the better
plan. The step height is the invested share (j/12) and is the same for every
start date; the real data sits in the colours. Only every n-th start date is
drawn; every figure uses all of them.

## Run it

```bash
cd ..
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python numpy pandas matplotlib yfinance imageio-ffmpeg scipy pillow
cd lump-sum-beats-the-drip

../.venv/bin/python LumpSum_Reel_Pipeline.py --smoke
../.venv/bin/python LumpSum_Reel_Pipeline.py
../.venv/bin/python LumpSum_Static_Pipeline.py
../.venv/bin/python LumpSum_Caption.py
```

The first run fetches from Yahoo Finance and caches to `_cache/`. Your figures
will differ from the table above as the data moves on; the asserts are bands
rather than snapshots, so the build still passes. `figures.json` records what
the posted version was built on.
