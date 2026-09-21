# quant.traderr reels

The code behind the 3D market visualisations on
[@quant.traderr](https://instagram.com/quant.traderr).

Every reel is one self-contained folder. Nothing is shared between them on
purpose: a reel you re-cut in six months should not break because a helper
changed for a different topic.

## The rule that makes these trustworthy

**Every number that reaches a frame is computed in the script and asserted
before a single frame is rendered.**

A `validate()` stage runs after the maths and before the render. It checks each
figure against a plausible band, writes them all to `figures.json` as strings,
and returns them. The render draws only strings from that file and never
re-derives a value. The caption module then gates itself against the same file:
a number typed from memory that is not in `figures.json` makes it exit 1
**before** anything is written.

So a data refresh that moves a figure out of band fails the build instead of
quietly shipping a video whose on-screen number contradicts its own caption.

If a fetch fails, the script raises. It never falls back to random data, because
a synthetic tape under a real-sounding caption is a lie with a chart on it.

## Reels

| Folder | Post | The number |
|---|---|---|
| [`the-ten-days`](the-ten-days) | The Ten Days | 313% becomes 118% without ten sessions |
| [`phase-portrait`](phase-portrait) | The Market's Phase Portrait | 87% of days sit below the previous high |
| [`shape-of-correlation`](shape-of-correlation) | The Shape of Correlation | average pairwise correlation 0.18 |
| [`when-diversification-dies`](when-diversification-dies) | When Diversification Dies | 0.37 correlation in a calm tape, 0.56 in a drawdown |
| [`the-market-pulls-back`](the-market-pulls-back) | The Market Pulls Back | 41 of 43 cells drift back toward the middle |
| [`the-curve-that-inverted`](the-curve-that-inverted) | The Curve That Inverted | inverted for 503 sessions straight |
| [`the-volatility-landscape`](the-volatility-landscape) | The Volatility Landscape | 72% on the 20 session window in Dec 2008 |
| [`the-bell-curve-lie`](the-bell-curve-lie) | The Bell Curve Lie | 63 days outside a shell a bell curve puts 0.1 days beyond |
| [`the-markets-memory`](the-markets-memory) | The Market's Memory | a big move follows a big down day 29% of the time, 13% on an average day |
| [`the-seasonality-mirage`](the-seasonality-mirage) | The Seasonality Mirage | September averages -0.5%, give or take 0.9 |

Each folder has its own README with the method, the figures the post was built
on, and a section on what the picture does **not** show. That last part is the
point: writing down what the maths does not support is what stops a good looking
render from making a claim it cannot carry.

## Running one

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

cd the-ten-days
../.venv/bin/python TenDays_Reel_Pipeline.py --smoke   # 3 frames, look at them
../.venv/bin/python TenDays_Reel_Pipeline.py           # full render -> topic.mp4
../.venv/bin/python TenDays_Static_Pipeline.py         # hero still
../.venv/bin/python TenDays_Caption.py                 # caption, gated
```

Data comes from Yahoo Finance through `yfinance` on the first run and is cached
in `_cache/` afterwards. The cache is not committed here, so your first run
fetches fresh data. That means your numbers may differ from the ones in the
post: the asserts are bands, not snapshots, so the build will still pass, and
`figures.json` in each folder records what the posted version was built on.

Output is 1080x1920 at 30 FPS, 11.7 seconds. That is native: Instagram
delivers reels at 1080x1920 at most, and handing it anything larger means
it downscales, which is exactly what smears hairlines and small type.

## Publishing

Two tools sit next to the reels because a reel that stays on disk is not a
post:

- `musikbett.py` builds a music bed out of sine waves, noise and envelopes and
  muxes it under the video without re-encoding the picture. Instagram's own
  music library only exists inside the app, so anything published through the
  API carries the audio that is in the file. Generating the bed keeps that
  free of anyone else's rights.
- `insta_post.py` publishes a finished reel through Metas Instagram API with
  Instagram Login: create a container from a public video URL, wait for
  processing, then publish. It needs the `instagram_business_content_publish`
  permission, and `--pruefen` says whether the token actually has it.

## Credit

The house style, the assertion gate and the per-reel structure follow a playbook
written by Vansh. The topics, the maths and the code in this repository are my
own work on top of it.

## Licence

MIT, see [LICENSE](LICENSE). Take it, change it, ship your own.
