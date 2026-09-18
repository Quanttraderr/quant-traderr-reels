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
| [`when-diversification-dies`](when-diversification-dies) | When Diversification Dies | 0.37 correlation in a calm tape, 0.56 in a drawdown |

## Running one

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

cd when-diversification-dies
../.venv/bin/python Tomography_Reel_Pipeline.py --smoke   # 3 frames, look at them
../.venv/bin/python Tomography_Reel_Pipeline.py           # full render -> topic.mp4
../.venv/bin/python Tomography_Static_Pipeline.py         # hero still
../.venv/bin/python Tomography_Caption.py                 # caption, gated
```

Data comes from Yahoo Finance through `yfinance` on the first run and is cached
in `_cache/` afterwards. The cache is not committed here, so your first run
fetches fresh data. That means your numbers may differ from the ones in the
post: the asserts are bands, not snapshots, so the build will still pass, and
`figures.json` in each folder records what the posted version was built on.

Output is 1440x2560 at 30 FPS, 11.7 seconds.

## Credit

The house style, the assertion gate and the per-reel structure follow a playbook
written by Vansh. The topics, the maths and the code in this repository are my
own work on top of it.

## Licence

MIT, see [LICENSE](LICENSE). Take it, change it, ship your own.
