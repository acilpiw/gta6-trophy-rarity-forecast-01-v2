# gta6-trophy-rarity-forecast-01-v2 — staircase-aware trophy forecaster

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/acilpiw/gta6-trophy-rarity-forecast-01-v2/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/acilpiw/gta6-trophy-rarity-forecast-01-v2/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Entry for the make.fun bounty **gta6-trophy-rarity-forecast-01**:
forecast the Exophase earn rate of every GTA VI PlayStation 5 trophy at
2026-11-26T00:00:00Z. Competition page:
https://make.fun/c/gta6-trophy-rarity-forecast-01

This is the second-generation model. The first generation fitted each
trophy on its own and trusted the practice cases more than they deserved.
The practice histories are flat — one stale page, then a single refresh —
while the final case carries the whole launch week: hundreds of reads,
many refreshes, the tracked-player count exploding from 638 upward, rates
dropping at the first refresh as new players dilute them, then story
trophies climbing back as those players progress. Version 2 is built for
that shape, and it calibrates itself on whatever history the case
actually contains instead of on constants chosen beforehand.

## How it works

Everything happens in logit space, the scale the score is charged on.

1. **Refresh states.** Repeated reads of the same page are collapsed, so
   the model sees the staircase Exophase actually published: distinct
   pages with their player counts and timestamps.
2. **Player forecast.** The tracked-player count at the target comes from
   a saturating growth fit (a grid over the ceiling, scored by fit
   quality) blended with a damped log-linear trend, so the explosive
   first days cannot be extrapolated forever.
3. **Candidate A — slope extrapolation.** Per trophy, logit(rate) is
   regressed on log(players) across the refresh states. That slope is the
   net of dilution and progression for the trophy. It is shrunk toward
   its tier group's mean (platinum / gold / silver / bronze), weighted by
   observation count and fit quality, then applied to the forecast player
   growth, anchored on the last observed rate.
4. **Candidate B — local velocity.** The freshest refresh-to-refresh
   logit moves give a per-hour velocity, shrunk the same way and damped
   toward zero over the horizon, because a launch-week surge does not
   run in a straight line for four more days.
5. **Self-calibration.** A rolling-origin backtest inside the case's own
   history truncates the states at several past checkpoints, re-runs the
   same pipeline, and scores the forecasts against the states that really
   followed. The residuals pick the A/B ensemble weight, a shrunk global
   bias shift, and the spread scale — including a coverage correction if
   the backtest band covered too little or too much.
6. **Spread.** Each trophy's sigma mixes global, per-tier and per-trophy
   residual dispersion, grows with the square root of the horizon ratio
   between the backtest and the real forecast, and adds a refresh-step
   risk term: the answer is one stepped page value, and every expected
   extra refresh is another jump that could land either way.
7. **Degenerate histories.** If a case has fewer than two distinct pages
   there is no slope to fit, so the forecast falls back to the last rates
   with a wide, dilution-aware band: the median shifts down by the player
   growth an unseen refresh can hide, most strongly for rare trophies,
   and the lower side of the band is far wider than the upper side.

Any internal failure falls back per case to a wide random-walk spread
around the last read. The output is always schema-valid: an invalid case
ranks below every valid entry, so the program is written to never crash,
never emit a non-monotone or out-of-range quantile, and finish far
inside the 8-second limit. It is deterministic and uses no packages.

## Run

    mkdir -p out
    MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts

Reads every `*.in` file in `$MAKEFUN_INPUT_DIR` (default `/input`) and
writes one `<id>.out` per case to `$MAKEFUN_OUTPUT_DIR` (default
`/output`), each holding five quantiles (0.05, 0.25, 0.5, 0.75, 0.95)
per trophy, in percent.

## License

MIT — see `LICENSE`.
