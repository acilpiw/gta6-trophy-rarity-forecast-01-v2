// gta6-trophy-rarity-forecast-01 — entry v2
//
// A runtime-adaptive forecaster for Exophase trophy earn rates.
//
// The page this contest forecasts is not a smooth series. Exophase only
// republishes a game's counts at its own refresh moments, so the recorded
// history is a staircase: long flat stretches where every read repeats the
// previous page, then a step where the tracked-player count jumps and most
// rates move at once. New players dilute almost every rate at a step, while
// ordinary play pushes the easy and story trophies back up between steps.
// A forecaster that ignores the staircase either trusts a stale flat page
// (and gets destroyed by the next refresh) or extrapolates one jump as if
// it were a trend (and gets destroyed by dilution ending).
//
// This program therefore works on the refresh states of a case — the
// distinct pages inside its history — and does everything in logit space,
// which is the scale the pinball loss is charged on:
//
//   1. compress the history into distinct refresh states;
//   2. forecast the tracked-player count at the target with a saturating
//      growth fit blended with a damped log-linear trend;
//   3. per trophy, regress logit(rate) on log(players) across the states
//      (the net dilution-plus-progression slope), shrink the slope toward
//      the trophy's tier group, and extrapolate to the forecast player
//      count — candidate A, anchored on the last observed logit;
//   4. candidate B: a damped local velocity read from the most recent
//      refresh-to-refresh logit moves, faded toward zero over the horizon;
//   5. self-calibration: a rolling-origin backtest inside the case's own
//      history re-runs the same pipeline from earlier checkpoints and
//      scores it against the states that actually followed, and the
//      residuals choose the ensemble weight, a global bias shift, and the
//      spread scale — nothing is tuned on the practice answers;
//   6. quantiles are logistic(median + z * sigma) with sigma built from
//      the backtest residual dispersion (global, per-tier and per-trophy
//      parts), grown for the longer real horizon, plus a refresh-step
//      risk term, because the answer itself is one stepped page value.
//
// Robustness rules the whole file: every case is wrapped so that any
// internal failure falls back to a wide house-style random-walk forecast
// around the last read, a flat history falls back to a dilution-aware
// wide band, and the output is always schema-valid. An invalid case ranks
// below every valid entry, so never crashing matters more than any single
// modelling choice. The program is deterministic and uses no packages.

import { readdirSync, readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";

// ---------------------------------------------------------------- math

function clamp(x, lo, hi) {
  if (!Number.isFinite(x)) return lo;
  return x < lo ? lo : x > hi ? hi : x;
}

function logitOf(ratePercent) {
  const p = clamp(ratePercent, 0.01, 99.99) / 100;
  return Math.log(p / (1 - p));
}

function logisticOf(z) {
  if (z >= 35) return 100;
  if (z <= -35) return 0;
  return 100 / (1 + Math.exp(-z));
}

// Inverse standard normal CDF (Acklam's rational approximation, refined
// with one Halley step). Accurate far beyond what five quantiles need.
function invNorm(p) {
  const a = [-3.969683028665376e+01, 2.209460984245205e+02,
    -2.759285104469687e+02, 1.383577518672690e+02,
    -3.066479806614716e+01, 2.506628277459239e+00];
  const b = [-5.447609879822406e+01, 1.615858368580409e+02,
    -1.556989798598866e+02, 6.680131188771972e+01,
    -1.328068155288572e+01];
  const c = [-7.784894002430293e-03, -3.223964580411365e-01,
    -2.400758277161838e+00, -2.549732539343734e+00,
    4.374664141464968e+00, 2.938163982698783e+00];
  const d = [7.784695709041462e-03, 3.224671290700398e-01,
    2.445134137142996e+00, 3.754408661907416e+00];
  const plow = 0.02425, phigh = 1 - plow;
  let q, r;
  if (p < plow) {
    q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > phigh) {
    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  q = p - 0.5;
  r = q * q;
  return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

function meanOf(xs) {
  if (xs.length === 0) return 0;
  let s = 0;
  for (const x of xs) s += x;
  return s / xs.length;
}

function medianOf(xs) {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((u, v) => u - v);
  const m = s.length >> 1;
  return s.length % 2 === 1 ? s[m] : 0.5 * (s[m - 1] + s[m]);
}

function rmsOf(xs) {
  if (xs.length === 0) return 0;
  let s = 0;
  for (const x of xs) s += x * x;
  return Math.sqrt(s / xs.length);
}

// Ordinary least squares y ~ x. Returns null when the fit is meaningless.
function linReg(xs, ys) {
  const n = xs.length;
  if (n < 2) return null;
  const mx = meanOf(xs), my = meanOf(ys);
  let sxx = 0, sxy = 0, syy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx, dy = ys[i] - my;
    sxx += dx * dx; sxy += dx * dy; syy += dy * dy;
  }
  if (sxx <= 1e-12) {
    return { slope: 0, intercept: my, r2: 0, resid: 0, n };
  }
  const slope = sxy / sxx;
  const intercept = my - slope * mx;
  let ssRes = 0;
  for (let i = 0; i < n; i++) {
    const e = ys[i] - (intercept + slope * xs[i]);
    ssRes += e * e;
  }
  const r2 = syy > 1e-12 ? clamp(1 - ssRes / syy, 0, 1) : 0;
  const resid = Math.sqrt(ssRes / Math.max(1, n - 2));
  return { slope, intercept, r2, resid, n };
}

function parseTime(s) {
  const t = Date.parse(s);
  return Number.isFinite(t) ? t : NaN;
}

// ------------------------------------------------------- case handling

// Compress raw reads into distinct refresh states. Consecutive reads with
// the same player count and the same rates are the same Exophase page seen
// again, not new information, so they collapse into one state. A state
// keeps two timestamps: `t`, the latest read of that page (used for
// ordering and gaps), and `since`, the first read of it — how long the
// page has already been frozen matters, because the players an unseen
// refresh will reveal have been accumulating since the page froze, not
// since it was last re-read.
function refreshStates(history) {
  const states = [];
  let lastKey = null;
  for (const read of history || []) {
    if (!read || typeof read !== "object") continue;
    const t = parseTime(read.at || read.slot);
    if (!Number.isFinite(t)) continue;
    const players = Number(read.players_tracked);
    if (!Number.isFinite(players) || players <= 0) continue;
    const rates = Array.isArray(read.rates) ? read.rates : [];
    const key = players + "|" + rates.join(",");
    const earned = Number(read.trophies_earned);
    if (key === lastKey && states.length > 0) {
      states[states.length - 1].t = t; // same page, fresher timestamp
      states[states.length - 1].earned =
        Number.isFinite(earned) ? earned : states[states.length - 1].earned;
    } else {
      states.push({
        t,
        since: t,
        players,
        earned: Number.isFinite(earned) ? earned : NaN,
        rates,
      });
      lastKey = key;
    }
  }
  states.sort((u, v) => u.t - v.t);
  return states;
}

// Last non-null rate of trophy j at or before state index `upto`, together
// with the state it came from (its player count anchors extrapolations).
function lastObservation(states, j, upto) {
  for (let i = Math.min(upto, states.length - 1); i >= 0; i--) {
    const r = states[i].rates[j];
    if (r !== null && r !== undefined && Number.isFinite(Number(r))) {
      return { rate: Number(r), stateIdx: i, players: states[i].players };
    }
  }
  return null;
}

// --------------------------------------------------- player forecasting

// Forecast the tracked-player count at targetMs. Two views are blended in
// log space: a saturating logistic-style fit (grid over the ceiling,
// scored by fit quality) and a damped log-linear trend whose growth rate
// decays with a long time constant, which stays sane when the early phase
// is explosive. The result never drops below the last observed count.
function forecastPlayers(states, targetMs) {
  const n = states.length;
  const last = states[n - 1];
  if (n < 2) return last.players;
  const t0 = states[0].t;
  const xs = states.map((s) => (s.t - t0) / 3600000);
  const horizonH = Math.max(0, (targetMs - last.t) / 3600000);
  const targetX = (targetMs - t0) / 3600000;

  const trend = linReg(xs, states.map((s) => Math.log(Math.max(1, s.players))));
  const g = trend ? Math.max(0, trend.slope) : 0;
  const tau = 96; // hours; growth halves roughly every ~2.8 days
  const effH = tau * (1 - Math.exp(-horizonH / tau));
  const pDamped = last.players * Math.exp(Math.min(g * effH, Math.log(45)));

  const maxP = Math.max(...states.map((s) => s.players));
  let best = null;
  const mults = [1.2, 1.4, 1.7, 2.1, 2.6, 3.4, 4.5, 6, 8, 11, 15, 21, 30, 45, 70];
  for (const mult of mults) {
    const pinf = maxP * mult;
    if (!(pinf > maxP * 1.05)) continue;
    const ys = states.map((s) =>
      Math.log(clamp(s.players / (pinf - s.players), 1e-6, 1e6)));
    const fit = linReg(xs, ys);
    if (!fit) continue;
    const yT = fit.intercept + fit.slope * targetX;
    const pred = pinf / (1 + Math.exp(-clamp(yT, -30, 30)));
    if (!best || fit.r2 > best.r2) best = { pred, r2: fit.r2 };
  }

  let blended;
  if (best && Number.isFinite(best.pred)) {
    const wLog = best.r2 >= 0.6 ? 0.65 : 0.35;
    blended = Math.exp(wLog * Math.log(Math.max(last.players, best.pred)) +
      (1 - wLog) * Math.log(pDamped));
  } else {
    blended = pDamped;
  }
  return clamp(blended, last.players, last.players * 60);
}

// -------------------------------------------------- per-trophy dynamics

// Fit logit(rate) against log(players) for one trophy over states 0..upto.
function trophySlopeFit(states, j, upto) {
  const xs = [], ys = [];
  const top = Math.min(upto, states.length - 1);
  for (let i = 0; i <= top; i++) {
    const r = states[i].rates[j];
    if (r === null || r === undefined || !Number.isFinite(Number(r))) continue;
    xs.push(Math.log(Math.max(1, states[i].players)));
    ys.push(logitOf(Number(r)));
  }
  if (xs.length < 2) return null;
  return linReg(xs, ys);
}

// Recent refresh-to-refresh logit velocity (per hour) for one trophy,
// weighted toward the freshest moves and read over at most the last few
// steps, so one ancient jump cannot dominate the local trend.
function recentVelocity(states, j, upto) {
  const top = Math.min(upto, states.length - 1);
  const moves = [];
  for (let i = 1; i <= top; i++) {
    const r0 = states[i - 1].rates[j], r1 = states[i].rates[j];
    if (r0 === null || r0 === undefined || r1 === null || r1 === undefined) continue;
    if (!Number.isFinite(Number(r0)) || !Number.isFinite(Number(r1))) continue;
    const dtH = (states[i].t - states[i - 1].t) / 3600000;
    if (!(dtH > 0.01)) continue;
    moves.push({ v: (logitOf(Number(r1)) - logitOf(Number(r0))) / dtH, i });
  }
  if (moves.length === 0) return null;
  const tail = moves.slice(-4);
  const weights = [0.35, 0.5, 0.7, 1]; // oldest..newest inside the tail
  let sw = 0, sv = 0;
  for (let k = 0; k < tail.length; k++) {
    const w = weights[weights.length - tail.length + k];
    sw += w; sv += w * tail[k].v;
  }
  return { vel: sv / sw, pairs: tail.length };
}

// The uncalibrated core: given states truncated at `upto`, produce the two
// median candidates per trophy for targetMs. Calibration (weights, bias,
// spread) is layered on top by the caller, which is what lets the internal
// backtest reuse exactly this code path.
function coreForecast(states, trophies, upto, targetMs) {
  const top = Math.min(upto, states.length - 1);
  const horizonH = Math.max(0, (targetMs - states[top].t) / 3600000);
  const pFore = forecastPlayers(states.slice(0, top + 1), targetMs);
  const logPFore = Math.log(Math.max(1, pFore));

  // Tier-level pooling targets: mean slope and mean velocity per tier,
  // weighted by how many observations back each trophy's own fit.
  const fits = trophies.map((_, j) => trophySlopeFit(states, j, top));
  const vels = trophies.map((_, j) => recentVelocity(states, j, top));
  const tierSlope = {}, tierVel = {}, tierCount = {};
  trophies.forEach((tr, j) => {
    const tier = tr.tier || "bronze";
    tierCount[tier] = (tierCount[tier] || 0) + 1;
    if (fits[j]) {
      const w = fits[j].n;
      const cur = tierSlope[tier] || { sw: 0, sv: 0 };
      cur.sw += w; cur.sv += w * fits[j].slope;
      tierSlope[tier] = cur;
    }
    if (vels[j]) {
      const w = vels[j].pairs;
      const cur = tierVel[tier] || { sw: 0, sv: 0 };
      cur.sw += w; cur.sv += w * vels[j].vel;
      tierVel[tier] = cur;
    }
  });
  let globalSlopeNum = 0, globalSlopeDen = 0;
  for (const tier of Object.keys(tierSlope)) {
    globalSlopeNum += tierSlope[tier].sv;
    globalSlopeDen += tierSlope[tier].sw;
  }
  const globalSlope = globalSlopeDen > 0 ? globalSlopeNum / globalSlopeDen : 0;

  // Mean trophies per tracked player, first vs last state: a cheap sanity
  // covariate. If overall progression per player is collapsing while the
  // fitted slopes say rates should rise, trust the slopes less.
  let mtpDrag = 1;
  const first = states[0], lastS = states[top];
  if (Number.isFinite(first.earned) && Number.isFinite(lastS.earned) &&
      first.players > 0 && lastS.players > 0 && top > 0) {
    const mtp0 = first.earned / first.players;
    const mtp1 = lastS.earned / lastS.players;
    if (mtp0 > 0 && mtp1 < mtp0 * 0.7 && globalSlope > 0) mtpDrag = 0.6;
  }

  const tauB = 30; // hours; local velocity fades with this time constant
  const effHB = tauB * (1 - Math.exp(-horizonH / tauB));

  const medA = [], medB = [];
  trophies.forEach((tr, j) => {
    const tier = tr.tier || "bronze";
    const obs = lastObservation(states, j, top);
    if (!obs) { medA.push(null); medB.push(null); return; }
    const anchor = logitOf(obs.rate);
    const dLogP = logPFore - Math.log(Math.max(1, obs.players));

    // Candidate A: shrunk slope extrapolation in log-player space.
    const tierMean = tierSlope[tier] && tierSlope[tier].sw > 0
      ? tierSlope[tier].sv / tierSlope[tier].sw : globalSlope;
    let slope = tierMean;
    if (fits[j]) {
      const quality = 0.5 + 0.5 * clamp(fits[j].r2, 0, 1);
      const w = (fits[j].n / (fits[j].n + 3)) * quality;
      slope = w * fits[j].slope + (1 - w) * tierMean;
    }
    slope = clamp(slope * mtpDrag, -2.5, 2.5);
    medA.push(anchor + slope * dLogP);

    // Candidate B: shrunk, damped local velocity from the anchor.
    const tierVelMean = tierVel[tier] && tierVel[tier].sw > 0
      ? tierVel[tier].sv / tierVel[tier].sw : 0;
    let vel = tierVelMean;
    if (vels[j]) {
      const w = vels[j].pairs / (vels[j].pairs + 2);
      vel = w * vels[j].vel + (1 - w) * tierVelMean;
    }
    vel = clamp(vel, -0.15, 0.15);
    medB.push(anchor + vel * effHB);
  });
  return { medA, medB, pFore };
}

// ------------------------------------------------------- self-calibration

// Rolling-origin backtest inside the case's own history. Earlier refresh
// states play the role of checkpoints; the pipeline re-forecasts the
// states that actually followed and the residuals tune the live forecast.
// Returns defaults whenever the history is too short to teach anything.
function calibrate(states, trophies) {
  const fallback = {
    wA: 0.7, bias: 0, rmsGlobal: null, rmsTier: {}, rmsOwn: [],
    ownCount: [], meanBtHorizonH: 0, stepJump: 0.55, coverMult: 1,
  };
  const n = states.length;
  if (n < 5) return fallback;

  // Typical size of one refresh step in logit space (pooled, robust).
  const jumps = [];
  for (let i = 1; i < n; i++) {
    for (let j = 0; j < trophies.length; j++) {
      const r0 = states[i - 1].rates[j], r1 = states[i].rates[j];
      if (r0 === null || r0 === undefined || r1 === null || r1 === undefined) continue;
      if (!Number.isFinite(Number(r0)) || !Number.isFinite(Number(r1))) continue;
      jumps.push(Math.abs(logitOf(Number(r1)) - logitOf(Number(r0))));
    }
  }
  const stepJump = jumps.length > 0 ? medianOf(jumps) : 0.55;

  // Checkpoint indices, spread through the later two thirds of history.
  const checkpoints = [];
  const step = Math.max(1, Math.floor(n / 6));
  for (let c = Math.max(2, Math.floor(n * 0.3)); c < n - 1; c += step) {
    checkpoints.push(c);
    if (checkpoints.length >= 6) break;
  }
  if (checkpoints.length === 0) return { ...fallback, stepJump };

  // First pass: gather (A, B, actual, horizon) tuples per checkpoint.
  const tuples = []; // {j, tier, a, b, y, horizonH}
  for (const c of checkpoints) {
    const futures = [...new Set([c + 1, n - 1])].filter((f) => f > c);
    for (const f of futures) {
      const core = coreForecast(states, trophies, c, states[f].t);
      const horizonH = Math.max(0.05, (states[f].t - states[c].t) / 3600000);
      for (let j = 0; j < trophies.length; j++) {
        const y = states[f].rates[j];
        if (y === null || y === undefined || !Number.isFinite(Number(y))) continue;
        if (core.medA[j] === null || core.medB[j] === null) continue;
        tuples.push({
          j, tier: trophies[j].tier || "bronze",
          a: core.medA[j], b: core.medB[j],
          y: logitOf(Number(y)), horizonH,
        });
      }
    }
  }
  if (tuples.length < 30) return { ...fallback, stepJump };

  // Ensemble weight: the blend whose medians miss least in absolute logit.
  const wGrid = [0, 0.25, 0.5, 0.65, 0.8, 1];
  let bestW = 0.7, bestErr = Infinity;
  for (const w of wGrid) {
    let err = 0;
    for (const t of tuples) err += Math.abs(t.y - (w * t.a + (1 - w) * t.b));
    err /= tuples.length;
    if (err < bestErr) { bestErr = err; bestW = w; }
  }

  const residuals = tuples.map((t) => ({
    ...t, res: t.y - (bestW * t.a + (1 - bestW) * t.b),
  }));
  const allRes = residuals.map((r) => r.res);
  const biasRaw = medianOf(allRes);
  // Shrink the bias: a median residual from a few dozen points is a hint,
  // not a fact, and an over-shifted median loses on every trophy at once.
  const bias = clamp(biasRaw * (allRes.length / (allRes.length + 25)), -0.6, 0.6);
  const centered = residuals.map((r) => r.res - bias);
  const rmsGlobal = rmsOf(centered);
  // The horizon the spread is later scaled from: a long-horizon quantile
  // of the backtest horizons, not their mean. Half the backtest tuples are
  // one-step forecasts; scaling the pooled RMS from their mean horizon up
  // to the real horizon double-counts growth the long tuples already show.
  const btHorizons = residuals.map((r) => r.horizonH).sort((u, v) => u - v);
  const meanBtHorizonH = btHorizons[Math.floor(0.75 * (btHorizons.length - 1))] || 1;

  const rmsTier = {};
  const tiersSeen = [...new Set(residuals.map((r) => r.tier))];
  for (const tier of tiersSeen) {
    const part = centered.filter((_, i) => residuals[i].tier === tier);
    if (part.length >= 8) rmsTier[tier] = rmsOf(part);
  }
  const rmsOwn = [], ownCount = [];
  for (let j = 0; j < trophies.length; j++) {
    const part = centered.filter((_, i) => residuals[i].j === j);
    ownCount.push(part.length);
    rmsOwn.push(part.length >= 3 ? rmsOf(part) : null);
  }

  // Coverage check on the backtest itself: if the provisional spread
  // (residual RMS) would have covered far less or far more than the
  // nominal 90% band, nudge the live spread toward honesty. The nudge is
  // deliberately small: the RMS already prices the dispersion, and a
  // heavy-tailed residual set mechanically covers less than 90% inside
  // ±1.645 RMS without the true spread being wrong — chasing the last
  // few points of coverage with width loses more pinball than it saves.
  let inside = 0;
  for (const r of centered) {
    if (Math.abs(r) <= 1.645 * Math.max(0.15, rmsGlobal)) inside++;
  }
  const cover = centered.length > 0 ? inside / centered.length : 0.9;
  let coverMult = 1;
  if (cover < 0.65) coverMult = 1.2;
  else if (cover < 0.78) coverMult = 1.08;
  else if (cover > 0.98) coverMult = 0.9;

  return {
    wA: bestW, bias, rmsGlobal, rmsTier, rmsOwn, ownCount,
    meanBtHorizonH, stepJump, coverMult,
  };
}

// ------------------------------------------------------------- fallbacks

// House-style emergency spread around a rate: a random walk on the logit
// scale whose width grows with the square root of the horizon in days.
function houseRow(rate, horizonDays, levels) {
  const z = levels.map((lv) => invNorm(clamp(lv, 0.001, 0.999)));
  const mid = logitOf(rate === null || rate === undefined ? 1 : rate);
  const sig = 0.5 * Math.sqrt(Math.max(0.01, horizonDays));
  const q = z.map((zz) => clamp(logisticOf(mid + zz * sig), 0.02, 99.98));
  for (let i = 1; i < q.length; i++) if (q[i] < q[i - 1]) q[i] = q[i - 1];
  return q;
}

// Flat-history fallback. With fewer than two distinct pages there is no
// slope to fit, but there is a known danger: an unseen refresh can multiply
// the tracked players several-fold in one step, and fresh players have
// earned almost nothing, so rates get dragged down — hardest for the
// trophies everyone already has. The band is therefore wide and asymmetric:
// the median itself shifts down by the expected dilution over the whole
// window from the first read to the target, and the lower side of the
// spread is much wider than the upper side.
function flatForecast(caseObj, states, trophies, levels) {
  const horizonDays = Math.max(0.01, (caseObj.horizon_minutes || 0) / 1440);
  const targetMs = parseTime(caseObj.target);
  // The stale window runs from when the frozen page was first seen, not
  // from its last identical re-read.
  const firstT = states.length > 0 ? states[0].since : parseTime(caseObj.at);
  const windowH = Number.isFinite(targetMs) && Number.isFinite(firstT)
    ? clamp((targetMs - firstT) / 3600000, 0.25, 24 * 14) : 6;
  // Expected log player growth hiding inside the stale window plus the
  // horizon itself; capped because dilution cannot exceed the player base.
  const logGrowth = Math.min(Math.log(1 + windowH * 0.9), Math.log(8));
  const rows = trophies.map((tr, j) => {
    const obs = states.length > 0 ? lastObservation(states, j, states.length - 1) : null;
    const rate = obs ? obs.rate : null;
    if (rate === null) return { id: tr.id, q: houseRow(1, horizonDays, levels) };
    const mid0 = logitOf(rate);
    // Dilution exposure: a trophy at 100% is earned by definition by every
    // tracked player, so fresh players cannot dilute it; a rare trophy is
    // exposed almost in full. Earners also keep arriving, so only part of
    // the raw dilution lands on the median.
    const exposure = clamp(1 - rate / 100, 0, 1);
    const shift = -0.75 * logGrowth * exposure;
    const mid = mid0 + shift;
    const sigDown = clamp(0.9 + 0.55 * logGrowth, 1.1, 2.4);
    const sigUp = clamp(0.55 + 0.3 * Math.sqrt(horizonDays) + 0.2 * logGrowth, 0.7, 1.5);
    const q = levels.map((lv) => {
      const z = invNorm(clamp(lv, 0.001, 0.999));
      const sig = z < 0 ? sigDown : sigUp;
      return clamp(logisticOf(mid + z * sig), 0.02, 99.98);
    });
    for (let i = 1; i < q.length; i++) if (q[i] < q[i - 1]) q[i] = q[i - 1];
    return { id: tr.id, q: q.map((x) => Math.round(x * 10000) / 10000) };
  });
  return rows;
}

// ------------------------------------------------------------- main path

function forecastCase(caseObj) {
  const trophies = Array.isArray(caseObj.trophies) ? caseObj.trophies : [];
  const levels = (Array.isArray(caseObj.levels) && caseObj.levels.length === 5)
    ? caseObj.levels : [0.05, 0.25, 0.5, 0.75, 0.95];
  const states = refreshStates(caseObj.history);
  const horizonMin = Number(caseObj.horizon_minutes) ||
    Math.max(1, (parseTime(caseObj.target) - parseTime(caseObj.at)) / 60000);
  const horizonDays = Math.max(0.01, horizonMin / 1440);
  const horizonH = horizonDays * 24;
  const targetMs = parseTime(caseObj.target);

  if (trophies.length === 0) return { trophies: [] };
  if (states.length < 2 || !Number.isFinite(targetMs)) {
    return { trophies: flatForecast(caseObj, states, trophies, levels) };
  }

  const cal = calibrate(states, trophies);
  const core = coreForecast(states, trophies, states.length - 1, targetMs);

  // Spread assembly. Backtest residuals were earned over shorter horizons
  // than the real one, so their dispersion is grown by the square root of
  // the horizon ratio (capped), then combined across global / tier / own
  // levels with shrinkage, then charged a refresh-step risk term: the
  // answer is whichever stepped page is live at the target, and each
  // expected extra refresh adds its own jump uncertainty.
  const btH = Math.max(1, cal.meanBtHorizonH || 1);
  const hScale = clamp(Math.sqrt(Math.max(1, horizonH) / btH), 1, 1.6);
  const sigGlobal = cal.rmsGlobal !== null && cal.rmsGlobal > 0
    ? cal.rmsGlobal * hScale
    : 0.55 * Math.sqrt(horizonDays) + 0.25;

  const gaps = [];
  for (let i = 1; i < states.length; i++) {
    gaps.push((states[i].t - states[i - 1].t) / 3600000);
  }
  const medianGapH = gaps.length > 0 ? Math.max(0.25, medianOf(gaps)) : 6;
  const expRefresh = clamp(horizonH / medianGapH, 0, 12);
  const stepComp = 0.22 * cal.stepJump * Math.sqrt(Math.min(expRefresh, 6));
  const floorSig = Math.max(0.35, 0.3 * Math.sqrt(horizonDays));

  // Tier median last-rate logit, used only when a trophy was never seen.
  const tierFallback = {};
  trophies.forEach((tr, j) => {
    const obs = lastObservation(states, j, states.length - 1);
    if (obs) {
      const tier = tr.tier || "bronze";
      (tierFallback[tier] = tierFallback[tier] || []).push(logitOf(obs.rate));
    }
  });
  for (const tier of Object.keys(tierFallback)) {
    tierFallback[tier] = medianOf(tierFallback[tier]);
  }

  const rows = trophies.map((tr, j) => {
    const tier = tr.tier || "bronze";
    let med;
    if (core.medA[j] !== null && core.medB[j] !== null) {
      med = cal.wA * core.medA[j] + (1 - cal.wA) * core.medB[j] + cal.bias;
    } else if (tierFallback[tier] !== undefined) {
      med = tierFallback[tier] + cal.bias;
    } else {
      med = logitOf(1) + cal.bias;
    }

    const sigTier = cal.rmsTier[tier] ? cal.rmsTier[tier] * hScale : sigGlobal;
    let sigOwn = sigTier;
    if (cal.rmsOwn[j] !== null && cal.rmsOwn[j] !== undefined) {
      const cnt = cal.ownCount[j] || 0;
      const ownScaled = cal.rmsOwn[j] * hScale;
      // Shrink the trophy's own dispersion toward its tier's: four or so
      // backtest points cannot carry a spread on their own.
      sigOwn = Math.sqrt((cnt * ownScaled * ownScaled + 4 * sigTier * sigTier) / (cnt + 4));
    }
    let sigma = Math.sqrt(
      0.4 * sigGlobal * sigGlobal + 0.3 * sigTier * sigTier + 0.3 * sigOwn * sigOwn);
    sigma = Math.sqrt(sigma * sigma + stepComp * stepComp);
    sigma = clamp(sigma * cal.coverMult, floorSig, 2.8);

    // A trophy pinned at the ceiling can only fall; keep its spread from
    // pretending otherwise by trimming the upper side through the clamp.
    const q = levels.map((lv) => {
      const z = invNorm(clamp(lv, 0.001, 0.999));
      return clamp(logisticOf(med + z * sigma), 0.02, 99.98);
    });
    for (let i = 1; i < q.length; i++) if (q[i] < q[i - 1]) q[i] = q[i - 1];
    return { id: tr.id, q: q.map((x) => Math.round(x * 10000) / 10000) };
  });
  return { trophies: rows };
}

// Emergency path: last read, house-style spread. Used whenever anything
// above throws, so a case is never lost to a modelling bug.
function emergencyCase(caseObj) {
  const trophies = Array.isArray(caseObj.trophies) ? caseObj.trophies : [];
  const levels = (Array.isArray(caseObj.levels) && caseObj.levels.length === 5)
    ? caseObj.levels : [0.05, 0.25, 0.5, 0.75, 0.95];
  const horizonDays = Math.max(0.01, (Number(caseObj.horizon_minutes) || 90) / 1440);
  const states = refreshStates(caseObj.history);
  const rows = trophies.map((tr, j) => {
    const obs = states.length > 0 ? lastObservation(states, j, states.length - 1) : null;
    const q = houseRow(obs ? obs.rate : 1, horizonDays, levels)
      .map((x) => Math.round(x * 10000) / 10000);
    return { id: tr.id, q };
  });
  return { trophies: rows };
}

function main() {
  const inDir = process.env.MAKEFUN_INPUT_DIR || "/input";
  const outDir = process.env.MAKEFUN_OUTPUT_DIR || "/output";
  let names = [];
  try {
    names = readdirSync(inDir).filter((f) => f.endsWith(".in")).sort();
  } catch {
    return;
  }
  try { mkdirSync(outDir, { recursive: true }); } catch { /* may already exist */ }
  for (const name of names) {
    const id = name.slice(0, -3);
    let caseObj = null;
    try {
      caseObj = JSON.parse(readFileSync(join(inDir, name), "utf8"));
    } catch {
      continue; // nothing sane can be written for an unreadable case
    }
    let out;
    try {
      out = forecastCase(caseObj);
    } catch {
      try { out = emergencyCase(caseObj); } catch { continue; }
    }
    // Final schema guard: exactly one row per trophy, finite, ordered.
    try {
      const want = new Set((caseObj.trophies || []).map((t) => t.id));
      const seen = new Set();
      const rows = [];
      for (const row of out.trophies || []) {
        if (!want.has(row.id) || seen.has(row.id)) continue;
        const q = (row.q || []).map((x) => clamp(Number(x), 0, 100));
        if (q.length !== 5 || q.some((x) => !Number.isFinite(x))) continue;
        for (let i = 1; i < q.length; i++) if (q[i] < q[i - 1]) q[i] = q[i - 1];
        seen.add(row.id);
        rows.push({ id: row.id, q });
      }
      for (const tr of caseObj.trophies || []) {
        if (!seen.has(tr.id)) {
          const e = emergencyCase(caseObj).trophies.find((r) => r.id === tr.id);
          if (e) rows.push(e);
        }
      }
      rows.sort((u, v) => u.id - v.id);
      writeFileSync(join(outDir, id + ".out"), JSON.stringify({ trophies: rows }));
    } catch { /* a single unwritable case must not stop the others */ }
  }
}

main();
