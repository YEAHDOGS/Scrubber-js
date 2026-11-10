// Tests for the JavaScript wrapper and the built dist/scrubber.wasm.
// Run with: npm test   (after `npm run build` if you changed anything in wasm/)
//
// The proof that scrubbing defeats the location attack lives in wasm/tests,
// where the attack scorer runs against the Rust core. These tests cover what
// the wrapper adds: memory handling, segmenting, joins, options and reports.

import assert from "node:assert/strict";
import { test } from "node:test";
import { createScrubber } from "./index.js";

const SR = 8000;
const TAU = 2 * Math.PI;
/** A flyover-like tone: this pitch, falling by GLIDE_HZ over the clip. */
const TONE_START_HZ = 110;
const GLIDE_HZ = 20;
/** Faint: about 29 dB below the noise, inaudible. */
const FAINT = 0.05;
/** Loud enough to stand above the noise floor in a spectrogram. */
const AUDIBLE = 0.3;

const scrubber = await createScrubber();

/** Small seeded generator, so the tests are repeatable. */
function gaussianSource(seed) {
  let a = seed >>> 0;
  const uniform = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return () => Math.sqrt(-2 * Math.log(1 - uniform())) * Math.cos(TAU * uniform());
}

/** Phase of the gliding tone at every sample: an S-shaped fall, like a pass overhead. */
function glidePhase(length) {
  const phase = new Float64Array(length);
  let acc = 0;
  for (let i = 0; i < length; i++) {
    const x = (i / length - 0.5) * 8;
    acc += (TAU * (TONE_START_HZ - GLIDE_HZ / (1 + Math.exp(-x)))) / SR;
    phase[i] = acc;
  }
  return phase;
}

function noiseWithTone(seconds, amplitude, seed) {
  const length = seconds * SR;
  const normal = gaussianSource(seed);
  const phase = glidePhase(length);
  const samples = new Float32Array(length);
  for (let i = 0; i < length; i++) samples[i] = normal() + amplitude * Math.sin(phase[i]);
  return samples;
}

/**
 * A matched filter for the known tone: how much of it is in `samples`.
 * Reads about amplitude / 2 when the tone is there and about 1 / sqrt(length) when it is not.
 */
function toneContent(samples) {
  const phase = glidePhase(samples.length);
  let re = 0;
  let im = 0;
  for (let i = 0; i < samples.length; i++) {
    re += samples[i] * Math.cos(phase[i]);
    im += samples[i] * Math.sin(phase[i]);
  }
  return Math.hypot(re, im) / samples.length;
}

test("a faint gliding tone is gone after scrubbing, and the input is untouched", () => {
  const input = noiseWithTone(40, FAINT, 1);
  const copy = input.slice();
  const { channels, report } = scrubber.scrub([input], SR);
  assert.deepEqual(input, copy);
  assert.equal(channels[0].length, input.length);
  assert.ok(toneContent(input) > 0.02, `tone should be measurable before: ${toneContent(input)}`);
  assert.ok(toneContent(channels[0]) < 0.006, `tone left after scrubbing: ${toneContent(channels[0])}`);
  assert.ok(report.replacedFraction > 0.5);
  assert.ok(Number.isFinite(report.peak) && report.peak > 0);
  // Plain noise is rebuilt entirely, and must come back at the level it went in.
  const rms = (x) => Math.sqrt(x.reduce((s, v) => s + v * v, 0) / x.length);
  const changeDb = 20 * Math.log10(rms(channels[0]) / rms(input));
  assert.ok(Math.abs(changeDb) < 0.5, `level changed by ${changeDb.toFixed(2)} dB`);
});

test("check reports an audible falling tone and changes nothing", () => {
  const input = noiseWithTone(40, AUDIBLE, 2);
  const copy = input.slice();
  const report = scrubber.check([input], SR);
  assert.deepEqual(input, copy);
  const [strongest] = report.ridges;
  assert.ok(strongest, "a ridge should be found");
  assert.ok(strongest.fLast < strongest.fFirst, "the tone falls in pitch");
  assert.ok(strongest.fLow > TONE_START_HZ - GLIDE_HZ - 10 && strongest.fHigh < TONE_START_HZ + 10);
  assert.ok(!("peak" in report));
});

test("a long recording is scrubbed in segments with no tone left at the joins", () => {
  const input = noiseWithTone(100, FAINT, 3);
  const options = { segmentSeconds: 30, contextSeconds: 5 };
  const { channels } = scrubber.scrub([input], SR, options);
  assert.equal(channels[0].length, input.length);
  assert.ok(channels[0].every(Number.isFinite));
  assert.ok(toneContent(channels[0]) < 0.004, `tone left: ${toneContent(channels[0])}`);
  // The level must not jump or dip where the pieces meet.
  const rms = (from, to) => Math.sqrt(channels[0].subarray(from, to).reduce((s, v) => s + v * v, 0) / (to - from));
  const whole = rms(0, input.length);
  for (const join of [30, 60, 90]) {
    const around = rms((join - 0.2) * SR, (join + 0.2) * SR);
    assert.ok(Math.abs(20 * Math.log10(around / whole)) < 1, `level at ${join} s is off by more than 1 dB`);
  }
});

test("a tone that crosses segment joins is reported as one ridge", () => {
  const input = noiseWithTone(100, AUDIBLE, 4);
  const whole = scrubber.check([input], SR, { segmentSeconds: Infinity });
  const pieces = scrubber.check([input], SR, { segmentSeconds: 30, contextSeconds: 5 });
  const longest = (report) => Math.max(...report.ridges.map((r) => r.tEnd - r.tStart));
  assert.ok(longest(whole) > 60, `one piece: longest ridge ${longest(whole)} s`);
  assert.ok(longest(pieces) > 0.9 * longest(whole), `segments: longest ridge ${longest(pieces)} s`);
});

test("grid hum is detected, and scrubbing removes it", () => {
  const length = 90 * SR;
  const normal = gaussianSource(5);
  const humming = new Float32Array(length);
  let phase = 0;
  for (let i = 0; i < length; i++) {
    const t = i / SR;
    phase += (TAU * (60 + 0.02 * Math.sin((TAU * t) / 37))) / SR;
    humming[i] = 0.3 * normal() + 0.05 * Math.sin(2 * phase) + 0.03 * Math.sin(4 * phase);
  }
  const before = scrubber.detectHum([humming], SR);
  assert.equal(before.present, true);
  assert.equal(before.nominalHz, 60);
  assert.ok(Math.abs(before.gridHz - 60) < 0.01);
  assert.deepEqual(before.lines.map((l) => l.hz), [120, 240]);
  const { channels } = scrubber.scrub([humming], SR);
  assert.equal(scrubber.detectHum(channels, SR).present, false);
});

test("channels get different noise, and a seed makes the output repeatable", () => {
  const input = noiseWithTone(10, FAINT, 6);
  const first = scrubber.scrub([input, input], SR, { seed: 7 });
  const again = scrubber.scrub([input, input], SR, { seed: 7 });
  const unseeded = scrubber.scrub([input, input], SR);
  assert.notDeepEqual(first.channels[0], first.channels[1]);
  assert.deepEqual(first.channels, again.channels);
  assert.notDeepEqual(first.channels[0], unseeded.channels[0]);
});

test("progress runs from 0 to 1 without going backwards", () => {
  const input = noiseWithTone(70, FAINT, 8);
  const seen = [];
  scrubber.scrub([input, input], SR, { segmentSeconds: 30, contextSeconds: 5, onProgress: (f) => seen.push(f) });
  assert.ok(seen.length > 10);
  assert.ok(seen.every((f, i) => i === 0 || f >= seen[i - 1]));
  assert.ok(seen[0] >= 0 && Math.abs(seen.at(-1) - 1) < 1e-6);
});

test("empty and very short audio pass through without error", () => {
  assert.equal(scrubber.scrub([new Float32Array(0)], SR).channels[0].length, 0);
  const short = Float32Array.from({ length: 100 }, (_, i) => Math.sin(i));
  assert.equal(scrubber.scrub([short], SR).channels[0].length, 100);
  assert.equal(scrubber.detectHum([short], SR).present, false);
});
