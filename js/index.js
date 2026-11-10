/**
 * Scrubber-js: remove aircraft Doppler signatures and grid hum from audio, so a
 * recording can't be located from the planes flying over it.
 *
 * This file is the whole JavaScript side. The signal processing lives in
 * `wasm/` (Rust, compiled to `dist/scrubber.wasm`); here we load that module,
 * move samples in and out of its memory, and split long recordings into
 * segments so memory stays bounded.
 *
 * Everything runs synchronously on the calling thread once the module is
 * loaded. In a browser, call it from a Worker.
 */

/** Defaults match upstream flyover-scrub's `ScrubConfig`. */
export const DEFAULT_CONFIG = Object.freeze({
  /** Protected band, Hz. */
  band: Object.freeze([20, 2000]),
  /** About 4 Hz bins: long enough to separate slow tones from speech. */
  windowSeconds: 0.25,
  hopFraction: 0.25,
  /** Width of the median across frequency for the noise-floor estimate. */
  floorFrequencyHz: 120,
  /** Smoothing of the floor over time. */
  floorTimeSeconds: 3,
  /** Bins this far above the floor are kept. 8 leaks medium-loud lines. */
  foregroundDb: 12,
  /** Low on purpose: a false ridge in pure noise costs nothing. */
  ridgeDb: 4,
  /** Ridges at least this long are treated as signatures. */
  ridgeMinSeconds: 1.5,
  /** Aircraft glide slowly; speech pitch moves fast. */
  ridgeMaxSlopeHzPerSecond: 40,
  ridgePadHz: 8,
  /** Keep perfectly steady tones (mains hum, appliances). Off: hum can date a recording. */
  keepSteadyTones: false,
  steadyMaxDriftHz: 1.5,
});

/** Order the WebAssembly module reads the config in (`config_from` in wasm/src/lib.rs). */
const CONFIG_ORDER = [
  (c) => c.band[0],
  (c) => c.band[1],
  (c) => c.windowSeconds,
  (c) => c.hopFraction,
  (c) => c.floorFrequencyHz,
  (c) => c.floorTimeSeconds,
  (c) => c.foregroundDb,
  (c) => c.ridgeDb,
  (c) => c.ridgeMinSeconds,
  (c) => c.ridgeMaxSlopeHzPerSecond,
  (c) => c.ridgePadHz,
  (c) => (c.keepSteadyTones ? 1 : 0),
  (c) => c.steadyMaxDriftHz,
];

/** Layout of the report buffers (constants of the same names in wasm/src/lib.rs). */
const REPORT_HEAD = 3;
const RIDGE_VALUES = 7;
const HUM_VALUES = 13;
const MAX_RIDGES_PER_SEGMENT = 512;
const BYTES_F32 = 4;
const BYTES_F64 = 8;

/** Long recordings are scrubbed in pieces this long, so memory stays bounded. */
const DEFAULT_SEGMENT_SECONDS = 180;
/** Each piece sees this much audio either side, so tones crossing a join are still found. */
const DEFAULT_CONTEXT_SECONDS = 10;
/** Neighbouring pieces are blended over this long at each join. */
const JOIN_FADE_SECONDS = 0.25;
/** Ridges that meet at a join within this pitch are reported as one. */
const JOIN_MATCH_HZ = 12;
/** Grid hum is looked for in at most this much audio. */
const DEFAULT_HUM_SECONDS = 600;
const DEFAULT_HUM_MIN_DB = 8;
const DEFAULT_WASM_URL = new URL("../dist/scrubber.wasm", import.meta.url);

/**
 * @typedef {object} Ridge A long narrowband tone: aircraft, drone, traffic, HVAC or hum.
 * @property {number} channel
 * @property {number} tStart Seconds from the start of the audio.
 * @property {number} tEnd
 * @property {number} fLow Hz
 * @property {number} fHigh Hz
 * @property {number} fFirst Pitch at its first frame, Hz.
 * @property {number} fLast Pitch at its last frame, Hz. A flyover falls; hum stays put.
 * @property {number} peakDb Strongest point, dB above the local noise floor.
 */

/**
 * @typedef {object} ScrubReport
 * @property {number} replacedFraction Share of in-band bins replaced, averaged over channels.
 * @property {number} ridgeFraction Share of in-band bins that sat on a ridge.
 * @property {Ridge[]} ridges Strongest first.
 * @property {number} peak Largest absolute sample in the output (may exceed 1; scale if you need to).
 */

/**
 * @typedef {object} HumReport Mains hum carrying the electrical network frequency (ENF).
 * @property {boolean} present
 * @property {number|null} nominalHz 50 or 60: which kind of grid.
 * @property {number|null} gridHz Median grid frequency implied by the hum.
 * @property {number|null} wanderHz Its standard deviation over the clip.
 * @property {{hz: number, db: number}[]} lines Harmonics that stand out.
 */

/**
 * Fetches or reads the module bytes from wherever they are.
 * @param {string|URL|Response|ArrayBuffer|ArrayBufferView|WebAssembly.Module} source
 * @returns {Promise<WebAssembly.Module>}
 */
async function compile(source) {
  if (source instanceof WebAssembly.Module) return source;
  if (source instanceof ArrayBuffer || ArrayBuffer.isView(source)) return WebAssembly.compile(source);
  if (typeof Response !== "undefined" && source instanceof Response) {
    return WebAssembly.compile(await source.arrayBuffer());
  }
  const url = new URL(source, import.meta.url);
  if (url.protocol === "file:") {
    const { readFile } = await import("node:fs/promises");
    return WebAssembly.compile(await readFile(url));
  }
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Scrubber: could not load ${url} (${response.status})`);
  return WebAssembly.compile(await response.arrayBuffer());
}

/** Two unpredictable 32-bit numbers, or two derived from a caller's seed. */
function seedWords(seed, channel, segment) {
  if (seed === undefined || seed === null) {
    return crypto.getRandomValues(new Uint32Array(2));
  }
  // The same seed must not give two channels or two segments the same noise.
  return Uint32Array.of((channel << 16) ^ segment, seed >>> 0);
}

/** Joins ridges that were cut in two by a segment boundary. */
function mergeAcrossJoins(ridges, joins, tolerance) {
  const merged = [];
  for (const ridge of ridges) {
    const earlier = merged.find(
      (m) =>
        m.channel === ridge.channel &&
        joins.some((t) => Math.abs(m.tEnd - t) <= tolerance && Math.abs(ridge.tStart - t) <= tolerance) &&
        Math.abs(m.fLast - ridge.fFirst) <= JOIN_MATCH_HZ,
    );
    if (!earlier) {
      merged.push(ridge);
      continue;
    }
    earlier.tEnd = ridge.tEnd;
    earlier.fLast = ridge.fLast;
    earlier.fLow = Math.min(earlier.fLow, ridge.fLow);
    earlier.fHigh = Math.max(earlier.fHigh, ridge.fHigh);
    earlier.peakDb = Math.max(earlier.peakDb, ridge.peakDb);
  }
  return merged;
}

/**
 * Loads the WebAssembly module and returns the scrubber.
 *
 * @param {string|URL|Response|ArrayBuffer|ArrayBufferView|WebAssembly.Module} [source]
 *   Where `scrubber.wasm` is. Defaults to the copy shipped beside this file. With a
 *   bundler, pass the URL it gives you (for Vite: `import url from "scrubber-js/scrubber.wasm?url"`).
 */
export async function createScrubber(source = DEFAULT_WASM_URL) {
  let onProgress = null;
  const module = await compile(source);
  const { exports: wasm } = await WebAssembly.instantiate(module, {
    env: { scrubber_progress: (fraction) => onProgress?.(fraction) },
  });

  /** Runs `use` with a block of module memory, then frees it. */
  const withMemory = (bytes, use) => {
    const ptr = wasm.scrubber_alloc(bytes);
    if (!ptr) throw new Error("Scrubber: out of memory. Try a smaller segmentSeconds.");
    try {
      return use(ptr);
    } finally {
      wasm.scrubber_free(ptr, bytes);
    }
  };
  // Views are made fresh each time: growing the memory detaches old ones.
  const f32 = (ptr, length) => new Float32Array(wasm.memory.buffer, ptr, length);
  const f64 = (ptr, length) => new Float64Array(wasm.memory.buffer, ptr, length);

  /**
   * Scrubs (or only checks) one stretch of one channel.
   * @returns {{samples: Float32Array|null, replaced: number, ridge: number, ridges: object[]}}
   */
  const runSegment = (samples, sampleRate, config, seed, checkOnly) => {
    const reportLength = REPORT_HEAD + MAX_RIDGES_PER_SEGMENT * RIDGE_VALUES;
    return withMemory(samples.length * BYTES_F32, (samplesPtr) =>
      withMemory(CONFIG_ORDER.length * BYTES_F64, (configPtr) =>
        withMemory(reportLength * BYTES_F64, (reportPtr) => {
          f32(samplesPtr, samples.length).set(samples);
          f64(configPtr, CONFIG_ORDER.length).set(CONFIG_ORDER.map((read) => read(config)));
          const written = wasm.scrubber_scrub(
            samplesPtr, samples.length, sampleRate, configPtr, seed[0], seed[1], checkOnly ? 1 : 0, reportPtr, reportLength,
          );
          const report = f64(reportPtr, reportLength);
          const ridges = Array.from({ length: written }, (_, i) => {
            const [tStart, tEnd, fLow, fHigh, fFirst, fLast, peakDb] = report.subarray(
              REPORT_HEAD + i * RIDGE_VALUES, REPORT_HEAD + (i + 1) * RIDGE_VALUES,
            );
            return { tStart, tEnd, fLow, fHigh, fFirst, fLast, peakDb };
          });
          return {
            samples: checkOnly ? null : f32(samplesPtr, samples.length).slice(),
            replaced: report[0],
            ridge: report[1],
            ridges,
          };
        }),
      ),
    );
  };

  /**
   * @param {Float32Array[]} channels
   * @param {number} sampleRate
   * @param {object} options
   * @param {boolean} checkOnly
   */
  const run = (channels, sampleRate, options, checkOnly) => {
    const config = { ...DEFAULT_CONFIG, ...options.config };
    const segment = Math.max(1, Math.round((options.segmentSeconds ?? DEFAULT_SEGMENT_SECONDS) * sampleRate));
    const context = Math.round((options.contextSeconds ?? DEFAULT_CONTEXT_SECONDS) * sampleRate);
    const fade = Math.min(context, Math.round(JOIN_FADE_SECONDS * sampleRate));
    const length = channels[0]?.length ?? 0;
    // One piece if the whole thing fits in a segment and its context.
    const pieces = length <= segment + 2 * context ? 1 : Math.ceil(length / segment);
    const total = channels.length * pieces;
    const out = checkOnly ? null : channels.map((c) => new Float32Array(c.length));
    const ridges = [];
    const joins = [];
    let replaced = 0;
    let ridge = 0;
    let done = 0;

    channels.forEach((samples, channel) => {
      for (let piece = 0; piece < pieces; piece++) {
        const coreStart = pieces === 1 ? 0 : piece * segment;
        const coreEnd = pieces === 1 ? length : Math.min(length, coreStart + segment);
        const start = Math.max(0, coreStart - context);
        const end = Math.min(length, coreEnd + context);
        if (channel === 0 && piece > 0) joins.push(coreStart / sampleRate);
        onProgress = options.onProgress && ((fraction) => options.onProgress((done + fraction) / total));
        const result = runSegment(
          samples.subarray(start, end), sampleRate, config, seedWords(options.seed, channel, piece), checkOnly,
        );
        onProgress = null;
        done += 1;
        const share = (coreEnd - coreStart) / length;
        replaced += (result.replaced * share) / channels.length;
        ridge += (result.ridge * share) / channels.length;
        for (const r of result.ridges) {
          // Keep the part of each ridge that lies in this piece's own stretch.
          const tStart = Math.max(r.tStart + start / sampleRate, coreStart / sampleRate);
          const tEnd = Math.min(r.tEnd + start / sampleRate, coreEnd / sampleRate);
          if (tEnd > tStart) ridges.push({ ...r, channel, tStart, tEnd });
        }
        if (checkOnly) continue;
        // Blend into the previous piece over a short fade centred on the join.
        // A straight-line blend of two scrubbed copies keeps nothing of the original.
        const fadeStart = piece === 0 ? coreStart : Math.max(start, coreStart - (fade >> 1));
        const fadeEnd = piece === 0 ? coreStart : Math.min(end, fadeStart + fade);
        const target = out[channel];
        for (let i = fadeStart; i < fadeEnd; i++) {
          const w = (i - fadeStart + 0.5) / (fadeEnd - fadeStart);
          target[i] = (1 - w) * target[i] + w * result.samples[i - start];
        }
        // Past the fade, write up to where the next piece's fade will end.
        const writeEnd = piece === pieces - 1 ? coreEnd : Math.min(end, coreEnd + fade);
        target.set(result.samples.subarray(fadeEnd - start, writeEnd - start), fadeEnd);
      }
    });

    const tolerance = 4 * config.windowSeconds * config.hopFraction;
    const report = {
      replacedFraction: replaced,
      ridgeFraction: ridge,
      ridges: mergeAcrossJoins(ridges, joins, tolerance).sort((a, b) => b.peakDb - a.peakDb),
    };
    if (checkOnly) return report;
    let peak = 0;
    for (const channel of out) {
      for (let i = 0; i < channel.length; i++) peak = Math.max(peak, Math.abs(channel[i]));
    }
    return { channels: out, report: { ...report, peak } };
  };

  return {
    /**
     * Removes aircraft signatures (and, by default, steady hum) from audio.
     * The input is not modified.
     *
     * @param {Float32Array[]} channels One Float32Array per channel, all the same length.
     * @param {number} sampleRate
     * @param {object} [options]
     * @param {Partial<typeof DEFAULT_CONFIG>} [options.config]
     * @param {number} [options.seed] Fixed noise seed, for repeatable output. Leave unset in real use.
     * @param {number} [options.segmentSeconds] Piece length for long recordings (default 180).
     *   `Infinity` scrubs in one piece, exactly as upstream does, at the cost of memory.
     * @param {number} [options.contextSeconds] Audio each piece sees either side (default 10).
     * @param {(fraction: number) => void} [options.onProgress]
     * @returns {{channels: Float32Array[], report: ScrubReport}}
     */
    scrub: (channels, sampleRate, options = {}) => run(channels, sampleRate, options, false),

    /**
     * Reports long tonal ridges without changing anything. Note: faint lines
     * below the noise floor are not detectable frame by frame but are still
     * recoverable by a matched filter, so a clean check does not mean safe.
     *
     * @param {Float32Array[]} channels
     * @param {number} sampleRate
     * @param {object} [options] As for `scrub`.
     * @returns {Omit<ScrubReport, "peak">}
     */
    check: (channels, sampleRate, options = {}) => run(channels, sampleRate, options, true),

    /**
     * Looks for mains hum that carries the grid frequency. Needs about 30 s of audio.
     *
     * @param {Float32Array[]} channels
     * @param {number} sampleRate
     * @param {object} [options]
     * @param {number} [options.minDb] How far a harmonic must stand out (default 8).
     * @param {number} [options.maxSeconds] Only the first this-many seconds are analysed (default 600).
     * @returns {HumReport}
     */
    detectHum: (channels, sampleRate, options = {}) => {
      const seconds = options.maxSeconds ?? DEFAULT_HUM_SECONDS;
      const length = Math.min(channels[0]?.length ?? 0, Math.round(seconds * sampleRate));
      return withMemory(length * BYTES_F32, (samplesPtr) =>
        withMemory(HUM_VALUES * BYTES_F64, (reportPtr) => {
          const mono = f32(samplesPtr, length).fill(0);
          for (const channel of channels) {
            for (let i = 0; i < length; i++) mono[i] += channel[i] / channels.length;
          }
          wasm.scrubber_hum(samplesPtr, length, sampleRate, options.minDb ?? DEFAULT_HUM_MIN_DB, reportPtr);
          const [present, nominalHz, gridHz, wanderHz, count, ...pairs] = f64(reportPtr, HUM_VALUES);
          if (!present) return { present: false, nominalHz: null, gridHz: null, wanderHz: null, lines: [] };
          const lines = Array.from({ length: count }, (_, i) => ({ hz: pairs[2 * i], db: pairs[2 * i + 1] }));
          return { present: true, nominalHz, gridHz, wanderHz, lines };
        }),
      );
    },
  };
}
