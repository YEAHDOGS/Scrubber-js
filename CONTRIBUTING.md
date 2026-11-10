# Contributing to Scrubber-js

Scrubber-js is a WebAssembly port of
[flyover-scrub](https://github.com/lizTheDeveloper/flyover-scrub). It removes
aircraft Doppler signatures and grid hum from audio, in the browser, so a
recording can't be located from the planes flying over it.

Contributions are welcome: bug reports, tests, speed, memory, documentation,
and keeping the port in step with upstream.

## What lives where

| Path | What it is |
|---|---|
| `js/index.js` | The JavaScript API. Loads the module, moves samples in and out, splits long recordings into segments. |
| `js/index.test.js` | Tests for the wrapper and the built module. |
| `wasm/src/` | The signal processing, in Rust: `scrub.rs`, `enf.rs`, `stft.rs`, `filters.rs`, `rng.rs`, and the WebAssembly interface in `lib.rs`. |
| `wasm/tests/` | Ports of the upstream tests, including the attack scorer that measures the defence. |
| `dist/scrubber.wasm` | The built module. It is committed so the package installs without a Rust toolchain. |
| `src/`, `tests/`, `eval/`, `research/` | Upstream's Python package and research, unchanged. This is the reference. |

## Setting up

You need Node 20 or newer. To change anything under `wasm/` you also need Rust:

```sh
rustup target add wasm32-unknown-unknown
```

Then:

```sh
npm test            # JavaScript tests against dist/scrubber.wasm
npm run test:core   # Rust tests, including the attack before and after scrubbing
npm run build       # rebuild dist/scrubber.wasm from wasm/
```

There are no npm dependencies and one Rust dependency (`rustfft`).

## Working on the JavaScript

Most wrapper changes need no Rust at all: edit `js/index.js`, run `npm test`.
Keep it dependency-free and keep it one file unless it grows past what one file
can hold comfortably.

## Working on the Rust core

1. Make the change under `wasm/src/`.
2. Run `npm run test:core`. It must pass with no compiler warnings.
3. Run `npm run build`, then `npm test`.
4. Commit `dist/scrubber.wasm` together with the source change, in the same commit.

If you change the WebAssembly interface (the `extern "C"` functions in
`lib.rs`, or the layout of the config and report buffers), change
`js/index.js` to match in the same commit. The constants that describe those
layouts carry the same names on both sides.

## Rules for changes

**The defence has to be measured, not assumed.** `wasm/tests/core.rs` runs the
location attack on a recording before and after scrubbing. A change to the
scrubber is only acceptable if the attack still finds the raw recording and
still loses the scrubbed one. If you change those tests, say why in the pull
request.

**Upstream is the reference for the algorithm.** This port follows
`src/flyover_scrub/scrub.py` and `enf.py` step for step, with the same edge
rules as the SciPy filters they use. If you find a flaw in the method itself,
or a better method, please raise it upstream too, so both implementations
benefit. Where the port differs on purpose, the difference is listed in the
README; add to that list if you add another.

**This is a defence.** The attack scorer exists in the tests so the scrubber
can be checked. Features for locating recordings, or for fetching bulk or
historical flight data, are out of scope and will not be merged.

**No new dependencies** without a clear case: what it costs in size and what
it saves. Native code that fits in a few dozen lines is preferred.

**Nothing leaves the device.** The library takes samples and returns samples.
It must never make a network request other than loading its own module.

## Pull requests

- Branch from `main`; one topic per pull request.
- Describe what changed and how you tested it. Paste the test output if you
  touched the scrubber.
- Don't commit audio or video files.

## Keeping in step with upstream

```sh
git remote add upstream https://github.com/lizTheDeveloper/flyover-scrub.git
git fetch upstream
git merge upstream/main
```

The Python under `src/` should merge cleanly, since this repo doesn't edit it.
After a merge, read the diff of `src/flyover_scrub/scrub.py` and `enf.py` and
carry any change of behaviour into `wasm/src/`.

## Reporting a weakness

If you find a way to recover a flight signature from scrubbed audio, that is
the most useful report this project can get. Open an issue with the smallest
example that shows it. If you would rather not publish the details yet, say so
in the issue and leave them out; a maintainer will follow up.

## Licence

MIT, the same as upstream. By contributing you agree that your contribution is
released under it.
