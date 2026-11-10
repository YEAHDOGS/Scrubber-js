// Compiles the Rust core to WebAssembly and copies it to dist/scrubber.wasm.
// Needs Rust with the wasm32-unknown-unknown target: rustup target add wasm32-unknown-unknown
import { execFileSync } from "node:child_process";
import { copyFileSync, mkdirSync, statSync } from "node:fs";
import { fileURLToPath } from "node:url";

const TARGET = "wasm32-unknown-unknown";
const ROOT = new URL("../", import.meta.url);
const MANIFEST = fileURLToPath(new URL("wasm/Cargo.toml", ROOT));
const BUILT = fileURLToPath(new URL(`wasm/target/${TARGET}/release/scrubber_core.wasm`, ROOT));
const DIST = fileURLToPath(new URL("dist/scrubber.wasm", ROOT));
const BYTES_PER_KB = 1024;

execFileSync("cargo", ["build", "--release", "--target", TARGET, "--manifest-path", MANIFEST], { stdio: "inherit" });
mkdirSync(fileURLToPath(new URL("dist/", ROOT)), { recursive: true });
copyFileSync(BUILT, DIST);
console.log(`dist/scrubber.wasm: ${Math.round(statSync(DIST).size / BYTES_PER_KB)} KB`);
