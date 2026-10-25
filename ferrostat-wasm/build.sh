#!/bin/sh
# Build the browser version into ferrostat-wasm/web (serve that directory).
# Requires: rustup target add wasm32-unknown-unknown
#           cargo install wasm-bindgen-cli --version <same as Cargo.lock>
set -e
cd "$(dirname "$0")/.."
cargo build -p ferrostat-wasm --release --target wasm32-unknown-unknown
wasm-bindgen --target web --no-typescript --out-dir ferrostat-wasm/web/pkg \
    target/wasm32-unknown-unknown/release/ferrostat_wasm.wasm
if command -v wasm-opt >/dev/null 2>&1; then
    wasm-opt -Oz -o ferrostat-wasm/web/pkg/ferrostat_wasm_bg.wasm ferrostat-wasm/web/pkg/ferrostat_wasm_bg.wasm
fi
echo "built ferrostat-wasm/web — serve it, e.g.: python3 -m http.server -d ferrostat-wasm/web 8080"
