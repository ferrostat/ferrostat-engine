#!/usr/bin/env bash
# Collects the release assets of one build into dist/. Shared by
# .gitlab-ci.yml and .github/workflows/release.yml so both publish the same
# files.
#
#   ci/package.sh <rust-target> <version>    after cargo build --release --target <rust-target>
#   ci/package.sh wasm32-web <version>       after ferrostat-wasm/build.sh
#
# Asset names carry no version, so "latest release" download links stay valid
# (the home page in site/ links to them):
#   ferrostat-<platform>.tar.gz|.zip     both programs plus README/LICENSE
#   ferrostat-<platform>[.exe]           command line
#   ferrostat-gui-<platform>[.exe]       desktop application
#   ferrostat-web.zip                    browser build, for self-hosting
set -euo pipefail

target=$1
version=$2
dir=ferrostat-$version-$target
mkdir -p dist "$dir"
cp README.md LICENSE* "$dir/" 2>/dev/null || true

if [ "$target" = wasm32-web ]; then
  cp -r ferrostat-wasm/web/. "$dir/"
  zip -qr dist/ferrostat-web.zip "$dir"
  exit 0
fi

case $target in
  x86_64-unknown-linux-gnu)      platform=linux-x86_64 ;;
  aarch64-unknown-linux-gnu)     platform=linux-aarch64 ;;
  armv7-unknown-linux-gnueabihf) platform=linux-armv7 ;;
  x86_64-pc-windows-msvc)        platform=windows-x86_64 ;;
  aarch64-apple-darwin)          platform=macos-aarch64 ;;
  x86_64-apple-darwin)           platform=macos-x86_64 ;;
  *) echo "package.sh: unknown target $target" >&2; exit 1 ;;
esac
ext=
case $target in *-windows-*) ext=.exe ;; esac

bin=target/$target/release
cp "$bin/ferrostat$ext" "$bin/ferrostat-gui$ext" "$dir/"
cp "$bin/ferrostat$ext" "dist/ferrostat-$platform$ext"
cp "$bin/ferrostat-gui$ext" "dist/ferrostat-gui-$platform$ext"
if [ -n "$ext" ]; then
  if command -v 7z >/dev/null; then
    7z a -bd "dist/ferrostat-$platform.zip" "$dir" >/dev/null
  else
    powershell -NoProfile -Command "Compress-Archive -Path '$dir' -DestinationPath 'dist/ferrostat-$platform.zip'"
  fi
else
  tar czf "dist/ferrostat-$platform.tar.gz" "$dir"
fi
