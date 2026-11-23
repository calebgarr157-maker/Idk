#!/usr/bin/env bash
# Fetch the original UChicago 2022 Case 3 data (aardisaputra/uchicago_case3) into case3/case3data/.
set -euo pipefail
cd "$(dirname "$0")"
[ -d case3data ] && { echo "case3data already present"; exit 0; }
tmp=$(mktemp -d)
git clone --depth 1 https://github.com/aardisaputra/uchicago_case3 "$tmp/src"
cp -r "$tmp/src/case3data" ./case3data
rm -rf "$tmp"
echo "fetched case3data"
