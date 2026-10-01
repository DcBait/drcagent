#!/usr/bin/env bash
# Fetch the open-source SkyWater sky130 files this project needs (sparse clones, ~15 MB).
#   - DRC deck:        efabless/mpw_precheck   (GPL-3.0, not vendored in this repo)
#   - rule docs:       google/skywater-pdk     docs/rules (Apache-2.0)
#   - standard cells:  google/skywater-pdk-libs-sky130_fd_sc_hd (Apache-2.0)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p pdk && cd pdk

clone() {  # clone <url> <dir> <paths...>
  local url=$1 dir=$2; shift 2
  [ -d "$dir/.git" ] || git clone -q --depth 1 --filter=blob:none --sparse "$url" "$dir"
  git -C "$dir" sparse-checkout set --no-cone "$@"
}

clone https://github.com/efabless/mpw_precheck.git mpw_precheck /checks/tech-files/sky130A_mr.drc
clone https://github.com/google/skywater-pdk.git skywater-pdk /docs/rules/periphery/
clone https://github.com/google/skywater-pdk-libs-sky130_fd_sc_hd.git sc_hd \
  '/cells/inv/sky130_fd_sc_hd__inv_[124].gds' '/cells/nand2/sky130_fd_sc_hd__nand2_[12].gds' \
  '/cells/nor2/sky130_fd_sc_hd__nor2_[12].gds' /cells/nand3/sky130_fd_sc_hd__nand3_1.gds \
  /cells/buf/sky130_fd_sc_hd__buf_1.gds
echo "PDK files ready in $(pwd)"
