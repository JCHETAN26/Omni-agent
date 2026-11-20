#!/usr/bin/env bash
# macOS only: torch, faiss-cpu and scikit-learn wheels each bundle their own
# LLVM libomp. Loading two copies in one process aborts ("OMP: Error #15") or
# segfaults inside faiss. Point the other copies at torch's runtime (same LLVM
# OpenMP ABI) so the process loads exactly one. Linux wheels share libgomp and
# need nothing. Re-run after reinstalling faiss-cpu or scikit-learn.
set -euo pipefail
[[ "$(uname -s)" == "Darwin" ]] || { echo "not macOS; nothing to do"; exit 0; }
PY="${1:-.venv/bin/python}"
SP=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
TORCH_OMP="$SP/torch/lib/libomp.dylib"
[[ -f "$TORCH_OMP" ]] || { echo "torch's libomp not found under $SP" >&2; exit 1; }
for copy in "$SP/faiss/.dylibs/libomp.dylib" "$SP/sklearn/.dylibs/libomp.dylib"; do
  [[ -e "$copy" || -L "$copy" ]] || continue
  if [[ -L "$copy" && "$(readlink "$copy")" == "$TORCH_OMP" ]]; then
    echo "already shared: $copy"; continue
  fi
  mv "$copy" "$copy.bundled"
  ln -s "$TORCH_OMP" "$copy"
  echo "shared: $copy -> torch libomp (original kept as .bundled)"
done
