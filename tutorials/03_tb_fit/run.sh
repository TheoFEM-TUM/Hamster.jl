#!/bin/bash
# Fit (hconf), then the band structure with the fitted parameters (hconf_bands), then the figures.
# Usage: ./run.sh [-t THREADS]   (in this folder, or in solution/ with ../run.sh)
set -e
hamster "$@"
mv hamster_out.h5 hamster_out_fit.h5
hamster --conf hconf_bands "$@"
julia --project="$(dirname "$(readlink -f "$(command -v hamster)")")" "$(dirname "$0")/plot.jl"
