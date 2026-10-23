#!/bin/bash
# Large cell, DOS with KPM, band edges with the sparse eigensolver, figures.  Usage: ./run.sh [-t THREADS]
set -e
cd "$(dirname "$0")"
julia make_cell.jl
hamster --conf hconf_dos "$@"      # first: a run without eigenvalues (skip_diag) removes an old Es.dat
hamster --conf hconf_edges "$@"
julia --project="$(dirname "$(readlink -f "$(command -v hamster)")")" plot.jl
