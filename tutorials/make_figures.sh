#!/bin/bash
# Runs all tutorials (the solutions of the exercises) and copies the figures to the documentation
# (docs/src/assets/tutorials). Also a test that all tutorials run.  Usage: ./make_figures.sh [-t THREADS]
set -e
T="$(cd "$(dirname "$0")" && pwd)"
ASSETS="$T/../docs/src/assets/tutorials"
PROJECT="$(dirname "$(readlink -f "$(command -v hamster)")")"
mkdir -p "$ASSETS"
plot() { julia --project="$PROJECT" "$@"; }

cd "$T/02_first_run" && hamster "$@" && plot plot.jl && cp bands.png "$ASSETS/02_bands.png"
cd "$T/03_tb_fit/solution" && bash ../run.sh "$@" && cp losses.png "$ASSETS/03_losses.png" && cp bands.png "$ASSETS/03_bands.png"
cd "$T/04_hyperopt" && hamster "$@" && plot plot.jl && cp hyperopt.png "$ASSETS/04_hyperopt.png"
cd "$T/05_soc/solution" && bash ../run.sh "$@" && cp bands.png "$ASSETS/05_bands.png"
cd "$T/06_ml/solution" && hamster "$@" && plot ../plot.jl && cp losses.png "$ASSETS/06_losses.png"
cd "$T/07_electrostatics/solution" && hamster "$@" && plot ../plot.jl && cp losses.png "$ASSETS/07_losses.png"
cd "$T/08_applications" && bash run.sh "$@" && cp dos.png "$ASSETS/08_dos.png"
echo "All tutorials finished; figures in $ASSETS"
