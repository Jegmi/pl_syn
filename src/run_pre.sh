#!/usr/bin/env bash

# set working directory
cd "$(dirname "$0")"
module load new gcc/4.8.2 python/3.6.1
echo run gen on cluster node
bsub -W 23:59 -N -B -u jannes.jegminat@alumni.uni-heidelberg.de -J “gen” -R "rusage[mem=50000]" "python main.py $@"
#python main.py "$@"
module purge
