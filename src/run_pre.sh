#!/usr/bin/env bash

# set working directory
cd "$(dirname "$0")"
module load new gcc/4.8.2 python/3.6.1
echo run pre on entry node
python main.py "$@"
module purge
