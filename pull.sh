#!/bin/bash

folder=$1
mkdir $folder

ssh jegminat@euler.ethz.ch bash ./$folder/run_post.sh 

# copy output
scp -r jegminat@euler.ethz.ch:/cluster/home/jegminat/$folder/* ./$folder

# clean up
cat ./$folder/lsf.o* >> ./$folder/log.txt
rm ./$folder/lsf.*

# rename
mv ./$folder "$(date '+%d-%b-%Y-')$folder"

