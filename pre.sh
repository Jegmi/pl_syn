#!/bin/bash

folder=$1
echo $folder

#### pre phase

# input argument: number of parameters parnum
if [ $# -eq 1 ]
  then 
	parnum=20
       	echo set num of grid points $parnum
  else
	parnum=$2
fi

## copy files to cluster
ssh jegminat@euler.ethz.ch mkdir ./$folder
scp -r ./src/* jegminat@euler.ethz.ch:/cluster/home/jegminat/$folder/

ssh jegminat@euler.ethz.ch bash ./$folder/run_pre.sh -s gen -N $parnum
