#!/bin/bash

folder=$1
echo $folder

#### main phase

# input argument: number of parameters parnum
if [ $# -eq 1 ]
  then 
	parnum=20
       	echo set num of grid points $parnum
  else
	parnum=$2
fi

ssh jegminat@euler.ethz.ch bash ./$folder/run_main.sh $folder -s main # bayesian
counter=0
parend=$((parnum-1))
while [ $counter -le $parend ]
    do
    echo submit job $counter
    ssh jegminat@euler.ethz.ch bash ./$folder/run_main.sh $folder -s main -N $parnum -i $counter -c True
    ((counter++))
done

