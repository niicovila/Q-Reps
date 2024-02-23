#!/bin/bash


alphas=( $(seq 0.1 3 0.1) )
for alpha in "${alphas[@]}"
do
    python qreps/qreps.py --alpha $alpha --track
done

