#!/bin/bash

alphas=( $(seq 0.0001 0.001 0.1) )
for alpha in "${alphas[@]}"
do
    echo "Running QREPS with alpha=$alpha"
    python qreps/qreps.py --alpha $alpha --track
done

