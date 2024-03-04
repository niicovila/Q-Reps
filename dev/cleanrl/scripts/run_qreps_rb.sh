#!/bin/bash

# Generate 50 evenly spaced alpha values between 0.0001 and 5
alphas=( $(seq 0 49 | awk '{print 0.0001 + $1 * ((5 - 0.0001) / 49)}') )

# learning_rates = [0.1, 1e-2, 1e-3, 1e-4]
learning_rates=(0.1 0.01 0.001 0.0001)

for alpha in "${alphas[@]}"
do
    for learning_rate in "${learning_rates[@]}"
    do
        echo "Running QREPS with alpha=$alpha and learning_rate=$learning_rate"
        python qreps/qreps_rb.py --alpha $alpha --learning_rate $learning_rate --track
    done
done
