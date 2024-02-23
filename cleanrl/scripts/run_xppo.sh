#!/bin/bash

alphas=( $(seq 0.5 3 0.5) )

for alpha in "${alphas[@]}"
do
    python xppo/xppo.py --alpha $alpha --track
    python xppo/xppo_v2.py --alpha $alpha --track
done
