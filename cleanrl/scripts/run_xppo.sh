#!/bin/bash

betas=( $(seq 0.1 0.1 1) )

for beta in "${betas[@]}"
do
    python xppo/xppo.py --beta $beta --track
    python xppo/xppo_v2.py --beta $beta --track
done
