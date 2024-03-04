#!/bin/bash

betas=( $(seq 3 0.1 4) )

for beta in "${betas[@]}"
do
    # python xppo/xppo.py --beta $beta --track
    
    python xppo/xppo_v2.py --beta $beta --track
done
