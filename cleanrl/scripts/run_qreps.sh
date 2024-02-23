#!/bin/bash


alphas=($(seq 0.01 0.1 10))

for alpha in "${alphas[@]}"
do
    python qreps/qreps.py --alpha $alpha --learning_rate 0.0025 --num_envs 4 --num_steps 128 --update_epochs 4 --update_policy_epochs 300 --policy_freq 1 --track
    python qreps/qreps.py --alpha $alpha --learning_rate 0.00025 --num_envs 4 --num_steps 128 --update_epochs 4 --update_policy_epochs 300 --policy_freq 1 --track
    python qreps/qreps.py --alpha $alpha --learning_rate 0.001 --num_envs 4 --num_steps 128 --update_epochs 4 --update_policy_epochs 300 --policy_freq 2 --track
    python qreps/qreps.py --alpha $alpha --learning_rate 0.01 --num_envs 4 --num_steps 128 --update_epochs 4 --update_policy_epochs 300 --policy_freq 2 --track

done

