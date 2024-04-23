#!/bin/bash
echo "Current directory: $(pwd)"
cd tests/main/qreps/working_scripts/v1

for file in *.py; do
    if [[ -f "$file" ]]; then
        echo "Executing $file with 5 different seeds"
        for seed in {1..5}; do
            echo "Seed: $seed"
            python "$file" --seed $seed
        done
        echo "Finished executing $file with 5 different seeds"
    fi
done
