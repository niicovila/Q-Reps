#!/bin/bash

# List of YAML files
yaml_files=(
    "/Users/nicolasvila/workplace/uni/tfg/tests/lightning_research_framework/configs/configs_beta/test_19.yaml"
    "/Users/nicolasvila/workplace/uni/tfg/tests/lightning_research_framework/configs/configs_beta/test_20.yaml"
    "/Users/nicolasvila/workplace/uni/tfg/tests/lightning_research_framework/configs/configs_beta/test_21.yaml"
    "/Users/nicolasvila/workplace/uni/tfg/tests/lightning_research_framework/configs/configs_beta/test_22.yaml"
    "/Users/nicolasvila/workplace/uni/tfg/tests/lightning_research_framework/configs/configs_beta/test_23.yaml"
)

# Loop through the YAML files and execute the command
counter=1
for yaml_file in "${yaml_files[@]}"
do
    output_path=".out_test_${counter}/"
    python scripts/train.py --config "$yaml_file" --path "$output_path"
    ((counter++))
done
