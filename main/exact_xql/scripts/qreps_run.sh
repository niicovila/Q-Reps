#!/bin/bash

# List of YAML files
yaml_files=(
    "/Users/nicolasvila/workplace/uni/tfg_v2/tests/lightning_research_framework/configs/configs_beta_qreps/test_1.yaml"
    "/Users/nicolasvila/workplace/uni/tfg_v2/tests/lightning_research_framework/configs/configs_beta_qreps/test_2.yaml"
    "/Users/nicolasvila/workplace/uni/tfg_v2/tests/lightning_research_framework/configs/configs_beta_qreps/test_3.yaml"
    "/Users/nicolasvila/workplace/uni/tfg_v2/tests/lightning_research_framework/configs/configs_beta_qreps/test_4.yaml"
    "/Users/nicolasvila/workplace/uni/tfg_v2/tests/lightning_research_framework/configs/configs_beta_qreps/test_5.yaml"
)

# Loop through the YAML files and execute the command
counter=1
for yaml_file in "${yaml_files[@]}"
do
    output_path=".out_test_${counter}/"
    python scripts/train.py --config "$yaml_file" --path "$output_path"
    ((counter++))
done
