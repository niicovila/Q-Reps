#!/bin/bash

# List of YAML files
yaml_files=(
    "/Users/nicolasvila/workplace/uni/tfg/tests/lightning_research_framework/configs/configs_beta/test_1.yaml"
    "/Users/nicolasvila/workplace/uni/tfg/tests/lightning_research_framework/configs/configs_beta/test_2.yaml"
    "/Users/nicolasvila/workplace/uni/tfg/tests/lightning_research_framework/configs/configs_beta/test_3.yaml"
    "/Users/nicolasvila/workplace/uni/tfg/tests/lightning_research_framework/configs/configs_beta/test_4.yaml"
    "/Users/nicolasvila/workplace/uni/tfg/tests/lightning_research_framework/configs/configs_beta/test_5.yaml"
    "/Users/nicolasvila/workplace/uni/tfg/tests/lightning_research_framework/configs/configs_beta/test_6.yaml"
    "/Users/nicolasvila/workplace/uni/tfg/tests/lightning_research_framework/configs/configs_beta/test_7.yaml"
    "/Users/nicolasvila/workplace/uni/tfg/tests/lightning_research_framework/configs/configs_beta/test_8.yaml"
    "/Users/nicolasvila/workplace/uni/tfg/tests/lightning_research_framework/configs/configs_beta/test_9.yaml"
    "/Users/nicolasvila/workplace/uni/tfg/tests/lightning_research_framework/configs/configs_beta/test_10.yaml"
    "/Users/nicolasvila/workplace/uni/tfg/tests/lightning_research_framework/configs/configs_beta/test_11.yaml"
    "/Users/nicolasvila/workplace/uni/tfg/tests/lightning_research_framework/configs/configs_beta/test_12.yaml"
    "/Users/nicolasvila/workplace/uni/tfg/tests/lightning_research_framework/configs/configs_beta/test_13.yaml"
    "/Users/nicolasvila/workplace/uni/tfg/tests/lightning_research_framework/configs/configs_beta/test_14.yaml"
    "/Users/nicolasvila/workplace/uni/tfg/tests/lightning_research_framework/configs/configs_beta/test_15.yaml"
    "/Users/nicolasvila/workplace/uni/tfg/tests/lightning_research_framework/configs/configs_beta/test_16.yaml"
    "/Users/nicolasvila/workplace/uni/tfg/tests/lightning_research_framework/configs/configs_beta/test_17.yaml"
    "/Users/nicolasvila/workplace/uni/tfg/tests/lightning_research_framework/configs/configs_beta/test_18.yaml"
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
