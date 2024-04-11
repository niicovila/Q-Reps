#!/bin/bash
CONFIG=1
export NUM_SEEDS=5

# Define a function to run the Python script
run_experiment() {
    num_envs=$1
    num_steps=$2
    anneal_lr=$3
    gamma=$4
    num_minibatches=$5
    policy_lr_start=$6
    q_lr_start=$7
    alpha=$8
    update_epochs=${9}
    autotune=${10}
    target_entropy_scale=${11}
    saddle_point_optimization=${12}
    use_kl_loss=${13}
    seed=${14}

    # Map lowercase strings to integer values
    anneal_lr_value=0
    autotune_value=0
    saddle_point_optimization_value=0
    use_kl_loss_value=0

    if [ "$anneal_lr" == "True" ]; then
        anneal_lr_value=1
    fi
    if [ "$autotune" == "True" ]; then
        autotune_value=1
    fi
    if [ "$saddle_point_optimization" == "True" ]; then
        saddle_point_optimization_value=1
    fi
    if [ "$use_kl_loss" == "True" ]; then
        use_kl_loss_value=1
    fi

    echo "starting experiment with config $CONFIG and seed $seed"
    echo "num_envs: $num_envs"

    # Run the Python script with the current configuration
    python Q-Reps/main/qreps/cleanrl/version_1/qreps_v1.py \
        --num_envs "$num_envs" \
        --num_steps "$num_steps" \
        --gamma "$gamma" \
        --num_minibatches "$num_minibatches" \
        --policy_lr_start "$policy_lr_start" \
        --q_lr_start "$q_lr_start" \
        --alpha "$alpha" \
        --update_epochs "$update_epochs" \
        --anneal_lr "$anneal_lr_value" \
        --target_entropy_scale "$target_entropy_scale" \
        --saddle_point_optimization "$saddle_point_optimization_value" \
        --use_kl_loss "$use_kl_loss_value" \
        --seed "$seed" \
        --wandb_project_name "Q-Reps-"$CONFIG""
}

export -f run_experiment

# Loop through each row in the DataFrame
tail -n +2 /Users/nicolasvila/workplace/uni/tfg_v2/Q-Reps/other/top_runs_CartPole.csv | while IFS=, read -r reward num_envs num_steps anneal_lr gamma num_minibatches policy_lr_start q_lr_start alpha eta update_epochs autotune target_entropy_scale saddle_point_optimization use_kl_loss; do
    # Run the function in parallel for each seed
    seq -w 0 $NUM_SEEDS| parallel run_experiment "$num_envs" "$num_steps" "$anneal_lr" "$gamma" "$num_minibatches" "$policy_lr_start" "$q_lr_start" "$alpha" "$update_epochs" "$autotune" "$target_entropy_scale" "$saddle_point_optimization" "$use_kl_loss" "{}"
    CONFIG=$((CONFIG+1))
done