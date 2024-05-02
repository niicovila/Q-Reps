
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import argparse 
from copy import deepcopy
import os
import random
import time
import sys
import gymnasium as gym
import numpy as np
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Categorical
from ray import train
from ray.tune.search import Repeater
from ray.tune.search.hebo import HEBOSearch
import ray.tune as tune  # Import the missing package
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter
from ray.tune.schedulers import AsyncHyperBandScheduler
from train_func_saddle import tune_saddle, tune_saddle_decoupled
from train_func_elbe import tune_elbe, tune_elbe_decoupled
from ray.tune.search.basic_variant import BasicVariantGenerator

import logging
FORMAT = "[%(asctime)s]: %(message)s"
logging.basicConfig(level=logging.INFO, format=FORMAT)
SEED_OFFSET = 1

config_ray = {
    "exp_name": "QREPS",
    "seed": 0,
    "torch_deterministic": True,
    "cuda": True,
    "track": False,
    "wandb_project_name": "CC",
    "wandb_entity": None,
    "capture_video": False,

    "env_id": "CartPole-v1",

    # Algorithm
    "total_timesteps": 100000,
    "num_envs": tune.choice([4, 8, 16, 32, 64, 128]),
    "gamma": tune.choice([0.95, 0.97, 0.99, 0.999]),

    "total_iterations": tune.choice([256, 512, 1024, 2048]),
    "num_minibatches": tune.choice([4, 8, 16, 32, 64]),
    "update_epochs": tune.choice([10, 25, 50, 100, 150]),
    "update_epochs_policy": tune.choice([10, 25, 50, 100, 150]),

    "alpha": tune.choice([2, 4, 8, 12, 32, 64, 100]),  
    "eta": tune.choice([2, 4, 8, 12, 32, 64, 100]),

    # Learning rates
    "beta": tune.choice([3e-05, 0.0001, 0.00025, 0.0003, 0.001, 0.003]),
    "policy_lr": tune.choice([3e-05, 0.0001, 0.00025, 0.0003, 0.001, 0.003]),
    "q_lr": tune.choice([3e-05, 0.0001, 0.00025, 0.0003, 0.001, 0.003]),
    "anneal_lr": tune.choice([True, False]),

    # Layer Init
    "layer_init": tune.choice(["default", 
                               "orthogonal_gain", 
                               "orthogonal", 
                               "xavier_normal", 
                               "xavier_uniform", 
                               "kaiming_normal", 
                               "kaiming_uniform", 
                               "sparse"]),
    # Architecture
    "policy_activation": tune.choice(["Tanh", "ReLU", "Sigmoid", "ELU"]),
    "num_hidden_layers": tune.choice([2, 4, 8]),
    "hidden_size": tune.choice([32, 64, 128, 512]),
    "actor_last_layer_std": 0.01,

    "q_activation": tune.choice(["Tanh", "ReLU", "Sigmoid", "ELU"]),
    "q_num_hidden_layers": tune.choice([2, 4, 8]),
    "q_hidden_size": tune.choice([16, 32, 64, 128, 512]),
    "q_last_layer_std": 1.0,

    "average_critics": tune.choice([True, False]),
    "use_policy": tune.choice([True, False]),

    "parametrized_sampler" : False,
    # "sampler_activation": tune.choice(["Tanh", "ReLU", "Sigmoid", "ELU"]),
    # "sampler_num_hidden_layers": tune.choice([2, 4, 8]),
    # "sampler_hidden_size": tune.choice([32, 64, 128, 512]),
    # "sampler_last_layer_std": tune.choice([0.01, 0.1, 1.0]),


    # Optimization
    "q_optimizer": tune.choice(["Adam", "SGD", "RMSprop"]),  # "Adam", "SGD", "RMSprop
    "actor_optimizer": tune.choice(["Adam", "SGD", "RMSprop"]), 
    "sampler_optimizer": "Adam",
    "eps": 1e-8,

    # Options
    "saddle_point_optimization": True,
    "normalize_delta": tune.choice([True, False]),
    "gae": True,
    "gae_lambda": 0.95,
    "use_kl_loss": tune.choice([True, False]),
    "q_histogram": False,

    "target_network": False,
    "tau": 1.0,
    "target_network_frequency": 0, 

    "minibatch_size": 0,
    "num_iterations": 0,
    "num_steps": 0,
}

config_rand = {
    "exp_name": "QREPS",
    "seed": tune.grid_search([1, 2, 3, 4, 5]),
    "torch_deterministic": True,
    "cuda": True,
    "track": False,
    "wandb_project_name": "CC",
    "wandb_entity": None,
    "capture_video": False,

    "env_id": "CartPole-v1",

    # Algorithm
    "total_timesteps": 100000,
    "num_envs": 4,
    "gamma": 0.99,

    "total_iterations": 2048,
    "num_minibatches": 16,
    "update_epochs": 50,
    "update_epochs_policy": 50,

    "alpha": tune.choice([4, 8, 12, 32]),  
    "eta": None,

    # Learning rates
    "beta": tune.choice([3e-05, 0.0001, 0.00025, 0.0003, 0.001, 0.003]),
    "policy_lr": tune.choice([3e-05, 0.0001, 0.00025, 0.0003, 0.001, 0.003]),
    "q_lr": tune.choice([3e-05, 0.0001, 0.00025, 0.0003, 0.001, 0.003]),
    "anneal_lr": tune.choice([True, False]),


    # Layer Init
    "layer_init": "orthogonal_gain",
    # Architecture
    "policy_activation": tune.choice(["Tanh", "ReLU", "Sigmoid", "ELU"]),
    "num_hidden_layers": tune.choice([2, 4, 8]),
    "hidden_size": tune.choice([32, 64, 128, 512]),
    "actor_last_layer_std": tune.choice([0.01, 0.1, 1.0]),

    "q_activation": tune.choice(["Tanh", "ReLU", "Sigmoid", "ELU"]),
    "q_num_hidden_layers": tune.choice([2, 4, 8]),
    "q_hidden_size": tune.choice([32, 64, 128, 512]),
    "q_last_layer_std": tune.choice([0.01, 0.1, 1.0]),
    "use_policy": tune.choice([True, False]),

    "parametrized_sampler" : False,
    # "sampler_activation": tune.choice(["Tanh", "ReLU", "Sigmoid", "ELU"]),
    # "sampler_num_hidden_layers": tune.choice([2, 4, 8]),
    # "sampler_hidden_size": tune.choice([32, 64, 128, 512]),
    # "sampler_last_layer_std": tune.choice([0.01, 0.1, 1.0]),


    # Optimization
    "q_optimizer": "Adam",  # "Adam", "SGD", "RMSprop
    "actor_optimizer": "Adam",
    "sampler_optimizer": "Adam",
    "eps": 1e-8,

    # Options
    "saddle_point_optimization": True,
    "average_critics": True,
    "normalize_delta": False,
    "use_kl_loss": False,
    "q_histogram": False,
    "use_policy": True,
    "gae": False,
    "gae_lambda": 0.95,

    "target_network": False,
    "tau": 1.0,
    "target_network_frequency": 0, 

    "minibatch_size": 0,
    "num_iterations": 0,
    "num_steps": 0,
}

config_optuna = {
    "exp_name": "QREPS",
    "seed": 0,
    "torch_deterministic": True,
    "cuda": True,
    "track": False,
    "wandb_project_name": "CC",
    "wandb_entity": None,
    "capture_video": False,
    "env_id": "CartPole-v1",
    "total_timesteps": 50000,

    "num_envs": None,
    "gamma": None,
    "total_iterations": None,
    "num_minibatches": None,
    "update_epochs": None,
    "alpha": None,
    "eta": None,
    "beta": None,
    "policy_lr": None,
    "q_lr": None,
    "policy_activation": None,
    "num_hidden_layers": None,
    "hidden_size": None,
    "actor_last_layer_std": None,
    "q_activation": None,
    "q_hidden_size": None,
    "q_num_hidden_layers": None,
    "q_last_layer_std": None,
    "sampler_activation": None,
    "sampler_hidden_size": None,
    "sampler_num_hidden_layers": None,
    "sampler_last_layer_std": None,
    "q_optimizer": None,
    "actor_optimizer": None,
    "sampler_optimizer": None,
    "eps": None,
    "ort_init": None,
    "average_critics": True,
    "normalize_delta": None,
    "use_kl_loss": None,
    "anneal_lr": None,
    "parametrized_sampler": None,
    "saddle_point_optimization": True,
    "gae": False,
    "gae_lambda": 0.95,
    "q_histogram": False,
    "target_network": False,
    "tau": 1.0,
    "target_network_frequency": 0,
    "minibatch_size": 0,
    "num_iterations": 0,
    "num_steps": 0,
    "logging_callback": None,
}

def define_by_run_func(trial):
    config_optuna["num_envs"] = trial.suggest_categorical("num_envs", [4, 8, 64, 128, 256])
    config_optuna["gamma"] = trial.suggest_categorical("gamma", [0.9, 0.95, 0.97, 0.99, 0.999])
    config_optuna["total_iterations"] = trial.suggest_categorical("total_iterations", [512, 1024, 2048, 4096])
    config_optuna["num_minibatches"] = trial.suggest_categorical("num_minibatches", [8, 16, 32, 64])
    config_optuna["update_epochs"] = trial.suggest_categorical("update_epochs", [10, 25, 50, 100, 150])


    config_optuna["alpha"] = trial.suggest_categorical("alpha", [4, 8, 12, 32])

    config_optuna["beta"] = trial.suggest_categorical("beta", [3e-05, 0.0001, 0.00025, 0.0003, 0.001, 0.003])
    config_optuna["policy_lr"] = trial.suggest_categorical("policy_lr", [3e-05, 0.0001, 0.00025, 0.0003, 0.001, 0.003])
    config_optuna["q_lr"] = trial.suggest_categorical("q_lr", [3e-05, 0.0001, 0.00025, 0.0003, 0.001, 0.003])

    config_optuna["policy_activation"] = trial.suggest_categorical("policy_activation", ["Tanh", "ReLU", "Sigmoid"])
    config_optuna["num_hidden_layers"] = trial.suggest_categorical("num_hidden_layers", [2, 4])
    config_optuna["hidden_size"] = trial.suggest_categorical("hidden_size", [64, 128, 512])
    config_optuna["actor_last_layer_std"] = trial.suggest_categorical("actor_last_layer_std", [0.01, 0.1, 1.0])
    config_optuna["actor_optimizer"] = trial.suggest_categorical("actor_optimizer", ["Adam", "SGD", "RMSprop"])

    
    config_optuna["q_activation"] = trial.suggest_categorical("q_activation", ["Tanh", "ReLU", "Sigmoid", "ELU"])
    config_optuna["q_hidden_size"] = trial.suggest_categorical("q_hidden_size", [64, 128, 512])
    config_optuna["q_num_hidden_layers"] = trial.suggest_categorical("q_num_hidden_layers", [2, 4])
    config_optuna["q_last_layer_std"] = trial.suggest_categorical("q_last_layer_std", [0.01, 0.1, 1.0])
    config_optuna["q_optimizer"] = trial.suggest_categorical("q_optimizer", ["Adam", "SGD", "RMSprop"])

    config_optuna["parametrized_sampler"] = trial.suggest_categorical("parametrized_sampler", [True, False])

    if config_optuna["parametrized_sampler"]:
        config_optuna["sampler_hidden_size"] = trial.suggest_categorical("sampler_hidden_size", [64, 128, 512])
        config_optuna["sampler_activation"] = trial.suggest_categorical("sampler_activation", ["Tanh", "ReLU", "Sigmoid", "ELU"])
        config_optuna["sampler_num_hidden_layers"] = trial.suggest_categorical("sampler_num_hidden_layers", [2, 4, 8])
        config_optuna["sampler_last_layer_std"] = trial.suggest_categorical("sampler_last_layer_std", [0.01, 0.1, 1.0])
        config_optuna["sampler_optimizer"] = trial.suggest_categorical("sampler_optimizer", ["Adam", "SGD", "RMSprop"])

    
    config_optuna["eps"] = trial.suggest_categorical("eps", [1e-4, 1e-8])
    config_optuna["layer_init"] = trial.suggest_categorical("layer_init", ["default", 
                               "orthogonal_gain", 
                               "orthogonal", 
                               "xavier_normal", 
                               "xavier_uniform", 
                               "kaiming_normal", 
                               "kaiming_uniform", 
                               "sparse"])
    
    config_optuna["normalize_delta"] = trial.suggest_categorical("normalize_delta", [True, False])
    config_optuna["use_kl_loss"] = trial.suggest_categorical("use_kl_loss", [True, False])
    config_optuna["anneal_lr"] = trial.suggest_categorical("anneal_lr", [True, False])
    config_optuna["saddle_point_optimization"] = trial.suggest_categorical("saddle_point_optimization", [True, False])
    config_optuna["average_critics"] = trial.suggest_categorical("average_critics", [True, False])
    # Return all constants in a dictionary.
    return config

config_elbe = {
    "exp_name": "QREPS",
    "seed": 0,
    "torch_deterministic": True,
    "cuda": True,
    "track": False,
    "wandb_project_name": "CC",
    "wandb_entity": None,
    "capture_video": False,

    "env_id": "CartPole-v1",

    # Algorithm
    "total_timesteps": 100000,
    "num_envs": 4,
    "gamma": 0.99,

    "total_iterations": 2048,
    "num_minibatches": 16,
    "update_epochs": 50,
    "update_epochs_policy": 50,

    "alpha": tune.choice([4, 8, 12, 32]),  
    "eta": None,

    # Learning rates
    "policy_lr": tune.choice([3e-05, 0.0001, 0.00025, 0.0003, 0.001, 0.003]),
    "q_lr": tune.choice([3e-05, 0.0001, 0.00025, 0.0003, 0.001, 0.003]),
    "anneal_lr": tune.choice([True, False]),

    # Layer Init
    "layer_init": "orthogonal_gain",

    # Architecture
    "policy_activation": tune.choice(["Tanh", "ReLU", "Sigmoid", "ELU"]),
    "num_hidden_layers": tune.choice([2, 4, 8]),
    "hidden_size": tune.choice([32, 64, 128, 512]),
    "actor_last_layer_std": tune.choice([0.01, 0.1, 1.0]),

    "q_activation": tune.choice(["Tanh", "ReLU", "Sigmoid", "ELU"]),
    "q_num_hidden_layers": tune.choice([2, 4, 8]),
    "q_hidden_size": tune.choice([32, 64, 128, 512]),
    "q_last_layer_std": tune.choice([0.01, 0.1, 1.0]),
    "use_policy": tune.choice([True, False]),

    # Optimization
    "q_optimizer": "Adam",  # "Adam", "SGD", "RMSprop
    "actor_optimizer": "Adam",
    "eps": 1e-8,

    # Options
    "average_critics": True,
    "normalize_delta": False,
    "use_kl_loss": False,
    "q_histogram": False,
    "use_policy": True,
    "gae": False,
    "gae_lambda": 0.95,

    "target_network": False,
    "tau": 1.0,
    "target_network_frequency": 0, 

    "minibatch_size": 0,
    "num_iterations": 0,
    "num_steps": 0,
}

optuna = False
hebo = True

num_cpus = int(sys.argv[1])
# ray.init(address=os.environ['ip_head'])
current_dir = os.getcwd()
# print(ray.nodes())
scheduler = AsyncHyperBandScheduler(grace_period=200, time_attr="training_iteration", max_t=5000)
num_samples = 1500
seeds = 3

if optuna:
    search_alg = OptunaSearch(space=define_by_run_func, metric="reward", mode="max")
    search_alg = ConcurrencyLimiter(search_alg, max_concurrent=num_cpus)
    search_alg = Repeater(search_alg, repeat=seeds)

    tuner = tune.Tuner(  
        tune_saddle,
        tune_config=tune.TuneConfig(
            metric="reward",
            mode="max",
            search_alg=search_alg,
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        run_config=train.RunConfig(
            name="qreps-optuna-tuning",
            stop={"reward": 500},
            storage_path=current_dir + "/ray_results", 
        )
    )

    result_grid = tuner.fit()
    print("Best config is:", result_grid.get_best_result().config)
    results_df = result_grid.get_dataframe()
    results_df.to_csv("tune_results_optuna.csv")

else:

    if hebo:
        search_alg = HEBOSearch(metric="reward", mode="max")
        search_alg = Repeater(search_alg, repeat=seeds)
        config = config_ray
        run = 'hebo'
        scheduler=None
    else:
        search_alg = BasicVariantGenerator(constant_grid_search=True)
        config = config_rand
        run = 'random'

    tuner = tune.Tuner(  
        tune_elbe_decoupled,
        tune_config=tune.TuneConfig(
            metric="reward",
            mode="max",
            search_alg=search_alg,
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        run_config=train.RunConfig(
            name="qreps-tuning",
            storage_path=current_dir + "/ray_results", 
        ),
        param_space=config,
    )

    result_grid = tuner.fit()
    print("Best config is:", result_grid.get_best_result().config)
    results_df = result_grid.get_dataframe()
    results_df.to_csv(f"tune_results_{run}_{num_samples}.csv")

