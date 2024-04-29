import time
import optuna

from utils.tuner import Tuner

tuner = Tuner(
    script="algos/qreps/qreps_main.py",
    metric="charts/episodic_return",
    metric_last_n_average_window=50,
    direction="maximize",
    aggregation_type="average",
    target_scores={
        "CartPole-v1": [0, 500],
        "Acrobot-v1": [-500, 0],
        "MountainCar-v0": [-200, 0],
        "LunarLander-v2": [-200, 250],
        "Pendulum-v1": [-16, 0],
    },
    params_fn=lambda trial: {
        "q-lr": trial.suggest_float("q-lr", 0.0001, 0.003, log=True),
        "num-envs": trial.suggest_categorical("num-envs", [4, 16, 32, 64]),
        "num-minibatches": trial.suggest_categorical("num-minibatches", [4, 8, 16, 32, 64]),
        "update-epochs": trial.suggest_categorical("update-epochs", [5, 10]),
        "total-iterations": trial.suggest_categorical("total-iterations", [128, 256, 512, 1024]),
        "alpha": trial.suggest_categorical("alpha", [2, 4, 8, 16]),
        "total-timesteps": 100000,

    },
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    sampler=optuna.samplers.TPESampler(),
)
start_time = time.time()
tuner.tune(
    num_trials=4,
    num_seeds=3,
)
print("Time taken: ", time.time() - start_time)