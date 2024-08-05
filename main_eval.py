import os
import glob
import click
import numpy as np
import torch
import matplotlib.pyplot as plt
import pickle
from scipy.stats import bootstrap

import unstable_baselines.common.util as util
from unstable_baselines.common.logger import Logger
from unstable_baselines.model_based_rl.mbpo.agent import MBPOAgent
from unstable_baselines.common.util import set_device_and_logger, load_config, set_global_seed

from evaluate import evaluate
from custom_envs.custom_envs import get_custom_env



@click.command(
    context_settings=dict(
        ignore_unknown_options=True,
        allow_extra_args=True,
    )
)
@click.argument("config-path", type=str, required=True)
@click.option("--log-dir", default=os.path.join("logs", "rmbpo"))
@click.option("--gpu", type=int, default=-1)
@click.option("--seed", type=int, default=30)
@click.option("--info", type=str, default="")
@click.option("--load-dir1", type=str, default="")  # Directory for policy A (RMBPO)
@click.option("--load-dir2", type=str, default="")  # Directory for policy B (RMBPO)
@click.option("--num-eval-runs", type=int, default=1)
@click.argument("args", nargs=-1)
def main(
    config_path, log_dir, gpu, seed, info, load_dir1, load_dir2, num_eval_runs, args
):
    # Load configuration
    args = load_config(config_path, args)

    # Set global seed
    set_global_seed(seed)

    # Get environment name and perturbation parameters
    env_name = args["env_name"]
    perturb_params = args["perturb_param"]
    perturb_values = args["perturb_value"]

    # Initialize logger
    logger = Logger(
        log_dir, env_name, seed=seed, info_str=info, print_to_terminal=False
    )
    logger.log_str("logging to {}".format(logger.log_path))

    # Set device and logger
    set_device_and_logger(gpu, logger)

    # Log parameters
    logger.log_str_object("parameters", log_dict=args)

    # Initialize environment
    logger.log_str("Initializing Environment")
    env = get_custom_env(env_name, seed=seed)

    # Initialize evaluation environments and descriptions
    eval_envs = []
    eval_env_descriptions = []
    for param, value in zip(perturb_params, perturb_values, strict=True):
        eval_envs.append(
            get_custom_env(
                env_name, seed=seed, perturb_param=param, perturb_value=value
            )
        )
        eval_env_descriptions.append(f"{param}_{value}")

    # Get observation and action spaces
    obs_space = env.observation_space
    action_space = env.action_space

    # Initialize agent
    logger.log_str("Initializing Agent")
    
    # Load all the .pt files in the directories
    policy_files1 = glob.glob(os.path.join(load_dir1, '*.pt'))
    policy_files2 = glob.glob(os.path.join(load_dir2, '*.pt'))

    # Initialize lists to store results
    eval_mean_returns1 = []
    eval_mean_returns2 = []
    eval_stddev_returns1 = []
    eval_stddev_returns2 = []

    # Evaluate each policy file
    for i, (policy_file1, policy_file2) in enumerate(zip(policy_files1, policy_files2, strict=True)):
        # Print loading progress
        percent = (i / len(policy_files1)) * 100
        num_hashes = int(percent // 2)
        load_bar = '#' * num_hashes + '-' * (50 - num_hashes)
        print(f"Loading: [{load_bar}] {percent:.2f}% of weights complete")

        # Initialize agents and load weights
        agent1 = MBPOAgent(obs_space, action_space, env_name=env_name, **args["agent"])
        agent1.load_state_dict(torch.load(policy_file1, map_location=util.device))

        agent2 = MBPOAgent(obs_space, action_space, env_name=env_name, **args["agent"])
        agent2.load_state_dict(torch.load(policy_file2, map_location=util.device))

        # Initialize log dictionaries
        log_dict1 = {}
        log_dict2 = {}

        # Evaluate agents
        h = args["trainer"]["max_trajectory_length"]
        log_dict1.update(evaluate(eval_envs, agent1, eval_env_descriptions, h, num_eval_runs))
        log_dict2.update(evaluate(eval_envs, agent2, eval_env_descriptions, h, num_eval_runs))

        # Append evaluation results
        eval_mean_returns1.append([log_dict1[f"performance/eval_return_{desc}"] for desc in eval_env_descriptions])
        eval_mean_returns2.append([log_dict2[f"performance/eval_return_{desc}"] for desc in eval_env_descriptions])
        eval_stddev_returns1.append([log_dict1[f"performance/eval_std_{desc}"] for desc in eval_env_descriptions])
        eval_stddev_returns2.append([log_dict2[f"performance/eval_std_{desc}"] for desc in eval_env_descriptions])

    # Get the interquartile means of the returns
    means1 = np.mean(np.array(eval_mean_returns1), axis=0)
    means2 = np.mean(np.array(eval_mean_returns2), axis=0)

    # Bootstrap confidence intervals
    r1 = bootstrap((np.array(eval_mean_returns1),), np.mean, n_resamples=1000, axis=0, confidence_level=0.95)
    r2 = bootstrap((np.array(eval_mean_returns2),), np.mean, n_resamples=1000, axis=0, confidence_level=0.95)
    confidence_low_1 = r1.confidence_interval.low
    confidence_high_1 = r1.confidence_interval.high
    confidence_low_2 = r2.confidence_interval.low
    confidence_high_2 = r2.confidence_interval.high

    # Save the final data
    with open('final_data.pkl', 'wb') as f:
        pickle.dump((perturb_values, eval_mean_returns1, eval_mean_returns2, eval_stddev_returns1, eval_stddev_returns2,
                     confidence_low_1, confidence_high_1, confidence_low_2, confidence_high_2), f)

    # Plot results
    plt.plot(perturb_values, means1, marker='o', color='orange', label='RMBPO')
    plt.fill_between(perturb_values, confidence_low_1, confidence_high_1, color='orange', alpha=0.1)
    plt.plot(perturb_values, means2, marker='o', color='blue', label='MBPO')
    plt.fill_between(perturb_values, confidence_low_2, confidence_high_2, color='blue', alpha=0.1)

    plt.xlabel('param_value')
    plt.ylabel('eval_mean_return')
    plt.title('Evaluation Mean Return vs. Param Value')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()


