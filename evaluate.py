import numpy as np
import torch


@torch.no_grad()
def evaluate(eval_envs, agent, descriptions, max_trajectory_length, num_eval_trajectories):
    # Ensure eval_envs and descriptions are lists
    eval_envs = eval_envs if isinstance(eval_envs, list) else [eval_envs]
    descriptions = descriptions if isinstance(descriptions, list) else [descriptions]
    
    ret_info = {}
    for i, env in enumerate(eval_envs):
        traj_returns, traj_lengths, traj_successes = [], [], []
        for _ in range(num_eval_trajectories):
            traj_return, traj_length, success = 0, 0, False
            obs, info = env.reset()
            for _ in range(max_trajectory_length):
                action = agent.select_action(obs, deterministic=True)["action"][0]
                next_obs, reward, done, truncated, info = env.step(action)
                traj_return += reward
                obs = next_obs
                traj_length += 1
                success = success or info.get("success", False)
                if done or truncated:
                    break
            traj_lengths.append(traj_length)
            traj_returns.append(traj_return)
            traj_successes.append(success)

        var, std = (np.var(traj_returns, ddof=1), np.sqrt(np.var(traj_returns, ddof=1))) if num_eval_trajectories > 1 else (0, 0)
        d = descriptions[i]
        ret_info.update({
            f"performance/eval_return_{d}": np.mean(traj_returns),
            f"performance/eval_median_{d}": np.median(traj_returns),
            f"performance/eval_variance_{d}": var,
            f"performance/eval_std_{d}": std,
            f"performance/eval_vmr_{d}": var / np.mean(traj_returns),
            f"performance/eval_cv_{d}": std / np.mean(traj_returns),
            f"performance/eval_mad_{d}": np.median(np.abs(traj_returns - np.mean(traj_returns))),  # MAD is median absolute deviation
            f"performance/eval_worst_{d}": np.amin(traj_returns),
            f"performance/eval_best_{d}": np.amax(traj_returns),
            f"performance/eval_length_{d}": np.mean(traj_lengths),
        })
        if traj_successes:
            ret_info[f"performance/eval_success_{d}"] = np.mean(traj_successes)

    ret_info["performance/eval_return"] = np.mean([ret_info[f"performance/eval_return_{d}"] for d in descriptions])
    return ret_info

