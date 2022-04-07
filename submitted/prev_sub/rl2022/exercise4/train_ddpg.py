import gym
import numpy as np
import time
from tqdm import tqdm
from typing import List, Tuple
import matplotlib.pyplot as plt

from rl2022.constants import EX4_PENDULUM_CONSTANTS as PENDULUM_CONSTANTS
from rl2022.constants import EX4_BIPEDAL_CONSTANTS as BIPEDAL_CONSTANTS
from rl2022.exercise4.agents import DDPG
from rl2022.exercise3.replay import ReplayBuffer

RENDER = False
    
PENDULUM_CONFIG = {
    "episode_length": 200,
    "max_timesteps": 100000,
    "eval_freq": 10000,
    "eval_episodes": 5,
    "policy_learning_rate": 1e-4,
    "critic_learning_rate": 1e-3,
    "critic_hidden_size": [400,],
    "policy_hidden_size": [300,],
    "tau": 0.1,
    "batch_size": 64,
    "gamma": 0.99,
    "buffer_capacity": int(1e6),
    "save_filename": "pendulum_latest.pt",
}
PENDULUM_CONFIG.update(PENDULUM_CONSTANTS)

BIPEDAL_CONFIG = {
    "eval_freq": 20000,
    "eval_episodes": 10,
    "policy_learning_rate": 1e-4, 
    "critic_learning_rate": 1e-3,
    "critic_hidden_size": [400,],
    "policy_hidden_size": [300,],
    "tau": 0.9,
    "batch_size": 64,
    "buffer_capacity": int(1e6),
    "episode_length": 200,
    "max_timesteps": 300000,
    "gamma": 0.99,
    "save_filename": "bipedal_latest.pt"
}
BIPEDAL_CONFIG.update(BIPEDAL_CONSTANTS)

# CONFIG = PENDULUM_CONFIG
CONFIG = BIPEDAL_CONFIG


def play_episode(
        env,
        agent,
        replay_buffer,
        train=True,
        explore=True,
        render=False,
        max_steps=200,
        batch_size=64,
):
    obs = env.reset()
    done = False
    losses = []
    if render:
        env.render()

    episode_timesteps = 0
    episode_return = 0

    while not done:
        action = agent.act(obs, explore=explore)
        nobs, reward, done, _ = env.step(action)
        if train:
            replay_buffer.push(
                np.array(obs, dtype=np.float32),
                np.array(action, dtype=np.float32),
                np.array(nobs, dtype=np.float32),
                np.array([reward], dtype=np.float32),
                np.array([done], dtype=np.float32),
            )
            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                loss = agent.update(batch)["q_loss"]
                losses.append(loss)

        episode_timesteps += 1
        episode_return += reward

        if render:
            env.render()

        if max_steps == episode_timesteps:
            break
        obs = nobs

    return episode_timesteps, episode_return, losses


def train(env: gym.Env, config, output: bool = True) -> Tuple[List[float], List[float]]:
    """
    Execute training of DDPG on given environment using the provided configuration

    :param env (gym.Env): environment to train on
    :param config: configuration dictionary mapping configuration keys to values
    :param output (bool): flag whether evaluation results should be printed
    :return (Tuple[List[float], List[float]]): eval returns during training, times of evaluation
    """
    timesteps_elapsed = 0

    agent = DDPG(
        action_space=env.action_space, observation_space=env.observation_space, **config
    )
    replay_buffer = ReplayBuffer(config["buffer_capacity"])

    eval_returns_all = []
    eval_times_all = []

    start_time = time.time()
    losses_all = []
    best_evaluation = 200.0
    j = 0
    with tqdm(total=config["max_timesteps"]) as pbar:
        while timesteps_elapsed < config["max_timesteps"]:
            elapsed_seconds = time.time() - start_time
            if elapsed_seconds > config["max_time"]:
                pbar.write(f"Training ended after {elapsed_seconds}s.")
                break
            agent.schedule_hyperparameters(timesteps_elapsed, config["max_timesteps"])
            episode_timesteps, _, losses = play_episode(
                env,
                agent,
                replay_buffer,
                train=True,
                explore=True,
                render=False,
                max_steps=config["episode_length"],
                batch_size=config["batch_size"],
            )
            timesteps_elapsed += episode_timesteps
            pbar.update(episode_timesteps)
            losses_all += losses

            if timesteps_elapsed % config["eval_freq"] < episode_timesteps:
                eval_returns = 0
                for _ in range(config["eval_episodes"]):
                    _, episode_return, _ = play_episode(
                        env,
                        agent,
                        replay_buffer,
                        train=False,
                        explore=False,
                        render=RENDER,
                        max_steps=config["episode_length"],
                        batch_size=config["batch_size"],
                    )
                    eval_returns += episode_return / config["eval_episodes"]
                if output:
                    pbar.write(
                        f"Evaluation at timestep {timesteps_elapsed} returned a mean returns of {eval_returns}"
                    )
                    # pbar.write(f"Epsilon = {agent.epsilon}")
                eval_returns_all.append(eval_returns)
                eval_times_all.append(time.time() - start_time)

                if eval_returns > best_evaluation:
                    best_evaluation =  eval_returns
                    print("Saving to: ", agent.save(str(j) + '_' + config["save_filename"]))
                    j = j + 1

                # if eval_returns >= config["target_return"]:
                #     pbar.write(
                #         f"Reached return {eval_returns} >= target return of {config['target_return']}"
                #     )
                #     break

    if config["save_filename"]:
        print("Saving to: ", agent.save('last_' + config["save_filename"]))

    return np.array(eval_returns_all), np.array(eval_times_all)


if __name__ == "__main__":
    env = gym.make(CONFIG["env"])
    _ = train(env, CONFIG)
    env.close()
