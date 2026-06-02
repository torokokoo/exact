"""RL Evaluator — runs SNN genomes in Gymnasium environments.

Replaces EXAMM's backpropagate_stochastic() entirely.
Converts observations to SNN input, decodes output spikes to actions,
and returns average episode reward as fitness.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym  # type: ignore[no-redef]

from ..genome.snn_genome import SNNGenome
from ..genome.snn import SNN

logger = logging.getLogger(__name__)


class RLEvaluator:
    """Evaluate SNN genomes on RL environments.

    Input encoding: direct current injection (obs values fed as current each sim step).
    Output decoding: spike count argmax for discrete actions.
    Fitness: average total reward over n_episodes.
    """

    def __init__(
        self,
        env_name: str = "CartPole-v1",
        n_episodes: int = 5,
        t_sim: int = 15,
        max_steps_per_episode: int = 500,
        render: bool = False,
    ):
        self.env_name = env_name
        self.n_episodes = n_episodes
        self.t_sim = t_sim
        self.max_steps_per_episode = max_steps_per_episode
        self.render = render

        # Create a temporary env to get observation/action space info
        env = gym.make(env_name)
        self.observation_size = env.observation_space.shape[0]

        if hasattr(env.action_space, "n"):
            self.action_type = "discrete"
            self.n_actions = env.action_space.n
        else:
            self.action_type = "continuous"
            self.n_actions = env.action_space.shape[0]

        env.close()

    def evaluate(self, genome: SNNGenome) -> float:
        """Evaluate a genome by running it in the RL environment.

        Returns average total reward across n_episodes.
        """
        snn = genome.build_snn()
        total_reward = 0.0

        for ep in range(self.n_episodes):
            reward = self._run_episode(snn)
            total_reward += reward

        fitness = total_reward / self.n_episodes
        return fitness

    def _run_episode(self, snn: SNN) -> float:
        """Run a single episode and return total reward."""
        render_mode = "human" if self.render else None
        env = gym.make(self.env_name, render_mode=render_mode)

        obs, _ = env.reset()
        snn.reset()

        episode_reward = 0.0
        steps = 0

        while steps < self.max_steps_per_episode:
            # Run SNN for t_sim timesteps on current observation
            outputs = snn.run(np.array(obs, dtype=np.float64), self.t_sim)

            # Decode action
            action = self._decode_action(outputs, snn)

            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            steps += 1

            if terminated or truncated:
                break

        env.close()
        return episode_reward

    def _decode_action(self, outputs: list[float], snn: SNN) -> int | np.ndarray:
        """Decode SNN output to an action.

        For discrete: argmax of output node accumulated values.
        Ties broken by membrane potential, then random.
        """
        if self.action_type == "discrete":
            if not outputs or all(v == 0 for v in outputs):
                # No spikes — use membrane potential as tiebreaker
                potentials = snn.get_output_potentials()
                if potentials and any(p != 0 for p in potentials):
                    return int(np.argmax(potentials))
                return np.random.randint(self.n_actions)
            return int(np.argmax(outputs))
        else:
            # Continuous: scale output values to action space bounds
            return np.array(outputs[:self.n_actions], dtype=np.float32)

    def evaluate_with_render(self, genome: SNNGenome, n_episodes: int = 1) -> float:
        """Evaluate with rendering enabled (for visualization)."""
        snn = genome.build_snn()
        total_reward = 0.0
        env = gym.make(self.env_name, render_mode="human")

        for ep in range(n_episodes):
            obs, _ = env.reset()
            snn.reset()
            episode_reward = 0.0
            steps = 0

            while steps < self.max_steps_per_episode:
                outputs = snn.run(np.array(obs, dtype=np.float64), self.t_sim)
                action = self._decode_action(outputs, snn)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                steps += 1
                if terminated or truncated:
                    break

            total_reward += episode_reward
            logger.info(f"Episode {ep + 1}: reward = {episode_reward:.2f}")

        env.close()
        return total_reward / n_episodes
