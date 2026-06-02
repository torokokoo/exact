"""Integration tests for SNN evaluation on RL environments."""

import pytest
from snn_examm.genome.snn_genome import create_seed_genome

# Skip all tests if gymnasium is not installed
gym = pytest.importorskip("gymnasium")


class TestRLEvaluator:
    def test_evaluator_creates(self):
        from snn_examm.evaluation.rl_evaluator import RLEvaluator
        evaluator = RLEvaluator(env_name="CartPole-v1", n_episodes=1, t_sim=5)
        assert evaluator.observation_size == 4
        assert evaluator.n_actions == 2
        assert evaluator.action_type == "discrete"

    def test_evaluate_produces_finite_reward(self):
        from snn_examm.evaluation.rl_evaluator import RLEvaluator
        evaluator = RLEvaluator(env_name="CartPole-v1", n_episodes=1, t_sim=5)
        genome = create_seed_genome(
            evaluator.observation_size,
            evaluator.n_actions,
        )
        fitness = evaluator.evaluate(genome)
        assert isinstance(fitness, float)
        assert fitness > 0  # CartPole gives at least 1 reward

    def test_evaluate_multiple_episodes(self):
        from snn_examm.evaluation.rl_evaluator import RLEvaluator
        evaluator = RLEvaluator(env_name="CartPole-v1", n_episodes=3, t_sim=10)
        genome = create_seed_genome(
            evaluator.observation_size,
            evaluator.n_actions,
        )
        fitness = evaluator.evaluate(genome)
        assert fitness > 0

    def test_acrobot_env(self):
        from snn_examm.evaluation.rl_evaluator import RLEvaluator
        evaluator = RLEvaluator(env_name="Acrobot-v1", n_episodes=1, t_sim=5, max_steps_per_episode=100)
        genome = create_seed_genome(
            evaluator.observation_size,
            evaluator.n_actions,
        )
        fitness = evaluator.evaluate(genome)
        assert isinstance(fitness, float)

    def test_mountain_car_env(self):
        from snn_examm.evaluation.rl_evaluator import RLEvaluator
        evaluator = RLEvaluator(env_name="MountainCar-v0", n_episodes=1, t_sim=5, max_steps_per_episode=100)
        genome = create_seed_genome(
            evaluator.observation_size,
            evaluator.n_actions,
        )
        fitness = evaluator.evaluate(genome)
        assert isinstance(fitness, float)
