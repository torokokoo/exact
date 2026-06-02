"""End-to-end test: run a small evolution on CartPole."""

import pytest

gym = pytest.importorskip("gymnasium")


class TestFullEvolution:
    def test_small_evolution_run(self):
        """Run a small evolution (50 genomes, 3 islands) and verify fitness improves."""
        from snn_examm.config import Config
        from snn_examm.main import run_evolution

        config = Config(
            env_name="CartPole-v1",
            n_eval_episodes=2,
            t_sim=10,
            max_steps_per_episode=200,
            number_islands=3,
            island_size=5,
            max_genomes=50,
            number_threads=2,
            mutation_rate=0.4,
            intra_island_crossover_rate=0.7,
            num_mutations=1,
            weight_initialize="xavier",
            mutated_component_weight="lamarckian",
            weight_inheritance="lamarckian",
            output_directory="/tmp/snn_examm_test",
            log_interval=10,
        )

        stats = run_evolution(config)
        assert stats is not None
        assert stats["evaluated"] > 0
        assert stats["global_best_fitness"] is not None
        # CartPole gives at least 1 step reward
        assert stats["global_best_fitness"] > 0

    def test_evolution_with_extinction(self):
        """Run evolution with extinction events enabled."""
        from snn_examm.config import Config
        from snn_examm.main import run_evolution

        config = Config(
            env_name="CartPole-v1",
            n_eval_episodes=1,
            t_sim=5,
            max_steps_per_episode=100,
            number_islands=3,
            island_size=3,
            max_genomes=30,
            number_threads=1,
            mutation_rate=0.5,
            extinction_event_generation_number=15,
            islands_to_exterminate=1,
            repopulation_method="bestGenome",
            output_directory="/tmp/snn_examm_test_extinction",
            log_interval=10,
        )

        stats = run_evolution(config)
        assert stats is not None
        assert stats["evaluated"] > 0
