from game.core import SurvivalGame, GameConfig
from game.agents import RuleBasedAgent
import numpy as np
from test_trained_agent import test_agent
from gwo import GreyWolfOptimizer  # você criaria esse arquivo/classe
import time

def game_fitness_function(weights):
    config = GameConfig()
    agent = RuleBasedAgent(config, danger_threshold=weights[0], lookahead_cells=weights[1], diff_to_center_to_move=weights[2])
    scores = []

    for _ in range(3):
        game = SurvivalGame(config, render=False)
        while not game.all_players_dead():
            state = game.get_state(0, include_internals=True)
            action = agent.predict(state)
            game.update([action])
        scores.append(game.players[0].score)

    return np.mean(scores)

if __name__ == "__main__":
    gwo = GreyWolfOptimizer(
        fitness_function=game_fitness_function,
        dim=3,
        n_wolves=100,
        max_iter=1000
    )

    best_weights, best_score = gwo.optimize()
    print(f"Melhores pesos: {best_weights} com score {best_score}")
    np.save("best_weights.npy", best_weights)

    test_agent(best_weights, num_tests=30, render=False)
