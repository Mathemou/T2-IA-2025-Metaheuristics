import numpy as np
from game.core import SurvivalGame, GameConfig
from game.agents import RuleBasedAgent
from genetic_algo import GeneticAlgorithm
from test_trained_agent import test_agent
import os
import sys
import time

# TREINA O RULEBASEDAGENT COM ALGORITMO GENÉTICO

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

POPULATION_SIZE = 100
GENERATIONS = 100
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.6

def game_fitness_function(population: np.ndarray) -> float:
    game_config = GameConfig(num_players=len(population),fps=60)
    agents = [RuleBasedAgent(config = game_config,danger_threshold = weights[0],lookahead_cells = weights[1],diff_to_center_to_move = weights[2]) for weights in population]

    total_scores = np.zeros(len(agents))
    for i in range(3):
        game = SurvivalGame(config=game_config, render=True)
        while not game.all_players_dead():
            actions = []
            for idx,agent in enumerate(agents):
                if game.players[idx].alive:
                    state = game.get_state(idx, include_internals=True)
                    action = agent.predict(state)
                    actions.append(action)
                else:
                    actions.append(0)

            game.update(actions)
            if game.render:
                game.render_frame()
        for idx, player in enumerate(game.players):
            total_scores[idx] += player.score

    average_scores = total_scores / 3
    print(f"Melhor: {np.max(average_scores):.2f} | Média: {np.mean(average_scores):.2f} | Std: {np.std(average_scores):.2f}")
    return average_scores

def train_and_test():
    print("\n--- Iniciando Treinamento com Algoritmo Genético ---")
    
    ga = GeneticAlgorithm(
        population_size=POPULATION_SIZE,
        gene_length=3,
        mutation_rate=MUTATION_RATE,
        crossover_rate=CROSSOVER_RATE
    )

    best_weights_overall = None
    best_fitness_overall = -np.inf

    for generation in range(GENERATIONS):
        start_generation = time.time()
        current_best_weights, current_best_fitness  = ga.evolve(game_fitness_function,parallel=True)

        if current_best_fitness > best_fitness_overall:
            best_fitness_overall = current_best_fitness
            best_weights_overall = current_best_weights
            print(f'Backup generation -> Melhor Fitness Geral: {best_fitness_overall:.2f}')
            np.save("best_weights.npy", best_weights_overall)
        
        end = time.time()
        print(f"{generation + 1}/{GENERATIONS} Best Fitness: {current_best_fitness:.2f} Melhor Fitness Geral: {best_fitness_overall:.2f} ({end-start_generation:.2f} s)")
   

    print("\n--- Treinamento Concluído ---")
    print(f"Melhor Fitness Geral Alcançado: {best_fitness_overall:.2f}")

    if best_weights_overall is not None:
        np.save("best_weights.npy", best_weights_overall)
        print("Melhores pesos salvos em \'best_weights.npy\'")
 
        test_agent(best_weights_overall, num_tests=30, render=True)
    else:
        print("Nenhum peso ótimo encontrado.")

if __name__ == "__main__":
    train_and_test()