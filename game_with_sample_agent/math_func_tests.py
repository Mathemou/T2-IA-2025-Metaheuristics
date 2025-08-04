import numpy as np
from genetic_algo import GeneticAlgorithm

def quadratic_function(x):
    return -(x - 5)**2 + 25

def fitness_wrapper(func, individual):
    return func(individual[0])

def run_math_test(func, gene_length=1, population_size=100, generations=50, mutation_rate=0.05, crossover_rate=0.7):
    print(f"\n--- Testando Algoritmo Genético com a função: {func.__name__} ---")
    ga = GeneticAlgorithm(population_size=population_size, gene_length=gene_length, mutation_rate=mutation_rate, crossover_rate=crossover_rate)

    best_individual_overall = None
    best_fitness_overall = -np.inf

    for i in range(generations):
        best_individual_generation,current_best_fitness = ga.evolve(lambda ind: fitness_wrapper(func, ind))

        if current_best_fitness > best_fitness_overall:
            best_fitness_overall = current_best_fitness
            best_individual_overall = best_individual_generation
        
        # print(f"Geração {i+1}: Melhor indivíduo = {best_individual_generation[0]:.4f}, Fitness = {current_best_fitness:.4f}")

    print(f"Resultado Final para {func.__name__}:")
    print(f"Melhor indivíduo encontrado: {best_individual_overall[0]:.4f}")
    print(f"Melhor Fitness (valor da função): {best_fitness_overall:.4f}")

if __name__ == "__main__":
    # Teste com a função quadrática (max = 25 em x=5)
    run_math_test(quadratic_function, gene_length=1, population_size=200, generations=300, mutation_rate=0.1, crossover_rate=0.8)