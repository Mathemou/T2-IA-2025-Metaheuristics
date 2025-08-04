import numpy as np

class GeneticAlgorithm:
    def __init__(self, population_size: int, gene_length: int, mutation_rate: float, crossover_rate: float):
        self.population_size = population_size
        self.gene_length = gene_length
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = self._initialize_population()

    def _initialize_population(self) -> np.ndarray:
        return np.random.uniform(0, 5, (self.population_size, self.gene_length))

    def _select_parents(self, fitness_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tournament_size = 3
        parent1_idx = max(np.random.choice(self.population_size, tournament_size), 
                        key=lambda i: fitness_scores[i])
        parent2_idx = max(np.random.choice(self.population_size, tournament_size), 
                        key=lambda i: fitness_scores[i])
        return self.population[parent1_idx], self.population[parent2_idx]

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if np.random.rand() < self.crossover_rate and self.gene_length > 1:
            crossover_point = np.random.randint(1, self.gene_length)
            child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
            child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
            return child1, child2
        return parent1, parent2

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        mutation_mask = np.random.rand(self.gene_length) < self.mutation_rate
        individual[mutation_mask] += np.random.normal(0, 0.1, np.sum(mutation_mask))
        return individual

    def evolve(self, fitness_function, parallel = False) -> np.ndarray:
        if parallel:
            fitness_scores = np.array(fitness_function(self.population))
        else:
            fitness_scores = np.array([fitness_function(individual) for individual in self.population])

        best_individual_idx = np.argmax(fitness_scores)
        best_individual = self.population[best_individual_idx].copy()

        new_population = [best_individual] 

        while len(new_population) < self.population_size:
            parent1, parent2 = self._select_parents(fitness_scores)
            child1, child2 = self._crossover(parent1, parent2)
            new_population.append(self._mutate(child1))
            if len(new_population) < self.population_size:
                new_population.append(self._mutate(child2))
        
        self.population = np.array(new_population)
        
        return self.population[best_individual_idx], fitness_scores[best_individual_idx]

    def get_population(self) -> np.ndarray:
        return self.population

    def set_population(self, new_population: np.ndarray):
        self.population = new_population

