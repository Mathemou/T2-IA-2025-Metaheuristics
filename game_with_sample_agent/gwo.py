import numpy as np


# Classe GreyWolfOptimizer implementa o algoritmo de otimização por lobos cinzentos
class GreyWolfOptimizer:
    def __init__(self, fitness_function, dim, n_wolves=20, max_iter=100, bounds=None):
        self.fitness_function = fitness_function
        self.dim = dim
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        
        # Definir limites para os pesos
        if bounds is None:
            self.lower_bound, self.upper_bound = 0, 5
        else:
            self.lower_bound, self.upper_bound = bounds
            
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (n_wolves, dim))

    def optimize(self):
        alpha, beta, delta = None, None, None
        alpha_score, beta_score, delta_score = -np.inf, -np.inf, -np.inf

        for iter in range(self.max_iter):
            for i in range(self.n_wolves):
                fitness = self.fitness_function(self.positions[i])

                if fitness > alpha_score:
                    alpha, alpha_score = self.positions[i].copy(), fitness
                elif fitness > beta_score:
                    beta, beta_score = self.positions[i].copy(), fitness
                elif fitness > delta_score:
                    delta, delta_score = self.positions[i].copy(), fitness

            a = 2 - iter * (2 / self.max_iter)
            for i in range(self.n_wolves):
                for j in range(self.dim):
                    r1, r2 = np.random.rand(), np.random.rand()
                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2
                    D_alpha = abs(C1 * alpha[j] - self.positions[i][j])
                    X1 = alpha[j] - A1 * D_alpha

                    r1, r2 = np.random.rand(), np.random.rand()
                    A2 = 2 * a * r1 - a
                    C2 = 2 * r2
                    D_beta = abs(C2 * beta[j] - self.positions[i][j])
                    X2 = beta[j] - A2 * D_beta

                    r1, r2 = np.random.rand(), np.random.rand()
                    A3 = 2 * a * r1 - a
                    C3 = 2 * r2
                    D_delta = abs(C3 * delta[j] - self.positions[i][j])
                    X3 = delta[j] - A3 * D_delta

                    self.positions[i][j] = np.clip((X1 + X2 + X3) / 3, self.lower_bound, self.upper_bound)

            print(f"Iter {iter}: Melhor Score = {alpha_score:.2f} Média do alfa, beta e delta = {np.mean([alpha_score, beta_score, delta_score]):.2f}")

        return alpha, alpha_score
