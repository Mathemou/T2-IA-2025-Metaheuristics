from game.core import SurvivalGame, GameConfig
from neural_agent import SimpleNeuralAgent, create_neural_agent
import numpy as np
from test_trained_agent import test_agent
from gwo import GreyWolfOptimizer
import time

def neural_fitness_function(weights):
    """Função de fitness para o agente neural"""
    config = GameConfig()
    
    # Criar agente neural com os pesos fornecidos
    agent = SimpleNeuralAgent(config, weights)
    scores = []

    # Testar o agente em múltiplos jogos
    for _ in range(3):
        game = SurvivalGame(config, render=False)
        step_count = 0
        max_steps = 100000  # Apesar que eu duvido muito, evita jogos infinitos
        
        while not game.all_players_dead() and step_count < max_steps:
            state = game.get_state(0, include_internals=True)
            action = agent.predict(state)
            game.update([action])
            step_count += 1
            
        scores.append(game.players[0].score)

    return np.mean(scores)

def test_neural_agent(weights, num_tests=10, render=False):
    """Testa o agente neural treinado"""
    config = GameConfig()
    agent = SimpleNeuralAgent(config, weights)
    scores = []
    
    print(f"\n--- Testando Agente Neural ({num_tests} jogos) ---")
    
    for test in range(num_tests):
        game = SurvivalGame(config, render=render)
        step_count = 0
        max_steps = 100000
        
        while not game.all_players_dead() and step_count < max_steps:
            state = game.get_state(0, include_internals=True)
            action = agent.predict(state)
            game.update([action])
            step_count += 1
            
            if render:
                game.render_frame()
                
        scores.append(game.players[0].score)
        
        if not render:
            print(f"Teste {test + 1}/{num_tests}: Score = {game.players[0].score}")
    
    print(f"\nResultados:")
    print(f"Score Médio: {np.mean(scores):.2f}")
    print(f"Score Máximo: {np.max(scores):.2f}")
    print(f"Score Mínimo: {np.min(scores):.2f}")
    print(f"Desvio Padrão: {np.std(scores):.2f}")
    
    return scores

def train_neural_gwo():
    """Treina o agente neural usando GWO"""
    print("\n--- Iniciando Treinamento Neural com GWO ---")
    
    # Criar um agente temporário para descobrir o número de pesos necessários
    config = GameConfig()
    temp_agent = SimpleNeuralAgent(config)
    dim = temp_agent.get_weights_count()
    
    print(f"Dimensões da rede neural: {dim} pesos")
    print(f"Arquitetura: {temp_agent.input_size} -> {temp_agent.hidden_size} -> {temp_agent.output_size}")
    
    # Configurar GWO com limites adequados para redes neurais
    gwo = GreyWolfOptimizer(
        fitness_function=neural_fitness_function,
        dim=dim,
        n_wolves=30,  # Menos lobos devido ao maior espaço de busca
        max_iter=500,   # Mais iterações
        bounds=(-3, 3)  # Limites adequados para pesos neurais
    )
    
    start_time = time.time()
    best_weights, best_score = gwo.optimize()
    end_time = time.time()
    
    print(f"\n--- Treinamento Concluído ---")
    print(f"Tempo total: {end_time - start_time:.2f} segundos")
    print(f"Melhor Score: {best_score:.2f}")
    
    # Salvar os melhores pesos
    np.save("best_neural_weights.npy", best_weights)
    print("Melhores pesos salvos em 'best_neural_weights.npy'")
    
    # Testar o agente treinado
    test_neural_agent(best_weights, num_tests=20, render=False)
    
    return best_weights, best_score

def demo_neural_agent():
    """Demonstra o agente neural treinado"""
    try:
        # Tentar carregar pesos salvos
        best_weights = np.load("best_neural_weights.npy")
        print("Carregando pesos salvos...")
        
        # Demonstrar com visualização
        print("\n--- Demonstração Visual ---")
        test_neural_agent(best_weights, num_tests=3, render=True)
        
    except FileNotFoundError:
        print("Arquivo de pesos não encontrado. Execute o treinamento primeiro.")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            demo_neural_agent()
        elif sys.argv[1] == "test":
            try:
                weights = np.load("best_neural_weights.npy")
                test_neural_agent(weights, num_tests=30, render=False)
            except FileNotFoundError:
                print("Arquivo de pesos não encontrado.")
        else:
            print("Uso: python gwo_neural_trainer.py [demo|test]")
    else:
        # Treinamento padrão
        train_neural_gwo()
