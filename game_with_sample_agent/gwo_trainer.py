from game.core import SurvivalGame, GameConfig
from game.agents import EnhancedNeuralAgent
import numpy as np
from gwo import GreyWolfOptimizer
import time

def fitness_function(weights):
    """Função de fitness para o agente neural que vê paredes"""
    config = GameConfig()
    
    # Criar agente neural melhorado
    agent = EnhancedNeuralAgent(config, weights)
    scores = []

    # Testar o agente em múltiplos jogos
    for _ in range(3):
        game = SurvivalGame(config, render=False)
        step_count = 0
        max_steps = 500000  # Limite mais razoável
        
        while not game.all_players_dead() and step_count < max_steps:
            state = game.get_state(0, include_internals=True)
            action = agent.predict(state)
            game.update([action])
            step_count += 1
            
        scores.append(game.players[0].score)

    return np.mean(scores)

def test_agent(weights, num_tests=10, render=False):
    config = GameConfig()
    agent = EnhancedNeuralAgent(config, weights)
    scores = []
    wall_deaths = 0
    obstacle_deaths = 0
    
    print(f"\n--- Testando Agente Neural Melhorado ({num_tests} jogos) ---")
    
    for test in trange(num_tests, desc="Testando agente"):
        game = SurvivalGame(config, render=render)
        
        step_count = 0
        max_steps = 100000000
        
        while not game.all_players_dead() and step_count < max_steps:
            state = game.get_state(0, include_internals=True)
            action = agent.predict(state)
            
            # Verificar se vai bater na parede
            player = game.players[0]
            if player.alive:
                old_y = player.y
            
            game.update([action])
            step_count += 1
            
            # Verificar causa da morte
            """ if not game.players[0].alive:  # Só verificar nos primeiros 5 testes
                if player.y <= config.player_radius or player.y >= config.screen_height - config.player_radius:
                    wall_deaths += 1
                    print(f"  Morte por parede no teste {test+1}")
                else:
                    obstacle_deaths += 1
                    print(f"  Morte por obstáculo no teste {test+1}") """
            
            if render and game.players[0].alive:
                game.render_frame()
                
        scores.append(game.players[0].score)
        
        """ if not render:
            print(f"Teste {test + 1}/{num_tests}: Score = {game.players[0].score:.2f}") """
    bonela_neural_agent_genetic_result = [38.32, 54.53, 61.16, 27.55, 16.08, 26.00, 25.33, 18.30,39.76, 48.17, 44.77, 47.54, 75.43, 23.68, 16.83, 15.81, 67.17, 53.54, 33.59,49.24, 52.65, 16.35, 44.05, 56.59, 63.23, 43.96, 43.82, 19.19, 28.36, 18.65]
    print(f"\nResultados:")
    print(f"Score Médio: {np.mean(scores):.2f}")
    print(f"Score Máximo: {np.max(scores):.2f}")
    print(f"Score Mínimo: {np.min(scores):.2f}")
    print(f"Desvio Padrão: {np.std(scores):.2f}")
    if np.mean(scores) > np.mean(bonela_neural_agent_genetic_result):
        print(f"O agente neural com o lobo foi {((np.mean(scores) - np.mean(bonela_neural_agent_genetic_result)) / np.mean(bonela_neural_agent_genetic_result)) * 100:.2f}% melhor que o agente do Bonela.")
    else:
        print(f"O agente neural com o lobo foi {((np.mean(bonela_neural_agent_genetic_result) - np.mean(scores)) / np.mean(bonela_neural_agent_genetic_result)) * 100:.2f}% pior que o agente do Bonela.")
    
    if wall_deaths + obstacle_deaths > 0:
        print(f"\nCausas de Morte (primeiros 5 testes):")
        print(f"  Mortes por parede: {wall_deaths}")
        print(f"  Mortes por obstáculo: {obstacle_deaths}")
    
    return scores

def train_gwo():
    """Treina o agente neural melhorado usando GWO"""
    print("\n--- Iniciando Treinamento Neural MELHORADO com GWO ---")
    
    # Criar um agente temporário para descobrir o número de pesos necessários
    config = GameConfig()
    temp_agent = EnhancedNeuralAgent(config)
    dim = temp_agent.get_weights_count()
    
    print(f"Dimensões da rede neural melhorada: {dim} pesos")
    print(f"Arquitetura: {temp_agent.input_size} -> {temp_agent.hidden1_size} -> {temp_agent.hidden2_size} -> {temp_agent.output_size}")
    print(f"Features adicionais: 4 (distância/perigo das paredes)")
    
    # Configurar GWO
    gwo = GreyWolfOptimizer(
        fitness_function=fitness_function,
        dim=dim,
        n_wolves=100,  
        max_iter=1000,   
        bounds=(-2.5, 2.5)  # Limites mais conservadores
    )
    
    start_time = time.time()
    best_weights, best_score = gwo.optimize()
    end_time = time.time()
    
    print(f"\n--- Treinamento Concluído ---")
    print(f"Tempo total: {end_time - start_time:.2f} segundos")
    print(f"Melhor Score: {best_score:.2f}")
    
    # Salvar os melhores pesos
    np.save("best_weights.npy", best_weights)
    print("Melhores pesos salvos em 'best_weights.npy'")
    
    # Testar o agente treinado
    #test_agent(best_weights, num_tests=30, render=False)
    
    return best_weights, best_score


def demo_agent():
    """Demonstra o agente melhorado treinado"""
    try:
        best_weights = np.load("best_weights.npy")
        print("Carregando pesos do agente melhorado...")
        
        print("\n--- Demonstração Visual do Agente Melhorado ---")
        test_agent(best_weights, num_tests=1, render=True)
        
    except FileNotFoundError:
        print("Arquivo de pesos do agente melhorado não encontrado.")
        print("Execute o treinamento primeiro com: python gwo_trainer.py")

def compare_bonela(weights, num_hyper_tests=10):
    config = GameConfig()
    agent = EnhancedNeuralAgent(config, weights)
    scores = []
    # resultados de 30 iterações do agente do Bonela otimizado com algoritmo genético
    bonela_neural_agent_genetic_result = [38.32, 54.53, 61.16, 27.55, 16.08, 26.00, 25.33, 18.30,39.76, 48.17, 44.77, 47.54, 75.43, 23.68, 16.83, 15.81, 67.17, 53.54, 33.59,49.24, 52.65, 16.35, 44.05, 56.59, 63.23, 43.96, 43.82, 19.19, 28.36, 18.65]
    quantas_vezes_superou_bonela = 0
    for _ in range(num_hyper_tests):
        game = SurvivalGame(GameConfig(), render=False)
        agent = EnhancedNeuralAgent(config, weights)
        scores = []
        for _ in range(30):
            step_count = 0
            max_steps = 100000000
            
            while not game.all_players_dead() and step_count < max_steps:
                state = game.get_state(0, include_internals=True)
                action = agent.predict(state)
                game.update([action])
                step_count += 1
                
            scores.append(game.players[0].score)
        if np.mean(scores) > np.mean(bonela_neural_agent_genetic_result):
            quantas_vezes_superou_bonela += 1
    print(f"\nO agente neural melhorado superou o agente do Bonela em {quantas_vezes_superou_bonela / num_hyper_tests * 100:.2f}% das vezes.")
if __name__ == "__main__":
    import sys
    from tqdm import trange
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            demo_agent()
        elif sys.argv[1] == "test":
            if len(sys.argv) > 2:
                num_tests = int(sys.argv[2])
            else:
                num_tests = 30
            try:
                weights = np.load("best_weights.npy")
                test_agent(weights, num_tests=num_tests, render=False)
            except FileNotFoundError:
                print("Arquivo de pesos não encontrado.")
        elif sys.argv[1] == "compare":
            weights = np.load("best_weights.npy")
            compare_bonela(weights)
        else:
            print("Uso: python gwo_trainer.py [demo|test]")
    else:
        # Treinamento padrão
        train_gwo()
