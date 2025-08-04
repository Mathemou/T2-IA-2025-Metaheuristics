from game.core import SurvivalGame, GameConfig
from neural_agent import EnhancedNeuralAgent, SimpleNeuralAgent
import numpy as np
from gwo import GreyWolfOptimizer
import time

def enhanced_fitness_function(weights):
    """Função de fitness para o agente neural melhorado"""
    config = GameConfig()
    
    # Criar agente neural melhorado
    agent = EnhancedNeuralAgent(config, weights)
    scores = []

    # Testar o agente em múltiplos jogos
    for _ in range(3):
        game = SurvivalGame(config, render=False)
        step_count = 0
        max_steps = 5000000  # Limite mais razoável
        
        while not game.all_players_dead() and step_count < max_steps:
            state = game.get_state(0, include_internals=True)
            action = agent.predict(state)
            game.update([action])
            step_count += 1
            
        scores.append(game.players[0].score)

    return np.mean(scores)

def test_enhanced_agent(weights, num_tests=10, render=False):
    """Testa o agente neural melhorado"""
    config = GameConfig()
    agent = EnhancedNeuralAgent(config, weights)
    scores = []
    wall_deaths = 0
    obstacle_deaths = 0
    
    print(f"\n--- Testando Agente Neural Melhorado ({num_tests} jogos) ---")
    
    for test in range(num_tests):
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
            if not game.players[0].alive and test < 5:  # Só verificar nos primeiros 5 testes
                if player.y <= config.player_radius or player.y >= config.screen_height - config.player_radius:
                    wall_deaths += 1
                    print(f"  Morte por parede no teste {test+1}")
                else:
                    obstacle_deaths += 1
                    print(f"  Morte por obstáculo no teste {test+1}")
            
            if render and game.players[0].alive:
                game.render_frame()
                
        scores.append(game.players[0].score)
        
        if not render:
            print(f"Teste {test + 1}/{num_tests}: Score = {game.players[0].score:.2f}")
    
    print(f"\nResultados:")
    print(f"Score Médio: {np.mean(scores):.2f}")
    print(f"Score Máximo: {np.max(scores):.2f}")
    print(f"Score Mínimo: {np.min(scores):.2f}")
    print(f"Desvio Padrão: {np.std(scores):.2f}")
    
    if wall_deaths + obstacle_deaths > 0:
        print(f"\nCausas de Morte (primeiros 5 testes):")
        print(f"  Mortes por parede: {wall_deaths}")
        print(f"  Mortes por obstáculo: {obstacle_deaths}")
    
    return scores

def train_enhanced_gwo():
    """Treina o agente neural melhorado usando GWO"""
    print("\n--- Iniciando Treinamento Neural MELHORADO com GWO ---")
    
    # Criar um agente temporário para descobrir o número de pesos necessários
    config = GameConfig()
    temp_agent = EnhancedNeuralAgent(config)
    dim = temp_agent.get_weights_count()
    
    print(f"Dimensões da rede neural melhorada: {dim} pesos")
    print(f"Arquitetura: {temp_agent.input_size} -> {temp_agent.hidden_size} -> {temp_agent.output_size}")
    print(f"Features adicionais: 4 (distância/perigo das paredes)")
    
    # Configurar GWO
    gwo = GreyWolfOptimizer(
        fitness_function=enhanced_fitness_function,
        dim=dim,
        n_wolves=50,  # População menor para eficiência
        max_iter=2200,   # Iterações suficientes
        bounds=(-2.5, 2.5)  # Limites mais conservadores
    )
    
    start_time = time.time()
    best_weights, best_score = gwo.optimize()
    end_time = time.time()
    
    print(f"\n--- Treinamento Concluído ---")
    print(f"Tempo total: {end_time - start_time:.2f} segundos")
    print(f"Melhor Score: {best_score:.2f}")
    
    # Salvar os melhores pesos
    np.save("best_enhanced_weights.npy", best_weights)
    print("Melhores pesos salvos em 'best_enhanced_weights.npy'")
    
    # Testar o agente treinado
    test_enhanced_agent(best_weights, num_tests=20, render=False)
    
    return best_weights, best_score

def compare_agents():
    """Compara o agente simples vs melhorado"""
    print("\n" + "="*60)
    print("COMPARAÇÃO: AGENTE SIMPLES vs MELHORADO")
    print("="*60)
    
    config = GameConfig()
    
    # Testar agente simples com pesos aleatórios
    simple_agent = SimpleNeuralAgent(config)
    simple_scores = []
    
    print("\n--- Testando Agente Simples (pesos aleatórios) ---")
    for i in range(10):
        game = SurvivalGame(config, render=False)
        step_count = 0
        
        while not game.all_players_dead() and step_count < 2000:
            state = game.get_state(0, include_internals=True)
            action = simple_agent.predict(state)
            game.update([action])
            step_count += 1
        
        simple_scores.append(game.players[0].score)
    
    # Testar agente melhorado com pesos aleatórios
    enhanced_agent = EnhancedNeuralAgent(config)
    enhanced_scores = []
    
    print("\n--- Testando Agente Melhorado (pesos aleatórios) ---")
    for i in range(10):
        game = SurvivalGame(config, render=False)
        step_count = 0
        
        while not game.all_players_dead() and step_count < 2000:
            state = game.get_state(0, include_internals=True)
            action = enhanced_agent.predict(state)
            game.update([action])
            step_count += 1
        
        enhanced_scores.append(game.players[0].score)
    
    print(f"\nResultados (pesos aleatórios):")
    print(f"Agente Simples    - Médio: {np.mean(simple_scores):.2f}, Max: {np.max(simple_scores):.2f}")
    print(f"Agente Melhorado  - Médio: {np.mean(enhanced_scores):.2f}, Max: {np.max(enhanced_scores):.2f}")
    
    # Testar agentes treinados se disponíveis
    try:
        simple_weights = np.load("best_neural_weights.npy")
        simple_trained = SimpleNeuralAgent(config, simple_weights)
        simple_trained_scores = []
        
        for i in range(10):
            game = SurvivalGame(config, render=False)
            step_count = 0
            
            while not game.all_players_dead() and step_count < 5000:
                state = game.get_state(0, include_internals=True)
                action = simple_trained.predict(state)
                game.update([action])
                step_count += 1
            
            simple_trained_scores.append(game.players[0].score)
        
        print(f"Agente Simples (treinado) - Médio: {np.mean(simple_trained_scores):.2f}, Max: {np.max(simple_trained_scores):.2f}")
        
    except FileNotFoundError:
        print("Agente simples treinado não encontrado")
    
    try:
        enhanced_weights = np.load("best_enhanced_weights.npy")
        enhanced_trained = EnhancedNeuralAgent(config, enhanced_weights)
        enhanced_trained_scores = []
        
        for i in range(10):
            game = SurvivalGame(config, render=False)
            step_count = 0
            
            while not game.all_players_dead() and step_count < 5000:
                state = game.get_state(0, include_internals=True)
                action = enhanced_trained.predict(state)
                game.update([action])
                step_count += 1
            
            enhanced_trained_scores.append(game.players[0].score)
        
        print(f"Agente Melhorado (treinado) - Médio: {np.mean(enhanced_trained_scores):.2f}, Max: {np.max(enhanced_trained_scores):.2f}")
        
    except FileNotFoundError:
        print("Agente melhorado treinado não encontrado")

def demo_enhanced_agent():
    """Demonstra o agente melhorado treinado"""
    try:
        best_weights = np.load("best_enhanced_weights.npy")
        print("Carregando pesos do agente melhorado...")
        
        print("\n--- Demonstração Visual do Agente Melhorado ---")
        test_enhanced_agent(best_weights, num_tests=3, render=True)
        
    except FileNotFoundError:
        print("Arquivo de pesos do agente melhorado não encontrado.")
        print("Execute o treinamento primeiro com: python enhanced_trainer.py")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            demo_enhanced_agent()
        elif sys.argv[1] == "test":
            try:
                weights = np.load("best_enhanced_weights.npy")
                test_enhanced_agent(weights, num_tests=30, render=False)
            except FileNotFoundError:
                print("Arquivo de pesos não encontrado.")
        elif sys.argv[1] == "compare":
            compare_agents()
        else:
            print("Uso: python enhanced_trainer.py [demo|test|compare]")
    else:
        # Treinamento padrão
        train_enhanced_gwo()
