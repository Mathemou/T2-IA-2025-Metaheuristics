"""
Script para analisar e demonstrar as melhorias do agente neural melhorado
"""

import numpy as np
from game.core import SurvivalGame, GameConfig
from neural_agent import SimpleNeuralAgent, EnhancedNeuralAgent
from game.agents import RuleBasedAgent

def analyze_wall_awareness():
    """Analisa se o agente melhorado tem consciência das paredes"""
    print("=== ANÁLISE DA CONSCIÊNCIA DAS PAREDES ===\n")
    
    config = GameConfig()
    
    # Criar agentes
    simple_agent = SimpleNeuralAgent(config)
    enhanced_agent = EnhancedNeuralAgent(config)
    
    # Simular situações perto das paredes
    situations = [
        ("Perto da parede superior", 30),
        ("No centro", config.screen_height // 2),
        ("Perto da parede inferior", config.screen_height - 30)
    ]
    
    for situation_name, y_position in situations:
        print(f"\n{situation_name} (Y = {y_position}):")
        
        # Criar estado simulado
        grid = np.zeros(25)  # Grid vazio (sem obstáculos)
        y_norm = y_position / config.screen_height
        speed = 0.5
        
        # Estado para agente simples
        simple_state = np.concatenate([grid, [y_norm, speed]])
        
        # Estado para agente melhorado (tem as mesmas informações + wall features)
        enhanced_state = np.concatenate([grid, [y_norm, speed]])
        
        # Testar ações múltiplas vezes
        simple_actions = []
        enhanced_actions = []
        
        for _ in range(10):
            simple_action = simple_agent.predict(simple_state)
            enhanced_action = enhanced_agent.predict(enhanced_state)
            simple_actions.append(simple_action)
            enhanced_actions.append(enhanced_action)
        
        # Análise das ações
        simple_action_counts = np.bincount(simple_actions, minlength=3)
        enhanced_action_counts = np.bincount(enhanced_actions, minlength=3)
        
        action_names = ["Parar", "Subir", "Descer"]
        
        print(f"  Agente Simples:")
        for i, count in enumerate(simple_action_counts):
            print(f"    {action_names[i]}: {count}/10 vezes")
        
        print(f"  Agente Melhorado:")
        for i, count in enumerate(enhanced_action_counts):
            print(f"    {action_names[i]}: {count}/10 vezes")
        
        # Análise da situação
        if y_position < 50:  # Perto da parede superior
            if enhanced_action_counts[2] > enhanced_action_counts[1]:  # Mais "descer" que "subir"
                print(f"  ✅ Agente melhorado prefere DESCER (correto perto da parede superior)")
            else:
                print(f"  ⚠️  Agente melhorado não mostra preferência clara")
        
        elif y_position > config.screen_height - 50:  # Perto da parede inferior
            if enhanced_action_counts[1] > enhanced_action_counts[2]:  # Mais "subir" que "descer"
                print(f"  ✅ Agente melhorado prefere SUBIR (correto perto da parede inferior)")
            else:
                print(f"  ⚠️  Agente melhorado não mostra preferência clara")

def test_wall_collision_rates():
    """Testa as taxas de colisão com paredes"""
    print("\n=== TESTE DE COLISÕES COM PAREDES ===\n")
    
    config = GameConfig()
    
    agents = [
        ("RuleBasedAgent", RuleBasedAgent(config)),
        ("SimpleNeuralAgent", SimpleNeuralAgent(config)),
        ("EnhancedNeuralAgent", EnhancedNeuralAgent(config))
    ]
    
    for agent_name, agent in agents:
        wall_deaths = 0
        obstacle_deaths = 0
        total_games = 20
        
        print(f"Testando {agent_name}...")
        
        for game_num in range(total_games):
            game = SurvivalGame(config, render=False)
            step_count = 0
            max_steps = 1000
            
            while not game.all_players_dead() and step_count < max_steps:
                state = game.get_state(0, include_internals=True)
                action = agent.predict(state)
                
                # Salvar posição antes da ação
                old_y = game.players[0].y
                
                game.update([action])
                step_count += 1
                
                # Verificar causa da morte
                if not game.players[0].alive:
                    if (game.players[0].y <= config.player_radius or 
                        game.players[0].y >= config.screen_height - config.player_radius):
                        wall_deaths += 1
                    else:
                        obstacle_deaths += 1
                    break
        
        wall_rate = (wall_deaths / total_games) * 100
        obstacle_rate = (obstacle_deaths / total_games) * 100
        
        print(f"  Mortes por parede: {wall_deaths}/{total_games} ({wall_rate:.1f}%)")
        print(f"  Mortes por obstáculo: {obstacle_deaths}/{total_games} ({obstacle_rate:.1f}%)")
        print(f"  Score médio: {np.mean([test_single_game(agent, config) for _ in range(5)]):.2f}\n")

def test_single_game(agent, config):
    """Testa um único jogo e retorna o score"""
    game = SurvivalGame(config, render=False)
    step_count = 0
    max_steps = 2000
    
    while not game.all_players_dead() and step_count < max_steps:
        state = game.get_state(0, include_internals=True)
        action = agent.predict(state)
        game.update([action])
        step_count += 1
    
    return game.players[0].score

def show_feature_extraction():
    """Mostra como o agente melhorado extrai features das paredes"""
    print("=== EXTRAÇÃO DE FEATURES DAS PAREDES ===\n")
    
    config = GameConfig()
    agent = EnhancedNeuralAgent(config)
    
    test_positions = [20, 100, 200, 300, 380]  # Diferentes posições Y
    
    print("Posição Y | Dist.Top | Dist.Bottom | Perigo Top | Perigo Bottom")
    print("-" * 65)
    
    for y_pos in test_positions:
        # Criar estado simulado
        grid = np.zeros(25)
        y_norm = y_pos / config.screen_height
        speed = 0.5
        state = np.concatenate([grid, [y_norm, speed]])
        
        # Extrair features das paredes
        wall_features = agent._extract_wall_features(state)
        
        print(f"{y_pos:8d} | {wall_features[0]:8.3f} | {wall_features[1]:9.3f} | "
              f"{wall_features[2]:9.1f} | {wall_features[3]:11.1f}")
    
    print("\nLegenda:")
    print("- Dist.Top/Bottom: 0 = na parede, 1 = no centro")
    print("- Perigo: 1 = muito perto da parede (< 50 pixels)")

def architecture_comparison():
    """Compara as arquiteturas dos agentes"""
    print("\n=== COMPARAÇÃO DAS ARQUITETURAS ===\n")
    
    config = GameConfig()
    
    simple = SimpleNeuralAgent(config)
    enhanced = EnhancedNeuralAgent(config)
    
    print("AGENTE SIMPLES:")
    print(f"  Entradas: {simple.input_size}")
    print(f"    - Grid de obstáculos: 25")
    print(f"    - Posição Y: 1") 
    print(f"  Neurônios ocultos: {simple.hidden_size}")
    print(f"  Saídas: {simple.output_size}")
    print(f"  Total de pesos: {simple.get_weights_count()}")
    
    print("\nAGENTE MELHORADO:")
    print(f"  Entradas: {enhanced.input_size}")
    print(f"    - Grid de obstáculos: 25")
    print(f"    - Posição Y: 1")
    print(f"    - Velocidade do jogo: 1")
    print(f"    - Distância até parede superior: 1")
    print(f"    - Distância até parede inferior: 1") 
    print(f"    - Flag perigo parede superior: 1")
    print(f"    - Flag perigo parede inferior: 1")
    print(f"  Neurônios ocultos: {enhanced.hidden_size}")
    print(f"  Saídas: {enhanced.output_size}")
    print(f"  Total de pesos: {enhanced.get_weights_count()}")
    
    improvement = ((enhanced.get_weights_count() - simple.get_weights_count()) / 
                   simple.get_weights_count() * 100)
    print(f"\nAumento na complexidade: +{improvement:.1f}% pesos")

def main():
    """Executa todas as análises"""
    print("ANÁLISE COMPLETA: AGENTE NEURAL MELHORADO")
    print("=" * 80)
    
    architecture_comparison()
    show_feature_extraction()
    analyze_wall_awareness()
    test_wall_collision_rates()
    
    print("\n" + "=" * 80)
    print("RESUMO DAS MELHORIAS")
    print("=" * 80)
    print("✅ Agente melhorado tem 4 features adicionais sobre paredes")
    print("✅ Calcula distâncias normalizadas até as paredes")
    print("✅ Detecta situações de perigo (muito perto das paredes)")
    print("✅ Arquitetura maior permite aprender comportamentos mais complexos")
    print("✅ Deve reduzir significativamente mortes por colisão com paredes")

if __name__ == "__main__":
    main()
