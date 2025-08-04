"""
Script para comparar diferentes tipos de agentes:
1. RuleBasedAgent com parâmetros padrão
2. RuleBasedAgent otimizado com GWO  
3. Neural Agent otimizado com GWO
"""

import numpy as np
from game.core import SurvivalGame, GameConfig
from game.agents import RuleBasedAgent
from neural_agent import SimpleNeuralAgent
import time

def test_agent_performance(agent, agent_name, num_tests=20, render=False):
    """Testa a performance de um agente"""
    config = GameConfig()
    scores = []
    
    print(f"\n--- Testando {agent_name} ({num_tests} jogos) ---")
    
    for test in range(num_tests):
        game = SurvivalGame(config, render=render)
        step_count = 0
        max_steps = 2000
        
        while not game.all_players_dead() and step_count < max_steps:
            state = game.get_state(0, include_internals=True)
            action = agent.predict(state)
            game.update([action])
            step_count += 1
            
            if render:
                game.render_frame()
                
        scores.append(game.players[0].score)
        
        if not render and test % 5 == 0:
            print(f"Teste {test + 1}/{num_tests}: Score = {game.players[0].score}")
    
    print(f"\n{agent_name} - Resultados:")
    print(f"Score Médio: {np.mean(scores):.2f}")
    print(f"Score Máximo: {np.max(scores):.2f}")
    print(f"Score Mínimo: {np.min(scores):.2f}")
    print(f"Desvio Padrão: {np.std(scores):.2f}")
    
    return scores

def compare_agents():
    """Compara todos os tipos de agentes"""
    config = GameConfig()
    
    print("="*60)
    print("COMPARAÇÃO DE AGENTES - SURVIVAL GAME")
    print("="*60)
    
    # 1. Agente baseado em regras (padrão)
    rule_agent_default = RuleBasedAgent(config)
    scores_rule_default = test_agent_performance(
        rule_agent_default, 
        "RuleBasedAgent (Padrão)", 
        num_tests=20
    )
    
    # 2. Agente baseado em regras (otimizado)
    try:
        optimized_weights = np.load("best_weights.npy")
        rule_agent_optimized = RuleBasedAgent(
            config,
            danger_threshold=optimized_weights[0],
            lookahead_cells=optimized_weights[1],
            diff_to_center_to_move=optimized_weights[2]
        )
        scores_rule_optimized = test_agent_performance(
            rule_agent_optimized,
            "RuleBasedAgent (Otimizado GWO)",
            num_tests=20
        )
    except FileNotFoundError:
        print("\nArquivo 'best_weights.npy' não encontrado.")
        print("Execute o treinamento do RuleBasedAgent primeiro.")
        scores_rule_optimized = None
    
    # 3. Agente neural (otimizado)
    try:
        neural_weights = np.load("best_neural_weights.npy")
        neural_agent = SimpleNeuralAgent(config, neural_weights)
        scores_neural = test_agent_performance(
            neural_agent,
            "Neural Agent (Otimizado GWO)",
            num_tests=20
        )
    except FileNotFoundError:
        print("\nArquivo 'best_neural_weights.npy' não encontrado.")
        print("Execute o treinamento do Neural Agent primeiro.")
        scores_neural = None
    
    # Resumo comparativo
    print("\n" + "="*60)
    print("RESUMO COMPARATIVO")
    print("="*60)
    
    results = [
        ("RuleBasedAgent (Padrão)", scores_rule_default),
        ("RuleBasedAgent (Otimizado)", scores_rule_optimized),
        ("Neural Agent (Otimizado)", scores_neural)
    ]
    
    for name, scores in results:
        if scores is not None:
            print(f"{name:25} | Médio: {np.mean(scores):6.2f} | "
                  f"Max: {np.max(scores):6.2f} | Std: {np.std(scores):5.2f}")
        else:
            print(f"{name:25} | Não testado (arquivo não encontrado)")
    
    # Análise de melhoria
    if scores_rule_optimized is not None:
        improvement_rule = ((np.mean(scores_rule_optimized) - np.mean(scores_rule_default)) / 
                           np.mean(scores_rule_default) * 100)
        print(f"\nMelhoria RuleBasedAgent Otimizado vs Padrão: {improvement_rule:+.1f}%")
    
    if scores_neural is not None:
        improvement_neural = ((np.mean(scores_neural) - np.mean(scores_rule_default)) / 
                             np.mean(scores_rule_default) * 100)
        print(f"Melhoria Neural Agent vs RuleBasedAgent Padrão: {improvement_neural:+.1f}%")
        
        if scores_rule_optimized is not None:
            improvement_neural_vs_rule = ((np.mean(scores_neural) - np.mean(scores_rule_optimized)) / 
                                         np.mean(scores_rule_optimized) * 100)
            print(f"Melhoria Neural Agent vs RuleBasedAgent Otimizado: {improvement_neural_vs_rule:+.1f}%")

def demo_best_agent():
    """Demonstra o melhor agente encontrado"""
    config = GameConfig()
    
    # Tentar encontrar o melhor agente
    best_agent = None
    best_name = ""
    
    try:
        neural_weights = np.load("best_neural_weights.npy")
        best_agent = SimpleNeuralAgent(config, neural_weights)
        best_name = "Neural Agent"
    except FileNotFoundError:
        try:
            rule_weights = np.load("best_weights.npy")
            best_agent = RuleBasedAgent(
                config,
                danger_threshold=rule_weights[0],
                lookahead_cells=rule_weights[1],
                diff_to_center_to_move=rule_weights[2]
            )
            best_name = "RuleBasedAgent Otimizado"
        except FileNotFoundError:
            best_agent = RuleBasedAgent(config)
            best_name = "RuleBasedAgent Padrão"
    
    print(f"\n--- Demonstração: {best_name} ---")
    test_agent_performance(best_agent, best_name, num_tests=3, render=True)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demo_best_agent()
    else:
        compare_agents()
