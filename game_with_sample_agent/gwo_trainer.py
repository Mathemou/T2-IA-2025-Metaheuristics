from game.core import SurvivalGame, GameConfig
from game.agents import EnhancedNeuralAgent
import numpy as np
from gwo import GreyWolfOptimizer
import time
import matplotlib.pyplot as plt
import sys
from tqdm import trange

def fitness_function(weights):
    config = GameConfig()
    
    agent = EnhancedNeuralAgent(config, weights)
    scores = []

    for _ in range(3):
        game = SurvivalGame(config, render=False)
        step_count = 0
        max_steps = 500000  
        
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
    
    for _ in trange(num_tests, desc="Testando agente"):
        game = SurvivalGame(config, render=render)
        
        step_count = 0
        max_steps = 100000000
        
        while not game.all_players_dead() and step_count < max_steps:
            state = game.get_state(0, include_internals=True)
            action = agent.predict(state)
            
            player = game.players[0]
            if player.alive:
                old_y = player.y
            
            game.update([action])
            step_count += 1
            
            if render and game.players[0].alive:
                game.render_frame()
                
        scores.append(game.players[0].score)
       
    bonela_rule_based_genetic_result = [12.69, 16.65, 6.97, 2.79, 15.94, 10.22, 21.90, 4.35, 6.22, 9.95, 19.94, 20.56, 15.74, 17.68, 7.16, 15.68, 2.37, 15.43, 15.13, 22.50, 25.82, 15.85, 17.02, 16.74, 14.69, 11.73, 13.80, 15.13, 12.35, 16.19]
    bonela_neural_agent_genetic_result = [38.32, 54.53, 61.16, 27.55, 16.08, 26.00, 25.33, 18.30, 39.76, 48.17, 44.77, 47.54, 75.43, 23.68, 16.83, 15.81, 67.17, 53.54, 33.59, 49.24, 52.65, 16.35, 44.05, 56.59, 63.23, 43.96, 43.82, 19.19, 28.36, 18.65]
    bonela_human_result = [27.34, 17.63, 39.33, 17.44, 1.16, 24.04, 29.21, 18.92, 25.71, 20.05, 31.88, 15.39, 22.50, 19.27, 26.33, 23.67, 16.82, 28.45, 12.59, 33.01, 21.74, 14.23, 27.90, 24.80, 11.35, 30.12, 17.08, 22.96, 9.41, 35.22]
    
    # Calcular estatísticas dos resultados de referência
    bonela_rule_mean = np.mean(bonela_rule_based_genetic_result)
    bonela_neural_mean = np.mean(bonela_neural_agent_genetic_result)
    bonela_human_mean = np.mean(bonela_human_result)
    
    print(f"\nResultados:")
    print(f"Score Médio: {np.mean(scores):.2f}")
    print(f"Score Máximo: {np.max(scores):.2f}")
    print(f"Score Mínimo: {np.min(scores):.2f}")
    print(f"Desvio Padrão: {np.std(scores):.2f}")
    
    print(f"\n--- Comparação com Benchmarks ---")
    
    # Comparação com agente neural do Bonela
    if np.mean(scores) > bonela_neural_mean:
        print(f"🟢 Agente Neural GWO foi {((np.mean(scores) - bonela_neural_mean) / bonela_neural_mean) * 100:.2f}% melhor que o Agente Neural Genético do Bonela (média: {bonela_neural_mean:.2f})")
    else:
        print(f"🔴 Agente Neural GWO foi {((bonela_neural_mean - np.mean(scores)) / bonela_neural_mean) * 100:.2f}% pior que o Agente Neural Genético do Bonela (média: {bonela_neural_mean:.2f})")
    
    # Comparação com agente baseado em regras do Bonela
    if np.mean(scores) > bonela_rule_mean:
        print(f"🟢 Agente Neural GWO foi {((np.mean(scores) - bonela_rule_mean) / bonela_rule_mean) * 100:.2f}% melhor que o Agente Genético Baseado em Regras do Bonela (média: {bonela_rule_mean:.2f})")
    else:
        print(f"🔴 Agente Neural GWO foi {((bonela_rule_mean - np.mean(scores)) / bonela_rule_mean) * 100:.2f}% pior que o Agente Genético Baseado em Regras do Bonela (média: {bonela_rule_mean:.2f})")
    
    # Comparação com humano do Bonela
    if np.mean(scores) > bonela_human_mean:
        print(f"🟢 Agente Neural GWO foi {((np.mean(scores) - bonela_human_mean) / bonela_human_mean) * 100:.2f}% melhor que o Humano do Bonela (média: {bonela_human_mean:.2f})")
    else:
        print(f"🔴 Agente Neural GWO foi {((bonela_human_mean - np.mean(scores)) / bonela_human_mean) * 100:.2f}% pior que o Humano do Bonela (média: {bonela_human_mean:.2f})")
    
    # Ranking
    results_dict = {
        "Agente Neural GWO": np.mean(scores),
        "Agente Neural Genético (Bonela)": bonela_neural_mean,
        "Humano (Bonela)": bonela_human_mean,
        "Agente Genético Regras (Bonela)": bonela_rule_mean
    }
    
    sorted_results = sorted(results_dict.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n--- Ranking de Desempenho ---")
    for i, (name, score) in enumerate(sorted_results, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{medal} {i}º lugar: {name} - {score:.2f}")
    
    gwo_position = next(i for i, (name, _) in enumerate(sorted_results, 1) if name == "Agente Neural GWO")
    print(f"\nO Agente Neural GWO ficou em {gwo_position}º lugar no ranking!")
    
    if wall_deaths + obstacle_deaths > 0:
        print(f"\nCausas de Morte (primeiros 5 testes):")
        print(f"  Mortes por parede: {wall_deaths}")
        print(f"  Mortes por obstáculo: {obstacle_deaths}")
    
    return scores

def train_gwo():
    """Treina o agente neural melhorado usando GWO"""
    print("\n--- Iniciando Treinamento Neural com GWO ---")
    
    config = GameConfig()
    temp_agent = EnhancedNeuralAgent(config)
    dim = temp_agent.get_weights_count()
    
    print(f"Dimensões da rede neural melhorada: {dim} pesos")
    print(f"Arquitetura: {temp_agent.input_size} -> {temp_agent.hidden1_size} -> {temp_agent.hidden2_size} -> {temp_agent.output_size}")
    print(f"Features adicionais: 4 (distância/perigo das paredes)")
    
    gwo = GreyWolfOptimizer(
        fitness_function=fitness_function,
        dim=dim,
        n_wolves=100,  
        max_iter=1000,   
        bounds=(-2.5, 2.5)  
    )
    
    start_time = time.time()
    best_weights, best_score = gwo.optimize()
    end_time = time.time()
    
    print(f"\n--- Treinamento Concluído ---")
    print(f"Tempo total: {end_time - start_time:.2f} segundos")
    print(f"Melhor Score: {best_score:.2f}")
    
    np.save("best_weights.npy", best_weights)
    print("Melhores pesos salvos em 'best_weights.npy'")
    
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
    bonela_rule_based_genetic_result = [12.69, 16.65, 6.97, 2.79, 15.94, 10.22, 21.90, 4.35, 6.22, 9.95, 19.94, 20.56, 15.74, 17.68, 7.16, 15.68, 2.37, 15.43, 15.13, 22.50, 25.82, 15.85, 17.02, 16.74, 14.69, 11.73, 13.80, 15.13, 12.35, 16.19]
    bonela_neural_agent_genetic_result = [38.32, 54.53, 61.16, 27.55, 16.08, 26.00, 25.33, 18.30, 39.76, 48.17, 44.77, 47.54, 75.43, 23.68, 16.83, 15.81, 67.17, 53.54, 33.59, 49.24, 52.65, 16.35, 44.05, 56.59, 63.23, 43.96, 43.82, 19.19, 28.36, 18.65]
    bonela_human_result = [27.34, 17.63, 39.33, 17.44, 1.16, 24.04, 29.21, 18.92, 25.71, 20.05, 31.88, 15.39, 22.50, 19.27, 26.33, 23.67, 16.82, 28.45, 12.59, 33.01, 21.74, 14.23, 27.90, 24.80, 11.35, 30.12, 17.08, 22.96, 9.41, 35.22]
    quantas_vezes_superou_bonela_neural = 0
    quantas_vezes_superou_bonela_rule_based = 0
    quantas_vezes_superou_bonela_human = 0
    for _ in trange(num_hyper_tests, desc="Comparando com o do Bonela (média de 30 jogos, 10 vezes)"):
        agent = EnhancedNeuralAgent(config, weights)
        scores = []
        for _ in range(30):
            game = SurvivalGame(config, render=False)
            step_count = 0
            max_steps = 100000000
            while not game.all_players_dead() and step_count < max_steps:
                state = game.get_state(0, include_internals=True)
                action = agent.predict(state)
                game.update([action])
                step_count += 1
            scores.append(game.players[0].score)
        if np.mean(scores) > np.mean(bonela_neural_agent_genetic_result):
            quantas_vezes_superou_bonela_neural += 1
        if np.mean(scores) > np.mean(bonela_rule_based_genetic_result):
            quantas_vezes_superou_bonela_rule_based += 1
        if np.mean(scores) > np.mean(bonela_human_result):
            quantas_vezes_superou_bonela_human += 1
    print(f"\nO agente neural melhorado superou o agente do Bonela (Rule Based) em {quantas_vezes_superou_bonela_rule_based / num_hyper_tests * 100:.2f}% das vezes.")
    print(f"O agente neural melhorado superou o agente do Bonela (Neural Agent) em {quantas_vezes_superou_bonela_neural / num_hyper_tests * 100:.2f}% das vezes.")
    print(f"O agente neural melhorado superou o agente do Bonela (Human) em {quantas_vezes_superou_bonela_human / num_hyper_tests * 100:.2f}% das vezes.")


def graph_agent_performance(weights, num_games=30, compare_benchmarks=False):
    """Gera gráfico com o desempenho do agente em múltiplos jogos"""
    config = GameConfig()
    agent = EnhancedNeuralAgent(config, weights)
    scores = []
    
    print(f"\n--- Gerando Gráfico de Desempenho ({num_games} jogos) ---")
    
    for game_num in trange(num_games, desc="Executando jogos"):
        game = SurvivalGame(config, render=False)
        step_count = 0
        max_steps = 100000000
        
        while not game.all_players_dead() and step_count < max_steps:
            state = game.get_state(0, include_internals=True)
            action = agent.predict(state)
            game.update([action])
            step_count += 1
            
        scores.append(game.players[0].score)
    
    mean_score = np.mean(scores)
    max_score = np.max(scores)
    min_score = np.min(scores)
    std_score = np.std(scores)
    
    print(f"\n--- Estatísticas de Desempenho ---")
    print(f"Média: {mean_score:.2f}")
    print(f"Máximo: {max_score:.2f}")
    print(f"Mínimo: {min_score:.2f}")
    print(f"Desvio Padrão: {std_score:.2f}")
    
    if compare_benchmarks:
        # Dados dos benchmarks (usando os primeiros 'num_games' resultados se necessário)
        bonela_rule_based_genetic_result = [12.69, 16.65, 6.97, 2.79, 15.94, 10.22, 21.90, 4.35, 6.22, 9.95, 19.94, 20.56, 15.74, 17.68, 7.16, 15.68, 2.37, 15.43, 15.13, 22.50, 25.82, 15.85, 17.02, 16.74, 14.69, 11.73, 13.80, 15.13, 12.35, 16.19]
        bonela_neural_agent_genetic_result = [38.32, 54.53, 61.16, 27.55, 16.08, 26.00, 25.33, 18.30, 39.76, 48.17, 44.77, 47.54, 75.43, 23.68, 16.83, 15.81, 67.17, 53.54, 33.59, 49.24, 52.65, 16.35, 44.05, 56.59, 63.23, 43.96, 43.82, 19.19, 28.36, 18.65]
        bonela_human_result = [27.34, 17.63, 39.33, 17.44, 1.16, 24.04, 29.21, 18.92, 25.71, 20.05, 31.88, 15.39, 22.50, 19.27, 26.33, 23.67, 16.82, 28.45, 12.59, 33.01, 21.74, 14.23, 27.90, 24.80, 11.35, 30.12, 17.08, 22.96, 9.41, 35.22]
        
        # Ajustar os dados dos benchmarks para o número de jogos solicitado
        bonela_rule = bonela_rule_based_genetic_result[:num_games] if num_games <= 30 else bonela_rule_based_genetic_result
        bonela_neural = bonela_neural_agent_genetic_result[:num_games] if num_games <= 30 else bonela_neural_agent_genetic_result
        bonela_human = bonela_human_result[:num_games] if num_games <= 30 else bonela_human_result
        
        # Criar figura comparativa mais detalhada
        plt.figure(figsize=(15, 12))
        
        # Gráfico 1: Comparação de desempenho linha por linha
        plt.subplot(3, 2, 1)
        games_range = range(1, len(scores) + 1)
        plt.plot(games_range, scores, 'b-', marker='o', markersize=5, linewidth=2, label='Agente Neural GWO', alpha=0.8)
        
        if len(bonela_neural) >= len(scores):
            plt.plot(games_range, bonela_neural[:len(scores)], 'r-', marker='s', markersize=4, linewidth=2, label='Neural Genético (Bonela)', alpha=0.7)
        if len(bonela_human) >= len(scores):
            plt.plot(games_range, bonela_human[:len(scores)], 'g-', marker='^', markersize=4, linewidth=2, label='Humano (Bonela)', alpha=0.7)
        if len(bonela_rule) >= len(scores):
            plt.plot(games_range, bonela_rule[:len(scores)], 'orange', marker='d', markersize=4, linewidth=2, label='Regras Genético (Bonela)', alpha=0.7)
        
        plt.title('Comparação de Desempenho por Jogo', fontsize=12, fontweight='bold')
        plt.xlabel('Número do Jogo')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Gráfico 2: Box Plot comparativo
        plt.subplot(3, 2, 2)
        data_to_plot = [scores]
        labels = ['GWO Neural']
        
        if len(bonela_neural) >= len(scores):
            data_to_plot.append(bonela_neural[:len(scores)])
            labels.append('Neural Gen.')
        if len(bonela_human) >= len(scores):
            data_to_plot.append(bonela_human[:len(scores)])
            labels.append('Humano')
        if len(bonela_rule) >= len(scores):
            data_to_plot.append(bonela_rule[:len(scores)])
            labels.append('Regras Gen.')
        
        box_plot = plt.boxplot(data_to_plot, labels=labels, patch_artist=True)
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightsalmon']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        plt.title('Distribuição dos Scores', fontsize=12, fontweight='bold')
        plt.ylabel('Score')
        plt.grid(True, alpha=0.3)
        
        # Gráfico 3: Histograma comparativo
        plt.subplot(3, 2, 3)
        plt.hist(scores, bins=15, alpha=0.7, color='blue', label='GWO Neural', edgecolor='black')
        if len(bonela_neural) >= len(scores):
            plt.hist(bonela_neural[:len(scores)], bins=15, alpha=0.5, color='red', label='Neural Gen.', edgecolor='black')
        plt.axvline(x=mean_score, color='blue', linestyle='--', alpha=0.8, label=f'Média GWO: {mean_score:.2f}')
        plt.title('Distribuição - GWO vs Neural Genético', fontsize=12)
        plt.xlabel('Score')
        plt.ylabel('Frequência')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Gráfico 4: Barras com médias
        plt.subplot(3, 2, 4)
        methods = ['GWO Neural']
        means = [mean_score]
        stds = [std_score]
        colors_bar = ['blue']
        
        if len(bonela_neural) >= len(scores):
            methods.append('Neural Gen.')
            means.append(np.mean(bonela_neural[:len(scores)]))
            stds.append(np.std(bonela_neural[:len(scores)]))
            colors_bar.append('red')
        if len(bonela_human) >= len(scores):
            methods.append('Humano')
            means.append(np.mean(bonela_human[:len(scores)]))
            stds.append(np.std(bonela_human[:len(scores)]))
            colors_bar.append('green')
        if len(bonela_rule) >= len(scores):
            methods.append('Regras Gen.')
            means.append(np.mean(bonela_rule[:len(scores)]))
            stds.append(np.std(bonela_rule[:len(scores)]))
            colors_bar.append('orange')
        
        bars = plt.bar(methods, means, yerr=stds, capsize=5, color=colors_bar, alpha=0.7, edgecolor='black')
        plt.title('Comparação de Médias', fontsize=12, fontweight='bold')
        plt.ylabel('Score Médio')
        plt.xticks(rotation=45)
        
        # Adicionar valores nas barras
        for bar, mean_val in zip(bars, means):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{mean_val:.1f}', 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.grid(True, alpha=0.3, axis='y')
        
        # Gráfico 5: Média móvel
        plt.subplot(3, 2, 5)
        window_size = min(5, len(scores)//2) if len(scores) > 5 else 2
        
        # Calcular médias móveis
        gwo_moving_avg = [np.mean(scores[max(0, i-window_size+1):i+1]) for i in range(len(scores))]
        plt.plot(games_range, gwo_moving_avg, 'b-', linewidth=3, label=f'GWO (média móvel {window_size})')
        
        if len(bonela_neural) >= len(scores):
            neural_moving_avg = [np.mean(bonela_neural[max(0, i-window_size+1):i+1]) for i in range(len(scores))]
            plt.plot(games_range, neural_moving_avg, 'r-', linewidth=2, label=f'Neural Gen. (média móvel {window_size})')
        
        plt.title('Tendência de Desempenho (Média Móvel)', fontsize=12, fontweight='bold')
        plt.xlabel('Número do Jogo')
        plt.ylabel('Score (Média Móvel)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Gráfico 6: Estatísticas resumo
        plt.subplot(3, 2, 6)
        plt.axis('off')
        
        # Criar tabela de estatísticas
        stats_text = f"""
ESTATÍSTICAS COMPARATIVAS

Agente Neural GWO:
• Média: {np.mean(scores):.2f}
• Mediana: {np.median(scores):.2f}
• Desvio Padrão: {np.std(scores):.2f}
• Mín/Máx: {np.min(scores):.1f}/{np.max(scores):.1f}

Neural Genético (Bonela):
• Média: {np.mean(bonela_neural[:len(scores)]):.2f}
• Mediana: {np.median(bonela_neural[:len(scores)]):.2f}
• Desvio Padrão: {np.std(bonela_neural[:len(scores)]):.2f}
• Mín/Máx: {np.min(bonela_neural[:len(scores)]):.1f}/{np.max(bonela_neural[:len(scores)]):.1f}

Humano (Bonela):
• Média: {np.mean(bonela_human[:len(scores)]):.2f}
• Mediana: {np.median(bonela_human[:len(scores)]):.2f}
• Desvio Padrão: {np.std(bonela_human[:len(scores)]):.2f}
• Mín/Máx: {np.min(bonela_human[:len(scores)]):.1f}/{np.max(bonela_human[:len(scores)]):.1f}

Regras Genético (Bonela):
• Média: {np.mean(bonela_rule[:len(scores)]):.2f}
• Mediana: {np.median(bonela_rule[:len(scores)]):.2f}
• Desvio Padrão: {np.std(bonela_rule[:len(scores)]):.2f}
• Mín/Máx: {np.min(bonela_rule[:len(scores)]):.1f}/{np.max(bonela_rule[:len(scores)]):.1f}
        """
        
        plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        plt.tight_layout()
        filename = f'agent_comparison_{num_games}_games.png'
        
    else:
        # Gráfico simples original (sem comparação)
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(range(1, num_games + 1), scores, 'b-', marker='o', markersize=4, linewidth=1.5)
        plt.axhline(y=mean_score, color='r', linestyle='--', alpha=0.7, label=f'Média: {mean_score:.2f}')
        plt.axhline(y=max_score, color='g', linestyle='--', alpha=0.7, label=f'Máximo: {max_score:.2f}')
        plt.axhline(y=min_score, color='orange', linestyle='--', alpha=0.7, label=f'Mínimo: {min_score:.2f}')
        
        plt.title(f'Desempenho do Agente Neural - {num_games} Jogos', fontsize=14, fontweight='bold')
        plt.xlabel('Número do Jogo')
        plt.ylabel('Score')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.hist(scores, bins=min(15, num_games//2), alpha=0.7, color='skyblue', edgecolor='black')
        plt.axvline(x=mean_score, color='r', linestyle='--', alpha=0.7, label=f'Média: {mean_score:.2f}')
        plt.title('Distribuição dos Scores', fontsize=12)
        plt.xlabel('Score')
        plt.ylabel('Frequência')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f'agent_performance_{num_games}_games.png'
    
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nGráfico salvo como: {filename}")
    
    plt.show()
    
    return scores


if __name__ == "__main__":
    
    
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
        elif sys.argv[1] == "graph":
            compare_mode = False
            num_games = 30
            
            # Verificar argumentos adicionais
            if len(sys.argv) > 2:
                if sys.argv[2] == "compare":
                    compare_mode = True
                    num_games = int(sys.argv[3]) if len(sys.argv) > 3 else 30
                else:
                    try:
                        num_games = int(sys.argv[2])
                        compare_mode = sys.argv[3] == "compare" if len(sys.argv) > 3 else False
                    except ValueError:
                        if sys.argv[2] == "compare":
                            compare_mode = True
                        else:
                            print("Argumento inválido. Use: python gwo_trainer.py graph [num_jogos] [compare] ou python gwo_trainer.py graph compare [num_jogos]")
                            exit(1)
            
            try:
                weights = np.load("best_weights.npy")
                if compare_mode:
                    print("Modo de comparação ativado - gerando gráficos comparativos detalhados")
                graph_agent_performance(weights, num_games=num_games, compare_benchmarks=compare_mode)
            except FileNotFoundError:
                print("Arquivo de pesos não encontrado. Execute o treinamento primeiro.")
            except ImportError:
                print("matplotlib não encontrado. Instale com: pip install matplotlib")
        else:
            print("Uso: python gwo_trainer.py [demo|test|compare|graph] [opções]")
            print("  demo: Demonstração visual do agente")
            print("  test [n]: Testa o agente n vezes (padrão: 30)")
            print("  compare: Compara com resultados do Bonela")
            print("  graph [n]: Gera gráfico simples com n jogos (padrão: 30)")
            print("  graph compare [n]: Gera gráficos comparativos com benchmarks (padrão: 30)")
            print("  graph [n] compare: Alternativa para gráficos comparativos")
    else:
        train_gwo()
