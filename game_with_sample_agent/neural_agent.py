import numpy as np
from game.agents import Agent
from game.core import GameConfig

class NeuralAgent(Agent):
    def __init__(self, config: GameConfig, weights=None):
        self.config = config
        self.grid_size = config.sensor_grid_size
        
        # Definir arquitetura da rede neural
        self.input_size = self.grid_size * self.grid_size + 2  # grid + posição y + velocidade
        self.hidden_size = 10  # neurônios na camada oculta
        self.output_size = 3   # 3 ações possíveis (0, 1, 2)
        
        # Calcular número total de pesos necessários
        # Pesos da camada de entrada para oculta: input_size * hidden_size
        # Bias da camada oculta: hidden_size
        # Pesos da camada oculta para saída: hidden_size * output_size
        # Bias da camada de saída: output_size
        self.total_weights = (self.input_size * self.hidden_size + 
                             self.hidden_size + 
                             self.hidden_size * self.output_size + 
                             self.output_size)
        
        if weights is not None:
            self.set_weights(weights)
        else:
            # Inicializar pesos aleatórios
            self.weights = np.random.uniform(-1, 1, self.total_weights)
    
    def set_weights(self, weights):
        """Define os pesos da rede neural"""
        if len(weights) != self.total_weights:
            raise ValueError(f"Esperado {self.total_weights} pesos, recebido {len(weights)}")
        self.weights = weights.copy()
    
    def get_weights_count(self):
        """Retorna o número total de pesos necessários"""
        return self.total_weights
    
    def _extract_weights(self):
        """Extrai os pesos e bias da rede neural do vetor de pesos"""
        idx = 0
        
        # Pesos da camada de entrada para oculta
        w1_size = self.input_size * self.hidden_size
        W1 = self.weights[idx:idx + w1_size].reshape(self.input_size, self.hidden_size)
        idx += w1_size
        
        # Bias da camada oculta
        b1 = self.weights[idx:idx + self.hidden_size]
        idx += self.hidden_size
        
        # Pesos da camada oculta para saída
        w2_size = self.hidden_size * self.output_size
        W2 = self.weights[idx:idx + w2_size].reshape(self.hidden_size, self.output_size)
        idx += w2_size
        
        # Bias da camada de saída
        b2 = self.weights[idx:idx + self.output_size]
        
        return W1, b1, W2, b2
    
    def _sigmoid(self, x):
        """Função de ativação sigmoid"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip para evitar overflow
    
    def _softmax(self, x):
        """Função de ativação softmax para a camada de saída"""
        exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exp_x / np.sum(exp_x)
    
    def _forward(self, state):
        """Propagação direta pela rede neural"""
        W1, b1, W2, b2 = self._extract_weights()
        
        # Camada oculta
        z1 = np.dot(state, W1) + b1
        a1 = self._sigmoid(z1)
        
        # Camada de saída
        z2 = np.dot(a1, W2) + b2
        a2 = self._softmax(z2)
        
        return a2
    
    def predict(self, state: np.ndarray) -> int:
        """Prediz a ação baseada no estado atual"""
        # Preparar entrada da rede neural
        grid_flat = state[:self.grid_size * self.grid_size]
        player_y = state[-2]  # Posição Y normalizada
        player_speed = state[-1] if len(state) > self.grid_size * self.grid_size + 1 else 0.0
        
        # Concatenar todas as features
        neural_input = np.concatenate([grid_flat, [player_y, player_speed]])
        
        # Obter probabilidades das ações
        action_probs = self._forward(neural_input)
        
        # Escolher a ação com maior probabilidade
        action = np.argmax(action_probs)
        
        return int(action)
    
    def predict_with_probs(self, state: np.ndarray):
        """Retorna tanto a ação quanto as probabilidades (útil para debug)"""
        grid_flat = state[:self.grid_size * self.grid_size]
        player_y = state[-2]
        player_speed = state[-1] if len(state) > self.grid_size * self.grid_size + 1 else 0.0
        
        neural_input = np.concatenate([grid_flat, [player_y, player_speed]])
        action_probs = self._forward(neural_input)
        action = np.argmax(action_probs)
        
        return int(action), action_probs


class EnhancedNeuralAgent(Agent):
    """Agente neural melhorado com informações sobre paredes"""
    def __init__(self, config: GameConfig, weights=None):
        self.config = config
        self.grid_size = config.sensor_grid_size
        
        # Arquitetura melhorada com mais features
        grid_features = self.grid_size * self.grid_size  # 25 - grid de obstáculos
        position_features = 1  # 1 - posição Y normalizada
        speed_features = 1     # 1 - velocidade do jogo
        wall_features = 4      # 4 - distâncias normalizadas para paredes + flags de perigo
        
        self.input_size = grid_features + position_features + speed_features + wall_features  # 31 total
        self.hidden_size = 12   # Mais neurônios para processar mais informações
        self.output_size = 3    # 3 ações
        
        # Número total de pesos
        self.total_weights = (self.input_size * self.hidden_size + 
                             self.hidden_size + 
                             self.hidden_size * self.output_size + 
                             self.output_size)
        
        if weights is not None:
            self.set_weights(weights)
        else:
            self.weights = np.random.uniform(-2, 2, self.total_weights)
    
    def set_weights(self, weights):
        if len(weights) != self.total_weights:
            raise ValueError(f"Esperado {self.total_weights} pesos, recebido {len(weights)}")
        self.weights = weights.copy()
    
    def get_weights_count(self):
        return self.total_weights
    
    def _extract_wall_features(self, state):
        """Extrai features relacionadas às paredes"""
        # Posição Y atual (já normalizada)
        player_y_norm = state[-2]
        player_y_actual = player_y_norm * self.config.screen_height
        
        # Calcular distâncias para as paredes
        distance_to_top = player_y_actual - self.config.player_radius
        distance_to_bottom = (self.config.screen_height - self.config.player_radius) - player_y_actual
        
        # Normalizar distâncias (0 = na parede, 1 = no centro)
        max_distance = self.config.screen_height / 2
        norm_dist_top = np.clip(distance_to_top / max_distance, 0, 1)
        norm_dist_bottom = np.clip(distance_to_bottom / max_distance, 0, 1)
        
        # Flags de perigo (1 = muito perto da parede)
        danger_threshold = 50  # pixels
        danger_top = 1.0 if distance_to_top < danger_threshold else 0.0
        danger_bottom = 1.0 if distance_to_bottom < danger_threshold else 0.0
        
        return np.array([norm_dist_top, norm_dist_bottom, danger_top, danger_bottom])
    
    def _extract_weights(self):
        idx = 0
        w1_size = self.input_size * self.hidden_size
        W1 = self.weights[idx:idx + w1_size].reshape(self.input_size, self.hidden_size)
        idx += w1_size
        
        b1 = self.weights[idx:idx + self.hidden_size]
        idx += self.hidden_size
        
        w2_size = self.hidden_size * self.output_size
        W2 = self.weights[idx:idx + w2_size].reshape(self.hidden_size, self.output_size)
        idx += w2_size
        
        b2 = self.weights[idx:idx + self.output_size]
        
        return W1, b1, W2, b2
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _forward(self, state):
        W1, b1, W2, b2 = self._extract_weights()
        
        # Camada oculta com ReLU
        z1 = np.dot(state, W1) + b1
        a1 = self._relu(z1)
        
        # Camada de saída (linear)
        z2 = np.dot(a1, W2) + b2
        
        return z2
    
    def predict(self, state: np.ndarray) -> int:
        # Preparar entrada básica
        grid_flat = state[:self.grid_size * self.grid_size]
        player_y = state[-2]
        player_speed = state[-1] if len(state) > self.grid_size * self.grid_size + 1 else 0.0
        
        # Extrair features das paredes
        wall_features = self._extract_wall_features(state)
        
        # Concatenar todas as features
        neural_input = np.concatenate([grid_flat, [player_y, player_speed], wall_features])
        
        # Obter saídas da rede
        outputs = self._forward(neural_input)
        
        # Escolher ação com maior valor
        action = np.argmax(outputs)
        
        return int(action)


class SimpleNeuralAgent(Agent):
    """Versão mais simples da rede neural com menos parâmetros"""
    def __init__(self, config: GameConfig, weights=None):
        self.config = config
        self.grid_size = config.sensor_grid_size
        
        # Arquitetura mais simples
        self.input_size = self.grid_size * self.grid_size + 1  # grid + posição y
        self.hidden_size = 6   # Menos neurônios
        self.output_size = 3   # 3 ações
        
        # Número total de pesos
        self.total_weights = (self.input_size * self.hidden_size + 
                             self.hidden_size + 
                             self.hidden_size * self.output_size + 
                             self.output_size)
        
        if weights is not None:
            self.set_weights(weights)
        else:
            self.weights = np.random.uniform(-2, 2, self.total_weights)
    
    def set_weights(self, weights):
        if len(weights) != self.total_weights:
            raise ValueError(f"Esperado {self.total_weights} pesos, recebido {len(weights)}")
        self.weights = weights.copy()
    
    def get_weights_count(self):
        return self.total_weights
    
    def _extract_weights(self):
        idx = 0
        w1_size = self.input_size * self.hidden_size
        W1 = self.weights[idx:idx + w1_size].reshape(self.input_size, self.hidden_size)
        idx += w1_size
        
        b1 = self.weights[idx:idx + self.hidden_size]
        idx += self.hidden_size
        
        w2_size = self.hidden_size * self.output_size
        W2 = self.weights[idx:idx + w2_size].reshape(self.hidden_size, self.output_size)
        idx += w2_size
        
        b2 = self.weights[idx:idx + self.output_size]
        
        return W1, b1, W2, b2
    
    def _tanh(self, x):
        return np.tanh(x)
    
    def _forward(self, state):
        W1, b1, W2, b2 = self._extract_weights()
        
        # Camada oculta com tanh
        z1 = np.dot(state, W1) + b1
        a1 = self._tanh(z1)
        
        # Camada de saída (linear)
        z2 = np.dot(a1, W2) + b2
        
        return z2
    
    def predict(self, state: np.ndarray) -> int:
        # Preparar entrada
        grid_flat = state[:self.grid_size * self.grid_size]
        player_y = state[-2]
        
        neural_input = np.concatenate([grid_flat, [player_y]])
        
        # Obter saídas da rede
        outputs = self._forward(neural_input)
        
        # Escolher ação com maior valor
        action = np.argmax(outputs)
        
        return int(action)


# Função auxiliar para criar agentes
def create_neural_agent(config, weights, agent_type="enhanced"):
    """Factory function para criar diferentes tipos de agentes neurais"""
    if agent_type == "simple":
        return SimpleNeuralAgent(config, weights)
    elif agent_type == "enhanced":
        return EnhancedNeuralAgent(config, weights)
    elif agent_type == "full":
        return NeuralAgent(config, weights)
    else:
        raise ValueError(f"Tipo de agente desconhecido: {agent_type}")
