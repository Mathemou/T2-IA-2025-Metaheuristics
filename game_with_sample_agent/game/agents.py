import numpy as np
from abc import ABC, abstractmethod
from typing import List
from game.core import GameConfig
import random

class Agent(ABC):
    @abstractmethod
    def predict(self, state: np.ndarray) -> int:
        pass

class HumanAgent(Agent):
    def predict(self, state: np.ndarray) -> int:
        return 0 #the input is from keyboard

class RuleBasedAgent(Agent):
    def __init__(self, config: GameConfig,danger_threshold = None, lookahead_cells = None, diff_to_center_to_move = None):
        self.config = config
        self.grid_size = config.sensor_grid_size
        self.sensor_range = config.sensor_range
        self.cell_size = self.sensor_range / self.grid_size

        if danger_threshold == None:
            self.danger_threshold = 0.3  # How close obstacles need to be to react
        else:
            self.danger_threshold = danger_threshold

        if lookahead_cells == None:
            self.lookahead_cells = 3  # How many cells ahead to check for obstacles
        else:
            self.lookahead_cells = int(np.rint(lookahead_cells))

        if diff_to_center_to_move == None:
            self.diff_to_center_to_move = 200
        else:
            self.diff_to_center_to_move = diff_to_center_to_move
        
    def predict(self, state: np.ndarray) -> int:
        # Reshape the state into grid if it's flattened
        grid_flat = state[:self.grid_size*self.grid_size]
        grid = grid_flat.reshape((self.grid_size, self.grid_size))
        player_y_normalized = state[-2] * self.config.screen_height # Second last element
        center_row = self.grid_size // 2
        
        # Check immediate danger in front (first column)
        first_col = grid[:, 0]
        if np.any(first_col):
            # Obstacle directly in front - need to dodge
            obstacle_rows = np.where(first_col)[0]
            
            # If obstacle is above center, go down
            if np.any(obstacle_rows < center_row):
                return 2
            # If obstacle is below center or covers center, go up
            else:
                return 1
        
        # Look ahead in the next few columns for obstacles
        for col in range(1, min(self.lookahead_cells, self.grid_size)):
            if np.any(grid[:, col]):
                # Calculate distance to obstacle
                distance = col * self.cell_size
                
                # If obstacle is getting close, prepare to dodge
                if distance < self.danger_threshold * self.sensor_range:
                    obstacle_rows = np.where(grid[:, col])[0]
                    
                    # Find the gap (if any)
                    obstacle_present = np.zeros(self.grid_size, dtype=bool)
                    obstacle_present[obstacle_rows] = True
                    
                    # Check for gaps above or below
                    gap_above = not np.any(obstacle_present[:center_row])
                    gap_below = not np.any(obstacle_present[center_row+1:])
                    
                    if gap_above and not gap_below:
                        return 1  # Move up
                    elif gap_below and not gap_above:
                        return 2  # Move down
                    elif gap_above and gap_below:
                        # Both gaps available, choose randomly
                        return random.choice([1, 2])
                    else:
                        # No gap, choose randomly (will probably hit)
                        return random.choice([0, 1, 2])
        #print(player_y_normalized)
        diff_to_center = player_y_normalized - (self.config.screen_height/2)
 
        if diff_to_center < -self.diff_to_center_to_move:
            return 2  # Must move down
        elif diff_to_center > self.diff_to_center_to_move:
            return 1  # Must move up

        # Default action - no movement needed
        return 0
    
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
    """Agente neural melhorado com informações sobre paredes e duas camadas ocultas"""
    def __init__(self, config: GameConfig, weights=None):
        self.config = config
        self.grid_size = config.sensor_grid_size

        # Arquitetura melhorada com mais features
        grid_features = self.grid_size * self.grid_size  # 25 - grid de obstáculos
        position_features = 1  # 1 - posição Y normalizada
        speed_features = 1     # 1 - velocidade do jogo
        wall_features = 4      # 4 - distâncias normalizadas para paredes + flags de perigo

        self.input_size = grid_features + position_features + speed_features + wall_features  # 31 total
        self.hidden1_size = 32   # Primeira camada oculta
        self.hidden2_size = 16   # Segunda camada oculta
        self.output_size = 3     # 3 ações

        # Número total de pesos
        # input -> hidden1: input_size * hidden1_size + hidden1_size (bias)
        # hidden1 -> hidden2: hidden1_size * hidden2_size + hidden2_size (bias)
        # hidden2 -> output: hidden2_size * output_size + output_size (bias)
        self.total_weights = (
            self.input_size * self.hidden1_size +
            self.hidden1_size +
            self.hidden1_size * self.hidden2_size +
            self.hidden2_size +
            self.hidden2_size * self.output_size +
            self.output_size
        )

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
        player_y_norm = state[-2]
        player_y_actual = player_y_norm * self.config.screen_height

        distance_to_top = player_y_actual - self.config.player_radius
        distance_to_bottom = (self.config.screen_height - self.config.player_radius) - player_y_actual

        max_distance = self.config.screen_height / 2
        norm_dist_top = np.clip(distance_to_top / max_distance, 0, 1)
        norm_dist_bottom = np.clip(distance_to_bottom / max_distance, 0, 1)

        danger_threshold = 50  # pixels
        danger_top = 1.0 if distance_to_top < danger_threshold else 0.0
        danger_bottom = 1.0 if distance_to_bottom < danger_threshold else 0.0

        return np.array([norm_dist_top, norm_dist_bottom, danger_top, danger_bottom])

    def _extract_weights(self):
        idx = 0
        # input -> hidden1
        w1_size = self.input_size * self.hidden1_size
        W1 = self.weights[idx:idx + w1_size].reshape(self.input_size, self.hidden1_size)
        idx += w1_size
        b1 = self.weights[idx:idx + self.hidden1_size]
        idx += self.hidden1_size

        # hidden1 -> hidden2
        w2_size = self.hidden1_size * self.hidden2_size
        W2 = self.weights[idx:idx + w2_size].reshape(self.hidden1_size, self.hidden2_size)
        idx += w2_size
        b2 = self.weights[idx:idx + self.hidden2_size]
        idx += self.hidden2_size

        # hidden2 -> output
        w3_size = self.hidden2_size * self.output_size
        W3 = self.weights[idx:idx + w3_size].reshape(self.hidden2_size, self.output_size)
        idx += w3_size
        b3 = self.weights[idx:idx + self.output_size]

        return W1, b1, W2, b2, W3, b3

    def _relu(self, x):
        return np.maximum(0, x)

    def _forward(self, state):
        W1, b1, W2, b2, W3, b3 = self._extract_weights()
        # Primeira camada oculta
        z1 = np.dot(state, W1) + b1
        a1 = self._relu(z1)
        # Segunda camada oculta
        z2 = np.dot(a1, W2) + b2
        a2 = self._relu(z2)
        # Saída
        z3 = np.dot(a2, W3) + b3
        return z3

    def predict(self, state: np.ndarray) -> int:
        grid_flat = state[:self.grid_size * self.grid_size]
        player_y = state[-2]
        player_speed = state[-1] if len(state) > self.grid_size * self.grid_size + 1 else 0.0

        wall_features = self._extract_wall_features(state)
        neural_input = np.concatenate([grid_flat, [player_y, player_speed], wall_features])

        outputs = self._forward(neural_input)
        action = np.argmax(outputs)
        return int(action)
