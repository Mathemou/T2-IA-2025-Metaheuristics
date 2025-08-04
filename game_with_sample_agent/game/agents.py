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