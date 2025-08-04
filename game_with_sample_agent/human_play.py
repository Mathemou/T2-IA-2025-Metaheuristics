from game.core import SurvivalGame
from game.config import GameConfig
from game.agents import HumanAgent
import pygame

def manual_play():
    config = GameConfig(render_grid = True)
    game = SurvivalGame(config, render=True)
    agent = HumanAgent()
    
    running = True
    while running:
        action = 0  # Nada
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            action = 1  # Cima
        elif keys[pygame.K_DOWN]:
            action = 2  # Baixo
        
        game.update([action])
        game.render_frame()
        
        if game.all_players_dead():
            print(f"Game Over!")
            running = False
    
    pygame.quit()

if __name__ == "__main__":
    manual_play()