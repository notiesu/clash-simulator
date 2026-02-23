import json
import pygame
import os
from src.clasher.gym_env import ClashRoyaleGymEnv
from wrappers.replaymodel import ReplayInferenceModel
from visualize_battle import BattleVisualizer
import argparse

from src.clasher.model_state import ReplayState

# Initialize Pygame
pygame.init()

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 100, 100)
BLUE = (100, 100, 255)
GREEN = (100, 255, 100)
GRAY = (128, 128, 128)
DARK_GRAY = (64, 64, 64)
YELLOW = (255, 255, 100)
PURPLE = (255, 100, 255)
CYAN = (100, 255, 255)
ORANGE = (255, 165, 0)

# Screen settings
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 900
# Make tiles square: 18 wide × 32 tall
TILE_SIZE = 22  # Square tiles
ARENA_WIDTH = 18 * TILE_SIZE   # 396 pixels
ARENA_HEIGHT = 32 * TILE_SIZE  # 704 pixels
ARENA_X = 50
ARENA_Y = 50

DECK = ["Cannon", "Fireball", "HogRider", "IceGolemite", "IceSpirits", "Musketeer", "Skeletons", "Log"]
class ReplayVisualizer(BattleVisualizer):
    def __init__(self, replay_path):
        super().__init__()
        self.p1_model = ReplayInferenceModel(replay_path=replay_path, player_id=1)
        self.env = ClashRoyaleGymEnv(opponent_policy=self.p1_model, opponent_state=ReplayState(), deck0=DECK, deck1=DECK)
        self.env.reset()
        self.engine = self.env.engine
        self.battle = self.env.battle
        self.obs, self.info = self.env.reset()
        self.p0_model = ReplayInferenceModel(replay_path=replay_path, player_id=0)
        #get the initial hands and set them in the environment
        self.replay = self.p0_model.replay_data
        self.p0_initial_hand = self.replay[0]['players'][0]['hand']
        self.p1_initial_hand = self.replay[0]['players'][1]['hand']
        self.env.set_player_deck(0, DECK, initial_hand=self.p0_initial_hand)
        self.env.set_player_deck(1, DECK, initial_hand=self.p1_initial_hand)


        self.state = ReplayState()
        

    def setup_test_battle(self):
        # overwriting to pass this
        pass

    def handle_events(self):
        """Handle pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.paused = not getattr(self, 'paused', False)
                elif event.key == pygame.K_r:
                    # Reset battle
                    self.env = ClashRoyaleGymEnv()
                    self.engine = self.env.engine
                    self.battle = self.env.battle
                    self.env.reset()
                elif event.key >= pygame.K_1 and event.key <= pygame.K_5:
                    # Set speed multiplier
                    self.speed = event.key - pygame.K_0
                elif event.key == pygame.K_i:
                    # Toggle investigation mode
                    self.investigation_mode = not getattr(self, 'investigation_mode', False)
                    if self.investigation_mode:
                        print("🔍 Investigation mode ON - taking screenshots every 30 ticks")
                        self.investigation_counter = 0
                        # Create investigation folder
                        import datetime
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        self.investigation_folder = f"investigation/{timestamp}"
                        os.makedirs(self.investigation_folder, exist_ok=True)
                    else:
                        print("🔍 Investigation mode OFF")
                elif event.key == pygame.K_s:
                    # Take single screenshot
                    import datetime
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"manual_screenshot_{timestamp}.png"
                    self.take_screenshot(filename)
                elif event.key == pygame.K_p:
                    # Take pathfinding debug screenshot
                    self.take_pathfinding_debug_screenshot()
        
        return True
    
    def run(self):
        """Main visualization loop"""
        print("🎮 Starting Battle Visualization")
        print("Controls:")
        print("  SPACE: Pause/Resume")
        print("  R: Reset Battle")
        print("  1-5: Speed multiplier (1x to 5x)")
        print("  I: Toggle investigation mode (auto screenshots)")
        print("  S: Take manual screenshot")
        print("  P: Take pathfinding debug screenshot")
        print("  ESC: Exit")
        
        self.paused = False
        self.speed = 1
        running = True
        
        while running:
            # Handle events
            running = self.handle_events()
            
            # Update battle
            if not self.paused and not self.battle.game_over:
                action, self.state = self.p0_model.predict(self.obs, valid_action_mask=None, state=self.state)
                self.obs, _,_,_, self.info = self.env.step(action=action)
                self.battle = self.env.battle # Update battle reference after step
                    
                # Investigation mode - take screenshots at intervals
                if getattr(self, 'investigation_mode', False):
                    if not hasattr(self, 'investigation_counter'):
                        self.investigation_counter = 0
                    self.investigation_counter += self.speed
                    
                    # Take screenshot every 30 ticks
                    if self.investigation_counter >= 30:
                        self.investigation_counter = 0
                        # Draw everything first
                        self.screen.fill(WHITE)
                        self.draw_arena()
                        self.draw_towers()
                        self.draw_entities()
                        self.draw_ui()
                        pygame.display.flip()
                        
                        # Take investigation screenshot
                        tick = getattr(self.battle, 'tick', 0)
                        # print(tick)
                        filename = f"{self.investigation_folder}/tick_{tick:04d}.png"
                        self.take_screenshot(filename)
            
            # Draw everything
            self.screen.fill(WHITE)
            self.draw_arena()
            self.draw_towers()
            self.draw_entities()
            self.draw_ui()
            
            # Show pause indicator
            if self.paused:
                pause_text = self.large_font.render("PAUSED", True, RED)
                pause_rect = pause_text.get_rect(center=(SCREEN_WIDTH//2, 30))
                self.screen.blit(pause_text, pause_rect)
            
            # Show speed indicator
            if self.speed > 1:
                speed_text = self.font.render(f"Speed: {self.speed}x", True, PURPLE)
                speed_rect = speed_text.get_rect(topleft=(10, 10))
                self.screen.blit(speed_text, speed_rect)
            
            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS display
        
        pygame.quit()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Run Clash Royale simulation with PPO Inference Model.")
    parser.add_argument("--replay_path", type=str, required=True, help="Path to the replay JSONL file.")
    args = parser.parse_args()

    visualizer = ReplayVisualizer(args.replay_path)
    visualizer.run()