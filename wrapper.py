#!/usr/bin/env python3
"""
🥊 Street Fighter RL Environment - Based on nicknochnack/StreetFighterRL
Simple PPO-based implementation using Stable Baselines3
"""

import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import retro

# --- FIX for TypeError in retro.make ---
_original_retro_make = retro.make

def _patched_retro_make(game, state=None, **kwargs):
    if not state:
        state = "ken_bison_12.state"
    return _original_retro_make(game=game, state=state, **kwargs)

retro.make = _patched_retro_make

print("🥊 Street Fighter RL Environment - Simple PPO Implementation")


class StreetFighter(gym.Env):
    """
    Custom Street Fighter environment based on nicknochnack/StreetFighterRL
    Simplified wrapper for PPO training
    """
    
    def __init__(self):
        super().__init__()
        
        # Create the retro environment
        try:
            self.game = retro.make(
                'StreetFighterIISpecialChampionEdition-Genesis', 
                state='ken_bison_12.state',
                use_restricted_actions=retro.Actions.FILTERED
            )
            print("✅ Street Fighter ROM found and loaded!")
        except FileNotFoundError:
            print("❌ Street Fighter ROM not found.")
            print("   To use Street Fighter: python -m retro.import /path/to/rom/file")
            raise
        
        # Define observation space - 84x84 grayscale
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )
        
        # Use the filtered action space from retro
        self.action_space = self.game.action_space
        
        # Health tracking for reward shaping
        self.player_health = 0
        self.enemy_health = 0
        
        # Position tracking
        self.agent_x = 0
        self.agent_y = 0
        self.enemy_x = 0
        self.enemy_y = 0
        
        # Round tracking for best of 3
        self.agent_rounds_won = 0
        self.enemy_rounds_won = 0
        
    def preprocess(self, observation):
        """
        Preprocess observation: convert to grayscale and resize to 84x84
        """
        # Convert to grayscale
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        
        # Resize to 84x84
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        
        # Add channel dimension
        return np.expand_dims(resized, axis=-1)
    
    def step(self, action):
        """
        Take a step in the environment
        """
        obs, reward, done, truncated, info = self.game.step(action)
        
        # Preprocess observation
        obs = self.preprocess(obs)
        
        # --- REWARD SHAPING ---
        current_player_health = info.get('agent_hp', 0)
        current_enemy_health = info.get('enemy_hp', 0)
        
        # Position tracking
        current_agent_x = info.get('agent_x', 0)
        current_agent_y = info.get('agent_y', 0)
        current_enemy_x = info.get('enemy_x', 0)
        current_enemy_y = info.get('enemy_y', 0)
        
        # Reward for damaging the opponent
        damage_to_enemy = self.enemy_health - current_enemy_health
        
        # Penalty for taking damage
        damage_to_player = self.player_health - current_player_health
        
        # Check for round end and update round counters
        round_ended = False
        if current_player_health <= 0 and self.player_health > 0:
            # Agent lost this round
            self.enemy_rounds_won += 1
            round_ended = True
        elif current_enemy_health <= 0 and self.enemy_health > 0:
            # Agent won this round
            self.agent_rounds_won += 1
            round_ended = True
        
        # Combine into a single reward signal
        # You can tune these weights
        reward = damage_to_enemy * 1.5 - damage_to_player * 1.0 - 0.001
        
        # Add round win/loss rewards
        if round_ended:
            if current_player_health <= 0:
                reward -= 20  # Penalty for losing round
            elif current_enemy_health <= 0:
                reward += 20  # Bonus for winning round
        
        # Check if best of 3 is complete
        if self.agent_rounds_won >= 2:
            # Agent won best of 3 - reset the game
            reward += 50  # Big bonus for winning match
            done = True
            # Reset game to beginning
            self.game.reset()
            self.agent_rounds_won = 0
            self.enemy_rounds_won = 0
        elif self.enemy_rounds_won >= 2:
            # Agent lost best of 3 - reset the game
            reward -= 50  # Big penalty for losing match
            done = True
            # Reset game to beginning
            self.game.reset()
            self.agent_rounds_won = 0
            self.enemy_rounds_won = 0

        # Update the stored health values for the next step
        self.player_health = current_player_health
        self.enemy_health = current_enemy_health
        
        # Update position values
        self.agent_x = current_agent_x
        self.agent_y = current_agent_y
        self.enemy_x = current_enemy_x
        self.enemy_y = current_enemy_y
        
        return obs, reward, done, truncated, info
    
    def reset(self, **kwargs):
        """
        Reset the environment
        """
        obs, info = self.game.reset(**kwargs)
        
        # Reset health scores from info dict
        self.player_health = info.get('agent_hp', 0)
        self.enemy_health = info.get('enemy_hp', 0)
        
        # Reset position values
        self.agent_x = info.get('agent_x', 0)
        self.agent_y = info.get('agent_y', 0)
        self.enemy_x = info.get('enemy_x', 0)
        self.enemy_y = info.get('enemy_y', 0)
        
        # Reset round counters
        self.agent_rounds_won = 0
        self.enemy_rounds_won = 0
        
        # Preprocess observation
        obs = self.preprocess(obs)
        
        return obs, info
    
    def render(self, mode='human'):
        """
        Render the environment
        """
        return self.game.render()
    
    def close(self):
        """
        Close the environment
        """
        self.game.close()


def make_env():
    """
    Create and return the Street Fighter environment
    """
    return StreetFighter()


# Export the main components
__all__ = [
    "StreetFighter", 
    "make_env"
]

print("🥊 Street Fighter RL wrapper loaded successfully!")