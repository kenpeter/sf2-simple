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
from discretizer import StreetFighter2Discretizer

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
            game = retro.make(
                "StreetFighterIISpecialChampionEdition-Genesis",
                state="ken_bison_12.state",
                use_restricted_actions=retro.Actions.FILTERED,
            )
            # Wrap with discretizer for special moves
            self.game = StreetFighter2Discretizer(game)
            print("✅ Street Fighter ROM found and loaded with special moves!")
        except FileNotFoundError:
            print("❌ Street Fighter ROM not found.")
            print("   To use Street Fighter: python -m retro.import /path/to/rom/file")
            raise

        # Define observation space - 84x84 grayscale
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

        # Use discrete action space from discretizer (includes special moves)
        self.action_space = self.game.action_space
        
        # Round tracking for best of 3
        self.agent_rounds_won = 0
        self.enemy_rounds_won = 0

    def preprocess(self, observation):
        """
        Preprocess observation: convert to grayscale and resize to 84x84
        """
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_CUBIC)
        state = np.reshape(resize, (84, 84, 1))
        return state

    def step(self, action):
        """
        Take a step in the environment
        """
        obs, reward, done, truncated, info = self.game.step(action)
        obs = self.preprocess(obs)
        
        # Preprocess frame from game - no frame delta like Test notebook
        frame_delta = obs
        
        # Simple health-based reward
        current_health = info.get('agent_hp', self.health)
        current_enemy_health = info.get('enemy_hp', self.enemy_health)
        
        # Normalized rewards: 
        # Agent hits opponent: +0.75 reward per HP damage dealt
        # Opponent hits agent: -0.25 reward per HP damage taken
        # This maintains 3:1 ratio but with normalized values
        reward = (self.enemy_health - current_enemy_health) * 0.75 + (current_health - self.health) * 0.25
        
        # Check for round end and update round counters
        round_ended = False
        if current_health <= 0 and self.health > 0:
            # Agent lost this round
            self.enemy_rounds_won += 1
            round_ended = True
        elif current_enemy_health <= 0 and self.enemy_health > 0:
            # Agent won this round
            self.agent_rounds_won += 1
            round_ended = True
        
        # Add normalized rewards for round win/loss
        if round_ended:
            if current_health <= 0:
                reward -= 1.0  # Normalized penalty for losing round
            elif current_enemy_health <= 0:
                reward += 1.0  # Normalized bonus for winning round
        
        # Check if single round is complete - no match rewards, just reset
        if self.agent_rounds_won >= 1:
            # Agent won single round - reset the game
            done = True
            # Reset game to beginning
            self.game.reset()
            self.agent_rounds_won = 0
            self.enemy_rounds_won = 0
        elif self.enemy_rounds_won >= 1:
            # Agent lost single round - reset the game
            done = True
            # Reset game to beginning
            self.game.reset()
            self.agent_rounds_won = 0
            self.enemy_rounds_won = 0
        
        # Update health tracking
        self.health = current_health
        self.enemy_health = current_enemy_health
        
        return frame_delta, reward, done, truncated, info

    def reset(self, **kwargs):
        """
        Reset the environment
        """
        result = self.game.reset()
        
        # Handle tuple return from game.reset()
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}
            
        obs = self.preprocess(obs)
        
        # Initialize health values like Test notebook
        self.health = 176
        self.enemy_health = 176
        
        # Reset round counters
        self.agent_rounds_won = 0
        self.enemy_rounds_won = 0
        
        return obs, info

    def render(self, mode="human"):
        """
        Render the environment
        """
        return self.game.render()

    def close(self):
        """
        Close the environment
        """
        self.game.close()
    
    def get_action_meaning(self, action):
        """
        Get the meaning of an action (for debugging)
        """
        return self.game.get_action_meaning(action)


def make_env():
    """
    Create and return the Street Fighter environment
    """
    return StreetFighter()


# Export the main components
__all__ = ["StreetFighter", "make_env"]

print("🥊 Street Fighter RL wrapper loaded successfully!")
