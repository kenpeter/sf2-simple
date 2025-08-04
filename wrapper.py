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
                "StreetFighterIISpecialChampionEdition-Genesis",
                state="ken_bison_12.state",
                use_restricted_actions=retro.Actions.FILTERED,
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

        # Use MultiBinary action space like Test notebook
        self.action_space = spaces.MultiBinary(12)

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
        
        # Health-based reward like Test notebook
        # Use agent_hp and enemy_hp keys
        current_health = info.get('agent_hp', self.health)
        current_enemy_health = info.get('enemy_hp', self.enemy_health)
        
        reward = (self.enemy_health - current_enemy_health)*2 + (current_health - self.health)
        
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


def make_env():
    """
    Create and return the Street Fighter environment
    """
    return StreetFighter()


# Export the main components
__all__ = ["StreetFighter", "make_env"]

print("🥊 Street Fighter RL wrapper loaded successfully!")
