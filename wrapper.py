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

        # Health tracking
        self.agent_hp = 176
        self.enemy_hp = 176

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
        obs, _, done, truncated, info = self.game.step(action)
        obs = self.preprocess(obs)

        # Get current health values from the game info
        current_agent_hp = info.get("agent_hp", self.agent_hp)
        current_enemy_hp = info.get("enemy_hp", self.enemy_hp)

        # Calculate health change (agent_hp_change - enemy_hp_change)
        # we want to create health advantage compared your opponent
        delta_hp_diff = (current_agent_hp - self.agent_hp) - (
            current_enemy_hp - self.enemy_hp
        )

        # Normalize health reward by max health (176) and add small time penalty
        # health double diff
        reward = delta_hp_diff / 176.0 - 0.0001

        # Update health tracking
        self.agent_hp = current_agent_hp
        self.enemy_hp = current_enemy_hp

        # Check for round end
        done = False
        if self.agent_hp <= 0 or self.enemy_hp <= 0:
            done = True
            if self.enemy_hp <= 0 and self.agent_hp > 0:
                reward += 1.0  # Large win bonus
                info["agent_won"] = True
            else:
                reward -= 1.0  # Large loss penalty
                info["agent_won"] = False

        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        """
        Reset the environment
        """
        result = self.game.reset(**kwargs)

        # Handle tuple return from game.reset()
        if isinstance(result, tuple):
            obs, info = result
        else:
            obs = result
            info = {}

        obs = self.preprocess(obs)

        # Initialize health values from the game info dictionary
        self.agent_hp = info.get("agent_hp", 176)
        self.enemy_hp = info.get("enemy_hp", 176)

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
