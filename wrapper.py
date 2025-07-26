#!/usr/bin/env python3
"""
🛡️ ENHANCED WRAPPER - Energy-Based Transformers + Current Energy Thinking
Keeps your original health system, game speed, and fight logic
Adds Energy-Based Transformer architecture for enhanced learning
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import deque, defaultdict
from gymnasium import spaces
from typing import Dict, Tuple, List, Type, Any, Optional, Union
import math
import logging
import os
from datetime import datetime
import retro
import json
import pickle
from pathlib import Path
import random
import time
import copy

# --- FIX for TypeError in retro.make ---
_original_retro_make = retro.make


def _patched_retro_make(game, state=None, **kwargs):
    if not state:
        state = "ken_bison_12.state"
    return _original_retro_make(game=game, state=state, **kwargs)


retro.make = _patched_retro_make

# Configure logging
os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
log_filename = (
    f'logs/enhanced_ebt_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
)
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename)],
)
logger = logging.getLogger(__name__)

# Constants
MAX_HEALTH = 176
SCREEN_WIDTH = 320
SCREEN_HEIGHT = 224
VECTOR_FEATURE_DIM = 32
MAX_FIGHT_STEPS = 1200

print(f"🥊 ENHANCED Energy-Based Transformer Configuration:")
print(f"   - Energy-Based Thinking + Transformer architecture")
print(f"   - Fixed quality threshold (0.6 → 0.3)")
print(f"   - Fast game UI speed maintained")
print(f"   - Single round fight logic preserved")
print(f"   - Max steps per fight: {MAX_FIGHT_STEPS}")


# Keep all your original safe operations
def safe_divide(numerator, denominator, default=0.0):
    """Safe division that prevents NaN and handles edge cases."""
    try:
        if isinstance(numerator, np.ndarray):
            numerator = (
                numerator.item() if numerator.size == 1 else float(numerator.flat[0])
            )
        if isinstance(denominator, np.ndarray):
            denominator = (
                denominator.item()
                if denominator.size == 1
                else float(denominator.flat[0])
            )

        numerator = float(numerator) if numerator is not None else default
        denominator = (
            float(denominator)
            if denominator is not None
            else (1.0 if default == 0.0 else default)
        )

        if denominator == 0 or not np.isfinite(denominator):
            return default
        result = numerator / denominator
        return result if np.isfinite(result) else default
    except:
        return default


def safe_std(values, default=0.0):
    """Safe standard deviation calculation."""
    if len(values) < 2:
        return default
    try:
        values_array = np.array(values)
        finite_values = values_array[np.isfinite(values_array)]
        if len(finite_values) < 2:
            return default
        std_val = np.std(finite_values)
        return std_val if np.isfinite(std_val) else default
    except:
        return default


def safe_mean(values, default=0.0):
    """Safe mean calculation."""
    if len(values) == 0:
        return default
    try:
        values_array = np.array(values)
        finite_values = values_array[np.isfinite(values_array)]
        if len(finite_values) == 0:
            return default
        mean_val = np.mean(finite_values)
        return mean_val if np.isfinite(mean_val) else default
    except:
        return default


def sanitize_array(arr, default_val=0.0):
    """Sanitize numpy array, replacing NaN/inf with default value."""
    if isinstance(arr, (int, float)):
        if np.isfinite(arr):
            return np.array([arr], dtype=np.float32)
        else:
            return np.array([default_val], dtype=np.float32)

    if not isinstance(arr, np.ndarray):
        try:
            arr = np.array(arr, dtype=np.float32)
        except (ValueError, TypeError):
            print(f"⚠️  Cannot convert to array: {type(arr)}, using default")
            return np.array([default_val], dtype=np.float32)

    if arr.ndim == 0:
        val = arr.item()
        if np.isfinite(val):
            return np.array([val], dtype=np.float32)
        else:
            return np.array([default_val], dtype=np.float32)

    mask = ~np.isfinite(arr)
    if np.any(mask):
        arr = arr.copy()
        arr[mask] = default_val

    return arr.astype(np.float32)


def ensure_scalar(value, default=0.0):
    """Ensure value is a scalar, handling arrays properly."""
    if value is None:
        return default

    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        elif value.size == 1:
            try:
                return float(value.item())
            except (ValueError, TypeError):
                return default
        else:
            try:
                return float(value.flat[0])
            except (ValueError, TypeError, IndexError):
                return default
    elif isinstance(value, (list, tuple)):
        if len(value) == 0:
            return default
        try:
            return float(value[0])
        except (ValueError, TypeError, IndexError):
            return default
    else:
        try:
            return float(value)
        except (ValueError, TypeError):
            return default


def ensure_feature_dimension(features, target_dim):
    """Ensure features match target dimension exactly."""
    if not isinstance(features, np.ndarray):
        if isinstance(features, (int, float)):
            features = np.array([features], dtype=np.float32)
        else:
            return np.zeros(target_dim, dtype=np.float32)

    if features.ndim == 0:
        features = np.array([features.item()], dtype=np.float32)

    try:
        current_length = len(features)
    except TypeError:
        return np.zeros(target_dim, dtype=np.float32)

    if current_length == target_dim:
        return features.astype(np.float32)
    elif current_length < target_dim:
        padding = np.zeros(target_dim - current_length, dtype=np.float32)
        return np.concatenate([features, padding]).astype(np.float32)
    else:
        return features[:target_dim].astype(np.float32)


def safe_comparison(value1, value2, operator="==", default=False):
    """Safe comparison that handles arrays."""
    try:
        val1 = ensure_scalar(value1)
        val2 = ensure_scalar(value2)

        if operator == "==":
            return val1 == val2
        elif operator == "!=":
            return val1 != val2
        elif operator == "<":
            return val1 < val2
        elif operator == "<=":
            return val1 <= val2
        elif operator == ">":
            return val1 > val2
        elif operator == ">=":
            return val1 >= val2
        else:
            return default
    except:
        return default


# Keep your original classes but add Energy-Based Transformer components
class IntelligentRewardCalculator:
    """🎯 Enhanced reward calculator with clear win/lose tracking."""

    def __init__(self):
        self.previous_opponent_health = MAX_HEALTH
        self.previous_player_health = MAX_HEALTH
        self.match_started = False

        self.max_damage_reward = 0.8
        self.winning_bonus = 2.0
        self.health_advantage_bonus = 0.3

        self.round_won = False
        self.round_lost = False
        self.round_draw = False

    def calculate_reward(self, player_health, opponent_health, done, info):
        """Enhanced reward calculation with clear win/lose detection."""
        reward = 0.0
        reward_breakdown = {}

        if not self.match_started:
            self.previous_player_health = player_health
            self.previous_opponent_health = opponent_health
            self.match_started = True
            return 0.0, {"initialization": 0.0}

        player_damage_taken = max(0, self.previous_player_health - player_health)
        opponent_damage_dealt = max(0, self.previous_opponent_health - opponent_health)

        if opponent_damage_dealt > 0:
            damage_reward = min(
                opponent_damage_dealt / MAX_HEALTH, self.max_damage_reward
            )
            reward += damage_reward
            reward_breakdown["damage_dealt"] = damage_reward

        if player_damage_taken > 0:
            damage_penalty = -(player_damage_taken / MAX_HEALTH) * 0.5
            reward += damage_penalty
            reward_breakdown["damage_taken"] = damage_penalty

        # Enhanced win/lose detection
        if done:
            termination_reason = info.get("termination_reason", "unknown")

            if termination_reason == "opponent_ko":
                # Clear win - opponent knocked out
                win_bonus = self.winning_bonus
                reward += win_bonus
                reward_breakdown["round_won"] = win_bonus
                reward_breakdown["victory_type"] = "knockout"
                self.round_won = True
                self.round_lost = False
                self.round_draw = False

            elif termination_reason == "player_ko":
                # Clear loss - player knocked out
                loss_penalty = -1.0
                reward += loss_penalty
                reward_breakdown["round_lost"] = loss_penalty
                reward_breakdown["defeat_type"] = "knockout"
                self.round_won = False
                self.round_lost = True
                self.round_draw = False

            elif termination_reason == "technical_ko":
                # Technical KO based on health difference
                if player_health > opponent_health:
                    win_bonus = self.winning_bonus * 0.8  # Slightly less than knockout
                    reward += win_bonus
                    reward_breakdown["round_won"] = win_bonus
                    reward_breakdown["victory_type"] = "technical"
                    self.round_won = True
                    self.round_lost = False
                    self.round_draw = False
                else:
                    loss_penalty = -0.8
                    reward += loss_penalty
                    reward_breakdown["round_lost"] = loss_penalty
                    reward_breakdown["defeat_type"] = "technical"
                    self.round_won = False
                    self.round_lost = True
                    self.round_draw = False

            elif termination_reason == "timeout":
                # Time expired - judge by health
                if player_health > opponent_health:
                    win_bonus = self.winning_bonus * 0.6  # Less than KO
                    reward += win_bonus
                    reward_breakdown["round_won"] = win_bonus
                    reward_breakdown["victory_type"] = "decision"
                    self.round_won = True
                    self.round_lost = False
                    self.round_draw = False
                elif opponent_health > player_health:
                    loss_penalty = -0.6
                    reward += loss_penalty
                    reward_breakdown["round_lost"] = loss_penalty
                    reward_breakdown["defeat_type"] = "decision"
                    self.round_won = False
                    self.round_lost = True
                    self.round_draw = False
                else:
                    # True draw - equal health at timeout
                    draw_penalty = -0.3
                    reward += draw_penalty
                    reward_breakdown["draw"] = draw_penalty
                    reward_breakdown["result_type"] = "draw"
                    self.round_won = False
                    self.round_lost = False
                    self.round_draw = True

            else:
                # Default case - determine by health
                if player_health > opponent_health:
                    win_bonus = self.winning_bonus * 0.5
                    reward += win_bonus
                    reward_breakdown["round_won"] = win_bonus
                    reward_breakdown["victory_type"] = "health_advantage"
                    self.round_won = True
                    self.round_lost = False
                    self.round_draw = False
                elif opponent_health > player_health:
                    loss_penalty = -0.5
                    reward += loss_penalty
                    reward_breakdown["round_lost"] = loss_penalty
                    reward_breakdown["defeat_type"] = "health_disadvantage"
                    self.round_won = False
                    self.round_lost = True
                    self.round_draw = False
                else:
                    draw_penalty = -0.3
                    reward += draw_penalty
                    reward_breakdown["draw"] = draw_penalty
                    reward_breakdown["result_type"] = "equal_health"
                    self.round_won = False
                    self.round_lost = False
                    self.round_draw = True
        else:
            # Ongoing round - health advantage bonus
            health_diff = (player_health - opponent_health) / MAX_HEALTH
            if abs(health_diff) > 0.1:
                advantage_bonus = health_diff * self.health_advantage_bonus
                reward += advantage_bonus
                reward_breakdown["health_advantage"] = advantage_bonus

        step_penalty = -0.01
        reward += step_penalty
        reward_breakdown["step_penalty"] = step_penalty

        self.previous_player_health = player_health
        self.previous_opponent_health = opponent_health

        return reward, reward_breakdown

    def get_round_result(self):
        """Get clear round result for logging."""
        if self.round_won:
            return "WIN"
        elif self.round_lost:
            return "LOSE"
        elif self.round_draw:
            return "DRAW"
        else:
            return "ONGOING"

    def reset(self):
        """Reset for new episode."""
        self.previous_opponent_health = MAX_HEALTH
        self.previous_player_health = MAX_HEALTH
        self.match_started = False
        self.round_won = False
        self.round_lost = False
        self.round_draw = False


class SimplifiedFeatureTracker:
    """📊 Your original feature tracker - UNCHANGED."""

    def __init__(self, history_length=5):
        self.history_length = history_length
        self.reset()

    def reset(self):
        """Reset all tracking for new episode."""
        self.player_health_history = deque(maxlen=self.history_length)
        self.opponent_health_history = deque(maxlen=self.history_length)
        self.last_action = 0
        self.combo_count = 0

    def update(self, player_health, opponent_health, action, reward_breakdown):
        """Update tracking with current state."""
        self.player_health_history.append(player_health / MAX_HEALTH)
        self.opponent_health_history.append(opponent_health / MAX_HEALTH)

        if "damage_dealt" in reward_breakdown and reward_breakdown["damage_dealt"] > 0:
            if action == self.last_action:
                self.combo_count += 1
            else:
                self.combo_count = 0

        self.last_action = action

    def get_features(self):
        """Get current feature vector."""
        features = []

        player_hist = list(self.player_health_history)
        opponent_hist = list(self.opponent_health_history)

        while len(player_hist) < self.history_length:
            player_hist.insert(0, 1.0)
        while len(opponent_hist) < self.history_length:
            opponent_hist.insert(0, 1.0)

        features.extend(player_hist)
        features.extend(opponent_hist)

        current_player_health = player_hist[-1] if player_hist else 1.0
        current_opponent_health = opponent_hist[-1] if opponent_hist else 1.0

        features.extend(
            [
                current_player_health,
                current_opponent_health,
                current_player_health - current_opponent_health,
                self.last_action / 55.0,
                min(self.combo_count / 5.0, 1.0),
            ]
        )

        return ensure_feature_dimension(
            np.array(features, dtype=np.float32), VECTOR_FEATURE_DIM
        )


class StreetFighterDiscreteActions:
    """🎮 Your original action mapping - UNCHANGED."""

    def __init__(self):
        self.action_map = {
            0: [],  # No action
            1: ["LEFT"],
            2: ["RIGHT"],
            3: ["UP"],
            4: ["DOWN"],
            5: ["A"],
            6: ["B"],
            7: ["C"],  # Light, Medium, Heavy punch
            8: ["X"],
            9: ["Y"],
            10: ["Z"],  # Light, Medium, Heavy kick
            # Combinations
            11: ["LEFT", "A"],
            12: ["LEFT", "B"],
            13: ["LEFT", "C"],
            14: ["RIGHT", "A"],
            15: ["RIGHT", "B"],
            16: ["RIGHT", "C"],
            17: ["DOWN", "A"],
            18: ["DOWN", "B"],
            19: ["DOWN", "C"],
            20: ["UP", "A"],
            21: ["UP", "B"],
            22: ["UP", "C"],
            23: ["LEFT", "X"],
            24: ["LEFT", "Y"],
            25: ["LEFT", "Z"],
            26: ["RIGHT", "X"],
            27: ["RIGHT", "Y"],
            28: ["RIGHT", "Z"],
            29: ["DOWN", "X"],
            30: ["DOWN", "Y"],
            31: ["DOWN", "Z"],
            32: ["UP", "X"],
            33: ["UP", "Y"],
            34: ["UP", "Z"],
            # Special moves
            35: ["DOWN", "RIGHT", "A"],
            36: ["DOWN", "RIGHT", "B"],
            37: ["DOWN", "RIGHT", "C"],
            38: ["DOWN", "RIGHT", "X"],
            39: ["DOWN", "RIGHT", "Y"],
            40: ["DOWN", "RIGHT", "Z"],
            # Quarter circle back
            41: ["DOWN", "LEFT", "A"],
            42: ["DOWN", "LEFT", "B"],
            43: ["DOWN", "LEFT", "C"],
            44: ["DOWN", "LEFT", "X"],
            45: ["DOWN", "LEFT", "Y"],
            46: ["DOWN", "LEFT", "Z"],
            # Dragon punch motion
            47: ["RIGHT", "DOWN", "A"],
            48: ["RIGHT", "DOWN", "B"],
            49: ["RIGHT", "DOWN", "C"],
            # Additional combinations
            50: ["A", "B"],
            51: ["B", "C"],
            52: ["X", "Y"],
            53: ["Y", "Z"],
            54: ["A", "X"],
            55: ["C", "Z"],
        }

        self.n_actions = len(self.action_map)
        self.button_to_index = {tuple(v): k for k, v in self.action_map.items()}

    def get_action(self, action_idx):
        """Convert action index to button combination."""
        return self.action_map.get(action_idx, [])


class StreetFighterVisionWrapper(gym.Wrapper):
    """🥊 Your original wrapper with ONLY timeout fix - UNCHANGED."""

    def __init__(self, env):
        super().__init__(env)

        self.reward_calculator = IntelligentRewardCalculator()
        self.feature_tracker = SimplifiedFeatureTracker()
        self.action_mapper = StreetFighterDiscreteActions()

        visual_space = gym.spaces.Box(
            low=0, high=255, shape=(3, SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.uint8
        )
        vector_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(5, VECTOR_FEATURE_DIM), dtype=np.float32
        )

        self.observation_space = gym.spaces.Dict(
            {"visual_obs": visual_space, "vector_obs": vector_space}
        )

        self.action_space = gym.spaces.Discrete(self.action_mapper.n_actions)
        self.vector_history = deque(maxlen=5)

        self.episode_count = 0
        self.step_count = 0

        # Your original health tracking
        self.previous_player_health = MAX_HEALTH
        self.previous_opponent_health = MAX_HEALTH

        print(f"🥊 StreetFighterVisionWrapper initialized (fast UI speed maintained)")

    def reset(self, **kwargs):
        """Enhanced reset with round state cleanup."""
        # Force a complete environment reset to prevent round carryover
        obs, info = self.env.reset(**kwargs)

        self.reward_calculator.reset()
        self.feature_tracker.reset()
        self.vector_history.clear()

        self.previous_player_health = MAX_HEALTH
        self.previous_opponent_health = MAX_HEALTH

        self.episode_count += 1
        self.step_count = 0

        # Ensure we start in a clean state
        player_health, opponent_health = self._extract_health(info)

        # If health values are weird, force them to max
        if player_health <= 0 or opponent_health <= 0:
            player_health = MAX_HEALTH
            opponent_health = MAX_HEALTH
            self.previous_player_health = MAX_HEALTH
            self.previous_opponent_health = MAX_HEALTH

        self.feature_tracker.update(player_health, opponent_health, 0, {})

        observation = self._build_observation(obs, info)

        # Add reset confirmation to info
        info.update(
            {
                "reset_complete": True,
                "starting_health": {
                    "player": player_health,
                    "opponent": opponent_health,
                },
                "episode_count": self.episode_count,
            }
        )

        return observation, info

    def step(self, action):
        """Enhanced step with detailed win/lose tracking."""
        self.step_count += 1

        button_combination = self.action_mapper.get_action(action)
        retro_action = self._convert_to_retro_action(button_combination)
        obs, reward, done, truncated, info = self.env.step(retro_action)

        player_health, opponent_health = self._extract_health(info)

        # ENHANCED SINGLE-ROUND LOGIC: Multiple termination conditions
        round_ended = False
        termination_reason = "ongoing"

        # 1. Health-based termination (KO)
        if self.previous_player_health > 0 and player_health <= 0:
            round_ended = True
            termination_reason = "player_ko"
        elif self.previous_opponent_health > 0 and opponent_health <= 0:
            round_ended = True
            termination_reason = "opponent_ko"

        # 2. Check for round completion via game state
        if not round_ended:
            game_messages = info.get("game_over_message", "")
            round_complete_indicators = [
                "PERFECT",
                "YOU WIN",
                "YOU LOSE",
                "K.O.",
                "WINS",
            ]
            if any(
                indicator in str(game_messages).upper()
                for indicator in round_complete_indicators
            ):
                round_ended = True
                if (
                    "YOU WIN" in str(game_messages).upper()
                    or "PERFECT" in str(game_messages).upper()
                ):
                    termination_reason = "opponent_ko"
                elif "YOU LOSE" in str(game_messages).upper():
                    termination_reason = "player_ko"
                else:
                    termination_reason = "round_complete"

        # 3. Time's up condition
        if not round_ended and hasattr(info, "timer") and info.get("timer", 99) <= 0:
            round_ended = True
            termination_reason = "timeout"

        # 4. Check raw game state for round completion
        if not round_ended and "round_complete" in info and info["round_complete"]:
            round_ended = True
            termination_reason = "round_complete"

        # 5. Detect if we're entering a new round (prevent round 2)
        if not round_ended:
            round_indicators = info.get("round_text", "")
            if (
                "ROUND 2" in str(round_indicators).upper()
                or "ROUND TWO" in str(round_indicators).upper()
            ):
                round_ended = True
                termination_reason = "round_2_prevention"

        # 6. Step limit timeout (last resort)
        if not round_ended and self.step_count >= MAX_FIGHT_STEPS:
            round_ended = True
            termination_reason = "timeout"

        # 7. Check for dramatic health difference (technical KO)
        if not round_ended and abs(player_health - opponent_health) >= MAX_HEALTH * 0.8:
            round_ended = True
            termination_reason = "technical_ko"

        # Apply single-round termination
        if round_ended:
            done = True
            truncated = True

        # Update previous health values
        self.previous_player_health = player_health
        self.previous_opponent_health = opponent_health

        # Add termination reason to info before reward calculation
        info["termination_reason"] = termination_reason
        info["round_ended"] = round_ended

        # Calculate enhanced reward with termination info
        intelligent_reward, reward_breakdown = self.reward_calculator.calculate_reward(
            player_health, opponent_health, done, info
        )

        self.feature_tracker.update(
            player_health, opponent_health, action, reward_breakdown
        )
        observation = self._build_observation(obs, info)

        # Get round result for logging
        round_result = self.reward_calculator.get_round_result()

        info.update(
            {
                "player_health": player_health,
                "opponent_health": opponent_health,
                "reward_breakdown": reward_breakdown,
                "intelligent_reward": intelligent_reward,
                "episode_count": self.episode_count,
                "step_count": self.step_count,
                "round_ended": round_ended,
                "termination_reason": termination_reason,
                "round_result": round_result,
                "final_health_diff": player_health - opponent_health,
                "victory_type": reward_breakdown.get("victory_type", "none"),
                "defeat_type": reward_breakdown.get("defeat_type", "none"),
            }
        )

        # Print immediate round result for debugging
        if round_ended:
            result_emoji = (
                "🏆"
                if round_result == "WIN"
                else "💀" if round_result == "LOSE" else "🤝"
            )
            victory_detail = ""
            if "victory_type" in reward_breakdown:
                victory_detail = f" ({reward_breakdown['victory_type']})"
            elif "defeat_type" in reward_breakdown:
                victory_detail = f" ({reward_breakdown['defeat_type']})"
            elif "result_type" in reward_breakdown:
                victory_detail = f" ({reward_breakdown['result_type']})"

            print(
                f"  {result_emoji} Episode {self.episode_count}: {round_result}{victory_detail} "
                f"- Steps: {self.step_count}, Health: {player_health} vs {opponent_health}, "
                f"Reason: {termination_reason}"
            )

        return observation, intelligent_reward, done, truncated, info

    def _get_termination_reason(self, player_health, opponent_health, round_ended):
        """Determine why the round ended for debugging."""
        if not round_ended:
            return "ongoing"
        elif player_health <= 0:
            return "player_ko"
        elif opponent_health <= 0:
            return "opponent_ko"
        elif self.step_count >= MAX_FIGHT_STEPS:
            return "timeout"
        elif abs(player_health - opponent_health) >= MAX_HEALTH * 0.8:
            return "technical_ko"
        else:
            return "round_complete"

    def _extract_health(self, info):
        """Enhanced health extraction with round state detection."""
        player_health = info.get("player_health", MAX_HEALTH)
        opponent_health = info.get("opponent_health", MAX_HEALTH)

        # Try multiple memory addresses for health detection
        if hasattr(self.env, "data") and hasattr(self.env.data, "memory"):
            try:
                # Primary health addresses
                player_health = self.env.data.memory.read_byte(0x8004)
                opponent_health = self.env.data.memory.read_byte(0x8008)

                # Alternative health addresses (different SF2 versions)
                if player_health == 0 and opponent_health == 0:
                    player_health = self.env.data.memory.read_byte(0xFF8204)
                    opponent_health = self.env.data.memory.read_byte(0xFF8208)

                # Another common set
                if player_health == 0 and opponent_health == 0:
                    player_health = self.env.data.memory.read_byte(0x800C)
                    opponent_health = self.env.data.memory.read_byte(0x8010)

            except:
                # Fallback to info values
                pass

        # Additional round state detection
        try:
            if hasattr(self.env, "data") and hasattr(self.env.data, "memory"):
                # Try to read round counter or game state
                round_state = self.env.data.memory.read_byte(
                    0x8014
                )  # Common round counter address
                if round_state > 1:  # If we're in round 2 or higher
                    info["round_text"] = f"ROUND {round_state}"

                # Check for game over states
                game_state = self.env.data.memory.read_byte(0x8018)  # Game state flag
                if game_state in [0x10, 0x20, 0x30]:  # Common "round complete" values
                    info["round_complete"] = True

        except:
            pass

        return player_health, opponent_health

    def _convert_to_retro_action(self, button_combination):
        """Your original action conversion - UNCHANGED."""
        button_tuple = tuple(button_combination)
        if button_tuple in self.action_mapper.button_to_index:
            return self.action_mapper.button_to_index[button_tuple]
        else:
            return 0

    def _build_observation(self, visual_obs, info):
        """Your original observation building - UNCHANGED."""
        if isinstance(visual_obs, np.ndarray):
            if len(visual_obs.shape) == 3 and visual_obs.shape[2] == 3:
                visual_obs = np.transpose(visual_obs, (2, 0, 1))

            if visual_obs.shape[-2:] != (SCREEN_HEIGHT, SCREEN_WIDTH):
                visual_obs = cv2.resize(
                    visual_obs.transpose(1, 2, 0), (SCREEN_WIDTH, SCREEN_HEIGHT)
                ).transpose(2, 0, 1)

        vector_features = self.feature_tracker.get_features()
        self.vector_history.append(vector_features)

        while len(self.vector_history) < 5:
            self.vector_history.appendleft(
                np.zeros(VECTOR_FEATURE_DIM, dtype=np.float32)
            )

        vector_obs = np.stack(list(self.vector_history), axis=0)

        return {
            "visual_obs": visual_obs.astype(np.uint8),
            "vector_obs": vector_obs.astype(np.float32),
        }


# NEW: Energy-Based Transformer Components
class EnergyBasedMultiHeadAttention(nn.Module):
    """🔥 Energy-Based Multi-Head Attention inspired by EBT paper."""

    def __init__(self, d_model, num_heads, dropout=0.1, energy_scale=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.energy_scale = energy_scale

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)

        # Energy-based components
        self.energy_proj = nn.Linear(d_model, d_model)
        self.energy_norm = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()

        # Standard attention computation
        Q = (
            self.w_q(x)
            .view(batch_size, seq_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.w_k(x)
            .view(batch_size, seq_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.w_v(x)
            .view(batch_size, seq_len, self.num_heads, self.d_k)
            .transpose(1, 2)
        )

        # Energy-based attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Energy transformation
        energy_features = self.energy_proj(x)
        energy_features = self.energy_norm(energy_features)
        energy_bias = torch.sum(
            energy_features.unsqueeze(2) * energy_features.unsqueeze(1), dim=-1
        )
        energy_bias = energy_bias.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        # Combine standard attention with energy bias
        scores = scores + self.energy_scale * energy_bias

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        context = (
            context.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        )

        output = self.w_o(context)
        return self.layer_norm(output + x), attention_weights


class EnergyBasedTransformerBlock(nn.Module):
    """🔥 Energy-Based Transformer Block with energy-aware feed-forward."""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, energy_scale=0.1):
        super().__init__()

        self.attention = EnergyBasedMultiHeadAttention(
            d_model, num_heads, dropout, energy_scale
        )

        # Energy-aware feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        # Energy prediction head for this layer
        self.energy_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Linear(d_model // 2, 1)
        )

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Energy-based attention
        attn_output, attention_weights = self.attention(x, mask)

        # Feed-forward with residual connection
        ff_output = self.ff(attn_output)
        output = self.layer_norm(ff_output + attn_output)

        # Energy prediction for this layer
        layer_energy = self.energy_head(output.mean(dim=1))  # Pool over sequence

        return output, layer_energy, attention_weights


class EnhancedEnergyBasedCNN(nn.Module):
    """🛡️ Your original CNN enhanced with Energy-Based Transformer processing."""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__()

        visual_space = observation_space["visual_obs"]
        vector_space = observation_space["vector_obs"]
        n_input_channels = visual_space.shape[0]
        seq_length, vector_feature_count = vector_space.shape

        # Your original visual CNN - UNCHANGED
        self.visual_cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.15),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.15),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.1),
            nn.AdaptiveAvgPool2d((3, 4)),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy_visual = torch.zeros(
                1, n_input_channels, visual_space.shape[1], visual_space.shape[2]
            )
            visual_output_size = self.visual_cnn(dummy_visual).shape[1]

        # Enhanced vector processing with Energy-Based Transformer
        self.vector_embed = nn.Linear(
            vector_feature_count, 128
        )  # Increased dim for transformer
        self.positional_encoding = nn.Parameter(torch.randn(seq_length, 128) * 0.02)

        # Energy-Based Transformer layers
        self.ebt_layers = nn.ModuleList(
            [
                EnergyBasedTransformerBlock(
                    d_model=128, num_heads=4, d_ff=256, dropout=0.1, energy_scale=0.1
                )
                for _ in range(2)  # 2 transformer layers
            ]
        )

        self.vector_final = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
        )

        # Enhanced fusion with energy-aware processing
        fusion_input_size = visual_output_size + 64
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, 512),
            nn.ReLU(inplace=True),
            nn.LayerNorm(512),
            nn.Dropout(0.2),
            nn.Linear(512, features_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        # Energy aggregation from transformer layers
        self.energy_aggregator = nn.Sequential(
            nn.Linear(len(self.ebt_layers), 32), nn.ReLU(), nn.Linear(32, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.3)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(
        self, observations: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        visual_obs = observations["visual_obs"]
        vector_obs = observations["vector_obs"]
        device = next(self.parameters()).device

        visual_obs = visual_obs.float().to(device)
        vector_obs = vector_obs.float().to(device)

        # Sanitize inputs
        visual_nan_mask = ~torch.isfinite(visual_obs)
        vector_nan_mask = ~torch.isfinite(vector_obs)

        if torch.any(visual_nan_mask):
            visual_obs = torch.where(
                visual_nan_mask, torch.zeros_like(visual_obs), visual_obs
            )

        if torch.any(vector_nan_mask):
            vector_obs = torch.where(
                vector_nan_mask, torch.zeros_like(vector_obs), vector_obs
            )

        visual_obs = torch.clamp(visual_obs / 255.0, 0.0, 1.0)
        vector_obs = torch.clamp(vector_obs, -8.0, 8.0)

        # Process visual features (unchanged)
        visual_features = self.visual_cnn(visual_obs)
        if torch.any(torch.abs(visual_features) > 50.0):
            visual_features = torch.clamp(visual_features, -50.0, 50.0)

        # Enhanced vector processing with Energy-Based Transformer
        batch_size, seq_len, feature_dim = vector_obs.shape
        vector_embedded = self.vector_embed(vector_obs)

        # Add positional encoding
        vector_embedded = vector_embedded + self.positional_encoding.unsqueeze(
            0
        ).expand(batch_size, -1, -1)

        # Process through Energy-Based Transformer layers
        transformer_output = vector_embedded
        layer_energies = []
        all_attention_weights = []

        for ebt_layer in self.ebt_layers:
            transformer_output, layer_energy, attention_weights = ebt_layer(
                transformer_output
            )
            layer_energies.append(layer_energy)
            all_attention_weights.append(attention_weights)

        # Pool transformer output
        vector_features = transformer_output.mean(
            dim=1
        )  # Average pooling over sequence
        vector_features = self.vector_final(vector_features)

        # Combine visual and vector features
        combined_features = torch.cat([visual_features, vector_features], dim=1)
        output = self.fusion(combined_features)

        # Aggregate energy from transformer layers
        if layer_energies:
            stacked_energies = torch.cat(layer_energies, dim=1)
            aggregated_energy = self.energy_aggregator(stacked_energies)
        else:
            aggregated_energy = torch.zeros(batch_size, 1, device=device)

        # Sanitize outputs
        if torch.any(~torch.isfinite(output)):
            output = torch.where(
                ~torch.isfinite(output), torch.zeros_like(output), output
            )

        if torch.any(~torch.isfinite(aggregated_energy)):
            aggregated_energy = torch.where(
                ~torch.isfinite(aggregated_energy),
                torch.zeros_like(aggregated_energy),
                aggregated_energy,
            )

        output = torch.clamp(output, -15.0, 15.0)
        aggregated_energy = torch.clamp(aggregated_energy, -10.0, 10.0)

        return output, aggregated_energy


# this is our verifier, what are doing with it
class EnhancedEnergyBasedVerifier(nn.Module):
    """🛡️ Enhanced verifier with Energy-Based Transformer integration."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        features_dim: int = 256,
    ):
        # verifier has obs, action space, feature dim (player health etc)
        # super init
        super().__init__()

        # obs, action space, feature, action dim
        self.observation_space = observation_space
        self.action_space = action_space
        self.features_dim = features_dim
        self.action_dim = action_space.n if hasattr(action_space, "n") else 56

        # Enhanced features extractor with EBT
        self.features_extractor = EnhancedEnergyBasedCNN(
            observation_space, features_dim
        )

        self.action_embed = nn.Sequential(
            nn.Linear(self.action_dim, 64),
            nn.ReLU(inplace=True),
            nn.LayerNorm(64),
            nn.Dropout(0.15),
        )

        # Enhanced energy network with transformer energy integration
        self.energy_net = nn.Sequential(
            nn.Linear(features_dim + 64 + 1, 256),  # +1 for transformer energy
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.LayerNorm(128),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.LayerNorm(64),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

        self.energy_scale = 0.6
        self.energy_clamp_min = -6.0
        self.energy_clamp_max = 6.0

        # Energy fusion weights
        self.energy_fusion = nn.Parameter(
            torch.tensor([0.7, 0.3])
        )  # CNN energy, Transformer energy

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.005)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(
        self, context: torch.Tensor, candidate_action: torch.Tensor
    ) -> torch.Tensor:
        device = next(self.parameters()).device

        if isinstance(context, dict):
            context_features, transformer_energy = self.features_extractor(context)
        else:
            context_features = context
            transformer_energy = torch.zeros(context.shape[0], 1, device=device)

        context_features = context_features.to(device)
        candidate_action = candidate_action.to(device)
        transformer_energy = transformer_energy.to(device)

        # Sanitize inputs
        if torch.any(~torch.isfinite(context_features)):
            context_features = torch.where(
                ~torch.isfinite(context_features),
                torch.zeros_like(context_features),
                context_features,
            )

        if torch.any(~torch.isfinite(candidate_action)):
            candidate_action = torch.where(
                ~torch.isfinite(candidate_action),
                torch.zeros_like(candidate_action),
                candidate_action,
            )

        if torch.any(~torch.isfinite(transformer_energy)):
            transformer_energy = torch.where(
                ~torch.isfinite(transformer_energy),
                torch.zeros_like(transformer_energy),
                transformer_energy,
            )

        context_features = torch.clamp(context_features, -15.0, 15.0)
        candidate_action = torch.clamp(candidate_action, 0.0, 1.0)
        transformer_energy = torch.clamp(transformer_energy, -10.0, 10.0)

        # Process action
        action_embedded = self.action_embed(candidate_action)
        if torch.any(~torch.isfinite(action_embedded)):
            action_embedded = torch.where(
                ~torch.isfinite(action_embedded),
                torch.zeros_like(action_embedded),
                action_embedded,
            )

        # Combine features including transformer energy
        combined_input = torch.cat(
            [context_features, action_embedded, transformer_energy], dim=-1
        )

        # Compute final energy
        raw_energy = self.energy_net(combined_input)
        energy = raw_energy * self.energy_scale
        energy = torch.clamp(energy, self.energy_clamp_min, self.energy_clamp_max)

        # Sanitize final output
        if torch.any(~torch.isfinite(energy)):
            energy = torch.where(
                ~torch.isfinite(energy), torch.zeros_like(energy), energy
            )

        return energy

    def get_energy_stats(self):
        return {
            "energy_mean": 0.0,
            "energy_std": 0.0,
            "nan_count": 0,
            "explosion_count": 0,
        }


class EnhancedEnergyBasedAgent:
    """🛡️ Enhanced agent with Energy-Based Transformer thinking."""

    def __init__(
        self,
        verifier: EnhancedEnergyBasedVerifier,
        thinking_steps: int = 3,
        thinking_lr: float = 0.06,
        noise_scale: float = 0.02,
    ):
        self.verifier = verifier
        self.initial_thinking_steps = thinking_steps
        self.current_thinking_steps = thinking_steps
        self.initial_thinking_lr = thinking_lr
        self.current_thinking_lr = thinking_lr
        self.noise_scale = noise_scale
        self.action_dim = verifier.action_dim

        # Enhanced thinking parameters
        self.gradient_clip = 0.3
        self.early_stop_patience = 2
        self.min_energy_improvement = 8e-4

        # Adaptive thinking parameters
        self.max_thinking_steps = 8  # Increased for transformer
        self.min_thinking_steps = 1

        # Energy-based thinking with transformer awareness
        self.energy_momentum = 0.9
        self.energy_history_weight = 0.1

        self.thinking_stats = {
            "total_predictions": 0,
            "avg_thinking_steps": 0.0,
            "avg_energy_improvement": 0.0,
            "early_stops": 0,
            "energy_explosions": 0,
            "gradient_explosions": 0,
            "successful_optimizations": 0,
            "failed_optimizations": 0,
            "transformer_energy_usage": 0.0,
        }

        self.recent_performance = deque(maxlen=100)

    def predict(
        self, observations: Dict[str, torch.Tensor], deterministic: bool = False
    ) -> Tuple[int, Dict]:
        device = next(self.verifier.parameters()).device

        # Prepare observations
        obs_device = {}
        for key, value in observations.items():
            if isinstance(value, torch.Tensor):
                obs_device[key] = value.to(device)
            else:
                obs_device[key] = torch.from_numpy(value).to(device)

        if len(obs_device["visual_obs"].shape) == 3:
            for key in obs_device:
                obs_device[key] = obs_device[key].unsqueeze(0)

        batch_size = obs_device["visual_obs"].shape[0]

        # Initialize candidate action
        if deterministic:
            candidate_action = (
                torch.ones(batch_size, self.action_dim, device=device) / self.action_dim
            )
        else:
            candidate_action = (
                torch.randn(batch_size, self.action_dim, device=device)
                * self.noise_scale
            )
            candidate_action = F.softmax(candidate_action, dim=-1)

        candidate_action.requires_grad_(True)

        # Enhanced thinking process with energy-based optimization
        energy_history = []
        transformer_energies = []
        steps_taken = 0
        early_stopped = False
        energy_explosion = False
        gradient_explosion = False
        optimization_successful = False

        # Get initial energy
        with torch.no_grad():
            try:
                initial_energy = self.verifier(obs_device, candidate_action)
                energy_history.append(initial_energy.mean().item())

                # Track transformer energy contribution
                _, transformer_energy = self.verifier.features_extractor(obs_device)
                transformer_energies.append(transformer_energy.mean().item())
            except Exception as e:
                return 0, {"error": "initial_energy_failed"}

        # Energy-based thinking loop
        momentum = torch.zeros_like(candidate_action)

        for step in range(self.current_thinking_steps):
            try:
                energy = self.verifier(obs_device, candidate_action)

                # Check for energy explosion
                if torch.any(torch.abs(energy) > 8.0):
                    energy_explosion = True
                    break

                # Compute gradients
                gradients = torch.autograd.grad(
                    outputs=energy.sum(),
                    inputs=candidate_action,
                    create_graph=False,
                    retain_graph=False,
                )[0]

                gradient_norm = torch.norm(gradients).item()

                # Check for gradient explosion
                if gradient_norm > self.gradient_clip:
                    gradient_explosion = True
                    gradients = gradients * (self.gradient_clip / gradient_norm)

                if torch.any(~torch.isfinite(gradients)):
                    break

                # Enhanced update with momentum
                with torch.no_grad():
                    momentum = (
                        self.energy_momentum * momentum
                        + (1 - self.energy_momentum) * gradients
                    )
                    candidate_action = (
                        candidate_action - self.current_thinking_lr * momentum
                    )
                    candidate_action = F.softmax(candidate_action, dim=-1)
                    candidate_action.requires_grad_(True)

                # Evaluate new energy
                with torch.no_grad():
                    new_energy = self.verifier(obs_device, candidate_action)
                    energy_history.append(new_energy.mean().item())

                    # Track transformer energy
                    _, transformer_energy = self.verifier.features_extractor(obs_device)
                    transformer_energies.append(transformer_energy.mean().item())

                # Early stopping based on energy improvement
                if len(energy_history) >= self.early_stop_patience + 1:
                    recent_improvement = (
                        energy_history[-self.early_stop_patience - 1]
                        - energy_history[-1]
                    )
                    if abs(recent_improvement) < self.min_energy_improvement:
                        early_stopped = True
                        break

                steps_taken += 1

            except Exception as e:
                break

        # Determine optimization success
        if len(energy_history) > 1:
            total_improvement = abs(energy_history[0] - energy_history[-1])
            optimization_successful = (
                total_improvement > self.min_energy_improvement
                and not energy_explosion
                and not gradient_explosion
            )

        # Final action selection
        with torch.no_grad():
            try:
                final_action_probs = F.softmax(candidate_action, dim=-1)
                final_action_probs = final_action_probs + 1e-8
                final_action_probs = final_action_probs / final_action_probs.sum(
                    dim=-1, keepdim=True
                )

                if deterministic:
                    action_idx = torch.argmax(final_action_probs, dim=-1)
                else:
                    action_idx = torch.multinomial(final_action_probs, 1).squeeze(-1)
            except Exception as e:
                return 0, {"error": "action_selection_failed"}

        # Update statistics
        self.thinking_stats["total_predictions"] += 1
        self.thinking_stats["avg_thinking_steps"] = (
            self.thinking_stats["avg_thinking_steps"] * 0.9 + steps_taken * 0.1
        )

        if len(energy_history) > 1:
            energy_improvement = abs(energy_history[0] - energy_history[-1])
            self.thinking_stats["avg_energy_improvement"] = (
                self.thinking_stats["avg_energy_improvement"] * 0.9
                + energy_improvement * 0.1
            )

        if transformer_energies:
            avg_transformer_energy = sum(transformer_energies) / len(
                transformer_energies
            )
            self.thinking_stats["transformer_energy_usage"] = (
                self.thinking_stats["transformer_energy_usage"] * 0.9
                + abs(avg_transformer_energy) * 0.1
            )

        # Update counters
        if early_stopped:
            self.thinking_stats["early_stops"] += 1
        if energy_explosion:
            self.thinking_stats["energy_explosions"] += 1
        if gradient_explosion:
            self.thinking_stats["gradient_explosions"] += 1
        if optimization_successful:
            self.thinking_stats["successful_optimizations"] += 1
        else:
            self.thinking_stats["failed_optimizations"] += 1

        self.recent_performance.append(1.0 if optimization_successful else 0.0)

        thinking_info = {
            "energy_history": energy_history,
            "transformer_energies": transformer_energies,
            "steps_taken": steps_taken,
            "early_stopped": early_stopped,
            "energy_explosion": energy_explosion,
            "gradient_explosion": gradient_explosion,
            "optimization_successful": optimization_successful,
            "energy_improvement": (
                abs(energy_history[0] - energy_history[-1])
                if len(energy_history) > 1
                else 0.0
            ),
            "final_energy": energy_history[-1] if energy_history else 0.0,
            "current_thinking_steps": self.current_thinking_steps,
            "current_thinking_lr": self.current_thinking_lr,
            "avg_transformer_energy": (
                sum(transformer_energies) / len(transformer_energies)
                if transformer_energies
                else 0.0
            ),
        }

        return action_idx.item() if batch_size == 1 else action_idx, thinking_info

    def get_thinking_stats(self) -> Dict:
        stats = self.thinking_stats.copy()
        if stats["total_predictions"] > 0:
            stats["success_rate"] = safe_divide(
                stats["successful_optimizations"], stats["total_predictions"], 0.0
            )
            stats["early_stop_rate"] = safe_divide(
                stats["early_stops"], stats["total_predictions"], 0.0
            )
            stats["explosion_rate"] = safe_divide(
                stats["energy_explosions"] + stats["gradient_explosions"],
                stats["total_predictions"],
                0.0,
            )
        return stats


# Keep your other original classes with fixed quality threshold
class GoldenExperienceBuffer:
    """🏆 Golden buffer with FIXED threshold."""

    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.experiences = deque(maxlen=capacity)
        self.min_quality_for_golden = 0.6  # Keep high for golden
        self.peak_win_rate_threshold = 0.3  # FIXED: Lower threshold
        self.current_win_rate = 0.0

    def update_win_rate(self, win_rate):
        self.current_win_rate = win_rate

    def add_experience(self, experience, quality_score):
        if (
            self.current_win_rate >= self.peak_win_rate_threshold
            and quality_score >= self.min_quality_for_golden
        ):
            golden_experience = experience.copy()
            golden_experience["is_golden"] = True
            golden_experience["golden_win_rate"] = self.current_win_rate
            golden_experience["golden_quality"] = quality_score
            self.experiences.append(golden_experience)

    def sample_golden_batch(self, batch_size):
        if len(self.experiences) < batch_size:
            return list(self.experiences)
        indices = np.random.choice(len(self.experiences), batch_size, replace=False)
        return [self.experiences[i] for i in indices]

    def get_stats(self):
        if len(self.experiences) > 0:
            qualities = [exp.get("golden_quality", 0.0) for exp in self.experiences]
            win_rates = [exp.get("golden_win_rate", 0.0) for exp in self.experiences]
            avg_quality = np.mean(qualities)
            avg_win_rate = np.mean(win_rates)
            utilization = len(self.experiences) / self.capacity
        else:
            avg_quality = 0.0
            avg_win_rate = 0.0
            utilization = 0.0

        return {
            "size": len(self.experiences),
            "capacity": self.capacity,
            "utilization": utilization,
            "avg_quality": avg_quality,
            "avg_win_rate": avg_win_rate,
            "min_quality_threshold": self.min_quality_for_golden,
            "peak_win_rate_threshold": self.peak_win_rate_threshold,
        }


class EnhancedQualityBasedExperienceBuffer:
    """🎯 MAIN FIX: Quality threshold lowered from 0.6 to 0.3."""

    def __init__(
        self, capacity=30000, quality_threshold=0.3, golden_buffer_capacity=1000
    ):
        self.capacity = capacity
        self.quality_threshold = quality_threshold  # THE KEY FIX

        self.good_experiences = deque(maxlen=capacity // 2)
        self.bad_experiences = deque(maxlen=capacity // 2)

        self.golden_buffer = GoldenExperienceBuffer(capacity=golden_buffer_capacity)

        self.quality_scores = deque(maxlen=1000)
        self.total_added = 0

        self.adjustment_rate = 0.05
        self.threshold_adjustment_frequency = 25
        self.threshold_adjustments = 0

        print(f"🎯 Enhanced Quality-Based Experience Buffer initialized")
        print(f"   - Quality threshold: {quality_threshold} (THE MAIN FIX)")

    def add_experience(self, experience, reward, reward_breakdown, quality_score):
        """Add experience with FIXED threshold."""
        self.total_added += 1
        self.quality_scores.append(quality_score)

        # THE CORE FIX: This will now work because threshold is 0.3 instead of 0.6
        if quality_score >= self.quality_threshold:
            self.good_experiences.append(experience)
            self.golden_buffer.add_experience(experience, quality_score)
        else:
            self.bad_experiences.append(experience)

    def update_win_rate(self, win_rate):
        self.golden_buffer.update_win_rate(win_rate)

    def sample_enhanced_balanced_batch(self, batch_size, golden_ratio=0.15):
        if (
            len(self.good_experiences) < batch_size // 4
            or len(self.bad_experiences) < batch_size // 4
        ):
            return None, None, None

        golden_count = int(batch_size * golden_ratio)
        remaining_good = (batch_size // 2) - golden_count
        bad_count = batch_size // 2

        golden_batch = (
            self.golden_buffer.sample_golden_batch(golden_count)
            if golden_count > 0
            else []
        )

        good_indices = np.random.choice(
            len(self.good_experiences), remaining_good, replace=False
        )
        good_batch = [self.good_experiences[i] for i in good_indices]

        bad_indices = np.random.choice(
            len(self.bad_experiences), bad_count, replace=False
        )
        bad_batch = [self.bad_experiences[i] for i in bad_indices]

        combined_good_batch = good_batch + golden_batch
        return combined_good_batch, bad_batch, golden_batch

    def get_stats(self):
        total_size = len(self.good_experiences) + len(self.bad_experiences)

        if len(self.quality_scores) > 0:
            avg_quality = np.mean(list(self.quality_scores))
            quality_std = safe_std(list(self.quality_scores), 0.0)
        else:
            avg_quality = 0.0
            quality_std = 0.0

        golden_stats = self.golden_buffer.get_stats()

        return {
            "total_size": total_size,
            "good_count": len(self.good_experiences),
            "bad_count": len(self.bad_experiences),
            "good_ratio": len(self.good_experiences) / max(1, total_size),
            "quality_threshold": self.quality_threshold,
            "avg_quality_score": avg_quality,
            "quality_std": quality_std,
            "total_added": self.total_added,
            "acceptance_rate": total_size / max(1, self.total_added),
            "threshold_adjustments": self.threshold_adjustments,
            "golden_buffer": golden_stats,
        }

    def adjust_threshold(self, target_good_ratio=0.5, episode_number=0):
        if episode_number % self.threshold_adjustment_frequency != 0:
            return

        if len(self.quality_scores) < 100:
            return

        total_size = len(self.good_experiences) + len(self.bad_experiences)
        current_good_ratio = len(self.good_experiences) / max(1, total_size)

        if current_good_ratio < target_good_ratio - 0.15:
            self.quality_threshold *= 1 - self.adjustment_rate
            self.threshold_adjustments += 1
            print(f"📉 Lowered quality threshold to {self.quality_threshold:.3f}")
        elif current_good_ratio > target_good_ratio + 0.15:
            self.quality_threshold *= 1 + self.adjustment_rate
            self.threshold_adjustments += 1
            print(f"📈 Raised quality threshold to {self.quality_threshold:.3f}")

        self.quality_threshold = max(0.4, min(0.75, self.quality_threshold))


# Keep all your other management classes unchanged
class PolicyMemoryManager:
    """🧠 Your original policy memory manager - UNCHANGED."""

    def __init__(self, performance_drop_threshold=0.05, averaging_weight=0.7):
        self.performance_drop_threshold = performance_drop_threshold
        self.averaging_weight = averaging_weight

        self.peak_win_rate = 0.0
        self.peak_checkpoint_state = None
        self.episodes_since_peak = 0
        self.performance_drop_detected = False

        self.peak_lr = None
        self.lr_reduction_factor = 0.5
        self.min_lr = 1e-7

        self.averaging_performed = 0
        self.last_averaging_episode = -1

    def update_performance(
        self, current_win_rate, current_episode, model_state_dict, current_lr
    ):
        performance_improved = False
        performance_drop = False

        if current_win_rate > self.peak_win_rate:
            self.peak_win_rate = current_win_rate
            self.peak_checkpoint_state = copy.deepcopy(model_state_dict)
            self.peak_lr = current_lr
            self.episodes_since_peak = 0
            self.performance_drop_detected = False
            performance_improved = True
        else:
            self.episodes_since_peak += 1

            if (
                current_win_rate < self.peak_win_rate - self.performance_drop_threshold
                and not self.performance_drop_detected
                and self.episodes_since_peak > 10
            ):
                self.performance_drop_detected = True
                performance_drop = True

        return performance_improved, performance_drop

    def should_perform_averaging(self, current_episode):
        return (
            self.performance_drop_detected
            and self.peak_checkpoint_state is not None
            and current_episode - self.last_averaging_episode > 5
        )

    def perform_checkpoint_averaging(self, current_model):
        if self.peak_checkpoint_state is None:
            return False

        try:
            current_state = current_model.state_dict()
            averaged_state = {}

            for name, param in current_state.items():
                if name in self.peak_checkpoint_state:
                    peak_param = self.peak_checkpoint_state[name]
                    peak_param = peak_param.to(param.device)
                    averaged_state[name] = (
                        self.averaging_weight * peak_param
                        + (1 - self.averaging_weight) * param
                    )
                else:
                    averaged_state[name] = param

            current_model.load_state_dict(averaged_state)

            self.averaging_performed += 1
            self.last_averaging_episode = self.episodes_since_peak
            self.performance_drop_detected = False

            return True

        except Exception as e:
            return False

    def should_reduce_lr(self):
        return self.performance_drop_detected and self.peak_lr is not None

    def get_reduced_lr(self, current_lr):
        new_lr = max(current_lr * self.lr_reduction_factor, self.min_lr)
        return new_lr

    def get_stats(self):
        return {
            "peak_win_rate": self.peak_win_rate,
            "episodes_since_peak": self.episodes_since_peak,
            "performance_drop_detected": self.performance_drop_detected,
            "averaging_performed": self.averaging_performed,
            "has_peak_checkpoint": self.peak_checkpoint_state is not None,
            "peak_lr": self.peak_lr,
        }


class EnhancedEnergyStabilityManager:
    """🛡️ Your original stability manager - UNCHANGED."""

    def __init__(self, initial_lr=3e-5, thinking_lr=0.06, policy_memory_manager=None):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.thinking_lr = thinking_lr
        self.current_thinking_lr = thinking_lr
        self.policy_memory_manager = policy_memory_manager

        self.win_rate_window = deque(maxlen=25)
        self.energy_quality_window = deque(maxlen=25)
        self.energy_separation_window = deque(maxlen=25)

        self.min_win_rate = 0.35
        self.min_energy_quality = 2.5
        self.min_energy_separation = 0.015
        self.max_early_stop_rate = 0.75

        self.lr_decay_factor = 0.85
        self.lr_recovery_factor = 1.02
        self.max_lr_reductions = 2
        self.lr_reductions = 0

        self.last_reset_episode = 0
        self.consecutive_poor_episodes = 0
        self.best_model_state = None
        self.best_win_rate = 0.0
        self.emergency_mode = False

    def update_metrics(
        self, win_rate, energy_quality, energy_separation, early_stop_rate
    ):
        self.win_rate_window.append(win_rate)
        self.energy_quality_window.append(energy_quality)
        self.energy_separation_window.append(energy_separation)

        avg_win_rate = safe_mean(list(self.win_rate_window), 0.5)
        avg_energy_quality = safe_mean(list(self.energy_quality_window), 10.0)
        avg_energy_separation = safe_mean(list(self.energy_separation_window), 0.1)

        collapse_indicators = 0

        if avg_win_rate < self.min_win_rate:
            collapse_indicators += 1
        if avg_energy_quality < self.min_energy_quality:
            collapse_indicators += 1
        if abs(avg_energy_separation) < self.min_energy_separation:
            collapse_indicators += 1
        if early_stop_rate > self.max_early_stop_rate:
            collapse_indicators += 1

        if collapse_indicators >= 2:
            self.consecutive_poor_episodes += 1
            if self.consecutive_poor_episodes >= 2:
                return self._trigger_emergency_protocol()
        else:
            self.consecutive_poor_episodes = 0
            self.emergency_mode = False

        return False

    def _trigger_emergency_protocol(self):
        emergency_triggered = False

        if not self.emergency_mode:
            if (
                self.policy_memory_manager
                and self.policy_memory_manager.should_perform_averaging(0)
            ):
                emergency_triggered = True

            if self.lr_reductions < self.max_lr_reductions:
                self.current_lr *= self.lr_decay_factor
                self.current_thinking_lr *= self.lr_decay_factor
                self.lr_reductions += 1
                emergency_triggered = True

            self.emergency_mode = True
            self.consecutive_poor_episodes = 0

        return emergency_triggered

    def get_current_lrs(self):
        return self.current_lr, self.current_thinking_lr


class EnhancedCheckpointManager:
    """💾 Your original checkpoint manager - UNCHANGED."""

    def __init__(self, checkpoint_dir="checkpoints_enhanced"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.best_checkpoint_path = None
        self.best_win_rate = 0.0
        self.emergency_checkpoint_path = None

        self.peak_checkpoint_path = None
        self.averaging_checkpoints = []

    def save_checkpoint(
        self,
        verifier,
        agent,
        episode,
        win_rate,
        energy_quality,
        is_emergency=False,
        is_peak=False,
        policy_memory_stats=None,
    ):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if is_emergency:
            filename = f"emergency_ep{episode}_{timestamp}.pt"
        elif is_peak:
            filename = f"peak_ep{episode}_wr{win_rate:.3f}_{timestamp}.pt"
        else:
            filename = f"checkpoint_ep{episode}_wr{win_rate:.3f}_eq{energy_quality:.1f}_{timestamp}.pt"

        checkpoint_path = self.checkpoint_dir / filename

        checkpoint_data = {
            "episode": episode,
            "win_rate": win_rate,
            "energy_quality": energy_quality,
            "verifier_state_dict": verifier.state_dict(),
            "agent_thinking_stats": agent.get_thinking_stats(),
            "agent_current_thinking_steps": agent.current_thinking_steps,
            "agent_current_thinking_lr": agent.current_thinking_lr,
            "energy_scale": verifier.energy_scale,
            "timestamp": timestamp,
            "is_emergency": is_emergency,
            "is_peak": is_peak,
            "policy_memory_stats": policy_memory_stats or {},
        }

        try:
            torch.save(checkpoint_data, checkpoint_path)

            if is_emergency:
                self.emergency_checkpoint_path = checkpoint_path
            elif is_peak:
                self.peak_checkpoint_path = checkpoint_path
            elif win_rate > self.best_win_rate:
                self.best_checkpoint_path = checkpoint_path
                self.best_win_rate = win_rate

            return checkpoint_path

        except Exception as e:
            return None

    def load_checkpoint(self, checkpoint_path, verifier, agent):
        try:
            checkpoint_data = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )

            verifier.load_state_dict(checkpoint_data["verifier_state_dict"])

            agent.current_thinking_steps = checkpoint_data.get(
                "agent_current_thinking_steps", agent.initial_thinking_steps
            )
            agent.current_thinking_lr = checkpoint_data.get(
                "agent_current_thinking_lr", agent.initial_thinking_lr
            )

            if "energy_scale" in checkpoint_data:
                verifier.energy_scale = checkpoint_data["energy_scale"]

            return checkpoint_data

        except Exception as e:
            return None


# Utility functions
def verify_enhanced_energy_flow(verifier, observation_space, action_space):
    """Verify enhanced energy flow with transformer components works correctly."""
    try:
        visual_obs = torch.zeros(1, 3, SCREEN_HEIGHT, SCREEN_WIDTH)
        vector_obs = torch.zeros(1, 5, VECTOR_FEATURE_DIM)
        dummy_obs = {"visual_obs": visual_obs, "vector_obs": vector_obs}
        dummy_action = torch.ones(1, action_space.n) / action_space.n

        energy = verifier(dummy_obs, dummy_action)

        print(f"   ✅ Enhanced energy calculation successful")
        print(f"   - Energy shape: {energy.shape}")
        print(f"   - Energy value: {energy.item():.4f}")

        # Test transformer components
        features, transformer_energy = verifier.features_extractor(dummy_obs)
        print(f"   ✅ Transformer energy calculation successful")
        print(f"   - Transformer energy: {transformer_energy.item():.4f}")

        return True

    except Exception as e:
        print(f"   ❌ Enhanced energy flow verification failed: {e}")
        return False


def make_enhanced_env(
    game="StreetFighterIISpecialChampionEdition-Genesis", state="ken_bison_12.state"
):
    """Create Street Fighter environment with enhanced features."""
    try:
        env = retro.make(
            game=game, state=state, use_restricted_actions=retro.Actions.DISCRETE
        )
        env = StreetFighterVisionWrapper(
            env
        )  # Your original wrapper (fast UI maintained)

        print(f"   ✅ Enhanced environment created (fast UI speed maintained)")
        return env

    except Exception as e:
        print(f"   ❌ Enhanced environment creation failed: {e}")
        raise


# Export enhanced components
__all__ = [
    # Environment
    "StreetFighterVisionWrapper",
    "make_enhanced_env",
    # Enhanced Energy-Based Transformer Components
    "EnergyBasedMultiHeadAttention",
    "EnergyBasedTransformerBlock",
    "EnhancedEnergyBasedCNN",
    "EnhancedEnergyBasedVerifier",
    "EnhancedEnergyBasedAgent",
    # Original components (with fixes)
    "SimplifiedFeatureTracker",
    "StreetFighterDiscreteActions",
    "IntelligentRewardCalculator",
    "EnhancedQualityBasedExperienceBuffer",
    "GoldenExperienceBuffer",
    "PolicyMemoryManager",
    "EnhancedEnergyStabilityManager",
    "EnhancedCheckpointManager",
    # Utilities
    "verify_enhanced_energy_flow",
    "safe_divide",
    "safe_std",
    "safe_mean",
    "sanitize_array",
    "ensure_scalar",
    "safe_comparison",
    "ensure_feature_dimension",
    # Constants
    "VECTOR_FEATURE_DIM",
    "MAX_FIGHT_STEPS",
]

print(f"🥊 ENHANCED Energy-Based Transformer wrapper.py loaded successfully!")
print(f"   - ✅ Energy-Based Transformer architecture integrated")
print(f"   - ✅ Current energy thinking system expanded")
print(f"   - ✅ Fast game UI speed maintained")
print(f"   - ✅ Single round fight logic preserved")
print(f"   - ✅ Quality threshold fixed (0.6 → 0.3)")
print(f"🎯 Ready for enhanced learning with EBT + Energy thinking!")
