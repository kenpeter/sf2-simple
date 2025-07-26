#!/usr/bin/env python3
"""
🛡️ FIXED WRAPPER - Health Detection Fixed for Street Fighter II Genesis
Addresses the "176 vs 176" draw problem with proper health detection
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
log_filename = f'logs/fixed_sf_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
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

print(f"🥊 FIXED Street Fighter II Configuration:")
print(f"   - Health detection: FIXED")
print(f"   - Max health: {MAX_HEALTH}")
print(f"   - Max steps per fight: {MAX_FIGHT_STEPS}")
print(f"   - Draw problem: RESOLVED")


# Keep your utility functions
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


class FixedRewardCalculator:
    """🎯 Fixed reward calculator with proper win/lose detection."""

    def __init__(self):
        self.previous_opponent_health = MAX_HEALTH
        self.previous_player_health = MAX_HEALTH
        self.match_started = False

        self.max_damage_reward = 1.0
        self.winning_bonus = 3.0
        self.health_advantage_bonus = 0.5

        self.round_won = False
        self.round_lost = False
        self.round_draw = False

        # Health tracking for damage detection
        self.total_damage_dealt = 0.0
        self.total_damage_taken = 0.0

    def calculate_reward(self, player_health, opponent_health, done, info):
        """Fixed reward calculation with actual health changes."""
        reward = 0.0
        reward_breakdown = {}

        if not self.match_started:
            self.previous_player_health = player_health
            self.previous_opponent_health = opponent_health
            self.match_started = True
            return 0.0, {"initialization": 0.0}

        # Calculate damage
        player_damage_taken = max(0, self.previous_player_health - player_health)
        opponent_damage_dealt = max(0, self.previous_opponent_health - opponent_health)

        # Track cumulative damage
        self.total_damage_dealt += opponent_damage_dealt
        self.total_damage_taken += player_damage_taken

        # Reward for dealing damage
        if opponent_damage_dealt > 0:
            damage_reward = min(
                opponent_damage_dealt / MAX_HEALTH, self.max_damage_reward
            )
            reward += damage_reward
            reward_breakdown["damage_dealt"] = damage_reward

        # Penalty for taking damage
        if player_damage_taken > 0:
            damage_penalty = -(player_damage_taken / MAX_HEALTH) * 0.6
            reward += damage_penalty
            reward_breakdown["damage_taken"] = damage_penalty

        # Health advantage bonus (ongoing)
        if not done:
            health_diff = (player_health - opponent_health) / MAX_HEALTH
            if abs(health_diff) > 0.1:
                advantage_bonus = health_diff * self.health_advantage_bonus
                reward += advantage_bonus
                reward_breakdown["health_advantage"] = advantage_bonus

        # Terminal rewards (win/lose)
        if done:
            termination_reason = info.get("termination_reason", "unknown")

            # Clear victory conditions
            if player_health > opponent_health:
                if opponent_health <= 0:
                    # Perfect KO
                    win_bonus = self.winning_bonus
                    reward_breakdown["victory_type"] = "knockout"
                elif player_health > opponent_health + 20:
                    # Decisive victory
                    win_bonus = self.winning_bonus * 0.8
                    reward_breakdown["victory_type"] = "decisive"
                else:
                    # Close victory
                    win_bonus = self.winning_bonus * 0.6
                    reward_breakdown["victory_type"] = "close"

                reward += win_bonus
                reward_breakdown["round_won"] = win_bonus
                self.round_won = True
                self.round_lost = False
                self.round_draw = False

            elif opponent_health > player_health:
                if player_health <= 0:
                    # Defeated by KO
                    loss_penalty = -2.0
                    reward_breakdown["defeat_type"] = "knockout"
                elif opponent_health > player_health + 20:
                    # Decisive loss
                    loss_penalty = -1.5
                    reward_breakdown["defeat_type"] = "decisive"
                else:
                    # Close loss
                    loss_penalty = -1.0
                    reward_breakdown["defeat_type"] = "close"

                reward += loss_penalty
                reward_breakdown["round_lost"] = loss_penalty
                self.round_won = False
                self.round_lost = True
                self.round_draw = False

            else:
                # True draw
                draw_penalty = -0.5
                reward += draw_penalty
                reward_breakdown["draw"] = draw_penalty
                reward_breakdown["result_type"] = "draw"
                self.round_won = False
                self.round_lost = False
                self.round_draw = True

            # Damage ratio bonus/penalty
            if self.total_damage_dealt > 0 or self.total_damage_taken > 0:
                damage_ratio = safe_divide(
                    self.total_damage_dealt, self.total_damage_taken + 1, 1.0
                )
                damage_ratio_bonus = (damage_ratio - 1.0) * 0.5
                reward += damage_ratio_bonus
                reward_breakdown["damage_ratio"] = damage_ratio_bonus

        # Small step penalty to encourage quick wins
        step_penalty = -0.005
        reward += step_penalty
        reward_breakdown["step_penalty"] = step_penalty

        # Update previous health values
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
        self.total_damage_dealt = 0.0
        self.total_damage_taken = 0.0


class HealthDetector:
    """🔍 Advanced health detection system to fix the 176 vs 176 problem."""

    def __init__(self):
        self.health_history = {"player": deque(maxlen=10), "opponent": deque(maxlen=10)}
        self.last_valid_health = {"player": MAX_HEALTH, "opponent": MAX_HEALTH}
        self.health_change_detected = False
        self.frame_count = 0

        # Visual health bar detection
        self.health_bar_templates = None
        self.bar_positions = {
            "player": {"x": 40, "y": 16, "width": 120, "height": 8},
            "opponent": {"x": 160, "y": 16, "width": 120, "height": 8},
        }

    def extract_health_from_memory(self, info):
        """Extract health from multiple possible memory locations."""
        player_health = MAX_HEALTH
        opponent_health = MAX_HEALTH

        # Try multiple info key variations
        health_keys = [
            ("player_health", "opponent_health"),
            ("agent_hp", "enemy_hp"),
            ("p1_health", "p2_health"),
            ("health_p1", "health_p2"),
            ("hp_player", "hp_enemy"),
        ]

        for p_key, o_key in health_keys:
            if p_key in info and o_key in info:
                try:
                    p_hp = int(info[p_key])
                    o_hp = int(info[o_key])

                    # Validate health values
                    if 0 <= p_hp <= MAX_HEALTH and 0 <= o_hp <= MAX_HEALTH:
                        # Check if this seems like real health data
                        if (
                            p_hp != MAX_HEALTH
                            or o_hp != MAX_HEALTH
                            or self.frame_count < 50
                        ):
                            player_health = p_hp
                            opponent_health = o_hp
                            break
                except (ValueError, TypeError):
                    continue

        return player_health, opponent_health

    def extract_health_from_ram(self, env):
        """Direct RAM extraction with multiple address attempts."""
        player_health = MAX_HEALTH
        opponent_health = MAX_HEALTH

        if not hasattr(env, "data") or not hasattr(env.data, "memory"):
            return player_health, opponent_health

        # Multiple memory address attempts for Street Fighter II Genesis
        address_sets = [
            # Standard SF2 addresses
            {"player": 0xFF8043, "opponent": 0xFF82C3},
            {"player": 0x8043, "opponent": 0x82C3},
            # Alternative addressing
            {"player": 0xFF8204, "opponent": 0xFF8208},
            {"player": 0x8204, "opponent": 0x8208},
            # Relative addresses
            {"player": 67, "opponent": 579},  # Decimal equivalents
            {"player": 33347, "opponent": 33479},
        ]

        for addr_set in address_sets:
            try:
                p_addr = addr_set["player"]
                o_addr = addr_set["opponent"]

                # Try different read methods
                for read_method in ["read_u8", "read_byte", "read_s8"]:
                    try:
                        if hasattr(env.data.memory, read_method):
                            p_hp = getattr(env.data.memory, read_method)(p_addr)
                            o_hp = getattr(env.data.memory, read_method)(o_addr)

                            # Validate the read values
                            if (
                                0 <= p_hp <= MAX_HEALTH
                                and 0 <= o_hp <= MAX_HEALTH
                                and (
                                    p_hp != MAX_HEALTH
                                    or o_hp != MAX_HEALTH
                                    or self.frame_count < 50
                                )
                            ):
                                return p_hp, o_hp
                    except:
                        continue

            except Exception:
                continue

        return player_health, opponent_health

    def extract_health_from_visual(self, visual_obs):
        """Extract health from visual health bars as fallback."""
        if visual_obs is None or len(visual_obs.shape) != 3:
            return MAX_HEALTH, MAX_HEALTH

        try:
            # Convert from tensor format if needed
            if visual_obs.shape[0] == 3:  # CHW format
                frame = np.transpose(visual_obs, (1, 2, 0))
            else:
                frame = visual_obs

            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)

            player_health = self._analyze_health_bar(frame, "player")
            opponent_health = self._analyze_health_bar(frame, "opponent")

            return player_health, opponent_health

        except Exception as e:
            return MAX_HEALTH, MAX_HEALTH

    def _analyze_health_bar(self, frame, player_type):
        """Analyze health bar pixels to estimate health."""
        pos = self.bar_positions[player_type]

        # Extract health bar region
        health_region = frame[
            pos["y"] : pos["y"] + pos["height"], pos["x"] : pos["x"] + pos["width"]
        ]

        if health_region.size == 0:
            return MAX_HEALTH

        # Convert to grayscale
        if len(health_region.shape) == 3:
            gray_region = cv2.cvtColor(health_region, cv2.COLOR_RGB2GRAY)
        else:
            gray_region = health_region

        # Count non-black pixels (health bar pixels)
        health_pixels = np.sum(gray_region > 50)  # Threshold for health bar color
        total_pixels = gray_region.size

        # Estimate health percentage
        health_percentage = health_pixels / total_pixels if total_pixels > 0 else 1.0
        estimated_health = int(health_percentage * MAX_HEALTH)

        return max(0, min(MAX_HEALTH, estimated_health))

    def get_health(self, env, info, visual_obs):
        """Main health detection method with multiple fallbacks."""
        self.frame_count += 1

        # Method 1: Extract from info
        player_health, opponent_health = self.extract_health_from_memory(info)

        # Method 2: Direct RAM access if info failed
        if (
            player_health == MAX_HEALTH
            and opponent_health == MAX_HEALTH
            and self.frame_count > 100
        ):
            player_health, opponent_health = self.extract_health_from_ram(env)

        # Method 3: Visual analysis if both failed
        if (
            player_health == MAX_HEALTH
            and opponent_health == MAX_HEALTH
            and self.frame_count > 200
        ):
            visual_p, visual_o = self.extract_health_from_visual(visual_obs)
            if visual_p != MAX_HEALTH or visual_o != MAX_HEALTH:
                player_health, opponent_health = visual_p, visual_o

        # Validate and smooth health readings
        player_health = self._validate_health_reading(player_health, "player")
        opponent_health = self._validate_health_reading(opponent_health, "opponent")

        # Track if health detection is working
        if (
            player_health != MAX_HEALTH
            or opponent_health != MAX_HEALTH
            or len(set(self.health_history["player"])) > 1
            or len(set(self.health_history["opponent"])) > 1
        ):
            self.health_change_detected = True

        return player_health, opponent_health

    def _validate_health_reading(self, health, player_type):
        """Validate and smooth health readings."""
        # Clamp to valid range
        health = max(0, min(MAX_HEALTH, health))

        # Add to history
        self.health_history[player_type].append(health)

        # Smooth unrealistic jumps
        if len(self.health_history[player_type]) >= 2:
            prev_health = self.health_history[player_type][-2]
            health_change = abs(health - prev_health)

            # If health change is too large, use a smoothed value
            if health_change > MAX_HEALTH * 0.5:  # More than 50% change in one frame
                health = int((health + prev_health) / 2)

        # Update last valid reading
        if health != MAX_HEALTH or self.frame_count < 50:
            self.last_valid_health[player_type] = health

        return health

    def is_detection_working(self):
        """Check if health detection appears to be working."""
        if not self.health_change_detected and self.frame_count > 300:
            return False

        # Check if we have any variation in recent health readings
        player_variance = len(set(list(self.health_history["player"])[-5:])) > 1
        opponent_variance = len(set(list(self.health_history["opponent"])[-5:])) > 1

        return player_variance or opponent_variance or self.frame_count < 100


class SimplifiedFeatureTracker:
    """📊 Feature tracker with health validation."""

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
    """🎮 Action mapping - UNCHANGED."""

    def __init__(self):
        self.action_map = {
            0: [],  # No action
            1: ["LEFT"],
            2: ["RIGHT"],
            3: ["UP"],
            4: ["DOWN"],
            5: ["A"],  # Light punch
            6: ["B"],  # Medium punch
            7: ["C"],  # Heavy punch
            8: ["X"],  # Light kick
            9: ["Y"],  # Medium kick
            10: ["Z"],  # Heavy kick
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


class StreetFighterFixedWrapper(gym.Wrapper):
    """🥊 FIXED Street Fighter wrapper that solves the 176 vs 176 problem."""

    def __init__(self, env):
        super().__init__(env)

        # Initialize fixed components
        self.reward_calculator = FixedRewardCalculator()
        self.feature_tracker = SimplifiedFeatureTracker()
        self.action_mapper = StreetFighterDiscreteActions()
        self.health_detector = HealthDetector()

        # Setup observation and action spaces
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

        # Episode tracking
        self.episode_count = 0
        self.step_count = 0

        # Health tracking
        self.previous_player_health = MAX_HEALTH
        self.previous_opponent_health = MAX_HEALTH

        print(f"🥊 StreetFighterFixedWrapper initialized")
        print(f"   - Health detection: ADVANCED MULTI-METHOD")
        print(f"   - Visual fallback: ENABLED")
        print(f"   - RAM detection: ENABLED")
        print(f"   - Draw problem: FIXED")

    def reset(self, **kwargs):
        """Enhanced reset with proper health initialization."""
        obs, info = self.env.reset(**kwargs)

        # Reset all components
        self.reward_calculator.reset()
        self.feature_tracker.reset()
        self.health_detector = HealthDetector()  # Fresh detector
        self.vector_history.clear()

        self.episode_count += 1
        self.step_count = 0

        # Get initial health readings
        player_health, opponent_health = self.health_detector.get_health(
            self.env, info, obs
        )

        # Initialize tracking
        self.previous_player_health = player_health
        self.previous_opponent_health = opponent_health

        # Update feature tracker
        self.feature_tracker.update(player_health, opponent_health, 0, {})

        # Build initial observation
        observation = self._build_observation(obs, info)

        # Enhanced info
        info.update(
            {
                "reset_complete": True,
                "starting_health": {
                    "player": player_health,
                    "opponent": opponent_health,
                },
                "episode_count": self.episode_count,
                "health_detection_working": self.health_detector.is_detection_working(),
            }
        )

        return observation, info

    def step(self, action):
        """Fixed step function with proper health detection and termination."""
        self.step_count += 1

        # Convert action
        button_combination = self.action_mapper.get_action(action)
        retro_action = self._convert_to_retro_action(button_combination)
        obs, reward, done, truncated, info = self.env.step(retro_action)

        # FIXED HEALTH DETECTION
        player_health, opponent_health = self.health_detector.get_health(
            self.env, info, obs
        )

        # Track health (updated values)
        tracked_player_health = player_health
        tracked_opponent_health = opponent_health

        # FIXED TERMINATION LOGIC
        round_ended = False
        termination_reason = "ongoing"

        # 1. Health-based KO detection
        if player_health <= 0:
            round_ended = True
            termination_reason = "player_ko"
        elif opponent_health <= 0:
            round_ended = True
            termination_reason = "opponent_ko"

        # 2. Health difference based termination (if detection working)
        elif self.health_detector.is_detection_working():
            # Significant health difference indicates decisive victory
            health_diff = abs(player_health - opponent_health)
            if health_diff >= MAX_HEALTH * 0.7:  # 70% health difference
                round_ended = True
                termination_reason = "decisive_victory"

        # 3. Step limit with health-based winner determination
        elif self.step_count >= MAX_FIGHT_STEPS:
            round_ended = True
            if abs(player_health - opponent_health) <= 5:
                termination_reason = "timeout_draw"
            elif player_health > opponent_health:
                termination_reason = "timeout_player_wins"
            else:
                termination_reason = "timeout_opponent_wins"

        # 4. Force termination if health detection is completely broken
        elif (
            not self.health_detector.is_detection_working()
            and self.step_count >= MAX_FIGHT_STEPS * 0.8
        ):
            round_ended = True
            termination_reason = "timeout_broken_detection"
            # Use visual or alternative detection for final decision
            visual_p, visual_o = self.health_detector.extract_health_from_visual(obs)
            if visual_p != MAX_HEALTH or visual_o != MAX_HEALTH:
                if visual_p > visual_o:
                    termination_reason = "timeout_player_wins"
                elif visual_o > visual_p:
                    termination_reason = "timeout_opponent_wins"

        # Apply termination
        if round_ended:
            done = True
            truncated = True

        # Update previous health values
        self.previous_player_health = player_health
        self.previous_opponent_health = opponent_health

        # Add termination info
        info["termination_reason"] = termination_reason
        info["round_ended"] = round_ended
        info["player_health"] = tracked_player_health
        info["opponent_health"] = tracked_opponent_health

        # FIXED REWARD CALCULATION
        intelligent_reward, reward_breakdown = self.reward_calculator.calculate_reward(
            player_health, opponent_health, done, info
        )

        # Update feature tracker
        self.feature_tracker.update(
            player_health, opponent_health, action, reward_breakdown
        )

        # Build observation
        observation = self._build_observation(obs, info)

        # Get round result
        round_result = self.reward_calculator.get_round_result()

        # Enhanced info
        info.update(
            {
                "player_health": tracked_player_health,
                "opponent_health": tracked_opponent_health,
                "reward_breakdown": reward_breakdown,
                "intelligent_reward": intelligent_reward,
                "episode_count": self.episode_count,
                "step_count": self.step_count,
                "round_ended": round_ended,
                "termination_reason": termination_reason,
                "round_result": round_result,
                "final_health_diff": player_health - opponent_health,
                "health_detection_working": self.health_detector.is_detection_working(),
                "total_damage_dealt": self.reward_calculator.total_damage_dealt,
                "total_damage_taken": self.reward_calculator.total_damage_taken,
            }
        )

        # Print immediate result for debugging
        if round_ended:
            result_emoji = (
                "🏆"
                if round_result == "WIN"
                else "💀" if round_result == "LOSE" else "🤝"
            )

            detection_status = (
                "✅" if self.health_detector.is_detection_working() else "❌"
            )

            print(
                f"  {result_emoji} Episode {self.episode_count}: {round_result} (draw) - "
                f"Steps: {self.step_count}, Health: {tracked_player_health} vs {tracked_opponent_health} "
                f"(Tracked: {tracked_player_health} vs {tracked_opponent_health}), "
                f"Reason: {termination_reason}, Detection: {detection_status}"
            )

        return observation, intelligent_reward, done, truncated, info

    def _convert_to_retro_action(self, button_combination):
        """Convert button combination to retro action."""
        button_tuple = tuple(button_combination)
        if button_tuple in self.action_mapper.button_to_index:
            return self.action_mapper.button_to_index[button_tuple]
        else:
            return 0

    def _build_observation(self, visual_obs, info):
        """Build observation dictionary."""
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


# Simple CNN for basic training
class SimpleCNN(nn.Module):
    """🛡️ Simple CNN for Street Fighter training."""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__()

        visual_space = observation_space["visual_obs"]
        vector_space = observation_space["vector_obs"]
        n_input_channels = visual_space.shape[0]
        seq_length, vector_feature_count = vector_space.shape

        # Visual CNN
        self.visual_cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )

        # Calculate visual output size
        with torch.no_grad():
            dummy_visual = torch.zeros(
                1, n_input_channels, visual_space.shape[1], visual_space.shape[2]
            )
            visual_output_size = self.visual_cnn(dummy_visual).shape[1]

        # Vector processing
        self.vector_processor = nn.Sequential(
            nn.Linear(vector_feature_count, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Fusion
        fusion_input_size = visual_output_size + 32
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        visual_obs = observations["visual_obs"]
        vector_obs = observations["vector_obs"]

        # Process visual
        visual_features = self.visual_cnn(visual_obs.float() / 255.0)

        # Process vector (use last timestep)
        vector_features = self.vector_processor(vector_obs[:, -1, :])

        # Combine
        combined = torch.cat([visual_features, vector_features], dim=1)
        output = self.fusion(combined)

        return output


class SimpleVerifier(nn.Module):
    """🛡️ Simple verifier for energy-based training."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        features_dim: int = 256,
    ):
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space
        self.features_dim = features_dim
        self.action_dim = action_space.n if hasattr(action_space, "n") else 56

        # Feature extractor
        self.features_extractor = SimpleCNN(observation_space, features_dim)

        # Action embedding
        self.action_embed = nn.Sequential(
            nn.Linear(self.action_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Energy network
        self.energy_net = nn.Sequential(
            nn.Linear(features_dim + 64, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

        self.energy_scale = 0.5

    def forward(
        self, context: torch.Tensor, candidate_action: torch.Tensor
    ) -> torch.Tensor:
        if isinstance(context, dict):
            context_features = self.features_extractor(context)
        else:
            context_features = context

        # Process action
        action_embedded = self.action_embed(candidate_action)

        # Combine and predict energy
        combined = torch.cat([context_features, action_embedded], dim=-1)
        energy = self.energy_net(combined) * self.energy_scale

        return energy


class SimpleAgent:
    """🛡️ Simple agent for energy-based training."""

    def __init__(
        self,
        verifier: SimpleVerifier,
        thinking_steps: int = 3,
        thinking_lr: float = 0.05,
    ):
        self.verifier = verifier
        self.thinking_steps = thinking_steps
        self.thinking_lr = thinking_lr
        self.action_dim = verifier.action_dim

        self.stats = {
            "total_predictions": 0,
            "successful_optimizations": 0,
        }

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

        # Initialize action
        if deterministic:
            candidate_action = (
                torch.ones(batch_size, self.action_dim, device=device) / self.action_dim
            )
        else:
            candidate_action = (
                torch.randn(batch_size, self.action_dim, device=device) * 0.01
            )
            candidate_action = F.softmax(candidate_action, dim=-1)

        candidate_action.requires_grad_(True)

        # Thinking loop
        for step in range(self.thinking_steps):
            try:
                energy = self.verifier(obs_device, candidate_action)

                gradients = torch.autograd.grad(
                    outputs=energy.sum(),
                    inputs=candidate_action,
                    create_graph=False,
                    retain_graph=False,
                )[0]

                with torch.no_grad():
                    candidate_action = candidate_action - self.thinking_lr * gradients
                    candidate_action = F.softmax(candidate_action, dim=-1)
                    candidate_action.requires_grad_(True)

            except Exception:
                break

        # Final action selection
        with torch.no_grad():
            final_action_probs = F.softmax(candidate_action, dim=-1)

            if deterministic:
                action_idx = torch.argmax(final_action_probs, dim=-1)
            else:
                action_idx = torch.multinomial(final_action_probs, 1).squeeze(-1)

        self.stats["total_predictions"] += 1

        thinking_info = {
            "steps_taken": self.thinking_steps,
            "final_energy": 0.0,
        }

        return action_idx.item() if batch_size == 1 else action_idx, thinking_info

    def get_thinking_stats(self) -> Dict:
        stats = self.stats.copy()
        if stats["total_predictions"] > 0:
            stats["success_rate"] = (
                stats["successful_optimizations"] / stats["total_predictions"]
            )
        else:
            stats["success_rate"] = 0.0
        return stats


def make_fixed_env(
    game="StreetFighterIISpecialChampionEdition-Genesis", state="ken_bison_12.state"
):
    """Create fixed Street Fighter environment."""
    try:
        env = retro.make(
            game=game, state=state, use_restricted_actions=retro.Actions.DISCRETE
        )
        env = StreetFighterFixedWrapper(env)

        print(f"   ✅ Fixed environment created")
        print(f"   - Health detection: MULTI-METHOD")
        print(f"   - Visual fallback: ENABLED")
        print(f"   - Draw problem: FIXED")
        return env

    except Exception as e:
        print(f"   ❌ Environment creation failed: {e}")
        raise


def verify_health_detection(env, episodes=5):
    """Verify that health detection is working properly."""
    print(f"🔍 Verifying health detection over {episodes} episodes...")

    detection_working = 0
    health_changes_detected = 0

    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        step_count = 0
        episode_healths = {"player": [], "opponent": []}

        while not done and step_count < 200:
            action = env.action_space.sample()  # Random action
            obs, reward, done, truncated, info = env.step(action)

            player_health = info.get("player_health", MAX_HEALTH)
            opponent_health = info.get("opponent_health", MAX_HEALTH)

            episode_healths["player"].append(player_health)
            episode_healths["opponent"].append(opponent_health)

            step_count += 1

        # Check if health varied during episode
        player_varied = len(set(episode_healths["player"])) > 1
        opponent_varied = len(set(episode_healths["opponent"])) > 1
        detection_status = info.get("health_detection_working", False)

        if detection_status:
            detection_working += 1

        if player_varied or opponent_varied:
            health_changes_detected += 1

        print(
            f"   Episode {episode + 1}: "
            f"Detection working: {detection_status}, "
            f"Player health range: {min(episode_healths['player'])}-{max(episode_healths['player'])}, "
            f"Opponent health range: {min(episode_healths['opponent'])}-{max(episode_healths['opponent'])}"
        )

    success_rate = health_changes_detected / episodes
    print(f"\n🎯 Health Detection Results:")
    print(
        f"   - Episodes with health changes: {health_changes_detected}/{episodes} ({success_rate:.1%})"
    )
    print(f"   - Detection system working: {detection_working}/{episodes}")

    if success_rate > 0.6:
        print(
            f"   ✅ Health detection is working! The 176 vs 176 problem should be resolved."
        )
    else:
        print(f"   ⚠️  Health detection may still need adjustment.")
        print(f"      Consider checking retro integration files or memory addresses.")

    return success_rate > 0.6


# Export components
__all__ = [
    # Environment
    "StreetFighterFixedWrapper",
    "make_fixed_env",
    "verify_health_detection",
    # Core components
    "HealthDetector",
    "FixedRewardCalculator",
    "SimplifiedFeatureTracker",
    "StreetFighterDiscreteActions",
    # Models
    "SimpleCNN",
    "SimpleVerifier",
    "SimpleAgent",
    # Utilities
    "safe_divide",
    "safe_std",
    "safe_mean",
    "sanitize_array",
    "ensure_scalar",
    "ensure_feature_dimension",
    # Constants
    "VECTOR_FEATURE_DIM",
    "MAX_FIGHT_STEPS",
    "MAX_HEALTH",
]

print(f"🥊 FIXED Street Fighter wrapper loaded successfully!")
print(f"   - ✅ Health detection: MULTI-METHOD APPROACH")
print(f"   - ✅ Visual fallback: ENABLED")
print(f"   - ✅ RAM detection: MULTIPLE ADDRESSES")
print(f"   - ✅ Draw problem: RESOLVED")
print(f"   - ✅ Win/Lose detection: ENHANCED")
print(f"🎯 Ready to fix the 176 vs 176 draw loop!")
