#!/usr/bin/env python3
"""
🛡️ ENHANCED WRAPPER WITH POLICY MEMORY AND GOLDEN EXPERIENCE BUFFER
Implements checkpoint averaging and golden experience buffer to prevent performance degradation.
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
log_filename = f'logs/enhanced_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
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

print(f"🥊 ENHANCED POLICY MEMORY Configuration:")
print(f"   - Features: {VECTOR_FEATURE_DIM}")
print(f"   - Training paradigm: Policy Memory + Golden Buffer")
print(f"   - Fight mode: Single Round with Memory Protection")
print(f"   - Checkpoint averaging: ✅ ACTIVE")
print(f"   - Golden experience buffer: ✅ ACTIVE")


# Enhanced safe operations
def safe_divide(numerator, denominator, default=0.0):
    """Safe division that prevents NaN and handles edge cases."""
    numerator = ensure_scalar(numerator, default)
    denominator = ensure_scalar(denominator, 1.0 if default == 0.0 else default)

    if denominator == 0 or not np.isfinite(denominator):
        return default
    result = numerator / denominator
    return result if np.isfinite(result) else default


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
            print(f"⚠️  Features is not array-like: {type(features)}, creating zeros")
            return np.zeros(target_dim, dtype=np.float32)

    if features.ndim == 0:
        features = np.array([features.item()], dtype=np.float32)

    try:
        current_length = len(features)
    except TypeError:
        print(f"⚠️  Cannot get length of features: {type(features)}, using zeros")
        return np.zeros(target_dim, dtype=np.float32)

    if current_length == target_dim:
        return features.astype(np.float32)
    elif current_length < target_dim:
        padding = np.zeros(target_dim - current_length, dtype=np.float32)
        return np.concatenate([features, padding]).astype(np.float32)
    else:
        print(f"⚠️  Truncating features from {current_length} to {target_dim}")
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


# MISSING CLASSES - Adding the core components that were referenced but not included


class IntelligentRewardCalculator:
    """🎯 Intelligent reward calculator that prevents point-scoring exploitation."""

    def __init__(self):
        self.previous_opponent_health = MAX_HEALTH
        self.previous_player_health = MAX_HEALTH
        self.match_started = False

        # Anti-exploitation parameters
        self.max_damage_reward = 0.8  # Cap damage rewards
        self.winning_bonus = 2.0  # Strong bonus for actually winning
        self.health_advantage_bonus = 0.3  # Bonus for health advantage

        # Track round outcome
        self.round_won = False
        self.round_lost = False

    def calculate_reward(self, player_health, opponent_health, done, info):
        """Calculate intelligent reward that prioritizes winning over point-scoring."""
        reward = 0.0
        reward_breakdown = {}

        # Initialize if first call
        if not self.match_started:
            self.previous_player_health = player_health
            self.previous_opponent_health = opponent_health
            self.match_started = True
            return 0.0, {"initialization": 0.0}

        # Calculate health changes
        player_damage_taken = max(0, self.previous_player_health - player_health)
        opponent_damage_dealt = max(0, self.previous_opponent_health - opponent_health)

        # Damage rewards (capped to prevent exploitation)
        if opponent_damage_dealt > 0:
            damage_reward = min(
                opponent_damage_dealt / MAX_HEALTH, self.max_damage_reward
            )
            reward += damage_reward
            reward_breakdown["damage_dealt"] = damage_reward

        # Damage penalties
        if player_damage_taken > 0:
            damage_penalty = -(player_damage_taken / MAX_HEALTH) * 0.5
            reward += damage_penalty
            reward_breakdown["damage_taken"] = damage_penalty

        # Round completion bonuses/penalties
        if done:
            if player_health > opponent_health:
                # Won the round - big bonus
                win_bonus = self.winning_bonus
                reward += win_bonus
                reward_breakdown["round_won"] = win_bonus
                self.round_won = True
            elif opponent_health > player_health:
                # Lost the round - penalty
                loss_penalty = -1.0
                reward += loss_penalty
                reward_breakdown["round_lost"] = loss_penalty
                self.round_lost = True
            else:
                # Draw - small penalty to encourage decisive action
                draw_penalty = -0.3
                reward += draw_penalty
                reward_breakdown["draw"] = draw_penalty
        else:
            # Health advantage bonus (small, to encourage maintaining advantage)
            health_diff = (player_health - opponent_health) / MAX_HEALTH
            if abs(health_diff) > 0.1:  # Only if significant difference
                advantage_bonus = health_diff * self.health_advantage_bonus
                reward += advantage_bonus
                reward_breakdown["health_advantage"] = advantage_bonus

        # Small step penalty to encourage efficiency
        step_penalty = -0.01
        reward += step_penalty
        reward_breakdown["step_penalty"] = step_penalty

        # Update previous health values
        self.previous_player_health = player_health
        self.previous_opponent_health = opponent_health

        return reward, reward_breakdown

    def reset(self):
        """Reset for new episode."""
        self.previous_opponent_health = MAX_HEALTH
        self.previous_player_health = MAX_HEALTH
        self.match_started = False
        self.round_won = False
        self.round_lost = False


class SimplifiedFeatureTracker:
    """📊 Simplified feature tracking for stable training."""

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

        # Track combos (simplified)
        if "damage_dealt" in reward_breakdown and reward_breakdown["damage_dealt"] > 0:
            if action == self.last_action:
                self.combo_count += 1
            else:
                self.combo_count = 0

        self.last_action = action

    def get_features(self):
        """Get current feature vector."""
        features = []

        # Health histories (padded to history_length)
        player_hist = list(self.player_health_history)
        opponent_hist = list(self.opponent_health_history)

        while len(player_hist) < self.history_length:
            player_hist.insert(0, 1.0)  # Full health default
        while len(opponent_hist) < self.history_length:
            opponent_hist.insert(0, 1.0)  # Full health default

        features.extend(player_hist)
        features.extend(opponent_hist)

        # Current states
        current_player_health = player_hist[-1] if player_hist else 1.0
        current_opponent_health = opponent_hist[-1] if opponent_hist else 1.0

        features.extend(
            [
                current_player_health,
                current_opponent_health,
                current_player_health - current_opponent_health,  # Health difference
                self.last_action / 55.0,  # Normalized last action
                min(self.combo_count / 5.0, 1.0),  # Normalized combo count
            ]
        )

        # Pad or truncate to exact dimension
        return ensure_feature_dimension(
            np.array(features, dtype=np.float32), VECTOR_FEATURE_DIM
        )


class StreetFighterDiscreteActions:
    """🎮 Street Fighter discrete action mapping."""

    def __init__(self):
        # Comprehensive action set for Street Fighter
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
            # Special moves (quarter circle forward + punch/kick)
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
            # Dragon punch motion (forward, down, down-forward + punch)
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

    def get_action(self, action_idx):
        """Convert action index to button combination."""
        return self.action_map.get(action_idx, [])


class StreetFighterVisionWrapper(gym.Wrapper):
    """🥊 Enhanced Street Fighter vision wrapper with policy memory support."""

    def __init__(self, env):
        super().__init__(env)

        # Initialize trackers
        self.reward_calculator = IntelligentRewardCalculator()
        self.feature_tracker = SimplifiedFeatureTracker()
        self.action_mapper = StreetFighterDiscreteActions()

        # Observation space
        visual_space = gym.spaces.Box(
            low=0, high=255, shape=(3, SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.uint8
        )
        vector_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(5, VECTOR_FEATURE_DIM),  # 5 frame history
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Dict(
            {"visual_obs": visual_space, "vector_obs": vector_space}
        )

        # Action space
        self.action_space = gym.spaces.Discrete(self.action_mapper.n_actions)

        # Frame history for vector observations
        self.vector_history = deque(maxlen=5)

        # Episode tracking
        self.episode_count = 0
        self.step_count = 0

        print(f"🥊 StreetFighterVisionWrapper initialized")
        print(f"   - Action space: {self.action_space.n} discrete actions")
        print(f"   - Visual observation: {visual_space.shape}")
        print(f"   - Vector observation: {vector_space.shape}")

    def reset(self, **kwargs):
        """Reset environment and trackers."""
        obs, info = self.env.reset(**kwargs)

        # Reset trackers
        self.reward_calculator.reset()
        self.feature_tracker.reset()
        self.vector_history.clear()

        self.episode_count += 1
        self.step_count = 0

        # Extract initial health values
        player_health, opponent_health = self._extract_health(info)

        # Initialize feature tracker
        self.feature_tracker.update(player_health, opponent_health, 0, {})

        # Build observation
        observation = self._build_observation(obs, info)

        return observation, info

    def step(self, action):
        """Execute action and return enhanced observation."""
        self.step_count += 1

        # Convert discrete action to button combination
        button_combination = self.action_mapper.get_action(action)

        # Execute action (for retro, we need to convert to the expected format)
        # This might need adjustment based on your specific retro setup
        retro_action = self._convert_to_retro_action(button_combination)
        obs, reward, done, truncated, info = self.env.step(retro_action)

        # Extract health values
        player_health, opponent_health = self._extract_health(info)

        # Calculate intelligent reward
        intelligent_reward, reward_breakdown = self.reward_calculator.calculate_reward(
            player_health, opponent_health, done, info
        )

        # Update feature tracker
        self.feature_tracker.update(
            player_health, opponent_health, action, reward_breakdown
        )

        # Build observation
        observation = self._build_observation(obs, info)

        # Enhanced info
        info.update(
            {
                "player_health": player_health,
                "opponent_health": opponent_health,
                "reward_breakdown": reward_breakdown,
                "intelligent_reward": intelligent_reward,
                "episode_count": self.episode_count,
                "step_count": self.step_count,
            }
        )

        return observation, intelligent_reward, done, truncated, info

    def _extract_health(self, info):
        """Extract health values from info or RAM."""
        # This is game-specific - adjust based on your Street Fighter ROM
        # Common RAM addresses for health in Street Fighter games
        player_health = info.get("player_health", MAX_HEALTH)
        opponent_health = info.get("opponent_health", MAX_HEALTH)

        # If not in info, try to extract from RAM
        if hasattr(self.env, "data") and hasattr(self.env.data, "memory"):
            # These addresses are examples - adjust for your specific game
            try:
                player_health = self.env.data.memory.read_byte(
                    0x8004
                )  # Example address
                opponent_health = self.env.data.memory.read_byte(
                    0x8008
                )  # Example address
            except:
                pass  # Fall back to defaults

        return player_health, opponent_health

    def _convert_to_retro_action(self, button_combination):
        """Convert button combination to retro action format - FIXED VERSION."""
        # For retro environments, we need to return an integer action index
        # that corresponds to the button combination in the retro action space

        # Map button combinations to retro action indices
        # This mapping needs to match your specific retro ROM's action space
        retro_action_map = {
            (): 0,  # No action
            ("LEFT",): 6,
            ("RIGHT",): 7,
            ("UP",): 4,
            ("DOWN",): 5,
            ("A",): 8,  # Light punch
            ("B",): 0,  # Medium punch (adjust index as needed)
            ("C",): 9,  # Heavy punch
            ("X",): 1,  # Light kick
            ("Y",): 2,  # Medium kick
            ("Z",): 10,  # Heavy kick
            # Add more combinations as needed
            ("LEFT", "A"): 11,
            ("RIGHT", "A"): 12,
            ("DOWN", "A"): 13,
            # ... add more combinations
        }

        # Convert list to tuple for dictionary lookup
        button_tuple = tuple(button_combination)

        # Try to find exact match first
        if button_tuple in retro_action_map:
            return retro_action_map[button_tuple]

        # Fallback: if no exact match, try individual buttons
        if len(button_combination) == 1:
            button = button_combination[0]
            single_button_map = {
                "LEFT": 6,
                "RIGHT": 7,
                "UP": 4,
                "DOWN": 5,
                "A": 8,
                "B": 0,
                "C": 9,
                "X": 1,
                "Y": 2,
                "Z": 10,
            }
            return single_button_map.get(button, 0)

        # Final fallback: return no action
        return 0

    def _build_observation(self, visual_obs, info):
        """Build combined observation."""
        # Process visual observation
        if isinstance(visual_obs, np.ndarray):
            if len(visual_obs.shape) == 3 and visual_obs.shape[2] == 3:
                # Convert HWC to CHW
                visual_obs = np.transpose(visual_obs, (2, 0, 1))

            # Resize if needed
            if visual_obs.shape[-2:] != (SCREEN_HEIGHT, SCREEN_WIDTH):
                visual_obs = cv2.resize(
                    visual_obs.transpose(1, 2, 0), (SCREEN_WIDTH, SCREEN_HEIGHT)
                ).transpose(2, 0, 1)

        # Get vector features
        vector_features = self.feature_tracker.get_features()

        # Add to history
        self.vector_history.append(vector_features)

        # Pad history if needed
        while len(self.vector_history) < 5:
            self.vector_history.appendleft(
                np.zeros(VECTOR_FEATURE_DIM, dtype=np.float32)
            )

        # Stack vector history
        vector_obs = np.stack(list(self.vector_history), axis=0)

        return {
            "visual_obs": visual_obs.astype(np.uint8),
            "vector_obs": vector_obs.astype(np.float32),
        }


class GoldenExperienceBuffer:
    """🏆 Golden Experience Buffer - Stores peak performance experiences."""

    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.experiences = deque(maxlen=capacity)
        self.min_quality_for_golden = 0.75  # Higher threshold for golden experiences
        self.peak_win_rate_threshold = (
            0.55  # Only store experiences from good performance periods
        )
        self.current_win_rate = 0.0

        print(f"🏆 Golden Experience Buffer initialized")
        print(f"   - Capacity: {capacity}")
        print(f"   - Min quality for golden: {self.min_quality_for_golden}")
        print(f"   - Peak win rate threshold: {self.peak_win_rate_threshold}")

    def update_win_rate(self, win_rate):
        """Update current win rate for filtering experiences."""
        self.current_win_rate = win_rate

    def add_experience(self, experience, quality_score):
        """Add experience to golden buffer if it meets criteria."""
        # Only add if we're performing well and the experience is high quality
        if (
            self.current_win_rate >= self.peak_win_rate_threshold
            and quality_score >= self.min_quality_for_golden
        ):

            # Add golden flag
            golden_experience = experience.copy()
            golden_experience["is_golden"] = True
            golden_experience["golden_win_rate"] = self.current_win_rate
            golden_experience["golden_quality"] = quality_score

            self.experiences.append(golden_experience)

    def sample_golden_batch(self, batch_size):
        """Sample batch from golden experiences."""
        if len(self.experiences) < batch_size:
            return list(self.experiences)  # Return all if not enough

        indices = np.random.choice(len(self.experiences), batch_size, replace=False)
        return [self.experiences[i] for i in indices]

    def get_stats(self):
        """Get golden buffer statistics."""
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

    def adjust_threshold(self, target_good_ratio=0.5, episode_number=0):
        """More conservative threshold adjustment."""
        # Only adjust every N episodes
        if episode_number % self.threshold_adjustment_frequency != 0:
            return

        if len(self.quality_scores) < 100:  # Need enough data
            return

        total_size = len(self.good_experiences) + len(self.bad_experiences)
        current_good_ratio = len(self.good_experiences) / max(1, total_size)

        # More gradual adjustments
        if current_good_ratio < target_good_ratio - 0.15:  # Wider tolerance
            self.quality_threshold *= 1 - self.adjustment_rate
            self.threshold_adjustments += 1
            print(f"📉 Lowered quality threshold to {self.quality_threshold:.3f}")
        elif current_good_ratio > target_good_ratio + 0.15:  # Wider tolerance
            self.quality_threshold *= 1 + self.adjustment_rate
            self.threshold_adjustments += 1
            print(f"📈 Raised quality threshold to {self.quality_threshold:.3f}")

        # Keep threshold in reasonable range
        self.quality_threshold = max(0.4, min(0.75, self.quality_threshold))


class EnhancedQualityBasedExperienceBuffer:
    """🎯 Enhanced quality-based experience buffer with golden experience integration."""

    def __init__(
        self, capacity=30000, quality_threshold=0.6, golden_buffer_capacity=1000
    ):
        self.capacity = capacity
        self.quality_threshold = quality_threshold

        # Separate good and bad experiences
        self.good_experiences = deque(maxlen=capacity // 2)
        self.bad_experiences = deque(maxlen=capacity // 2)

        # Golden experience buffer integration
        self.golden_buffer = GoldenExperienceBuffer(capacity=golden_buffer_capacity)

        # Quality tracking
        self.quality_scores = deque(maxlen=1000)
        self.total_added = 0

        # Adaptive threshold parameters
        self.adjustment_rate = 0.05
        self.threshold_adjustment_frequency = 25
        self.threshold_adjustments = 0

        print(f"🎯 Enhanced Quality-Based Experience Buffer initialized")
        print(
            f"   - Capacity: {capacity:,} (Good: {capacity//2:,}, Bad: {capacity//2:,})"
        )
        print(f"   - Quality threshold: {quality_threshold}")
        print(f"   - Golden buffer capacity: {golden_buffer_capacity:,}")

    def add_experience(self, experience, reward, reward_breakdown, quality_score):
        """Add experience to appropriate buffer based on quality."""
        self.total_added += 1
        self.quality_scores.append(quality_score)

        # Classify experience
        if quality_score >= self.quality_threshold:
            self.good_experiences.append(experience)

            # Add to golden buffer if it meets higher criteria
            self.golden_buffer.add_experience(experience, quality_score)
        else:
            self.bad_experiences.append(experience)

    def update_win_rate(self, win_rate):
        """Update win rate for golden buffer filtering."""
        self.golden_buffer.update_win_rate(win_rate)

    def sample_enhanced_balanced_batch(self, batch_size, golden_ratio=0.15):
        """Sample balanced batch with golden experiences included."""
        if (
            len(self.good_experiences) < batch_size // 4
            or len(self.bad_experiences) < batch_size // 4
        ):
            return None, None, None

        # Calculate batch composition
        golden_count = int(batch_size * golden_ratio)
        remaining_good = (batch_size // 2) - golden_count
        bad_count = batch_size // 2

        # Sample golden experiences
        golden_batch = (
            self.golden_buffer.sample_golden_batch(golden_count)
            if golden_count > 0
            else []
        )

        # Sample regular good experiences
        good_indices = np.random.choice(
            len(self.good_experiences), remaining_good, replace=False
        )
        good_batch = [self.good_experiences[i] for i in good_indices]

        # Sample bad experiences
        bad_indices = np.random.choice(
            len(self.bad_experiences), bad_count, replace=False
        )
        bad_batch = [self.bad_experiences[i] for i in bad_indices]

        # Combine good and golden for good batch
        combined_good_batch = good_batch + golden_batch

        return combined_good_batch, bad_batch, golden_batch

    def get_stats(self):
        """Get comprehensive buffer statistics."""
        total_size = len(self.good_experiences) + len(self.bad_experiences)

        if len(self.quality_scores) > 0:
            avg_quality = np.mean(list(self.quality_scores))
            quality_std = safe_std(list(self.quality_scores), 0.0)
        else:
            avg_quality = 0.0
            quality_std = 0.0

        # Golden buffer stats
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
        """More conservative threshold adjustment."""
        # Only adjust every N episodes
        if episode_number % self.threshold_adjustment_frequency != 0:
            return

        if len(self.quality_scores) < 100:  # Need enough data
            return

        total_size = len(self.good_experiences) + len(self.bad_experiences)
        current_good_ratio = len(self.good_experiences) / max(1, total_size)

        # More gradual adjustments
        if current_good_ratio < target_good_ratio - 0.15:  # Wider tolerance
            self.quality_threshold *= 1 - self.adjustment_rate
            self.threshold_adjustments += 1
            print(f"📉 Lowered quality threshold to {self.quality_threshold:.3f}")
        elif current_good_ratio > target_good_ratio + 0.15:  # Wider tolerance
            self.quality_threshold *= 1 + self.adjustment_rate
            self.threshold_adjustments += 1
            print(f"📈 Raised quality threshold to {self.quality_threshold:.3f}")

        # Keep threshold in reasonable range
        self.quality_threshold = max(0.4, min(0.75, self.quality_threshold))


class PolicyMemoryManager:
    """🧠 Policy Memory Manager - Implements checkpoint averaging and policy memory."""

    def __init__(self, performance_drop_threshold=0.05, averaging_weight=0.7):
        self.performance_drop_threshold = performance_drop_threshold
        self.averaging_weight = (
            averaging_weight  # Weight for best checkpoint in averaging
        )

        # Track peak performance
        self.peak_win_rate = 0.0
        self.peak_checkpoint_state = None
        self.episodes_since_peak = 0
        self.performance_drop_detected = False

        # Learning rate memory
        self.peak_lr = None
        self.lr_reduction_factor = 0.5
        self.min_lr = 1e-7

        # Averaging history
        self.averaging_performed = 0
        self.last_averaging_episode = -1

        print(f"🧠 Policy Memory Manager initialized")
        print(f"   - Performance drop threshold: {performance_drop_threshold}")
        print(
            f"   - Averaging weight (best/current): {averaging_weight}/{1-averaging_weight}"
        )
        print(f"   - LR reduction factor: {self.lr_reduction_factor}")

    def update_performance(
        self, current_win_rate, current_episode, model_state_dict, current_lr
    ):
        """Update performance tracking and detect drops."""
        performance_improved = False
        performance_drop = False

        # Check for new peak
        if current_win_rate > self.peak_win_rate:
            print(
                f"🏆 NEW PEAK PERFORMANCE: {current_win_rate:.3f} (prev: {self.peak_win_rate:.3f})"
            )
            self.peak_win_rate = current_win_rate
            self.peak_checkpoint_state = copy.deepcopy(model_state_dict)
            self.peak_lr = current_lr
            self.episodes_since_peak = 0
            self.performance_drop_detected = False
            performance_improved = True
        else:
            self.episodes_since_peak += 1

            # Check for performance drop
            if (
                current_win_rate < self.peak_win_rate - self.performance_drop_threshold
                and not self.performance_drop_detected
                and self.episodes_since_peak > 10
            ):

                print(f"📉 PERFORMANCE DROP DETECTED!")
                print(
                    f"   Current: {current_win_rate:.3f} | Peak: {self.peak_win_rate:.3f}"
                )
                print(
                    f"   Drop: {self.peak_win_rate - current_win_rate:.3f} (threshold: {self.performance_drop_threshold})"
                )
                self.performance_drop_detected = True
                performance_drop = True

        return performance_improved, performance_drop

    def should_perform_averaging(self, current_episode):
        """Check if we should perform checkpoint averaging."""
        return (
            self.performance_drop_detected
            and self.peak_checkpoint_state is not None
            and current_episode - self.last_averaging_episode > 5
        )

    def perform_checkpoint_averaging(self, current_model):
        """Perform checkpoint averaging between current model and peak checkpoint."""
        if self.peak_checkpoint_state is None:
            print("⚠️  No peak checkpoint available for averaging")
            return False

        try:
            current_state = current_model.state_dict()
            averaged_state = {}

            print(f"🔄 Performing checkpoint averaging...")
            print(
                f"   Weights: {self.averaging_weight:.1f} (peak) + {1-self.averaging_weight:.1f} (current)"
            )

            for name, param in current_state.items():
                if name in self.peak_checkpoint_state:
                    peak_param = self.peak_checkpoint_state[name]

                    # Ensure tensors are on the same device
                    peak_param = peak_param.to(param.device)

                    # Weighted average
                    averaged_state[name] = (
                        self.averaging_weight * peak_param
                        + (1 - self.averaging_weight) * param
                    )
                else:
                    # If parameter doesn't exist in peak (shouldn't happen), use current
                    averaged_state[name] = param

            # Load averaged state
            current_model.load_state_dict(averaged_state)

            self.averaging_performed += 1
            self.last_averaging_episode = self.episodes_since_peak
            self.performance_drop_detected = False  # Reset flag

            print(f"✅ Checkpoint averaging completed (#{self.averaging_performed})")
            return True

        except Exception as e:
            print(f"❌ Checkpoint averaging failed: {e}")
            return False

    def should_reduce_lr(self):
        """Check if learning rate should be reduced."""
        return self.performance_drop_detected and self.peak_lr is not None

    def get_reduced_lr(self, current_lr):
        """Get reduced learning rate."""
        new_lr = max(current_lr * self.lr_reduction_factor, self.min_lr)
        print(f"📉 Reducing learning rate: {current_lr:.2e} → {new_lr:.2e}")
        return new_lr

    def get_stats(self):
        """Get policy memory statistics."""
        return {
            "peak_win_rate": self.peak_win_rate,
            "episodes_since_peak": self.episodes_since_peak,
            "performance_drop_detected": self.performance_drop_detected,
            "averaging_performed": self.averaging_performed,
            "has_peak_checkpoint": self.peak_checkpoint_state is not None,
            "peak_lr": self.peak_lr,
        }


class EnhancedFixedEnergyBasedStreetFighterCNN(nn.Module):
    """🛡️ Enhanced CNN with improved regularization for policy memory."""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__()

        visual_space = observation_space["visual_obs"]
        vector_space = observation_space["vector_obs"]
        n_input_channels = visual_space.shape[0]
        seq_length, vector_feature_count = vector_space.shape

        print(f"🔧 Enhanced Energy-Based CNN Feature Extractor Configuration:")
        print(f"   - Visual channels: {n_input_channels}")
        print(f"   - Visual size: {visual_space.shape[1]}x{visual_space.shape[2]}")
        print(f"   - Vector sequence: {seq_length} x {vector_feature_count}")
        print(f"   - Output features: {features_dim}")

        # Enhanced CNN architecture with stronger regularization
        self.visual_cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.15),  # Increased dropout
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.15),  # Increased dropout
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.1),  # Additional dropout layer
            nn.AdaptiveAvgPool2d((3, 4)),
            nn.Flatten(),
        )

        # Calculate visual output size
        with torch.no_grad():
            dummy_visual = torch.zeros(
                1, n_input_channels, visual_space.shape[1], visual_space.shape[2]
            )
            visual_output_size = self.visual_cnn(dummy_visual).shape[1]

        # Enhanced vector processing with stronger regularization
        self.vector_embed = nn.Linear(vector_feature_count, 64)
        self.vector_norm = nn.LayerNorm(64)
        self.vector_dropout = nn.Dropout(0.2)  # Increased dropout
        self.vector_gru = nn.GRU(
            64, 64, batch_first=True, dropout=0.15
        )  # Increased dropout
        self.vector_final = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),  # Increased dropout
        )

        # Enhanced fusion layer with stronger regularization
        fusion_input_size = visual_output_size + 32
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, 512),
            nn.ReLU(inplace=True),
            nn.LayerNorm(512),
            nn.Dropout(0.2),  # Increased dropout
            nn.Linear(512, features_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),  # Additional final dropout
        )

        # Conservative weight initialization
        self.apply(self._init_weights)

        # Monitoring
        self.activation_monitor = {
            "nan_count": 0,
            "explosion_count": 0,
            "forward_count": 0,
        }

        print(f"   - Visual output size: {visual_output_size}")
        print(f"   - Fusion input size: {fusion_input_size}")
        print(f"   ✅ Enhanced Energy-Based Feature Extractor initialized")

    def _init_weights(self, m):
        """Extra conservative weight initialization for policy memory."""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.3)  # Even more conservative gain
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param, gain=0.3)  # More conservative gain
                elif "bias" in name:
                    nn.init.constant_(param, 0)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        visual_obs = observations["visual_obs"]
        vector_obs = observations["vector_obs"]
        device = next(self.parameters()).device

        visual_obs = visual_obs.float().to(device)
        vector_obs = vector_obs.float().to(device)

        self.activation_monitor["forward_count"] += 1

        # Enhanced NaN safety
        visual_nan_mask = ~torch.isfinite(visual_obs)
        vector_nan_mask = ~torch.isfinite(vector_obs)

        if torch.any(visual_nan_mask):
            self.activation_monitor["nan_count"] += torch.sum(visual_nan_mask).item()
            visual_obs = torch.where(
                visual_nan_mask, torch.zeros_like(visual_obs), visual_obs
            )

        if torch.any(vector_nan_mask):
            self.activation_monitor["nan_count"] += torch.sum(vector_nan_mask).item()
            vector_obs = torch.where(
                vector_nan_mask, torch.zeros_like(vector_obs), vector_obs
            )

        # Enhanced input normalization
        visual_obs = torch.clamp(visual_obs / 255.0, 0.0, 1.0)
        vector_obs = torch.clamp(vector_obs, -8.0, 8.0)  # Tighter bounds

        # Process visual features
        visual_features = self.visual_cnn(visual_obs)

        # Enhanced explosion monitoring
        if torch.any(torch.abs(visual_features) > 50.0):  # Tighter bound
            self.activation_monitor["explosion_count"] += 1
            visual_features = torch.clamp(visual_features, -50.0, 50.0)

        # Process vector features
        batch_size, seq_len, feature_dim = vector_obs.shape
        vector_embedded = self.vector_embed(vector_obs)
        vector_embedded = self.vector_norm(vector_embedded)
        vector_embedded = self.vector_dropout(vector_embedded)

        gru_output, _ = self.vector_gru(vector_embedded)
        vector_features = gru_output[:, -1, :]
        vector_features = self.vector_final(vector_features)

        # Combine and process
        combined_features = torch.cat([visual_features, vector_features], dim=1)
        output = self.fusion(combined_features)

        # Enhanced safety checks
        if torch.any(~torch.isfinite(output)):
            self.activation_monitor["nan_count"] += torch.sum(
                ~torch.isfinite(output)
            ).item()
            output = torch.where(
                ~torch.isfinite(output), torch.zeros_like(output), output
            )

        output = torch.clamp(output, -15.0, 15.0)  # Tighter bounds

        return output

    def get_activation_stats(self):
        """Get activation monitoring statistics."""
        return self.activation_monitor.copy()


class EnhancedFixedEnergyBasedStreetFighterVerifier(nn.Module):
    """🛡️ Enhanced Energy-Based Verifier with policy memory support."""

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

        # Enhanced feature extractor
        self.features_extractor = EnhancedFixedEnergyBasedStreetFighterCNN(
            observation_space, features_dim
        )

        # Enhanced action embedding with stronger regularization
        self.action_embed = nn.Sequential(
            nn.Linear(self.action_dim, 64),
            nn.ReLU(inplace=True),
            nn.LayerNorm(64),
            nn.Dropout(0.15),  # Increased dropout
        )

        # Enhanced energy network with stronger regularization
        self.energy_net = nn.Sequential(
            nn.Linear(features_dim + 64, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Dropout(0.2),  # Increased dropout
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.LayerNorm(128),
            nn.Dropout(0.15),  # Increased dropout
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.LayerNorm(64),
            nn.Dropout(0.1),  # Additional dropout
            nn.Linear(64, 1),
        )

        # More conservative energy scaling parameters
        self.energy_scale = 0.6  # Further reduced scale
        self.energy_clamp_min = -6.0  # Tighter bounds
        self.energy_clamp_max = 6.0  # Tighter bounds

        # Energy landscape monitoring
        self.energy_monitor = {
            "forward_count": 0,
            "nan_count": 0,
            "explosion_count": 0,
            "energy_history": deque(maxlen=1000),
            "gradient_norms": deque(maxlen=1000),
        }

        # Extra conservative initialization
        self.apply(self._init_weights)

        print(f"✅ Enhanced EnergyBasedStreetFighterVerifier initialized")
        print(f"   - Energy scale: {self.energy_scale} (more conservative)")
        print(
            f"   - Energy bounds: [{self.energy_clamp_min}, {self.energy_clamp_max}] (tighter)"
        )
        print(f"   - Enhanced regularization: ✅ ACTIVE")

    def _init_weights(self, m):
        """Extra conservative weight initialization for policy memory."""
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.005)  # Even smaller std
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(
        self, context: torch.Tensor, candidate_action: torch.Tensor
    ) -> torch.Tensor:
        """Enhanced energy calculation with tighter bounds."""
        device = next(self.parameters()).device
        self.energy_monitor["forward_count"] += 1

        # Ensure inputs are on correct device and finite
        if isinstance(context, dict):
            context_features = self.features_extractor(context)
        else:
            context_features = context

        context_features = context_features.to(device)
        candidate_action = candidate_action.to(device)

        # Enhanced safety checks
        if torch.any(~torch.isfinite(context_features)):
            nan_count = torch.sum(~torch.isfinite(context_features)).item()
            self.energy_monitor["nan_count"] += nan_count
            context_features = torch.where(
                ~torch.isfinite(context_features),
                torch.zeros_like(context_features),
                context_features,
            )

        if torch.any(~torch.isfinite(candidate_action)):
            nan_count = torch.sum(~torch.isfinite(candidate_action)).item()
            self.energy_monitor["nan_count"] += nan_count
            candidate_action = torch.where(
                ~torch.isfinite(candidate_action),
                torch.zeros_like(candidate_action),
                candidate_action,
            )

        # Tighter input clamping
        context_features = torch.clamp(context_features, -15.0, 15.0)
        candidate_action = torch.clamp(candidate_action, 0.0, 1.0)

        # Embed action
        action_embedded = self.action_embed(candidate_action)

        # Check for NaN in action embedding
        if torch.any(~torch.isfinite(action_embedded)):
            nan_count = torch.sum(~torch.isfinite(action_embedded)).item()
            self.energy_monitor["nan_count"] += nan_count
            action_embedded = torch.where(
                ~torch.isfinite(action_embedded),
                torch.zeros_like(action_embedded),
                action_embedded,
            )

        # Combine features and action
        combined_input = torch.cat([context_features, action_embedded], dim=-1)

        # Calculate raw energy
        raw_energy = self.energy_net(combined_input)

        # Monitor for energy explosions with tighter bounds
        if torch.any(torch.abs(raw_energy) > 30.0):  # Tighter bound
            self.energy_monitor["explosion_count"] += 1

        # Scale energy with more conservative scaling
        energy = raw_energy * self.energy_scale

        # Clamp energy within tighter bounds
        energy = torch.clamp(energy, self.energy_clamp_min, self.energy_clamp_max)

        # Final safety check
        if torch.any(~torch.isfinite(energy)):
            nan_count = torch.sum(~torch.isfinite(energy)).item()
            self.energy_monitor["nan_count"] += nan_count
            energy = torch.where(
                ~torch.isfinite(energy), torch.zeros_like(energy), energy
            )

        # Store energy for monitoring
        self.energy_monitor["energy_history"].append(energy.mean().item())

        return energy

    def get_energy_stats(self):
        """Get energy monitoring statistics."""
        stats = self.energy_monitor.copy()
        if len(stats["energy_history"]) > 0:
            stats["energy_mean"] = safe_mean(list(stats["energy_history"]), 0.0)
            stats["energy_std"] = safe_std(list(stats["energy_history"]), 0.0)
        else:
            stats["energy_mean"] = 0.0
            stats["energy_std"] = 0.0
        return stats


class EnhancedFixedStabilizedEnergyBasedAgent:
    """🛡️ Enhanced Energy-Based Agent with policy memory integration."""

    def __init__(
        self,
        verifier: EnhancedFixedEnergyBasedStreetFighterVerifier,
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

        # More conservative thinking process parameters
        self.gradient_clip = 0.3  # Further reduced
        self.early_stop_patience = 2
        self.min_energy_improvement = 8e-4  # Slightly higher threshold

        # Adaptive thinking parameters
        self.max_thinking_steps = 5  # Reduced
        self.min_thinking_steps = 1

        # Statistics
        self.thinking_stats = {
            "total_predictions": 0,
            "avg_thinking_steps": 0.0,
            "avg_energy_improvement": 0.0,
            "early_stops": 0,
            "energy_explosions": 0,
            "gradient_explosions": 0,
            "successful_optimizations": 0,
            "failed_optimizations": 0,
        }

        # Performance-based adaptation
        self.recent_performance = deque(maxlen=100)

        print(f"✅ Enhanced StabilizedEnergyBasedAgent initialized")
        print(f"   - Thinking steps: {thinking_steps} (reduced for stability)")
        print(f"   - Thinking LR: {thinking_lr} (reduced for stability)")
        print(f"   - Gradient clip: {self.gradient_clip} (more conservative)")
        print(f"   - Noise scale: {noise_scale} (reduced for stability)")

    def predict(
        self, observations: Dict[str, torch.Tensor], deterministic: bool = False
    ) -> Tuple[int, Dict]:
        """Enhanced action prediction with tighter stability controls."""
        device = next(self.verifier.parameters()).device

        # Ensure observations are on device
        obs_device = {}
        for key, value in observations.items():
            if isinstance(value, torch.Tensor):
                obs_device[key] = value.to(device)
            else:
                obs_device[key] = torch.from_numpy(value).to(device)

        # Add batch dimension if needed
        if len(obs_device["visual_obs"].shape) == 3:
            for key in obs_device:
                obs_device[key] = obs_device[key].unsqueeze(0)

        batch_size = obs_device["visual_obs"].shape[0]

        # Initialize candidate action with even less noise
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

        # Track thinking process
        energy_history = []
        steps_taken = 0
        early_stopped = False
        energy_explosion = False
        gradient_explosion = False
        optimization_successful = False

        # Initial energy
        with torch.no_grad():
            try:
                initial_energy = self.verifier(obs_device, candidate_action)
                energy_history.append(initial_energy.mean().item())
            except Exception as e:
                print(f"⚠️ Initial energy calculation failed: {e}")
                return 0, {"error": "initial_energy_failed"}

        # Thinking loop with tighter bounds
        for step in range(self.current_thinking_steps):
            try:
                # Calculate current energy
                energy = self.verifier(obs_device, candidate_action)

                # Check for energy explosion with tighter bound
                if torch.any(torch.abs(energy) > 8.0):  # Even tighter bound
                    energy_explosion = True
                    break

                # Calculate gradient
                gradients = torch.autograd.grad(
                    outputs=energy.sum(),
                    inputs=candidate_action,
                    create_graph=False,
                    retain_graph=False,
                )[0]

                # Enhanced gradient monitoring
                gradient_norm = torch.norm(gradients).item()

                if gradient_norm > self.gradient_clip:
                    gradient_explosion = True
                    gradients = gradients * (self.gradient_clip / gradient_norm)

                # Check for NaN gradients
                if torch.any(~torch.isfinite(gradients)):
                    break

                # Update candidate action with smaller steps
                with torch.no_grad():
                    candidate_action = (
                        candidate_action - self.current_thinking_lr * gradients
                    )
                    candidate_action = F.softmax(candidate_action, dim=-1)
                    candidate_action.requires_grad_(True)

                # Track energy improvement
                with torch.no_grad():
                    new_energy = self.verifier(obs_device, candidate_action)
                    energy_history.append(new_energy.mean().item())

                # Enhanced early stopping
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
                print(f"⚠️ Thinking step {step} failed: {e}")
                break

        # Determine optimization success
        if len(energy_history) > 1:
            total_improvement = abs(energy_history[0] - energy_history[-1])
            optimization_successful = (
                total_improvement > self.min_energy_improvement
                and not energy_explosion
                and not gradient_explosion
            )

        # Final action selection with enhanced safety
        with torch.no_grad():
            try:
                final_action_probs = F.softmax(candidate_action, dim=-1)
                # Add small epsilon for numerical stability
                final_action_probs = final_action_probs + 1e-8
                final_action_probs = final_action_probs / final_action_probs.sum(
                    dim=-1, keepdim=True
                )

                if deterministic:
                    action_idx = torch.argmax(final_action_probs, dim=-1)
                else:
                    action_idx = torch.multinomial(final_action_probs, 1).squeeze(-1)
            except Exception as e:
                print(f"⚠️ Final action selection failed: {e}")
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

        # Track recent performance
        self.recent_performance.append(1.0 if optimization_successful else 0.0)

        # Information about thinking process
        thinking_info = {
            "energy_history": energy_history,
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
        }

        return action_idx.item() if batch_size == 1 else action_idx, thinking_info

    def get_thinking_stats(self) -> Dict:
        """Get comprehensive thinking statistics."""
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


class EnhancedEnergyStabilityManager:
    """🛡️ Enhanced Energy Landscape Stability Manager with policy memory."""

    def __init__(self, initial_lr=3e-5, thinking_lr=0.06, policy_memory_manager=None):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.thinking_lr = thinking_lr
        self.current_thinking_lr = thinking_lr
        self.policy_memory_manager = policy_memory_manager

        # Performance tracking
        self.win_rate_window = deque(maxlen=25)  # Longer window for stability
        self.energy_quality_window = deque(maxlen=25)
        self.energy_separation_window = deque(maxlen=25)

        # More conservative emergency thresholds
        self.min_win_rate = 0.35  # Increased
        self.min_energy_quality = 2.5  # Reduced
        self.min_energy_separation = 0.015  # Reduced
        self.max_early_stop_rate = 0.75  # Reduced

        # More conservative adaptive parameters
        self.lr_decay_factor = 0.85  # Less aggressive decay
        self.lr_recovery_factor = 1.02  # Much slower recovery
        self.max_lr_reductions = 2  # Fewer reductions
        self.lr_reductions = 0

        # State tracking
        self.last_reset_episode = 0
        self.consecutive_poor_episodes = 0
        self.best_model_state = None
        self.best_win_rate = 0.0
        self.emergency_mode = False

        print(f"🛡️  Enhanced EnergyStabilityManager initialized")
        print(f"   - Min energy separation: {self.min_energy_separation}")
        print(f"   - Min energy quality: {self.min_energy_quality}")
        print(
            f"   - Policy memory integration: {'✅' if policy_memory_manager else '❌'}"
        )

    def update_metrics(
        self, win_rate, energy_quality, energy_separation, early_stop_rate
    ):
        """Update performance metrics and check for instability with policy memory."""
        self.win_rate_window.append(win_rate)
        self.energy_quality_window.append(energy_quality)
        self.energy_separation_window.append(energy_separation)

        # Check for energy landscape collapse
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

        # Trigger emergency if multiple indicators
        if collapse_indicators >= 2:  # More sensitive
            self.consecutive_poor_episodes += 1
            if self.consecutive_poor_episodes >= 2:  # Faster response
                print(f"🚨 ENERGY LANDSCAPE INSTABILITY DETECTED!")
                return self._trigger_emergency_protocol()
        else:
            self.consecutive_poor_episodes = 0
            self.emergency_mode = False

        return False

    def _trigger_emergency_protocol(self):
        """🚨 Enhanced emergency protocol with policy memory."""
        print(f"🛡️  ACTIVATING ENHANCED EMERGENCY STABILIZATION PROTOCOL")

        emergency_triggered = False

        if not self.emergency_mode:
            # First try policy memory if available
            if (
                self.policy_memory_manager
                and self.policy_memory_manager.should_perform_averaging(0)
            ):
                print(f"   🧠 Attempting policy memory recovery...")
                # This will be handled by the training script
                emergency_triggered = True

            # Also reduce learning rates
            if self.lr_reductions < self.max_lr_reductions:
                self.current_lr *= self.lr_decay_factor
                self.current_thinking_lr *= self.lr_decay_factor
                self.lr_reductions += 1

                print(f"   📉 Learning rates reduced:")
                print(f"      - Main LR: {self.current_lr:.2e}")
                print(f"      - Thinking LR: {self.current_thinking_lr:.3f}")
                emergency_triggered = True

            self.emergency_mode = True
            self.consecutive_poor_episodes = 0

        return emergency_triggered

    def get_current_lrs(self):
        """Get current learning rates."""
        return self.current_lr, self.current_thinking_lr


class EnhancedCheckpointManager:
    """Enhanced checkpoint management with policy memory support."""

    def __init__(self, checkpoint_dir="checkpoints_enhanced"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.best_checkpoint_path = None
        self.best_win_rate = 0.0
        self.emergency_checkpoint_path = None

        # Policy memory checkpoints
        self.peak_checkpoint_path = None
        self.averaging_checkpoints = []

        print(f"💾 Enhanced CheckpointManager initialized: {checkpoint_dir}")

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
        """Save checkpoint with policy memory metadata."""
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
                print(f"🚨 Emergency checkpoint saved: {filename}")
            elif is_peak:
                self.peak_checkpoint_path = checkpoint_path
                print(f"🏆 Peak checkpoint saved: {filename}")
            elif win_rate > self.best_win_rate:
                self.best_checkpoint_path = checkpoint_path
                self.best_win_rate = win_rate
                print(f"💾 New best checkpoint saved: {filename}")
            else:
                print(f"💾 Checkpoint saved: {filename}")

            return checkpoint_path

        except Exception as e:
            print(f"❌ Failed to save checkpoint: {e}")
            return None

    def load_checkpoint(self, checkpoint_path, verifier, agent):
        """Load checkpoint and restore state with policy memory support."""
        try:
            checkpoint_data = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )

            # Restore verifier
            verifier.load_state_dict(checkpoint_data["verifier_state_dict"])

            # Restore agent parameters
            agent.current_thinking_steps = checkpoint_data.get(
                "agent_current_thinking_steps", agent.initial_thinking_steps
            )
            agent.current_thinking_lr = checkpoint_data.get(
                "agent_current_thinking_lr", agent.initial_thinking_lr
            )

            # Restore energy scale
            if "energy_scale" in checkpoint_data:
                verifier.energy_scale = checkpoint_data["energy_scale"]

            print(f"✅ Checkpoint restored from: {checkpoint_path.name}")
            print(f"   - Episode: {checkpoint_data['episode']}")
            print(f"   - Win rate: {checkpoint_data['win_rate']:.3f}")
            print(f"   - Energy quality: {checkpoint_data['energy_quality']:.1f}")

            if checkpoint_data.get("is_peak", False):
                print(f"   🏆 Peak performance checkpoint restored")

            return checkpoint_data

        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            return None


# Utility functions for environment creation
def verify_fixed_energy_flow(verifier, observation_space, action_space):
    """Verify that energy flow is working correctly."""
    print(f"🔍 Verifying enhanced energy flow...")

    try:
        # Create dummy observations
        visual_obs = torch.zeros(1, 3, SCREEN_HEIGHT, SCREEN_WIDTH)
        vector_obs = torch.zeros(1, 5, VECTOR_FEATURE_DIM)

        dummy_obs = {"visual_obs": visual_obs, "vector_obs": vector_obs}

        # Create dummy action
        dummy_action = torch.ones(1, action_space.n) / action_space.n

        # Test forward pass
        energy = verifier(dummy_obs, dummy_action)

        print(f"   ✅ Energy calculation successful")
        print(f"   - Energy shape: {energy.shape}")
        print(f"   - Energy value: {energy.item():.4f}")
        print(
            f"   - Energy bounds: [{verifier.energy_clamp_min}, {verifier.energy_clamp_max}]"
        )

        return True

    except Exception as e:
        print(f"   ❌ Energy flow verification failed: {e}")
        return False


def make_fixed_env(
    game="StreetFighterIISpecialChampionEdition-Genesis", state="ken_bison_12.state"
):
    """Create enhanced Street Fighter environment with policy memory support."""
    print(f"🎮 Creating enhanced Street Fighter environment...")
    print(f"   - Game: {game}")
    print(f"   - State: {state}")

    try:
        # Create base retro environment
        env = retro.make(
            game=game, state=state, use_restricted_actions=retro.Actions.DISCRETE
        )

        # Wrap with our enhanced vision wrapper
        env = StreetFighterVisionWrapper(env)

        print(f"   ✅ Enhanced environment created successfully")
        print(f"   - Observation space: {env.observation_space}")
        print(f"   - Action space: {env.action_space}")

        return env

    except Exception as e:
        print(f"   ❌ Environment creation failed: {e}")
        raise


# Use enhanced versions for backwards compatibility
FixedEnergyBasedStreetFighterCNN = EnhancedFixedEnergyBasedStreetFighterCNN
FixedEnergyBasedStreetFighterVerifier = EnhancedFixedEnergyBasedStreetFighterVerifier
FixedStabilizedEnergyBasedAgent = EnhancedFixedStabilizedEnergyBasedAgent
FixedEnergyStabilityManager = EnhancedEnergyStabilityManager
CheckpointManager = EnhancedCheckpointManager
QualityBasedExperienceBuffer = EnhancedQualityBasedExperienceBuffer

# Export all components with enhanced versions
__all__ = [
    # Core enhanced components
    "StreetFighterVisionWrapper",
    "EnhancedFixedEnergyBasedStreetFighterCNN",
    "EnhancedFixedEnergyBasedStreetFighterVerifier",
    "EnhancedFixedStabilizedEnergyBasedAgent",
    "SimplifiedFeatureTracker",
    "StreetFighterDiscreteActions",
    # Enhanced quality-based components
    "IntelligentRewardCalculator",
    "EnhancedQualityBasedExperienceBuffer",
    "GoldenExperienceBuffer",
    # Policy memory components
    "PolicyMemoryManager",
    # Enhanced stability components
    "EnhancedEnergyStabilityManager",
    "EnhancedCheckpointManager",
    # Aliases for backwards compatibility
    "FixedEnergyBasedStreetFighterCNN",
    "FixedEnergyBasedStreetFighterVerifier",
    "FixedStabilizedEnergyBasedAgent",
    "FixedEnergyStabilityManager",
    "CheckpointManager",
    "QualityBasedExperienceBuffer",
    # Utilities
    "verify_fixed_energy_flow",
    "make_fixed_env",
    "safe_divide",
    "safe_std",
    "safe_mean",
    "sanitize_array",
    "ensure_scalar",
    "safe_comparison",
    "ensure_feature_dimension",
    # Constants
    "VECTOR_FEATURE_DIM",
]

print(f"🥊 ENHANCED POLICY MEMORY - Complete wrapper.py loaded successfully!")
print(f"   - Training paradigm: Policy Memory + Golden Buffer")
print(f"   - Fight mode: Single decisive rounds with memory protection")
print(f"   - Checkpoint averaging: ✅ ACTIVE")
print(f"   - Golden experience buffer: ✅ ACTIVE")
print(f"   - Enhanced regularization: ✅ ACTIVE")
print(f"   - Conservative learning parameters: ✅ ACTIVE")
print(f"🎯 Ready for stable 60% win rate training with policy memory!")
