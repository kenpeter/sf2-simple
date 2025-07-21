#!/usr/bin/env python3
"""
🛡️ FIXED WRAPPER WITH POLICY MEMORY AND GOLDEN EXPERIENCE BUFFER
Implements checkpoint averaging and golden experience buffer to prevent performance degradation.
FIXES: Quality threshold, health detection, single fight episodes, gradient stability
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
log_filename = f'logs/fixed_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
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
MAX_FIGHT_STEPS = 1200  # Max steps per single fight (~1 minute at 60 FPS)

print(f"🥊 FIXED POLICY MEMORY Configuration:")
print(f"   - Features: {VECTOR_FEATURE_DIM}")
print(f"   - Training paradigm: Policy Memory + Golden Buffer + FIXES")
print(f"   - Fight mode: SINGLE ROUND with proper health detection")
print(f"   - Max steps per fight: {MAX_FIGHT_STEPS}")
print(f"   - Quality threshold: LOWERED for learning")
print(f"   - Gradient stability: ENHANCED")


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


class FixedIntelligentRewardCalculator:
    """🎯 FIXED intelligent reward calculator with better quality scoring."""

    def __init__(self):
        self.previous_opponent_health = MAX_HEALTH
        self.previous_player_health = MAX_HEALTH
        self.match_started = False

        # FIXED: More balanced reward parameters
        self.max_damage_reward = 0.5  # Reduced to prevent exploitation
        self.winning_bonus = 3.0  # INCREASED for stronger win signal
        self.health_advantage_bonus = 0.2
        self.step_penalty = -0.005  # REDUCED step penalty

        # Track round outcome
        self.round_won = False
        self.round_lost = False

    def calculate_reward(self, player_health, opponent_health, done, info):
        """Calculate intelligent reward with stronger win signals."""
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
            damage_penalty = -(player_damage_taken / MAX_HEALTH) * 0.3
            reward += damage_penalty
            reward_breakdown["damage_taken"] = damage_penalty

        # FIXED: Stronger round completion signals
        if done:
            if player_health > opponent_health:
                # Won the round - BIG bonus
                win_bonus = self.winning_bonus
                reward += win_bonus
                reward_breakdown["round_won"] = win_bonus
                self.round_won = True
                print(f"🏆 ROUND WON! Bonus: {win_bonus}")
            elif opponent_health > player_health:
                # Lost the round - penalty
                loss_penalty = -1.5  # Increased penalty
                reward += loss_penalty
                reward_breakdown["round_lost"] = loss_penalty
                self.round_lost = True
                print(f"💀 ROUND LOST! Penalty: {loss_penalty}")
            else:
                # Draw - small penalty
                draw_penalty = -0.5
                reward += draw_penalty
                reward_breakdown["draw"] = draw_penalty
                print(f"🤝 DRAW! Penalty: {draw_penalty}")
        else:
            # Health advantage bonus (small)
            health_diff = (player_health - opponent_health) / MAX_HEALTH
            if abs(health_diff) > 0.1:
                advantage_bonus = health_diff * self.health_advantage_bonus
                reward += advantage_bonus
                reward_breakdown["health_advantage"] = advantage_bonus

        # Small step penalty to encourage efficiency
        reward += self.step_penalty
        reward_breakdown["step_penalty"] = self.step_penalty

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


class FixedStreetFighterVisionWrapper(gym.Wrapper):
    """🥊 FIXED Street Fighter vision wrapper with proper health detection and single fights."""

    def __init__(self, env):
        super().__init__(env)

        # Initialize trackers
        self.reward_calculator = FixedIntelligentRewardCalculator()
        self.feature_tracker = SimplifiedFeatureTracker()
        self.action_mapper = StreetFighterDiscreteActions()

        # Observation space
        visual_space = gym.spaces.Box(
            low=0, high=255, shape=(3, SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.uint8
        )
        vector_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(5, VECTOR_FEATURE_DIM), dtype=np.float32
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

        # FIXED: Health tracking for single fight detection
        self.initial_player_health = MAX_HEALTH
        self.initial_opponent_health = MAX_HEALTH
        self.current_player_health = MAX_HEALTH
        self.current_opponent_health = MAX_HEALTH
        self.fight_ended = False

        # Known Street Fighter II RAM addresses (adjust for your specific ROM)
        self.player_health_addr = 0xFF8A  # Player 1 health
        self.opponent_health_addr = 0xFF8B  # Player 2 health

        print(f"🥊 FIXED StreetFighterVisionWrapper initialized")
        print(f"   - Action space: {self.action_space.n} discrete actions")
        print(f"   - Max steps per fight: {MAX_FIGHT_STEPS}")
        print(f"   - Health detection: RAM addresses 0xFF8A, 0xFF8B")

    def reset(self, **kwargs):
        """Reset environment and trackers."""
        obs, info = self.env.reset(**kwargs)

        # Reset trackers
        self.reward_calculator.reset()
        self.feature_tracker.reset()
        self.vector_history.clear()

        self.episode_count += 1
        self.step_count = 0
        self.fight_ended = False

        # FIXED: Initialize health values properly
        self.initial_player_health = MAX_HEALTH
        self.initial_opponent_health = MAX_HEALTH
        self.current_player_health = MAX_HEALTH
        self.current_opponent_health = MAX_HEALTH

        # Try to get actual health values
        player_health, opponent_health = self._extract_health_fixed()
        if player_health > 0:
            self.initial_player_health = player_health
            self.current_player_health = player_health
        if opponent_health > 0:
            self.initial_opponent_health = opponent_health
            self.current_opponent_health = opponent_health

        print(
            f"🔄 Episode {self.episode_count} reset - Initial health: P1={self.current_player_health}, P2={self.current_opponent_health}"
        )

        # Initialize feature tracker
        self.feature_tracker.update(
            self.current_player_health, self.current_opponent_health, 0, {}
        )

        # Build observation
        observation = self._build_observation(obs, info)

        return observation, info

    def step(self, action):
        """Execute action and return enhanced observation with FIXED single fight detection."""
        self.step_count += 1

        # Convert discrete action to button combination
        button_combination = self.action_mapper.get_action(action)

        # Execute action in environment
        obs, reward, done, truncated, info = self.env.step(action)

        # FIXED: Extract health values with multiple methods
        player_health, opponent_health = self._extract_health_fixed()

        # Update current health
        prev_player_health = self.current_player_health
        prev_opponent_health = self.current_opponent_health
        self.current_player_health = player_health
        self.current_opponent_health = opponent_health

        # FIXED: Detect fight end conditions
        fight_ended_reason = None

        # Method 1: Health dropped to 0 or below
        if prev_player_health > 0 and player_health <= 0:
            done = True
            fight_ended_reason = "player_defeated"
        elif prev_opponent_health > 0 and opponent_health <= 0:
            done = True
            fight_ended_reason = "opponent_defeated"

        # Method 2: Significant health drop (KO detection)
        elif prev_player_health - player_health >= 50:  # Large damage = KO
            done = True
            fight_ended_reason = "player_ko"
        elif prev_opponent_health - opponent_health >= 50:  # Large damage = KO
            done = True
            fight_ended_reason = "opponent_ko"

        # Method 3: Time limit per fight
        elif self.step_count >= MAX_FIGHT_STEPS:
            done = True
            fight_ended_reason = "timeout"

        # Debug logging every 100 steps
        if self.step_count % 100 == 0:
            print(
                f"Step {self.step_count}: P1={player_health}, P2={opponent_health}, Done={done}"
            )

        if done and fight_ended_reason:
            print(
                f"🥊 Fight ended ({fight_ended_reason}): P1={player_health}, P2={opponent_health} after {self.step_count} steps"
            )

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
                "fight_ended_reason": fight_ended_reason,
            }
        )

        return observation, intelligent_reward, done, truncated, info

    def _extract_health_fixed(self):
        """FIXED health extraction with multiple fallback methods."""
        player_health = self.current_player_health  # Default to last known
        opponent_health = self.current_opponent_health  # Default to last known

        try:
            # Method 1: Try RAM access through unwrapped environment
            if hasattr(self.env, "unwrapped") and hasattr(self.env.unwrapped, "data"):
                ram_data = self.env.unwrapped.data
                if isinstance(ram_data, (np.ndarray, list)) and len(ram_data) > max(
                    self.player_health_addr, self.opponent_health_addr
                ):
                    p_health = int(ram_data[self.player_health_addr])
                    o_health = int(ram_data[self.opponent_health_addr])
                    if 0 <= p_health <= MAX_HEALTH:
                        player_health = p_health
                    if 0 <= o_health <= MAX_HEALTH:
                        opponent_health = o_health
        except Exception as e:
            pass

        try:
            # Method 2: Try different RAM addresses (common Street Fighter addresses)
            if hasattr(self.env, "unwrapped"):
                ram = self.env.unwrapped.data
                # Try multiple known addresses
                addresses = [
                    (0x8004, 0x8008),  # Common SF2 addresses
                    (0xFF8A, 0xFF8B),  # Alternative addresses
                    (0x8A, 0x8B),  # Short addresses
                ]

                for p_addr, o_addr in addresses:
                    try:
                        if len(ram) > max(p_addr, o_addr):
                            p_h = int(ram[p_addr])
                            o_h = int(ram[o_addr])
                            if 0 <= p_h <= MAX_HEALTH and 0 <= o_h <= MAX_HEALTH:
                                player_health = p_h
                                opponent_health = o_h
                                break
                    except:
                        continue
        except:
            pass

        # Ensure values are in valid range
        player_health = max(0, min(player_health, MAX_HEALTH))
        opponent_health = max(0, min(opponent_health, MAX_HEALTH))

        return player_health, opponent_health

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
        self.min_quality_for_golden = 0.6  # FIXED: Reasonable threshold
        self.peak_win_rate_threshold = 0.3  # FIXED: Lower threshold
        self.current_win_rate = 0.0

        print(f"🏆 FIXED Golden Experience Buffer initialized")
        print(f"   - Capacity: {capacity}")
        print(f"   - Min quality for golden: {self.min_quality_for_golden}")
        print(f"   - Peak win rate threshold: {self.peak_win_rate_threshold}")

    def update_win_rate(self, win_rate):
        """Update current win rate for filtering experiences."""
        self.current_win_rate = win_rate

    def add_experience(self, experience, quality_score):
        """Add experience to golden buffer if it meets criteria."""
        # FIXED: More lenient criteria for golden experiences
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


class FixedQualityBasedExperienceBuffer:
    """🎯 FIXED quality-based experience buffer with proper thresholds."""

    def __init__(
        self, capacity=30000, quality_threshold=0.3, golden_buffer_capacity=1000
    ):  # FIXED: Lower threshold
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

        # FIXED: More conservative adaptive parameters
        self.adjustment_rate = 0.02  # Slower adjustments
        self.threshold_adjustment_frequency = 50  # Less frequent adjustments
        self.threshold_adjustments = 0

        print(f"🎯 FIXED Quality-Based Experience Buffer initialized")
        print(
            f"   - Capacity: {capacity:,} (Good: {capacity//2:,}, Bad: {capacity//2:,})"
        )
        print(f"   - Quality threshold: {quality_threshold} (LOWERED)")
        print(f"   - Golden buffer capacity: {golden_buffer_capacity:,}")

    def add_experience(self, experience, reward, reward_breakdown, quality_score):
        """Add experience to appropriate buffer based on quality."""
        self.total_added += 1
        self.quality_scores.append(quality_score)

        # FIXED: More logging for debugging
        if self.total_added % 100 == 0:
            print(
                f"📊 Added {self.total_added} experiences. Quality: {quality_score:.3f}, Threshold: {self.quality_threshold:.3f}"
            )

        # Classify experience
        if quality_score >= self.quality_threshold:
            self.good_experiences.append(experience)
            print(f"✅ Good experience added (quality: {quality_score:.3f})")

            # Add to golden buffer if it meets higher criteria
            self.golden_buffer.add_experience(experience, quality_score)
        else:
            self.bad_experiences.append(experience)

    def update_win_rate(self, win_rate):
        """Update win rate for golden buffer filtering."""
        self.golden_buffer.update_win_rate(win_rate)

    def sample_enhanced_balanced_batch(self, batch_size, golden_ratio=0.15):
        """Sample balanced batch with golden experiences included."""
        # FIXED: More lenient batch requirements
        min_good_needed = max(1, batch_size // 8)  # Need at least 1/8 of batch size
        min_bad_needed = max(1, batch_size // 8)

        if (
            len(self.good_experiences) < min_good_needed
            or len(self.bad_experiences) < min_bad_needed
        ):
            print(
                f"⚠️ Not enough experiences: Good={len(self.good_experiences)}, Bad={len(self.bad_experiences)}"
            )
            return None, None, None

        # Calculate batch composition
        golden_count = min(
            int(batch_size * golden_ratio), len(self.golden_buffer.experiences)
        )
        good_count = min(batch_size // 2, len(self.good_experiences))
        remaining_good = max(0, good_count - golden_count)
        bad_count = min(batch_size - good_count, len(self.bad_experiences))

        # Sample golden experiences
        golden_batch = (
            self.golden_buffer.sample_golden_batch(golden_count)
            if golden_count > 0
            else []
        )

        # Sample regular good experiences
        if remaining_good > 0:
            good_indices = np.random.choice(
                len(self.good_experiences), remaining_good, replace=False
            )
            good_batch = [self.good_experiences[i] for i in good_indices]
        else:
            good_batch = []

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

    def adjust_threshold(
        self, target_good_ratio=0.4, episode_number=0
    ):  # FIXED: Lower target
        """More conservative threshold adjustment."""
        # Only adjust every N episodes
        if episode_number % self.threshold_adjustment_frequency != 0:
            return

        if len(self.quality_scores) < 50:  # Need less data
            return

        total_size = len(self.good_experiences) + len(self.bad_experiences)
        current_good_ratio = len(self.good_experiences) / max(1, total_size)

        # More gradual adjustments with wider tolerance
        if current_good_ratio < target_good_ratio - 0.2:  # Much wider tolerance
            old_threshold = self.quality_threshold
            self.quality_threshold *= 1 - self.adjustment_rate
            self.threshold_adjustments += 1
            print(
                f"📉 Lowered quality threshold: {old_threshold:.3f} → {self.quality_threshold:.3f}"
            )
        elif current_good_ratio > target_good_ratio + 0.3:  # Much wider tolerance
            old_threshold = self.quality_threshold
            self.quality_threshold *= 1 + self.adjustment_rate
            self.threshold_adjustments += 1
            print(
                f"📈 Raised quality threshold: {old_threshold:.3f} → {self.quality_threshold:.3f}"
            )

        # Keep threshold in reasonable range
        self.quality_threshold = max(0.1, min(0.7, self.quality_threshold))


class PolicyMemoryManager:
    """🧠 Policy Memory Manager - Implements checkpoint averaging and policy memory."""

    def __init__(
        self, performance_drop_threshold=0.08, averaging_weight=0.7
    ):  # FIXED: Higher threshold
        self.performance_drop_threshold = performance_drop_threshold
        self.averaging_weight = averaging_weight

        # Track peak performance
        self.peak_win_rate = 0.0
        self.peak_checkpoint_state = None
        self.episodes_since_peak = 0
        self.performance_drop_detected = False

        # Learning rate memory
        self.peak_lr = None
        self.lr_reduction_factor = 0.7  # FIXED: Less aggressive reduction
        self.min_lr = 5e-7  # FIXED: Higher minimum

        # Averaging history
        self.averaging_performed = 0
        self.last_averaging_episode = -1

        print(f"🧠 FIXED Policy Memory Manager initialized")
        print(f"   - Performance drop threshold: {performance_drop_threshold}")
        print(f"   - Averaging weight: {averaging_weight}")
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

            # FIXED: More lenient drop detection
            if (
                current_win_rate < self.peak_win_rate - self.performance_drop_threshold
                and not self.performance_drop_detected
                and self.episodes_since_peak > 15  # FIXED: More episodes needed
                and self.peak_win_rate > 0.1
            ):  # FIXED: Only if we had some success

                print(f"📉 PERFORMANCE DROP DETECTED!")
                print(
                    f"   Current: {current_win_rate:.3f} | Peak: {self.peak_win_rate:.3f}"
                )
                print(f"   Drop: {self.peak_win_rate - current_win_rate:.3f}")
                self.performance_drop_detected = True
                performance_drop = True

        return performance_improved, performance_drop

    def should_perform_averaging(self, current_episode):
        """Check if we should perform checkpoint averaging."""
        return (
            self.performance_drop_detected
            and self.peak_checkpoint_state is not None
            and current_episode - self.last_averaging_episode > 10
        )  # FIXED: More episodes between averaging

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


class FixedEnergyBasedStreetFighterCNN(nn.Module):
    """🛡️ FIXED CNN with better stability and regularization."""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__()

        visual_space = observation_space["visual_obs"]
        vector_space = observation_space["vector_obs"]
        n_input_channels = visual_space.shape[0]
        seq_length, vector_feature_count = vector_space.shape

        print(f"🔧 FIXED Energy-Based CNN Configuration:")
        print(f"   - Visual channels: {n_input_channels}")
        print(f"   - Output features: {features_dim}")

        # FIXED: More conservative CNN architecture
        self.visual_cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.1),  # FIXED: Reduced dropout
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((3, 4)),
            nn.Flatten(),
        )

        # Calculate visual output size
        with torch.no_grad():
            dummy_visual = torch.zeros(
                1, n_input_channels, visual_space.shape[1], visual_space.shape[2]
            )
            visual_output_size = self.visual_cnn(dummy_visual).shape[1]

        # FIXED: Simpler vector processing
        self.vector_embed = nn.Linear(vector_feature_count, 32)
        self.vector_gru = nn.GRU(32, 32, batch_first=True)
        self.vector_final = nn.Linear(32, 32)

        # FIXED: Simpler fusion layer
        fusion_input_size = visual_output_size + 32
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),  # FIXED: Reduced dropout
            nn.Linear(512, features_dim),
            nn.ReLU(inplace=True),
        )

        # FIXED: More conservative initialization
        self.apply(self._init_weights)

        print(f"   ✅ FIXED Energy-Based Feature Extractor initialized")

    def _init_weights(self, m):
        """FIXED: More conservative weight initialization."""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.5)  # FIXED: More conservative gain
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param, gain=0.5)
                elif "bias" in name:
                    nn.init.constant_(param, 0)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        visual_obs = observations["visual_obs"]
        vector_obs = observations["vector_obs"]
        device = next(self.parameters()).device

        visual_obs = visual_obs.float().to(device)
        vector_obs = vector_obs.float().to(device)

        # FIXED: Better input normalization
        visual_obs = torch.clamp(visual_obs / 255.0, 0.0, 1.0)
        vector_obs = torch.clamp(vector_obs, -5.0, 5.0)  # FIXED: Tighter bounds

        # Process visual features
        visual_features = self.visual_cnn(visual_obs)
        visual_features = torch.clamp(
            visual_features, -10.0, 10.0
        )  # FIXED: Clamp intermediate

        # Process vector features
        batch_size, seq_len, feature_dim = vector_obs.shape
        vector_embedded = self.vector_embed(vector_obs)
        gru_output, _ = self.vector_gru(vector_embedded)
        vector_features = self.vector_final(gru_output[:, -1, :])

        # Combine and process
        combined_features = torch.cat([visual_features, vector_features], dim=1)
        output = self.fusion(combined_features)

        # FIXED: Final clamping
        output = torch.clamp(output, -8.0, 8.0)

        return output


class FixedEnergyBasedStreetFighterVerifier(nn.Module):
    """🛡️ FIXED Energy-Based Verifier with better stability."""

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

        # FIXED: Feature extractor
        self.features_extractor = FixedEnergyBasedStreetFighterCNN(
            observation_space, features_dim
        )

        # FIXED: Simpler action embedding
        self.action_embed = nn.Sequential(
            nn.Linear(self.action_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),  # FIXED: Reduced dropout
        )

        # FIXED: Simpler energy network
        self.energy_net = nn.Sequential(
            nn.Linear(features_dim + 64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

        # FIXED: More conservative energy parameters
        self.energy_scale = 1.0  # FIXED: Normal scale
        self.energy_clamp_min = -4.0  # FIXED: Reasonable bounds
        self.energy_clamp_max = 4.0

        # Initialize conservatively
        self.apply(self._init_weights)

        print(f"✅ FIXED EnergyBasedStreetFighterVerifier initialized")
        print(f"   - Energy scale: {self.energy_scale}")
        print(f"   - Energy bounds: [{self.energy_clamp_min}, {self.energy_clamp_max}]")

    def _init_weights(self, m):
        """FIXED: Conservative weight initialization."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.5)  # FIXED: Conservative gain
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(
        self, context: torch.Tensor, candidate_action: torch.Tensor
    ) -> torch.Tensor:
        """FIXED: Energy calculation with better stability."""
        device = next(self.parameters()).device

        # Get context features
        if isinstance(context, dict):
            context_features = self.features_extractor(context)
        else:
            context_features = context

        context_features = context_features.to(device)
        candidate_action = candidate_action.to(device)

        # FIXED: Better input validation and clamping
        context_features = torch.clamp(context_features, -10.0, 10.0)
        candidate_action = torch.clamp(candidate_action, 0.0, 1.0)

        # Embed action
        action_embedded = self.action_embed(candidate_action)
        action_embedded = torch.clamp(action_embedded, -5.0, 5.0)  # FIXED: Clamp

        # Combine features
        combined_input = torch.cat([context_features, action_embedded], dim=-1)

        # Calculate energy
        raw_energy = self.energy_net(combined_input)
        energy = raw_energy * self.energy_scale

        # FIXED: Final clamping
        energy = torch.clamp(energy, self.energy_clamp_min, self.energy_clamp_max)

        return energy

    def get_energy_stats(self):
        """Get basic energy statistics."""
        return {
            "energy_scale": self.energy_scale,
            "energy_bounds": (self.energy_clamp_min, self.energy_clamp_max),
        }


class FixedStabilizedEnergyBasedAgent:
    """🛡️ FIXED Energy-Based Agent with better stability."""

    def __init__(
        self,
        verifier: FixedEnergyBasedStreetFighterVerifier,
        thinking_steps: int = 2,
        thinking_lr: float = 0.03,
        noise_scale: float = 0.01,
    ):  # FIXED: Reduced params
        self.verifier = verifier
        self.initial_thinking_steps = thinking_steps
        self.current_thinking_steps = thinking_steps
        self.initial_thinking_lr = thinking_lr
        self.current_thinking_lr = thinking_lr
        self.noise_scale = noise_scale
        self.action_dim = verifier.action_dim

        # FIXED: More conservative parameters
        self.gradient_clip = 0.5  # FIXED: Higher clip
        self.early_stop_patience = 3
        self.min_energy_improvement = 1e-3

        # Statistics
        self.thinking_stats = {
            "total_predictions": 0,
            "avg_thinking_steps": 0.0,
            "successful_optimizations": 0,
            "failed_optimizations": 0,
        }

        print(f"✅ FIXED StabilizedEnergyBasedAgent initialized")
        print(f"   - Thinking steps: {thinking_steps}")
        print(f"   - Thinking LR: {thinking_lr}")
        print(f"   - Gradient clip: {self.gradient_clip}")

    def predict(
        self, observations: Dict[str, torch.Tensor], deterministic: bool = False
    ) -> Tuple[int, Dict]:
        """FIXED: More stable action prediction."""
        device = next(self.verifier.parameters()).device

        # Prepare observations
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

        # FIXED: Initialize with uniform distribution
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

        # FIXED: Simpler thinking loop
        steps_taken = 0
        optimization_successful = True

        try:
            for step in range(self.current_thinking_steps):
                # Calculate energy
                energy = self.verifier(obs_device, candidate_action)

                # Calculate gradient
                gradients = torch.autograd.grad(
                    outputs=energy.sum(),
                    inputs=candidate_action,
                    create_graph=False,
                    retain_graph=False,
                )[0]

                # FIXED: Gradient clipping
                grad_norm = torch.norm(gradients).item()
                if grad_norm > self.gradient_clip:
                    gradients = gradients * (self.gradient_clip / grad_norm)

                # Update action
                with torch.no_grad():
                    candidate_action = (
                        candidate_action - self.current_thinking_lr * gradients
                    )
                    candidate_action = F.softmax(candidate_action, dim=-1)
                    candidate_action.requires_grad_(True)

                steps_taken += 1

        except Exception as e:
            print(f"⚠️ Thinking step failed: {e}")
            optimization_successful = False

        # Final action selection
        with torch.no_grad():
            final_action_probs = F.softmax(candidate_action, dim=-1)

            if deterministic:
                action_idx = torch.argmax(final_action_probs, dim=-1)
            else:
                action_idx = torch.multinomial(final_action_probs, 1).squeeze(-1)

        # Update statistics
        self.thinking_stats["total_predictions"] += 1
        if optimization_successful:
            self.thinking_stats["successful_optimizations"] += 1
        else:
            self.thinking_stats["failed_optimizations"] += 1

        thinking_info = {
            "steps_taken": steps_taken,
            "optimization_successful": optimization_successful,
            "current_thinking_steps": self.current_thinking_steps,
            "current_thinking_lr": self.current_thinking_lr,
        }

        return action_idx.item() if batch_size == 1 else action_idx, thinking_info

    def get_thinking_stats(self) -> Dict:
        """Get thinking statistics."""
        stats = self.thinking_stats.copy()
        if stats["total_predictions"] > 0:
            stats["success_rate"] = (
                stats["successful_optimizations"] / stats["total_predictions"]
            )
        else:
            stats["success_rate"] = 0.0
        return stats


class FixedEnergyStabilityManager:
    """🛡️ FIXED Energy Stability Manager."""

    def __init__(
        self, initial_lr=1e-5, thinking_lr=0.03, policy_memory_manager=None
    ):  # FIXED: Lower LR
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.thinking_lr = thinking_lr
        self.current_thinking_lr = thinking_lr
        self.policy_memory_manager = policy_memory_manager

        # FIXED: More lenient thresholds
        self.min_win_rate = 0.1  # FIXED: Much lower
        self.consecutive_poor_episodes = 0
        self.emergency_mode = False

        print(f"🛡️ FIXED EnergyStabilityManager initialized")
        print(f"   - Initial LR: {initial_lr:.2e}")
        print(f"   - Min win rate threshold: {self.min_win_rate}")

    def update_metrics(
        self, win_rate, energy_quality, energy_separation, early_stop_rate
    ):
        """FIXED: More lenient stability checking."""
        # Only trigger emergency if win rate is extremely low for extended period
        if win_rate < self.min_win_rate:
            self.consecutive_poor_episodes += 1
            if self.consecutive_poor_episodes >= 20:  # FIXED: Much higher threshold
                print(
                    f"🚨 STABILITY EMERGENCY: {self.consecutive_poor_episodes} poor episodes"
                )
                return self._trigger_emergency_protocol()
        else:
            self.consecutive_poor_episodes = 0
            self.emergency_mode = False

        return False

    def _trigger_emergency_protocol(self):
        """FIXED: More conservative emergency response."""
        if not self.emergency_mode:
            # Reduce learning rates more conservatively
            self.current_lr *= 0.8  # FIXED: Less aggressive reduction
            self.current_thinking_lr *= 0.8
            self.emergency_mode = True
            self.consecutive_poor_episodes = 0
            print(f"📉 Emergency LR reduction: {self.current_lr:.2e}")
            return True
        return False

    def get_current_lrs(self):
        """Get current learning rates."""
        return self.current_lr, self.current_thinking_lr


class FixedCheckpointManager:
    """FIXED checkpoint management."""

    def __init__(self, checkpoint_dir="checkpoints_fixed"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.best_checkpoint_path = None
        self.best_win_rate = 0.0

        print(f"💾 FIXED CheckpointManager initialized: {checkpoint_dir}")

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
        """Save checkpoint with metadata."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if is_emergency:
            filename = f"emergency_ep{episode}_{timestamp}.pt"
        elif is_peak:
            filename = f"peak_ep{episode}_wr{win_rate:.3f}_{timestamp}.pt"
        else:
            filename = f"checkpoint_ep{episode}_wr{win_rate:.3f}_{timestamp}.pt"

        checkpoint_path = self.checkpoint_dir / filename

        checkpoint_data = {
            "episode": episode,
            "win_rate": win_rate,
            "verifier_state_dict": verifier.state_dict(),
            "agent_thinking_stats": agent.get_thinking_stats(),
            "timestamp": timestamp,
            "is_emergency": is_emergency,
            "is_peak": is_peak,
            "policy_memory_stats": policy_memory_stats or {},
        }

        try:
            torch.save(checkpoint_data, checkpoint_path)

            if win_rate > self.best_win_rate:
                self.best_checkpoint_path = checkpoint_path
                self.best_win_rate = win_rate
                print(f"💾 New best checkpoint saved: {filename} (WR: {win_rate:.1%})")
            else:
                print(f"💾 Checkpoint saved: {filename}")

            return checkpoint_path

        except Exception as e:
            print(f"❌ Failed to save checkpoint: {e}")
            return None


# Utility functions
def verify_fixed_energy_flow(verifier, observation_space, action_space):
    """Verify energy flow works correctly."""
    print(f"🔍 Verifying FIXED energy flow...")

    try:
        visual_obs = torch.zeros(1, 3, SCREEN_HEIGHT, SCREEN_WIDTH)
        vector_obs = torch.zeros(1, 5, VECTOR_FEATURE_DIM)
        dummy_obs = {"visual_obs": visual_obs, "vector_obs": vector_obs}
        dummy_action = torch.ones(1, action_space.n) / action_space.n

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
    """Create FIXED Street Fighter environment."""
    print(f"🎮 Creating FIXED Street Fighter environment...")
    print(f"   - Game: {game}")
    print(f"   - State: {state}")
    print(f"   - Single fight mode: ✅ ENABLED")

    try:
        # Create base retro environment
        env = retro.make(
            game=game, state=state, use_restricted_actions=retro.Actions.DISCRETE
        )

        # Wrap with our FIXED vision wrapper
        env = FixedStreetFighterVisionWrapper(env)

        print(f"   ✅ FIXED environment created successfully")
        print(f"   - Observation space: {env.observation_space}")
        print(f"   - Action space: {env.action_space}")

        return env

    except Exception as e:
        print(f"   ❌ Environment creation failed: {e}")
        raise


# Export FIXED components
__all__ = [
    # FIXED core components
    "FixedStreetFighterVisionWrapper",
    "FixedEnergyBasedStreetFighterCNN",
    "FixedEnergyBasedStreetFighterVerifier",
    "FixedStabilizedEnergyBasedAgent",
    "SimplifiedFeatureTracker",
    "StreetFighterDiscreteActions",
    # FIXED reward and experience components
    "FixedIntelligentRewardCalculator",
    "FixedQualityBasedExperienceBuffer",
    "GoldenExperienceBuffer",
    # Policy memory components
    "PolicyMemoryManager",
    # FIXED stability components
    "FixedEnergyStabilityManager",
    "FixedCheckpointManager",
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
    "MAX_FIGHT_STEPS",
]

print(f"🥊 FIXED POLICY MEMORY - Complete wrapper.py loaded successfully!")
print(f"   - Training paradigm: FIXED Policy Memory + Golden Buffer")
print(f"   - Fight mode: SINGLE DECISIVE ROUNDS with proper health detection")
print(f"   - Quality threshold: LOWERED to 0.3 for learning")
print(f"   - Gradient stability: ENHANCED with better clipping")
print(f"   - Health detection: MULTIPLE fallback methods")
print(f"   - Max steps per fight: {MAX_FIGHT_STEPS} (~1 minute)")
print(f"🎯 Ready for stable learning with FIXED quality thresholds!")
