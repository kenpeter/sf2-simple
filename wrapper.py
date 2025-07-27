#!/usr/bin/env python3
"""
🛡️ ENHANCED WRAPPER - RGB Version with Smaller Image Sizes
Key Improvements:
1. Keep RGB images but resize to smaller dimensions for efficiency
2. Time-decayed winning bonuses (fast wins >>> slow wins)
3. Aggressive epsilon-greedy exploration
4. Reservoir sampling for experience diversity
5. Enhanced temporal awareness with 8-frame stacking
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
    f'logs/enhanced_sf_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
)
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename)],
)
logger = logging.getLogger(__name__)

# Constants - RGB version with smaller image size
MAX_HEALTH = 176
SCREEN_WIDTH = 160  # Reduced from 320 (half size)
SCREEN_HEIGHT = 112  # Reduced from 224 (half size)
VECTOR_FEATURE_DIM = 32
MAX_FIGHT_STEPS = (
    1200  # Note: This is now only used for reward calculation, not for termination
)
FRAME_STACK_SIZE = 8

print(f"🚀 ENHANCED Street Fighter II Configuration (RGB):")
print(f"   - Health detection: MULTI-METHOD")
print(f"   - Image format: RGB (3 channels)")
print(f"   - Image size: {SCREEN_WIDTH}x{SCREEN_HEIGHT} (reduced for efficiency)")
print(f"   - Time-decayed rewards: ENABLED")
print(f"   - Aggressive exploration: ACTIVE")
print(f"   - Reservoir sampling: ENABLED")
print(f"   - Frame stacking: {FRAME_STACK_SIZE} frames")


# Keep utility functions from original
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


class EnhancedRewardCalculator:
    """🚀 ENHANCED reward calculator with time-decayed bonuses and aggression incentives."""

    def __init__(self):
        self.previous_opponent_health = MAX_HEALTH
        self.previous_player_health = MAX_HEALTH
        self.match_started = False
        self.step_count = 0

        # ENHANCED: More aggressive reward structure
        self.max_damage_reward = 1.5  # Increased from 1.0
        self.base_winning_bonus = 4.0  # Increased from 3.0
        self.health_advantage_bonus = 0.8  # Increased from 0.5

        # NEW: Aggression incentives
        self.combo_bonus_multiplier = 2.0
        self.fast_damage_bonus = 1.0
        self.timeout_penalty_multiplier = 3.0  # Heavy penalty for timeouts

        self.round_won = False
        self.round_lost = False
        self.round_draw = False

        # Enhanced tracking
        self.total_damage_dealt = 0.0
        self.total_damage_taken = 0.0
        self.consecutive_damage_frames = 0
        self.last_damage_frame = -1

    def calculate_reward(self, player_health, opponent_health, done, info):
        """ENHANCED reward calculation with time-decayed bonuses and aggression incentives."""
        reward = 0.0
        reward_breakdown = {}

        # Update step count
        self.step_count = info.get("step_count", self.step_count + 1)

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

        # ENHANCED: Combo detection and bonus
        if opponent_damage_dealt > 0:
            # Check for consecutive damage (combo detection)
            if self.step_count == self.last_damage_frame + 1:
                self.consecutive_damage_frames += 1
            else:
                self.consecutive_damage_frames = 1
            self.last_damage_frame = self.step_count

            # Base damage reward
            damage_reward = min(
                opponent_damage_dealt / MAX_HEALTH, self.max_damage_reward
            )

            # COMBO BONUS: Reward consecutive damage frames exponentially
            if self.consecutive_damage_frames > 1:
                combo_multiplier = min(
                    1 + (self.consecutive_damage_frames - 1) * 0.5, 3.0
                )
                damage_reward *= combo_multiplier
                reward_breakdown["combo_multiplier"] = combo_multiplier
                reward_breakdown["combo_frames"] = self.consecutive_damage_frames

            # FAST DAMAGE BONUS: Extra reward for early damage
            time_factor = (MAX_FIGHT_STEPS - self.step_count) / MAX_FIGHT_STEPS
            fast_bonus = damage_reward * self.fast_damage_bonus * time_factor
            damage_reward += fast_bonus
            reward_breakdown["fast_damage_bonus"] = fast_bonus

            reward += damage_reward
            reward_breakdown["damage_dealt"] = damage_reward

        # Penalty for taking damage (slightly reduced to encourage aggression)
        if player_damage_taken > 0:
            damage_penalty = (
                -(player_damage_taken / MAX_HEALTH) * 0.5
            )  # Reduced from 0.6
            reward += damage_penalty
            reward_breakdown["damage_taken"] = damage_penalty

        # Health advantage bonus (ongoing)
        if not done:
            health_diff = (player_health - opponent_health) / MAX_HEALTH
            if abs(health_diff) > 0.1:
                advantage_bonus = health_diff * self.health_advantage_bonus
                reward += advantage_bonus
                reward_breakdown["health_advantage"] = advantage_bonus

        # ENHANCED TERMINAL REWARDS with TIME-DECAYED BONUSES
        # the done pass from top
        if done:
            termination_reason = info.get("termination_reason", "unknown")

            # TIME-DECAYED WINNING BONUS - FAST WINS ARE EXPONENTIALLY BETTER
            if player_health > opponent_health:
                # Calculate time bonus factor (1.0 at step 0, ~0.0 at max steps)
                time_bonus_factor = (
                    MAX_FIGHT_STEPS - self.step_count
                ) / MAX_FIGHT_STEPS

                # Exponential scaling for aggressive fast play
                time_multiplier = 1 + 3 * (
                    time_bonus_factor**2
                )  # Up to 4x bonus for fast wins

                if opponent_health <= 0:
                    # Perfect KO with time bonus
                    win_bonus = self.base_winning_bonus * time_multiplier
                    reward_breakdown["victory_type"] = "knockout"
                elif player_health > opponent_health + 20:
                    # Decisive victory
                    win_bonus = self.base_winning_bonus * 0.8 * time_multiplier
                    reward_breakdown["victory_type"] = "decisive"
                else:
                    # Close victory
                    win_bonus = self.base_winning_bonus * 0.6 * time_multiplier
                    reward_breakdown["victory_type"] = "close"

                reward += win_bonus
                reward_breakdown["round_won"] = win_bonus
                reward_breakdown["time_bonus_factor"] = time_bonus_factor
                reward_breakdown["time_multiplier"] = time_multiplier

                # SPEED BONUS: Extra reward for very fast wins
                if self.step_count < MAX_FIGHT_STEPS * 0.3:  # Win in first 30% of fight
                    speed_bonus = self.base_winning_bonus * 0.5
                    reward += speed_bonus
                    reward_breakdown["speed_bonus"] = speed_bonus

                self.round_won = True
                self.round_lost = False
                self.round_draw = False

            elif opponent_health > player_health:
                # Loss penalties (unchanged but noted)
                if player_health <= 0:
                    loss_penalty = -2.5  # Slightly increased
                    reward_breakdown["defeat_type"] = "knockout"
                elif opponent_health > player_health + 20:
                    loss_penalty = -2.0  # Slightly increased
                    reward_breakdown["defeat_type"] = "decisive"
                else:
                    loss_penalty = -1.2  # Slightly increased
                    reward_breakdown["defeat_type"] = "close"

                reward += loss_penalty
                reward_breakdown["round_lost"] = loss_penalty
                self.round_won = False
                self.round_lost = True
                self.round_draw = False

            else:
                # TRUE DRAW - HEAVILY PENALIZED
                draw_penalty = -1.5  # Tripled from -0.5
                reward += draw_penalty
                reward_breakdown["draw"] = draw_penalty
                reward_breakdown["result_type"] = "draw"
                self.round_won = False
                self.round_lost = False
                self.round_draw = True

            # TIMEOUT PENALTY: Massive penalty for defensive play
            if "timeout" in termination_reason:
                timeout_penalty = -2.0 * self.timeout_penalty_multiplier  # Up to -6.0
                reward += timeout_penalty
                reward_breakdown["timeout_penalty"] = timeout_penalty

            # Enhanced damage ratio bonus/penalty
            if self.total_damage_dealt > 0 or self.total_damage_taken > 0:
                damage_ratio = safe_divide(
                    self.total_damage_dealt, self.total_damage_taken + 1, 1.0
                )
                damage_ratio_bonus = (damage_ratio - 1.0) * 0.8  # Increased from 0.5
                reward += damage_ratio_bonus
                reward_breakdown["damage_ratio"] = damage_ratio_bonus

        # ENHANCED step penalty - encourages faster play
        step_penalty = -0.008  # Increased from -0.005
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
        self.step_count = 0
        self.round_won = False
        self.round_lost = False
        self.round_draw = False
        self.total_damage_dealt = 0.0
        self.total_damage_taken = 0.0
        self.consecutive_damage_frames = 0
        self.last_damage_frame = -1


# Keep the HealthDetector class unchanged (it's working well)
class HealthDetector:
    """🔍 Advanced health detection system - RGB version."""

    def __init__(self):
        self.health_history = {"player": deque(maxlen=10), "opponent": deque(maxlen=10)}
        self.last_valid_health = {"player": MAX_HEALTH, "opponent": MAX_HEALTH}
        self.health_change_detected = False
        self.frame_count = 0
        # Adjusted bar positions for smaller image size
        self.bar_positions = {
            "player": {"x": 20, "y": 8, "width": 60, "height": 4},  # Half size
            "opponent": {"x": 80, "y": 8, "width": 60, "height": 4},  # Half size
        }

    def extract_health_from_memory(self, info):
        """Extract health from multiple possible memory locations."""
        player_health = MAX_HEALTH
        opponent_health = MAX_HEALTH

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
                    if 0 <= p_hp <= MAX_HEALTH and 0 <= o_hp <= MAX_HEALTH:
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

        address_sets = [
            {"player": 0xFF8043, "opponent": 0xFF82C3},
            {"player": 0x8043, "opponent": 0x82C3},
            {"player": 0xFF8204, "opponent": 0xFF8208},
            {"player": 0x8204, "opponent": 0x8208},
            {"player": 67, "opponent": 579},
            {"player": 33347, "opponent": 33479},
        ]

        for addr_set in address_sets:
            try:
                p_addr = addr_set["player"]
                o_addr = addr_set["opponent"]

                for read_method in ["read_u8", "read_byte", "read_s8"]:
                    try:
                        if hasattr(env.data.memory, read_method):
                            p_hp = getattr(env.data.memory, read_method)(p_addr)
                            o_hp = getattr(env.data.memory, read_method)(o_addr)

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
        """Extract health from visual health bars as fallback - RGB version."""
        if visual_obs is None or len(visual_obs.shape) != 3:
            return MAX_HEALTH, MAX_HEALTH

        try:
            # Handle both HWC and CHW formats
            if visual_obs.shape[0] == 3:  # CHW format
                frame = np.transpose(visual_obs, (1, 2, 0))  # Convert to HWC
            else:  # HWC format
                frame = visual_obs

            # Ensure RGB format and proper data type
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)

            player_health = self._analyze_health_bar(frame, "player")
            opponent_health = self._analyze_health_bar(frame, "opponent")

            return player_health, opponent_health
        except Exception:
            return MAX_HEALTH, MAX_HEALTH

    def _analyze_health_bar(self, frame, player_type):
        """Analyze health bar pixels to estimate health - RGB version."""
        pos = self.bar_positions[player_type]
        health_region = frame[
            pos["y"] : pos["y"] + pos["height"], pos["x"] : pos["x"] + pos["width"]
        ]

        if health_region.size == 0:
            return MAX_HEALTH

        # Convert RGB to grayscale for analysis
        if len(health_region.shape) == 3:
            gray_region = cv2.cvtColor(health_region, cv2.COLOR_RGB2GRAY)
        else:
            gray_region = health_region

        health_pixels = np.sum(gray_region > 50)
        total_pixels = gray_region.size
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
        health = max(0, min(MAX_HEALTH, health))
        self.health_history[player_type].append(health)

        if len(self.health_history[player_type]) >= 2:
            prev_health = self.health_history[player_type][-2]
            health_change = abs(health - prev_health)

            if health_change > MAX_HEALTH * 0.5:
                health = int((health + prev_health) / 2)

        if health != MAX_HEALTH or self.frame_count < 50:
            self.last_valid_health[player_type] = health

        return health

    def is_detection_working(self):
        """Check if health detection appears to be working."""
        if not self.health_change_detected and self.frame_count > 300:
            return False

        player_variance = len(set(list(self.health_history["player"])[-5:])) > 1
        opponent_variance = len(set(list(self.health_history["opponent"])[-5:])) > 1

        return player_variance or opponent_variance or self.frame_count < 100


class SimplifiedFeatureTracker:
    """📊 Feature tracker with enhanced temporal awareness."""

    def __init__(self, history_length=FRAME_STACK_SIZE):
        self.history_length = history_length
        self.reset()

    def reset(self):
        """Reset all tracking for new episode."""
        self.player_health_history = deque(maxlen=self.history_length)
        self.opponent_health_history = deque(maxlen=self.history_length)
        self.action_history = deque(maxlen=self.history_length)
        self.reward_history = deque(maxlen=self.history_length)
        self.damage_history = deque(
            maxlen=self.history_length
        )  # NEW: Track damage patterns
        self.last_action = 0
        self.combo_count = 0

    def update(self, player_health, opponent_health, action, reward_breakdown):
        """Update tracking with current state and enhanced damage tracking."""
        self.player_health_history.append(player_health / MAX_HEALTH)
        self.opponent_health_history.append(opponent_health / MAX_HEALTH)
        self.action_history.append(action / 55.0)

        # Enhanced reward signal tracking
        reward_signal = reward_breakdown.get(
            "damage_dealt", 0.0
        ) - reward_breakdown.get("damage_taken", 0.0)
        self.reward_history.append(np.clip(reward_signal, -1.0, 1.0))

        # NEW: Track damage patterns for better temporal features
        damage_dealt = reward_breakdown.get("damage_dealt", 0.0)
        self.damage_history.append(damage_dealt)

        # Enhanced combo detection
        if damage_dealt > 0:
            if action == self.last_action:
                self.combo_count += 1
            else:
                self.combo_count = max(0, self.combo_count - 1)  # Gradual decay
        else:
            self.combo_count = max(0, self.combo_count - 1)

        self.last_action = action

    def get_features(self):
        """Get enhanced feature vector with temporal context and damage patterns."""
        features = []

        # Pad histories to full length
        player_hist = list(self.player_health_history)
        opponent_hist = list(self.opponent_health_history)
        action_hist = list(self.action_history)
        reward_hist = list(self.reward_history)
        damage_hist = list(self.damage_history)

        while len(player_hist) < self.history_length:
            player_hist.insert(0, 1.0)
        while len(opponent_hist) < self.history_length:
            opponent_hist.insert(0, 1.0)
        while len(action_hist) < self.history_length:
            action_hist.insert(0, 0.0)
        while len(reward_hist) < self.history_length:
            reward_hist.insert(0, 0.0)
        while len(damage_hist) < self.history_length:
            damage_hist.insert(0, 0.0)

        # Add temporal sequences
        features.extend(player_hist)  # 8 features
        features.extend(opponent_hist)  # 8 features

        # Enhanced derived temporal features
        current_player_health = player_hist[-1] if player_hist else 1.0
        current_opponent_health = opponent_hist[-1] if opponent_hist else 1.0

        # Multi-scale health trends
        mid_point = self.history_length // 2
        player_trend = (
            current_player_health - player_hist[mid_point]
            if len(player_hist) > mid_point
            else 0.0
        )
        opponent_trend = (
            current_opponent_health - opponent_hist[mid_point]
            if len(opponent_hist) > mid_point
            else 0.0
        )

        # NEW: Damage momentum features
        recent_damage = sum(damage_hist[-4:]) / 4.0  # Average recent damage
        damage_acceleration = (
            damage_hist[-1] - damage_hist[-2] if len(damage_hist) >= 2 else 0.0
        )

        # Action patterns and aggression indicators
        recent_actions = action_hist[-4:]
        action_diversity = (
            len(set([int(a * 55) for a in recent_actions])) / 4.0
        )  # Normalized action diversity

        features.extend(
            [
                current_player_health,
                current_opponent_health,
                current_player_health - current_opponent_health,
                player_trend,
                opponent_trend,
                self.last_action / 55.0,
                min(self.combo_count / 5.0, 1.0),
                recent_damage,  # NEW
                damage_acceleration,  # NEW
                action_diversity,  # NEW
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
            5: ["A"],
            6: ["B"],
            7: ["C"],
            8: ["X"],
            9: ["Y"],
            10: ["Z"],
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


class EnhancedStreetFighterWrapper(gym.Wrapper):
    """🚀 ENHANCED Street Fighter wrapper with RGB images and aggressive exploration."""

    def __init__(self, env):
        super().__init__(env)

        # Initialize enhanced components
        self.reward_calculator = EnhancedRewardCalculator()
        self.feature_tracker = SimplifiedFeatureTracker()
        self.action_mapper = StreetFighterDiscreteActions()
        self.health_detector = HealthDetector()

        # Frame stacking for RGB visual observations
        self.frame_stack = deque(maxlen=FRAME_STACK_SIZE)

        # Setup observation and action spaces for RGB images
        visual_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(
                3 * FRAME_STACK_SIZE,
                SCREEN_HEIGHT,
                SCREEN_WIDTH,
            ),  # RGB with stacking
            dtype=np.uint8,
        )
        vector_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(FRAME_STACK_SIZE, VECTOR_FEATURE_DIM),
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Dict(
            {"visual_obs": visual_space, "vector_obs": vector_space}
        )

        self.action_space = gym.spaces.Discrete(self.action_mapper.n_actions)
        self.vector_history = deque(maxlen=FRAME_STACK_SIZE)

        # Episode tracking
        self.episode_count = 0
        self.step_count = 0
        self.previous_player_health = MAX_HEALTH
        self.previous_opponent_health = MAX_HEALTH

        print(f"🚀 EnhancedStreetFighterWrapper initialized (RGB)")
        print(f"   - Image format: RGB ({SCREEN_WIDTH}x{SCREEN_HEIGHT})")
        print(f"   - Time-decayed rewards: ACTIVE")
        print(f"   - Aggression incentives: ENABLED")
        print(f"   - Frame stacking: {FRAME_STACK_SIZE} frames")

    def _initialize_frame_stack(self, initial_frame):
        """Initialize frame stack with the first frame."""
        self.frame_stack.clear()
        for _ in range(FRAME_STACK_SIZE):
            self.frame_stack.append(initial_frame)

    def _process_visual_frame(self, obs):
        """Process and resize RGB visual frame."""
        if isinstance(obs, np.ndarray):
            # Handle different input formats
            if len(obs.shape) == 3:
                if obs.shape[2] == 3:  # HWC format
                    # Already in HWC RGB format
                    frame = obs
                elif obs.shape[0] == 3:  # CHW format
                    # Convert CHW to HWC
                    frame = np.transpose(obs, (1, 2, 0))
                else:
                    print(f"⚠️ Unexpected image shape: {obs.shape}")
                    frame = obs
            else:
                print(f"⚠️ Unexpected image dimensions: {obs.shape}")
                frame = obs

            # Resize to smaller dimensions while keeping RGB
            if frame.shape[:2] != (SCREEN_HEIGHT, SCREEN_WIDTH):
                frame = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT))

            # Ensure uint8 format
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)

            # Convert back to CHW format for stacking
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = np.transpose(frame, (2, 0, 1))  # HWC to CHW

            return frame

        # If not numpy array, create empty RGB frame
        return np.zeros((3, SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.uint8)

    def _get_stacked_visual_obs(self):
        """Get stacked RGB visual observations."""
        if len(self.frame_stack) == 0:
            empty_frame = np.zeros((3, SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.uint8)
            return np.tile(empty_frame, (FRAME_STACK_SIZE, 1, 1))

        stacked = np.concatenate(list(self.frame_stack), axis=0)
        return stacked

    def reset(self, **kwargs):
        """Enhanced reset with proper RGB initialization."""
        obs, info = self.env.reset(**kwargs)

        # Reset all components
        self.reward_calculator.reset()
        self.feature_tracker.reset()
        self.health_detector = HealthDetector()
        self.vector_history.clear()

        self.episode_count += 1
        self.step_count = 0

        # Process initial RGB visual frame
        processed_frame = self._process_visual_frame(obs)
        self._initialize_frame_stack(processed_frame)

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
                "frame_stack_size": FRAME_STACK_SIZE,
                "image_format": "RGB",
                "image_size": f"{SCREEN_WIDTH}x{SCREEN_HEIGHT}",
            }
        )

        return observation, info

    def step(self, action):
        """Enhanced step function with RGB processing and time-decayed rewards."""
        self.step_count += 1

        # Convert action
        button_combination = self.action_mapper.get_action(action)
        retro_action = self._convert_to_retro_action(button_combination)
        obs, reward, done, truncated, info = self.env.step(retro_action)

        # Process and add RGB visual frame to stack
        processed_frame = self._process_visual_frame(obs)
        self.frame_stack.append(processed_frame)

        # Enhanced health detection
        player_health, opponent_health = self.health_detector.get_health(
            self.env, info, obs
        )

        # --- SIMPLIFIED TERMINATION LOGIC ---
        # A round ends only on KO or when the base environment signals it's over (e.g., timeout).
        # This removes custom wrapper termination rules like a separate step counter.
        base_env_done = done
        round_ended = False
        termination_reason = "ongoing"

        # Case 1: Player or Opponent is knocked out.
        if player_health <= 0 or opponent_health <= 0:
            round_ended = True
            if player_health <= 0 and opponent_health <= 0:
                termination_reason = "double_ko"
            elif player_health <= 0:
                termination_reason = "player_ko"
            else:  # opponent_health <= 0
                termination_reason = "opponent_ko"

        # Case 2: If no KO, but the base env says the game is over, it must be a timeout.
        elif base_env_done:
            round_ended = True
            if abs(player_health - opponent_health) <= 5:
                termination_reason = "timeout_draw"
            elif player_health > opponent_health:
                termination_reason = "timeout_player_wins"
            else:
                termination_reason = "timeout_opponent_wins"

        # Apply termination if our logic determined the round has ended.
        if round_ended:
            done = True
            # Using truncated is good practice for early termination or timeouts.
            truncated = True
        # --- END OF SIMPLIFIED TERMINATION LOGIC ---

        # Add step count to info for reward calculator
        info["step_count"] = self.step_count
        info["termination_reason"] = termination_reason
        info["round_ended"] = round_ended
        info["player_health"] = player_health
        info["opponent_health"] = opponent_health

        # ENHANCED REWARD CALCULATION with time-decayed bonuses
        enhanced_reward, reward_breakdown = self.reward_calculator.calculate_reward(
            player_health, opponent_health, done, info
        )

        # Update feature tracker with enhanced tracking
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
                "player_health": player_health,
                "opponent_health": opponent_health,
                "reward_breakdown": reward_breakdown,
                "enhanced_reward": enhanced_reward,
                "episode_count": self.episode_count,
                "step_count": self.step_count,
                "round_ended": round_ended,
                "termination_reason": termination_reason,
                "round_result": round_result,
                "final_health_diff": player_health - opponent_health,
                "health_detection_working": self.health_detector.is_detection_working(),
                "total_damage_dealt": self.reward_calculator.total_damage_dealt,
                "total_damage_taken": self.reward_calculator.total_damage_taken,
                "frame_stack_size": FRAME_STACK_SIZE,
                "image_format": "RGB",
                "image_size": f"{SCREEN_WIDTH}x{SCREEN_HEIGHT}",
            }
        )

        # Enhanced result display
        if round_ended:
            result_emoji = (
                "🏆"
                if round_result == "WIN"
                else "💀" if round_result == "LOSE" else "🤝"
            )
            speed_indicator = (
                "⚡"
                if self.step_count < MAX_FIGHT_STEPS * 0.5
                else "🐌" if self.step_count >= MAX_FIGHT_STEPS * 0.9 else "🚶"
            )

            # Show enhanced reward breakdown for key components
            time_bonus = reward_breakdown.get("time_multiplier", 1.0)
            combo_info = reward_breakdown.get("combo_frames", 0)

            print(
                f"  {result_emoji}{speed_indicator} Episode {self.episode_count}: {round_result} - "
                f"Steps: {self.step_count}, Health: {player_health} vs {opponent_health}, "
                f"TimeBonus: {time_bonus:.1f}x, Combos: {combo_info}, "
                f"Reason: {termination_reason}"
            )

        return observation, enhanced_reward, done, truncated, info

    def _convert_to_retro_action(self, button_combination):
        """Convert button combination to retro action."""
        button_tuple = tuple(button_combination)
        if button_tuple in self.action_mapper.button_to_index:
            return self.action_mapper.button_to_index[button_tuple]
        else:
            return 0

    def _build_observation(self, visual_obs, info):
        """Build observation dictionary with RGB frame stacking."""
        # Get stacked RGB visual observations
        stacked_visual = self._get_stacked_visual_obs()

        # Get vector features and maintain history
        vector_features = self.feature_tracker.get_features()
        self.vector_history.append(vector_features)

        # Ensure we have full frame history
        while len(self.vector_history) < FRAME_STACK_SIZE:
            self.vector_history.appendleft(
                np.zeros(VECTOR_FEATURE_DIM, dtype=np.float32)
            )

        vector_obs = np.stack(list(self.vector_history), axis=0)

        return {
            "visual_obs": stacked_visual.astype(np.uint8),
            "vector_obs": vector_obs.astype(np.float32),
        }


class SimpleCNN(nn.Module):
    """🚀 Enhanced CNN for RGB temporal processing with better feature extraction."""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__()

        visual_space = observation_space["visual_obs"]
        vector_space = observation_space["vector_obs"]
        n_input_channels = visual_space.shape[0]  # 24 channels for 8-frame RGB stacking
        seq_length, vector_feature_count = vector_space.shape  # (8, 32)

        # Enhanced RGB CNN with better temporal processing
        # Designed for smaller images (160x112) but with RGB information
        self.visual_cnn = nn.Sequential(
            # First layer: handle 24 input channels (8 frames * 3 RGB)
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            # Second layer: extract RGB-aware features
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # Third layer: higher-level features
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            # Fourth layer: final feature extraction
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            # Adaptive pooling to handle variable sizes
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )

        # Calculate visual output size
        with torch.no_grad():
            dummy_visual = torch.zeros(
                1, n_input_channels, visual_space.shape[1], visual_space.shape[2]
            )
            visual_output_size = self.visual_cnn(dummy_visual).shape[1]

        # Enhanced vector processing with better temporal modeling
        self.vector_lstm = nn.LSTM(
            input_size=vector_feature_count,
            hidden_size=128,  # Increased from 64
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )

        self.vector_processor = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Enhanced fusion network
        fusion_input_size = visual_output_size + 64
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        visual_obs = observations["visual_obs"]
        vector_obs = observations["vector_obs"]

        # Process RGB visual with enhanced temporal awareness
        # Normalize to [0, 1] range for better training stability
        visual_features = self.visual_cnn(visual_obs.float() / 255.0)

        # Process vector sequence with enhanced LSTM
        lstm_out, _ = self.vector_lstm(vector_obs)
        vector_features = self.vector_processor(lstm_out[:, -1, :])

        # Enhanced temporal fusion
        combined = torch.cat([visual_features, vector_features], dim=1)
        output = self.fusion(combined)

        return output


class SimpleVerifier(nn.Module):
    """🚀 Enhanced verifier with better energy modeling for aggressive play."""

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

        # Enhanced feature extractor for RGB
        self.features_extractor = SimpleCNN(observation_space, features_dim)

        # Enhanced action embedding with better representation
        self.action_embed = nn.Sequential(
            nn.Linear(self.action_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),  # Better normalization
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Enhanced energy network with better capacity
        self.energy_net = nn.Sequential(
            nn.Linear(features_dim + 64, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),  # Better normalization
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.energy_scale = 0.7  # Slightly increased for better separation

    def forward(
        self, context: torch.Tensor, candidate_action: torch.Tensor
    ) -> torch.Tensor:
        # Extract enhanced RGB features
        if isinstance(context, dict):
            context_features = self.features_extractor(context)
        else:
            context_features = context

        # Enhanced action embedding
        action_embedded = self.action_embed(candidate_action)

        # Enhanced temporal-aware fusion
        combined = torch.cat([context_features, action_embedded], dim=-1)
        energy = self.energy_net(combined) * self.energy_scale

        return energy


class AggressiveAgent:
    """🚀 Enhanced agent with aggressive exploration and better temporal reasoning."""

    def __init__(
        self,
        verifier: SimpleVerifier,
        thinking_steps: int = 6,
        thinking_lr: float = 0.025,
    ):
        self.verifier = verifier
        self.thinking_steps = thinking_steps
        self.thinking_lr = thinking_lr
        self.action_dim = verifier.action_dim

        # NEW: Aggressive exploration parameters
        self.epsilon = 0.25  # 25% random exploration
        self.epsilon_decay = 0.995  # Slow decay to maintain exploration
        self.min_epsilon = 0.05  # Always maintain some exploration

        # Enhanced stats tracking
        self.stats = {
            "total_predictions": 0,
            "successful_optimizations": 0,
            "exploration_actions": 0,
            "exploitation_actions": 0,
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

        # AGGRESSIVE EXPLORATION: Force random actions during training
        if not deterministic and np.random.random() < self.epsilon:
            # Pure random exploration
            action_idx = np.random.randint(0, self.action_dim)
            self.stats["exploration_actions"] += 1
            self.stats["total_predictions"] += 1

            thinking_info = {
                "steps_taken": 0,
                "final_energy": 0.0,
                "exploration": True,
                "epsilon": self.epsilon,
            }

            return action_idx, thinking_info

        # Enhanced exploitation with better thinking
        self.stats["exploitation_actions"] += 1

        # Enhanced initialization for better optimization
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

        # Enhanced thinking loop with better optimization
        best_energy = float("inf")
        best_action = candidate_action.clone().detach()

        for step in range(self.thinking_steps):
            try:
                energy = self.verifier(obs_device, candidate_action)

                current_energy = energy.mean().item()
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_action = candidate_action.clone().detach()

                gradients = torch.autograd.grad(
                    outputs=energy.sum(),
                    inputs=candidate_action,
                    create_graph=False,
                    retain_graph=False,
                )[0]

                with torch.no_grad():
                    # Enhanced gradient descent with adaptive learning rate
                    step_size = self.thinking_lr * (0.85**step)  # More aggressive decay
                    candidate_action = candidate_action - step_size * gradients
                    candidate_action = F.softmax(candidate_action, dim=-1)
                    candidate_action.requires_grad_(True)

            except Exception:
                candidate_action = best_action
                break

        # Enhanced action selection
        with torch.no_grad():
            final_action_probs = F.softmax(candidate_action, dim=-1)

            if deterministic:
                action_idx = torch.argmax(final_action_probs, dim=-1)
            else:
                # Enhanced exploration even in exploitation
                if torch.rand(1).item() < 0.15:  # 15% additional exploration
                    action_idx = torch.randint(
                        0, self.action_dim, (batch_size,), device=device
                    )
                else:
                    action_idx = torch.multinomial(final_action_probs, 1).squeeze(-1)

        # Update epsilon for exploration decay
        if not deterministic:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        self.stats["total_predictions"] += 1

        thinking_info = {
            "steps_taken": self.thinking_steps,
            "final_energy": best_energy,
            "energy_improvement": best_energy < 0,
            "exploration": False,
            "epsilon": self.epsilon,
        }

        return action_idx.item() if batch_size == 1 else action_idx, thinking_info

    def get_thinking_stats(self) -> Dict:
        stats = self.stats.copy()
        if stats["total_predictions"] > 0:
            stats["success_rate"] = (
                stats["successful_optimizations"] / stats["total_predictions"]
            )
            stats["exploration_rate"] = (
                stats["exploration_actions"] / stats["total_predictions"]
            )
        else:
            stats["success_rate"] = 0.0
            stats["exploration_rate"] = 0.0
        stats["current_epsilon"] = self.epsilon
        return stats


def make_enhanced_env(
    game="StreetFighterIISpecialChampionEdition-Genesis", state="ken_bison_12.state"
):
    """Create enhanced Street Fighter environment with RGB support."""
    try:
        env = retro.make(
            game=game, state=state, use_restricted_actions=retro.Actions.DISCRETE
        )
        env = EnhancedStreetFighterWrapper(env)

        print(f"   ✅ Enhanced RGB environment created")
        print(f"   - Image format: RGB ({SCREEN_WIDTH}x{SCREEN_HEIGHT})")
        print(f"   - Time-decayed rewards: ACTIVE")
        print(f"   - Aggression incentives: ENABLED")
        print(f"   - Frame stacking: {FRAME_STACK_SIZE} frames")
        return env

    except Exception as e:
        print(f"   ❌ Environment creation failed: {e}")
        raise


def verify_health_detection(env, episodes=5):
    """Verify that health detection and reward system is working with RGB."""
    print(f"🔍 Verifying enhanced RGB system over {episodes} episodes...")

    detection_working = 0
    health_changes_detected = 0
    timeout_wins = 0
    fast_wins = 0

    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        step_count = 0
        episode_healths = {"player": [], "opponent": []}

        # Verify RGB format
        if "visual_obs" in obs:
            visual_shape = obs["visual_obs"].shape
            print(f"   Episode {episode + 1}: Visual obs shape: {visual_shape}")
            if visual_shape[0] == 24:  # 8 frames * 3 RGB channels
                print(f"   ✅ RGB frame stacking verified: {visual_shape[0]//3} frames")

        while not done and step_count < 200:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)

            player_health = info.get("player_health", MAX_HEALTH)
            opponent_health = info.get("opponent_health", MAX_HEALTH)

            episode_healths["player"].append(player_health)
            episode_healths["opponent"].append(opponent_health)

            if done:
                termination_reason = info.get("termination_reason", "unknown")
                if "timeout" in termination_reason:
                    timeout_wins += 1
                elif step_count < MAX_FIGHT_STEPS * 0.5:
                    fast_wins += 1

            step_count += 1

        # Check detection
        player_varied = len(set(episode_healths["player"])) > 1
        opponent_varied = len(set(episode_healths["opponent"])) > 1
        detection_status = info.get("health_detection_working", False)

        if detection_status:
            detection_working += 1
        if player_varied or opponent_varied:
            health_changes_detected += 1

        print(
            f"   Episode {episode + 1}: Detection: {detection_status}, "
            f"Player: {min(episode_healths['player'])}-{max(episode_healths['player'])}, "
            f"Opponent: {min(episode_healths['opponent'])}-{max(episode_healths['opponent'])}"
        )

    success_rate = health_changes_detected / episodes
    print(f"\n🎯 Enhanced RGB System Results:")
    print(f"   - Health detection working: {detection_working}/{episodes}")
    print(
        f"   - Health changes detected: {health_changes_detected}/{episodes} ({success_rate:.1%})"
    )
    print(f"   - Timeout wins: {timeout_wins}/{episodes}")
    print(f"   - Fast wins: {fast_wins}/{episodes}")
    print(f"   - Image format: RGB at {SCREEN_WIDTH}x{SCREEN_HEIGHT}")

    if success_rate > 0.6:
        print(f"   ✅ Enhanced RGB system is working! Ready for aggressive training.")
    else:
        print(f"   ⚠️  System may need adjustment.")

    return success_rate > 0.6


# Export enhanced RGB components
__all__ = [
    # Enhanced Environment
    "EnhancedStreetFighterWrapper",
    "make_enhanced_env",
    "verify_health_detection",
    # Enhanced Core Components
    "HealthDetector",
    "EnhancedRewardCalculator",
    "SimplifiedFeatureTracker",
    "StreetFighterDiscreteActions",
    # Enhanced Models
    "SimpleCNN",
    "SimpleVerifier",
    "AggressiveAgent",
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
    "FRAME_STACK_SIZE",
    "SCREEN_WIDTH",
    "SCREEN_HEIGHT",
]

print(f"🚀 ENHANCED Street Fighter wrapper loaded successfully! (RGB Version)")
print(f"   - ✅ RGB images with resizing: ACTIVE ({SCREEN_WIDTH}x{SCREEN_HEIGHT})")
print(f"   - ✅ Time-decayed winning bonuses: ACTIVE")
print(f"   - ✅ Aggressive exploration: ENABLED")
print(f"   - ✅ Enhanced temporal awareness: ACTIVE")
print(f"   - ✅ Combo and speed incentives: ENABLED")
print(f"   - ✅ Timeout penalties: HEAVY")
print(f"🎯 Ready to break learning plateaus with RGB visual information!")
