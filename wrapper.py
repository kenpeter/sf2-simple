#!/usr/bin/env python3
"""
🛡️ COMPLETE WRAPPER WITH QUALITY-BASED EXPERIENCE BUFFER - SINGLE ROUND FIGHT
Integrates intelligent reward shaping with quality-based labeling for 60% win rate.
Modified for single round fights.
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
    f'logs/single_round_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
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

print(f"🥊 SINGLE ROUND FIGHT Configuration:")
print(f"   - Features: {VECTOR_FEATURE_DIM}")
print(f"   - Training paradigm: Quality-Based Experience Labeling")
print(f"   - Fight mode: Single Round (first to zero health wins)")


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


class IntelligentRewardCalculator:
    """
    🧠 Calculates intelligent rewards with detailed breakdown for quality scoring.
    Modified for single round fights.
    """

    def __init__(self):
        # Reward component weights (optimized for single round winning)
        self.weights = {
            "damage_dealt": 20.0,  # Increased for single round
            "damage_avoided": 10.0,  # Increased for single round
            "health_advantage": 8.0,  # Increased for single round
            "combo_progression": 15.0,  # Increased for single round
            "positioning": 6.0,
            "pressure": 5.0,
            "momentum": 10.0,  # Increased for single round
            "defensive": 4.0,
            "win_progress": 15.0,  # Increased for single round
            "frame_efficiency": 2.0,
        }

        # Game state tracking
        self.prev_player_health = None
        self.prev_opponent_health = None
        self.prev_score = None
        self.combo_count = 0
        self.last_hit_frame = -1
        self.current_frame = 0
        self.pressure_frames = 0
        self.defensive_frames = 0

        # History tracking
        self.health_history = deque(maxlen=10)
        self.opponent_health_history = deque(maxlen=10)
        self.momentum_tracker = deque(maxlen=10)

        # Game constants
        self.max_health = MAX_HEALTH
        self.screen_width = SCREEN_WIDTH
        self.optimal_distance_min = 60
        self.optimal_distance_max = 100

        print("🧠 Intelligent Reward Calculator initialized for single round fights")
        print(f"   - {len(self.weights)} reward components")
        print(f"   - Optimized for decisive single round gameplay")

    def calculate_reward_with_breakdown(
        self, info: Dict, prev_info: Dict = None
    ) -> Tuple[float, Dict]:
        """
        Calculate intelligent reward with detailed breakdown for single round.

        Returns:
            (total_reward, reward_breakdown_dict)
        """
        self.current_frame += 1
        reward_breakdown = {}
        total_reward = 0.0

        # Extract current state
        player_health = ensure_scalar(
            info.get("agent_hp", self.max_health), self.max_health
        )
        opponent_health = ensure_scalar(
            info.get("enemy_hp", self.max_health), self.max_health
        )
        score = ensure_scalar(info.get("score", 0), 0)
        player_x = ensure_scalar(
            info.get("agent_x", self.screen_width // 2), self.screen_width // 2
        )
        opponent_x = ensure_scalar(
            info.get("enemy_x", self.screen_width // 2), self.screen_width // 2
        )

        # Store in history
        self.health_history.append(player_health)
        self.opponent_health_history.append(opponent_health)

        # 1. DAMAGE AND HEALTH REWARDS (Enhanced for single round)
        damage_reward = self._calculate_damage_rewards(
            player_health, opponent_health, prev_info
        )
        total_reward += damage_reward
        reward_breakdown["damage"] = damage_reward

        # 2. COMBO AND SCORE REWARDS (Enhanced for single round)
        combo_reward = self._calculate_combo_rewards(score, prev_info)
        total_reward += combo_reward
        reward_breakdown["combo"] = combo_reward

        # 3. POSITIONING REWARDS
        spacing_reward = self._calculate_spacing_rewards(player_x, opponent_x)
        total_reward += spacing_reward
        reward_breakdown["spacing"] = spacing_reward

        # 4. PRESSURE AND TACTICAL REWARDS (Enhanced for single round)
        pressure_reward = self._calculate_pressure_rewards(
            player_x, opponent_x, player_health, opponent_health
        )
        total_reward += pressure_reward
        reward_breakdown["pressure"] = pressure_reward

        # 5. MOMENTUM REWARDS (Enhanced for single round)
        momentum_reward = self._calculate_momentum_rewards()
        total_reward += momentum_reward
        reward_breakdown["momentum"] = momentum_reward

        # 6. WIN PROGRESS REWARDS (Enhanced for single round)
        win_progress_reward = self._calculate_win_progress_rewards(
            player_health, opponent_health
        )
        total_reward += win_progress_reward
        reward_breakdown["win_progress"] = win_progress_reward

        # 7. FRAME EFFICIENCY (small time penalty to encourage decisive action)
        efficiency_reward = (
            -0.002 * self.weights["frame_efficiency"]
        )  # Slightly increased penalty
        total_reward += efficiency_reward
        reward_breakdown["efficiency"] = efficiency_reward

        # Apply context scaling (enhanced for single round urgency)
        total_reward = self._apply_context_scaling(
            total_reward, player_health, opponent_health
        )

        # Update previous state
        self.prev_player_health = player_health
        self.prev_opponent_health = opponent_health
        self.prev_score = score

        # Normalize total reward (slightly wider range for single round)
        total_reward = np.clip(total_reward, -1.0, 3.0)

        return total_reward, reward_breakdown

    def _calculate_damage_rewards(self, player_health, opponent_health, prev_info):
        """Calculate damage-related rewards with enhanced single round focus."""
        reward = 0.0

        if prev_info is not None:
            prev_player_health = ensure_scalar(
                prev_info.get("agent_hp", player_health), player_health
            )
            prev_opponent_health = ensure_scalar(
                prev_info.get("enemy_hp", opponent_health), opponent_health
            )

            # Damage dealt (positive reward) - enhanced for single round
            damage_dealt = max(0, prev_opponent_health - opponent_health)
            if damage_dealt > 0:
                # Scale reward based on opponent's remaining health (more critical in single round)
                health_factor = (
                    1.0 + (1.0 - opponent_health / self.max_health) * 1.0
                )  # Doubled factor
                reward += (
                    damage_dealt
                    * self.weights["damage_dealt"]
                    * health_factor
                    * 0.015  # Increased multiplier
                )

            # Damage received (negative reward) - enhanced penalty for single round
            damage_received = max(0, prev_player_health - player_health)
            if damage_received > 0:
                reward -= (
                    damage_received * self.weights["damage_dealt"] * 0.01
                )  # Doubled penalty

            # Health advantage change - more critical in single round
            current_advantage = player_health - opponent_health
            prev_advantage = prev_player_health - prev_opponent_health
            advantage_change = current_advantage - prev_advantage
            if advantage_change > 0:
                reward += (
                    advantage_change * self.weights["health_advantage"] * 0.005
                )  # Increased

        return reward

    def _calculate_combo_rewards(self, score, prev_info):
        """Calculate combo-related rewards with enhanced single round focus."""
        reward = 0.0

        if prev_info is not None:
            prev_score = ensure_scalar(prev_info.get("score", 0), 0)
            score_increase = score - prev_score

            if score_increase > 0:
                # Detect combo continuation
                if self.current_frame - self.last_hit_frame <= 60:  # 1 second window
                    self.combo_count += 1
                    # Exponential bonus for longer combos (enhanced for single round)
                    combo_multiplier = 1.0 + (
                        self.combo_count * 0.3
                    )  # Increased multiplier
                    reward += (
                        score_increase
                        * self.weights["combo_progression"]
                        * combo_multiplier
                        * 0.002  # Doubled base reward
                    )
                else:
                    self.combo_count = 1
                    reward += (
                        score_increase * self.weights["combo_progression"] * 0.002
                    )  # Doubled

                self.last_hit_frame = self.current_frame
            else:
                # Reset combo if no hit for 1 second
                if self.current_frame - self.last_hit_frame > 60:
                    self.combo_count = 0

        return reward

    def _calculate_spacing_rewards(self, player_x, opponent_x):
        """Calculate positioning rewards."""
        reward = 0.0
        distance = abs(player_x - opponent_x)

        # Optimal spacing reward
        if self.optimal_distance_min <= distance <= self.optimal_distance_max:
            reward += self.weights["positioning"] * 0.002

        # Penalty for poor spacing
        if distance < 30:  # Too close (vulnerable)
            reward -= self.weights["positioning"] * 0.001
        elif distance > 150:  # Too far (can't attack)
            reward -= self.weights["positioning"] * 0.0005

        # Center control bonus
        screen_center = self.screen_width // 2
        player_center_dist = abs(player_x - screen_center)
        opponent_center_dist = abs(opponent_x - screen_center)
        if player_center_dist < opponent_center_dist:
            reward += self.weights["positioning"] * 0.0005

        # Corner awareness
        player_corner_dist = min(player_x, self.screen_width - player_x)
        if player_corner_dist < 50:  # Too close to corner
            reward -= self.weights["positioning"] * 0.001

        return reward

    def _calculate_pressure_rewards(
        self, player_x, opponent_x, player_health, opponent_health
    ):
        """Calculate tactical pressure rewards with enhanced single round focus."""
        reward = 0.0
        distance = abs(player_x - opponent_x)

        # Offensive pressure when ahead (enhanced for single round)
        if player_health > opponent_health and distance < 80:
            self.pressure_frames += 1
            reward += self.weights["pressure"] * 0.002  # Doubled
        else:
            self.pressure_frames = max(0, self.pressure_frames - 1)

        # Sustained pressure bonus (enhanced)
        if self.pressure_frames > 30:  # Half second of pressure
            reward += self.weights["pressure"] * 0.004  # Doubled

        # Defensive spacing when behind (enhanced)
        if player_health < opponent_health and distance > 100:
            self.defensive_frames += 1
            reward += self.weights["defensive"] * 0.001  # Doubled
        else:
            self.defensive_frames = max(0, self.defensive_frames - 1)

        return reward

    def _calculate_momentum_rewards(self):
        """Calculate momentum rewards with enhanced single round focus."""
        reward = 0.0

        if len(self.health_history) >= 5 and len(self.opponent_health_history) >= 5:
            # Calculate health momentum over last 5 frames
            health_list = list(self.health_history)
            opponent_health_list = list(self.opponent_health_history)

            player_momentum = (health_list[-1] - health_list[-5]) / 5
            opponent_momentum = (
                opponent_health_list[-1] - opponent_health_list[-5]
            ) / 5

            # Positive momentum = opponent losing health faster than us
            relative_momentum = opponent_momentum - player_momentum
            self.momentum_tracker.append(relative_momentum)

            # Reward positive momentum (enhanced for single round)
            if relative_momentum > 0:
                reward += relative_momentum * self.weights["momentum"] * 0.02  # Doubled

            # Sustained momentum bonus (enhanced)
            if len(self.momentum_tracker) >= 5:
                momentum_list = list(self.momentum_tracker)
                recent_momentum = sum(momentum_list[-5:]) / 5
                if recent_momentum > 0.5:
                    reward += self.weights["momentum"] * 0.004  # Doubled

        return reward

    def _calculate_win_progress_rewards(self, player_health, opponent_health):
        """Calculate win condition progress rewards with enhanced single round focus."""
        reward = 0.0

        player_health_pct = player_health / self.max_health
        opponent_health_pct = opponent_health / self.max_health

        # Reward for maintaining high health (enhanced)
        if player_health_pct > 0.7:
            reward += self.weights["win_progress"] * 0.002  # Doubled

        # Big bonus for getting opponent to critical health (enhanced)
        if opponent_health_pct < 0.3:
            reward += self.weights["win_progress"] * 0.01  # Doubled

        if opponent_health_pct < 0.1:
            reward += self.weights["win_progress"] * 0.02  # Doubled - Close to victory!

        # Health advantage scaling (enhanced)
        health_advantage = player_health_pct - opponent_health_pct
        if health_advantage > 0:
            reward += health_advantage * self.weights["win_progress"] * 0.004  # Doubled

        return reward

    def _apply_context_scaling(self, base_reward, player_health, opponent_health):
        """Apply intelligent scaling based on game context with single round urgency."""
        # Scale up rewards when the game is close (more critical decisions in single round)
        health_diff = abs(player_health - opponent_health)
        if health_diff < 30:  # Very close game
            base_reward *= 1.5  # Increased scaling
        elif health_diff < 60:  # Somewhat close
            base_reward *= 1.3  # Increased scaling

        # Scale up rewards in critical health situations (enhanced for single round)
        if player_health < 40 or opponent_health < 40:
            base_reward *= 1.4  # Increased scaling

        return base_reward

    def calculate_win_reward(
        self, won: bool, player_health: int, opponent_health: int, episode_length: int
    ) -> float:
        """Calculate final win/loss reward for single round."""
        if won:
            base_win_reward = 2.0  # Increased base reward for single round victory
            health_bonus = (
                player_health / self.max_health
            ) * 0.5  # Increased health bonus

            # Speed bonus (more important in single round)
            if episode_length < 800:  # Quick victory
                speed_bonus = 0.5
            elif episode_length < 1500:  # Moderate speed
                speed_bonus = 0.2
            else:
                speed_bonus = 0.0

            return base_win_reward + health_bonus + speed_bonus
        else:
            return -0.2  # Slightly increased loss penalty

    def get_experience_quality_score(
        self, reward: float, reward_breakdown: Dict
    ) -> float:
        """
        Calculate quality score for experience buffer labeling (enhanced for single round).
        """
        # Initialize strategic bonus
        strategic_bonus = 0.0

        # Base quality from total reward using tanh normalization
        base_quality = math.tanh(reward * 1.5) * 0.5 + 0.5  # Adjusted for single round

        # Strategic bonuses for key actions (enhanced for single round)
        if reward_breakdown.get("damage", 0) > 0.02:  # Significant damage dealing
            strategic_bonus += 0.25

        if reward_breakdown.get("combo", 0) > 0.01:  # Building combos
            strategic_bonus += 0.2

        if reward_breakdown.get("spacing", 0) > 0.001:  # Good positioning
            strategic_bonus += 0.1

        if reward_breakdown.get("momentum", 0) > 0.002:  # Positive momentum
            strategic_bonus += 0.15

        if reward_breakdown.get("win_progress", 0) > 0.01:  # Progress toward winning
            strategic_bonus += 0.2

        # Penalties for bad actions (enhanced for single round)
        if reward_breakdown.get("damage", 0) < -0.02:  # Taking significant damage
            strategic_bonus -= 0.25

        if reward_breakdown.get("spacing", 0) < -0.001:  # Poor positioning
            strategic_bonus -= 0.1

        # Final quality score
        quality_score = min(1.0, max(0.0, base_quality + strategic_bonus))

        return quality_score

    def reset_episode(self):
        """Reset tracking for new episode."""
        self.health_history.clear()
        self.opponent_health_history.clear()
        self.momentum_tracker.clear()

        self.combo_count = 0
        self.last_hit_frame = -1
        self.current_frame = 0
        self.pressure_frames = 0
        self.defensive_frames = 0

        self.prev_player_health = None
        self.prev_opponent_health = None
        self.prev_score = None


class QualityBasedExperienceBuffer:
    """
    🎯 Experience buffer using quality-based labeling instead of broken percentiles.
    """

    def __init__(self, capacity=40000, quality_threshold=0.55):
        self.capacity = capacity
        self.quality_threshold = quality_threshold

        # Separate storage for good and bad experiences
        self.good_experiences = deque(maxlen=capacity // 2)
        self.bad_experiences = deque(maxlen=capacity // 2)

        # Quality tracking for monitoring
        self.quality_scores = deque(maxlen=1000)
        self.total_added = 0
        self.good_count = 0
        self.bad_count = 0

        print(f"🎯 Quality-Based Experience Buffer initialized")
        print(f"   - Quality threshold: {quality_threshold}")
        print(f"   - Uses absolute quality, not relative percentiles")

    def add_experience(
        self,
        experience: Dict,
        reward: float,
        reward_breakdown: Dict,
        quality_score: float,
    ):
        """
        Add experience with quality-based labeling.

        Args:
            experience: The experience dictionary (obs, action, etc.)
            reward: The intelligent reward for this action
            reward_breakdown: Dictionary with reward components
            quality_score: Pre-calculated quality score
        """
        self.total_added += 1

        # Store quality score for monitoring
        self.quality_scores.append(quality_score)

        # Add metadata to experience
        experience["quality_score"] = quality_score
        experience["reward"] = reward
        experience["reward_breakdown"] = reward_breakdown

        # Label based on absolute quality threshold
        if quality_score >= self.quality_threshold:
            experience["is_good"] = True
            self.good_experiences.append(experience)
            self.good_count += 1
        else:
            experience["is_good"] = False
            self.bad_experiences.append(experience)
            self.bad_count += 1

    def sample_balanced_batch(self, batch_size: int):
        """Sample balanced batch of good and bad experiences."""
        target_per_class = batch_size // 2

        # Check if we have enough experiences
        if len(self.good_experiences) < target_per_class:
            return None, None

        if len(self.bad_experiences) < target_per_class:
            return None, None

        # Sample randomly from each class
        good_indices = np.random.choice(
            len(self.good_experiences), target_per_class, replace=False
        )
        bad_indices = np.random.choice(
            len(self.bad_experiences), target_per_class, replace=False
        )

        good_batch = [self.good_experiences[i] for i in good_indices]
        bad_batch = [self.bad_experiences[i] for i in bad_indices]

        return good_batch, bad_batch

    def get_stats(self) -> Dict:
        """Get comprehensive buffer statistics."""
        total_size = len(self.good_experiences) + len(self.bad_experiences)

        # Quality statistics
        if self.quality_scores:
            avg_quality = np.mean(list(self.quality_scores))
            quality_std = np.std(list(self.quality_scores))
        else:
            avg_quality = 0.0
            quality_std = 0.0

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
        }

    def adjust_threshold(self, target_good_ratio=0.5):
        """Dynamically adjust quality threshold to maintain good/bad balance."""
        if len(self.quality_scores) < 100:  # Need enough data
            return

        total_size = len(self.good_experiences) + len(self.bad_experiences)
        current_good_ratio = len(self.good_experiences) / max(1, total_size)

        if current_good_ratio < target_good_ratio - 0.1:  # Too few good examples
            self.quality_threshold *= 0.95  # Lower threshold
            print(f"📉 Lowered quality threshold to {self.quality_threshold:.3f}")
        elif current_good_ratio > target_good_ratio + 0.1:  # Too many good examples
            self.quality_threshold *= 1.05  # Raise threshold
            print(f"📈 Raised quality threshold to {self.quality_threshold:.3f}")

        # Keep threshold in reasonable range
        self.quality_threshold = max(0.3, min(0.8, self.quality_threshold))

    def emergency_purge(self, keep_ratio=0.3):
        """Emergency purge - keep only the best experiences."""
        print(f"🚨 Emergency buffer purge - keeping top {keep_ratio*100:.0f}%")

        total_purged = 0

        # Purge good experiences
        if len(self.good_experiences) > 0:
            experiences = list(self.good_experiences)
            keep_count = max(1, int(len(experiences) * keep_ratio))

            # Sort by quality and keep best
            experiences.sort(key=lambda x: x.get("quality_score", 0.5), reverse=True)
            self.good_experiences.clear()

            for exp in experiences[:keep_count]:
                self.good_experiences.append(exp)

            total_purged += len(experiences) - keep_count

        # Purge bad experiences (keep some for contrast)
        if len(self.bad_experiences) > 0:
            experiences = list(self.bad_experiences)
            keep_count = max(1, int(len(experiences) * keep_ratio))

            # For bad experiences, keep the "least bad" ones
            experiences.sort(key=lambda x: x.get("quality_score", 0.5), reverse=True)
            self.bad_experiences.clear()

            for exp in experiences[:keep_count]:
                self.bad_experiences.append(exp)

            total_purged += len(experiences) - keep_count

        print(f"   📊 Purged {total_purged} experiences")


class SimplifiedFeatureTracker:
    """Simplified Strategic Feature Tracker."""

    def __init__(self, history_length=8):
        self.history_length = history_length

        # Core feature tracking
        self.player_health_history = deque(maxlen=history_length)
        self.opponent_health_history = deque(maxlen=history_length)
        self.score_history = deque(maxlen=history_length)
        self.score_change_history = deque(maxlen=history_length)
        self.combo_counter = 0
        self.max_combo_this_round = 0
        self.last_score_increase_frame = -1
        self.current_frame = 0
        self.player_damage_dealt_history = deque(maxlen=history_length)
        self.opponent_damage_dealt_history = deque(maxlen=history_length)
        self.button_features_history = deque(maxlen=history_length)
        self.previous_button_features = np.zeros(12, dtype=np.float32)

        # Combat metrics
        self.close_combat_count = 0
        self.total_frames = 0

        # Normalization
        self.feature_rolling_mean = None
        self.feature_rolling_std = None
        self.normalization_alpha = 0.999
        self.feature_nan_count = 0

        # Game constants
        self.DANGER_ZONE_HEALTH = MAX_HEALTH * 0.25
        self.CORNER_THRESHOLD = 53
        self.CLOSE_DISTANCE = 71
        self.OPTIMAL_SPACING_MIN = 62
        self.OPTIMAL_SPACING_MAX = 98
        self.COMBO_TIMEOUT_FRAMES = 60
        self.MIN_SCORE_INCREASE_FOR_HIT = 50

        # Previous states
        self.prev_player_health = None
        self.prev_opponent_health = None
        self.prev_score = None

        print(f"🔧 SimplifiedFeatureTracker initialized")

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features with consistent dimension handling."""
        features = ensure_feature_dimension(features, VECTOR_FEATURE_DIM)

        nan_mask = ~np.isfinite(features)
        if np.any(nan_mask):
            self.feature_nan_count += np.sum(nan_mask)
            features = sanitize_array(features, 0.0)

        if self.feature_rolling_mean is None:
            self.feature_rolling_mean = np.zeros(VECTOR_FEATURE_DIM, dtype=np.float32)
            self.feature_rolling_std = np.ones(VECTOR_FEATURE_DIM, dtype=np.float32)

        self.feature_rolling_mean = (
            self.normalization_alpha * self.feature_rolling_mean
            + (1 - self.normalization_alpha) * features
        )
        squared_diff = (features - self.feature_rolling_mean) ** 2
        self.feature_rolling_std = np.sqrt(
            self.normalization_alpha * (self.feature_rolling_std**2)
            + (1 - self.normalization_alpha) * squared_diff
        )
        safe_std = np.maximum(self.feature_rolling_std, 1e-6)

        try:
            normalized = (features - self.feature_rolling_mean) / safe_std
            normalized = np.where(np.isfinite(normalized), normalized, 0.0)
        except Exception as e:
            print(f"⚠️  Normalization error: {e}, using zeros")
            normalized = np.zeros(VECTOR_FEATURE_DIM, dtype=np.float32)

        normalized = np.clip(normalized, -3.0, 3.0)
        normalized = sanitize_array(normalized, 0.0)
        return normalized.astype(np.float32)

    def update(self, info: Dict, button_features: np.ndarray) -> np.ndarray:
        """Main update function with simplified features."""
        self.current_frame += 1

        try:
            player_health = ensure_scalar(info.get("agent_hp", MAX_HEALTH), MAX_HEALTH)
            opponent_health = ensure_scalar(
                info.get("enemy_hp", MAX_HEALTH), MAX_HEALTH
            )
            score = ensure_scalar(info.get("score", 0), 0)
            player_x = ensure_scalar(
                info.get("agent_x", SCREEN_WIDTH / 2), SCREEN_WIDTH / 2
            )
            opponent_x = ensure_scalar(
                info.get("enemy_x", SCREEN_WIDTH / 2), SCREEN_WIDTH / 2
            )

            button_features = sanitize_array(button_features, 0.0)
            self.player_health_history.append(player_health)
            self.opponent_health_history.append(opponent_health)

            # Score and combo tracking
            score_change = 0
            if self.prev_score is not None and np.isfinite(self.prev_score):
                score_change = score - self.prev_score
                if not np.isfinite(score_change):
                    score_change = 0

            if safe_comparison(score_change, self.MIN_SCORE_INCREASE_FOR_HIT, ">="):
                if safe_comparison(
                    self.current_frame - self.last_score_increase_frame,
                    self.COMBO_TIMEOUT_FRAMES,
                    "<=",
                ):
                    self.combo_counter += 1
                else:
                    self.combo_counter = 1
                self.last_score_increase_frame = self.current_frame
                self.max_combo_this_round = max(
                    self.max_combo_this_round, self.combo_counter
                )
            elif safe_comparison(
                self.current_frame - self.last_score_increase_frame,
                self.COMBO_TIMEOUT_FRAMES,
                ">",
            ):
                self.combo_counter = 0

            self.score_history.append(score)
            self.score_change_history.append(score_change)
            self.button_features_history.append(self.previous_button_features.copy())
            self.previous_button_features = button_features.copy()

            # Damage calculation
            player_damage = 0
            opponent_damage = 0
            if self.prev_opponent_health is not None and np.isfinite(
                self.prev_opponent_health
            ):
                damage_calc = ensure_scalar(self.prev_opponent_health) - ensure_scalar(
                    opponent_health
                )
                player_damage = max(0, damage_calc) if np.isfinite(damage_calc) else 0

            if self.prev_player_health is not None and np.isfinite(
                self.prev_player_health
            ):
                damage_calc = ensure_scalar(self.prev_player_health) - ensure_scalar(
                    player_health
                )
                opponent_damage = max(0, damage_calc) if np.isfinite(damage_calc) else 0

            self.player_damage_dealt_history.append(player_damage)
            self.opponent_damage_dealt_history.append(opponent_damage)

            # Close combat tracking
            self.total_frames += 1
            distance = abs(player_x - opponent_x)
            if np.isfinite(distance) and safe_comparison(
                distance, self.CLOSE_DISTANCE, "<="
            ):
                self.close_combat_count += 1

            # Calculate simplified features
            features = self._calculate_simplified_features(info, distance)
            features = ensure_feature_dimension(features, VECTOR_FEATURE_DIM)
            features = self._normalize_features(features)

            # Update previous states
            self.prev_player_health = player_health
            self.prev_opponent_health = opponent_health
            self.prev_score = score

            return features

        except Exception as e:
            print(f"⚠️  Critical error in SimplifiedFeatureTracker.update(): {e}")
            return np.zeros(VECTOR_FEATURE_DIM, dtype=np.float32)

    def _calculate_simplified_features(self, info, distance) -> np.ndarray:
        """Calculate simplified features."""
        features = np.zeros(VECTOR_FEATURE_DIM, dtype=np.float32)

        try:
            player_health = ensure_scalar(info.get("agent_hp", MAX_HEALTH), MAX_HEALTH)
            opponent_health = ensure_scalar(
                info.get("enemy_hp", MAX_HEALTH), MAX_HEALTH
            )
            player_x = ensure_scalar(
                info.get("agent_x", SCREEN_WIDTH / 2), SCREEN_WIDTH / 2
            )
            opponent_x = ensure_scalar(
                info.get("enemy_x", SCREEN_WIDTH / 2), SCREEN_WIDTH / 2
            )

            distance = distance if np.isfinite(distance) else 80.0

            # Basic health features (0-2)
            features[0] = (
                1.0
                if safe_comparison(player_health, self.DANGER_ZONE_HEALTH, "<=")
                else 0.0
            )
            features[1] = (
                1.0
                if safe_comparison(opponent_health, self.DANGER_ZONE_HEALTH, "<=")
                else 0.0
            )

            if safe_comparison(opponent_health, 0, ">"):
                health_ratio = safe_divide(player_health, opponent_health, 1.0)
                features[2] = np.clip(health_ratio, 0.0, 3.0)
            else:
                features[2] = 3.0

            # Combat momentum (3-5)
            features[3] = self._calculate_momentum(self.player_health_history)
            features[4] = self._calculate_momentum(self.player_damage_dealt_history)
            features[5] = self._calculate_momentum(self.opponent_damage_dealt_history)

            # Position features (6-10)
            features[6] = np.clip(
                min(player_x, SCREEN_WIDTH - player_x) / (SCREEN_WIDTH / 2), 0.0, 1.0
            )
            features[7] = np.clip(
                min(opponent_x, SCREEN_WIDTH - opponent_x) / (SCREEN_WIDTH / 2),
                0.0,
                1.0,
            )
            features[8] = (
                1.0
                if safe_comparison(
                    min(player_x, SCREEN_WIDTH - player_x), self.CORNER_THRESHOLD, "<="
                )
                else 0.0
            )
            features[9] = (
                1.0
                if safe_comparison(
                    min(opponent_x, SCREEN_WIDTH - opponent_x),
                    self.CORNER_THRESHOLD,
                    "<=",
                )
                else 0.0
            )

            player_center_dist = abs(player_x - SCREEN_WIDTH / 2)
            opponent_center_dist = abs(opponent_x - SCREEN_WIDTH / 2)
            features[10] = np.sign(opponent_center_dist - player_center_dist)

            # Distance and positioning (11-13)
            features[11] = np.clip(distance / 200.0, 0.0, 1.0)
            optimal_spacing = safe_comparison(
                distance, self.OPTIMAL_SPACING_MIN, ">="
            ) and safe_comparison(distance, self.OPTIMAL_SPACING_MAX, "<=")
            features[12] = 1.0 if optimal_spacing else 0.0
            features[13] = safe_divide(
                self.close_combat_count, max(1, self.total_frames), 0.0
            )

            # Score and combo features (14-16)
            features[14] = self._calculate_enhanced_score_momentum()
            features[15] = min(self.combo_counter / 10.0, 1.0)
            features[16] = min(self.max_combo_this_round / 10.0, 1.0)

            # Game state features (17-19)
            agent_status = ensure_scalar(info.get("agent_status", 0), 0)
            enemy_status = ensure_scalar(info.get("enemy_status", 0), 0)
            status_diff = (agent_status - enemy_status) / 100.0
            features[17] = np.clip(status_diff, -1.0, 1.0)

            # Single round specific features (18-19) - no multi-round tracking
            features[18] = 0.0  # No multi-round victories in single round mode
            features[19] = 0.0  # No multi-round victories in single round mode

            # Button features (20-31)
            try:
                if len(self.button_features_history) > 0:
                    button_hist = self.button_features_history[-1]
                    if isinstance(button_hist, np.ndarray) and len(button_hist) == 12:
                        features[20:32] = button_hist
                    else:
                        features[20:32] = np.zeros(12, dtype=np.float32)
                else:
                    features[20:32] = np.zeros(12, dtype=np.float32)
            except Exception as e:
                features[20:32] = np.zeros(12, dtype=np.float32)

            features = sanitize_array(features, 0.0)
            return features

        except Exception as e:
            print(f"⚠️  Error in _calculate_simplified_features: {e}")
            return np.zeros(VECTOR_FEATURE_DIM, dtype=np.float32)

    def _calculate_momentum(self, history):
        if len(history) < 2:
            return 0.0
        values = [v for v in list(history) if np.isfinite(v)]
        if len(values) < 2:
            return 0.0
        changes = [values[i] - values[i - 1] for i in range(1, len(values))]
        finite_changes = [c for c in changes if np.isfinite(c)]
        if not finite_changes:
            return 0.0
        momentum = safe_mean(
            finite_changes[-3:] if len(finite_changes) >= 3 else finite_changes
        )
        return np.clip(momentum, -50.0, 50.0)

    def _calculate_enhanced_score_momentum(self) -> float:
        if len(self.score_change_history) < 2:
            return 0.0
        recent_changes = [
            max(0, c) for c in list(self.score_change_history)[-5:] if np.isfinite(c)
        ]
        if not recent_changes:
            return 0.0
        base_momentum = safe_mean(recent_changes)
        combo_multiplier = 1.0 + (self.combo_counter * 0.1)
        momentum = (base_momentum * combo_multiplier) / 100.0
        return np.clip(momentum, -2.0, 5.0) if np.isfinite(momentum) else 0.0

    def get_combo_stats(self) -> Dict:
        """Get simplified combo stats."""
        return {
            "current_combo": self.combo_counter,
            "max_combo_this_round": self.max_combo_this_round,
            "frame_count": self.current_frame,
        }


class StreetFighterDiscreteActions:
    def __init__(self):
        self.button_names = [
            "B",
            "Y",
            "SELECT",
            "START",
            "UP",
            "DOWN",
            "LEFT",
            "RIGHT",
            "A",
            "X",
            "L",
            "R",
        ]
        self.num_buttons = 12
        self.action_combinations = [
            [],
            [6],
            [7],
            [4],
            [5],
            [6, 4],
            [7, 4],
            [6, 5],
            [7, 5],
            [1],
            [0],
            [1, 6],
            [1, 7],
            [0, 6],
            [0, 7],
            [9],
            [8],
            [9, 6],
            [9, 7],
            [8, 6],
            [8, 7],
            [10],
            [11],
            [10, 6],
            [10, 7],
            [11, 6],
            [11, 7],
            [1, 4],
            [0, 4],
            [9, 4],
            [8, 4],
            [10, 4],
            [11, 4],
            [1, 5],
            [0, 5],
            [9, 5],
            [8, 5],
            [10, 5],
            [11, 5],
            [5, 7],
            [5, 6],
            [5, 7, 1],
            [5, 7, 9],
            [5, 7, 10],
            [5, 6, 1],
            [5, 6, 9],
            [5, 6, 10],
            [5, 7, 0],
            [5, 7, 8],
            [5, 7, 11],
            [4, 5],
            [7, 1],
            [7, 9],
            [7, 10],
            [6],
            [6, 5],
            [4, 6],
        ]
        self.num_actions = len(self.action_combinations)

    def discrete_to_multibinary(self, action_index: int) -> np.ndarray:
        multibinary_action = np.zeros(self.num_buttons, dtype=np.uint8)
        if 0 <= action_index < self.num_actions:
            for button_idx in self.action_combinations[action_index]:
                if 0 <= button_idx < self.num_buttons:
                    multibinary_action[button_idx] = 1
        return multibinary_action

    def get_button_features(self, action_index: int) -> np.ndarray:
        return self.discrete_to_multibinary(action_index).astype(np.float32)


class FixedEnergyBasedStreetFighterCNN(nn.Module):
    """🛡️ FIXED CNN feature extractor with better initialization."""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__()

        visual_space = observation_space["visual_obs"]
        vector_space = observation_space["vector_obs"]
        n_input_channels = visual_space.shape[0]
        seq_length, vector_feature_count = vector_space.shape

        print(f"🔧 FIXED Energy-Based CNN Feature Extractor Configuration:")
        print(f"   - Visual channels: {n_input_channels}")
        print(f"   - Visual size: {visual_space.shape[1]}x{visual_space.shape[2]}")
        print(f"   - Vector sequence: {seq_length} x {vector_feature_count}")
        print(f"   - Output features: {features_dim}")

        # Conservative CNN architecture
        self.visual_cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.05),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.05),
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

        # Vector processing
        self.vector_embed = nn.Linear(vector_feature_count, 64)
        self.vector_norm = nn.LayerNorm(64)
        self.vector_dropout = nn.Dropout(0.1)
        self.vector_gru = nn.GRU(64, 64, batch_first=True, dropout=0.05)
        self.vector_final = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.05),
        )

        # Fusion layer
        fusion_input_size = visual_output_size + 32
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, 512),
            nn.ReLU(inplace=True),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, features_dim),
            nn.ReLU(inplace=True),
        )

        # Better weight initialization
        self.apply(self._init_weights)

        # Monitoring
        self.activation_monitor = {
            "nan_count": 0,
            "explosion_count": 0,
            "forward_count": 0,
        }

        print(f"   - Visual output size: {visual_output_size}")
        print(f"   - Fusion input size: {fusion_input_size}")
        print(f"   ✅ FIXED Energy-Based Feature Extractor initialized")

    def _init_weights(self, m):
        """Better weight initialization."""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.5)
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

        self.activation_monitor["forward_count"] += 1

        # NaN safety
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

        # Input normalization
        visual_obs = torch.clamp(visual_obs / 255.0, 0.0, 1.0)
        vector_obs = torch.clamp(vector_obs, -10.0, 10.0)

        # Process visual features
        visual_features = self.visual_cnn(visual_obs)

        # Monitor for explosions
        if torch.any(torch.abs(visual_features) > 100.0):
            self.activation_monitor["explosion_count"] += 1
            visual_features = torch.clamp(visual_features, -100.0, 100.0)

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

        # Final safety checks
        if torch.any(~torch.isfinite(output)):
            self.activation_monitor["nan_count"] += torch.sum(
                ~torch.isfinite(output)
            ).item()
            output = torch.where(
                ~torch.isfinite(output), torch.zeros_like(output), output
            )

        output = torch.clamp(output, -20.0, 20.0)

        return output

    def get_activation_stats(self):
        """Get activation monitoring statistics."""
        return self.activation_monitor.copy()


class FixedEnergyBasedStreetFighterVerifier(nn.Module):
    """🛡️ FIXED Energy-Based Transformer Verifier with proper scaling."""

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
        self.features_extractor = FixedEnergyBasedStreetFighterCNN(
            observation_space, features_dim
        )

        # Action embedding
        self.action_embed = nn.Sequential(
            nn.Linear(self.action_dim, 64),
            nn.ReLU(inplace=True),
            nn.LayerNorm(64),
            nn.Dropout(0.05),
        )

        # Energy network with proper scaling
        self.energy_net = nn.Sequential(
            nn.Linear(features_dim + 64, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.LayerNorm(128),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.LayerNorm(64),
            nn.Linear(64, 1),
        )

        # Energy scaling parameters
        self.energy_scale = 1.0
        self.energy_clamp_min = -10.0
        self.energy_clamp_max = 10.0

        # Energy landscape monitoring
        self.energy_monitor = {
            "forward_count": 0,
            "nan_count": 0,
            "explosion_count": 0,
            "energy_history": deque(maxlen=1000),
            "gradient_norms": deque(maxlen=1000),
        }

        # Better initialization
        self.apply(self._init_weights)

        print(f"✅ FIXED EnergyBasedStreetFighterVerifier initialized")
        print(f"   - Energy scale: {self.energy_scale}")
        print(f"   - Energy bounds: [{self.energy_clamp_min}, {self.energy_clamp_max}]")

    def _init_weights(self, m):
        """Better weight initialization for energy network."""
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(
        self, context: torch.Tensor, candidate_action: torch.Tensor
    ) -> torch.Tensor:
        """🛡️ FIXED energy calculation with proper scaling."""
        device = next(self.parameters()).device
        self.energy_monitor["forward_count"] += 1

        # Ensure inputs are on correct device and finite
        if isinstance(context, dict):
            context_features = self.features_extractor(context)
        else:
            context_features = context

        context_features = context_features.to(device)
        candidate_action = candidate_action.to(device)

        # Safety checks
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

        # Input clamping
        context_features = torch.clamp(context_features, -20.0, 20.0)
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

        # Monitor for energy explosions
        if torch.any(torch.abs(raw_energy) > 50.0):
            self.energy_monitor["explosion_count"] += 1

        # Scale energy
        energy = raw_energy * self.energy_scale

        # Clamp energy within reasonable bounds
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


class FixedStabilizedEnergyBasedAgent:
    """🛡️ FIXED Energy-Based Agent with adaptive thinking."""

    def __init__(
        self,
        verifier: FixedEnergyBasedStreetFighterVerifier,
        thinking_steps: int = 3,
        thinking_lr: float = 0.1,
        noise_scale: float = 0.1,
    ):
        self.verifier = verifier
        self.initial_thinking_steps = thinking_steps
        self.current_thinking_steps = thinking_steps
        self.initial_thinking_lr = thinking_lr
        self.current_thinking_lr = thinking_lr
        self.noise_scale = noise_scale
        self.action_dim = verifier.action_dim

        # Thinking process parameters
        self.gradient_clip = 1.0
        self.early_stop_patience = 3
        self.min_energy_improvement = 1e-4

        # Adaptive thinking parameters
        self.max_thinking_steps = 8
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

        print(f"✅ FIXED StabilizedEnergyBasedAgent initialized")
        print(f"   - Thinking steps: {thinking_steps}")
        print(f"   - Thinking LR: {thinking_lr}")
        print(f"   - Gradient clip: {self.gradient_clip}")

    def predict(
        self, observations: Dict[str, torch.Tensor], deterministic: bool = False
    ) -> Tuple[int, Dict]:
        """🛡️ FIXED action prediction with proper energy scaling."""
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

        # Thinking loop
        for step in range(self.current_thinking_steps):
            try:
                # Calculate current energy
                energy = self.verifier(obs_device, candidate_action)

                # Check for energy explosion
                if torch.any(torch.abs(energy) > 15.0):
                    energy_explosion = True
                    break

                # Calculate gradient
                gradients = torch.autograd.grad(
                    outputs=energy.sum(),
                    inputs=candidate_action,
                    create_graph=False,
                    retain_graph=False,
                )[0]

                # Gradient monitoring
                gradient_norm = torch.norm(gradients).item()

                if gradient_norm > self.gradient_clip:
                    gradient_explosion = True
                    gradients = gradients * (self.gradient_clip / gradient_norm)

                # Check for NaN gradients
                if torch.any(~torch.isfinite(gradients)):
                    break

                # Update candidate action
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

                # Early stopping
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

        # Final action selection
        with torch.no_grad():
            try:
                final_action_probs = F.softmax(candidate_action, dim=-1)
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


class FixedEnergyStabilityManager:
    """🛡️ FIXED Energy Landscape Stability Manager."""

    def __init__(self, initial_lr=1e-4, thinking_lr=0.1):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.thinking_lr = thinking_lr
        self.current_thinking_lr = thinking_lr

        # Performance tracking
        self.win_rate_window = deque(maxlen=20)
        self.energy_quality_window = deque(maxlen=20)
        self.energy_separation_window = deque(maxlen=20)

        # Realistic emergency thresholds
        self.min_win_rate = 0.20
        self.min_energy_quality = 5.0
        self.min_energy_separation = 0.05
        self.max_early_stop_rate = 0.9

        # Adaptive parameters
        self.lr_decay_factor = 0.7
        self.lr_recovery_factor = 1.1
        self.max_lr_reductions = 5
        self.lr_reductions = 0

        # State tracking
        self.last_reset_episode = 0
        self.consecutive_poor_episodes = 0
        self.best_model_state = None
        self.best_win_rate = 0.0
        self.emergency_mode = False

        print(f"🛡️  FIXED EnergyStabilityManager initialized")
        print(f"   - Min energy separation: {self.min_energy_separation}")
        print(f"   - Min energy quality: {self.min_energy_quality}")

    def update_metrics(
        self, win_rate, energy_quality, energy_separation, early_stop_rate
    ):
        """Update performance metrics and check for instability."""
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
        if collapse_indicators >= 3:
            self.consecutive_poor_episodes += 1
            if self.consecutive_poor_episodes >= 5:
                print(f"🚨 ENERGY LANDSCAPE COLLAPSE DETECTED!")
                return self._trigger_emergency_protocol()
        else:
            self.consecutive_poor_episodes = 0
            self.emergency_mode = False

        return False

    def _trigger_emergency_protocol(self):
        """🚨 Emergency protocol for energy landscape collapse."""
        print(f"🛡️  ACTIVATING EMERGENCY STABILIZATION PROTOCOL")

        if not self.emergency_mode:
            if self.lr_reductions < self.max_lr_reductions:
                self.current_lr *= self.lr_decay_factor
                self.current_thinking_lr *= self.lr_decay_factor
                self.lr_reductions += 1

                print(f"   📉 Learning rates reduced:")
                print(f"      - Main LR: {self.current_lr:.2e}")
                print(f"      - Thinking LR: {self.current_thinking_lr:.3f}")

            self.emergency_mode = True
            self.consecutive_poor_episodes = 0

            return True

        return False

    def get_current_lrs(self):
        """Get current learning rates."""
        return self.current_lr, self.current_thinking_lr


class CheckpointManager:
    """Advanced checkpoint management."""

    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.best_checkpoint_path = None
        self.best_win_rate = 0.0
        self.emergency_checkpoint_path = None

        print(f"💾 CheckpointManager initialized: {checkpoint_dir}")

    def save_checkpoint(
        self, verifier, agent, episode, win_rate, energy_quality, is_emergency=False
    ):
        """Save checkpoint with comprehensive state."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if is_emergency:
            filename = f"emergency_ep{episode}_{timestamp}.pt"
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
        }

        try:
            torch.save(checkpoint_data, checkpoint_path)

            if is_emergency:
                self.emergency_checkpoint_path = checkpoint_path
                print(f"🚨 Emergency checkpoint saved: {filename}")
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
        """Load checkpoint and restore state."""
        try:
            try:
                checkpoint_data = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
            except Exception as first_error:
                checkpoint_data = torch.load(checkpoint_path, map_location="cpu")

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

            return checkpoint_data

        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            return None


class StreetFighterVisionWrapper(gym.Wrapper):
    """🥊 ENHANCED Street Fighter environment wrapper for SINGLE ROUND FIGHTS."""

    def __init__(self, env, frame_stack=8, rendering=False):
        super().__init__(env)
        self.frame_stack = frame_stack
        self.rendering = rendering

        sample_obs = env.reset()
        if isinstance(sample_obs, tuple):
            sample_obs = sample_obs[0]
        if hasattr(sample_obs, "shape"):
            self.target_size = sample_obs.shape[:2]
        else:
            self.target_size = (224, 320)

        self.discrete_actions = StreetFighterDiscreteActions()
        self.action_space = spaces.Discrete(self.discrete_actions.num_actions)

        self.observation_space = spaces.Dict(
            {
                "visual_obs": spaces.Box(
                    low=0,
                    high=255,
                    shape=(3 * frame_stack, *self.target_size),
                    dtype=np.uint8,
                ),
                "vector_obs": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(frame_stack, VECTOR_FEATURE_DIM),
                    dtype=np.float32,
                ),
            }
        )

        self.frame_buffer = deque(maxlen=frame_stack)
        self.vector_features_history = deque(maxlen=frame_stack)
        self.strategic_tracker = SimplifiedFeatureTracker(history_length=frame_stack)

        # Initialize reward calculator and experience buffer
        self.reward_calculator = IntelligentRewardCalculator()
        self.experience_buffer = None  # Will be set by training script

        self.full_hp = 176
        self.prev_player_health = self.full_hp
        self.prev_opponent_health = self.full_hp

        # SINGLE ROUND TRACKING - only track current round
        self.current_round_won = False
        self.current_round_lost = False
        self.wins, self.losses = 0, 0  # Track total single rounds won/lost
        self.total_damage_dealt, self.total_damage_received = 0, 0

        # Episode tracking
        self.episode_steps = 0
        self.max_episode_steps = 12000  # Reduced for single round
        self.episode_rewards = deque(maxlen=100)
        self.episode_count = 0

        # Previous info for reward calculation
        self.prev_info = None

        print(f"🥊 SINGLE ROUND FIGHT wrapper initialized")
        print(f"   - Each episode is ONE decisive round")
        print(f"   - First fighter to reach 0 HP loses")
        print(f"   - Max steps per round: {self.max_episode_steps}")

    def _sanitize_info(self, info: Dict) -> Dict:
        """Converts array values from a vectorized env's info dict to scalars."""
        sanitized = {}
        for k, v in info.items():
            sanitized[k] = ensure_scalar(v, 0)
        return sanitized

    def _create_initial_vector_features(self, info):
        initial_button_features = np.zeros(12, dtype=np.float32)
        try:
            sanitized_info = self._sanitize_info(info)
            initial_features = self.strategic_tracker.update(
                sanitized_info, initial_button_features
            )
            initial_features = ensure_feature_dimension(
                initial_features, VECTOR_FEATURE_DIM
            )
            return initial_features
        except Exception as e:
            print(f"⚠️  Error creating initial features: {e}, using zeros")
            return np.zeros(VECTOR_FEATURE_DIM, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Reset reward calculator for new single round
        self.reward_calculator.reset_episode()

        self.prev_player_health = self.full_hp
        self.prev_opponent_health = self.full_hp
        self.episode_steps = 0
        self.episode_count += 1
        self.prev_info = None

        # Reset single round tracking
        self.current_round_won = False
        self.current_round_lost = False

        processed_frame = self._preprocess_frame(obs)
        initial_vector_features = self._create_initial_vector_features(info)

        self.frame_buffer.clear()
        self.vector_features_history.clear()

        for _ in range(self.frame_stack):
            self.frame_buffer.append(processed_frame)
            self.vector_features_history.append(initial_vector_features.copy())

        return self._get_observation(), info

    def step(self, discrete_action):
        self.episode_steps += 1

        multibinary_action = self.discrete_actions.discrete_to_multibinary(
            discrete_action
        )
        observation, base_reward, done, truncated, info = self.env.step(
            multibinary_action
        )

        if self.rendering:
            self.env.render()

        sanitized_info = self._sanitize_info(info)

        # Calculate intelligent reward with breakdown
        intelligent_reward, reward_breakdown = (
            self.reward_calculator.calculate_reward_with_breakdown(
                sanitized_info, self.prev_info
            )
        )

        # SINGLE ROUND FIGHT LOGIC - Check for knockout conditions
        curr_player_health = ensure_scalar(
            sanitized_info.get("agent_hp", self.full_hp), self.full_hp
        )
        curr_opponent_health = ensure_scalar(
            sanitized_info.get("enemy_hp", self.full_hp), self.full_hp
        )

        # Single round fight ends when either fighter reaches 0 health
        single_round_done = False
        won = False

        if curr_player_health <= 0:
            # AI lost this single round
            single_round_done = True
            won = False
            self.current_round_lost = True
            self.losses += 1
            print(
                f"💀 AI KNOCKED OUT! Single Round Lost! Total: {self.wins}W/{self.losses}L"
            )

        elif curr_opponent_health <= 0:
            # AI won this single round
            single_round_done = True
            won = True
            self.current_round_won = True
            self.wins += 1
            print(
                f"🏆 AI WINS BY KNOCKOUT! Single Round Won! Total: {self.wins}W/{self.losses}L"
            )

        # Add final win/loss reward for single round completion
        if single_round_done:
            final_reward = self.reward_calculator.calculate_win_reward(
                won, curr_player_health, curr_opponent_health, self.episode_steps
            )
            intelligent_reward += final_reward
            reward_breakdown["final"] = final_reward
            done = True  # End episode immediately after single round decision

        # Episode ends if max steps reached (draw/timeout)
        if safe_comparison(self.episode_steps, self.max_episode_steps, ">="):
            truncated = True
            # In case of timeout, determine winner by health
            if curr_player_health > curr_opponent_health:
                won = True
                self.wins += 1
                print(
                    f"⏰ TIME! AI wins by health advantage! Total: {self.wins}W/{self.losses}L"
                )
            elif curr_opponent_health > curr_player_health:
                won = False
                self.losses += 1
                print(
                    f"⏰ TIME! AI loses by health disadvantage! Total: {self.wins}W/{self.losses}L"
                )
            else:
                # True draw - no winner
                won = False
                print(
                    f"⏰ TIME! True draw - no winner! Total: {self.wins}W/{self.losses}L"
                )

            # Add timeout reward
            timeout_reward = self.reward_calculator.calculate_win_reward(
                won, curr_player_health, curr_opponent_health, self.episode_steps
            )
            intelligent_reward += timeout_reward
            reward_breakdown["timeout"] = timeout_reward

        processed_frame = self._preprocess_frame(observation)
        self.frame_buffer.append(processed_frame)

        button_features = self.discrete_actions.get_button_features(discrete_action)

        try:
            vector_features = self.strategic_tracker.update(
                sanitized_info, button_features
            )
            vector_features = ensure_feature_dimension(
                vector_features, VECTOR_FEATURE_DIM
            )
        except Exception as e:
            print(f"⚠️  Vector feature update error: {e}, using zeros")
            vector_features = np.zeros(VECTOR_FEATURE_DIM, dtype=np.float32)

        self.vector_features_history.append(vector_features)

        # Calculate quality score for experience labeling
        quality_score = self.reward_calculator.get_experience_quality_score(
            intelligent_reward, reward_breakdown
        )

        # Create experience and add to buffer if available
        if self.experience_buffer is not None:
            experience = {
                "observations": self._get_observation(),
                "action": discrete_action,
                "step_number": self.episode_steps,
                "info": sanitized_info.copy(),
            }

            self.experience_buffer.add_experience(
                experience, intelligent_reward, reward_breakdown, quality_score
            )

        # Update stats
        self._update_stats()

        # Add reward components to info
        sanitized_info.update(
            {
                "intelligent_reward": intelligent_reward,
                "quality_score": quality_score,
                "reward_breakdown": reward_breakdown,
                "single_round_won": self.current_round_won,
                "single_round_lost": self.current_round_lost,
            }
        )
        sanitized_info.update(self._get_stats())

        # Update previous info
        self.prev_info = sanitized_info.copy()

        return (
            self._get_observation(),
            intelligent_reward,
            done,
            truncated,
            sanitized_info,
        )

    def set_experience_buffer(self, buffer):
        """Set the experience buffer for quality-based labeling."""
        self.experience_buffer = buffer

    def _update_stats(self):
        """Update comprehensive statistics for single round fights."""
        try:
            total_games = self.wins + self.losses
            win_rate = safe_divide(self.wins, total_games, 0.0)

            self.stats = {
                "win_rate": win_rate,
                "wins": self.wins,
                "losses": self.losses,
                "total_single_rounds": total_games,
                "episode_steps": self.episode_steps,
                "current_round_won": self.current_round_won,
                "current_round_lost": self.current_round_lost,
            }
        except Exception as e:
            print(f"⚠️  Error updating stats: {e}")
            self.stats = {"error": "stats_update_failed"}

    def _get_stats(self):
        """Get current statistics."""
        return getattr(self, "stats", {})

    def _get_observation(self):
        try:
            visual_obs = np.concatenate(list(self.frame_buffer), axis=2).transpose(
                2, 0, 1
            )
            vector_obs = np.stack(list(self.vector_features_history))

            visual_obs = sanitize_array(visual_obs, 0.0).astype(np.uint8)
            vector_obs = sanitize_array(vector_obs, 0.0).astype(np.float32)

            expected_shape = (self.frame_stack, VECTOR_FEATURE_DIM)
            if vector_obs.shape != expected_shape:
                corrected_vector_obs = np.zeros(expected_shape, dtype=np.float32)
                min_frames = min(vector_obs.shape[0], expected_shape[0])
                min_features = min(vector_obs.shape[1], expected_shape[1])
                corrected_vector_obs[:min_frames, :min_features] = vector_obs[
                    :min_frames, :min_features
                ]
                vector_obs = corrected_vector_obs

            return {"visual_obs": visual_obs, "vector_obs": vector_obs}
        except Exception as e:
            print(f"⚠️  Error constructing observation: {e}, using fallback")
            visual_obs = np.zeros(
                (3 * self.frame_stack, *self.target_size), dtype=np.uint8
            )
            vector_obs = np.zeros(
                (self.frame_stack, VECTOR_FEATURE_DIM), dtype=np.float32
            )
            return {"visual_obs": visual_obs, "vector_obs": vector_obs}

    def _preprocess_frame(self, frame):
        if frame is None:
            return np.zeros((*self.target_size, 3), dtype=np.uint8)
        try:
            if frame.shape[:2] == self.target_size:
                return frame
            return cv2.resize(frame, (self.target_size[1], self.target_size[0]))
        except Exception as e:
            print(f"⚠️  Frame preprocessing error: {e}, using black frame")
            return np.zeros((*self.target_size, 3), dtype=np.uint8)


def verify_fixed_energy_flow(verifier, env, device=None):
    """Verify FIXED energy flow and gradient computation for single round fights."""
    print("\n🔬 SINGLE ROUND Energy-Based Transformer Verification")
    print("=" * 70)

    if device is None:
        device = next(verifier.parameters()).device

    obs, _ = env.reset()

    obs_tensor = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            value = sanitize_array(value, 0.0)
            obs_tensor[key] = torch.from_numpy(value).unsqueeze(0).float().to(device)
        else:
            obs_tensor[key] = torch.tensor(value).unsqueeze(0).float().to(device)

    visual_obs = obs_tensor["visual_obs"]
    print(f"🖼️  Visual Feature Analysis:")
    print(f"   - Shape: {visual_obs.shape}")
    print(f"   - Range: {visual_obs.min().item():.1f} to {visual_obs.max().item():.1f}")
    print(f"   - NaN count: {torch.sum(~torch.isfinite(visual_obs)).item()}")

    vector_obs = obs_tensor["vector_obs"]
    print(f"🔍 Vector Feature Analysis:")
    print(f"   - Shape: {vector_obs.shape}")
    print(f"   - Features: {vector_obs.shape[-1]} (Fixed)")
    print(f"   - Range: {vector_obs.min().item():.3f} to {vector_obs.max().item():.3f}")
    print(f"   - NaN count: {torch.sum(~torch.isfinite(vector_obs)).item()}")

    verifier.train()
    for param in verifier.parameters():
        param.requires_grad = True

    # Test with random action
    batch_size = obs_tensor["visual_obs"].shape[0]
    random_action = torch.randn(
        batch_size, verifier.action_dim, device=device, requires_grad=True
    )
    random_action = F.softmax(random_action, dim=-1)

    try:
        energy = verifier(obs_tensor, random_action)
        print(f"✅ SINGLE ROUND Energy calculation successful")
        print(f"   - Energy output: {energy.item():.6f}")
        print(f"   - Energy scale: {verifier.energy_scale}")

        gradients = torch.autograd.grad(
            outputs=energy.sum(),
            inputs=random_action,
            create_graph=False,
            retain_graph=False,
        )[0]

        print("✅ SINGLE ROUND Gradient computation successful")
        print(f"   - Gradient norm: {torch.norm(gradients).item():.6f}")

        print(
            "✅ EXCELLENT: SINGLE ROUND Energy-Based Transformer verification successful!"
        )
        return True

    except Exception as e:
        print(f"❌ Energy/gradient computation failed: {e}")
        return False


def make_fixed_env(
    game="StreetFighterIISpecialChampionEdition-Genesis",
    state="ken_bison_12.state",
    render_mode=None,
):
    """Create SINGLE ROUND Energy-Based environment."""
    try:
        env = retro.make(game=game, state=state, render_mode=render_mode)
        env = StreetFighterVisionWrapper(
            env, frame_stack=8, rendering=(render_mode is not None)
        )

        print(f"✅ SINGLE ROUND Energy-Based environment created")
        print(f"   - Feature dimension: {VECTOR_FEATURE_DIM}")
        print(f"   - Single round fights: ✅ ACTIVE")
        print(f"   - Quality-based experience labeling: ✅ ACTIVE")
        print(f"   - Intelligent reward shaping: ✅ ACTIVE")
        print(f"   - Strategic feedback: ✅ ACTIVE")

        return env
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        raise


# Export all components
__all__ = [
    # Core components
    "StreetFighterVisionWrapper",
    "FixedEnergyBasedStreetFighterCNN",
    "FixedEnergyBasedStreetFighterVerifier",
    "FixedStabilizedEnergyBasedAgent",
    "SimplifiedFeatureTracker",
    "StreetFighterDiscreteActions",
    # Quality-based components
    "IntelligentRewardCalculator",
    "QualityBasedExperienceBuffer",
    # Stability components
    "FixedEnergyStabilityManager",
    "CheckpointManager",
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

print(f"🥊 SINGLE ROUND FIGHT - Complete wrapper.py loaded successfully!")
print(f"   - Training paradigm: Quality-Based Experience Labeling")
print(f"   - Fight mode: Single decisive rounds")
print(f"   - Intelligent reward shaping: ✅ ACTIVE")
print(f"   - Absolute quality thresholds: ✅ ACTIVE")
print(f"   - Strategic feedback: ✅ ACTIVE")
print(f"   - Dense learning signals: ✅ ACTIVE")
print(f"🎯 Ready for single round 60% win rate training!")
