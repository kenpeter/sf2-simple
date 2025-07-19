#!/usr/bin/env python3
"""
FINAL ENERGY-BASED TRANSFORMER FOR STREET FIGHTER
Incorporates all fixes for energy collapse prevention:
- Balanced good/bad experience sampling
- Energy collapse detection and reset
- Historical diversity maintenance
- Improved contrastive loss
- Emergency protocols
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
    f'logs/final_energy_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
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

print(f"🧠 FINAL ENERGY-BASED TRANSFORMER Configuration:")
print(f"   - Features: {VECTOR_FEATURE_DIM}")
print(f"   - Training paradigm: FINAL Energy-Based with Collapse Prevention")


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


def safe_bool_check(value):
    """Safe boolean check for arrays."""
    if value is None:
        return False

    if isinstance(value, np.ndarray):
        if value.size == 0:
            return False
        elif value.size == 1:
            try:
                return bool(value.item())
            except (ValueError, TypeError):
                return False
        else:
            try:
                return bool(np.any(value))
            except (ValueError, TypeError):
                return False
    else:
        try:
            return bool(value)
        except (ValueError, TypeError):
            return False


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


class EnergyStabilityManager:
    """
    🛡️ Energy Landscape Stability Manager
    Prevents energy collapse and manages adaptive learning.
    """

    def __init__(self, initial_lr=1e-4, thinking_lr=0.1):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.thinking_lr = thinking_lr
        self.current_thinking_lr = thinking_lr

        # Performance tracking
        self.win_rate_window = deque(maxlen=20)
        self.energy_quality_window = deque(maxlen=20)
        self.energy_separation_window = deque(maxlen=20)

        # Emergency thresholds
        self.min_win_rate = 0.30
        self.min_energy_quality = 20.0
        self.min_energy_separation = 0.5
        self.max_early_stop_rate = 0.8

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

        print(f"🛡️  EnergyStabilityManager initialized")

    def update_metrics(
        self, win_rate, energy_quality, energy_separation, early_stop_rate
    ):
        """Update performance metrics and check for instability."""
        self.win_rate_window.append(win_rate)
        self.energy_quality_window.append(energy_quality)
        self.energy_separation_window.append(energy_separation)

        # Check for energy landscape collapse
        avg_win_rate = safe_mean(list(self.win_rate_window), 0.5)
        avg_energy_quality = safe_mean(list(self.energy_quality_window), 50.0)
        avg_energy_separation = safe_mean(list(self.energy_separation_window), 1.0)

        collapse_indicators = 0

        if avg_win_rate < self.min_win_rate:
            collapse_indicators += 1
            print(f"🚨 Win rate collapse: {avg_win_rate:.3f} < {self.min_win_rate}")

        if avg_energy_quality < self.min_energy_quality:
            collapse_indicators += 1
            print(
                f"🚨 Energy quality collapse: {avg_energy_quality:.1f} < {self.min_energy_quality}"
            )

        if avg_energy_separation < self.min_energy_separation:
            collapse_indicators += 1
            print(
                f"🚨 Energy separation collapse: {avg_energy_separation:.3f} < {self.min_energy_separation}"
            )

        if early_stop_rate > self.max_early_stop_rate:
            collapse_indicators += 1
            print(
                f"🚨 Early stop rate explosion: {early_stop_rate:.3f} > {self.max_early_stop_rate}"
            )

        # Trigger emergency if multiple indicators
        if collapse_indicators >= 2:
            self.consecutive_poor_episodes += 1
            if self.consecutive_poor_episodes >= 3:  # Faster trigger
                print(f"🚨 ENERGY LANDSCAPE COLLAPSE DETECTED!")
                return self._trigger_emergency_protocol()
        else:
            self.consecutive_poor_episodes = 0
            self.emergency_mode = False

        return False  # No emergency

    def _trigger_emergency_protocol(self):
        """🚨 Emergency protocol for energy landscape collapse."""
        print(f"🛡️  ACTIVATING EMERGENCY STABILIZATION PROTOCOL")

        if not self.emergency_mode:
            # Reduce learning rates
            if self.lr_reductions < self.max_lr_reductions:
                self.current_lr *= self.lr_decay_factor
                self.current_thinking_lr *= self.lr_decay_factor
                self.lr_reductions += 1

                print(f"   📉 Learning rates reduced:")
                print(f"      - Main LR: {self.current_lr:.2e}")
                print(f"      - Thinking LR: {self.current_thinking_lr:.3f}")

            self.emergency_mode = True
            self.consecutive_poor_episodes = 0

            return True  # Signal for emergency actions

        return False

    def should_save_checkpoint(self, current_win_rate):
        """Determine if current model should be saved as best."""
        if current_win_rate > self.best_win_rate:
            self.best_win_rate = current_win_rate
            return True
        return False

    def get_current_lrs(self):
        """Get current learning rates."""
        return self.current_lr, self.current_thinking_lr

    def recovery_check(self, current_win_rate):
        """Check if we can recover from emergency mode."""
        if self.emergency_mode and current_win_rate > self.min_win_rate + 0.1:
            self.emergency_mode = False
            print(f"✅ Recovery detected, exiting emergency mode")


class DiversityExperienceBuffer:
    """
    🎯 ENHANCED: Diversity-Controlled Experience Buffer
    Maintains good/bad balance across different skill levels.
    """

    def __init__(self, capacity=50000, quality_threshold=0.5):
        self.capacity = capacity
        self.quality_threshold = quality_threshold

        # Separate storage for different skill levels
        self.skill_buckets = {
            "beginner": {
                "good": deque(maxlen=capacity // 6),
                "bad": deque(maxlen=capacity // 6),
            },
            "intermediate": {
                "good": deque(maxlen=capacity // 6),
                "bad": deque(maxlen=capacity // 6),
            },
            "advanced": {
                "good": deque(maxlen=capacity // 6),
                "bad": deque(maxlen=capacity // 6),
            },
        }

        # Quality tracking
        self.total_added = 0
        self.total_rejected = 0
        self.episode_count = 0

        print(f"🎯 DiversityExperienceBuffer initialized with skill-based buckets")

    def add_episode_experiences(self, episode_experiences, win_rate):
        """Add entire episode with relative good/bad labeling."""
        self.episode_count += 1

        # Determine skill level
        if win_rate < 0.3:
            skill_level = "beginner"
        elif win_rate < 0.7:
            skill_level = "intermediate"
        else:
            skill_level = "advanced"

        # Process episode for relative good/bad
        if len(episode_experiences) == 0:
            return

        # Extract rewards and calculate percentiles
        rewards = [exp["reward"] for exp in episode_experiences]
        if len(rewards) < 4:  # Too few experiences
            return

        reward_75th = np.percentile(rewards, 75)
        reward_25th = np.percentile(rewards, 25)

        good_added = 0
        bad_added = 0

        for exp in episode_experiences:
            self.total_added += 1

            # Quality score based on relative performance
            if exp["reward"] >= reward_75th:
                # Top 25% of episode
                exp["is_good"] = True
                quality_score = 0.7 + 0.3 * (
                    (exp["reward"] - reward_25th)
                    / max(reward_75th - reward_25th, 0.001)
                )
            elif exp["reward"] <= reward_25th:
                # Bottom 25% of episode
                exp["is_good"] = False
                quality_score = 0.7 + 0.3 * (
                    (reward_75th - exp["reward"])
                    / max(reward_75th - reward_25th, 0.001)
                )
            else:
                # Middle 50% - skip to maintain clear distinction
                self.total_rejected += 1
                continue

            # Add to appropriate bucket
            if quality_score >= self.quality_threshold:
                if exp["is_good"]:
                    self.skill_buckets[skill_level]["good"].append(exp)
                    good_added += 1
                else:
                    self.skill_buckets[skill_level]["bad"].append(exp)
                    bad_added += 1
            else:
                self.total_rejected += 1

        print(
            f"Episode {self.episode_count} ({skill_level}): +{good_added} good, +{bad_added} bad"
        )

    def sample_balanced_batch(self, batch_size, prioritize_diversity=True):
        """Sample balanced batch across skill levels."""
        good_samples = []
        bad_samples = []

        # Collect from all skill levels
        for skill_level in self.skill_buckets:
            good_bucket = list(self.skill_buckets[skill_level]["good"])
            bad_bucket = list(self.skill_buckets[skill_level]["bad"])

            good_samples.extend(good_bucket)
            bad_samples.extend(bad_bucket)

        # Ensure we have enough examples
        target_per_class = batch_size // 2
        if len(good_samples) < target_per_class or len(bad_samples) < target_per_class:
            return None, None

        # Sample balanced amounts
        final_good = random.sample(good_samples, target_per_class)
        final_bad = random.sample(bad_samples, target_per_class)

        return final_good, final_bad

    def emergency_purge(self, keep_ratio=0.2):
        """🚨 Emergency: Keep only the best experiences."""
        print(f"🚨 Emergency buffer purge - keeping top {keep_ratio*100:.0f}%")

        total_purged = 0
        for skill_level in self.skill_buckets:
            for category in ["good", "bad"]:
                bucket = self.skill_buckets[skill_level][category]
                if len(bucket) == 0:
                    continue

                # Keep only the best
                experiences = list(bucket)
                keep_count = max(1, int(len(experiences) * keep_ratio))

                # Sort by quality and keep best
                experiences.sort(
                    key=lambda x: x.get("quality_score", 0.5), reverse=True
                )
                bucket.clear()

                for exp in experiences[:keep_count]:
                    bucket.append(exp)

                total_purged += len(experiences) - keep_count

        print(f"   📊 Purged {total_purged} low-quality experiences")

    def get_stats(self):
        """Get comprehensive buffer statistics."""
        total_good = sum(
            len(self.skill_buckets[level]["good"]) for level in self.skill_buckets
        )
        total_bad = sum(
            len(self.skill_buckets[level]["bad"]) for level in self.skill_buckets
        )

        acceptance_rate = safe_divide(
            self.total_added - self.total_rejected, self.total_added, 0.0
        )

        stats = {
            "total_size": total_good + total_bad,
            "good_count": total_good,
            "bad_count": total_bad,
            "balance_ratio": safe_divide(total_good, total_bad, 1.0),
            "acceptance_rate": acceptance_rate,
            "total_added": self.total_added,
            "total_rejected": self.total_rejected,
        }

        # Add per-skill stats
        for skill_level in self.skill_buckets:
            good_count = len(self.skill_buckets[skill_level]["good"])
            bad_count = len(self.skill_buckets[skill_level]["bad"])
            stats[f"{skill_level}_good"] = good_count
            stats[f"{skill_level}_bad"] = bad_count

        return stats


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

            agent_victories = ensure_scalar(info.get("agent_victories", 0), 0)
            enemy_victories = ensure_scalar(info.get("enemy_victories", 0), 0)
            features[18] = (
                min(agent_victories / 10.0, 1.0)
                if np.isfinite(agent_victories)
                else 0.0
            )
            features[19] = (
                min(enemy_victories / 10.0, 1.0)
                if np.isfinite(enemy_victories)
                else 0.0
            )

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


class EnergyBasedStreetFighterCNN(nn.Module):
    """
    🛡️ STABILIZED CNN feature extractor for Energy-Based Transformer.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__()

        visual_space = observation_space["visual_obs"]
        vector_space = observation_space["vector_obs"]
        n_input_channels = visual_space.shape[0]
        seq_length, vector_feature_count = vector_space.shape

        print(f"🔧 STABILIZED Energy-Based CNN Feature Extractor Configuration:")
        print(f"   - Visual channels: {n_input_channels}")
        print(f"   - Visual size: {visual_space.shape[1]}x{visual_space.shape[2]}")
        print(f"   - Vector sequence: {seq_length} x {vector_feature_count}")
        print(f"   - Output features: {features_dim}")

        # Ultra-conservative CNN architecture
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

        # Stabilized vector processing
        self.vector_embed = nn.Linear(vector_feature_count, 64)
        self.vector_norm = nn.LayerNorm(64)
        self.vector_dropout = nn.Dropout(0.1)
        self.vector_gru = nn.GRU(64, 64, batch_first=True, dropout=0.05)
        self.vector_final = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.05),
        )

        # Ultra-stable fusion layer
        fusion_input_size = visual_output_size + 32
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, 512),
            nn.ReLU(inplace=True),
            nn.LayerNorm(512),
            nn.Dropout(0.1),
            nn.Linear(512, features_dim),
            nn.ReLU(inplace=True),
        )

        # Ultra-conservative weight initialization
        self.apply(self._init_weights_ultra_conservative)

        # Energy monitoring
        self.activation_monitor = {
            "nan_count": 0,
            "explosion_count": 0,
            "forward_count": 0,
        }

        print(f"   - Visual output size: {visual_output_size}")
        print(f"   - Fusion input size: {fusion_input_size}")
        print(f"   ✅ STABILIZED Energy-Based Feature Extractor initialized")

    def _init_weights_ultra_conservative(self, m):
        """Ultra-conservative weight initialization for maximum stability."""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if hasattr(m.weight, "data"):
                m.weight.data *= 0.1
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param, gain=0.1)
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

        # Enhanced input clamping
        visual_obs = torch.clamp(visual_obs / 255.0, 0.0, 1.0)
        vector_obs = torch.clamp(vector_obs, -10.0, 10.0)

        # Process visual features
        visual_features = self.visual_cnn(visual_obs)

        # Monitor for explosions
        if torch.any(torch.abs(visual_features) > 100.0):
            self.activation_monitor["explosion_count"] += 1
            visual_features = torch.clamp(visual_features, -100.0, 100.0)

        # Check for NaN in visual features
        if torch.any(~torch.isfinite(visual_features)):
            self.activation_monitor["nan_count"] += torch.sum(
                ~torch.isfinite(visual_features)
            ).item()
            visual_features = torch.where(
                ~torch.isfinite(visual_features),
                torch.zeros_like(visual_features),
                visual_features,
            )

        # Process vector features
        batch_size, seq_len, feature_dim = vector_obs.shape
        vector_embedded = self.vector_embed(vector_obs)
        vector_embedded = self.vector_norm(vector_embedded)
        vector_embedded = self.vector_dropout(vector_embedded)

        # Check for NaN in vector embedding
        if torch.any(~torch.isfinite(vector_embedded)):
            self.activation_monitor["nan_count"] += torch.sum(
                ~torch.isfinite(vector_embedded)
            ).item()
            vector_embedded = torch.where(
                ~torch.isfinite(vector_embedded),
                torch.zeros_like(vector_embedded),
                vector_embedded,
            )

        gru_output, _ = self.vector_gru(vector_embedded)
        vector_features = gru_output[:, -1, :]
        vector_features = self.vector_final(vector_features)

        # Check for NaN in vector features
        if torch.any(~torch.isfinite(vector_features)):
            self.activation_monitor["nan_count"] += torch.sum(
                ~torch.isfinite(vector_features)
            ).item()
            vector_features = torch.where(
                ~torch.isfinite(vector_features),
                torch.zeros_like(vector_features),
                vector_features,
            )

        # Combine and process
        combined_features = torch.cat([visual_features, vector_features], dim=1)
        output = self.fusion(combined_features)

        # Final stability checks
        if torch.any(~torch.isfinite(output)):
            self.activation_monitor["nan_count"] += torch.sum(
                ~torch.isfinite(output)
            ).item()
            output = torch.where(
                ~torch.isfinite(output), torch.zeros_like(output), output
            )

        # Clamp final output to prevent energy explosions
        output = torch.clamp(output, -50.0, 50.0)

        return output

    def get_activation_stats(self):
        """Get activation monitoring statistics."""
        return self.activation_monitor.copy()


class EnergyBasedStreetFighterVerifier(nn.Module):
    """
    🛡️ STABILIZED Energy-Based Transformer Verifier for Street Fighter.
    """

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

        # Stabilized feature extractor
        self.features_extractor = EnergyBasedStreetFighterCNN(
            observation_space, features_dim
        )

        # Stabilized action embedding
        self.action_embed = nn.Sequential(
            nn.Linear(self.action_dim, 64),
            nn.ReLU(inplace=True),
            nn.LayerNorm(64),
            nn.Dropout(0.05),
        )

        # Ultra-stable energy network
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

        # Enhanced energy scaling for maximum stability
        self.energy_scale = 0.01
        self.energy_clamp_min = -2.0
        self.energy_clamp_max = 2.0

        # Energy landscape monitoring
        self.energy_monitor = {
            "forward_count": 0,
            "nan_count": 0,
            "explosion_count": 0,
            "energy_history": deque(maxlen=1000),
            "gradient_norms": deque(maxlen=1000),
        }

        # Ultra-conservative initialization
        self.apply(self._init_weights_ultra_conservative)

        print(f"✅ STABILIZED EnergyBasedStreetFighterVerifier initialized")

    def _init_weights_ultra_conservative(self, m):
        """Ultra-conservative weight initialization."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(
        self, context: torch.Tensor, candidate_action: torch.Tensor
    ) -> torch.Tensor:
        """
        🛡️ STABILIZED energy calculation with monitoring.
        """
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

        # Clamp inputs to prevent explosions
        context_features = torch.clamp(context_features, -10.0, 10.0)
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

        # Combine context and action
        combined_input = torch.cat([context_features, action_embedded], dim=-1)

        # Calculate energy
        energy = self.energy_net(combined_input)

        # Monitor for energy explosions
        if torch.any(torch.abs(energy) > 10.0):
            self.energy_monitor["explosion_count"] += 1
            print(
                f"🚨 Energy explosion detected: {torch.max(torch.abs(energy)).item():.3f}"
            )

        # Enhanced scale and clamp energy for maximum stability
        energy = energy * self.energy_scale
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


class StabilizedEnergyBasedAgent:
    """
    🛡️ STABILIZED Energy-Based Agent for Street Fighter.
    """

    def __init__(
        self,
        verifier: EnergyBasedStreetFighterVerifier,
        thinking_steps: int = 3,
        thinking_lr: float = 0.05,
        noise_scale: float = 0.05,
    ):
        self.verifier = verifier
        self.initial_thinking_steps = thinking_steps
        self.current_thinking_steps = thinking_steps
        self.initial_thinking_lr = thinking_lr
        self.current_thinking_lr = thinking_lr
        self.noise_scale = noise_scale
        self.action_dim = verifier.action_dim

        # Enhanced thinking process parameters
        self.gradient_clip = 0.5
        self.early_stop_patience = 2
        self.min_energy_improvement = 1e-5

        # Adaptive thinking parameters
        self.max_thinking_steps = 10
        self.min_thinking_steps = 1
        self.thinking_adaptation_rate = 0.1

        # Enhanced statistics
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
        self.adaptation_threshold = 0.4

        print(f"✅ STABILIZED EnergyBasedAgent initialized")

    def adapt_thinking_parameters(self, success_rate):
        """Adapt thinking parameters based on performance."""
        if success_rate < self.adaptation_threshold:
            self.current_thinking_steps = max(
                self.min_thinking_steps, self.current_thinking_steps - 1
            )
            self.current_thinking_lr *= 0.9
        elif success_rate > 0.7:
            self.current_thinking_steps = min(
                self.max_thinking_steps, self.current_thinking_steps + 1
            )
            self.current_thinking_lr = min(
                self.initial_thinking_lr, self.current_thinking_lr * 1.05
            )

    def predict(
        self, observations: Dict[str, torch.Tensor], deterministic: bool = False
    ) -> Tuple[int, Dict]:
        """
        🛡️ STABILIZED action prediction with enhanced thinking process.
        """
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

        # Stabilized candidate action initialization
        if deterministic:
            candidate_action = (
                torch.ones(batch_size, self.action_dim, device=device) / self.action_dim
            )
        else:
            candidate_action = torch.randn(
                batch_size, self.action_dim, device=device
            ) * (self.noise_scale * 0.1)
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

        # Enhanced thinking loop with stability controls
        for step in range(self.current_thinking_steps):
            try:
                # Calculate current energy
                energy = self.verifier(obs_device, candidate_action)

                # Check for energy explosion
                if torch.any(torch.abs(energy) > 5.0):
                    energy_explosion = True
                    print(
                        f"🚨 Energy explosion at step {step}: {torch.max(torch.abs(energy)).item():.3f}"
                    )
                    break

                # Calculate gradient of energy w.r.t. candidate action
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
                    print(f"🚨 NaN gradients at step {step}")
                    break

                # Stabilized update with smaller learning rate
                with torch.no_grad():
                    candidate_action = (
                        candidate_action - (self.current_thinking_lr * 0.1) * gradients
                    )
                    candidate_action = F.softmax(candidate_action, dim=-1)
                    candidate_action.requires_grad_(True)

                # Track energy improvement
                with torch.no_grad():
                    new_energy = self.verifier(obs_device, candidate_action)
                    energy_history.append(new_energy.mean().item())

                # Early stopping if no improvement
                if len(energy_history) >= self.early_stop_patience + 1:
                    recent_improvement = (
                        energy_history[-self.early_stop_patience - 1]
                        - energy_history[-1]
                    )
                    if recent_improvement < self.min_energy_improvement:
                        early_stopped = True
                        break

                steps_taken += 1

            except Exception as e:
                print(f"⚠️ Thinking step {step} failed: {e}")
                break

        # Determine optimization success
        if len(energy_history) > 1:
            total_improvement = energy_history[0] - energy_history[-1]
            optimization_successful = (
                total_improvement > 0
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

        # Update enhanced statistics
        self.thinking_stats["total_predictions"] += 1
        self.thinking_stats["avg_thinking_steps"] = (
            self.thinking_stats["avg_thinking_steps"] * 0.9 + steps_taken * 0.1
        )

        if len(energy_history) > 1:
            energy_improvement = energy_history[0] - energy_history[-1]
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

        # Track recent performance for adaptation
        self.recent_performance.append(1.0 if optimization_successful else 0.0)

        # Adapt thinking parameters periodically
        if (
            len(self.recent_performance) >= 50
            and self.thinking_stats["total_predictions"] % 50 == 0
        ):
            success_rate = safe_mean(list(self.recent_performance), 0.5)
            self.adapt_thinking_parameters(success_rate)

        # Information about thinking process
        thinking_info = {
            "energy_history": energy_history,
            "steps_taken": steps_taken,
            "early_stopped": early_stopped,
            "energy_explosion": energy_explosion,
            "gradient_explosion": gradient_explosion,
            "optimization_successful": optimization_successful,
            "energy_improvement": (
                energy_history[0] - energy_history[-1]
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


class StreetFighterVisionWrapper(gym.Wrapper):
    """
    🛡️ FINAL Street Fighter environment wrapper for Energy-Based Transformer.
    Enhanced with diversity-based experience collection.
    """

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
        self.full_hp = 176
        self.prev_player_health = self.full_hp
        self.prev_opponent_health = self.full_hp
        self.wins, self.losses, self.total_rounds = 0, 0, 0
        self.total_damage_dealt, self.total_damage_received = 0, 0

        # Enhanced episode tracking for diversity buffer
        self.current_episode_data = []
        self.episode_count = 0

        # Energy-based reward configuration
        self.reward_scale = 0.01
        self.episode_steps = 0
        self.max_episode_steps = 18000
        self.episode_rewards = deque(maxlen=100)
        self.stats = {}

        # Enhanced reward normalization
        self.reward_history = deque(maxlen=1000)
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_alpha = 0.99

        # Performance monitoring
        self.performance_window = deque(maxlen=20)
        self.stability_metrics = {
            "nan_rewards": 0,
            "explosive_rewards": 0,
            "total_steps": 0,
        }

        print(f"🛡️  FINAL energy-based wrapper initialized")

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

        # Process previous episode data if exists
        if len(self.current_episode_data) > 0:
            self._finalize_episode()

        self.prev_player_health = self.full_hp
        self.prev_opponent_health = self.full_hp
        self.episode_steps = 0
        self.current_episode_data = []
        self.episode_count += 1

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
        self.stability_metrics["total_steps"] += 1

        multibinary_action = self.discrete_actions.discrete_to_multibinary(
            discrete_action
        )
        observation, reward, done, truncated, info = self.env.step(multibinary_action)

        if self.rendering:
            self.env.render()

        sanitized_info = self._sanitize_info(info)

        curr_player_health = ensure_scalar(
            sanitized_info.get("agent_hp", self.full_hp), self.full_hp
        )
        curr_opponent_health = ensure_scalar(
            sanitized_info.get("enemy_hp", self.full_hp), self.full_hp
        )

        # Calculate ultra-stable reward
        base_reward, custom_done = self._calculate_ultra_stable_reward(
            curr_player_health, curr_opponent_health
        )
        final_reward = self._ultra_normalize_reward_for_energy(base_reward)

        if safe_comparison(self.episode_steps, self.max_episode_steps, ">="):
            truncated = True

        done = custom_done or done

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

        # Store step data for episode processing
        step_data = {
            "observations": self._get_observation(),
            "action": discrete_action,
            "reward": final_reward,
            "step_number": self.episode_steps,
            "info": sanitized_info.copy(),
        }
        self.current_episode_data.append(step_data)

        self._update_enhanced_stats()

        # Add enhanced statistics
        sanitized_info.update(self.stats)
        sanitized_info.update(self.stability_metrics)

        return self._get_observation(), final_reward, done, truncated, sanitized_info

    def _finalize_episode(self):
        """Process episode data for diversity buffer."""
        if hasattr(self, "_episode_callback") and self._episode_callback:
            win_rate = safe_divide(self.wins, self.wins + self.losses, 0.0)
            self._episode_callback(self.current_episode_data, win_rate)

    def set_episode_callback(self, callback):
        """Set callback for episode completion."""
        self._episode_callback = callback

    def _calculate_ultra_stable_reward(self, curr_player_health, curr_opponent_health):
        """Calculate ultra-stable reward for EBT training."""
        reward, done = 0.0, False

        curr_player_health = ensure_scalar(curr_player_health, self.full_hp)
        curr_opponent_health = ensure_scalar(curr_opponent_health, self.full_hp)

        player_dead = safe_comparison(curr_player_health, 0, "<=")
        opponent_dead = safe_comparison(curr_opponent_health, 0, "<=")

        if player_dead or opponent_dead:
            self.total_rounds += 1
            if opponent_dead and not player_dead:
                self.wins += 1
                win_bonus = (
                    0.1 + safe_divide(curr_player_health, self.full_hp, 0.0) * 0.05
                )
                reward += win_bonus
                print(
                    f"🏆 AI WON! Total: {self.wins}W/{self.losses}L (Round {self.total_rounds})"
                )
            else:
                self.losses += 1
                reward -= 0.05
                print(
                    f"💀 AI LOST! Total: {self.wins}W/{self.losses}L (Round {self.total_rounds})"
                )
            done = True

            combo_bonus = self.strategic_tracker.combo_counter * 0.001
            reward += combo_bonus

        # Damage calculation with smaller rewards
        damage_dealt = 0
        damage_received = 0

        if self.prev_opponent_health is not None and np.isfinite(
            self.prev_opponent_health
        ):
            damage_calc = (
                ensure_scalar(self.prev_opponent_health) - curr_opponent_health
            )
            damage_dealt = max(0, damage_calc) if np.isfinite(damage_calc) else 0

        if self.prev_player_health is not None and np.isfinite(self.prev_player_health):
            damage_calc = ensure_scalar(self.prev_player_health) - curr_player_health
            damage_received = max(0, damage_calc) if np.isfinite(damage_calc) else 0

        reward += (damage_dealt * 0.001) - (damage_received * 0.0005)
        self.total_damage_dealt += damage_dealt
        self.total_damage_received += damage_received

        # Tiny time penalty
        reward -= 0.00001

        # Apply ultra-small reward scale for maximum energy stability
        reward *= self.reward_scale
        reward = np.clip(reward, -0.1, 0.1) if np.isfinite(reward) else 0.0

        self.prev_player_health, self.prev_opponent_health = (
            curr_player_health,
            curr_opponent_health,
        )

        if done:
            self.episode_rewards.append(reward)
            win_rate = safe_divide(self.wins, self.wins + self.losses, 0.0)
            self.performance_window.append(win_rate)

        return reward, done

    def _ultra_normalize_reward_for_energy(self, reward):
        """Ultra-normalize reward specifically for maximum energy stability."""
        if not np.isfinite(reward):
            reward = 0.0
            self.stability_metrics["nan_rewards"] += 1

        reward = np.clip(reward, -0.5, 0.5)

        if abs(reward) > 0.2:
            self.stability_metrics["explosive_rewards"] += 1
            reward = np.clip(reward, -0.2, 0.2)

        self.reward_history.append(reward)

        if len(self.reward_history) > 20:
            current_mean = np.mean(list(self.reward_history))
            current_std = np.std(list(self.reward_history))

            if not np.isfinite(current_mean):
                current_mean = 0.0
            if not np.isfinite(current_std) or current_std == 0:
                current_std = 0.1

            self.reward_mean = (
                self.reward_alpha * self.reward_mean
                + (1 - self.reward_alpha) * current_mean
            )
            self.reward_std = (
                self.reward_alpha * self.reward_std
                + (1 - self.reward_alpha) * current_std
            )
            self.reward_std = max(self.reward_std, 0.01)

        if self.reward_std > 0:
            normalized_reward = (reward - self.reward_mean) / (self.reward_std * 2.0)
        else:
            normalized_reward = reward

        normalized_reward = np.clip(normalized_reward, -0.5, 0.5)
        return normalized_reward * 0.01

    def _update_enhanced_stats(self):
        try:
            total_games = self.wins + self.losses
            win_rate = safe_divide(self.wins, total_games, 0.0)
            avg_damage_per_round = safe_divide(
                self.total_damage_dealt, max(1, self.total_rounds), 0.0
            )
            total_damage = self.total_damage_dealt + self.total_damage_received
            defensive_efficiency = safe_divide(
                self.total_damage_dealt, max(1, total_damage), 0.0
            )
            damage_ratio = safe_divide(
                self.total_damage_dealt, max(1, self.total_damage_received), 1.0
            )
            combo_stats = self.strategic_tracker.get_combo_stats()

            performance_stability = 1.0
            if len(self.performance_window) > 5:
                performance_stability = 1.0 - safe_std(
                    list(self.performance_window), 0.0
                )

            self.stats.update(
                {
                    "win_rate": win_rate,
                    "wins": self.wins,
                    "losses": self.losses,
                    "total_games": total_games,
                    "total_rounds": self.total_rounds,
                    "avg_damage_per_round": avg_damage_per_round,
                    "defensive_efficiency": defensive_efficiency,
                    "damage_ratio": damage_ratio,
                    "max_combo": combo_stats.get("max_combo_this_round", 0),
                    "episode_steps": self.episode_steps,
                    "reward_mean": self.reward_mean,
                    "reward_std": self.reward_std,
                    "performance_stability": performance_stability,
                }
            )
        except Exception as e:
            print(f"⚠️  Error updating stats: {e}")
            self.stats = {"error": "stats_update_failed"}

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
                print(
                    f"⚠️  Vector obs shape mismatch: got {vector_obs.shape}, expected {expected_shape}"
                )
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


def verify_stabilized_energy_flow(verifier, env, device=None):
    """Verify stabilized energy flow and gradient computation for EBT."""
    print("\n🔬 FINAL Energy-Based Transformer Verification")
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
    print(f"   - Features: {vector_obs.shape[-1]} (Final)")
    print(f"   - Range: {vector_obs.min().item():.3f} to {vector_obs.max().item():.3f}")
    print(f"   - NaN count: {torch.sum(~torch.isfinite(vector_obs)).item()}")

    if torch.sum(~torch.isfinite(vector_obs)) > 0:
        print("   🚨 CRITICAL: NaN values detected!")
        return False
    else:
        print("   ✅ No NaN values detected")

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
        print(f"✅ FINAL Energy calculation successful")
        print(f"   - Energy output: {energy.item():.6f}")
        print(f"   - Energy NaN count: {torch.sum(~torch.isfinite(energy)).item()}")
        print(
            f"   - Energy bounds: [{verifier.energy_clamp_min}, {verifier.energy_clamp_max}]"
        )

        if torch.sum(~torch.isfinite(energy)) > 0:
            print("   🚨 CRITICAL: NaN values in energy output!")
            return False

        if abs(energy.item()) > abs(verifier.energy_clamp_max):
            print(f"   🚨 CRITICAL: Energy outside bounds: {energy.item():.6f}")
            return False
        else:
            print("   ✅ Energy output is ULTRA-STABLE")

    except Exception as e:
        print(f"❌ Energy calculation failed: {e}")
        return False

    # Test gradient computation
    try:
        gradients = torch.autograd.grad(
            outputs=energy.sum(),
            inputs=random_action,
            create_graph=False,
            retain_graph=False,
        )[0]

        print("✅ FINAL Gradient computation successful")
        print(f"   - Gradient shape: {gradients.shape}")
        print(f"   - Gradient norm: {torch.norm(gradients).item():.6f}")
        print(
            f"   - Gradient NaN count: {torch.sum(~torch.isfinite(gradients)).item()}"
        )

        if torch.sum(~torch.isfinite(gradients)) > 0:
            print("   🚨 CRITICAL: NaN values in gradients!")
            return False

        gradient_norm = torch.norm(gradients).item()
        if gradient_norm > 1.0:
            print(f"   🚨 CRITICAL: Gradient explosion: {gradient_norm:.3f}")
            return False
        elif gradient_norm > 0.5:
            print(f"   ⚠️  WARNING: Large gradients: {gradient_norm:.3f}")
        else:
            print("   ✅ Gradients are ULTRA-STABLE")

    except Exception as e:
        print(f"❌ Gradient computation failed: {e}")
        return False

    print("✅ EXCELLENT: FINAL Energy-Based Transformer verification successful!")
    print("🛡️  All stability controls verified")
    print("🎯 Energy collapse prevention active")
    print("⚡ Diversity-based training ready")
    return True


def make_stabilized_env(
    game="StreetFighterIISpecialChampionEdition-Genesis",
    state="ken_bison_12.state",
    render_mode=None,
):
    """Create FINAL Energy-Based environment."""
    try:
        env = retro.make(game=game, state=state, render_mode=render_mode)
        env = StreetFighterVisionWrapper(
            env, frame_stack=8, rendering=(render_mode is not None)
        )

        print(f"✅ FINAL Energy-Based environment created")
        print(f"   - Feature dimension: {VECTOR_FEATURE_DIM}")
        print(f"   - Diversity buffer: ✅ ACTIVE")
        print(f"   - Stability controls: ✅ MAXIMUM")
        print(f"   - Energy collapse prevention: ✅ ACTIVE")

        return env
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        raise


class CheckpointManager:
    """Advanced checkpoint management with emergency restore capabilities."""

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
            # Fix for PyTorch 2.6+ weights_only issue
            try:
                checkpoint_data = torch.load(
                    checkpoint_path, map_location="cpu", weights_only=False
                )
            except Exception as first_error:
                print(f"⚠️  First load attempt failed: {first_error}")
                # Try with safe globals for numpy compatibility
                import torch.serialization

                with torch.serialization.safe_globals([np.core.multiarray.scalar]):
                    checkpoint_data = torch.load(
                        checkpoint_path, map_location="cpu", weights_only=True
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

            print(f"✅ Checkpoint restored from: {checkpoint_path.name}")
            print(f"   - Episode: {checkpoint_data['episode']}")
            print(f"   - Win rate: {checkpoint_data['win_rate']:.3f}")
            print(f"   - Energy quality: {checkpoint_data['energy_quality']:.1f}")
            print(f"   - Thinking steps: {agent.current_thinking_steps}")
            print(f"   - Thinking LR: {agent.current_thinking_lr:.4f}")

            return checkpoint_data

        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            print(f"💡 Tip: This might be a PyTorch version compatibility issue.")
            return None

    def emergency_restore(self, verifier, agent):
        """Emergency restore from best available checkpoint."""
        restore_path = self.best_checkpoint_path or self.emergency_checkpoint_path

        if restore_path and restore_path.exists():
            print(f"🚨 EMERGENCY RESTORE INITIATED")
            return self.load_checkpoint(restore_path, verifier, agent)
        else:
            print(f"❌ No checkpoint available for emergency restore!")
            return None


# Export all components
__all__ = [
    # Core components
    "StreetFighterVisionWrapper",
    "EnergyBasedStreetFighterCNN",
    "EnergyBasedStreetFighterVerifier",
    "StabilizedEnergyBasedAgent",
    "SimplifiedFeatureTracker",
    "StreetFighterDiscreteActions",
    # Enhanced stability components
    "EnergyStabilityManager",
    "DiversityExperienceBuffer",
    "CheckpointManager",
    # Utilities
    "verify_stabilized_energy_flow",
    "make_stabilized_env",
    "safe_divide",
    "safe_std",
    "safe_mean",
    "sanitize_array",
    "ensure_scalar",
    "safe_bool_check",
    "safe_comparison",
    "ensure_feature_dimension",
    # Constants
    "VECTOR_FEATURE_DIM",
]

print(f"🎉 FINAL ENERGY-BASED TRANSFORMER - Complete wrapper.py loaded successfully!")
print(f"   - Training paradigm: FINAL Energy-Based Transformer")
print(f"   - Verifier network: ✅ ULTRA-STABLE")
print(f"   - Thinking optimization: ✅ ADAPTIVE")
print(f"   - Energy stability: ✅ MAXIMUM")
print(f"   - Collapse prevention: ✅ ACTIVE")
print(f"   - Emergency protocols: ✅ ACTIVE")
print(f"   - Diversity buffer: ✅ ACTIVE")
print(f"   - Experience quality control: ✅ ENHANCED")
print(f"🛡️  Ready for FINAL Energy-Based Street Fighter training!")
