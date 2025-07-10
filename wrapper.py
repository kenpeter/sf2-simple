#!/usr/bin/env python3
"""
wrapper.py - ENERGY-BASED TRANSFORMER FOR STREET FIGHTER
CORE CONCEPT:
- Verifier-based action scoring (no policy network)
- Gradient-based "thinking" optimization
- Energy landscape learning for Street Fighter
- System 2 reasoning for strategic decisions
REPLACES: PPO entirely with Energy-Based approach
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

# Import the stabilized bait-punish system
try:
    from bait_punish_system import (
        SimpleBlockPunishDetector,
        integrate_bait_punish_system,
        AdaptiveRewardShaper,
    )

    BAIT_PUNISH_AVAILABLE = True
except ImportError:
    BAIT_PUNISH_AVAILABLE = False
    print("⚠️  Bait-punish system not available, running without it")

# --- FIX for TypeError in retro.make ---
_original_retro_make = retro.make


def _patched_retro_make(game, state=None, **kwargs):
    if not state:
        state = "ken_bison_12.state"
    return _original_retro_make(game=game, state=state, **kwargs)


retro.make = _patched_retro_make

# Configure logging
os.makedirs("logs", exist_ok=True)
log_filename = f'logs/energy_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
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

# Dynamic feature dimension based on bait-punish availability
BASE_VECTOR_FEATURE_DIM = 45
ENHANCED_VECTOR_FEATURE_DIM = 52
VECTOR_FEATURE_DIM = (
    ENHANCED_VECTOR_FEATURE_DIM if BAIT_PUNISH_AVAILABLE else BASE_VECTOR_FEATURE_DIM
)

print(f"🧠 ENERGY-BASED TRANSFORMER Configuration:")
print(f"   - Base features: {BASE_VECTOR_FEATURE_DIM}")
print(f"   - Enhanced features: {ENHANCED_VECTOR_FEATURE_DIM}")
print(
    f"   - Current mode: {VECTOR_FEATURE_DIM} ({'Enhanced' if BAIT_PUNISH_AVAILABLE else 'Base'})"
)
print(f"   - Training paradigm: Energy-Based (NOT PPO)")


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


class OscillationTracker:
    def __init__(self, history_length=16):
        self.history_length = history_length
        self.player_x_history = deque(maxlen=history_length)
        self.opponent_x_history = deque(maxlen=history_length)
        self.player_velocity_history = deque(maxlen=history_length)
        self.opponent_velocity_history = deque(maxlen=history_length)
        self.movement_threshold = 0.3
        self.direction_change_threshold = 0.1
        self.velocity_smoothing_factor = 0.3
        self.direction_change_timestamps = deque(maxlen=1800)
        self.player_direction_changes = 0
        self.opponent_direction_changes = 0
        self.player_oscillation_amplitude = 0.0
        self.opponent_oscillation_amplitude = 0.0
        self.space_control_score = 0.0
        self.aggressive_forward_count = 0
        self.defensive_backward_count = 0
        self.neutral_dance_count = 0
        self.CLOSE_RANGE = 45
        self.MID_RANGE = 80
        self.FAR_RANGE = 125
        self.prev_player_x = None
        self.prev_opponent_x = None
        self.prev_player_velocity = 0.0
        self.prev_opponent_velocity = 0.0
        self.frame_count = 0

    def update(
        self,
        player_x: float,
        opponent_x: float,
        player_attacking: bool = False,
        opponent_attacking: bool = False,
    ) -> Dict:
        self.frame_count += 1
        player_x = ensure_scalar(player_x, SCREEN_WIDTH / 2)
        opponent_x = ensure_scalar(opponent_x, SCREEN_WIDTH / 2)
        player_attacking = safe_bool_check(player_attacking)
        opponent_attacking = safe_bool_check(opponent_attacking)

        player_velocity, opponent_velocity = 0.0, 0.0

        if self.prev_player_x is not None:
            raw_velocity = player_x - ensure_scalar(self.prev_player_x, player_x)
            if np.isfinite(raw_velocity):
                player_velocity = (
                    self.velocity_smoothing_factor * raw_velocity
                    + (1 - self.velocity_smoothing_factor) * self.prev_player_velocity
                )
                if not np.isfinite(player_velocity):
                    player_velocity = 0.0

        if self.prev_opponent_x is not None:
            raw_velocity = opponent_x - ensure_scalar(self.prev_opponent_x, opponent_x)
            if np.isfinite(raw_velocity):
                opponent_velocity = (
                    self.velocity_smoothing_factor * raw_velocity
                    + (1 - self.velocity_smoothing_factor) * self.prev_opponent_velocity
                )
                if not np.isfinite(opponent_velocity):
                    opponent_velocity = 0.0

        self.player_x_history.append(player_x)
        self.opponent_x_history.append(opponent_x)
        self.player_velocity_history.append(player_velocity)
        self.opponent_velocity_history.append(opponent_velocity)

        distance = abs(player_x - opponent_x)
        if not np.isfinite(distance):
            distance = 80.0

        movement_analysis = self._analyze_movement_patterns(
            player_x,
            opponent_x,
            player_velocity,
            opponent_velocity,
            distance,
            player_attacking,
            opponent_attacking,
        )

        self.prev_player_x, self.prev_opponent_x = player_x, opponent_x
        self.prev_player_velocity, self.prev_opponent_velocity = (
            player_velocity,
            opponent_velocity,
        )

        return movement_analysis

    def _analyze_movement_patterns(
        self,
        player_x,
        opponent_x,
        player_velocity,
        opponent_velocity,
        distance,
        player_attacking,
        opponent_attacking,
    ) -> Dict:
        player_moving_forward = (
            safe_comparison(player_x, opponent_x, "<") and player_velocity > 0.5
        ) or (safe_comparison(player_x, opponent_x, ">") and player_velocity < -0.5)

        player_moving_backward = (
            safe_comparison(player_x, opponent_x, "<") and player_velocity < -0.5
        ) or (safe_comparison(player_x, opponent_x, ">") and player_velocity > 0.5)

        self.space_control_score = self._calculate_space_control(
            player_x, opponent_x, player_velocity, opponent_velocity, distance
        )

        return {
            "player_moving_forward": player_moving_forward,
            "player_moving_backward": player_moving_backward,
            "distance": distance,
            "space_control_score": self.space_control_score,
        }

    def _calculate_space_control(
        self, player_x, opponent_x, player_velocity, opponent_velocity, distance
    ) -> float:
        screen_center = SCREEN_WIDTH / 2
        player_center_dist = abs(player_x - screen_center)
        opponent_center_dist = abs(opponent_x - screen_center)
        center_control = safe_divide(
            opponent_center_dist - player_center_dist, SCREEN_WIDTH / 2, 0.0
        )

        total_control = center_control * 0.5
        return np.clip(total_control, -1.0, 1.0) if np.isfinite(total_control) else 0.0

    def get_oscillation_features(self) -> np.ndarray:
        """Return exactly 12 features for compatibility."""
        features = np.zeros(12, dtype=np.float32)
        features[0] = np.clip(self.space_control_score, -1.0, 1.0)
        return features.astype(np.float32)

    def get_stats(self) -> Dict:
        return {
            "space_control_score": self.space_control_score,
            "frame_count": self.frame_count,
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


class StrategicFeatureTracker:
    def __init__(self, history_length=8):
        self.history_length = history_length
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
        self.oscillation_tracker = OscillationTracker(history_length=16)
        self.close_combat_count = 0
        self.total_frames = 0
        self.feature_rolling_mean = None
        self.feature_rolling_std = None
        self.normalization_alpha = 0.999
        self.feature_nan_count = 0
        self.DANGER_ZONE_HEALTH = MAX_HEALTH * 0.25
        self.CORNER_THRESHOLD = 53
        self.CLOSE_DISTANCE = 71
        self.OPTIMAL_SPACING_MIN = 62
        self.OPTIMAL_SPACING_MAX = 98
        self.COMBO_TIMEOUT_FRAMES = 60
        self.MIN_SCORE_INCREASE_FOR_HIT = 50
        self.prev_player_health = None
        self.prev_opponent_health = None
        self.prev_score = None
        self._bait_punish_integrated = False
        self.current_feature_dim = VECTOR_FEATURE_DIM

        print(
            f"🔧 StrategicFeatureTracker initialized with {self.current_feature_dim} features"
        )

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features with consistent dimension handling."""
        features = ensure_feature_dimension(features, self.current_feature_dim)

        nan_mask = ~np.isfinite(features)
        if np.any(nan_mask):
            self.feature_nan_count += np.sum(nan_mask)
            features = sanitize_array(features, 0.0)

        if self.feature_rolling_mean is None:
            self.feature_rolling_mean = np.zeros(
                self.current_feature_dim, dtype=np.float32
            )
            self.feature_rolling_std = np.ones(
                self.current_feature_dim, dtype=np.float32
            )

        if len(self.feature_rolling_mean) != self.current_feature_dim:
            self.feature_rolling_mean = np.zeros(
                self.current_feature_dim, dtype=np.float32
            )
            self.feature_rolling_std = np.ones(
                self.current_feature_dim, dtype=np.float32
            )

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
            if isinstance(features, np.ndarray) and isinstance(
                self.feature_rolling_mean, np.ndarray
            ):
                normalized = (features - self.feature_rolling_mean) / safe_std
                normalized = np.where(np.isfinite(normalized), normalized, 0.0)
            else:
                normalized = safe_divide(
                    features - self.feature_rolling_mean, safe_std, 0.0
                )
                if not isinstance(normalized, np.ndarray):
                    normalized = np.array([normalized], dtype=np.float32)
        except Exception as e:
            print(f"⚠️  Normalization division error: {e}, using zeros")
            normalized = np.zeros(self.current_feature_dim, dtype=np.float32)

        if isinstance(normalized, np.ndarray):
            normalized = np.clip(normalized, -3.0, 3.0)
        else:
            normalized = np.clip(normalized, -3.0, 3.0)

        normalized = sanitize_array(normalized, 0.0)
        normalized = ensure_feature_dimension(normalized, self.current_feature_dim)
        return normalized.astype(np.float32)

    def update(self, info: Dict, button_features: np.ndarray) -> np.ndarray:
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

            try:
                attack_buttons = button_features[[0, 1, 8, 9, 10, 11]]
                player_attacking = safe_bool_check(np.any(attack_buttons > 0.5))
            except:
                player_attacking = False

            oscillation_analysis = self.oscillation_tracker.update(
                player_x, opponent_x, player_attacking
            )

            self.total_frames += 1
            distance = abs(player_x - opponent_x)
            if np.isfinite(distance) and safe_comparison(
                distance, self.CLOSE_DISTANCE, "<="
            ):
                self.close_combat_count += 1

            features = self._calculate_enhanced_features(
                info, distance, oscillation_analysis
            )
            features = ensure_feature_dimension(features, self.current_feature_dim)
            features = self._normalize_features(features)
            features = ensure_feature_dimension(features, self.current_feature_dim)

            self.prev_player_health = player_health
            self.prev_opponent_health = opponent_health
            self.prev_score = score

            # Bait-punish integration
            if BAIT_PUNISH_AVAILABLE and not self._bait_punish_integrated:
                try:
                    integrate_bait_punish_system(self)
                    self._bait_punish_integrated = True
                    self.current_feature_dim = ENHANCED_VECTOR_FEATURE_DIM
                    self.feature_rolling_mean = None
                    self.feature_rolling_std = None
                    print(
                        f"✅ Bait-Punish system integrated, features expanded to {self.current_feature_dim}"
                    )
                except Exception as e:
                    print(
                        f"⚠️  Bait-Punish integration failed: {e}, continuing without it"
                    )
                    globals()["BAIT_PUNISH_AVAILABLE"] = False

            return features

        except Exception as e:
            print(f"⚠️  Critical error in StrategicFeatureTracker.update(): {e}")
            import traceback

            traceback.print_exc()
            return np.zeros(self.current_feature_dim, dtype=np.float32)

    def _calculate_enhanced_features(
        self, info, distance, oscillation_analysis
    ) -> np.ndarray:
        """Calculate features with dimension awareness."""
        features = np.zeros(BASE_VECTOR_FEATURE_DIM, dtype=np.float32)

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

            # Calculate base features (0-44)
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

            features[3] = (
                self._calculate_momentum(self.player_health_history)
                + self._calculate_momentum(self.opponent_health_history)
            ) / 2.0
            features[4] = self._calculate_momentum(self.player_damage_dealt_history)
            features[5] = self._calculate_momentum(self.opponent_damage_dealt_history)

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

            player_y = ensure_scalar(
                info.get("agent_y", SCREEN_HEIGHT / 2), SCREEN_HEIGHT / 2
            )
            opponent_y = ensure_scalar(
                info.get("enemy_y", SCREEN_HEIGHT / 2), SCREEN_HEIGHT / 2
            )

            y_diff = (player_y - opponent_y) / (SCREEN_HEIGHT / 2)
            features[11] = np.clip(y_diff, -1.0, 1.0) if np.isfinite(y_diff) else 0.0

            space_control = oscillation_analysis.get("space_control_score", 0.0)
            features[12] = space_control if np.isfinite(space_control) else 0.0

            optimal_spacing = safe_comparison(
                distance, self.OPTIMAL_SPACING_MIN, ">="
            ) and safe_comparison(distance, self.OPTIMAL_SPACING_MAX, "<=")
            features[13] = 1.0 if optimal_spacing else 0.0

            player_forward = oscillation_analysis.get("player_moving_forward", False)
            player_backward = oscillation_analysis.get("player_moving_backward", False)
            features[14] = 1.0 if player_forward else (-1.0 if player_backward else 0.0)
            features[15] = 1.0 if player_backward else 0.0

            features[16] = safe_divide(
                self.close_combat_count, max(1, self.total_frames), 0.0
            )
            features[17] = self._calculate_enhanced_score_momentum()

            agent_status = ensure_scalar(info.get("agent_status", 0), 0)
            enemy_status = ensure_scalar(info.get("enemy_status", 0), 0)

            status_diff = (agent_status - enemy_status) / 100.0
            features[18] = np.clip(status_diff, -1.0, 1.0)

            agent_victories = ensure_scalar(info.get("agent_victories", 0), 0)
            enemy_victories = ensure_scalar(info.get("enemy_victories", 0), 0)
            features[19] = (
                min(agent_victories / 10.0, 1.0)
                if np.isfinite(agent_victories)
                else 0.0
            )
            features[20] = (
                min(enemy_victories / 10.0, 1.0)
                if np.isfinite(enemy_victories)
                else 0.0
            )

            # Oscillation features (21-32)
            try:
                oscillation_features = (
                    self.oscillation_tracker.get_oscillation_features()
                )
                if (
                    isinstance(oscillation_features, np.ndarray)
                    and len(oscillation_features) == 12
                ):
                    features[21:33] = oscillation_features
                else:
                    features[21:33] = np.zeros(12, dtype=np.float32)
            except Exception as e:
                features[21:33] = np.zeros(12, dtype=np.float32)

            # Button features (33-44)
            try:
                if len(self.button_features_history) > 0:
                    button_hist = self.button_features_history[-1]
                    if isinstance(button_hist, np.ndarray) and len(button_hist) == 12:
                        features[33:45] = button_hist
                    else:
                        features[33:45] = np.zeros(12, dtype=np.float32)
                else:
                    features[33:45] = np.zeros(12, dtype=np.float32)
            except Exception as e:
                features[33:45] = np.zeros(12, dtype=np.float32)

            features = sanitize_array(features, 0.0)

            # Handle bait-punish features expansion properly
            if (
                BAIT_PUNISH_AVAILABLE
                and self._bait_punish_integrated
                and self.current_feature_dim == ENHANCED_VECTOR_FEATURE_DIM
            ):
                enhanced_features = np.zeros(
                    ENHANCED_VECTOR_FEATURE_DIM, dtype=np.float32
                )
                enhanced_features[:BASE_VECTOR_FEATURE_DIM] = features

                try:
                    bait_punish_features = getattr(
                        self, "last_bait_punish_features", np.zeros(7, dtype=np.float32)
                    )
                    if (
                        isinstance(bait_punish_features, np.ndarray)
                        and len(bait_punish_features) == 7
                    ):
                        enhanced_features[45:52] = bait_punish_features
                    else:
                        enhanced_features[45:52] = np.zeros(7, dtype=np.float32)
                except Exception as e:
                    enhanced_features[45:52] = np.zeros(7, dtype=np.float32)

                return enhanced_features
            else:
                return features

        except Exception as e:
            print(f"⚠️  Error in _calculate_enhanced_features: {e}")
            return np.zeros(self.current_feature_dim, dtype=np.float32)

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
        combo_stats = {
            "current_combo": self.combo_counter,
            "max_combo_this_round": self.max_combo_this_round,
        }
        combo_stats.update(self.oscillation_tracker.get_stats())
        return combo_stats


class EnergyBasedStreetFighterCNN(nn.Module):
    """
    CNN feature extractor for Energy-Based Transformer.
    Extracts visual and vector features for the energy verifier.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__()
        visual_space = observation_space["visual_obs"]
        vector_space = observation_space["vector_obs"]
        n_input_channels = visual_space.shape[0]
        seq_length, vector_feature_count = vector_space.shape

        print(f"🔧 Energy-Based CNN Feature Extractor Configuration:")
        print(f"   - Visual channels: {n_input_channels}")
        print(f"   - Visual size: {visual_space.shape[1]}x{visual_space.shape[2]}")
        print(f"   - Vector sequence: {seq_length} x {vector_feature_count}")
        print(f"   - Output features: {features_dim}")

        # Conservative CNN architecture for energy stability
        self.visual_cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.1),
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

        with torch.no_grad():
            dummy_visual = torch.zeros(
                1, n_input_channels, visual_space.shape[1], visual_space.shape[2]
            )
            visual_output_size = self.visual_cnn(dummy_visual).shape[1]

        self.vector_embed = nn.Linear(vector_feature_count, 64)
        self.vector_norm = nn.LayerNorm(64)
        self.vector_dropout = nn.Dropout(0.2)
        self.vector_gru = nn.GRU(64, 64, batch_first=True, dropout=0.1)
        self.vector_final = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        fusion_input_size = visual_output_size + 32
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, 512),
            nn.ReLU(inplace=True),
            nn.LayerNorm(512),
            nn.Dropout(0.2),
            nn.Linear(512, features_dim),
            nn.ReLU(inplace=True),
        )

        # Conservative weight initialization
        self.apply(self._init_weights_conservative)
        print(f"   - Visual output size: {visual_output_size}")
        print(f"   - Fusion input size: {fusion_input_size}")
        print(f"   ✅ Energy-Based Feature Extractor initialized")

    def _init_weights_conservative(self, m):
        """Conservative weight initialization for energy stability."""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if hasattr(m.weight, "data"):
                m.weight.data *= 0.3  # Reduced scale for stability
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.3)  # Reduced gain for stability
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param, gain=0.3)
                elif "bias" in name:
                    nn.init.constant_(param, 0)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        visual_obs = observations["visual_obs"]
        vector_obs = observations["vector_obs"]
        device = next(self.parameters()).device

        visual_obs = visual_obs.float().to(device)
        vector_obs = vector_obs.float().to(device)

        # NaN safety
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

        # Clamp inputs for stability
        visual_obs = torch.clamp(visual_obs / 255.0, 0.0, 1.0)
        visual_features = self.visual_cnn(visual_obs)

        # Check for NaN in visual features
        if torch.any(~torch.isfinite(visual_features)):
            visual_features = torch.where(
                ~torch.isfinite(visual_features),
                torch.zeros_like(visual_features),
                visual_features,
            )

        batch_size, seq_len, feature_dim = vector_obs.shape
        vector_embedded = self.vector_embed(vector_obs)
        vector_embedded = self.vector_norm(vector_embedded)
        vector_embedded = self.vector_dropout(vector_embedded)

        # Check for NaN in vector embedding
        if torch.any(~torch.isfinite(vector_embedded)):
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
            vector_features = torch.where(
                ~torch.isfinite(vector_features),
                torch.zeros_like(vector_features),
                vector_features,
            )

        combined_features = torch.cat([visual_features, vector_features], dim=1)
        output = self.fusion(combined_features)

        # Final safety checks
        if torch.any(~torch.isfinite(output)):
            output = torch.where(
                ~torch.isfinite(output), torch.zeros_like(output), output
            )

        return output


class EnergyBasedStreetFighterVerifier(nn.Module):
    """
    Energy-Based Transformer Verifier for Street Fighter.
    This is the core of the EBT approach - it learns to score context-action pairs.
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

        # Feature extractor for context (state)
        self.features_extractor = EnergyBasedStreetFighterCNN(
            observation_space, features_dim
        )

        # Action embedding
        self.action_embed = nn.Sequential(
            nn.Linear(self.action_dim, 64),
            nn.ReLU(inplace=True),
            nn.LayerNorm(64),
            nn.Dropout(0.1),
        )

        # Energy network - this is the core verifier
        self.energy_net = nn.Sequential(
            nn.Linear(features_dim + 64, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.LayerNorm(128),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.LayerNorm(64),
            nn.Linear(64, 1),  # Single energy score output
        )

        # Energy scaling for stability
        self.energy_scale = 0.1
        self.energy_clamp_min = -5.0
        self.energy_clamp_max = 5.0

        # Conservative initialization
        self.apply(self._init_weights_conservative)

        print(f"✅ EnergyBasedStreetFighterVerifier initialized")
        print(f"   - Features dim: {features_dim}")
        print(f"   - Action dim: {self.action_dim}")
        print(f"   - Energy scale: {self.energy_scale}")

    def _init_weights_conservative(self, m):
        """Conservative weight initialization for energy stability."""
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.1)  # Very small gain
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(
        self, context: torch.Tensor, candidate_action: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate energy score for context-action pair.
        This is the core EBT verifier function.

        Args:
            context: Processed observations (context)
            candidate_action: One-hot encoded action (candidate)

        Returns:
            energy: Energy score (lower = better compatibility)
        """
        device = next(self.parameters()).device

        # Ensure inputs are on correct device and finite
        if isinstance(context, dict):
            context_features = self.features_extractor(context)
        else:
            context_features = context

        context_features = context_features.to(device)
        candidate_action = candidate_action.to(device)

        # Safety checks
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

        # Embed action
        action_embedded = self.action_embed(candidate_action)

        # Check for NaN in action embedding
        if torch.any(~torch.isfinite(action_embedded)):
            action_embedded = torch.where(
                ~torch.isfinite(action_embedded),
                torch.zeros_like(action_embedded),
                action_embedded,
            )

        # Combine context and action
        combined_input = torch.cat([context_features, action_embedded], dim=-1)

        # Calculate energy
        energy = self.energy_net(combined_input)

        # Scale and clamp energy for stability
        energy = energy * self.energy_scale
        energy = torch.clamp(energy, self.energy_clamp_min, self.energy_clamp_max)

        # Final safety check
        if torch.any(~torch.isfinite(energy)):
            energy = torch.where(
                ~torch.isfinite(energy), torch.zeros_like(energy), energy
            )

        return energy


class EnergyBasedAgent:
    """
    Energy-Based Agent for Street Fighter.
    Uses iterative optimization to find actions that minimize energy.
    """

    def __init__(
        self,
        verifier: EnergyBasedStreetFighterVerifier,
        thinking_steps: int = 5,
        thinking_lr: float = 0.1,
        noise_scale: float = 0.1,
    ):

        self.verifier = verifier
        self.thinking_steps = thinking_steps
        self.thinking_lr = thinking_lr
        self.noise_scale = noise_scale
        self.action_dim = verifier.action_dim

        # Thinking process parameters
        self.gradient_clip = 1.0
        self.early_stop_patience = 3
        self.min_energy_improvement = 1e-4

        # Statistics
        self.thinking_stats = {
            "total_predictions": 0,
            "avg_thinking_steps": 0.0,
            "avg_energy_improvement": 0.0,
            "early_stops": 0,
            "energy_explosions": 0,
        }

        print(f"✅ EnergyBasedAgent initialized")
        print(f"   - Thinking steps: {thinking_steps}")
        print(f"   - Learning rate: {thinking_lr}")
        print(f"   - Noise scale: {noise_scale}")

    def predict(
        self, observations: Dict[str, torch.Tensor], deterministic: bool = False
    ) -> Tuple[int, Dict]:
        """
        Predict action using energy-based thinking.
        This implements the core EBT "thinking" process.

        Args:
            observations: Environment observations
            deterministic: Whether to use deterministic prediction

        Returns:
            action: Predicted action index
            info: Information about the thinking process
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
        if len(obs_device["visual_obs"].shape) == 3:  # Single observation
            for key in obs_device:
                obs_device[key] = obs_device[key].unsqueeze(0)

        batch_size = obs_device["visual_obs"].shape[0]

        # Initialize candidate action randomly (this is the EBT starting point)
        if deterministic:
            # For deterministic prediction, start with uniform distribution
            candidate_action = (
                torch.ones(batch_size, self.action_dim, device=device) / self.action_dim
            )
        else:
            # Start with random action + small noise
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

        # Initial energy
        with torch.no_grad():
            initial_energy = self.verifier(obs_device, candidate_action)
            energy_history.append(initial_energy.mean().item())

        # EBT THINKING LOOP - This is the core of the approach!
        for step in range(self.thinking_steps):
            # Calculate current energy
            energy = self.verifier(obs_device, candidate_action)

            # Check for energy explosion
            if torch.any(torch.abs(energy) > 20.0):
                energy_explosion = True
                print(f"⚠️ Energy explosion at step {step}")
                break

            # Calculate gradient of energy w.r.t. candidate action
            try:
                gradients = torch.autograd.grad(
                    outputs=energy.sum(),
                    inputs=candidate_action,
                    create_graph=False,  # No need for higher-order gradients in inference
                    retain_graph=False,
                )[0]

                # Clip gradients for stability
                gradient_norm = torch.norm(gradients)
                if gradient_norm > self.gradient_clip:
                    gradients = gradients * (self.gradient_clip / gradient_norm)

                # Update candidate action (gradient descent on energy)
                with torch.no_grad():
                    candidate_action = candidate_action - self.thinking_lr * gradients
                    candidate_action = F.softmax(
                        candidate_action, dim=-1
                    )  # Ensure valid probabilities
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
                print(f"⚠️ Thinking gradient calculation failed at step {step}: {e}")
                break

        # Final action selection
        with torch.no_grad():
            final_action_probs = F.softmax(candidate_action, dim=-1)
            if deterministic:
                action_idx = torch.argmax(final_action_probs, dim=-1)
            else:
                action_idx = torch.multinomial(final_action_probs, 1).squeeze(-1)

        # Update statistics
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

        # Information about thinking process
        thinking_info = {
            "energy_history": energy_history,
            "steps_taken": steps_taken,
            "early_stopped": early_stopped,
            "energy_explosion": energy_explosion,
            "energy_improvement": (
                energy_history[0] - energy_history[-1]
                if len(energy_history) > 1
                else 0.0
            ),
            "final_energy": energy_history[-1] if energy_history else 0.0,
        }

        return action_idx.item() if batch_size == 1 else action_idx, thinking_info

    def get_thinking_stats(self) -> Dict:
        """Get statistics about the thinking process."""
        return self.thinking_stats.copy()


class StreetFighterVisionWrapper(gym.Wrapper):
    """
    Street Fighter environment wrapper for Energy-Based Transformer.
    Handles observation processing and reward calculation.
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

        print(f"🖼️  Using frames: {self.target_size[0]}x{self.target_size[1]} (H x W)")

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

        print(f"🔧 Energy-Based Observation space configured:")
        print(f"   - Visual: {self.observation_space['visual_obs'].shape}")
        print(f"   - Vector: {self.observation_space['vector_obs'].shape}")
        print(f"   - Vector features: {VECTOR_FEATURE_DIM}")

        self.frame_buffer = deque(maxlen=frame_stack)
        self.vector_features_history = deque(maxlen=frame_stack)
        self.strategic_tracker = StrategicFeatureTracker(history_length=frame_stack)
        self.full_hp = 176
        self.prev_player_health = self.full_hp
        self.prev_opponent_health = self.full_hp
        self.wins, self.losses, self.total_rounds = 0, 0, 0
        self.total_damage_dealt, self.total_damage_received = 0, 0

        # Initialize bait-punish system if available
        if BAIT_PUNISH_AVAILABLE:
            self.reward_shaper = AdaptiveRewardShaper()
            print("✅ Adaptive reward shaper initialized for energy stability")
        else:
            self.reward_shaper = None

        # Energy-based reward configuration
        self.reward_scale = 0.05
        self.episode_steps = 0
        self.max_episode_steps = 18000
        self.episode_rewards = deque(maxlen=100)
        self.stats = {}

        # Reward normalization
        self.reward_history = deque(maxlen=1000)
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_alpha = 0.99

        print(f"🛡️  Energy-based wrapper initialized")

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
        self.prev_player_health = self.full_hp
        self.prev_opponent_health = self.full_hp
        self.episode_steps = 0

        processed_frame = self._preprocess_frame(obs)
        initial_vector_features = self._create_initial_vector_features(info)

        self.frame_buffer.clear()
        self.vector_features_history.clear()

        for _ in range(self.frame_stack):
            self.frame_buffer.append(processed_frame)
            self.vector_features_history.append(initial_vector_features.copy())

        self.strategic_tracker = StrategicFeatureTracker(
            history_length=self.frame_stack
        )

        return self._get_observation(), info

    def step(self, discrete_action):
        self.episode_steps += 1

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

        # Calculate energy-stable reward
        base_reward, custom_done = self._calculate_energy_stable_reward(
            curr_player_health, curr_opponent_health
        )

        # Add bait-punish reward if available
        if self.reward_shaper is not None:
            bait_punish_info = getattr(
                self.strategic_tracker, "last_bait_punish_info", {}
            )
            try:
                final_reward = self.reward_shaper.shape_reward(
                    base_reward, bait_punish_info, sanitized_info
                )
            except Exception as e:
                print(f"⚠️  Reward shaping error: {e}, using base reward")
                final_reward = base_reward
        else:
            final_reward = base_reward

        # Normalize reward for energy stability
        final_reward = self._normalize_reward_for_energy(final_reward)

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
        self._update_enhanced_stats()

        # Add bait-punish stats if available
        if hasattr(self.strategic_tracker, "bait_punish_detector"):
            try:
                bait_punish_stats = (
                    self.strategic_tracker.bait_punish_detector.get_learning_stats()
                )
                sanitized_info.update(bait_punish_stats)
            except Exception as e:
                print(f"⚠️  Error getting bait-punish stats: {e}")

        if self.reward_shaper is not None:
            try:
                adaptation_stats = self.reward_shaper.get_adaptation_stats()
                sanitized_info.update(adaptation_stats)
            except Exception as e:
                print(f"⚠️  Error getting adaptation stats: {e}")

        sanitized_info.update(self.stats)

        return self._get_observation(), final_reward, done, truncated, sanitized_info

    def _calculate_energy_stable_reward(self, curr_player_health, curr_opponent_health):
        """Calculate energy-stable reward for EBT training."""
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
                    0.5 + safe_divide(curr_player_health, self.full_hp, 0.0) * 0.2
                )
                reward += win_bonus
                print(
                    f"🏆 AI WON! Total: {self.wins}W/{self.losses}L (Round {self.total_rounds})"
                )
            else:
                self.losses += 1
                reward -= 0.2
                print(
                    f"💀 AI LOST! Total: {self.wins}W/{self.losses}L (Round {self.total_rounds})"
                )
            done = True
            combo_bonus = self.strategic_tracker.combo_counter * 0.01
            reward += combo_bonus

        # Damage calculation
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

        # Smaller damage rewards for energy stability
        reward += (damage_dealt * 0.01) - (damage_received * 0.005)
        self.total_damage_dealt += damage_dealt
        self.total_damage_received += damage_received

        # Strategic bonuses (very small)
        try:
            osc_tracker = self.strategic_tracker.oscillation_tracker
            space_control = osc_tracker.space_control_score
            if np.isfinite(space_control) and safe_comparison(space_control, 0, ">"):
                reward += space_control * 0.001
        except Exception:
            pass

        # Small time penalty
        reward -= 0.0001

        # Apply reward scale for energy stability
        reward *= self.reward_scale

        # Hard clip to prevent energy explosion
        reward = np.clip(reward, -0.5, 0.5) if np.isfinite(reward) else 0.0

        self.prev_player_health, self.prev_opponent_health = (
            curr_player_health,
            curr_opponent_health,
        )

        if done:
            self.episode_rewards.append(reward)

        return reward, done

    def _normalize_reward_for_energy(self, reward):
        """Normalize reward specifically for energy stability."""
        if not np.isfinite(reward):
            reward = 0.0

        reward = np.clip(reward, -1.0, 1.0)
        self.reward_history.append(reward)

        # Update running statistics
        if len(self.reward_history) > 10:
            current_mean = np.mean(list(self.reward_history))
            current_std = np.std(list(self.reward_history))

            self.reward_mean = (
                self.reward_alpha * self.reward_mean
                + (1 - self.reward_alpha) * current_mean
            )
            self.reward_std = (
                self.reward_alpha * self.reward_std
                + (1 - self.reward_alpha) * current_std
            )

            self.reward_std = max(self.reward_std, 0.1)

        # Light normalization
        if self.reward_std > 0:
            normalized_reward = (reward - self.reward_mean) / self.reward_std
        else:
            normalized_reward = reward

        # Final clipping for energy stability
        normalized_reward = np.clip(normalized_reward, -2.0, 2.0)

        # Final scale down
        return normalized_reward * 0.1

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
                    "space_control_score": combo_stats.get("space_control_score", 0.0),
                    "episode_steps": self.episode_steps,
                    "current_feature_dim": self.strategic_tracker.current_feature_dim,
                    "reward_mean": self.reward_mean,
                    "reward_std": self.reward_std,
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


def verify_energy_flow(verifier, env, device=None):
    """Verify energy flow and gradient computation for EBT."""
    print("\n🔬 Energy-Based Transformer Verification")
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
    print(
        f"   - Features: {vector_obs.shape[-1]} ({'Enhanced' if vector_obs.shape[-1] == 52 else 'Base'})"
    )
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
        print(f"✅ Energy calculation successful")
        print(f"   - Energy output: {energy.item():.6f}")
        print(f"   - Energy NaN count: {torch.sum(~torch.isfinite(energy)).item()}")

        if torch.sum(~torch.isfinite(energy)) > 0:
            print("   🚨 CRITICAL: NaN values in energy output!")
            return False

        if abs(energy.item()) > 10.0:
            print(f"   🚨 CRITICAL: Energy value too large: {energy.item():.3f}")
            return False
        else:
            print("   ✅ Energy output is stable")

    except Exception as e:
        print(f"❌ Energy calculation failed: {e}")
        return False

    # Test gradient computation (core of EBT)
    try:
        gradients = torch.autograd.grad(
            outputs=energy.sum(),
            inputs=random_action,
            create_graph=False,
            retain_graph=False,
        )[0]

        print("✅ Gradient computation successful")
        print(f"   - Gradient shape: {gradients.shape}")
        print(f"   - Gradient norm: {torch.norm(gradients).item():.6f}")
        print(
            f"   - Gradient NaN count: {torch.sum(~torch.isfinite(gradients)).item()}"
        )

        if torch.sum(~torch.isfinite(gradients)) > 0:
            print("   🚨 CRITICAL: NaN values in gradients!")
            return False

        gradient_norm = torch.norm(gradients).item()
        if gradient_norm > 10.0:
            print(f"   🚨 CRITICAL: Gradient explosion: {gradient_norm:.3f}")
            return False
        elif gradient_norm > 5.0:
            print(f"   ⚠️  WARNING: Large gradients: {gradient_norm:.3f}")
        else:
            print("   ✅ Gradients are stable")

    except Exception as e:
        print(f"❌ Gradient computation failed: {e}")
        return False

    print("✅ EXCELLENT: Energy-Based Transformer verification successful!")
    return True


def make_env(
    game="StreetFighterIISpecialChampionEdition-Genesis",
    state="ken_bison_12.state",
    render_mode=None,
):
    """Create Energy-Based environment."""
    try:
        env = retro.make(game=game, state=state, render_mode=render_mode)
        env = StreetFighterVisionWrapper(
            env, frame_stack=8, rendering=(render_mode is not None)
        )

        print(f"✅ Energy-Based environment created")
        print(f"   - Feature dimension: {VECTOR_FEATURE_DIM}")
        print(
            f"   - Bait-punish: {'Available' if BAIT_PUNISH_AVAILABLE else 'Not available'}"
        )

        return env
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        raise


# Export components
__all__ = [
    "StreetFighterVisionWrapper",
    "EnergyBasedStreetFighterCNN",
    "EnergyBasedStreetFighterVerifier",
    "EnergyBasedAgent",
    "verify_energy_flow",
    "OscillationTracker",
    "StrategicFeatureTracker",
    "StreetFighterDiscreteActions",
    "safe_divide",
    "safe_std",
    "safe_mean",
    "sanitize_array",
    "ensure_scalar",
    "safe_bool_check",
    "safe_comparison",
    "ensure_feature_dimension",
    "make_env",
    "VECTOR_FEATURE_DIM",
    "BASE_VECTOR_FEATURE_DIM",
    "ENHANCED_VECTOR_FEATURE_DIM",
    "BAIT_PUNISH_AVAILABLE",
]

if BAIT_PUNISH_AVAILABLE:
    __all__.append("AdaptiveRewardShaper")

print(f"🎉 ENERGY-BASED TRANSFORMER - Complete wrapper.py loaded successfully!")
print(f"   - Training paradigm: Energy-Based Transformer (NOT PPO)")
print(f"   - Verifier network: ✅ ACTIVE")
print(f"   - Thinking optimization: ✅ ACTIVE")
print(f"   - Energy stability: ✅ ACTIVE")
print(
    f"   - Bait-punish integration: ✅ {'ACTIVE' if BAIT_PUNISH_AVAILABLE else 'STANDBY'}"
)
print(f"   - Ready for Energy-Based Street Fighter training!")
