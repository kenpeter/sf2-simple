#!/usr/bin/env python3
"""
wrapper_fixed.py - COMPLETE ARRAY AMBIGUITY FIX + DIMENSION CONSISTENCY
FIXES: All remaining boolean array operations that cause "truth value of an array" errors
SOLUTION: Comprehensive array-to-scalar conversion at every potential boolean operation point
NEW FIX: Consistent feature dimension handling (45 base, optionally 52 with bait-punish)
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import deque, defaultdict
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Dict, Tuple, List, Type, Any, Optional, Union
import math
import logging
import os
from datetime import datetime
import retro

# Import the bait-punish system (note: you'll need to create this file)
try:
    # Use the fixed version
    from bait_punish_system import integrate_bait_punish_system, AdaptiveRewardShaper

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
# --- END OF FIX ---

# Configure logging
os.makedirs("logs", exist_ok=True)
log_filename = f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
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

# CRITICAL FIX: Dynamic feature dimension based on bait-punish availability
BASE_VECTOR_FEATURE_DIM = 45  # Base features without bait-punish
ENHANCED_VECTOR_FEATURE_DIM = 52  # Enhanced features with bait-punish
VECTOR_FEATURE_DIM = (
    ENHANCED_VECTOR_FEATURE_DIM if BAIT_PUNISH_AVAILABLE else BASE_VECTOR_FEATURE_DIM
)

print(f"🔧 Feature Dimension Configuration:")
print(f"   - Base features: {BASE_VECTOR_FEATURE_DIM}")
print(f"   - Enhanced features: {ENHANCED_VECTOR_FEATURE_DIM}")
print(
    f"   - Current mode: {VECTOR_FEATURE_DIM} ({'Enhanced' if BAIT_PUNISH_AVAILABLE else 'Base'})"
)


# CRITICAL: Enhanced safe operations with more aggressive scalar conversion
def safe_divide(numerator, denominator, default=0.0):
    """Safe division that prevents NaN and handles edge cases."""
    # CRITICAL FIX: Ensure both inputs are scalars
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
    # CRITICAL FIX: Handle scalar inputs
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

    # Handle 0-dimensional arrays
    if arr.ndim == 0:
        val = arr.item()
        if np.isfinite(val):
            return np.array([val], dtype=np.float32)
        else:
            return np.array([default_val], dtype=np.float32)

    # Handle regular arrays
    mask = ~np.isfinite(arr)
    if np.any(mask):
        arr = arr.copy()
        arr[mask] = default_val

    return arr.astype(np.float32)


def ensure_scalar(value, default=0.0):
    """CRITICAL FIX: Ensure value is a scalar, handling arrays properly."""
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
            # For multi-element arrays, take the first element
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
    """CRITICAL FIX: Safe boolean check for arrays."""
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
            # For multi-element arrays, use any() for existence check
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
    """CRITICAL FIX: Safe comparison that handles arrays."""
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
    """CRITICAL NEW FIX: Ensure features match target dimension exactly."""
    if not isinstance(features, np.ndarray):
        if isinstance(features, (int, float)):
            features = np.array([features], dtype=np.float32)
        else:
            print(f"⚠️  Features is not array-like: {type(features)}, creating zeros")
            return np.zeros(target_dim, dtype=np.float32)

    # Handle 0-dimensional arrays
    if features.ndim == 0:
        features = np.array([features.item()], dtype=np.float32)

    # Get current length safely
    try:
        current_length = len(features)
    except TypeError:
        print(f"⚠️  Cannot get length of features: {type(features)}, using zeros")
        return np.zeros(target_dim, dtype=np.float32)

    # Adjust to target dimension
    if current_length == target_dim:
        return features.astype(np.float32)
    elif current_length < target_dim:
        # Pad with zeros
        padding = np.zeros(target_dim - current_length, dtype=np.float32)
        return np.concatenate([features, padding]).astype(np.float32)
    else:
        # Truncate
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
        self.optimal_range_violations = 0
        self.whiff_bait_attempts = 0
        self.successful_whiff_punishes = 0
        self.neutral_game_duration = 0
        self.advantage_transitions = 0
        self.space_control_score = 0.0
        self.aggressive_forward_count = 0
        self.defensive_backward_count = 0
        self.neutral_dance_count = 0
        self.CLOSE_RANGE = 45
        self.MID_RANGE = 80
        self.FAR_RANGE = 125
        self.WHIFF_BAIT_RANGE = 62
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

        # CRITICAL FIX: Ensure scalar values
        player_x = ensure_scalar(player_x, SCREEN_WIDTH / 2)
        opponent_x = ensure_scalar(opponent_x, SCREEN_WIDTH / 2)
        player_attacking = safe_bool_check(player_attacking)
        opponent_attacking = safe_bool_check(opponent_attacking)

        player_velocity, opponent_velocity = 0.0, 0.0

        # CRITICAL FIX: Safe scalar comparisons
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

        # Direction change detection with safe comparisons
        if (
            len(self.player_velocity_history) >= 2
            and abs(self.prev_player_velocity) > self.direction_change_threshold
            and abs(player_velocity) > self.direction_change_threshold
        ):

            if (self.prev_player_velocity > 0 and player_velocity < 0) or (
                self.prev_player_velocity < 0 and player_velocity > 0
            ):
                self.player_direction_changes += 1
                self.direction_change_timestamps.append(self.frame_count)

        if (
            len(self.opponent_velocity_history) >= 2
            and abs(self.prev_opponent_velocity) > self.direction_change_threshold
            and abs(opponent_velocity) > self.direction_change_threshold
        ):

            if (self.prev_opponent_velocity > 0 and opponent_velocity < 0) or (
                self.prev_opponent_velocity < 0 and opponent_velocity > 0
            ):
                self.opponent_direction_changes += 1

        # Oscillation amplitude calculation
        if len(self.player_x_history) >= 8:
            recent_positions = list(self.player_x_history)[-8:]
            finite_positions = [p for p in recent_positions if np.isfinite(p)]
            if len(finite_positions) >= 2:
                self.player_oscillation_amplitude = max(finite_positions) - min(
                    finite_positions
                )
            else:
                self.player_oscillation_amplitude = 0.0

        if len(self.opponent_x_history) >= 8:
            recent_positions = list(self.opponent_x_history)[-8:]
            finite_positions = [p for p in recent_positions if np.isfinite(p)]
            if len(finite_positions) >= 2:
                self.opponent_oscillation_amplitude = max(finite_positions) - min(
                    finite_positions
                )
            else:
                self.opponent_oscillation_amplitude = 0.0

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
        # CRITICAL FIX: Safe boolean operations
        player_moving_forward = (
            safe_comparison(player_x, opponent_x, "<") and player_velocity > 0.5
        ) or (safe_comparison(player_x, opponent_x, ">") and player_velocity < -0.5)

        player_moving_backward = (
            safe_comparison(player_x, opponent_x, "<") and player_velocity < -0.5
        ) or (safe_comparison(player_x, opponent_x, ">") and player_velocity > 0.5)

        opponent_moving_forward = (
            safe_comparison(opponent_x, player_x, "<") and opponent_velocity > 0.5
        ) or (safe_comparison(opponent_x, player_x, ">") and opponent_velocity < -0.5)

        opponent_moving_backward = (
            safe_comparison(opponent_x, player_x, "<") and opponent_velocity < -0.5
        ) or (safe_comparison(opponent_x, player_x, ">") and opponent_velocity > 0.5)

        neutral_game = (
            not player_attacking
            and not opponent_attacking
            and distance > self.CLOSE_RANGE
            and abs(player_velocity) < 2.0
            and abs(opponent_velocity) < 2.0
        )

        if neutral_game:
            self.neutral_game_duration += 1
        else:
            if self.neutral_game_duration > 0:
                self.advantage_transitions += 1
            self.neutral_game_duration = 0

        if player_moving_forward and distance > self.MID_RANGE:
            self.aggressive_forward_count += 1
        elif player_moving_backward and distance < self.MID_RANGE:
            self.defensive_backward_count += 1
        elif (
            abs(player_velocity) > self.movement_threshold
            and self.MID_RANGE <= distance <= self.FAR_RANGE
        ):
            self.neutral_dance_count += 1

        if (
            distance > self.WHIFF_BAIT_RANGE
            and distance < self.MID_RANGE + 5
            and player_moving_forward
            and not player_attacking
        ):
            self.whiff_bait_attempts += 1

        self.space_control_score = self._calculate_enhanced_space_control(
            player_x, opponent_x, player_velocity, opponent_velocity, distance
        )

        return {
            "player_moving_forward": player_moving_forward,
            "player_moving_backward": player_moving_backward,
            "opponent_moving_forward": opponent_moving_forward,
            "opponent_moving_backward": opponent_moving_backward,
            "neutral_game": neutral_game,
            "distance": distance,
            "space_control_score": self.space_control_score,
        }

    def _calculate_enhanced_space_control(
        self, player_x, opponent_x, player_velocity, opponent_velocity, distance
    ) -> float:
        screen_center = SCREEN_WIDTH / 2
        player_center_dist = abs(player_x - screen_center)
        opponent_center_dist = abs(opponent_x - screen_center)
        center_control = safe_divide(
            opponent_center_dist - player_center_dist, SCREEN_WIDTH / 2, 0.0
        )

        movement_initiative = 0.0
        velocity_diff = abs(player_velocity) - abs(opponent_velocity)
        if abs(velocity_diff) > 0.1:
            movement_initiative = 0.3 if velocity_diff > 0 else -0.3

        range_control = 0.0
        if self.CLOSE_RANGE <= distance <= self.MID_RANGE:
            range_control = 0.4
        elif distance > self.FAR_RANGE:
            range_control = -0.3
        elif self.MID_RANGE < distance <= self.FAR_RANGE:
            range_control = 0.2

        oscillation_effectiveness = 0.0
        if self.frame_count > 60:
            freq = self.get_rolling_window_frequency()
            if 1.0 <= freq <= 3.0:
                oscillation_effectiveness = 0.3

        total_control = (
            center_control * 0.3
            + movement_initiative * 0.3
            + range_control * 0.2
            + oscillation_effectiveness * 0.2
        )

        return np.clip(total_control, -1.0, 1.0) if np.isfinite(total_control) else 0.0

    def get_rolling_window_frequency(self) -> float:
        if len(self.direction_change_timestamps) < 2:
            return 0.0
        window_frames = 600
        recent_changes = sum(
            1
            for ts in self.direction_change_timestamps
            if self.frame_count - ts <= window_frames
        )
        window_seconds = min(
            window_frames / 60.0, max(self.frame_count / 60.0, 1 / 60.0)
        )
        frequency = safe_divide(recent_changes, window_seconds, 0.0)
        return frequency

    def get_oscillation_features(self) -> np.ndarray:
        """CRITICAL FIX: Ensure this always returns exactly 12 features."""
        try:
            features = np.zeros(12, dtype=np.float32)
            if self.frame_count == 0:
                return features

            rolling_freq = self.get_rolling_window_frequency()
            features[0] = np.clip(rolling_freq / 5.0, 0.0, 1.0)

            if self.frame_count > 0:
                opponent_freq = safe_divide(
                    self.opponent_direction_changes, max(1, self.frame_count / 60), 0.0
                )
                features[1] = np.clip(opponent_freq / 5.0, 0.0, 1.0)
            else:
                features[1] = 0.0

            features[2] = np.clip(self.player_oscillation_amplitude / 90.0, 0.0, 1.0)
            features[3] = np.clip(self.opponent_oscillation_amplitude / 90.0, 0.0, 1.0)
            features[4] = np.clip(self.space_control_score, -1.0, 1.0)
            features[5] = np.clip(self.neutral_game_duration / 180.0, 0.0, 1.0)

            total_movement = (
                self.aggressive_forward_count
                + self.defensive_backward_count
                + self.neutral_dance_count
            )
            if total_movement > 0:
                features[6] = safe_divide(
                    self.aggressive_forward_count, total_movement, 0.0
                )
                features[7] = safe_divide(
                    self.defensive_backward_count, total_movement, 0.0
                )
                features[8] = safe_divide(self.neutral_dance_count, total_movement, 0.0)
            else:
                features[6] = features[7] = features[8] = 0.33

            if self.frame_count > 0:
                frame_rate = max(1, self.frame_count / 60)
                features[9] = np.clip(
                    safe_divide(self.whiff_bait_attempts, frame_rate, 0.0), 0.0, 1.0
                )
                features[10] = np.clip(
                    safe_divide(self.advantage_transitions, frame_rate, 0.0), 0.0, 1.0
                )
            else:
                features[9] = features[10] = 0.0

            if (
                len(self.player_velocity_history) > 0
                and len(self.opponent_velocity_history) > 0
            ):
                velocity_diff = (
                    self.player_velocity_history[-1]
                    - self.opponent_velocity_history[-1]
                )
                features[11] = (
                    np.clip(velocity_diff / 5.0, -1.0, 1.0)
                    if np.isfinite(velocity_diff)
                    else 0.0
                )
            else:
                features[11] = 0.0

            features = sanitize_array(features, 0.0)

            # CRITICAL FIX: Final dimension check
            if len(features) != 12:
                print(
                    f"⚠️  Oscillation features wrong dimension: got {len(features)}, expected 12"
                )
                return ensure_feature_dimension(features, 12)

            return features.astype(np.float32)

        except Exception as e:
            print(f"⚠️  Error in get_oscillation_features: {e}")
            return np.zeros(12, dtype=np.float32)

    def get_stats(self) -> Dict:
        return {
            "player_direction_changes": self.player_direction_changes,
            "opponent_direction_changes": self.opponent_direction_changes,
            "player_oscillation_amplitude": self.player_oscillation_amplitude,
            "opponent_oscillation_amplitude": self.opponent_oscillation_amplitude,
            "space_control_score": self.space_control_score,
            "neutral_game_duration": self.neutral_game_duration,
            "whiff_bait_attempts": self.whiff_bait_attempts,
            "advantage_transitions": self.advantage_transitions,
            "rolling_window_frequency": self.get_rolling_window_frequency(),
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

        # CRITICAL NEW FIX: Track current feature dimension mode
        self.current_feature_dim = VECTOR_FEATURE_DIM
        print(
            f"🔧 StrategicFeatureTracker initialized with {self.current_feature_dim} features"
        )

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """CRITICAL FIX: Normalize features with consistent dimension handling."""
        # CRITICAL FIX: Ensure features match current dimension
        features = ensure_feature_dimension(features, self.current_feature_dim)

        nan_mask = ~np.isfinite(features)
        if np.any(nan_mask):
            self.feature_nan_count += np.sum(nan_mask)
            if self.feature_nan_count % 100 == 0:
                print(f"⚠️  Feature NaN cleaned (total: {self.feature_nan_count})")
            features = sanitize_array(features, 0.0)

        if self.feature_rolling_mean is None:
            self.feature_rolling_mean = np.zeros(
                self.current_feature_dim, dtype=np.float32
            )
            self.feature_rolling_std = np.ones(
                self.current_feature_dim, dtype=np.float32
            )

        # CRITICAL FIX: Handle dimension mismatch in rolling statistics
        if len(self.feature_rolling_mean) != self.current_feature_dim:
            print(f"⚠️  Rolling statistics dimension mismatch, reinitializing")
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

        # CRITICAL FIX: Handle safe_divide returning arrays
        try:
            if isinstance(features, np.ndarray) and isinstance(
                self.feature_rolling_mean, np.ndarray
            ):
                # Element-wise division for arrays
                normalized = (features - self.feature_rolling_mean) / safe_std
                normalized = np.where(np.isfinite(normalized), normalized, 0.0)
            else:
                # Fallback to safe_divide for problematic cases
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

        # CRITICAL FIX: Final dimension check after normalization
        normalized = ensure_feature_dimension(normalized, self.current_feature_dim)
        return normalized.astype(np.float32)

    def update(self, info: Dict, button_features: np.ndarray) -> np.ndarray:
        self.current_frame += 1

        try:
            # CRITICAL FIX: Ensure all extracted values are scalars
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

            # CRITICAL FIX: Safe scalar comparisons for combo logic
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

            # CRITICAL FIX: Use safe_bool_check for button features analysis
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

            # CRITICAL FIX: Calculate features based on current mode
            features = self._calculate_enhanced_features(
                info, distance, oscillation_analysis
            )

            # CRITICAL FIX: Ensure features match current dimension before normalization
            features = ensure_feature_dimension(features, self.current_feature_dim)

            # Now normalize the features
            features = self._normalize_features(features)

            # CRITICAL FIX: Final validation before return
            features = ensure_feature_dimension(features, self.current_feature_dim)

            self.prev_player_health = player_health
            self.prev_opponent_health = opponent_health
            self.prev_score = score

            # Bait-punish integration
            if BAIT_PUNISH_AVAILABLE and not self._bait_punish_integrated:
                try:
                    integrate_bait_punish_system(self)
                    self._bait_punish_integrated = True
                    # Update current feature dim to enhanced mode
                    self.current_feature_dim = ENHANCED_VECTOR_FEATURE_DIM
                    # Reinitialize normalization statistics for new dimension
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
        """CRITICAL FIX: Calculate features with dimension awareness."""

        # Always start with base features (45)
        features = np.zeros(BASE_VECTOR_FEATURE_DIM, dtype=np.float32)

        try:
            # CRITICAL FIX: Ensure all extracted values are scalars
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
            # CRITICAL FIX: Safe comparisons for health danger zones
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

            # CRITICAL FIX: Safe corner detection
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

            # CRITICAL FIX: Safe optimal spacing check
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

            # CRITICAL FIX: Safe oscillation feature extraction (21-32)
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
                    print(
                        f"⚠️  Oscillation features wrong type/size: {type(oscillation_features)}, len: {len(oscillation_features) if hasattr(oscillation_features, '__len__') else 'N/A'}"
                    )
                    features[21:33] = np.zeros(12, dtype=np.float32)
            except Exception as e:
                print(f"⚠️  Error getting oscillation features: {e}")
                features[21:33] = np.zeros(12, dtype=np.float32)

            # CRITICAL FIX: Safe button features extraction (33-44)
            try:
                if len(self.button_features_history) > 0:
                    button_hist = self.button_features_history[-1]
                    if isinstance(button_hist, np.ndarray) and len(button_hist) == 12:
                        features[33:45] = button_hist
                    else:
                        print(
                            f"⚠️  Button history wrong type/size: {type(button_hist)}, len: {len(button_hist) if hasattr(button_hist, '__len__') else 'N/A'}"
                        )
                        features[33:45] = np.zeros(12, dtype=np.float32)
                else:
                    features[33:45] = np.zeros(12, dtype=np.float32)
            except Exception as e:
                print(f"⚠️  Error getting button features: {e}")
                features[33:45] = np.zeros(12, dtype=np.float32)

            features = sanitize_array(features, 0.0)

            # CRITICAL NEW FIX: Handle bait-punish features expansion properly
            if (
                BAIT_PUNISH_AVAILABLE
                and self._bait_punish_integrated
                and self.current_feature_dim == ENHANCED_VECTOR_FEATURE_DIM
            ):
                # Expand to enhanced features (45-51: 7 additional features)
                enhanced_features = np.zeros(
                    ENHANCED_VECTOR_FEATURE_DIM, dtype=np.float32
                )
                enhanced_features[:BASE_VECTOR_FEATURE_DIM] = features

                try:
                    # Add 7 additional bait-punish features
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
                    print(f"⚠️  Error adding bait-punish features: {e}")
                    enhanced_features[45:52] = np.zeros(7, dtype=np.float32)

                return enhanced_features
            else:
                # Return base features only
                return features

        except Exception as e:
            print(f"⚠️  Error in _calculate_enhanced_features: {e}")
            import traceback

            traceback.print_exc()
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


class FixedStreetFighterCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        visual_space = observation_space["visual_obs"]
        vector_space = observation_space["vector_obs"]
        n_input_channels = visual_space.shape[0]
        seq_length, vector_feature_count = vector_space.shape

        print(f"🔧 ENHANCED FeatureExtractor Configuration (NaN-SAFE):")
        print(f"   - Visual channels: {n_input_channels}")
        print(
            f"   - Visual size: {visual_space.shape[1]}x{visual_space.shape[2]} (FULL SIZE)"
        )
        print(
            f"   - Vector sequence: {seq_length} x {vector_feature_count} (NaN-protected)"
        )
        print(
            f"   - Vector features: {vector_feature_count} ({'Enhanced' if vector_feature_count == 52 else 'Base'})"
        )
        print(f"   - Output features: {features_dim}")

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
            nn.Dropout2d(0.1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
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

        self.apply(self._init_weights_conservative)
        print(f"   - Visual output size: {visual_output_size}")
        print(f"   - Fusion input size: {fusion_input_size}")
        print(f"   ✅ NaN-SAFE Feature Extractor initialized")

    def _init_weights_conservative(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if hasattr(m.weight, "data"):
                m.weight.data *= 0.5
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
        visual_features = self.visual_cnn(visual_obs)

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

        if torch.any(~torch.isfinite(vector_embedded)):
            vector_embedded = torch.where(
                ~torch.isfinite(vector_embedded),
                torch.zeros_like(vector_embedded),
                vector_embedded,
            )

        gru_output, _ = self.vector_gru(vector_embedded)
        vector_features = gru_output[:, -1, :]
        vector_features = self.vector_final(vector_features)

        if torch.any(~torch.isfinite(vector_features)):
            vector_features = torch.where(
                ~torch.isfinite(vector_features),
                torch.zeros_like(vector_features),
                vector_features,
            )

        combined_features = torch.cat([visual_features, vector_features], dim=1)
        output = self.fusion(combined_features)

        if torch.any(~torch.isfinite(output)):
            output = torch.where(
                ~torch.isfinite(output), torch.zeros_like(output), output
            )

        if self.training:
            output = torch.clamp(output, -5.0, 5.0)

        return output


class FixedStreetFighterPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        net_arch: Optional[Union[list, dict]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        *args,
        **kwargs,
    ):
        kwargs["features_extractor_class"] = FixedStreetFighterCNN
        kwargs["features_extractor_kwargs"] = {"features_dim": 256}
        if net_arch is None:
            net_arch = dict(pi=[128, 64], vf=[128, 64])
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs,
        )
        print("✅ NaN-SAFE Policy initialized")

    def forward(self, obs, deterministic: bool = False):
        features = self.extract_features(obs)
        if torch.any(~torch.isfinite(features)):
            features = torch.where(
                ~torch.isfinite(features), torch.zeros_like(features), features
            )

        latent_pi, latent_vf = self.mlp_extractor(features)

        if torch.any(~torch.isfinite(latent_pi)):
            latent_pi = torch.where(
                ~torch.isfinite(latent_pi), torch.zeros_like(latent_pi), latent_pi
            )
        if torch.any(~torch.isfinite(latent_vf)):
            latent_vf = torch.where(
                ~torch.isfinite(latent_vf), torch.zeros_like(latent_vf), latent_vf
            )

        action_logits = self.action_net(latent_pi)
        if torch.any(~torch.isfinite(action_logits)):
            action_logits = torch.where(
                ~torch.isfinite(action_logits),
                torch.zeros_like(action_logits),
                action_logits,
            )

        action_logits = torch.clamp(action_logits, -10.0, 10.0)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        values = self.value_net(latent_vf)
        log_prob = distribution.log_prob(actions)

        if self.training:
            values = torch.clamp(values, -50.0, 50.0)

        if torch.any(~torch.isfinite(values)):
            values = torch.where(
                ~torch.isfinite(values), torch.zeros_like(values), values
            )
        if torch.any(~torch.isfinite(log_prob)):
            log_prob = torch.where(
                ~torch.isfinite(log_prob), torch.full_like(log_prob, -1.0), log_prob
            )

        return actions, values, log_prob


class StreetFighterVisionWrapper(gym.Wrapper):
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

        print(
            f"🖼️  Using FULL SIZE frames: {self.target_size[0]}x{self.target_size[1]} (H x W)"
        )

        self.discrete_actions = StreetFighterDiscreteActions()
        self.action_space = spaces.Discrete(self.discrete_actions.num_actions)

        # CRITICAL NEW FIX: Dynamic observation space based on feature mode
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

        print(f"🔧 Observation space configured:")
        print(f"   - Visual: {self.observation_space['visual_obs'].shape}")
        print(f"   - Vector: {self.observation_space['vector_obs'].shape}")
        print(
            f"   - Vector features: {VECTOR_FEATURE_DIM} ({'Enhanced' if BAIT_PUNISH_AVAILABLE else 'Base'})"
        )

        self.frame_buffer = deque(maxlen=frame_stack)
        self.vector_features_history = deque(maxlen=frame_stack)
        self.strategic_tracker = StrategicFeatureTracker(history_length=frame_stack)
        self.full_hp = 176
        self.prev_player_health = self.full_hp
        self.prev_opponent_health = self.full_hp
        self.wins, self.losses, self.total_rounds = 0, 0, 0
        self.total_damage_dealt, self.total_damage_received = 0, 0

        if BAIT_PUNISH_AVAILABLE:
            self.reward_shaper = AdaptiveRewardShaper()
            print("✅ Adaptive reward shaper initialized")
        else:
            self.reward_shaper = None

        self.reward_scale = 0.1
        self.episode_steps = 0
        self.max_episode_steps = 18000
        self.episode_rewards = deque(maxlen=100)
        self.stats = {}

    def _sanitize_info(self, info: Dict) -> Dict:
        """Converts array values from a vectorized env's info dict to scalars."""
        sanitized = {}
        for k, v in info.items():
            # CRITICAL FIX: Use ensure_scalar instead of item()
            sanitized[k] = ensure_scalar(v, 0)
        return sanitized

    def _create_initial_vector_features(self, info):
        initial_button_features = np.zeros(12, dtype=np.float32)
        try:
            sanitized_info = self._sanitize_info(info)
            initial_features = self.strategic_tracker.update(
                sanitized_info, initial_button_features
            )

            # CRITICAL FIX: Ensure features match current dimension
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

        # CRITICAL FIX: Use ensure_scalar for health values
        curr_player_health = ensure_scalar(
            sanitized_info.get("agent_hp", self.full_hp), self.full_hp
        )
        curr_opponent_health = ensure_scalar(
            sanitized_info.get("enemy_hp", self.full_hp), self.full_hp
        )

        base_reward, custom_done = self._calculate_base_reward(
            curr_player_health, curr_opponent_health
        )

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

        final_reward = final_reward if np.isfinite(final_reward) else 0.0

        # CRITICAL FIX: Safe episode step comparison
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

            # CRITICAL NEW FIX: Ensure vector features match current dimension
            vector_features = ensure_feature_dimension(
                vector_features, VECTOR_FEATURE_DIM
            )

        except Exception as e:
            print(f"⚠️  Vector feature update error: {e}, using zeros")
            vector_features = np.zeros(VECTOR_FEATURE_DIM, dtype=np.float32)

        self.vector_features_history.append(vector_features)
        self._update_enhanced_stats()

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
                    "player_oscillation_frequency": combo_stats.get(
                        "rolling_window_frequency", 0.0
                    ),
                    "space_control_score": combo_stats.get("space_control_score", 0.0),
                    "episode_steps": self.episode_steps,
                    "current_feature_dim": self.strategic_tracker.current_feature_dim,
                }
            )
        except Exception as e:
            print(f"⚠️  Error updating stats: {e}")
            self.stats = {"error": "stats_update_failed"}

    def _calculate_base_reward(self, curr_player_health, curr_opponent_health):
        reward, done = 0.0, False

        # CRITICAL FIX: Ensure scalar comparisons using safe_comparison
        curr_player_health = ensure_scalar(curr_player_health, self.full_hp)
        curr_opponent_health = ensure_scalar(curr_opponent_health, self.full_hp)

        # CRITICAL FIX: Use safe_comparison for health checks
        player_dead = safe_comparison(curr_player_health, 0, "<=")
        opponent_dead = safe_comparison(curr_opponent_health, 0, "<=")

        if player_dead or opponent_dead:
            self.total_rounds += 1
            if opponent_dead and not player_dead:
                self.wins += 1
                win_bonus = (
                    25.0 + safe_divide(curr_player_health, self.full_hp, 0.0) * 10.0
                )
                reward += win_bonus
                print(
                    f"🏆 AI WON! Total: {self.wins}W/{self.losses}L (Round {self.total_rounds})"
                )
            else:
                self.losses += 1
                reward -= 10.0
                print(
                    f"💀 AI LOST! Total: {self.wins}W/{self.losses}L (Round {self.total_rounds})"
                )
            done = True
            combo_bonus = self.strategic_tracker.combo_counter * 0.02
            reward += combo_bonus

        damage_dealt = 0
        damage_received = 0

        # CRITICAL FIX: Safe damage calculations
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

        reward += (damage_dealt * 0.1) - (damage_received * 0.05)
        self.total_damage_dealt += damage_dealt
        self.total_damage_received += damage_received

        # Strategic bonuses with safe operations
        try:
            osc_tracker = self.strategic_tracker.oscillation_tracker
            rolling_freq = osc_tracker.get_rolling_window_frequency()
            if (
                np.isfinite(rolling_freq)
                and safe_comparison(rolling_freq, 1.0, ">=")
                and safe_comparison(rolling_freq, 3.0, "<=")
            ):
                reward += 0.01
            space_control = osc_tracker.space_control_score
            if np.isfinite(space_control) and safe_comparison(space_control, 0, ">"):
                reward += space_control * 0.005
        except Exception as e:
            print(f"⚠️  Error in strategic bonuses: {e}")

        reward -= 0.001
        reward *= self.reward_scale
        reward = np.clip(reward, -2.0, 2.0) if np.isfinite(reward) else 0.0

        self.prev_player_health, self.prev_opponent_health = (
            curr_player_health,
            curr_opponent_health,
        )

        if done:
            self.episode_rewards.append(reward)

        return reward, done

    def _get_observation(self):
        try:
            visual_obs = np.concatenate(list(self.frame_buffer), axis=2).transpose(
                2, 0, 1
            )
            vector_obs = np.stack(list(self.vector_features_history))

            visual_obs = sanitize_array(visual_obs, 0.0).astype(np.uint8)
            vector_obs = sanitize_array(vector_obs, 0.0).astype(np.float32)

            # CRITICAL NEW FIX: Ensure vector_obs matches expected dimension
            expected_shape = (self.frame_stack, VECTOR_FEATURE_DIM)
            if vector_obs.shape != expected_shape:
                print(
                    f"⚠️  Vector obs shape mismatch: got {vector_obs.shape}, expected {expected_shape}"
                )
                # Create properly shaped vector observation
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


def verify_gradient_flow(model, env, device=None):
    """CRITICAL FIX: Enhanced gradient flow verification with NaN detection."""
    print("\n🔬 NaN-SAFE Gradient Flow Verification")
    print("=" * 70)

    if device is None:
        device = next(model.policy.parameters()).device

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
    print(f"   - Memory usage: {visual_obs.numel() * 4 / 1024 / 1024:.1f} MB")
    print(f"   - Range: {visual_obs.min().item():.1f} to {visual_obs.max().item():.1f}")
    print(f"   - NaN count: {torch.sum(~torch.isfinite(visual_obs)).item()}")

    vector_obs = obs_tensor["vector_obs"]
    print(f"🔍 Vector Feature Analysis (NaN-SAFE):")
    print(f"   - Shape: {vector_obs.shape}")
    print(
        f"   - Features: {vector_obs.shape[-1]} ({'Enhanced with bait-punish' if vector_obs.shape[-1] == 52 else 'Base features'})"
    )
    print(f"   - Range: {vector_obs.min().item():.3f} to {vector_obs.max().item():.3f}")
    print(f"   - NaN count: {torch.sum(~torch.isfinite(vector_obs)).item()}")

    if torch.sum(~torch.isfinite(vector_obs)) > 0:
        print("   🚨 CRITICAL: NaN values detected in vector features!")
        return False
    else:
        print("   ✅ No NaN values detected")

    if vector_obs.abs().max() > 10.0:
        print("   ⚠️  WARNING: Large vector values detected!")
    else:
        print("   ✅ Vector features in stable range")

    model.policy.train()
    for param in model.policy.parameters():
        param.requires_grad = True

    try:
        actions, values, log_probs = model.policy(obs_tensor)
        print(f"✅ Policy forward pass successful")
        print(f"   - Value output: {values.item():.3f}")
        print(f"   - Value NaN count: {torch.sum(~torch.isfinite(values)).item()}")
        print(
            f"   - Log prob NaN count: {torch.sum(~torch.isfinite(log_probs)).item()}"
        )

        if (
            torch.sum(~torch.isfinite(values)) > 0
            or torch.sum(~torch.isfinite(log_probs)) > 0
        ):
            print("   🚨 CRITICAL: NaN values in policy outputs!")
            return False

        if abs(values.item()) > 100.0:
            print("   🚨 CRITICAL: Value function output too large!")
            return False
        else:
            print("   ✅ Value function output is stable")

    except Exception as e:
        print(f"❌ Policy forward pass failed: {e}")
        return False

    loss = values.mean() + log_probs.mean() * 0.1
    model.policy.zero_grad()

    try:
        loss.backward()
        print("✅ Backward pass successful")
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        return False

    total_params, params_with_grads, total_grad_norm, nan_grad_count = 0, 0, 0.0, 0

    for name, param in model.policy.named_parameters():
        total_params += param.numel()
        if param.grad is not None:
            params_with_grads += param.numel()
            grad_norm = param.grad.norm().item()
            if np.isfinite(grad_norm):
                total_grad_norm += grad_norm
            else:
                nan_grad_count += 1

    coverage = (params_with_grads / total_params) * 100
    avg_grad_norm = safe_divide(total_grad_norm, max(params_with_grads, 1), 0.0)

    print(f"📊 Gradient Analysis:")
    print(f"   - Coverage: {coverage:.1f}%")
    print(f"   - Average norm: {avg_grad_norm:.6f}")
    print(f"   - NaN gradients: {nan_grad_count}")

    if nan_grad_count > 0:
        print("   🚨 CRITICAL: NaN gradients detected!")
        return False

    if coverage > 95 and avg_grad_norm < 10.0 and nan_grad_count == 0:
        print("✅ EXCELLENT: Stable, NaN-free gradient flow ready for training!")
        return True
    else:
        print("❌ Gradient flow issues detected")
        return False


# Export components
__all__ = [
    "StreetFighterVisionWrapper",
    "FixedStreetFighterCNN",
    "FixedStreetFighterPolicy",
    "verify_gradient_flow",
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
    "VECTOR_FEATURE_DIM",
    "BASE_VECTOR_FEATURE_DIM",
    "ENHANCED_VECTOR_FEATURE_DIM",
]

if BAIT_PUNISH_AVAILABLE:
    __all__.append("AdaptiveRewardShaper")
