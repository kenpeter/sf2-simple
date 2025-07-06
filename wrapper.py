#!/usr/bin/env python3
"""
wrapper.py - STABILIZED VERSION with training instability fixes
FIXES: Value loss normalization, reward scaling, feature stability, conservative architecture
Addresses: High value loss (35+), Low explained variance (0.11), Low clip fraction (0.04)
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import deque
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Dict, Tuple, List, Type, Any, Optional, Union
import math
import logging
import os
from datetime import datetime
import retro

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
SCREEN_WIDTH = 180
SCREEN_HEIGHT = 128
VECTOR_FEATURE_DIM = 45  # 21 strategic + 12 oscillation + 12 button


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
        self.CLOSE_RANGE = 25
        self.MID_RANGE = 45
        self.FAR_RANGE = 70
        self.WHIFF_BAIT_RANGE = 35
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
        player_velocity, opponent_velocity = 0.0, 0.0
        if self.prev_player_x is not None:
            raw_velocity = player_x - self.prev_player_x
            player_velocity = (
                self.velocity_smoothing_factor * raw_velocity
                + (1 - self.velocity_smoothing_factor) * self.prev_player_velocity
            )
        if self.prev_opponent_x is not None:
            raw_velocity = opponent_x - self.prev_opponent_x
            opponent_velocity = (
                self.velocity_smoothing_factor * raw_velocity
                + (1 - self.velocity_smoothing_factor) * self.prev_opponent_velocity
            )

        self.player_x_history.append(player_x)
        self.opponent_x_history.append(opponent_x)
        self.player_velocity_history.append(player_velocity)
        self.opponent_velocity_history.append(opponent_velocity)

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

        if len(self.player_x_history) >= 8:
            self.player_oscillation_amplitude = max(
                list(self.player_x_history)[-8:]
            ) - min(list(self.player_x_history)[-8:])
        if len(self.opponent_x_history) >= 8:
            self.opponent_oscillation_amplitude = max(
                list(self.opponent_x_history)[-8:]
            ) - min(list(self.opponent_x_history)[-8:])

        movement_analysis = self._analyze_movement_patterns(
            player_x,
            opponent_x,
            player_velocity,
            opponent_velocity,
            abs(player_x - opponent_x),
            player_attacking,
            opponent_attacking,
        )
        (
            self.prev_player_x,
            self.prev_opponent_x,
            self.prev_player_velocity,
            self.prev_opponent_velocity,
        ) = (player_x, opponent_x, player_velocity, opponent_velocity)
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
        player_moving_forward = (player_x < opponent_x and player_velocity > 0) or (
            player_x > opponent_x and player_velocity < 0
        )
        player_moving_backward = (player_x < opponent_x and player_velocity < 0) or (
            player_x > opponent_x and player_velocity > 0
        )
        opponent_moving_forward = (opponent_x < player_x and opponent_velocity > 0) or (
            opponent_x > player_x and opponent_velocity < 0
        )
        opponent_moving_backward = (
            opponent_x < player_x and opponent_velocity < 0
        ) or (opponent_x > player_x and opponent_velocity > 0)
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
        center_control = (
            abs(opponent_x - screen_center) - abs(player_x - screen_center)
        ) / (SCREEN_WIDTH / 2)

        movement_initiative = 0.0
        if abs(player_velocity) > abs(opponent_velocity) + 0.1:
            movement_initiative = 0.3 if player_velocity > 0 else -0.3
        elif abs(opponent_velocity) > abs(player_velocity) + 0.1:
            movement_initiative = -0.3 if opponent_velocity > 0 else 0.3

        range_control = 0.0
        if self.CLOSE_RANGE <= distance <= self.MID_RANGE:
            range_control = 0.4
        elif distance > self.FAR_RANGE:
            range_control = -0.3
        elif self.MID_RANGE < distance <= self.FAR_RANGE:
            range_control = 0.2

        oscillation_effectiveness = 0.0
        if self.frame_count > 60 and 1.0 <= self.get_rolling_window_frequency() <= 3.0:
            oscillation_effectiveness = 0.3

        total_control = (
            center_control * 0.3
            + movement_initiative * 0.3
            + range_control * 0.2
            + oscillation_effectiveness * 0.2
        )
        return np.clip(total_control, -1.0, 1.0)

    def get_rolling_window_frequency(self) -> float:
        if len(self.direction_change_timestamps) < 2:
            return 0.0
        window_frames = 600
        recent_changes = sum(
            1
            for ts in self.direction_change_timestamps
            if self.frame_count - ts <= window_frames
        )
        window_seconds = min(window_frames / 60.0, self.frame_count / 60.0)
        return recent_changes / window_seconds if window_seconds > 0 else 0.0

    def get_oscillation_features(self) -> np.ndarray:
        features = np.zeros(12, dtype=np.float32)
        if self.frame_count == 0:
            return features

        rolling_freq = self.get_rolling_window_frequency()
        features[0] = np.clip(rolling_freq / 5.0, 0.0, 1.0)
        features[1] = np.clip(
            (self.opponent_direction_changes / max(1, self.frame_count / 60)) / 5.0,
            0.0,
            1.0,
        )
        features[2] = np.clip(self.player_oscillation_amplitude / 50.0, 0.0, 1.0)
        features[3] = np.clip(self.opponent_oscillation_amplitude / 50.0, 0.0, 1.0)
        features[4] = np.clip(self.space_control_score, -1.0, 1.0)
        features[5] = np.clip(self.neutral_game_duration / 180.0, 0.0, 1.0)

        total_movement = (
            self.aggressive_forward_count
            + self.defensive_backward_count
            + self.neutral_dance_count
        )
        if total_movement > 0:
            features[6] = self.aggressive_forward_count / total_movement
            features[7] = self.defensive_backward_count / total_movement
            features[8] = self.neutral_dance_count / total_movement

        features[9] = np.clip(
            self.whiff_bait_attempts / max(1, self.frame_count / 60), 0.0, 1.0
        )
        features[10] = np.clip(
            self.advantage_transitions / max(1, self.frame_count / 60), 0.0, 1.0
        )

        if (
            len(self.player_velocity_history) > 0
            and len(self.opponent_velocity_history) > 0
        ):
            velocity_diff = (
                self.player_velocity_history[-1] - self.opponent_velocity_history[-1]
            )
            features[11] = np.clip(velocity_diff / 5.0, -1.0, 1.0)

        return features

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
        self.recent_damage_events = deque(maxlen=5)
        self.button_features_history = deque(maxlen=history_length)
        self.previous_button_features = np.zeros(12, dtype=np.float32)
        self.oscillation_tracker = OscillationTracker(history_length=16)
        self.close_combat_count = 0
        self.total_frames = 0

        # STABILITY FIX: Feature normalization for stability
        self.feature_rolling_mean = np.zeros(VECTOR_FEATURE_DIM, dtype=np.float32)
        self.feature_rolling_std = np.ones(VECTOR_FEATURE_DIM, dtype=np.float32)
        self.normalization_alpha = 0.999

        self.DANGER_ZONE_HEALTH = MAX_HEALTH * 0.25
        self.CORNER_THRESHOLD = 30
        self.CLOSE_DISTANCE = 40
        self.OPTIMAL_SPACING_MIN = 35
        self.OPTIMAL_SPACING_MAX = 55
        self.COMBO_TIMEOUT_FRAMES = 60
        self.MIN_SCORE_INCREASE_FOR_HIT = 50
        self.prev_player_health = None
        self.prev_opponent_health = None
        self.prev_score = None

    def update(self, info: Dict, button_features: np.ndarray) -> np.ndarray:
        self.current_frame += 1
        player_health = info.get("agent_hp", MAX_HEALTH)
        opponent_health = info.get("enemy_hp", MAX_HEALTH)
        score = info.get("score", 0)
        player_x = info.get("agent_x", SCREEN_WIDTH / 2)
        opponent_x = info.get("enemy_x", SCREEN_WIDTH / 2)

        self.player_health_history.append(player_health)
        self.opponent_health_history.append(opponent_health)

        score_change = score - self.prev_score if self.prev_score is not None else 0
        if score_change >= self.MIN_SCORE_INCREASE_FOR_HIT:
            if (
                self.current_frame - self.last_score_increase_frame
                <= self.COMBO_TIMEOUT_FRAMES
            ):
                self.combo_counter += 1
            else:
                self.combo_counter = 1
            self.last_score_increase_frame = self.current_frame
            self.max_combo_this_round = max(
                self.max_combo_this_round, self.combo_counter
            )
        elif (
            self.current_frame - self.last_score_increase_frame
            > self.COMBO_TIMEOUT_FRAMES
        ):
            self.combo_counter = 0

        self.score_history.append(score)
        self.score_change_history.append(score_change)
        self.button_features_history.append(self.previous_button_features.copy())
        self.previous_button_features = button_features.copy()

        player_damage = (
            max(0, self.prev_opponent_health - opponent_health)
            if self.prev_opponent_health is not None
            else 0
        )
        opponent_damage = (
            max(0, self.prev_player_health - player_health)
            if self.prev_player_health is not None
            else 0
        )
        self.player_damage_dealt_history.append(player_damage)
        self.opponent_damage_dealt_history.append(opponent_damage)

        player_attacking = np.any(button_features[[0, 1, 8, 9, 10, 11]])
        oscillation_analysis = self.oscillation_tracker.update(
            player_x, opponent_x, player_attacking
        )

        self.total_frames += 1
        distance = abs(player_x - opponent_x)
        if distance <= self.CLOSE_DISTANCE:
            self.close_combat_count += 1

        features = self._calculate_enhanced_features(
            info, distance, oscillation_analysis
        )

        # STABILITY FIX: Normalize features for training stability
        features = self._normalize_features(features)

        self.prev_player_health = player_health
        self.prev_opponent_health = opponent_health
        self.prev_score = score
        return features

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """STABILITY FIX: Normalize features to prevent value explosions that cause high value loss."""
        # Update rolling statistics for stability
        self.feature_rolling_mean = (
            self.normalization_alpha * self.feature_rolling_mean
            + (1 - self.normalization_alpha) * features
        )
        squared_diff = (features - self.feature_rolling_mean) ** 2
        self.feature_rolling_std = np.sqrt(
            self.normalization_alpha * (self.feature_rolling_std**2)
            + (1 - self.normalization_alpha) * squared_diff
        )

        # Prevent division by zero
        safe_std = np.maximum(self.feature_rolling_std, 1e-6)

        # Normalize and clip to prevent extreme values that destabilize training
        normalized = (features - self.feature_rolling_mean) / safe_std
        normalized = np.clip(normalized, -3.0, 3.0)

        return normalized.astype(np.float32)

    def _calculate_enhanced_features(
        self, info, distance, oscillation_analysis
    ) -> np.ndarray:
        features = np.zeros(VECTOR_FEATURE_DIM, dtype=np.float32)
        player_health, opponent_health = info.get("agent_hp", MAX_HEALTH), info.get(
            "enemy_hp", MAX_HEALTH
        )
        player_x, opponent_x = info.get("agent_x", SCREEN_WIDTH / 2), info.get(
            "enemy_x", SCREEN_WIDTH / 2
        )

        # Strategic features (21) - with stability improvements
        features[0] = 1.0 if player_health <= self.DANGER_ZONE_HEALTH else 0.0
        features[1] = 1.0 if opponent_health <= self.DANGER_ZONE_HEALTH else 0.0

        # STABILITY FIX: Safer health ratio calculation to prevent extreme values
        if opponent_health > 0:
            health_ratio = player_health / opponent_health
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
            min(opponent_x, SCREEN_WIDTH - opponent_x) / (SCREEN_WIDTH / 2), 0.0, 1.0
        )
        features[8] = (
            1.0
            if min(player_x, SCREEN_WIDTH - player_x) <= self.CORNER_THRESHOLD
            else 0.0
        )
        features[9] = (
            1.0
            if min(opponent_x, SCREEN_WIDTH - opponent_x) <= self.CORNER_THRESHOLD
            else 0.0
        )
        features[10] = np.sign(
            abs(opponent_x - SCREEN_WIDTH / 2) - abs(player_x - SCREEN_WIDTH / 2)
        )
        features[11] = np.clip(
            (info.get("agent_y", 64) - info.get("enemy_y", 64)) / (SCREEN_HEIGHT / 2),
            -1.0,
            1.0,
        )
        features[12] = oscillation_analysis.get("space_control_score", 0.0)
        features[13] = (
            1.0
            if self.OPTIMAL_SPACING_MIN <= distance <= self.OPTIMAL_SPACING_MAX
            else 0.0
        )
        features[14] = (
            1.0
            if oscillation_analysis.get("player_moving_forward", False)
            else (
                -1.0
                if oscillation_analysis.get("player_moving_backward", False)
                else 0.0
            )
        )
        features[15] = (
            1.0 if oscillation_analysis.get("player_moving_backward", False) else 0.0
        )
        features[16] = (
            self.close_combat_count / self.total_frames
            if self.total_frames > 0
            else 0.0
        )
        features[17] = self._calculate_enhanced_score_momentum()
        features[18] = np.clip(
            (info.get("agent_status", 0) - info.get("enemy_status", 0)) / 100.0,
            -1.0,
            1.0,
        )
        features[19] = min(info.get("agent_victories", 0) / 10.0, 1.0)
        features[20] = min(info.get("enemy_victories", 0) / 10.0, 1.0)

        # Oscillation features (12)
        features[21:33] = self.oscillation_tracker.get_oscillation_features()

        # Previous button state (12)
        features[33:45] = (
            self.button_features_history[-1]
            if len(self.button_features_history) > 0
            else np.zeros(12)
        )

        return features

    def _calculate_momentum(self, history):
        if len(history) < 2:
            return 0.0
        values = list(history)
        changes = [values[i] - values[i - 1] for i in range(1, len(values))]
        momentum = np.mean(changes[-3:]) if changes else 0.0
        # STABILITY FIX: Clip momentum to prevent extreme values
        return np.clip(momentum, -50.0, 50.0)

    def _calculate_enhanced_score_momentum(self) -> float:
        if len(self.score_change_history) < 2:
            return 0.0
        base_momentum = np.mean(
            [max(0, c) for c in list(self.score_change_history)[-5:]]
        )
        combo_multiplier = 1.0 + (self.combo_counter * 0.1)
        momentum = (base_momentum * combo_multiplier) / 100.0
        # STABILITY FIX: Clip score momentum to prevent value function instability
        return np.clip(momentum, -2.0, 5.0)

    def get_combo_stats(self) -> Dict:
        combo_stats = {
            "current_combo": self.combo_counter,
            "max_combo_this_round": self.max_combo_this_round,
        }
        combo_stats.update(self.oscillation_tracker.get_stats())
        return combo_stats


# --- STABILIZED FEATURE EXTRACTOR FOR TRAINING STABILITY ---


class FixedStreetFighterCNN(BaseFeaturesExtractor):
    """
    STABILIZED Feature extractor with training stability fixes.
    FIXES: Gradient explosion, feature scaling, conservative architecture for value loss stability.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        visual_space = observation_space["visual_obs"]
        vector_space = observation_space["vector_obs"]
        n_input_channels = visual_space.shape[0]
        seq_length, vector_feature_count = vector_space.shape

        print(f"🔧 STABILIZED FeatureExtractor Configuration:")
        print(f"   - Visual channels: {n_input_channels}")
        print(f"   - Vector sequence: {seq_length} x {vector_feature_count}")
        print(f"   - Output features: {features_dim}")

        # === STABILIZED VISUAL PROCESSING ===
        self.visual_cnn = nn.Sequential(
            # Reduced complexity for stability
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
        )

        # Calculate visual output size
        with torch.no_grad():
            dummy_visual = torch.zeros(
                1, n_input_channels, visual_space.shape[1], visual_space.shape[2]
            )
            visual_output_size = self.visual_cnn(dummy_visual).shape[1]

        # === STABILIZED VECTOR PROCESSING ===
        self.vector_embed = nn.Linear(vector_feature_count, 64)
        self.vector_norm = nn.LayerNorm(64)
        self.vector_dropout = nn.Dropout(0.2)

        # Simple GRU for stability (more stable than attention)
        self.vector_gru = nn.GRU(64, 32, batch_first=True, dropout=0.1)

        self.vector_final = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        # === STABILIZED FUSION LAYER ===
        fusion_input_size = visual_output_size + 32
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),  # LayerNorm for stability
            nn.Dropout(0.2),
            nn.Linear(256, features_dim),
            nn.ReLU(inplace=True),
        )

        # STABILITY FIX: Conservative weight initialization
        self.apply(self._init_weights_conservative)

        print(f"   - Fusion input size: {fusion_input_size}")
        print(f"   - Final output size: {features_dim}")
        print("   ✅ STABILIZED Feature Extractor initialized for training stability")

    def _init_weights_conservative(self, m):
        """STABILITY FIX: Conservative weight initialization to prevent gradient explosion."""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if hasattr(m.weight, "data"):
                m.weight.data *= 0.5  # Scale down for stability
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.5)  # Reduced gain
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
        """STABILIZED forward pass with gradient stability controls."""
        visual_obs = observations["visual_obs"]
        vector_obs = observations["vector_obs"]

        device = next(self.parameters()).device
        visual_obs = visual_obs.float().to(device)
        vector_obs = vector_obs.float().to(device)

        # STABILITY FIX: Careful normalization
        visual_obs = torch.clamp(visual_obs / 255.0, 0.0, 1.0)

        # Process visual features
        visual_features = self.visual_cnn(visual_obs)

        # STABILIZED vector processing
        batch_size, seq_len, feature_dim = vector_obs.shape

        # Embed and normalize vector features
        vector_embedded = self.vector_embed(vector_obs)
        vector_embedded = self.vector_norm(vector_embedded)
        vector_embedded = self.vector_dropout(vector_embedded)

        # Process with GRU
        gru_output, _ = self.vector_gru(vector_embedded)
        vector_features = gru_output[:, -1, :]  # Take last output
        vector_features = self.vector_final(vector_features)

        # Fuse features
        combined_features = torch.cat([visual_features, vector_features], dim=1)
        output = self.fusion(combined_features)

        # STABILITY FIX: Gradient clipping within forward pass
        if self.training:
            output = torch.clamp(output, -10.0, 10.0)

        return output


# --- STABILIZED POLICY CLASS ---


class FixedStreetFighterPolicy(ActorCriticPolicy):
    """STABILIZED: Custom policy with conservative parameters for training stability."""

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

        # STABILITY FIX: Conservative network architecture
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
        print("✅ STABILIZED Policy initialized for training stability")

    def forward(self, obs, deterministic: bool = False):
        """STABILIZED forward pass with value stability controls."""
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        values = self.value_net(latent_vf)
        log_prob = distribution.log_prob(actions)

        # STABILITY FIX: Clamp values to prevent explosion that causes high value loss
        if self.training:
            values = torch.clamp(values, -100.0, 100.0)

        return actions, values, log_prob


# --- STABILIZED STREET FIGHTER ENVIRONMENT WRAPPER ---


class StreetFighterVisionWrapper(gym.Wrapper):
    """
    STABILIZED: Street Fighter wrapper with reward scaling and stability improvements.
    FIXES: Reward scaling for value stability, episode length limits, normalized rewards.
    """

    def __init__(self, env, frame_stack=8, rendering=False):
        super().__init__(env)
        self.frame_stack = frame_stack
        self.rendering = rendering
        self.target_size = (128, 180)
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

        self.strategic_tracker = StrategicFeatureTracker(history_length=frame_stack)
        self.full_hp = 176
        self.prev_player_health = self.full_hp
        self.prev_opponent_health = self.full_hp
        self.wins, self.losses, self.total_rounds = 0, 0, 0
        self.total_damage_dealt, self.total_damage_received = 0, 0

        # STABILITY FIX: Reward scaling and episode management
        self.reward_scale = 0.1  # Scale down rewards for value function stability
        self.episode_steps = 0
        self.max_episode_steps = 18000  # ~5 minutes at 60fps
        self.episode_rewards = deque(maxlen=100)

        self.stats = {}

    def _create_initial_vector_features(self, info):
        """STABILITY FIX: Create well-normalized initial vector features."""
        player_health = info.get("agent_hp", self.full_hp)
        opponent_health = info.get("enemy_hp", self.full_hp)
        player_x = info.get("agent_x", SCREEN_WIDTH / 2)
        opponent_x = info.get("enemy_x", SCREEN_WIDTH / 2)

        initial_button_features = np.zeros(12, dtype=np.float32)
        initial_features = self.strategic_tracker.update(info, initial_button_features)

        # STABILITY FIX: Ensure features are in reasonable range
        initial_features = np.clip(initial_features, -5.0, 5.0)
        return initial_features

    def reset(self, **kwargs):
        """STABILIZED reset with proper initialization."""
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
        initial_button_features = np.zeros(12, dtype=np.float32)
        self.strategic_tracker.update(info, initial_button_features)

        return self._get_observation(), info

    def step(self, discrete_action):
        self.episode_steps += 1

        multibinary_action = self.discrete_actions.discrete_to_multibinary(
            discrete_action
        )
        observation, reward, done, truncated, info = self.env.step(multibinary_action)

        if self.rendering:
            self.env.render()

        curr_player_health = info.get("agent_hp", self.full_hp)
        curr_opponent_health = info.get("enemy_hp", self.full_hp)

        custom_reward, custom_done = self._calculate_stabilized_reward(
            curr_player_health, curr_opponent_health
        )

        # STABILITY FIX: Episode length limit to prevent extremely long episodes
        if self.episode_steps >= self.max_episode_steps:
            truncated = True
            print(f"Episode truncated at {self.episode_steps} steps")

        done = custom_done or done

        processed_frame = self._preprocess_frame(observation)
        self.frame_buffer.append(processed_frame)

        button_features = self.discrete_actions.get_button_features(discrete_action)
        vector_features = self.strategic_tracker.update(info, button_features)
        self.vector_features_history.append(vector_features)

        self._update_enhanced_stats()
        info.update(self.stats)

        return self._get_observation(), custom_reward, done, truncated, info

    def _calculate_stabilized_reward(self, curr_player_health, curr_opponent_health):
        """STABILITY FIX: Properly scaled and normalized reward function to prevent value loss explosion."""
        reward, done = 0.0, False

        # Win/Loss rewards - scaled down for stability
        if curr_player_health <= 0 or curr_opponent_health <= 0:
            self.total_rounds += 1
            if curr_opponent_health <= 0 < curr_player_health:
                self.wins += 1
                # STABILITY FIX: Much smaller win reward to prevent value function instability
                win_bonus = 10.0 + (curr_player_health / self.full_hp) * 5.0
                reward += win_bonus
                print(
                    f"🏆 AI WON! Total: {self.wins}W/{self.losses}L (Round {self.total_rounds})"
                )
            else:
                self.losses += 1
                # STABILITY FIX: Small loss penalty instead of large negative reward
                reward -= 5.0
                print(
                    f"💀 AI LOST! Total: {self.wins}W/{self.losses}L (Round {self.total_rounds})"
                )
            done = True

        # Damage-based rewards - scaled down for stability
        damage_dealt = max(0, self.prev_opponent_health - curr_opponent_health)
        damage_received = max(0, self.prev_player_health - curr_player_health)

        # STABILITY FIX: Much smaller damage rewards to prevent value explosion
        reward += (damage_dealt * 0.1) - (damage_received * 0.05)

        self.total_damage_dealt += damage_dealt
        self.total_damage_received += damage_received

        # Strategic bonuses - very small for stability
        osc_tracker = self.strategic_tracker.oscillation_tracker
        rolling_freq = osc_tracker.get_rolling_window_frequency()
        if 1.0 <= rolling_freq <= 3.0:
            reward += 0.01
        if osc_tracker.space_control_score > 0:
            reward += osc_tracker.space_control_score * 0.005

        # STABILITY FIX: Small step penalty to encourage efficiency
        reward -= 0.001

        # STABILITY FIX: Apply reward scaling and clipping
        reward *= self.reward_scale
        reward = np.clip(reward, -2.0, 2.0)

        self.prev_player_health, self.prev_opponent_health = (
            curr_player_health,
            curr_opponent_health,
        )

        if done:
            self.episode_rewards.append(reward)

        return reward, done

    def _get_observation(self):
        visual_obs = np.concatenate(list(self.frame_buffer), axis=2).transpose(2, 0, 1)
        vector_obs = np.stack(list(self.vector_features_history))
        return {"visual_obs": visual_obs, "vector_obs": vector_obs}

    def _preprocess_frame(self, frame):
        if frame is None:
            return np.zeros((*self.target_size, 3), dtype=np.uint8)
        return cv2.resize(frame, (self.target_size[1], self.target_size[0]))

    def _update_enhanced_stats(self):
        total_games = self.wins + self.losses
        win_rate = self.wins / total_games if total_games > 0 else 0.0
        avg_damage_per_round = self.total_damage_dealt / max(1, self.total_rounds)
        defensive_efficiency = self.total_damage_dealt / max(
            1, self.total_damage_dealt + self.total_damage_received
        )
        damage_ratio = self.total_damage_dealt / max(1, self.total_damage_received)
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
            }
        )


# --- GRADIENT FLOW VERIFICATION ---


def verify_gradient_flow(model, env, device=None):
    """Verify gradient flow with focus on training stability."""
    print("\n🔬 STABILIZED Gradient Flow Verification")
    print("=" * 50)

    if device is None:
        device = next(model.policy.parameters()).device

    # Get sample observation
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    # Convert to tensors
    obs_tensor = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            obs_tensor[key] = torch.from_numpy(value).unsqueeze(0).float().to(device)
        else:
            obs_tensor[key] = torch.tensor(value).unsqueeze(0).float().to(device)

    # Check vector features for stability
    vector_obs = obs_tensor["vector_obs"]
    print(f"🔍 Vector Feature Stability:")
    print(f"   - Shape: {vector_obs.shape}")
    print(f"   - Range: {vector_obs.min().item():.3f} to {vector_obs.max().item():.3f}")

    if vector_obs.abs().max() > 10.0:
        print("   ⚠️  WARNING: Large vector values detected!")
    else:
        print("   ✅ Vector features in stable range")

    model.policy.train()
    for param in model.policy.parameters():
        param.requires_grad = True

    # Test forward pass
    try:
        actions, values, log_probs = model.policy(obs_tensor)
        print(f"✅ Policy forward pass successful")
        print(f"   - Value output: {values.item():.3f}")

        if abs(values.item()) > 100.0:
            print("   🚨 CRITICAL: Value function output too large!")
            return False
        else:
            print("   ✅ Value function output is stable")

    except Exception as e:
        print(f"❌ Policy forward pass failed: {e}")
        return False

    # Test backward pass
    loss = values.mean() + log_probs.mean() * 0.1
    model.policy.zero_grad()

    try:
        loss.backward()
        print("✅ Backward pass successful")
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        return False

    # Analyze gradients
    total_params = 0
    params_with_grads = 0
    total_grad_norm = 0.0

    for name, param in model.policy.named_parameters():
        total_params += param.numel()
        if param.grad is not None:
            params_with_grads += param.numel()
            total_grad_norm += param.grad.norm().item()

    coverage = (params_with_grads / total_params) * 100
    avg_grad_norm = total_grad_norm / max(params_with_grads, 1)

    print(f"📊 Gradient Analysis:")
    print(f"   - Coverage: {coverage:.1f}%")
    print(f"   - Average norm: {avg_grad_norm:.6f}")

    if coverage > 95 and avg_grad_norm < 10.0:
        print("✅ EXCELLENT: Stable gradient flow ready for training!")
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
]
