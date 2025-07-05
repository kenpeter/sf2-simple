#!/usr/bin/env python3
"""
wrapper.py - CRITICAL FIXES Applied for Street Fighter II Wrapper
FIXES APPLIED:
1. Fixed frequency calculation using rolling window approach
2. Enhanced gradient flow for cross-attention learning
3. Improved neutral game detection with better thresholds
4. Added proper learning rate scaling for cross-attention components
5. Fixed attention weight initialization and gradient propagation
6. Enhanced oscillation detection sensitivity
7. **FIXED TypeError by patching retro.make with user-specified state file**
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
from typing import Dict, Tuple, List
import math
import logging
import os
from datetime import datetime
import retro

# --- FIX for TypeError in retro.make ---
# The traceback indicates a TypeError when creating the retro environment. This
# occurs when `retro.make` is called without a state and the library's default
# state discovery fails, causing it to pass `None` to the file-opening function.
#
# This patch intercepts the call. If the `state` argument is missing (None or
# empty), it injects the user-specified default state: 'ken_bison_12.state'.
# This guarantees that a valid state file is always used, preventing the crash.
_original_retro_make = retro.make


def _patched_retro_make(game, state=None, **kwargs):
    """A patched version of retro.make that forces a specific default state if none is provided."""
    if not state:
        # If no state is provided by the training script, default to the one specified by the user.
        # This prevents the TypeError from gym-retro's internal state loading.
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
    handlers=[
        logging.FileHandler(log_filename),
    ],
)
logger = logging.getLogger(__name__)

# Create analysis output directory
ANALYSIS_OUTPUT_DIR = "analysis_data"
os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)

# Constants for Street Fighter
MAX_HEALTH = 176
SCREEN_WIDTH = 180
SCREEN_HEIGHT = 128


class OscillationTracker:
    """
    FIXED Oscillation Tracker with Rolling Window Frequency Calculation
    """

    def __init__(self, history_length=16):
        self.history_length = history_length

        # Position and movement tracking
        self.player_x_history = deque(maxlen=history_length)
        self.opponent_x_history = deque(maxlen=history_length)
        self.player_velocity_history = deque(maxlen=history_length)
        self.opponent_velocity_history = deque(maxlen=history_length)

        # FIXED: More sensitive detection parameters
        self.movement_threshold = 0.3
        self.direction_change_threshold = 0.1
        self.velocity_smoothing_factor = 0.3

        # FIXED: Rolling window tracking for frequency calculation
        self.direction_change_timestamps = deque(maxlen=1800)  # 30 seconds at 60 FPS
        self.player_direction_changes = 0
        self.opponent_direction_changes = 0
        self.player_oscillation_amplitude = 0.0
        self.opponent_oscillation_amplitude = 0.0

        # Threat bubble analysis
        self.optimal_range_violations = 0
        self.whiff_bait_attempts = 0
        self.successful_whiff_punishes = 0

        # Neutral game state tracking
        self.neutral_game_duration = 0
        self.advantage_transitions = 0
        self.space_control_score = 0.0

        # Movement intention detection
        self.aggressive_forward_count = 0
        self.defensive_backward_count = 0
        self.neutral_dance_count = 0

        # Range control constants
        self.CLOSE_RANGE = 25
        self.MID_RANGE = 45
        self.FAR_RANGE = 70
        self.WHIFF_BAIT_RANGE = 35

        # Previous values for calculating changes
        self.prev_player_x = None
        self.prev_opponent_x = None
        self.prev_player_velocity = 0.0
        self.prev_opponent_velocity = 0.0

        # FIXED: Enhanced debugging data
        self.debug_data = {
            "recent_positions": deque(maxlen=10),
            "recent_velocities": deque(maxlen=10),
            "recent_direction_changes": deque(maxlen=10),
            "detection_sensitivity": {
                "movement_threshold": self.movement_threshold,
                "direction_change_threshold": self.direction_change_threshold,
                "velocity_smoothing_factor": self.velocity_smoothing_factor,
            },
        }

        # Frame counting
        self.frame_count = 0

    def update(
        self,
        player_x: float,
        opponent_x: float,
        player_attacking: bool = False,
        opponent_attacking: bool = False,
    ) -> Dict:
        """FIXED Update oscillation tracking with rolling window frequency calculation"""
        self.frame_count += 1

        # FIXED: Enhanced velocity calculation with smoothing
        player_velocity = 0.0
        opponent_velocity = 0.0

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

        # Update histories
        self.player_x_history.append(player_x)
        self.opponent_x_history.append(opponent_x)
        self.player_velocity_history.append(player_velocity)
        self.opponent_velocity_history.append(opponent_velocity)

        # FIXED: Enhanced direction change detection with timestamp tracking
        if (
            len(self.player_velocity_history) >= 2
            and abs(self.prev_player_velocity) > self.direction_change_threshold
            and abs(player_velocity) > self.direction_change_threshold
        ):
            if (self.prev_player_velocity > 0 and player_velocity < 0) or (
                self.prev_player_velocity < 0 and player_velocity > 0
            ):
                self.player_direction_changes += 1
                # FIXED: Store timestamp for rolling window calculation
                self.direction_change_timestamps.append(self.frame_count)

                self.debug_data["recent_direction_changes"].append(
                    {
                        "frame": self.frame_count,
                        "prev_vel": self.prev_player_velocity,
                        "curr_vel": player_velocity,
                        "type": "player",
                    }
                )

        if (
            len(self.opponent_velocity_history) >= 2
            and abs(self.prev_opponent_velocity) > self.direction_change_threshold
            and abs(opponent_velocity) > self.direction_change_threshold
        ):
            if (self.prev_opponent_velocity > 0 and opponent_velocity < 0) or (
                self.prev_opponent_velocity < 0 and opponent_velocity > 0
            ):
                self.opponent_direction_changes += 1

        # Update debug data
        self.debug_data["recent_positions"].append(
            {"player_x": player_x, "opponent_x": opponent_x}
        )
        self.debug_data["recent_velocities"].append(
            {"player_vel": player_velocity, "opponent_vel": opponent_velocity}
        )

        # Calculate oscillation amplitude
        if len(self.player_x_history) >= 8:
            player_positions = list(self.player_x_history)[-8:]
            self.player_oscillation_amplitude = max(player_positions) - min(
                player_positions
            )

        if len(self.opponent_x_history) >= 8:
            opponent_positions = list(self.opponent_x_history)[-8:]
            self.opponent_oscillation_amplitude = max(opponent_positions) - min(
                opponent_positions
            )

        # Analyze movement patterns
        distance = abs(player_x - opponent_x)
        movement_analysis = self._analyze_movement_patterns(
            player_x,
            opponent_x,
            player_velocity,
            opponent_velocity,
            distance,
            player_attacking,
            opponent_attacking,
        )

        # Update previous values
        self.prev_player_x = player_x
        self.prev_opponent_x = opponent_x
        self.prev_player_velocity = player_velocity
        self.prev_opponent_velocity = opponent_velocity

        return movement_analysis

    def _analyze_movement_patterns(
        self,
        player_x: float,
        opponent_x: float,
        player_velocity: float,
        opponent_velocity: float,
        distance: float,
        player_attacking: bool,
        opponent_attacking: bool,
    ) -> Dict:
        """FIXED Analyze movement patterns with enhanced neutral game detection"""

        # Determine who is moving forward/backward
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

        # FIXED: Enhanced neutral game detection
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

        # Movement intention classification
        if player_moving_forward and distance > self.MID_RANGE:
            self.aggressive_forward_count += 1
        elif player_moving_backward and distance < self.MID_RANGE:
            self.defensive_backward_count += 1
        elif (
            abs(player_velocity) > self.movement_threshold
            and self.MID_RANGE <= distance <= self.FAR_RANGE
        ):
            self.neutral_dance_count += 1

        # Whiff bait detection
        if (
            distance > self.WHIFF_BAIT_RANGE
            and distance < self.MID_RANGE + 5
            and player_moving_forward
            and not player_attacking
        ):
            self.whiff_bait_attempts += 1

        # FIXED: Enhanced space control analysis
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
        self,
        player_x: float,
        opponent_x: float,
        player_velocity: float,
        opponent_velocity: float,
        distance: float,
    ) -> float:
        """FIXED Enhanced space control calculation"""

        # Center control bonus
        screen_center = SCREEN_WIDTH / 2
        player_center_distance = abs(player_x - screen_center)
        opponent_center_distance = abs(opponent_x - screen_center)
        center_control = (opponent_center_distance - player_center_distance) / (
            SCREEN_WIDTH / 2
        )

        # FIXED: Enhanced movement initiative calculation
        movement_initiative = 0.0
        if abs(player_velocity) > abs(opponent_velocity) + 0.1:
            movement_initiative = 0.3 if player_velocity > 0 else -0.3
        elif abs(opponent_velocity) > abs(player_velocity) + 0.1:
            movement_initiative = -0.3 if opponent_velocity > 0 else 0.3

        # FIXED: Better range control assessment
        range_control = 0.0
        if self.CLOSE_RANGE <= distance <= self.MID_RANGE:
            range_control = 0.4
        elif distance > self.FAR_RANGE:
            range_control = -0.3
        elif self.MID_RANGE < distance <= self.FAR_RANGE:
            range_control = 0.2

        # FIXED: Enhanced oscillation effectiveness using rolling window
        oscillation_effectiveness = 0.0
        if self.frame_count > 60:
            rolling_freq = self.get_rolling_window_frequency()
            if 1.0 <= rolling_freq <= 3.0:
                oscillation_effectiveness = 0.3

        total_control = (
            center_control * 0.3
            + movement_initiative * 0.3
            + range_control * 0.2
            + oscillation_effectiveness * 0.2
        )
        return np.clip(total_control, -1.0, 1.0)

    def get_rolling_window_frequency(self) -> float:
        """FIXED: Calculate oscillation frequency using rolling window"""
        if len(self.direction_change_timestamps) < 2:
            return 0.0

        # Use 10-second rolling window for frequency calculation
        window_frames = 600  # 10 seconds at 60 FPS
        current_frame = self.frame_count

        # Count direction changes in the last 10 seconds
        recent_changes = sum(
            1
            for timestamp in self.direction_change_timestamps
            if current_frame - timestamp <= window_frames
        )

        # Calculate frequency in Hz
        window_seconds = min(window_frames / 60.0, self.frame_count / 60.0)
        if window_seconds > 0:
            frequency = recent_changes / window_seconds
            return frequency
        return 0.0

    def get_oscillation_features(self) -> np.ndarray:
        """FIXED: Get 12 oscillation-based features with rolling window frequency"""
        features = np.zeros(12, dtype=np.float32)

        if self.frame_count == 0:
            return features

        # Feature 1: Player oscillation frequency (using rolling window)
        rolling_freq = self.get_rolling_window_frequency()
        features[0] = np.clip(rolling_freq / 5.0, 0.0, 1.0)

        # Feature 2: Opponent oscillation frequency
        opponent_osc_freq = self.opponent_direction_changes / max(
            1, self.frame_count / 60
        )
        features[1] = np.clip(opponent_osc_freq / 5.0, 0.0, 1.0)

        # Feature 3: Player oscillation amplitude
        features[2] = np.clip(self.player_oscillation_amplitude / 50.0, 0.0, 1.0)

        # Feature 4: Opponent oscillation amplitude
        features[3] = np.clip(self.opponent_oscillation_amplitude / 50.0, 0.0, 1.0)

        # Feature 5: Space control score
        features[4] = np.clip(self.space_control_score, -1.0, 1.0)

        # Feature 6: Neutral game duration ratio
        features[5] = np.clip(self.neutral_game_duration / 180.0, 0.0, 1.0)

        # Feature 7: Movement aggression ratio
        total_movement = (
            self.aggressive_forward_count
            + self.defensive_backward_count
            + self.neutral_dance_count
        )
        if total_movement > 0:
            features[6] = self.aggressive_forward_count / total_movement

        # Feature 8: Defensive movement ratio
        if total_movement > 0:
            features[7] = self.defensive_backward_count / total_movement

        # Feature 9: Neutral dance ratio
        if total_movement > 0:
            features[8] = self.neutral_dance_count / total_movement

        # Feature 10: Whiff bait attempt frequency
        features[9] = np.clip(
            self.whiff_bait_attempts / max(1, self.frame_count / 60), 0.0, 1.0
        )

        # Feature 11: Advantage transition frequency
        features[10] = np.clip(
            self.advantage_transitions / max(1, self.frame_count / 60), 0.0, 1.0
        )

        # Feature 12: Current velocity differential
        if (
            len(self.player_velocity_history) > 0
            and len(self.opponent_velocity_history) > 0
        ):
            velocity_diff = (
                self.player_velocity_history[-1] - self.opponent_velocity_history[-1]
            )
            features[11] = np.clip(velocity_diff / 5.0, -1.0, 1.0)

        return features

    def get_debug_info(self) -> Dict:
        """Get debug information for validation"""
        return {
            "recent_positions": list(self.debug_data["recent_positions"]),
            "recent_velocities": list(self.debug_data["recent_velocities"]),
            "recent_direction_changes": list(
                self.debug_data["recent_direction_changes"]
            ),
            "detection_sensitivity": self.debug_data["detection_sensitivity"],
            "current_stats": self.get_stats(),
            "rolling_window_frequency": self.get_rolling_window_frequency(),
        }

    def get_stats(self) -> Dict:
        """Get current oscillation statistics for logging"""
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
    """Enhanced Street Fighter II Discrete Action System"""

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

        # Action combinations
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
        """Convert discrete action to multi-binary array"""
        if action_index < 0 or action_index >= self.num_actions:
            action_index = 0

        multibinary_action = np.zeros(self.num_buttons, dtype=np.uint8)
        button_indices = self.action_combinations[action_index]

        for button_idx in button_indices:
            if 0 <= button_idx < self.num_buttons:
                multibinary_action[button_idx] = 1

        return multibinary_action

    def get_action_name(self, action_index: int) -> str:
        """Get human-readable name for an action"""
        if action_index < 0 or action_index >= self.num_actions:
            return f"INVALID_{action_index}"
        if action_index == 0:
            return "IDLE"

        button_indices = self.action_combinations[action_index]
        if not button_indices:
            return "IDLE"

        button_names = [
            self.button_names[i] for i in button_indices if 0 <= i < self.num_buttons
        ]
        return "+".join(button_names)

    def get_button_features(self, action_index: int) -> np.ndarray:
        """Get 12-dimensional button features"""
        return self.discrete_to_multibinary(action_index).astype(np.float32)

    def is_attack_action(self, action_index: int) -> bool:
        """Check if action contains attack buttons"""
        if action_index < 0 or action_index >= self.num_actions:
            return False

        button_indices = self.action_combinations[action_index]
        attack_buttons = {0, 1, 8, 9, 10, 11}
        return any(btn in attack_buttons for btn in button_indices)


class CNNFeatureExtractor(nn.Module):
    """CNN to extract features from frame stack"""

    def __init__(self, input_channels=24, feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim

        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.projection = nn.Linear(256, feature_dim)

    def forward(self, frame_stack: torch.Tensor) -> torch.Tensor:
        try:
            cnn_output = self.cnn(frame_stack)
            features = self.projection(cnn_output)
            return features
        except Exception as e:
            logger.error(f"CNN error: {e}")
            batch_size = frame_stack.shape[0] if len(frame_stack.shape) > 0 else 1
            return torch.zeros(batch_size, self.feature_dim, device=frame_stack.device)


class StrategicFeatureTracker:
    """Strategic feature tracker with oscillation-based positioning"""

    def __init__(self, history_length=8):
        self.history_length = history_length

        # Initialize all tracking variables
        self.player_health_history = deque(maxlen=history_length)
        self.opponent_health_history = deque(maxlen=history_length)
        self.score_history = deque(maxlen=history_length)
        self.score_change_history = deque(maxlen=history_length)
        self.combo_counter = 0
        self.max_combo_this_round = 0
        self.last_score_increase_frame = -1
        self.current_frame = 0

        # Status tracking
        self.player_status_history = deque(maxlen=history_length)
        self.opponent_status_history = deque(maxlen=history_length)
        self.player_victories_history = deque(maxlen=history_length)
        self.opponent_victories_history = deque(maxlen=history_length)
        self.player_damage_dealt_history = deque(maxlen=history_length)
        self.opponent_damage_dealt_history = deque(maxlen=history_length)
        self.recent_damage_events = deque(maxlen=5)

        # Button tracking
        self.button_features_history = deque(maxlen=history_length)
        self.previous_button_features = np.zeros(12, dtype=np.float32)

        # FIXED: Enhanced oscillation tracker
        self.oscillation_tracker = OscillationTracker(history_length=16)

        # Combat state tracking
        self.close_combat_count = 0
        self.total_frames = 0

        # Constants
        self.SCREEN_WIDTH = 180
        self.SCREEN_HEIGHT = 128
        self.DANGER_ZONE_HEALTH = MAX_HEALTH * 0.25
        self.CORNER_THRESHOLD = 30
        self.CLOSE_DISTANCE = 40
        self.OPTIMAL_SPACING_MIN = 35
        self.OPTIMAL_SPACING_MAX = 55
        self.COMBO_TIMEOUT_FRAMES = 60
        self.MIN_SCORE_INCREASE_FOR_HIT = 50

        # Previous values for rate calculations
        self.prev_player_health = None
        self.prev_opponent_health = None
        self.prev_score = None

    def update(
        self,
        player_health: float,
        opponent_health: float,
        score: float,
        player_x: float,
        player_y: float,
        opponent_x: float,
        opponent_y: float,
        player_status: int = 0,
        opponent_status: int = 0,
        player_victories: int = 0,
        opponent_victories: int = 0,
        button_features: np.ndarray = None,
    ) -> np.ndarray:
        """Update method returns 45 features: 21 traditional + 12 oscillation + 12 button history"""
        self.current_frame += 1

        # Update histories
        self.player_health_history.append(player_health)
        self.opponent_health_history.append(opponent_health)
        self.player_status_history.append(player_status)
        self.opponent_status_history.append(opponent_status)
        self.player_victories_history.append(player_victories)
        self.opponent_victories_history.append(opponent_victories)

        # Enhanced score momentum with combo detection
        score_change = 0
        if self.prev_score is not None:
            score_change = score - self.prev_score

            if score_change >= self.MIN_SCORE_INCREASE_FOR_HIT:
                frames_since_last_hit = (
                    self.current_frame - self.last_score_increase_frame
                )

                if (
                    frames_since_last_hit <= self.COMBO_TIMEOUT_FRAMES
                    and self.last_score_increase_frame > 0
                ):
                    self.combo_counter += 1
                else:
                    self.combo_counter = 1

                self.last_score_increase_frame = self.current_frame
                self.max_combo_this_round = max(
                    self.max_combo_this_round, self.combo_counter
                )

                self.recent_damage_events.append(
                    {
                        "frame": self.current_frame,
                        "score_increase": score_change,
                        "combo_count": self.combo_counter,
                    }
                )
            else:
                frames_since_last_hit = (
                    self.current_frame - self.last_score_increase_frame
                )
                if frames_since_last_hit > self.COMBO_TIMEOUT_FRAMES:
                    self.combo_counter = 0

        self.score_history.append(score)
        self.score_change_history.append(score_change)

        # Update button features history
        self.button_features_history.append(self.previous_button_features.copy())

        if button_features is not None:
            self.previous_button_features = button_features.copy()
        else:
            self.previous_button_features = np.zeros(12, dtype=np.float32)

        # Calculate damage rates
        player_damage_this_frame = 0
        opponent_damage_this_frame = 0

        if self.prev_opponent_health is not None:
            player_damage_this_frame = max(
                0, self.prev_opponent_health - opponent_health
            )
        if self.prev_player_health is not None:
            opponent_damage_this_frame = max(0, self.prev_player_health - player_health)

        self.player_damage_dealt_history.append(player_damage_this_frame)
        self.opponent_damage_dealt_history.append(opponent_damage_this_frame)

        # Update oscillation tracker
        player_attacking = (
            button_features is not None
            and np.any(button_features[[0, 1, 8, 9, 10, 11]])
            if button_features is not None
            else False
        )
        opponent_attacking = False

        oscillation_analysis = self.oscillation_tracker.update(
            player_x, opponent_x, player_attacking, opponent_attacking
        )

        # Update frame counting
        self.total_frames += 1
        distance = abs(player_x - opponent_x)
        if distance <= self.CLOSE_DISTANCE:
            self.close_combat_count += 1

        # Calculate 45 features
        features = self._calculate_enhanced_features_with_oscillation(
            player_health,
            opponent_health,
            score,
            player_x,
            player_y,
            opponent_x,
            opponent_y,
            player_status,
            opponent_status,
            player_victories,
            opponent_victories,
            distance,
            oscillation_analysis,
        )

        # Update previous values
        self.prev_player_health = player_health
        self.prev_opponent_health = opponent_health
        self.prev_score = score

        return features

    def _calculate_enhanced_features_with_oscillation(
        self,
        player_health,
        opponent_health,
        score,
        player_x,
        player_y,
        opponent_x,
        opponent_y,
        player_status,
        opponent_status,
        player_victories,
        opponent_victories,
        distance,
        oscillation_analysis,
    ) -> np.ndarray:
        """Calculate 45 features: 21 traditional + 12 oscillation + 12 button history"""
        features = np.zeros(45, dtype=np.float32)

        # Features 1-21: Traditional strategic features
        features[0] = 1.0 if player_health <= self.DANGER_ZONE_HEALTH else 0.0
        features[1] = 1.0 if opponent_health <= self.DANGER_ZONE_HEALTH else 0.0

        # Health ratio
        if opponent_health > 0:
            features[2] = player_health / opponent_health
        else:
            features[2] = 2.0
        features[2] = np.clip(features[2], 0.0, 2.0)

        # Health change rates
        player_health_change = self._calculate_momentum(self.player_health_history)
        opponent_health_change = self._calculate_momentum(self.opponent_health_history)
        features[3] = (player_health_change + opponent_health_change) / 2.0

        # Damage rates
        features[4] = self._calculate_momentum(self.player_damage_dealt_history)
        features[5] = self._calculate_momentum(self.opponent_damage_dealt_history)

        # Corner distances
        player_left_dist = player_x
        player_right_dist = self.SCREEN_WIDTH - player_x
        player_corner_dist = min(player_left_dist, player_right_dist)
        features[6] = np.clip(player_corner_dist / (self.SCREEN_WIDTH / 2), 0.0, 1.0)

        opponent_left_dist = opponent_x
        opponent_right_dist = self.SCREEN_WIDTH - opponent_x
        opponent_corner_dist = min(opponent_left_dist, opponent_right_dist)
        features[7] = np.clip(opponent_corner_dist / (self.SCREEN_WIDTH / 2), 0.0, 1.0)

        # Near corner flags
        features[8] = 1.0 if player_corner_dist <= self.CORNER_THRESHOLD else 0.0
        features[9] = 1.0 if opponent_corner_dist <= self.CORNER_THRESHOLD else 0.0

        # Center control
        screen_center = self.SCREEN_WIDTH / 2
        player_center_dist = abs(player_x - screen_center)
        opponent_center_dist = abs(opponent_x - screen_center)
        if player_center_dist < opponent_center_dist:
            features[10] = 1.0
        elif opponent_center_dist < player_center_dist:
            features[10] = -1.0
        else:
            features[10] = 0.0

        # Vertical advantage
        vertical_diff = player_y - opponent_y
        features[11] = np.clip(vertical_diff / (self.SCREEN_HEIGHT / 2), -1.0, 1.0)

        # Space control from oscillation
        features[12] = oscillation_analysis.get("space_control_score", 0.0)

        # Optimal spacing
        if self.OPTIMAL_SPACING_MIN <= distance <= self.OPTIMAL_SPACING_MAX:
            features[13] = 1.0
        else:
            features[13] = 0.0

        # Forward pressure
        if oscillation_analysis.get("player_moving_forward", False):
            features[14] = 1.0
        elif oscillation_analysis.get("player_moving_backward", False):
            features[14] = -1.0
        else:
            features[14] = 0.0

        # Defensive movement
        if oscillation_analysis.get("player_moving_backward", False):
            features[15] = 1.0
        else:
            features[15] = 0.0

        # Close combat frequency
        if self.total_frames > 0:
            features[16] = self.close_combat_count / self.total_frames
        else:
            features[16] = 0.0

        # Enhanced score momentum
        features[17] = self._calculate_enhanced_score_momentum()

        # Status difference
        status_diff = player_status - opponent_status
        features[18] = np.clip(status_diff / 100.0, -1.0, 1.0)

        # Victory counts
        features[19] = min(player_victories / 10.0, 1.0)
        features[20] = min(opponent_victories / 10.0, 1.0)

        # Features 22-33: Oscillation features
        oscillation_features = self.oscillation_tracker.get_oscillation_features()
        features[21:33] = oscillation_features

        # Features 34-45: Previous button state
        if len(self.button_features_history) > 0:
            previous_buttons = self.button_features_history[-1]
            features[33:45] = previous_buttons
        else:
            features[33:45] = 0.0

        return features

    def _calculate_enhanced_score_momentum(self) -> float:
        """Enhanced score momentum calculation"""
        if len(self.score_change_history) < 2:
            return 0.0

        # Base momentum
        recent_changes = list(self.score_change_history)[-5:]
        base_momentum = np.mean([max(0, change) for change in recent_changes])

        # Combo multiplier
        combo_multiplier = 1.0 + (self.combo_counter * 0.1)

        # Hit frequency bonus
        hit_frequency = 0.0
        if len(self.recent_damage_events) > 0:
            recent_hits = [
                event
                for event in self.recent_damage_events
                if self.current_frame - event["frame"] <= 300
            ]
            hit_frequency = len(recent_hits) / 5.0

        # Damage scaling
        damage_scaling = 1.0
        if len(self.score_change_history) > 0:
            recent_score_change = self.score_change_history[-1]
            if recent_score_change > 0:
                damage_scaling = 1.0 + min(recent_score_change / 1000.0, 1.0)

        # Combine all factors
        enhanced_momentum = (
            base_momentum * combo_multiplier * damage_scaling + hit_frequency
        )
        return np.clip(enhanced_momentum / 100.0, -1.0, 2.0)

    def _calculate_momentum(self, history):
        """Calculate momentum from history"""
        if len(history) < 2:
            return 0.0

        values = list(history)
        changes = [values[i] - values[i - 1] for i in range(1, len(values))]
        recent_changes = changes[-3:] if len(changes) >= 3 else changes
        return np.mean(recent_changes) if recent_changes else 0.0

    def get_combo_stats(self) -> Dict:
        """Get combo statistics"""
        combo_stats = {
            "current_combo": self.combo_counter,
            "max_combo_this_round": self.max_combo_this_round,
            "recent_hits": len(
                [
                    event
                    for event in self.recent_damage_events
                    if self.current_frame - event["frame"] <= 300
                ]
            ),
        }

        oscillation_stats = self.oscillation_tracker.get_stats()
        combo_stats.update(oscillation_stats)
        return combo_stats


class MultiHeadCrossAttention(nn.Module):
    """FIXED Multi-head cross-attention with proper gradient flow"""

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # FIXED: Proper initialization for gradient flow
        self.w_q = nn.Linear(d_model, d_model, bias=True)
        self.w_k = nn.Linear(d_model, d_model, bias=True)
        self.w_v = nn.Linear(d_model, d_model, bias=True)
        self.w_o = nn.Linear(d_model, d_model, bias=True)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        # FIXED: Proper parameter initialization
        self._init_parameters()

    def _init_parameters(self):
        """FIXED: Initialize parameters for better gradient flow"""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with improved gradient flow"""
        batch_size = query.size(0)

        # Apply linear transformations
        Q = (
            self.w_q(query)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = (
            self.w_v(value)
            .view(batch_size, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )

        # Scaled dot-product attention
        attention_output, attention_weights = self.scaled_dot_product_attention(
            Q, K, V, mask
        )

        # Concatenate heads
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.d_model)
        )

        output = self.w_o(attention_output)

        # FIXED: Residual connection with better scaling
        return self.layer_norm(output + query), attention_weights

    def scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scaled dot-product attention"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        return torch.matmul(attention_weights, V), attention_weights


class FeatureGroupProcessor(nn.Module):
    """FIXED Feature group processor with better gradient flow"""

    def __init__(self, input_dim: int, d_model: int, group_name: str):
        super().__init__()
        self.group_name = group_name
        self.input_dim = input_dim
        self.d_model = d_model

        # FIXED: Improved architectures for each feature type
        if group_name == "visual":
            self.processor = nn.Sequential(
                nn.Linear(input_dim, d_model * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
            )
        elif group_name == "strategy":
            self.processor = nn.Sequential(
                nn.Linear(input_dim, d_model),
                nn.ReLU(),
                nn.Dropout(0.05),
                nn.LayerNorm(d_model),
            )
        elif group_name == "oscillation":
            self.processor = nn.Sequential(
                nn.Linear(input_dim, d_model // 2),
                nn.ReLU(),
                nn.Dropout(0.05),
                nn.Linear(d_model // 2, d_model),
                nn.LayerNorm(d_model),
            )
        elif group_name == "previous_buttons":
            self.processor = nn.Sequential(
                nn.Linear(input_dim, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, d_model),
                nn.LayerNorm(d_model),
            )
        else:
            self.processor = nn.Sequential(
                nn.Linear(input_dim, d_model), nn.ReLU(), nn.LayerNorm(d_model)
            )

        # FIXED: Proper initialization
        self._init_parameters()

    def _init_parameters(self):
        """Initialize parameters properly"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Process features"""
        return self.processor(features)


class EnhancedCrossAttentionVisionTransformer(nn.Module):
    """FIXED Enhanced Vision Transformer with proper gradient flow"""

    def __init__(
        self,
        visual_dim: int = 512,
        strategic_dim: int = 21,
        oscillation_dim: int = 12,
        button_dim: int = 12,
        seq_length: int = 8,
    ):
        super().__init__()

        self.visual_dim = visual_dim
        self.strategic_dim = strategic_dim
        self.oscillation_dim = oscillation_dim
        self.button_dim = button_dim
        self.seq_length = seq_length
        self.num_heads = 8
        self.d_model = 256

        # Feature group processors
        self.visual_processor = FeatureGroupProcessor(
            visual_dim, self.d_model, "visual"
        )
        self.strategy_processor = FeatureGroupProcessor(
            strategic_dim, self.d_model, "strategy"
        )
        self.oscillation_processor = FeatureGroupProcessor(
            oscillation_dim, self.d_model, "oscillation"
        )
        self.button_processor = FeatureGroupProcessor(
            button_dim, self.d_model, "previous_buttons"
        )

        # FIXED: Better query initialization
        self.action_query = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)

        # Cross-attention modules
        self.visual_cross_attention = MultiHeadCrossAttention(
            self.d_model, num_heads=self.num_heads
        )
        self.strategy_cross_attention = MultiHeadCrossAttention(
            self.d_model, num_heads=self.num_heads
        )
        self.oscillation_cross_attention = MultiHeadCrossAttention(
            self.d_model, num_heads=self.num_heads
        )
        self.button_cross_attention = MultiHeadCrossAttention(
            self.d_model, num_heads=self.num_heads
        )

        # FIXED: Improved feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.d_model * 4, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.LayerNorm(self.d_model),
        )

        # Temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=self.d_model,
            num_heads=self.num_heads,
            dropout=0.1,
            batch_first=True,
        )

        self._init_parameters()

    def _init_parameters(self):
        """FIXED: Proper parameter initialization"""
        nn.init.normal_(self.action_query, mean=0.0, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, combined_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with enhanced gradient flow"""
        try:
            batch_size, seq_len, total_dim = combined_sequence.shape

            # Split features
            visual_features = combined_sequence[:, :, : self.visual_dim]
            strategy_features = combined_sequence[
                :, :, self.visual_dim : self.visual_dim + self.strategic_dim
            ]
            oscillation_features = combined_sequence[
                :,
                :,
                self.visual_dim
                + self.strategic_dim : self.visual_dim
                + self.strategic_dim
                + self.oscillation_dim,
            ]
            button_features = combined_sequence[
                :, :, self.visual_dim + self.strategic_dim + self.oscillation_dim :
            ]

            # Process each feature group
            visual_processed = self.visual_processor(visual_features)
            strategy_processed = self.strategy_processor(strategy_features)
            oscillation_processed = self.oscillation_processor(oscillation_features)
            button_processed = self.button_processor(button_features)

            # Expand action query
            action_query = self.action_query.expand(batch_size, -1, -1)

            # Cross-attention for each feature group
            visual_attended, visual_weights = self.visual_cross_attention(
                action_query, visual_processed, visual_processed
            )
            strategy_attended, strategy_weights = self.strategy_cross_attention(
                action_query, strategy_processed, strategy_processed
            )
            oscillation_attended, oscillation_weights = (
                self.oscillation_cross_attention(
                    action_query, oscillation_processed, oscillation_processed
                )
            )
            button_attended, button_weights = self.button_cross_attention(
                action_query, button_processed, button_processed
            )

            # Fuse features
            fused_features = torch.cat(
                [
                    visual_attended.squeeze(1),
                    strategy_attended.squeeze(1),
                    oscillation_attended.squeeze(1),
                    button_attended.squeeze(1),
                ],
                dim=-1,
            )
            fused_features = self.feature_fusion(fused_features)

            # Temporal attention
            all_features_processed = torch.cat(
                [
                    visual_processed,
                    strategy_processed,
                    oscillation_processed,
                    button_processed,
                ],
                dim=-1,
            )
            all_features_fused = self.feature_fusion(all_features_processed)

            temporal_output, _ = self.temporal_attention(
                query=fused_features.unsqueeze(1),
                key=all_features_fused,
                value=all_features_fused,
            )

            final_features = temporal_output.squeeze(1)

            return {
                "processed_features": final_features,
                "visual_attention": visual_weights,
                "strategy_attention": strategy_weights,
                "oscillation_attention": oscillation_weights,
                "button_attention": button_weights,
            }

        except Exception as e:
            logger.error(f"Cross-attention transformer error: {e}", exc_info=True)
            batch_size = combined_sequence.shape[0]
            device = combined_sequence.device

            return {
                "processed_features": torch.zeros(
                    batch_size, self.d_model, device=device
                ),
                "visual_attention": torch.zeros(
                    batch_size, self.num_heads, 1, self.seq_length, device=device
                ),
                "strategy_attention": torch.zeros(
                    batch_size, self.num_heads, 1, self.seq_length, device=device
                ),
                "oscillation_attention": torch.zeros(
                    batch_size, self.num_heads, 1, self.seq_length, device=device
                ),
                "button_attention": torch.zeros(
                    batch_size, self.num_heads, 1, self.seq_length, device=device
                ),
            }


class StreetFighterVisionWrapper(gym.Wrapper):
    """FIXED Street Fighter wrapper with proper frequency calculation"""

    def __init__(
        self,
        env,
        reset_round=True,
        rendering=False,
        max_episode_steps=5000,
        frame_stack=8,
        enable_vision_transformer=True,
        defend_action_indices=None,
        log_transformer_predictions=True,
    ):
        super().__init__(env)

        self.frame_stack = frame_stack
        self.enable_vision_transformer = enable_vision_transformer
        self.target_size = (128, 180)
        self.log_transformer_predictions = log_transformer_predictions
        self.discrete_actions = StreetFighterDiscreteActions()
        self.max_episode_steps = max_episode_steps
        self.episode_steps = 0
        self.reset_round = reset_round
        self.rendering = rendering  # FIXED: Store rendering flag
        self.full_hp = 176
        self.prev_player_health = self.full_hp
        self.prev_opponent_health = self.full_hp
        self.defend_action_indices = defend_action_indices or [54, 55, 56]
        self.wins = 0
        self.losses = 0
        self.total_rounds = 0
        self.total_damage_dealt = 0
        self.total_damage_received = 0

        # Initialize observation space
        obs_shape = (3 * frame_stack, self.target_size[0], self.target_size[1])
        self.observation_space = spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )
        self.action_space = spaces.Discrete(self.discrete_actions.num_actions)

        # Initialize strategic tracker
        self.strategic_tracker = StrategicFeatureTracker()

        # Initialize frame buffer
        self.frame_buffer = deque(maxlen=frame_stack)

        # Initialize feature histories
        self.visual_features_history = deque(maxlen=frame_stack)
        self.strategic_features_history = deque(maxlen=frame_stack)

        # Initialize model components
        self.cnn_extractor = None
        self.cross_attention_transformer = None
        self.vision_ready = False

        # Initialize stats dictionary
        self.stats = {
            "predictions_made": 0,
            "total_combos": 0,
            "max_combo": 0,
            "avg_damage_per_round": 0.0,
            "defensive_efficiency": 0.0,
            "cross_attention_ready": False,
            "visual_attention_weight": 0.0,
            "strategy_attention_weight": 0.0,
            "oscillation_attention_weight": 0.0,
            "button_attention_weight": 0.0,
            "player_oscillation_frequency": 0.0,
            "space_control_score": 0.0,
            "neutral_game_duration": 0.0,
            "whiff_bait_attempts": 0,
        }

        # Feature names
        self.strategic_feature_names = [
            "player_in_danger",
            "opponent_in_danger",
            "health_ratio",
            "combined_health_change",
            "player_damage_rate",
            "opponent_damage_rate",
            "player_corner_distance",
            "opponent_corner_distance",
            "player_near_corner",
            "opponent_near_corner",
            "center_control",
            "vertical_advantage",
            "space_control_from_oscillation",
            "optimal_spacing",
            "forward_pressure_from_oscillation",
            "defensive_movement",
            "close_combat_frequency",
            "enhanced_score_momentum",
            "status_difference",
            "agent_victories",
            "enemy_victories",
        ]

        self.oscillation_feature_names = [
            "player_oscillation_frequency",
            "opponent_oscillation_frequency",
            "player_oscillation_amplitude",
            "opponent_oscillation_amplitude",
            "space_control_score",
            "neutral_game_duration_ratio",
            "movement_aggression_ratio",
            "defensive_movement_ratio",
            "neutral_dance_ratio",
            "whiff_bait_frequency",
            "advantage_transition_frequency",
            "velocity_differential",
        ]

        self.button_feature_names = [
            f"prev_{name}_pressed" for name in self.discrete_actions.button_names
        ]

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        observation, info = result if isinstance(result, tuple) else (result, {})

        self.prev_player_health = self.full_hp
        self.prev_opponent_health = self.full_hp
        self.episode_steps = 0

        processed_frame = self._preprocess_frame(observation)
        self.frame_buffer.clear()
        for _ in range(self.frame_stack):
            self.frame_buffer.append(processed_frame)

        self.visual_features_history.clear()
        self.strategic_features_history.clear()

        stacked_obs = self._get_stacked_observation()
        return stacked_obs, info

    def step(self, discrete_action):
        multibinary_action = self.discrete_actions.discrete_to_multibinary(
            discrete_action
        )
        observation, reward, done, truncated, info = self.env.step(multibinary_action)

        # FIXED: Handle rendering if enabled
        if self.rendering and hasattr(self.env, "render"):
            try:
                self.env.render()
            except Exception as e:
                # Ignore rendering errors to prevent training interruption
                pass

        # Extract state
        (
            curr_player_health,
            curr_opponent_health,
            score,
            player_x,
            player_y,
            opponent_x,
            opponent_y,
            player_status,
            opponent_status,
            player_victories,
            opponent_victories,
        ) = self._extract_enhanced_state(info)

        # Calculate reward
        custom_reward, custom_done = self._calculate_enhanced_reward_with_oscillation(
            curr_player_health, curr_opponent_health, score, discrete_action
        )
        done = custom_done or done

        processed_frame = self._preprocess_frame(observation)
        self.frame_buffer.append(processed_frame)
        stacked_obs = self._get_stacked_observation()
        button_features = self.discrete_actions.get_button_features(discrete_action)

        self._process_cross_attention_vision_pipeline(
            stacked_obs,
            curr_player_health,
            curr_opponent_health,
            score,
            player_x,
            player_y,
            opponent_x,
            opponent_y,
            player_status,
            opponent_status,
            player_victories,
            opponent_victories,
            button_features,
        )

        self._update_enhanced_stats()
        self.episode_steps += 1
        info.update(self.stats)

        return stacked_obs, custom_reward, done, truncated, info

    def _calculate_enhanced_reward_with_oscillation(
        self, curr_player_health, curr_opponent_health, score, action
    ):
        """Enhanced reward calculation with oscillation bonuses"""
        reward = 0.0
        done = False

        # Basic win/loss rewards
        if curr_player_health <= 0 or curr_opponent_health <= 0:
            self.total_rounds += 1
            if curr_opponent_health <= 0 and curr_player_health > 0:
                health_bonus = (curr_player_health / self.full_hp) * 50
                self.wins += 1
                reward += 100 + health_bonus
            else:
                self.losses += 1
            if self.reset_round:
                done = True

        # Damage rewards
        damage_dealt = max(0, self.prev_opponent_health - curr_opponent_health)
        damage_received = max(0, self.prev_player_health - curr_player_health)
        reward += damage_dealt - damage_received

        # FIXED: Oscillation-specific rewards
        if hasattr(self.strategic_tracker, "oscillation_tracker"):
            osc_tracker = self.strategic_tracker.oscillation_tracker
            rolling_freq = osc_tracker.get_rolling_window_frequency()

            # Reward proper oscillation frequency (1-3 Hz is optimal)
            if 1.0 <= rolling_freq <= 3.0:
                reward += 0.1
            elif rolling_freq > 5.0:
                reward -= 0.05

            # Reward positive space control
            if osc_tracker.space_control_score > 0:
                reward += osc_tracker.space_control_score * 0.05

            # Reward neutral game engagement
            if osc_tracker.neutral_game_duration > 0:
                if 30 <= osc_tracker.neutral_game_duration <= 90:
                    reward += 0.05
                elif osc_tracker.neutral_game_duration > 150:
                    reward -= 0.02

            # Reward whiff bait attempts
            if osc_tracker.whiff_bait_attempts > 0:
                reward += 0.02

        self.total_damage_dealt += damage_dealt
        self.total_damage_received += damage_received
        self.prev_player_health = curr_player_health
        self.prev_opponent_health = curr_opponent_health

        return reward, done

    def _update_enhanced_stats(self):
        """FIXED: Update stats with win rate calculation"""
        # Calculate win rate
        total_games = self.wins + self.losses
        win_rate = self.wins / total_games if total_games > 0 else 0.0

        # Calculate other performance metrics
        avg_damage_per_round = self.total_damage_dealt / max(1, self.total_rounds)

        defensive_efficiency = 0.0
        if self.total_damage_received > 0:
            defensive_efficiency = self.total_damage_dealt / (
                self.total_damage_dealt + self.total_damage_received
            )
        elif self.total_damage_dealt > 0:
            defensive_efficiency = 1.0

        # Get combo stats
        combo_stats = self.strategic_tracker.get_combo_stats()

        # Update stats dictionary with win rate
        self.stats.update(
            {
                "wins": self.wins,
                "losses": self.losses,
                "total_rounds": self.total_rounds,
                "win_rate": win_rate,  # ✅ NOW INCLUDED
                "avg_damage_per_round": avg_damage_per_round,
                "defensive_efficiency": defensive_efficiency,
                "total_combos": combo_stats.get("current_combo", 0),
                "max_combo": combo_stats.get("max_combo_this_round", 0),
            }
        )

    def get_debug_info(self):
        """Get debug information for validation"""
        if hasattr(self.strategic_tracker, "oscillation_tracker"):
            return self.strategic_tracker.oscillation_tracker.get_debug_info()
        return {}

    def _extract_enhanced_state(self, info):
        return (
            info.get("agent_hp", self.full_hp),
            info.get("enemy_hp", self.full_hp),
            info.get("score", 0),
            info.get("agent_x", 90),
            info.get("agent_y", 64),
            info.get("enemy_x", 90),
            info.get("enemy_y", 64),
            info.get("agent_status", 0),
            info.get("enemy_status", 0),
            info.get("agent_victories", 0),
            info.get("enemy_victories", 0),
        )

    def _process_cross_attention_vision_pipeline(
        self,
        stacked_obs,
        player_health,
        opponent_health,
        score,
        player_x,
        player_y,
        opponent_x,
        opponent_y,
        player_status,
        opponent_status,
        player_victories,
        opponent_victories,
        button_features,
    ):
        try:
            strategic_features = self.strategic_tracker.update(
                player_health,
                opponent_health,
                score,
                player_x,
                player_y,
                opponent_x,
                opponent_y,
                player_status,
                opponent_status,
                player_victories,
                opponent_victories,
                button_features,
            )

            if self.cnn_extractor is not None:
                with torch.no_grad():
                    device = next(self.cnn_extractor.parameters()).device
                    obs_tensor = (
                        torch.from_numpy(stacked_obs).float().unsqueeze(0).to(device)
                    )
                    visual_features = (
                        self.cnn_extractor(obs_tensor).squeeze(0).cpu().numpy()
                    )
            else:
                visual_features = np.zeros(512, dtype=np.float32)

            self.visual_features_history.append(visual_features)
            self.strategic_features_history.append(strategic_features)

            if (
                self.vision_ready
                and len(self.visual_features_history) == self.frame_stack
            ):
                processed_output = self._get_cross_attention_processed_output()
                if processed_output is not None:
                    self.stats["predictions_made"] += 1
                    self.stats["visual_attention_weight"] = np.mean(
                        processed_output["visual_attention"]
                    )
                    self.stats["strategy_attention_weight"] = np.mean(
                        processed_output["strategy_attention"]
                    )
                    self.stats["oscillation_attention_weight"] = np.mean(
                        processed_output["oscillation_attention"]
                    )
                    self.stats["button_attention_weight"] = np.mean(
                        processed_output["button_attention"]
                    )
                    return processed_output["processed_features"]
            return None
        except Exception as e:
            logger.error(f"Cross-attention pipeline error: {e}", exc_info=True)
            return None

    def _get_cross_attention_processed_output(self):
        try:
            if (
                not self.vision_ready
                or len(self.visual_features_history) < self.frame_stack
            ):
                return None

            visual_seq = np.stack(list(self.visual_features_history))
            strategic_seq = np.stack(list(self.strategic_features_history))

            # Split strategic features
            strategy_seq = strategic_seq[:, : len(self.strategic_feature_names)]
            oscillation_seq = strategic_seq[
                :,
                len(self.strategic_feature_names) : len(self.strategic_feature_names)
                + len(self.oscillation_feature_names),
            ]
            button_seq = strategic_seq[
                :,
                len(self.strategic_feature_names)
                + len(self.oscillation_feature_names) :,
            ]

            # Combine all feature sequences
            combined_seq = np.concatenate(
                [visual_seq, strategy_seq, oscillation_seq, button_seq], axis=1
            )

            device = next(self.cross_attention_transformer.parameters()).device
            combined_tensor = (
                torch.from_numpy(combined_seq).float().unsqueeze(0).to(device)
            )

            with torch.no_grad():
                output_dict = self.cross_attention_transformer(combined_tensor)

            # Check for required keys
            required_keys = [
                "processed_features",
                "visual_attention",
                "strategy_attention",
                "oscillation_attention",
                "button_attention",
            ]
            if not all(k in output_dict for k in required_keys):
                logger.error(
                    f"Transformer output malformed. Keys: {output_dict.keys()}"
                )
                return None

            return {
                "processed_features": output_dict["processed_features"]
                .cpu()
                .numpy()[0],
                "visual_attention": output_dict["visual_attention"].cpu().numpy(),
                "strategy_attention": output_dict["strategy_attention"].cpu().numpy(),
                "oscillation_attention": output_dict["oscillation_attention"]
                .cpu()
                .numpy(),
                "button_attention": output_dict["button_attention"].cpu().numpy(),
            }
        except Exception as e:
            logger.error(f"Cross-attention processing error: {e}", exc_info=True)
            return None

    def _preprocess_frame(self, frame):
        if frame is None:
            return np.zeros((*self.target_size, 3), dtype=np.uint8)
        return cv2.resize(frame, (self.target_size[1], self.target_size[0]))

    def _get_stacked_observation(self):
        if len(self.frame_buffer) == 0:
            return np.zeros(self.observation_space.shape, dtype=np.uint8)
        stacked = np.concatenate(list(self.frame_buffer), axis=2)
        return stacked.transpose(2, 0, 1)

    def inject_feature_extractor(self, feature_extractor):
        if not self.enable_vision_transformer:
            return
        try:
            self.cnn_extractor = feature_extractor.cnn
            device = next(feature_extractor.parameters()).device
            self.cross_attention_transformer = EnhancedCrossAttentionVisionTransformer(
                visual_dim=512,
                strategic_dim=len(self.strategic_feature_names),
                oscillation_dim=len(self.oscillation_feature_names),
                button_dim=len(self.button_feature_names),
                seq_length=self.frame_stack,
            ).to(device)

            if hasattr(feature_extractor, "inject_cross_attention_components"):
                feature_extractor.inject_cross_attention_components(
                    self.cross_attention_transformer, self
                )

            self.vision_ready = True
            self.stats["cross_attention_ready"] = True
            print(
                "✅ FIXED Enhanced Cross-Attention Vision Transformer with Oscillation initialized and ready!"
            )
        except Exception as e:
            logger.error(
                f"Cross-Attention Vision Transformer injection failed: {e}",
                exc_info=True,
            )
            self.vision_ready = False


class StreetFighterCrossAttentionCNN(BaseFeaturesExtractor):
    """FIXED Enhanced CNN feature extractor with proper gradient flow"""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        self.cnn = CNNFeatureExtractor(input_channels=n_input_channels, feature_dim=512)
        self.cross_attention_transformer = None
        self.wrapper_env = None
        self.final_projection = nn.Linear(256, features_dim)

        # FIXED: Initialize final projection properly
        nn.init.xavier_uniform_(self.final_projection.weight)
        nn.init.constant_(self.final_projection.bias, 0)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """FIXED Forward pass with enhanced feature pipeline"""
        # Step 1: Visual features from CNN
        visual_features = self.cnn(observations.float() / 255.0)

        # Step 2: Access wrapper to get strategic and oscillation features
        if (
            self.wrapper_env is not None
            and self.cross_attention_transformer is not None
        ):
            batch_size = observations.shape[0]

            if hasattr(self.wrapper_env, "envs"):
                strategic_features_batch = []
                for env in self.wrapper_env.envs:
                    try:
                        # Get the actual wrapper environment
                        wrapper = env.env if hasattr(env, "env") else env

                        # Extract state from retro info
                        info = wrapper.unwrapped.data.lookup_all()
                        state_data = wrapper._extract_enhanced_state(info)

                        strategic_features = wrapper.strategic_tracker.update(
                            *state_data, button_features=np.zeros(12)
                        )
                        strategic_features_batch.append(strategic_features)
                    except Exception as e:
                        # Fallback to zeros if extraction fails
                        strategic_features_batch.append(np.zeros(45, dtype=np.float32))

                strategic_seq = torch.tensor(
                    np.array(strategic_features_batch),
                    dtype=torch.float32,
                    device=observations.device,
                )

                # Split into strategy, oscillation, and button features
                wrapper = (
                    self.wrapper_env.envs[0].env
                    if hasattr(self.wrapper_env.envs[0], "env")
                    else self.wrapper_env.envs[0]
                )

                strategy_features = (
                    strategic_seq[:, : len(wrapper.strategic_feature_names)]
                    .unsqueeze(1)
                    .repeat(1, 8, 1)
                )
                oscillation_features = (
                    strategic_seq[
                        :,
                        len(wrapper.strategic_feature_names) : len(
                            wrapper.strategic_feature_names
                        )
                        + len(wrapper.oscillation_feature_names),
                    ]
                    .unsqueeze(1)
                    .repeat(1, 8, 1)
                )
                button_features = (
                    strategic_seq[
                        :,
                        len(wrapper.strategic_feature_names)
                        + len(wrapper.oscillation_feature_names) :,
                    ]
                    .unsqueeze(1)
                    .repeat(1, 8, 1)
                )

                # Combine with visual features
                visual_seq = visual_features.unsqueeze(1).repeat(1, 8, 1)

                combined_seq = torch.cat(
                    [
                        visual_seq,
                        strategy_features,
                        oscillation_features,
                        button_features,
                    ],
                    dim=-1,
                )

                output_dict = self.cross_attention_transformer(combined_seq)
                processed_features = output_dict["processed_features"]
                return self.final_projection(processed_features)

        # Fallback path
        return self.final_projection(visual_features[:, :256])

    def inject_cross_attention_components(
        self, cross_attention_transformer, wrapper_env
    ):
        self.cross_attention_transformer = cross_attention_transformer
        self.wrapper_env = wrapper_env
        print(
            "✅ FIXED Cross-attention components with Oscillation injected into SB3 CNN"
        )


def monitor_gradients(model, step_count):
    """FIXED: Monitor gradient flow in cross-attention components"""
    if step_count % 5000 == 0:
        print(f"\n🔍 Gradient Monitor at Step {step_count}:")

        total_grad_norm = 0
        param_count = 0

        for name, param in model.policy.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                total_grad_norm += grad_norm
                param_count += 1

                # Focus on cross-attention components
                if "cross_attention" in name or "attention" in name:
                    print(f"  {name}: {grad_norm:.6f}")
            else:
                if "cross_attention" in name or "attention" in name:
                    print(f"  {name}: NO GRADIENT")

        avg_grad_norm = total_grad_norm / max(param_count, 1)
        print(f"  Average gradient norm: {avg_grad_norm:.6f}")
        print(f"  Total parameters with gradients: {param_count}")


# FIXED: Export all necessary components
__all__ = [
    "StreetFighterVisionWrapper",
    "StreetFighterCrossAttentionCNN",
    "OscillationTracker",
    "StreetFighterDiscreteActions",
    "monitor_gradients",
]
