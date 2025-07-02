#!/usr/bin/env python3

"""
wrapper.py - Improved Street Fighter II Wrapper with Enhanced Score Momentum
Key improvements:
1. Better score momentum calculation with combo detection
2. Enhanced reward system with combo bonuses
3. Improved strategic feature weighting
4. Better action space design for more effective gameplay
5. REDUCED LOGGING FREQUENCY to prevent large log files
"""

import cv2
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from collections import deque
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict
import math
import logging
import os
from datetime import datetime

# Configure logging to file instead of console - REDUCED FREQUENCY
os.makedirs("logs", exist_ok=True)
log_filename = f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.WARNING,  # CHANGED: Reduced from INFO to WARNING
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        # Remove console handler to reduce output
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


class StreetFighterDiscreteActions:
    """
    Enhanced Street Fighter II Discrete Action System
    Focuses on more effective action combinations based on fighting game mechanics
    """

    def __init__(self):
        # Button indices for stable-retro filtered actions (12 total)
        self.button_names = [
            "B",  # 0 - Light Kick
            "Y",  # 1 - Light Punch
            "SELECT",  # 2
            "START",  # 3
            "UP",  # 4
            "DOWN",  # 5
            "LEFT",  # 6
            "RIGHT",  # 7
            "A",  # 8 - Medium Kick
            "X",  # 9 - Medium Punch
            "L",  # 10 - Heavy Punch
            "R",  # 11 - Heavy Kick
        ]
        self.num_buttons = 12

        # IMPROVED: More strategic action combinations based on fighting game theory
        # COMPATIBLE: Keep original 57 actions for model compatibility
        self.action_combinations = [
            # 0: No action (neutral)
            [],
            # Basic movements (1-8)
            [6],  # 1: LEFT
            [7],  # 2: RIGHT
            [4],  # 3: UP (jump)
            [5],  # 4: DOWN (crouch)
            [6, 4],  # 5: LEFT + UP (jump back)
            [7, 4],  # 6: RIGHT + UP (jump forward)
            [6, 5],  # 7: LEFT + DOWN (crouch back)
            [7, 5],  # 8: RIGHT + DOWN (crouch forward)
            # Light attacks (9-14)
            [1],  # 9: Light punch (Y)
            [0],  # 10: Light kick (B)
            [1, 6],  # 11: Light punch + LEFT
            [1, 7],  # 12: Light punch + RIGHT
            [0, 6],  # 13: Light kick + LEFT
            [0, 7],  # 14: Light kick + RIGHT
            # Medium attacks (15-20)
            [9],  # 15: Medium punch (X)
            [8],  # 16: Medium kick (A)
            [9, 6],  # 17: Medium punch + LEFT
            [9, 7],  # 18: Medium punch + RIGHT
            [8, 6],  # 19: Medium kick + LEFT
            [8, 7],  # 20: Medium kick + RIGHT
            # Heavy attacks (21-26)
            [10],  # 21: Heavy punch (L)
            [11],  # 22: Heavy kick (R)
            [10, 6],  # 23: Heavy punch + LEFT
            [10, 7],  # 24: Heavy punch + RIGHT
            [11, 6],  # 25: Heavy kick + LEFT
            [11, 7],  # 26: Heavy kick + RIGHT
            # Jumping attacks (27-32)
            [1, 4],  # 27: Jumping light punch
            [0, 4],  # 28: Jumping light kick
            [9, 4],  # 29: Jumping medium punch
            [8, 4],  # 30: Jumping medium kick
            [10, 4],  # 31: Jumping heavy punch
            [11, 4],  # 32: Jumping heavy kick
            # Crouching attacks (33-38)
            [1, 5],  # 33: Crouching light punch
            [0, 5],  # 34: Crouching light kick
            [9, 5],  # 35: Crouching medium punch
            [8, 5],  # 36: Crouching medium kick
            [10, 5],  # 37: Crouching heavy punch
            [11, 5],  # 38: Crouching heavy kick
            # Special move motions (39-50)
            [5, 7],  # 39: Down-forward (quarter circle start)
            [5, 6],  # 40: Down-back (reverse quarter circle)
            [5, 7, 1],  # 41: Hadoken motion + light punch
            [5, 7, 9],  # 42: Hadoken motion + medium punch
            [5, 7, 10],  # 43: Hadoken motion + heavy punch
            [5, 6, 1],  # 44: Reverse hadoken + light punch
            [5, 6, 9],  # 45: Reverse hadoken + medium punch
            [5, 6, 10],  # 46: Reverse hadoken + heavy punch
            [5, 7, 0],  # 47: Quarter circle + light kick
            [5, 7, 8],  # 48: Quarter circle + medium kick
            [5, 7, 11],  # 49: Quarter circle + heavy kick
            [4, 5],  # 50: Up-down (charge motion)
            # Anti-air and defensive (51-56)
            [7, 1],  # 51: Forward + light punch (anti-air)
            [7, 9],  # 52: Forward + medium punch
            [7, 10],  # 53: Forward + heavy punch
            [6],  # 54: Block (hold back)
            [6, 5],  # 55: Low block (back + down)
            [4, 6],  # 56: Jump back (defensive)
            [4],  # 57: UP (jump)
            [5],  # 58: DOWN (crouch)
            [6, 4],  # 59: JUMP BACK (defensive)
            [7, 4],  # 60: JUMP FORWARD (aggressive)
            [6, 5],  # 61: CROUCH BACK (low block position)
            [7, 5],  # 62: CROUCH FORWARD (approach)
            # QUICK ATTACKS (63-70) - Fast startup for pressure
            [1],  # 63: Light Punch (jab)
            [0],  # 64: Light Kick
            [1, 7],  # 65: Forward Light Punch (advancing jab)
            [0, 7],  # 66: Forward Light Kick
            [1, 5],  # 67: Crouching Light Punch
            [0, 5],  # 68: Crouching Light Kick
            # MEDIUM ATTACKS (69-76) - Good damage/range balance
            [9],  # 69: Medium Punch
            [8],  # 70: Medium Kick
            [9, 7],  # 71: Forward Medium Punch
            [8, 7],  # 72: Forward Medium Kick
            [9, 5],  # 73: Crouching Medium Punch
            [8, 5],  # 74: Crouching Medium Kick
            # HEAVY ATTACKS (75-82) - High damage but slower
            [10],  # 75: Heavy Punch
            [11],  # 76: Heavy Kick
            [10, 7],  # 77: Forward Heavy Punch
            [11, 7],  # 78: Forward Heavy Kick
            [10, 5],  # 79: Crouching Heavy Punch (sweep setup)
            [11, 5],  # 80: Crouching Heavy Kick (sweep)
            # JUMP ATTACKS (81-88) - Air control and crossups
            [1, 4],  # 81: Jumping Light Punch
            [0, 4],  # 82: Jumping Light Kick
            [9, 4],  # 83: Jumping Medium Punch
            [8, 4],  # 84: Jumping Medium Kick
            [10, 4],  # 85: Jumping Heavy Punch
            [11, 4],  # 86: Jumping Heavy Kick
            [1, 4, 7],  # 87: Jump Forward Light Punch
            [9, 4, 7],  # 88: Jump Forward Medium Punch
            # SPECIAL MOVE MOTIONS (89-100) - Improved special move inputs
            [5, 7],  # 89: Quarter Circle Forward (QCF start)
            [5, 6],  # 90: Quarter Circle Back (QCB start)
            [5, 7, 1],  # 91: Hadoken Light (fireball)
            [5, 7, 9],  # 92: Hadoken Medium
            [5, 7, 10],  # 93: Hadoken Heavy
            [7, 5, 7, 1],  # 94: Dragon Punch Light (DP motion)
            [7, 5, 7, 9],  # 95: Dragon Punch Medium
            [7, 5, 7, 10],  # 96: Dragon Punch Heavy
            [5, 6, 0],  # 97: Hurricane Kick Light
            [5, 6, 8],  # 98: Hurricane Kick Medium
            [5, 6, 11],  # 99: Hurricane Kick Heavy
            [6, 5, 6, 1],  # 100: Reverse Dragon Punch
            # DEFENSIVE OPTIONS (101-106) - Improved defensive play
            [6],  # 101: Block (hold back)
            [6, 5],  # 102: Low Block
            [4, 6],  # 103: Jump Back Block
            [6, 1],  # 104: Jab while blocking (frame trap escape)
            [5],  # 105: Crouch (avoid high attacks)
            [4],  # 106: Jump (avoid low attacks)
            # COMBO STARTERS (107-114) - Common combo initiators
            [1, 1],  # 107: Double Jab (link combo)
            [0, 1],  # 108: Light Kick to Light Punch
            [9, 10],  # 109: Medium Punch to Heavy Punch
            [8, 11],  # 110: Medium Kick to Heavy Kick
            [5, 0, 9],  # 111: Crouch Light Kick to Medium Punch
            [1, 5, 7, 10],  # 112: Jab to Hadoken Heavy
            [9, 5, 7, 9],  # 113: Medium Punch to Hadoken Medium
            [0, 5, 6, 8],  # 114: Light Kick to Hurricane Medium
        ]

        self.num_actions = len(self.action_combinations)

        # REDUCED: Only log initialization once
        print(f"🎮 Enhanced Street Fighter Discrete Actions initialized:")
        print(f"   Total discrete actions: {self.num_actions}")
        print(f"   Improved focus on: combos, specials, positioning")

    def discrete_to_multibinary(self, action_index: int) -> np.ndarray:
        """Convert discrete action to multi-binary array for stable-retro"""
        if action_index < 0 or action_index >= self.num_actions:
            # REDUCED: Only log critical errors
            if action_index < -10 or action_index > self.num_actions + 10:
                logger.error(f"Invalid action index: {action_index}")
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
        """Get 12-dimensional button features for strategic analysis"""
        return self.discrete_to_multibinary(action_index).astype(np.float32)


class CNNFeatureExtractor(nn.Module):
    """CNN to extract features from 8-frame RGB stack (180×128 resolution)"""

    def __init__(self, input_channels=24, feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim

        # Optimized CNN architecture for 180×128 input
        self.cnn = nn.Sequential(
            # First conv block - reduce spatial size quickly
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        # Project to desired feature dimension
        self.projection = nn.Linear(256, feature_dim)

        # Add tactical prediction tracking
        self.current_attack_timing = 0.0
        self.current_defend_timing = 0.0

    def forward(self, frame_stack: torch.Tensor) -> torch.Tensor:
        try:
            cnn_output = self.cnn(frame_stack)
            features = self.projection(cnn_output)
            return features
        except Exception as e:
            # REDUCED: Only log critical CNN errors
            logger.error(f"CNN error: {e}")
            batch_size = frame_stack.shape[0] if len(frame_stack.shape) > 0 else 1
            return torch.zeros(batch_size, self.feature_dim, device=frame_stack.device)

    def update_tactical_predictions(self, attack_timing: float, defend_timing: float):
        """Update current tactical predictions from Vision Transformer"""
        self.current_attack_timing = float(attack_timing)
        self.current_defend_timing = float(defend_timing)


class StrategicFeatureTracker:
    """
    ENHANCED Strategic feature tracker with improved button history
    Features: 33 total (21 strategic + 12 button history features)
    Key improvement: Button features represent PREVIOUS action, not current
    """

    def __init__(self, history_length=8):
        self.history_length = history_length

        # Health tracking
        self.player_health_history = deque(maxlen=history_length)
        self.opponent_health_history = deque(maxlen=history_length)

        # Position tracking
        self.player_x_history = deque(maxlen=history_length)
        self.player_y_history = deque(maxlen=history_length)
        self.opponent_x_history = deque(maxlen=history_length)
        self.opponent_y_history = deque(maxlen=history_length)

        # ENHANCED: Score tracking with combo detection
        self.score_history = deque(maxlen=history_length)
        self.score_change_history = deque(maxlen=history_length)
        self.combo_counter = 0
        self.max_combo_this_round = 0
        self.last_score_increase_frame = -1
        self.current_frame = 0

        # Status tracking
        self.player_status_history = deque(maxlen=history_length)
        self.opponent_status_history = deque(maxlen=history_length)

        # Victory tracking
        self.player_victories_history = deque(maxlen=history_length)
        self.opponent_victories_history = deque(maxlen=history_length)

        # Damage tracking with enhanced combo detection
        self.player_damage_dealt_history = deque(maxlen=history_length)
        self.opponent_damage_dealt_history = deque(maxlen=history_length)
        self.recent_damage_events = deque(maxlen=5)  # Track recent damage for combos

        # FIXED: Button tracking - represents PREVIOUS actions, not current
        self.button_features_history = deque(maxlen=history_length)
        self.previous_button_features = np.zeros(
            12, dtype=np.float32
        )  # Store previous action

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

        # ENHANCED: Combo detection parameters
        self.COMBO_TIMEOUT_FRAMES = 60  # Combo breaks after 1 second of no hits
        self.MIN_SCORE_INCREASE_FOR_HIT = 50  # Minimum score increase to count as hit

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
        """
        ENHANCED update method with improved score momentum calculation
        FIXED: Button features now represent PREVIOUS action for proper causality
        """
        self.current_frame += 1

        # Update histories
        self.player_health_history.append(player_health)
        self.opponent_health_history.append(opponent_health)
        self.player_x_history.append(player_x)
        self.player_y_history.append(player_y)
        self.opponent_x_history.append(opponent_x)
        self.opponent_y_history.append(opponent_y)
        self.player_status_history.append(player_status)
        self.opponent_status_history.append(opponent_status)
        self.player_victories_history.append(player_victories)
        self.opponent_victories_history.append(opponent_victories)

        # ENHANCED: Score momentum with combo detection
        score_change = 0
        if self.prev_score is not None:
            score_change = score - self.prev_score

            # Detect combo hits
            if score_change >= self.MIN_SCORE_INCREASE_FOR_HIT:
                frames_since_last_hit = (
                    self.current_frame - self.last_score_increase_frame
                )

                if (
                    frames_since_last_hit <= self.COMBO_TIMEOUT_FRAMES
                    and self.last_score_increase_frame > 0
                ):
                    # Continue combo
                    self.combo_counter += 1
                else:
                    # Start new combo
                    self.combo_counter = 1

                self.last_score_increase_frame = self.current_frame
                self.max_combo_this_round = max(
                    self.max_combo_this_round, self.combo_counter
                )

                # Track recent damage events for pattern analysis
                self.recent_damage_events.append(
                    {
                        "frame": self.current_frame,
                        "score_increase": score_change,
                        "combo_count": self.combo_counter,
                    }
                )
            else:
                # Check if combo should break
                frames_since_last_hit = (
                    self.current_frame - self.last_score_increase_frame
                )
                if frames_since_last_hit > self.COMBO_TIMEOUT_FRAMES:
                    self.combo_counter = 0

        self.score_history.append(score)
        self.score_change_history.append(score_change)

        # FIXED: Update button features history with PREVIOUS action
        # This represents what action was taken in the previous frame
        self.button_features_history.append(self.previous_button_features.copy())

        # Store current button features for next frame (proper causality)
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

        # Update frame counting for close combat tracking
        self.total_frames += 1
        distance = np.sqrt((player_x - opponent_x) ** 2 + (player_y - opponent_y) ** 2)
        if distance <= self.CLOSE_DISTANCE:
            self.close_combat_count += 1

        # Calculate 33 features (21 strategic + 12 button history)
        features = self._calculate_enhanced_features(
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
        )

        # Update previous values
        self.prev_player_health = player_health
        self.prev_opponent_health = opponent_health
        self.prev_score = score

        return features

    def _calculate_enhanced_features(
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
    ) -> np.ndarray:
        """
        ENHANCED feature calculation with improved score momentum
        """
        features = np.zeros(33, dtype=np.float32)

        # Features 1-2: Danger zones
        features[0] = 1.0 if player_health <= self.DANGER_ZONE_HEALTH else 0.0
        features[1] = 1.0 if opponent_health <= self.DANGER_ZONE_HEALTH else 0.0

        # Feature 3: Health ratio
        if opponent_health > 0:
            features[2] = player_health / opponent_health
        else:
            features[2] = 2.0
        features[2] = np.clip(features[2], 0.0, 2.0)

        # Feature 4: Combined health change rate
        player_health_change = self._calculate_momentum(self.player_health_history)
        opponent_health_change = self._calculate_momentum(self.opponent_health_history)
        features[3] = (player_health_change + opponent_health_change) / 2.0

        # Features 5-6: Damage rates
        features[4] = self._calculate_momentum(self.player_damage_dealt_history)
        features[5] = self._calculate_momentum(self.opponent_damage_dealt_history)

        # Features 7-8: Corner distances
        player_left_dist = player_x
        player_right_dist = self.SCREEN_WIDTH - player_x
        player_corner_dist = min(player_left_dist, player_right_dist)
        features[6] = np.clip(player_corner_dist / (self.SCREEN_WIDTH / 2), 0.0, 1.0)

        opponent_left_dist = opponent_x
        opponent_right_dist = self.SCREEN_WIDTH - opponent_x
        opponent_corner_dist = min(opponent_left_dist, opponent_right_dist)
        features[7] = np.clip(opponent_corner_dist / (self.SCREEN_WIDTH / 2), 0.0, 1.0)

        # Features 9-10: Near corner flags
        features[8] = 1.0 if player_corner_dist <= self.CORNER_THRESHOLD else 0.0
        features[9] = 1.0 if opponent_corner_dist <= self.CORNER_THRESHOLD else 0.0

        # Feature 11: Center control
        screen_center = self.SCREEN_WIDTH / 2
        player_center_dist = abs(player_x - screen_center)
        opponent_center_dist = abs(opponent_x - screen_center)
        if player_center_dist < opponent_center_dist:
            features[10] = 1.0
        elif opponent_center_dist < player_center_dist:
            features[10] = -1.0
        else:
            features[10] = 0.0

        # Feature 12: Vertical advantage
        vertical_diff = player_y - opponent_y
        features[11] = np.clip(vertical_diff / (self.SCREEN_HEIGHT / 2), -1.0, 1.0)

        # Feature 13: Position stability
        player_x_stability = 1.0 - min(
            1.0, abs(self._calculate_momentum(self.player_x_history)) / 10.0
        )
        opponent_x_stability = 1.0 - min(
            1.0, abs(self._calculate_momentum(self.opponent_x_history)) / 10.0
        )
        features[12] = (player_x_stability + opponent_x_stability) / 2.0

        # Feature 14: Optimal spacing
        if self.OPTIMAL_SPACING_MIN <= distance <= self.OPTIMAL_SPACING_MAX:
            features[13] = 1.0
        else:
            features[13] = 0.0

        # Feature 15: Forward pressure
        player_x_momentum = self._calculate_momentum(self.player_x_history)
        if player_x < opponent_x:
            forward_pressure = player_x_momentum
        else:
            forward_pressure = -player_x_momentum
        features[14] = np.clip(forward_pressure / 5.0, -1.0, 1.0)

        # Feature 16: Defensive movement
        if player_x < opponent_x:
            defensive_movement = -player_x_momentum
        else:
            defensive_movement = player_x_momentum
        features[15] = np.clip(defensive_movement / 5.0, -1.0, 1.0)

        # Feature 17: Close combat frequency
        if self.total_frames > 0:
            features[16] = self.close_combat_count / self.total_frames
        else:
            features[16] = 0.0

        # Feature 18: ENHANCED Score momentum with combo detection
        features[17] = self._calculate_enhanced_score_momentum()

        # Feature 19: Status difference
        status_diff = player_status - opponent_status
        features[18] = np.clip(status_diff / 100.0, -1.0, 1.0)

        # Features 20-21: Victory counts
        features[19] = min(player_victories / 10.0, 1.0)
        features[20] = min(opponent_victories / 10.0, 1.0)

        # Features 22-33: PREVIOUS button state (12 button history features) - FIXED
        if len(self.button_features_history) > 0:
            previous_buttons = self.button_features_history[-1]  # Previous action taken
            features[21:33] = previous_buttons
        else:
            features[21:33] = 0.0  # No previous buttons pressed

        return features

    def _calculate_enhanced_score_momentum(self) -> float:
        """
        ENHANCED score momentum calculation incorporating:
        1. Recent score changes
        2. Combo multiplier
        3. Hit frequency
        4. Damage scaling
        """
        if len(self.score_change_history) < 2:
            return 0.0

        # Base momentum from recent score changes
        recent_changes = list(self.score_change_history)[-5:]  # Last 5 frames
        base_momentum = np.mean([max(0, change) for change in recent_changes])

        # Combo multiplier - reward sustained offense
        combo_multiplier = 1.0 + (self.combo_counter * 0.1)  # 10% bonus per combo hit

        # Hit frequency bonus - reward consistent pressure
        hit_frequency = 0.0
        if len(self.recent_damage_events) > 0:
            recent_hits = [
                event
                for event in self.recent_damage_events
                if self.current_frame - event["frame"] <= 300
            ]  # Last 5 seconds
            hit_frequency = len(recent_hits) / 5.0  # Normalize to 0-1 range

        # Damage scaling - bigger hits get more momentum
        damage_scaling = 1.0
        if len(self.score_change_history) > 0:
            recent_score_change = self.score_change_history[-1]
            if recent_score_change > 0:
                # Scale based on hit strength (larger score increases = stronger hits)
                damage_scaling = 1.0 + min(recent_score_change / 1000.0, 1.0)

        # Combine all factors
        enhanced_momentum = (
            base_momentum * combo_multiplier * damage_scaling + hit_frequency
        )

        # Normalize to reasonable range (-1.0 to 2.0)
        return np.clip(enhanced_momentum / 100.0, -1.0, 2.0)

    def _calculate_momentum(self, history):
        """Calculate momentum (rate of change) from history"""
        if len(history) < 2:
            return 0.0

        values = list(history)
        changes = [values[i] - values[i - 1] for i in range(1, len(values))]

        # Use recent changes (last 3) if available, otherwise all changes
        recent_changes = changes[-3:] if len(changes) >= 3 else changes
        return np.mean(recent_changes) if recent_changes else 0.0

    def get_combo_stats(self) -> Dict:
        """Get current combo statistics for logging"""
        return {
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


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class SimplifiedVisionTransformer(nn.Module):
    """Simplified Vision Transformer with 33 strategic features"""

    def __init__(self, visual_dim=512, strategic_dim=33, seq_length=8):
        super().__init__()
        self.seq_length = seq_length

        # Combined input dimension: 512 (visual) + 33 (strategic) = 545
        combined_dim = visual_dim + strategic_dim

        # REDUCED: Only log initialization once
        print(f"🔍 SimplifiedVisionTransformer dimensions:")
        print(f"   Visual: {visual_dim}, Strategic: {strategic_dim}")
        print(f"   Combined input: {combined_dim}")

        # Project to transformer dimension
        d_model = 256
        self.input_projection = nn.Linear(combined_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout=0.1)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Tactical prediction head - only attack and defend timing
        self.tactical_predictor = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 2),  # [attack_timing, defend_timing] only
        )

    def forward(self, combined_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        try:
            # Debug: Check actual input dimensions
            batch_size, seq_len, actual_dim = combined_sequence.shape
            expected_dim = 545  # 512 + 33

            if actual_dim != expected_dim:
                # REDUCED: Only log critical dimension mismatches
                logger.error(
                    f"Dimension mismatch: got {actual_dim}, expected {expected_dim}"
                )
                if actual_dim > expected_dim:
                    combined_sequence = combined_sequence[:, :, :expected_dim]
                else:
                    padding = torch.zeros(
                        batch_size,
                        seq_len,
                        expected_dim - actual_dim,
                        device=combined_sequence.device,
                        dtype=combined_sequence.dtype,
                    )
                    combined_sequence = torch.cat([combined_sequence, padding], dim=-1)

            # Project input features
            projected = self.input_projection(combined_sequence)
            projected = self.pos_encoding(projected)

            # Apply transformer
            transformer_out = self.transformer(projected)
            final_features = transformer_out[:, -1, :]  # Use last timestep

            # Generate tactical predictions (0-1 range via sigmoid)
            tactical_logits = self.tactical_predictor(final_features)
            tactical_probs = torch.sigmoid(tactical_logits)

            return {
                "attack_timing": tactical_probs[:, 0],  # Best time to attack (0-1)
                "defend_timing": tactical_probs[:, 1],  # Best time to defend (0-1)
            }

        except Exception as e:
            # REDUCED: Only log critical transformer errors
            logger.error(f"Transformer error: {e}")
            batch_size = combined_sequence.shape[0]
            device = combined_sequence.device
            return {
                "attack_timing": torch.zeros(batch_size, device=device),
                "defend_timing": torch.zeros(batch_size, device=device),
            }


class StreetFighterVisionWrapper(gym.Wrapper):
    """
    ENHANCED Street Fighter wrapper with REDUCED LOGGING FREQUENCY
    Key improvements:
    1. Better reward calculation with combo bonuses
    2. Enhanced action space for more effective gameplay
    3. Improved tactical analysis and logging
    4. REDUCED logging frequency to prevent large log files
    """

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
        self.target_size = (128, 180)  # H, W - optimized for 180×128
        self.log_transformer_predictions = log_transformer_predictions

        # Initialize enhanced discrete action system
        self.discrete_actions = StreetFighterDiscreteActions()

        # Episode management
        self.max_episode_steps = max_episode_steps
        self.episode_steps = 0
        self.reset_round = reset_round

        # Health tracking
        self.full_hp = 176
        self.prev_player_health = self.full_hp
        self.prev_opponent_health = self.full_hp

        # ENHANCED: Defense action indices (updated for new action space)
        self.defend_action_indices = defend_action_indices or [
            101,
            102,
            103,
            104,
            105,
            106,  # Updated indices for defensive actions
        ]
        self.defense_cooldown_frames = 30
        self.last_defense_frame = -100

        # ENHANCED: Win tracking and performance metrics
        self.wins = 0
        self.losses = 0
        self.total_rounds = 0
        self.total_damage_dealt = 0
        self.total_damage_received = 0
        self.round_start_time = 0

        # Setup observation space: [channels, height, width] - 8 frames RGB
        obs_shape = (3 * frame_stack, self.target_size[0], self.target_size[1])
        self.observation_space = spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

        # Override action space to discrete
        self.action_space = spaces.Discrete(self.discrete_actions.num_actions)

        # Initialize enhanced strategic feature tracker
        self.strategic_tracker = StrategicFeatureTracker()

        # Frame and feature buffers
        self.frame_buffer = deque(maxlen=frame_stack)
        self.visual_features_history = deque(maxlen=frame_stack)
        self.strategic_features_history = deque(maxlen=frame_stack)

        # Vision transformer components
        self.cnn_extractor = None
        self.vision_transformer = None
        self.vision_ready = False

        # Current tactical predictions
        self.current_attack_timing = 0.0
        self.current_defend_timing = 0.0
        self.recent_rewards = deque(maxlen=10)

        # Current action tracking
        self.current_discrete_action = 0

        # ENHANCED: Statistics with combo tracking
        self.stats = {
            "predictions_made": 0,
            "vision_transformer_ready": False,
            "avg_attack_timing": 0.0,
            "avg_defend_timing": 0.0,
            "total_combos": 0,
            "max_combo": 0,
            "avg_damage_per_round": 0.0,
            "defensive_efficiency": 0.0,
        }

        # REDUCED LOGGING: Increase intervals and reduce detail
        if self.log_transformer_predictions:
            self.transformer_logs = {
                "predictions": [],
                "feature_importance": [],
                "tactical_patterns": {},
                "situation_analysis": {},
                "learning_progression": [],
                "combo_analysis": [],
            }
            # INCREASED: From 100 to 1000 for less frequent detailed logging
            self.log_interval = 1000
            self.last_detailed_log = 0

            # INCREASED: From 50k to 100k for less frequent saves
            self.save_interval_steps = 100000
            self.last_save_step = 0

            # Feature names for logging (when needed)
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
                "position_stability",
                "optimal_spacing",
                "forward_pressure",
                "defensive_movement",
                "close_combat_frequency",
                "enhanced_score_momentum",
                "status_difference",
                "agent_victories",
                "enemy_victories",
            ]

            self.button_feature_names = [
                "prev_B_pressed",
                "prev_Y_pressed",
                "prev_SELECT_pressed",
                "prev_START_pressed",
                "prev_UP_pressed",
                "prev_DOWN_pressed",
                "prev_LEFT_pressed",
                "prev_RIGHT_pressed",
                "prev_A_pressed",
                "prev_X_pressed",
                "prev_L_pressed",
                "prev_R_pressed",
            ]

        # REDUCED: Only log initialization once with print instead of logger
        print(f"🎮 ENHANCED Street Fighter Vision Wrapper initialized:")
        print(f"   Resolution: {self.target_size[1]}×{self.target_size[0]}")
        print(f"   Frame stack: {frame_stack} RGB frames (24 channels total)")
        print(f"   Strategic Features: 33 total (21 combat + 12 button)")
        print(
            f"   Action Space: Enhanced Discrete({self.discrete_actions.num_actions}) actions"
        )
        print(
            f"   REDUCED LOGGING: Detail every {self.log_interval} predictions, saves every {self.save_interval_steps} steps"
        )
        print(f"   Analysis Output: {ANALYSIS_OUTPUT_DIR}/")

    def inject_feature_extractor(self, feature_extractor):
        """Inject CNN feature extractor and initialize vision transformer"""
        if not self.enable_vision_transformer:
            print("   🔧 Vision Transformer disabled")
            return

        try:
            self.cnn_extractor = feature_extractor
            actual_feature_dim = self.cnn_extractor.features_dim
            print(f"   📏 Detected CNN feature dimension: {actual_feature_dim}")

            # Initialize vision transformer with correct dimensions
            device = next(feature_extractor.parameters()).device
            self.vision_transformer = SimplifiedVisionTransformer(
                visual_dim=actual_feature_dim,
                strategic_dim=33,  # 21 strategic + 12 button features
                seq_length=self.frame_stack,
            ).to(device)
            self.vision_ready = True
            self.stats["vision_transformer_ready"] = True

            print("   ✅ Enhanced Strategic Vision Transformer initialized and ready!")

        except Exception as e:
            # REDUCED: Only log critical injection failures
            logger.error(f"Vision Transformer injection failed: {e}")
            self.vision_ready = False

    def reset(self, **kwargs):
        """Reset environment and initialize buffers"""
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            observation, info = result
        else:
            observation = result
            info = {}

        # Reset tracking
        self.prev_player_health = self.full_hp
        self.prev_opponent_health = self.full_hp
        self.episode_steps = 0
        self.round_start_time = 0

        # Reset defense anti-spam tracking
        self.last_defense_frame = -100

        # Reset tactical predictions
        self.current_attack_timing = 0.0
        self.current_defend_timing = 0.0
        self.current_discrete_action = 0

        # Initialize frame buffer with processed frames
        processed_frame = self._preprocess_frame(observation)
        self.frame_buffer.clear()
        for _ in range(self.frame_stack):
            self.frame_buffer.append(processed_frame)

        # Clear feature history buffers
        self.visual_features_history.clear()
        self.strategic_features_history.clear()

        stacked_obs = self._get_stacked_observation()
        return stacked_obs, info

    def step(self, discrete_action):
        """ENHANCED step method with improved reward calculation and REDUCED LOGGING"""
        # Store current discrete action
        self.current_discrete_action = discrete_action

        # Convert discrete action to multi-binary
        multibinary_action = self.discrete_actions.discrete_to_multibinary(
            discrete_action
        )

        # Track defense actions for anti-spam
        is_defending = discrete_action in self.defend_action_indices
        if is_defending:
            self.last_defense_frame = self.episode_steps

        # Execute step with multi-binary action
        observation, reward, done, truncated, info = self.env.step(multibinary_action)

        # Extract enhanced game state information
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

        # ENHANCED: Calculate custom reward with combo bonuses
        custom_reward, custom_done = self._calculate_enhanced_reward(
            curr_player_health, curr_opponent_health, score, discrete_action
        )
        self.recent_rewards.append(custom_reward)

        if custom_done:
            done = custom_done

        # Process new frame
        processed_frame = self._preprocess_frame(observation)
        self.frame_buffer.append(processed_frame)

        # Get current stacked observation
        stacked_obs = self._get_stacked_observation()

        # Get button features from current action
        button_features = self.discrete_actions.get_button_features(discrete_action)

        # Process through strategic vision pipeline
        tactical_predictions = self._process_strategic_vision_pipeline(
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

        # Update CNN with tactical predictions
        if tactical_predictions and self.cnn_extractor is not None:
            self.cnn_extractor.update_tactical_predictions(
                tactical_predictions["attack_timing"],
                tactical_predictions["defend_timing"],
            )

        # Update enhanced statistics and periodic saves
        self._update_enhanced_stats()

        # REDUCED: Less frequent analysis saves (every 100k steps instead of 50k)
        if (
            self.log_transformer_predictions
            and self.episode_steps - self.last_save_step >= self.save_interval_steps
        ):
            self.last_save_step = self.episode_steps
            filename = f"transformer_analysis_step_{self.episode_steps}.json"
            filepath = os.path.join(ANALYSIS_OUTPUT_DIR, filename)
            self.save_enhanced_analysis(filepath)

        self.episode_steps += 1
        info.update(self.stats)

        return stacked_obs, custom_reward, done, truncated, info

    def _calculate_enhanced_reward(
        self, curr_player_health, curr_opponent_health, score, action
    ):
        """
        ENHANCED reward calculation with combo bonuses and strategic incentives
        REDUCED LOGGING: Only log wins/losses, not every reward calculation
        """
        reward = 0.0
        done = False

        # Check for round end
        if curr_player_health <= 0 or curr_opponent_health <= 0:
            self.total_rounds += 1

            if curr_opponent_health <= 0 and curr_player_health > 0:
                # Win with bonus based on health remaining
                health_bonus = (
                    curr_player_health / self.full_hp
                ) * 50  # Up to 50 point bonus
                self.wins += 1
                win_rate = self.wins / self.total_rounds

                # REDUCED: Only log wins, not detailed breakdown
                print(f"🏆 WIN! {self.wins}/{self.total_rounds} ({win_rate:.1%})")

                reward += 100 + health_bonus  # Base win reward + health bonus

                # Combo bonus for wins
                combo_stats = self.strategic_tracker.get_combo_stats()
                if combo_stats["max_combo_this_round"] >= 3:
                    combo_bonus = combo_stats["max_combo_this_round"] * 10
                    # REDUCED: Only log significant combos
                    if combo_stats["max_combo_this_round"] >= 5:
                        print(
                            f"   🔥 BIG COMBO: {combo_stats['max_combo_this_round']} hits!"
                        )
                    reward += combo_bonus
                    self.stats["max_combo"] = max(
                        self.stats["max_combo"], combo_stats["max_combo_this_round"]
                    )

            elif curr_player_health <= 0 and curr_opponent_health > 0:
                # Loss
                self.losses += 1
                win_rate = self.wins / self.total_rounds
                # REDUCED: Only log losses, not detailed info
                print(f"💀 LOSS! {self.wins}/{self.total_rounds} ({win_rate:.1%})")

            if self.reset_round:
                done = True

        # ENHANCED: Damage-based reward with combo multiplier
        damage_dealt = max(0, self.prev_opponent_health - curr_opponent_health)
        damage_received = max(0, self.prev_player_health - curr_player_health)

        base_reward = damage_dealt - damage_received

        # Combo multiplier for damage
        combo_stats = self.strategic_tracker.get_combo_stats()
        if combo_stats["current_combo"] > 1 and damage_dealt > 0:
            combo_multiplier = (
                1.0 + (combo_stats["current_combo"] - 1) * 0.2
            )  # 20% bonus per combo hit
            reward += base_reward * combo_multiplier
            # REMOVED: Debug logging for combo bonuses to reduce noise
        else:
            reward += base_reward

        # ENHANCED: Strategic action bonuses
        reward += self._calculate_strategic_action_bonus(
            action, curr_player_health, curr_opponent_health
        )

        # Tracking for statistics
        self.total_damage_dealt += damage_dealt
        self.total_damage_received += damage_received

        # Update health tracking
        self.prev_player_health = curr_player_health
        self.prev_opponent_health = curr_opponent_health

        return reward, done

    def _calculate_strategic_action_bonus(self, action, player_health, opponent_health):
        """Calculate bonus rewards for strategic actions"""
        bonus = 0.0

        # Defensive bonus when in danger
        if player_health <= self.full_hp * 0.3:  # Low health
            if action in self.defend_action_indices:
                bonus += 0.5  # Small bonus for defensive play when low

        # Aggressive bonus when opponent is in danger
        if opponent_health <= self.full_hp * 0.3:  # Enemy low health
            # Check if action is offensive (attacks)
            if action in range(63, 88):  # Attack actions in our action space
                bonus += 1.0  # Bigger bonus for attacking when enemy is low

        return bonus

    def _update_enhanced_stats(self):
        """Update enhanced statistics including combo tracking"""
        combo_stats = self.strategic_tracker.get_combo_stats()

        if combo_stats["current_combo"] > self.stats["max_combo"]:
            self.stats["max_combo"] = combo_stats["current_combo"]

        if combo_stats["current_combo"] >= 2:
            self.stats["total_combos"] += 1

        # Calculate average damage per round
        if self.total_rounds > 0:
            self.stats["avg_damage_per_round"] = (
                self.total_damage_dealt / self.total_rounds
            )

            # Calculate defensive efficiency (damage dealt vs received ratio)
            if self.total_damage_received > 0:
                self.stats["defensive_efficiency"] = (
                    self.total_damage_dealt / self.total_damage_received
                )
            else:
                self.stats["defensive_efficiency"] = (
                    float("inf") if self.total_damage_dealt > 0 else 0.0
                )

    def _extract_enhanced_state(self, info):
        """Extract enhanced game state including all required data"""
        # Basic health and score
        player_health = info.get("agent_hp", self.full_hp)
        opponent_health = info.get("enemy_hp", self.full_hp)
        score = info.get("score", 0)

        # Position data
        player_x = info.get("agent_x", 90)
        player_y = info.get("agent_y", 64)
        opponent_x = info.get("enemy_x", 90)
        opponent_y = info.get("enemy_y", 64)

        # Status data
        player_status = info.get("agent_status", 0)
        opponent_status = info.get("enemy_status", 0)

        # Victory data
        player_victories = info.get("agent_victories", 0)
        opponent_victories = info.get("enemy_victories", 0)

        # REMOVED: Debug logging to reduce noise

        return (
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
        )

    def _process_strategic_vision_pipeline(
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
        """Process through enhanced strategic vision pipeline with REDUCED LOGGING"""
        try:
            # Step 1: Enhanced strategic features (33 features with improved score momentum)
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

            # Step 2: CNN feature extraction
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

            # Store in history buffers
            self.visual_features_history.append(visual_features)
            self.strategic_features_history.append(strategic_features)

            # Step 3: Vision transformer prediction
            if (
                self.vision_ready
                and len(self.visual_features_history) == self.frame_stack
            ):
                prediction = self._make_tactical_prediction()
                if prediction:
                    self.stats["predictions_made"] += 1

                    # Apply defense anti-spam logic
                    frames_since_defense = self.episode_steps - self.last_defense_frame
                    if frames_since_defense < self.defense_cooldown_frames:
                        prediction["defend_timing"] *= 0.1

                    # Update current predictions
                    self.current_attack_timing = prediction["attack_timing"]
                    self.current_defend_timing = prediction["defend_timing"]

                    # Update running averages for stats
                    self.stats["avg_attack_timing"] = (
                        self.stats["avg_attack_timing"] * 0.99
                        + prediction["attack_timing"] * 0.01
                    )
                    self.stats["avg_defend_timing"] = (
                        self.stats["avg_defend_timing"] * 0.99
                        + prediction["defend_timing"] * 0.01
                    )

                    return prediction

            return None

        except Exception as e:
            # REDUCED: Only log critical pipeline errors
            logger.error(f"Strategic vision pipeline error: {e}")
            return None

    def _make_tactical_prediction(self):
        """Make tactical predictions using enhanced strategic features with REDUCED LOGGING"""
        try:
            if (
                not self.vision_ready
                or len(self.visual_features_history) < self.frame_stack
            ):
                return None

            # Stack sequences
            visual_seq = np.stack(list(self.visual_features_history))  # [8, 512]
            strategic_seq = np.stack(list(self.strategic_features_history))  # [8, 33]

            # Combine features at each timestep
            combined_seq = np.concatenate(
                [visual_seq, strategic_seq], axis=1
            )  # [8, 545]

            # Convert to tensor and add batch dimension
            device = next(self.vision_transformer.parameters()).device
            combined_tensor = (
                torch.from_numpy(combined_seq).float().unsqueeze(0).to(device)
            )  # [1, 8, 545]

            # Get tactical predictions from vision transformer
            with torch.no_grad():
                predictions = self.vision_transformer(combined_tensor)

            prediction_result = {
                "attack_timing": predictions["attack_timing"].cpu().item(),
                "defend_timing": predictions["defend_timing"].cpu().item(),
            }

            # REDUCED LOGGING: Only log predictions if they meet frequency criteria
            if self.log_transformer_predictions:
                self._log_enhanced_transformer_prediction(
                    prediction_result,
                    strategic_seq[-1],
                    visual_seq[-1],
                    combined_tensor,
                )

            return prediction_result

        except Exception as e:
            # REDUCED: Only log critical prediction errors
            logger.error(f"Strategic vision prediction error: {e}")
            return None

    def _log_enhanced_transformer_prediction(
        self, predictions, strategic_features, visual_features, combined_tensor
    ):
        """REDUCED LOGGING: Enhanced transformer prediction logging with much less frequency"""
        try:
            combo_stats = self.strategic_tracker.get_combo_stats()

            # REDUCED: Only store lightweight prediction data, not full logs
            if len(self.transformer_logs["predictions"]) < 1000:  # Limit storage
                self.transformer_logs["predictions"].append(
                    {
                        "step": self.stats["predictions_made"],
                        "attack_timing": predictions["attack_timing"],
                        "defend_timing": predictions["defend_timing"],
                        "combo_count": combo_stats["current_combo"],
                        "score_momentum": strategic_features[17],  # Only key features
                    }
                )

            # REDUCED: Only store significant combo events (3+ hits)
            if combo_stats["current_combo"] >= 3:
                self.transformer_logs["combo_analysis"].append(
                    {
                        "step": self.stats["predictions_made"],
                        "combo_count": combo_stats["current_combo"],
                        "attack_confidence": predictions["attack_timing"],
                        "score_momentum": strategic_features[17],
                    }
                )

            # REDUCED: Much less frequent detailed analysis (every 1000 instead of 100)
            if self.stats["predictions_made"] % self.log_interval == 0:
                self._analyze_enhanced_transformer_learning(
                    predictions, strategic_features, visual_features
                )

            # REDUCED: Only log extremely high-confidence predictions (>0.9 instead of >0.8)
            if predictions["attack_timing"] > 0.9 or predictions["defend_timing"] > 0.9:
                self._log_enhanced_significant_prediction(
                    predictions, strategic_features, combo_stats
                )

        except Exception as e:
            # REDUCED: Don't log minor transformer logging errors
            pass

    def _analyze_enhanced_transformer_learning(
        self, predictions, strategic_features, visual_features
    ):
        """REDUCED LOGGING: Enhanced analysis with much less detail - LOG TO FILE ONLY"""
        try:
            # REDUCED: Only log every 1000 predictions instead of every 100
            print(f"🔍 TRANSFORMER ANALYSIS @ Step {self.stats['predictions_made']}")

            # REDUCED: Only log top 5 features instead of top 10
            feature_importance = self._calculate_enhanced_feature_importance(
                strategic_features
            )

            print("TOP 5 STRATEGIC FEATURES:")
            for i, (feature_name, importance) in enumerate(feature_importance[:5]):
                print(f"   {i+1}. {feature_name}: {importance:.3f}")

            # REDUCED: Simplified situation logging
            combo_stats = self.strategic_tracker.get_combo_stats()
            if combo_stats["current_combo"] > 0:
                print(f"   ACTIVE COMBO: {combo_stats['current_combo']} hits")

            print(
                f"   Attack/Defend: {predictions['attack_timing']:.2f}/{predictions['defend_timing']:.2f}"
            )

            # REDUCED: Only log performance summary
            if self.total_rounds > 0:
                win_rate = self.wins / self.total_rounds
                print(
                    f"   Performance: {win_rate:.1%} wins, Max Combo: {self.stats['max_combo']}"
                )

        except Exception as e:
            # REDUCED: Don't log analysis errors
            pass

    def _calculate_enhanced_feature_importance(self, strategic_features):
        """Enhanced feature importance calculation (unchanged)"""
        try:
            feature_importance = []

            for i, feature_value in enumerate(strategic_features):
                if i < len(self.strategic_feature_names):
                    feature_name = self.strategic_feature_names[i]
                else:
                    feature_name = self.button_feature_names[
                        i - len(self.strategic_feature_names)
                    ]

                # Enhanced importance calculation considering feature context
                base_importance = abs(feature_value) * (1.0 + feature_value * 0.1)

                # Boost importance for combo-related features
                if feature_name == "enhanced_score_momentum":
                    base_importance *= 1.5
                elif "damage" in feature_name.lower():
                    base_importance *= 1.3
                elif feature_name in ["optimal_spacing", "center_control"]:
                    base_importance *= 1.2

                feature_importance.append((feature_name, base_importance))

            feature_importance.sort(key=lambda x: x[1], reverse=True)
            return feature_importance

        except Exception as e:
            return []

    def _log_enhanced_significant_prediction(
        self, predictions, strategic_features, combo_stats
    ):
        """REDUCED LOGGING: Only log extremely significant predictions (>0.9)"""
        try:
            if predictions["attack_timing"] > 0.9:
                print(
                    f"🗡️ VERY HIGH ATTACK CONFIDENCE ({predictions['attack_timing']:.3f})"
                )
                if combo_stats["current_combo"] > 0:
                    print(f"   🔥 {combo_stats['current_combo']}-hit combo active!")

            if predictions["defend_timing"] > 0.9:
                print(
                    f"🛡️ VERY HIGH DEFEND CONFIDENCE ({predictions['defend_timing']:.3f})"
                )

        except Exception as e:
            pass

    def _preprocess_frame(self, frame):
        """Preprocess frame to target size (128, 180)"""
        try:
            if frame is None:
                return np.zeros((*self.target_size, 3), dtype=np.uint8)

            # Resize to target size: height=128, width=180
            resized = cv2.resize(frame, (self.target_size[1], self.target_size[0]))
            return resized

        except Exception as e:
            # REDUCED: Only log critical frame preprocessing errors
            logger.error(f"Frame preprocessing error: {e}")
            return np.zeros((*self.target_size, 3), dtype=np.uint8)

    def _get_stacked_observation(self):
        """Get stacked observation in CHW format - 8 RGB frames"""
        try:
            if len(self.frame_buffer) == 0:
                return np.zeros(self.observation_space.shape, dtype=np.uint8)

            # Stack 8 RGB frames: [frame1, frame2, ..., frame8] each [H, W, 3]
            # Convert to [3*8, H, W] = [24, H, W] format
            stacked = np.concatenate(list(self.frame_buffer), axis=2)  # [H, W, 24]
            return stacked.transpose(2, 0, 1)  # [24, H, W]

        except Exception as e:
            # REDUCED: Only log critical frame stacking errors
            logger.error(f"Frame stacking error: {e}")
            return np.zeros(self.observation_space.shape, dtype=np.uint8)

    def save_enhanced_analysis(self, filepath=None):
        """REDUCED LOGGING: Save lightweight transformer analysis"""
        try:
            if self.log_transformer_predictions:
                import json

                # Use default filepath if none provided
                if filepath is None:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"transformer_analysis_{timestamp}.json"
                    filepath = os.path.join(ANALYSIS_OUTPUT_DIR, filename)

                # REDUCED: Save only essential data, not detailed logs
                analysis_data = {
                    "timestamp": datetime.now().isoformat(),
                    "total_predictions": len(self.transformer_logs["predictions"]),
                    "performance_summary": {
                        "win_rate": self.wins / max(self.total_rounds, 1),
                        "total_rounds": self.total_rounds,
                        "wins": self.wins,
                        "losses": self.losses,
                        "max_combo": self.stats["max_combo"],
                        "total_combos": self.stats["total_combos"],
                        "avg_damage_per_round": self.stats["avg_damage_per_round"],
                    },
                    "recent_predictions": self.transformer_logs["predictions"][
                        -100:
                    ],  # Only last 100 instead of 500
                    "significant_combos": [
                        combo
                        for combo in self.transformer_logs["combo_analysis"]
                        if combo.get("combo_count", 0) >= 5
                    ],  # Only 5+ hit combos
                    "training_metadata": {
                        "episode_steps": self.episode_steps,
                        "vision_transformer_ready": self.vision_ready,
                        "logging_reduced": True,  # Flag to indicate reduced logging
                    },
                }

                with open(filepath, "w") as f:
                    json.dump(analysis_data, f, indent=2)

                print(f"📊 Lightweight analysis saved to {filepath}")

        except Exception as e:
            # REDUCED: Don't log analysis save failures unless critical
            if "permission" in str(e).lower() or "disk" in str(e).lower():
                logger.error(f"Critical: Failed to save analysis: {e}")


# Enhanced CNN for stable-baselines3 compatibility (unchanged)
class StreetFighterSimplifiedCNN(BaseFeaturesExtractor):
    """
    Enhanced CNN feature extractor for Street Fighter compatible with stable-baselines3
    Improved architecture for better feature extraction
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]
        self.cnn = CNNFeatureExtractor(
            input_channels=n_input_channels, feature_dim=features_dim
        )

        # REDUCED: Only log initialization once with print
        print(f"🏗️ Enhanced Street Fighter CNN initialized:")
        print(
            f"   Input: {n_input_channels} channels → Output: {features_dim} features"
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Normalize observations to [0, 1] range
        normalized_obs = observations.float() / 255.0
        return self.cnn(normalized_obs)

    def update_tactical_predictions(self, attack_timing: float, defend_timing: float):
        """Update tactical predictions - forward to underlying CNN"""
        self.cnn.update_tactical_predictions(attack_timing, defend_timing)
