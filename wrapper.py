#!/usr/bin/env python3
"""
wrapper.py - Enhanced Street Fighter II Wrapper with Cross-Attention Vision Transformer
Key improvements:
1. Cross-attention mechanism with Q="what button should I press now?"
2. Separate processing of visual (512), strategy (21), and previous button (12) features
3. Enhanced tactical prediction with button-specific outputs
4. Better feature interaction through multi-head cross-attention
5. REDUCED LOGGING FREQUENCY to prevent large log files
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
from typing import Dict, Tuple
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
        # button names -> action_combo -> later convert to muti binary
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

        # Keep original action combinations for compatibility
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
        ]

        self.num_actions = len(self.action_combinations)
        print(f"🎮 Enhanced Street Fighter Discrete Actions initialized:")
        print(f"   Total discrete actions: {self.num_actions}")

    def discrete_to_multibinary(self, action_index: int) -> np.ndarray:
        """Convert discrete action to multi-binary array for stable-retro"""
        if action_index < 0 or action_index >= self.num_actions:
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


class MultiHeadCrossAttention(nn.Module):
    """Multi-head cross-attention module for feature interaction"""

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear layers for Q, K, V
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [batch_size, seq_len_q, d_model]
            key: [batch_size, seq_len_k, d_model]
            value: [batch_size, seq_len_v, d_model]
            mask: Optional attention mask
        """
        batch_size = query.size(0)

        # Apply linear transformations and reshape for multi-head attention
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

        # Concatenate heads and apply output projection
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.d_model)
        )

        output = self.w_o(attention_output)

        # --- FIX: Return both the processed tensor and the attention weights ---
        return self.layer_norm(output + query), attention_weights

    def scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scaled dot-product attention mechanism"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        return torch.matmul(attention_weights, V), attention_weights


class FeatureGroupProcessor(nn.Module):
    """Process individual feature groups before cross-attention"""

    def __init__(self, input_dim: int, d_model: int, group_name: str):
        super().__init__()
        self.group_name = group_name
        self.input_dim = input_dim
        self.d_model = d_model

        # Feature-specific processing
        if group_name == "visual":
            # Visual features need more complex processing
            self.processor = nn.Sequential(
                nn.Linear(input_dim, d_model * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model * 2, d_model),
                nn.LayerNorm(d_model),
            )
        elif group_name == "strategy":
            # Strategy features are already well-structured
            self.processor = nn.Sequential(
                nn.Linear(input_dim, d_model),
                nn.ReLU(),
                nn.Dropout(0.05),
                nn.LayerNorm(d_model),
            )
        elif group_name == "previous_buttons":
            # Previous button features are binary, need special handling
            self.processor = nn.Sequential(
                nn.Linear(input_dim, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, d_model),
                nn.LayerNorm(d_model),
            )
        else:
            # Default processor
            self.processor = nn.Sequential(
                nn.Linear(input_dim, d_model), nn.ReLU(), nn.LayerNorm(d_model)
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Process features and add positional encoding if needed"""
        return self.processor(features)


class EnhancedCrossAttentionVisionTransformer(nn.Module):
    """
    Enhanced Vision Transformer with Cross-Attention for Street Fighter II

    Architecture:
    1. Process visual (512), strategy (21), and previous button (12) features separately
    2. Use cross-attention with Q="what button should I press now?"
    3. Generate tactical predictions and button-specific outputs
    """

    def __init__(
        self,
        visual_dim: int = 512,
        strategic_dim: int = 21,
        button_dim: int = 12,
        seq_length: int = 8,
    ):
        super().__init__()

        self.visual_dim = visual_dim
        self.strategic_dim = strategic_dim
        self.button_dim = button_dim
        self.seq_length = seq_length

        # Model dimension for transformer
        self.d_model = 256

        # Feature group processors
        self.visual_processor = FeatureGroupProcessor(
            visual_dim, self.d_model, "visual"
        )
        self.strategy_processor = FeatureGroupProcessor(
            strategic_dim, self.d_model, "strategy"
        )
        self.button_processor = FeatureGroupProcessor(
            button_dim, self.d_model, "previous_buttons"
        )

        # Learnable query: "What button should I press now?"
        self.action_query = nn.Parameter(torch.randn(1, 1, self.d_model))

        # Cross-attention modules for each feature group
        self.visual_cross_attention = MultiHeadCrossAttention(self.d_model, num_heads=8)
        self.strategy_cross_attention = MultiHeadCrossAttention(
            self.d_model, num_heads=8
        )
        self.button_cross_attention = MultiHeadCrossAttention(self.d_model, num_heads=8)

        # Feature fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.d_model * 3, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.LayerNorm(self.d_model),
        )

        # Temporal attention across sequence
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=self.d_model, num_heads=8, dropout=0.1, batch_first=True
        )

        # No prediction heads - features go directly to agent.predict

        # Initialize parameters
        self._init_parameters()

        print(f"🎯 Enhanced Cross-Attention Vision Transformer initialized:")
        print(f"   Visual features: {visual_dim} → {self.d_model}")
        print(f"   Strategy features: {strategic_dim} → {self.d_model}")
        print(f"   Previous button features: {button_dim} → {self.d_model}")
        print(f"   Cross-attention heads: 8 per feature group")
        print(f"   Learnable query: 'What button should I press now?'")

    def _init_parameters(self):
        """Initialize parameters with appropriate scaling"""
        # Initialize the action query
        nn.init.normal_(self.action_query, mean=0.0, std=0.02)

        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, combined_sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with cross-attention mechanism

        Args:
            combined_sequence: [batch_size, seq_length, 545] where 545 = 512 + 21 + 12

        Returns:
            Dict with tactical predictions and button probabilities
        """
        try:
            batch_size, seq_len, total_dim = combined_sequence.shape

            # Verify input dimensions
            expected_dim = self.visual_dim + self.strategic_dim + self.button_dim  # 545
            if total_dim != expected_dim:
                logger.warning(
                    f"Input dimension mismatch: got {total_dim}, expected {expected_dim}"
                )
                # Pad or truncate as needed
                if total_dim < expected_dim:
                    padding = torch.zeros(
                        batch_size,
                        seq_len,
                        expected_dim - total_dim,
                        device=combined_sequence.device,
                        dtype=combined_sequence.dtype,
                    )
                    combined_sequence = torch.cat([combined_sequence, padding], dim=-1)
                else:
                    combined_sequence = combined_sequence[:, :, :expected_dim]

            # Split features into groups
            visual_features = combined_sequence[
                :, :, : self.visual_dim
            ]  # [batch, seq, 512]
            strategy_features = combined_sequence[
                :, :, self.visual_dim : self.visual_dim + self.strategic_dim
            ]  # [batch, seq, 21]
            button_features = combined_sequence[
                :, :, self.visual_dim + self.strategic_dim :
            ]  # [batch, seq, 12]

            # Process each feature group
            visual_processed = self.visual_processor(
                visual_features
            )  # [batch, seq, d_model]
            strategy_processed = self.strategy_processor(
                strategy_features
            )  # [batch, seq, d_model]
            button_processed = self.button_processor(
                button_features
            )  # [batch, seq, d_model]

            # Expand action query for batch size
            action_query = self.action_query.expand(
                batch_size, -1, -1
            )  # [batch, 1, d_model]

            # --- FIX: Unpack the tuple (processed_tensor, attention_weights) ---
            visual_attended, visual_weights = self.visual_cross_attention(
                query=action_query, key=visual_processed, value=visual_processed
            )

            strategy_attended, strategy_weights = self.strategy_cross_attention(
                query=action_query, key=strategy_processed, value=strategy_processed
            )

            button_attended, button_weights = self.button_cross_attention(
                query=action_query, key=button_processed, value=button_processed
            )

            # Fuse cross-attention outputs (using the processed tensors)
            fused_features = torch.cat(
                [
                    visual_attended.squeeze(1),
                    strategy_attended.squeeze(1),
                    button_attended.squeeze(1),
                ],
                dim=-1,
            )  # [batch, d_model * 3]

            fused_features = self.feature_fusion(fused_features)  # [batch, d_model]

            # Apply temporal attention across the sequence for context
            # Use the fused features as query, and all processed features as key/value
            all_features = torch.cat(
                [visual_processed, strategy_processed, button_processed], dim=-1
            )  # [batch, seq, d_model * 3]
            all_features = all_features.view(
                batch_size * seq_len, -1
            )  # Reshape for fusion layer
            all_features = self.feature_fusion(all_features)
            all_features = all_features.view(batch_size, seq_len, -1)  # Reshape back

            temporal_output, _ = self.temporal_attention(
                query=fused_features.unsqueeze(1),  # [batch, 1, d_model]
                key=all_features,  # [batch, seq, d_model]
                value=all_features,  # [batch, seq, d_model]
            )

            final_features = temporal_output.squeeze(1)  # [batch, d_model]

            # --- FIX: Return the actual attention weights in the dictionary ---
            return {
                "processed_features": final_features,
                "visual_attention": visual_weights,
                "strategy_attention": strategy_weights,
                "button_attention": button_weights,
            }

        except Exception as e:
            logger.error(f"Cross-attention transformer error: {e}", exc_info=True)
            batch_size = combined_sequence.shape[0]
            device = combined_sequence.device

            # Return safe defaults
            return {
                "processed_features": torch.zeros(
                    batch_size, self.d_model, device=device
                ),
                "visual_attention": torch.zeros(
                    batch_size, self.d_model, device=device
                ),
                "strategy_attention": torch.zeros(
                    batch_size, self.d_model, device=device
                ),
                "button_attention": torch.zeros(
                    batch_size, self.d_model, device=device
                ),
            }

    def get_processed_features(self, combined_sequence: torch.Tensor) -> torch.Tensor:
        """Get processed features for agent.predict"""
        with torch.no_grad():
            output = self.forward(combined_sequence)
            return output["processed_features"]


class StreetFighterVisionWrapper(gym.Wrapper):
    """
    ENHANCED Street Fighter wrapper with Cross-Attention Vision Transformer
    Key improvements:
    1. Cross-attention mechanism with Q="what button should I press now?"
    2. Separate processing of visual, strategy, and previous button features
    3. Enhanced tactical prediction with button-specific outputs
    4. Better feature interaction through multi-head cross-attention
    5. REDUCED logging frequency to prevent large log files
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
            54,
            55,
            56,  # Updated indices for defensive actions
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

        # Cross-attention vision transformer components
        self.cnn_extractor = None
        self.cross_attention_transformer = None
        self.vision_ready = False

        # Current action tracking
        self.current_discrete_action = 0

        self.recent_rewards = deque(maxlen=10)

        # ENHANCED: Statistics with combo tracking and cross-attention analysis
        self.stats = {
            "predictions_made": 0,
            "vision_transformer_ready": False,
            "total_combos": 0,
            "max_combo": 0,
            "avg_damage_per_round": 0.0,
            "defensive_efficiency": 0.0,
            "cross_attention_ready": False,
            "visual_attention_weight": 0.0,
            "strategy_attention_weight": 0.0,
            "button_attention_weight": 0.0,
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
                "attention_analysis": [],  # Track attention weights
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

        print(
            f"🎮 ENHANCED Street Fighter Vision Wrapper with Cross-Attention initialized:"
        )
        print(f"   Resolution: {self.target_size[1]}×{self.target_size[0]}")
        print(f"   Frame stack: {frame_stack} RGB frames (24 channels total)")
        print(
            f"   Strategic Features: {len(self.strategic_feature_names) + len(self.button_feature_names)} total ({len(self.strategic_feature_names)} combat + {len(self.button_feature_names)} button)"
        )
        print(
            f"   Cross-Attention: Visual(512) + Strategy({len(self.strategic_feature_names)}) + PrevButtons({len(self.button_feature_names)})"
        )
        print(
            f"   Action Space: Enhanced Discrete({self.discrete_actions.num_actions}) actions"
        )
        print(
            f"   REDUCED LOGGING: Detail every {self.log_interval} predictions, saves every {self.save_interval_steps} steps"
        )
        print(f"   Analysis Output: {ANALYSIS_OUTPUT_DIR}/")

    def inject_feature_extractor(self, feature_extractor):
        """Inject CNN feature extractor and initialize cross-attention vision transformer"""
        if not self.enable_vision_transformer:
            print("   🔧 Cross-Attention Vision Transformer disabled")
            return

        try:
            self.cnn_extractor = feature_extractor.cnn
            device = next(feature_extractor.parameters()).device
            self.cross_attention_transformer = EnhancedCrossAttentionVisionTransformer(
                visual_dim=512,
                strategic_dim=len(self.strategic_feature_names),
                button_dim=len(self.button_feature_names),
                seq_length=self.frame_stack,
            ).to(device)

            if hasattr(feature_extractor, "inject_cross_attention_components"):
                feature_extractor.inject_cross_attention_components(
                    self.cross_attention_transformer, self
                )

            self.vision_ready = True
            self.stats["vision_transformer_ready"] = True
            self.stats["cross_attention_ready"] = True

            print(
                "   ✅ Enhanced Cross-Attention Vision Transformer initialized and ready!"
            )
            print("   🔄 Features flow: CNN → Cross-Attention → Agent.predict")

        except Exception as e:
            logger.error(
                f"Cross-Attention Vision Transformer injection failed: {e}",
                exc_info=True,
            )
            self.vision_ready = False

    def reset(self, **kwargs):
        """Reset environment and initialize buffers"""
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple):
            observation, info = result
        else:
            observation = result
            info = {}

        self.prev_player_health = self.full_hp
        self.prev_opponent_health = self.full_hp
        self.episode_steps = 0
        self.last_defense_frame = -100
        self.current_discrete_action = 0

        processed_frame = self._preprocess_frame(observation)
        self.frame_buffer.clear()
        for _ in range(self.frame_stack):
            self.frame_buffer.append(processed_frame)

        self.visual_features_history.clear()
        self.strategic_features_history.clear()

        stacked_obs = self._get_stacked_observation()
        return stacked_obs, info

    def step(self, discrete_action):
        """ENHANCED step method with cross-attention predictions and REDUCED LOGGING"""
        self.current_discrete_action = discrete_action
        multibinary_action = self.discrete_actions.discrete_to_multibinary(
            discrete_action
        )

        if discrete_action in self.defend_action_indices:
            self.last_defense_frame = self.episode_steps

        observation, reward, done, truncated, info = self.env.step(multibinary_action)

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

        custom_reward, custom_done = self._calculate_enhanced_reward(
            curr_player_health, curr_opponent_health, score, discrete_action
        )
        self.recent_rewards.append(custom_reward)
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

        if (
            self.log_transformer_predictions
            and self.episode_steps > 0
            and self.episode_steps % self.save_interval_steps == 0
        ):
            self.save_enhanced_analysis()

        self.episode_steps += 1
        info.update(self.stats)

        return stacked_obs, custom_reward, done, truncated, info

    def _calculate_enhanced_reward(
        self, curr_player_health, curr_opponent_health, score, action
    ):
        """ENHANCED reward calculation with combo bonuses and strategic incentives"""
        reward = 0.0
        done = False

        if curr_player_health <= 0 or curr_opponent_health <= 0:
            self.total_rounds += 1
            if curr_opponent_health <= 0 and curr_player_health > 0:
                health_bonus = (curr_player_health / self.full_hp) * 50
                self.wins += 1
                win_rate = self.wins / self.total_rounds
                print(f"🏆 WIN! {self.wins}/{self.total_rounds} ({win_rate:.1%})")
                reward += 100 + health_bonus
                combo_stats = self.strategic_tracker.get_combo_stats()
                if combo_stats["max_combo_this_round"] >= 3:
                    combo_bonus = combo_stats["max_combo_this_round"] * 10
                    if combo_stats["max_combo_this_round"] >= 5:
                        print(
                            f"   🔥 BIG COMBO: {combo_stats['max_combo_this_round']} hits!"
                        )
                    reward += combo_bonus
                    self.stats["max_combo"] = max(
                        self.stats["max_combo"], combo_stats["max_combo_this_round"]
                    )
            elif curr_player_health <= 0 and curr_opponent_health > 0:
                self.losses += 1
                win_rate = self.wins / self.total_rounds
                print(f"💀 LOSS! {self.wins}/{self.total_rounds} ({win_rate:.1%})")
            if self.reset_round:
                done = True

        damage_dealt = max(0, self.prev_opponent_health - curr_opponent_health)
        damage_received = max(0, self.prev_player_health - curr_player_health)
        base_reward = damage_dealt - damage_received
        combo_stats = self.strategic_tracker.get_combo_stats()
        if combo_stats["current_combo"] > 1 and damage_dealt > 0:
            combo_multiplier = 1.0 + (combo_stats["current_combo"] - 1) * 0.2
            reward += base_reward * combo_multiplier
        else:
            reward += base_reward

        reward += self._calculate_strategic_action_bonus(
            action, curr_player_health, curr_opponent_health
        )
        self.total_damage_dealt += damage_dealt
        self.total_damage_received += damage_received
        self.prev_player_health = curr_player_health
        self.prev_opponent_health = curr_opponent_health
        return reward, done

    def _calculate_strategic_action_bonus(self, action, player_health, opponent_health):
        bonus = 0.0
        if player_health <= self.full_hp * 0.3 and action in self.defend_action_indices:
            bonus += 0.5
        if opponent_health <= self.full_hp * 0.3 and action in range(9, 57):
            bonus += 1.0
        return bonus

    def _update_enhanced_stats(self):
        combo_stats = self.strategic_tracker.get_combo_stats()
        if combo_stats["current_combo"] > self.stats["max_combo"]:
            self.stats["max_combo"] = combo_stats["current_combo"]
        if combo_stats["current_combo"] >= 2:
            self.stats["total_combos"] += 1
        if self.total_rounds > 0:
            self.stats["avg_damage_per_round"] = (
                self.total_damage_dealt / self.total_rounds
            )
            if self.total_damage_received > 0:
                self.stats["defensive_efficiency"] = (
                    self.total_damage_dealt / self.total_damage_received
                )
            else:
                self.stats["defensive_efficiency"] = (
                    float("inf") if self.total_damage_dealt > 0 else 0.0
                )

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
                    self.stats["button_attention_weight"] = np.mean(
                        processed_output["button_attention"]
                    )
                    return processed_output["processed_features"]
            return None
        except Exception as e:
            logger.error(
                f"Cross-attention strategic vision pipeline error: {e}", exc_info=True
            )
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
            strategy_seq = strategic_seq[:, : len(self.strategic_feature_names)]
            button_seq = strategic_seq[:, len(self.strategic_feature_names) :]
            combined_seq = np.concatenate(
                [visual_seq, strategy_seq, button_seq], axis=1
            )

            device = next(self.cross_attention_transformer.parameters()).device
            combined_tensor = (
                torch.from_numpy(combined_seq).float().unsqueeze(0).to(device)
            )

            with torch.no_grad():
                output = self.cross_attention_transformer(combined_tensor)

            processed_features = output["processed_features"].cpu().numpy()[0]
            attention_weights = {
                "visual": output["visual_attention"].cpu().numpy()[0],
                "strategy": output["strategy_attention"].cpu().numpy()[0],
                "button": output["button_attention"].cpu().numpy()[0],
            }

            if (
                self.log_transformer_predictions
                and self.stats["predictions_made"] % self.log_interval == 0
            ):
                self._log_cross_attention_processing(
                    processed_features, attention_weights, strategic_seq[-1]
                )

            return {"processed_features": processed_features, **attention_weights}
        except Exception as e:
            logger.error(
                f"Cross-attention feature processing error: {e}", exc_info=True
            )
            return None

    def _log_cross_attention_processing(
        self, processed_features, attention_weights, strategic_features
    ):
        try:
            print(f"🎯 CROSS-ATTENTION @ Step {self.stats['predictions_made']}")
            print(
                f"   🔍 Attention Weights: Visual={np.mean(attention_weights['visual']):.3f}, "
                f"Strategy={np.mean(attention_weights['strategy']):.3f}, "
                f"Buttons={np.mean(attention_weights['button']):.3f}"
            )
        except Exception:
            pass

    def _preprocess_frame(self, frame):
        if frame is None:
            return np.zeros((*self.target_size, 3), dtype=np.uint8)
        return cv2.resize(frame, (self.target_size[1], self.target_size[0]))

    def _get_stacked_observation(self):
        if len(self.frame_buffer) == 0:
            return np.zeros(self.observation_space.shape, dtype=np.uint8)
        stacked = np.concatenate(list(self.frame_buffer), axis=2)
        return stacked.transpose(2, 0, 1)

    def save_enhanced_analysis(self, filepath=None):
        if not self.log_transformer_predictions:
            return
        try:
            import json

            if filepath is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = os.path.join(
                    ANALYSIS_OUTPUT_DIR, f"cross_attention_analysis_{timestamp}.json"
                )

            win_rate = self.wins / max(self.total_rounds, 1)
            analysis_data = {
                "timestamp": datetime.now().isoformat(),
                "performance_summary": {
                    "win_rate": win_rate,
                    "total_rounds": self.total_rounds,
                    "max_combo": self.stats["max_combo"],
                },
                "attention_summary": {
                    "avg_visual": self.stats["visual_attention_weight"],
                    "avg_strategy": self.stats["strategy_attention_weight"],
                    "avg_button": self.stats["button_attention_weight"],
                },
            }
            with open(filepath, "w") as f:
                json.dump(analysis_data, f, indent=2)
            print(f"📊 Cross-attention analysis saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save cross-attention analysis: {e}")


class StreetFighterCrossAttentionCNN(BaseFeaturesExtractor):
    """
    Enhanced CNN feature extractor that integrates with cross-attention transformer
    This class is the main entry point for SB3's policy.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        # This is the CNN part that will be injected into the wrapper
        self.cnn = CNNFeatureExtractor(input_channels=n_input_channels, feature_dim=512)
        self.cross_attention_transformer = None
        self.wrapper_env = None
        self.final_projection = nn.Linear(256, features_dim)

        print(f"🎯 Street Fighter Cross-Attention CNN initialized for SB3:")
        print(f"   CNN Output: 512 features")
        print(f"   Cross-Attention Output: 256 features")
        print(f"   Final Policy Input: {features_dim} features")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        This forward pass is called by the SB3 agent.
        It relies on the wrapper to perform the full pipeline processing.
        """
        if self.wrapper_env is not None and self.wrapper_env.vision_ready:
            batch_size = observations.shape[0]
            processed_features_batch = []

            if hasattr(self.wrapper_env, "envs"):  # VecEnv
                for i in range(batch_size):
                    env = self.wrapper_env.envs[i].env
                    if len(env.visual_features_history) == env.frame_stack:
                        # Re-run pipeline for this observation
                        output = env._get_cross_attention_processed_output()
                        if output is not None and "processed_features" in output:
                            processed_features_batch.append(
                                torch.tensor(
                                    output["processed_features"], device=self.device
                                )
                            )
                        else:  # Fallback
                            processed_features_batch.append(
                                torch.zeros(
                                    self.final_projection.in_features,
                                    device=self.device,
                                )
                            )
                    else:  # Not ready yet
                        processed_features_batch.append(
                            torch.zeros(
                                self.final_projection.in_features, device=self.device
                            )
                        )

                if processed_features_batch:
                    # Project from 256 to the final features_dim
                    stacked_features = torch.stack(processed_features_batch)
                    return self.final_projection(stacked_features)

        # Fallback if the pipeline isn't ready or something fails
        normalized_obs = observations.float() / 255.0
        visual_features = self.cnn(normalized_obs)
        # Project from 512 (CNN) -> 256 (like transformer) -> final_dim
        projected_to_256 = visual_features[:, :256]
        return self.final_projection(projected_to_256)

    def inject_cross_attention_components(
        self, cross_attention_transformer, wrapper_env
    ):
        """Inject cross-attention transformer and wrapper environment"""
        self.cross_attention_transformer = cross_attention_transformer
        self.wrapper_env = wrapper_env  # This is the VecEnv
        print("✅ Cross-attention components injected into SB3 CNN")
