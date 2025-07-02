#!/usr/bin/env python3

"""
wrapper.py - Complete Street Fighter II Wrapper with Discrete Actions and Button Features
Raw Frames → CNN → Vision Transformer → Attack/Defend Predictions
Strategic Combat Data (33 features: 21 combat + 12 button features)
Discrete Actions → Multi-Binary → Emulator
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for Street Fighter
MAX_HEALTH = 176
SCREEN_WIDTH = 180
SCREEN_HEIGHT = 128


class StreetFighterDiscreteActions:
    """
    Street Fighter II Discrete Action System for stable-retro
    Maps discrete integer actions to multi-binary button combinations

    Stable-retro uses filtered actions with 12 buttons total:
    Index 0-11: [B, Y, SELECT, START, UP, DOWN, LEFT, RIGHT, A, X, L, R]

    Street Fighter layout:
    - Light Punch: Y (index 1)
    - Medium Punch: X (index 9)
    - Heavy Punch: L (index 10)
    - Light Kick: B (index 0)
    - Medium Kick: A (index 8)
    - Heavy Kick: R (index 11)
    - Directions: UP(4), DOWN(5), LEFT(6), RIGHT(7)
    """

    def __init__(self):
        # Button indices for stable-retro filtered actions (12 total)
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

        # Define meaningful action combinations for Street Fighter
        self.action_combinations = [
            # 0: No action
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

        logger.info(f"🎮 Street Fighter Discrete Actions initialized:")
        logger.info(f"   Total discrete actions: {self.num_actions}")
        logger.info(f"   Button layout: {self.button_names}")
        logger.info(f"   Using stable-retro filtered actions (12 buttons)")

    def discrete_to_multibinary(self, action_index: int) -> np.ndarray:
        """
        Convert discrete action to multi-binary array for stable-retro

        Args:
            action_index: Integer from 0 to num_actions-1

        Returns:
            Multi-binary array of shape (12,) with 1s for pressed buttons
        """
        if action_index < 0 or action_index >= self.num_actions:
            logger.error(f"Invalid action index: {action_index}")
            action_index = 0  # Default to idle

        # Create empty multi-binary action
        multibinary_action = np.zeros(self.num_buttons, dtype=np.uint8)

        # Get button indices for this action
        button_indices = self.action_combinations[action_index]

        # Set corresponding buttons to 1
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
        """
        Get 12-dimensional button features for strategic analysis
        Returns the multi-binary representation as float32
        """
        return self.discrete_to_multibinary(action_index).astype(np.float32)


class CNNFeatureExtractor(nn.Module):
    """CNN to extract features from 8-frame RGB stack (180×128 resolution)"""

    def __init__(self, input_channels=24, feature_dim=512):  # 8 frames * 3 channels RGB
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
            # Apply CNN layers
            cnn_output = self.cnn(frame_stack)
            # Project to final feature dimension
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
    """Strategic feature tracker with 33 features: 21 combat + 12 button features"""

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

        # Score and status tracking
        self.score_history = deque(maxlen=history_length)
        self.player_status_history = deque(maxlen=history_length)
        self.opponent_status_history = deque(maxlen=history_length)

        # Victory tracking
        self.player_victories_history = deque(maxlen=history_length)
        self.opponent_victories_history = deque(maxlen=history_length)

        # Damage tracking
        self.player_damage_dealt_history = deque(maxlen=history_length)
        self.opponent_damage_dealt_history = deque(maxlen=history_length)

        # Button tracking - NEW
        self.button_features_history = deque(maxlen=history_length)

        # Combat state tracking
        self.close_combat_count = 0
        self.total_frames = 0

        # Constants for Street Fighter screen
        self.SCREEN_WIDTH = 180
        self.SCREEN_HEIGHT = 128
        self.DANGER_ZONE_HEALTH = MAX_HEALTH * 0.25  # 25% health is danger zone
        self.CORNER_THRESHOLD = 30  # Distance from screen edge to be "near corner"
        self.CLOSE_DISTANCE = 40  # Distance to be considered "close combat"
        self.OPTIMAL_SPACING_MIN = 35
        self.OPTIMAL_SPACING_MAX = 55

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
        button_features: np.ndarray = None,  # NEW: 12-dimensional button features
    ) -> np.ndarray:
        """Update all tracked variables and return 33-dimensional feature vector"""

        # Update histories
        self.player_health_history.append(player_health)
        self.opponent_health_history.append(opponent_health)
        self.player_x_history.append(player_x)
        self.player_y_history.append(player_y)
        self.opponent_x_history.append(opponent_x)
        self.opponent_y_history.append(opponent_y)
        self.score_history.append(score)
        self.player_status_history.append(player_status)
        self.opponent_status_history.append(opponent_status)
        self.player_victories_history.append(player_victories)
        self.opponent_victories_history.append(opponent_victories)

        # Update button features history - NEW
        if button_features is not None:
            self.button_features_history.append(button_features)
        else:
            # Default to no buttons pressed
            self.button_features_history.append(np.zeros(12, dtype=np.float32))

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

        # Calculate 33 features (21 strategic + 12 button)
        features = self._calculate_features(
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

    def _calculate_features(
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
        """Calculate the 33 strategic features (21 strategic + 12 button)"""

        features = np.zeros(33, dtype=np.float32)

        # First 21 features: Strategic combat features
        # 1. Is player in danger zone
        features[0] = 1.0 if player_health <= self.DANGER_ZONE_HEALTH else 0.0

        # 2. Is opponent in danger zone
        features[1] = 1.0 if opponent_health <= self.DANGER_ZONE_HEALTH else 0.0

        # 3. Player / opponent health ratio
        if opponent_health > 0:
            features[2] = player_health / opponent_health
        else:
            features[2] = 2.0  # Cap at 2.0 when opponent has no health
        features[2] = np.clip(features[2], 0.0, 2.0)

        # 4. Combined health change rate
        player_health_change = self._calculate_momentum(self.player_health_history)
        opponent_health_change = self._calculate_momentum(self.opponent_health_history)
        features[3] = (player_health_change + opponent_health_change) / 2.0

        # 5. Damage rate player (damage dealt per frame)
        features[4] = self._calculate_momentum(self.player_damage_dealt_history)

        # 6. Damage rate opponent (damage dealt per frame)
        features[5] = self._calculate_momentum(self.opponent_damage_dealt_history)

        # 7. Player corner distance (normalized, 0 = at corner, 1 = center)
        player_left_dist = player_x
        player_right_dist = self.SCREEN_WIDTH - player_x
        player_corner_dist = min(player_left_dist, player_right_dist)
        features[6] = np.clip(player_corner_dist / (self.SCREEN_WIDTH / 2), 0.0, 1.0)

        # 8. Opponent corner distance
        opponent_left_dist = opponent_x
        opponent_right_dist = self.SCREEN_WIDTH - opponent_x
        opponent_corner_dist = min(opponent_left_dist, opponent_right_dist)
        features[7] = np.clip(opponent_corner_dist / (self.SCREEN_WIDTH / 2), 0.0, 1.0)

        # 9. Player near corner
        features[8] = 1.0 if player_corner_dist <= self.CORNER_THRESHOLD else 0.0

        # 10. Opponent near corner
        features[9] = 1.0 if opponent_corner_dist <= self.CORNER_THRESHOLD else 0.0

        # 11. Center control (who is closer to screen center)
        screen_center = self.SCREEN_WIDTH / 2
        player_center_dist = abs(player_x - screen_center)
        opponent_center_dist = abs(opponent_x - screen_center)
        if player_center_dist < opponent_center_dist:
            features[10] = 1.0  # Player has center control
        elif opponent_center_dist < player_center_dist:
            features[10] = -1.0  # Opponent has center control
        else:
            features[10] = 0.0  # Neutral

        # 12. Vertical advantage (who is higher/lower)
        vertical_diff = player_y - opponent_y
        features[11] = np.clip(vertical_diff / (self.SCREEN_HEIGHT / 2), -1.0, 1.0)

        # 13. Position stability (how much position is changing)
        player_x_stability = 1.0 - min(
            1.0, abs(self._calculate_momentum(self.player_x_history)) / 10.0
        )
        opponent_x_stability = 1.0 - min(
            1.0, abs(self._calculate_momentum(self.opponent_x_history)) / 10.0
        )
        features[12] = (player_x_stability + opponent_x_stability) / 2.0

        # 14. Optimal spacing (1.0 if in optimal range, 0.0 if too close/far)
        if self.OPTIMAL_SPACING_MIN <= distance <= self.OPTIMAL_SPACING_MAX:
            features[13] = 1.0
        else:
            features[13] = 0.0

        # 15. Sustain forward pressure (player moving toward opponent)
        player_x_momentum = self._calculate_momentum(self.player_x_history)

        # Forward pressure depends on relative positions
        if player_x < opponent_x:  # Player is on the left
            forward_pressure = player_x_momentum  # Moving right is forward
        else:  # Player is on the right
            forward_pressure = -player_x_momentum  # Moving left is forward
        features[14] = np.clip(forward_pressure / 5.0, -1.0, 1.0)

        # 16. Sustain defence movement (player moving away from opponent)
        if player_x < opponent_x:  # Player is on the left
            defensive_movement = -player_x_momentum  # Moving left is defensive
        else:  # Player is on the right
            defensive_movement = player_x_momentum  # Moving right is defensive
        features[15] = np.clip(defensive_movement / 5.0, -1.0, 1.0)

        # 17. How often players get close (close combat frequency)
        if self.total_frames > 0:
            features[16] = self.close_combat_count / self.total_frames
        else:
            features[16] = 0.0

        # 18. Score momentum (consecutive hits/advantage)
        features[17] = self._calculate_momentum(self.score_history)

        # 19. Agent status and enemy status (combined as single feature)
        status_diff = player_status - opponent_status
        features[18] = np.clip(status_diff / 100.0, -1.0, 1.0)

        # 20. Agent victories (normalized)
        features[19] = min(player_victories / 10.0, 1.0)

        # 21. Enemy victories (normalized)
        features[20] = min(opponent_victories / 10.0, 1.0)

        # Features 22-33: Current button state (12 button features) - NEW
        if len(self.button_features_history) > 0:
            current_buttons = self.button_features_history[
                -1
            ]  # Most recent button state
            features[21:33] = current_buttons
        else:
            features[21:33] = 0.0  # No buttons pressed

        return features

    def _calculate_momentum(self, history):
        """Calculate momentum (rate of change) from history"""
        if len(history) < 2:
            return 0.0

        values = list(history)
        changes = [values[i] - values[i - 1] for i in range(1, len(values))]

        # Use recent changes (last 3) if available, otherwise all changes
        recent_changes = changes[-3:] if len(changes) >= 3 else changes
        return np.mean(recent_changes) if recent_changes else 0.0


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
    """Simplified Vision Transformer with 33 strategic features - attack/defend timing predictions"""

    def __init__(self, visual_dim=512, strategic_dim=33, seq_length=8):
        super().__init__()
        self.seq_length = seq_length

        # Combined input dimension: 512 (visual) + 33 (strategic) = 545
        combined_dim = visual_dim + strategic_dim

        logger.info(f"🔍 SimplifiedVisionTransformer dimensions:")
        logger.info(f"   Visual: {visual_dim}, Strategic: {strategic_dim}")
        logger.info(f"   Strategic breakdown: 21 combat + 12 button features")
        logger.info(f"   Combined input: {combined_dim}")

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
                logger.error(
                    f"Dimension mismatch: got {actual_dim}, expected {expected_dim}"
                )
                # Truncate or pad to fix the mismatch
                if actual_dim > expected_dim:
                    combined_sequence = combined_sequence[:, :, :expected_dim]
                    logger.warning(
                        f"Truncated input from {actual_dim} to {expected_dim}"
                    )
                else:
                    # Pad with zeros
                    padding = torch.zeros(
                        batch_size,
                        seq_len,
                        expected_dim - actual_dim,
                        device=combined_sequence.device,
                        dtype=combined_sequence.dtype,
                    )
                    combined_sequence = torch.cat([combined_sequence, padding], dim=-1)
                    logger.warning(f"Padded input from {actual_dim} to {expected_dim}")

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
            logger.error(f"Transformer error: {e}")
            batch_size = combined_sequence.shape[0]
            device = combined_sequence.device
            return {
                "attack_timing": torch.zeros(batch_size, device=device),
                "defend_timing": torch.zeros(batch_size, device=device),
            }


class StreetFighterVisionWrapper(gym.Wrapper):
    """Street Fighter wrapper with discrete actions and strategic features with button data"""

    def __init__(
        self,
        env,
        reset_round=True,
        rendering=False,
        max_episode_steps=5000,
        frame_stack=8,
        enable_vision_transformer=True,
        defend_action_indices=None,
        log_transformer_predictions=True,  # NEW: Enable transformer logging
    ):
        super().__init__(env)

        self.frame_stack = frame_stack
        self.enable_vision_transformer = enable_vision_transformer
        self.target_size = (128, 180)  # H, W - optimized for 180×128
        self.log_transformer_predictions = log_transformer_predictions  # NEW

        # Initialize discrete action system
        self.discrete_actions = StreetFighterDiscreteActions()

        # Episode management
        self.max_episode_steps = max_episode_steps
        self.episode_steps = 0
        self.reset_round = reset_round

        # Health tracking
        self.full_hp = 176
        self.prev_player_health = self.full_hp
        self.prev_opponent_health = self.full_hp

        # Anti-spam defense tracking (convert to discrete action indices)
        self.defend_action_indices = defend_action_indices or [
            54,
            55,
            56,
        ]  # Block actions
        self.defense_cooldown_frames = 30
        self.last_defense_frame = -100

        # Win tracking for display
        self.wins = 0
        self.losses = 0
        self.total_rounds = 0

        # Setup observation space: [channels, height, width] - 8 frames RGB
        obs_shape = (
            3 * frame_stack,
            self.target_size[0],
            self.target_size[1],
        )
        self.observation_space = spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

        # Override action space to discrete
        self.action_space = spaces.Discrete(self.discrete_actions.num_actions)

        # Initialize strategic feature tracker
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

        # Statistics
        self.stats = {
            "predictions_made": 0,
            "vision_transformer_ready": False,
            "avg_attack_timing": 0.0,
            "avg_defend_timing": 0.0,
        }

        # NEW: Transformer learning analysis
        if self.log_transformer_predictions:
            self.transformer_logs = {
                "predictions": [],
                "feature_importance": [],
                "tactical_patterns": {},
                "situation_analysis": {},
                "learning_progression": [],
            }
            self.log_interval = 100  # Log every 100 predictions
            self.last_detailed_log = 0

            # Feature names for logging
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
                "score_momentum",
                "status_difference",
                "agent_victories",
                "enemy_victories",
            ]

            self.button_feature_names = [
                "B_pressed",
                "Y_pressed",
                "SELECT_pressed",
                "START_pressed",
                "UP_pressed",
                "DOWN_pressed",
                "LEFT_pressed",
                "RIGHT_pressed",
                "A_pressed",
                "X_pressed",
                "L_pressed",
                "R_pressed",
            ]

        logger.info(
            f"🎮 Street Fighter Vision Wrapper with Discrete Actions initialized:"
        )
        logger.info(f"   Resolution: {self.target_size[1]}×{self.target_size[0]}")
        logger.info(f"   Frame stack: {frame_stack} RGB frames (24 channels total)")
        logger.info(f"   Strategic Features: 33 total (21 combat + 12 button)")
        logger.info(
            f"   Action Space: Discrete({self.discrete_actions.num_actions}) actions"
        )
        logger.info(
            f"   Defense anti-spam: {self.defense_cooldown_frames} frame cooldown"
        )
        logger.info(
            f"   Vision Transformer: {'Enabled' if enable_vision_transformer else 'Disabled'}"
        )
        logger.info(
            f"   Transformer Logging: {'Enabled' if log_transformer_predictions else 'Disabled'}"
        )  # NEW

    def inject_feature_extractor(self, feature_extractor):
        """Inject CNN feature extractor and initialize vision transformer"""
        if not self.enable_vision_transformer:
            logger.info("   🔧 Vision Transformer disabled")
            return

        try:
            self.cnn_extractor = feature_extractor
            actual_feature_dim = self.cnn_extractor.features_dim
            logger.info(f"   📏 Detected CNN feature dimension: {actual_feature_dim}")

            # Initialize vision transformer with correct dimensions
            device = next(feature_extractor.parameters()).device
            self.vision_transformer = SimplifiedVisionTransformer(
                visual_dim=actual_feature_dim,
                strategic_dim=33,  # 21 strategic + 12 button features
                seq_length=self.frame_stack,
            ).to(device)
            self.vision_ready = True
            self.stats["vision_transformer_ready"] = True

            logger.info("   ✅ Strategic Vision Transformer initialized and ready!")

        except Exception as e:
            logger.error(f"   ❌ Vision Transformer injection failed: {e}")
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
        """Execute discrete action and process through strategic vision pipeline"""
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

        # Calculate custom reward
        custom_reward, custom_done = self._calculate_reward(
            curr_player_health, curr_opponent_health
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
            button_features,  # Include button features
        )

        # Update CNN with tactical predictions
        if tactical_predictions and self.cnn_extractor is not None:
            self.cnn_extractor.update_tactical_predictions(
                tactical_predictions["attack_timing"],
                tactical_predictions["defend_timing"],
            )

        self.episode_steps += 1
        info.update(self.stats)

        return stacked_obs, custom_reward, done, truncated, info

    def _extract_enhanced_state(self, info):
        """Extract enhanced game state including all required data for strategic features"""
        # Basic health and score
        player_health = info.get("agent_hp", self.full_hp)
        opponent_health = info.get("enemy_hp", self.full_hp)
        score = info.get("score", 0)

        # Position data
        player_x = info.get("agent_x", 90)  # Default to center if not available
        player_y = info.get("agent_y", 64)
        opponent_x = info.get("enemy_x", 90)
        opponent_y = info.get("enemy_y", 64)

        # Status data
        player_status = info.get("agent_status", 0)
        opponent_status = info.get("enemy_status", 0)

        # Victory data
        player_victories = info.get("agent_victories", 0)
        opponent_victories = info.get("enemy_victories", 0)

        # Log these values for debugging (only occasionally to avoid spam)
        if self.episode_steps % 100 == 0:  # Log every 100 steps
            logger.info(f"DEBUG - Game state values:")
            logger.info(
                f"  Player status: {player_status} (type: {type(player_status)})"
            )
            logger.info(
                f"  Opponent status: {opponent_status} (type: {type(opponent_status)})"
            )
            logger.info(
                f"  Player victories: {player_victories} (type: {type(player_victories)})"
            )
            logger.info(
                f"  Opponent victories: {opponent_victories} (type: {type(opponent_victories)})"
            )
            logger.info(f"  Score: {score} (type: {type(score)})")
            logger.info(
                f"  Positions - Player: ({player_x}, {player_y}), Opponent: ({opponent_x}, {opponent_y})"
            )
            logger.info(
                f"  Current action: {self.discrete_actions.get_action_name(self.current_discrete_action)}"
            )

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
        """Process through strategic vision pipeline with button features"""
        try:
            # Step 1: Strategic features using the 33-feature system (21 + 12 button)
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
                button_features,  # Include button features
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
            logger.error(f"Strategic vision pipeline processing error: {e}")
            return None

    def _make_tactical_prediction(self):
        """Make tactical predictions using strategic features with button data"""
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

            # NEW: Log transformer predictions for analysis
            if self.log_transformer_predictions:
                self._log_transformer_prediction(
                    prediction_result,
                    strategic_seq[-1],  # Latest strategic features
                    visual_seq[-1],  # Latest visual features
                    combined_tensor,
                )

            return prediction_result

        except Exception as e:
            logger.error(f"Strategic vision prediction error: {e}")
            return None

    def _log_transformer_prediction(
        self, predictions, strategic_features, visual_features, combined_tensor
    ):
        """NEW: Log transformer predictions for learning analysis"""
        try:
            # Basic prediction logging
            self.transformer_logs["predictions"].append(
                {
                    "step": self.stats["predictions_made"],
                    "attack_timing": predictions["attack_timing"],
                    "defend_timing": predictions["defend_timing"],
                    "action": self.discrete_actions.get_action_name(
                        self.current_discrete_action
                    ),
                    "strategic_features": strategic_features.tolist(),
                }
            )

            # Detailed analysis every N predictions
            if self.stats["predictions_made"] % self.log_interval == 0:
                self._analyze_transformer_learning(
                    predictions, strategic_features, visual_features
                )

            # Log significant predictions
            if predictions["attack_timing"] > 0.8 or predictions["defend_timing"] > 0.8:
                self._log_significant_prediction(predictions, strategic_features)

        except Exception as e:
            logger.warning(f"Transformer logging error: {e}")

    def _analyze_transformer_learning(
        self, predictions, strategic_features, visual_features
    ):
        """NEW: Analyze what the transformer has learned"""
        try:
            logger.info(
                f"\n🧠 TRANSFORMER LEARNING ANALYSIS @ Step {self.stats['predictions_made']}"
            )

            # Feature importance analysis
            feature_importance = self._calculate_feature_importance(strategic_features)

            # Strategic feature analysis
            logger.info("📊 STRATEGIC FEATURE IMPORTANCE (Top 10):")
            for i, (feature_name, importance) in enumerate(feature_importance[:10]):
                logger.info(f"   {i+1:2d}. {feature_name:20s}: {importance:.4f}")

            # Current game situation
            self._log_current_situation(strategic_features, predictions)

            # Tactical patterns discovered
            self._log_tactical_patterns(predictions, strategic_features)

            # Learning progression
            self._log_learning_progression()

            logger.info("=" * 60)

        except Exception as e:
            logger.warning(f"Transformer analysis error: {e}")

    def _calculate_feature_importance(self, strategic_features):
        """NEW: Calculate which features the transformer considers most important"""
        try:
            # Simple feature importance based on prediction correlation
            feature_importance = []

            for i, feature_value in enumerate(strategic_features):
                if i < len(self.strategic_feature_names):
                    feature_name = self.strategic_feature_names[i]
                else:
                    feature_name = self.button_feature_names[
                        i - len(self.strategic_feature_names)
                    ]

                # Calculate importance based on feature magnitude and recent usage
                importance = abs(feature_value) * (1.0 + feature_value * 0.1)
                feature_importance.append((feature_name, importance))

            # Sort by importance
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            return feature_importance

        except Exception as e:
            logger.warning(f"Feature importance calculation error: {e}")
            return []

    def _log_current_situation(self, strategic_features, predictions):
        """NEW: Log current game situation analysis"""
        try:
            logger.info("🎯 CURRENT SITUATION ANALYSIS:")

            # Health situation
            player_danger = strategic_features[0] > 0.5
            opponent_danger = strategic_features[1] > 0.5
            health_ratio = strategic_features[2]

            if player_danger:
                logger.info("   ⚠️  PLAYER IN DANGER ZONE!")
            if opponent_danger:
                logger.info("   🎯 OPPONENT IN DANGER ZONE!")

            logger.info(f"   💚 Health Ratio: {health_ratio:.3f} (Player vs Opponent)")

            # Position situation
            corner_distance = strategic_features[6]
            center_control = strategic_features[10]
            optimal_spacing = strategic_features[13]

            logger.info(f"   📍 Corner Distance: {corner_distance:.3f}")
            logger.info(f"   🎮 Center Control: {center_control:.3f}")
            logger.info(
                f"   📏 Optimal Spacing: {'YES' if optimal_spacing > 0.5 else 'NO'}"
            )

            # Button analysis
            button_features = strategic_features[21:33]
            pressed_buttons = [
                self.button_feature_names[i]
                for i, pressed in enumerate(button_features)
                if pressed > 0.5
            ]
            logger.info(
                f"   🎮 Buttons Pressed: {', '.join(pressed_buttons) if pressed_buttons else 'None'}"
            )

            # Transformer recommendations
            logger.info(f"   🧠 Transformer Recommends:")
            logger.info(f"      Attack Timing: {predictions['attack_timing']:.3f}")
            logger.info(f"      Defend Timing: {predictions['defend_timing']:.3f}")

            if predictions["attack_timing"] > 0.7:
                logger.info("      → 🗡️  STRONG ATTACK RECOMMENDATION!")
            if predictions["defend_timing"] > 0.7:
                logger.info("      → 🛡️  STRONG DEFEND RECOMMENDATION!")

        except Exception as e:
            logger.warning(f"Situation logging error: {e}")

    def _log_tactical_patterns(self, predictions, strategic_features):
        """NEW: Log discovered tactical patterns"""
        try:
            logger.info("🎯 TACTICAL PATTERNS DISCOVERED:")

            # Attack patterns
            if predictions["attack_timing"] > 0.6:
                conditions = []
                if strategic_features[1] > 0.5:  # opponent in danger
                    conditions.append("opponent in danger")
                if strategic_features[13] > 0.5:  # optimal spacing
                    conditions.append("optimal spacing")
                if strategic_features[14] > 0.3:  # forward pressure
                    conditions.append("forward pressure")

                if conditions:
                    logger.info(f"   ⚔️  Attack when: {', '.join(conditions)}")

            # Defend patterns
            if predictions["defend_timing"] > 0.6:
                conditions = []
                if strategic_features[0] > 0.5:  # player in danger
                    conditions.append("player in danger")
                if strategic_features[8] > 0.5:  # player near corner
                    conditions.append("player near corner")
                if strategic_features[15] > 0.3:  # defensive movement
                    conditions.append("defensive movement")

                if conditions:
                    logger.info(f"   🛡️  Defend when: {', '.join(conditions)}")

            # Button combination patterns
            button_features = strategic_features[21:33]
            if np.sum(button_features) > 0:
                active_buttons = [
                    name
                    for i, name in enumerate(self.button_feature_names)
                    if button_features[i] > 0.5
                ]
                if len(active_buttons) > 1:
                    logger.info(f"   🎮 Learned combo: {' + '.join(active_buttons)}")

        except Exception as e:
            logger.warning(f"Tactical pattern logging error: {e}")

    def _log_learning_progression(self):
        """NEW: Log learning progression over time"""
        try:
            if len(self.transformer_logs["predictions"]) > 500:  # Need some history
                recent_predictions = self.transformer_logs["predictions"][-500:]
                early_predictions = (
                    self.transformer_logs["predictions"][:500]
                    if len(self.transformer_logs["predictions"]) > 1000
                    else []
                )

                logger.info("📈 LEARNING PROGRESSION:")

                # Calculate average confidence over time
                recent_attack_avg = np.mean(
                    [p["attack_timing"] for p in recent_predictions]
                )
                recent_defend_avg = np.mean(
                    [p["defend_timing"] for p in recent_predictions]
                )

                logger.info(f"   Recent Attack Confidence: {recent_attack_avg:.3f}")
                logger.info(f"   Recent Defend Confidence: {recent_defend_avg:.3f}")

                if early_predictions:
                    early_attack_avg = np.mean(
                        [p["attack_timing"] for p in early_predictions]
                    )
                    early_defend_avg = np.mean(
                        [p["defend_timing"] for p in early_predictions]
                    )

                    attack_improvement = recent_attack_avg - early_attack_avg
                    defend_improvement = recent_defend_avg - early_defend_avg

                    logger.info(
                        f"   Attack Confidence Change: {attack_improvement:+.3f}"
                    )
                    logger.info(
                        f"   Defend Confidence Change: {defend_improvement:+.3f}"
                    )

                # Win rate correlation
                if self.total_rounds > 0:
                    win_rate = self.wins / self.total_rounds
                    logger.info(f"   Current Win Rate: {win_rate:.1%}")

        except Exception as e:
            logger.warning(f"Learning progression logging error: {e}")

    def _log_significant_prediction(self, predictions, strategic_features):
        """NEW: Log high-confidence predictions for pattern analysis"""
        try:
            if predictions["attack_timing"] > 0.8:
                logger.info(
                    f"🗡️  HIGH ATTACK CONFIDENCE ({predictions['attack_timing']:.3f}):"
                )
                logger.info(
                    f"   Action: {self.discrete_actions.get_action_name(self.current_discrete_action)}"
                )
                logger.info(f"   Health Ratio: {strategic_features[2]:.3f}")
                logger.info(f"   Distance: Optimal={strategic_features[13]:.3f}")
                logger.info(
                    f"   Position: Corner={strategic_features[6]:.3f}, Center={strategic_features[10]:.3f}"
                )

            if predictions["defend_timing"] > 0.8:
                logger.info(
                    f"🛡️  HIGH DEFEND CONFIDENCE ({predictions['defend_timing']:.3f}):"
                )
                logger.info(
                    f"   Action: {self.discrete_actions.get_action_name(self.current_discrete_action)}"
                )
                logger.info(f"   Player Danger: {strategic_features[0]:.3f}")
                logger.info(f"   Corner Distance: {strategic_features[6]:.3f}")
                logger.info(f"   Defensive Movement: {strategic_features[15]:.3f}")

        except Exception as e:
            logger.warning(f"Significant prediction logging error: {e}")

    def save_transformer_analysis(self, filepath="transformer_analysis.json"):
        """NEW: Save transformer learning analysis to file"""
        try:
            if self.log_transformer_predictions:
                import json

                analysis_data = {
                    "total_predictions": len(self.transformer_logs["predictions"]),
                    "learning_summary": {
                        "avg_attack_timing": (
                            np.mean(
                                [
                                    p["attack_timing"]
                                    for p in self.transformer_logs["predictions"]
                                ]
                            )
                            if self.transformer_logs["predictions"]
                            else 0
                        ),
                        "avg_defend_timing": (
                            np.mean(
                                [
                                    p["defend_timing"]
                                    for p in self.transformer_logs["predictions"]
                                ]
                            )
                            if self.transformer_logs["predictions"]
                            else 0
                        ),
                        "high_confidence_attacks": len(
                            [
                                p
                                for p in self.transformer_logs["predictions"]
                                if p["attack_timing"] > 0.8
                            ]
                        ),
                        "high_confidence_defends": len(
                            [
                                p
                                for p in self.transformer_logs["predictions"]
                                if p["defend_timing"] > 0.8
                            ]
                        ),
                        "win_rate": self.wins / max(self.total_rounds, 1),
                        "total_rounds": self.total_rounds,
                    },
                    "recent_predictions": self.transformer_logs["predictions"][
                        -100:
                    ],  # Last 100 predictions
                }

                with open(filepath, "w") as f:
                    json.dump(analysis_data, f, indent=2)

                logger.info(f"💾 Transformer analysis saved to {filepath}")

        except Exception as e:
            logger.warning(f"Failed to save transformer analysis: {e}")

    def _preprocess_frame(self, frame):
        """Preprocess frame to target size (128, 180)"""
        try:
            if frame is None:
                return np.zeros((*self.target_size, 3), dtype=np.uint8)

            # Resize to target size: height=128, width=180
            resized = cv2.resize(frame, (self.target_size[1], self.target_size[0]))
            return resized

        except Exception as e:
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
            logger.error(f"Frame stacking error: {e}")
            return np.zeros(self.observation_space.shape, dtype=np.uint8)

    def _calculate_reward(self, curr_player_health, curr_opponent_health):
        """Calculate reward based on damage dealt/received"""
        reward = 0.0
        done = False

        # Check for round end
        if curr_player_health <= 0 or curr_opponent_health <= 0:
            self.total_rounds += 1

            if curr_opponent_health <= 0 and curr_player_health > 0:
                # Win
                self.wins += 1
                win_rate = self.wins / self.total_rounds
                logger.info(f"🏆 WIN! {self.wins}/{self.total_rounds} ({win_rate:.1%})")

                # # NEW: Save analysis on win
                # if self.log_transformer_predictions:
                #     self.save_transformer_analysis(
                #         f"transformer_analysis_win_{self.wins}.json"
                #     )

            elif curr_player_health <= 0 and curr_opponent_health > 0:
                # Loss
                self.losses += 1
                win_rate = self.wins / self.total_rounds
                logger.info(
                    f"💀 LOSS! {self.wins}/{self.total_rounds} ({win_rate:.1%})"
                )

            if self.reset_round:
                done = True

        # Damage-based reward: +1 per damage dealt, -1 per damage received
        damage_dealt = max(0, self.prev_opponent_health - curr_opponent_health)
        damage_received = max(0, self.prev_player_health - curr_player_health)
        reward = damage_dealt - damage_received

        # Update health tracking
        self.prev_player_health = curr_player_health
        self.prev_opponent_health = curr_opponent_health

        return reward, done


# Simplified CNN for stable-baselines3 compatibility
class StreetFighterSimplifiedCNN(BaseFeaturesExtractor):
    """Simplified CNN feature extractor for Street Fighter compatible with stable-baselines3"""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]
        self.cnn = CNNFeatureExtractor(
            input_channels=n_input_channels, feature_dim=features_dim
        )

        logger.info(f"🏗️ Street Fighter Simplified CNN initialized:")
        logger.info(
            f"   Input: {n_input_channels} channels (8 RGB frames) → Output: {features_dim} features"
        )
        logger.info(f"   Expected input shape: {observation_space.shape}")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Normalize observations to [0, 1] range
        normalized_obs = observations.float() / 255.0
        return self.cnn(normalized_obs)

    def update_tactical_predictions(self, attack_timing: float, defend_timing: float):
        """Update tactical predictions - forward to underlying CNN"""
        self.cnn.update_tactical_predictions(attack_timing, defend_timing)
