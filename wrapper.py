#!/usr/bin/env python3

"""
wrapper.py - Simplified Vision Pipeline for Street Fighter II with Position & Score Tracking
Raw Frames → OpenCV → CNN ↘
                           Vision Transformer → Attack/Defend Predictions Only
Health/Score/Position Data → Enhanced Momentum Tracker ↗
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
from typing import Dict, List, Tuple, Optional
import math
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for Street Fighter
MAX_HEALTH = 176
CRITICAL_HEALTH_THRESHOLD = MAX_HEALTH * 0.3
SCREEN_WIDTH = 180  # Game screen width for position normalization
SCREEN_HEIGHT = 128  # Game screen height for position normalization


class EnhancedOpenCVDetector:
    """Simple OpenCV detector for Street Fighter - focuses only on motion"""

    def __init__(self):
        # Simple motion detection only
        self.prev_frame_gray = None
        self.frame_count = 0
        self.motion_history = deque(maxlen=5)  # Track recent motion

        logger.info("🔍 OpenCV detector for Street Fighter:")
        logger.info("   Simple motion detection only")
        logger.info("   No complex activity analysis")

    def detect_activity(self, frame: np.ndarray) -> Dict:
        """
        Simple motion detection for Street Fighter
        Input: frame [128, 180, 3] - RGB frame (H×W×C)
        Output: basic motion analysis
        """
        if frame is None or frame.size == 0:
            return self._empty_detection()

        try:
            self.frame_count += 1

            # Convert to grayscale for motion detection
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            # Detect motion
            motion_info = self._detect_motion(gray)

            # Simple result - just motion data
            result = {
                "motion_detected": motion_info["has_motion"],
                "motion_intensity": motion_info["intensity"],
            }

            return result

        except Exception as e:
            logger.warning(f"OpenCV detection error: {e}")
            return self._empty_detection()

    def _detect_motion(self, gray: np.ndarray) -> Dict:
        """Detect motion for general activity analysis"""
        try:
            if self.prev_frame_gray is None:
                self.prev_frame_gray = gray.copy()
                return {"has_motion": False, "intensity": 0.0, "motion_areas": []}

            # Calculate frame difference
            frame_diff = cv2.absdiff(self.prev_frame_gray, gray)

            # Threshold to get motion areas
            _, motion_mask = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)

            # Calculate motion intensity
            motion_pixels = np.sum(motion_mask > 0)
            total_pixels = motion_mask.shape[0] * motion_mask.shape[1]
            motion_intensity = motion_pixels / total_pixels

            # Find contours
            contours, _ = cv2.findContours(
                motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            motion_areas = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 50:  # Filter small motions
                    x, y, w, h = cv2.boundingRect(contour)
                    motion_areas.append((x, y, w, h))

            # Update motion history
            self.motion_history.append(motion_intensity)

            # Update previous frame
            self.prev_frame_gray = gray.copy()

            return {
                "has_motion": motion_intensity > 0.01,
                "intensity": motion_intensity,
                "motion_areas": motion_areas,
            }

        except Exception as e:
            logger.warning(f"Motion detection error: {e}")
            return {"has_motion": False, "intensity": 0.0, "motion_areas": []}

    def _empty_detection(self) -> Dict:
        """Return empty detection result"""
        return {
            "motion_detected": False,
            "motion_intensity": 0.0,
        }


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

        # Add tactical prediction tracking (simplified to attack/defend only)
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


class EnhancedMomentumTracker:
    """Enhanced tracker for health, score, and relative position momentum in Street Fighter"""

    def __init__(self, history_length=8):
        self.history_length = history_length

        # Health tracking
        self.player_health_history = deque(maxlen=history_length)
        self.opponent_health_history = deque(maxlen=history_length)

        # Score tracking
        self.score_history = deque(maxlen=history_length)

        # Relative position tracking (agent - enemy)
        self.x_diff_history = deque(maxlen=history_length)  # agent_x - enemy_x
        self.y_diff_history = deque(maxlen=history_length)  # agent_y - enemy_y

        # Distance tracking
        self.distance_history = deque(maxlen=history_length)

    def update(
        self,
        player_health: float,
        opponent_health: float,
        score: float,
        player_x: float,
        player_y: float,
        opponent_x: float,
        opponent_y: float,
    ) -> np.ndarray:
        """Update all tracked variables and return enhanced feature vector"""

        # Calculate relative positions (agent - enemy)
        x_diff = player_x - opponent_x  # Positive = agent is to the right
        y_diff = (
            player_y - opponent_y
        )  # Positive = agent is below (depending on coordinate system)

        # Calculate distance between characters
        distance = np.sqrt(x_diff**2 + y_diff**2)

        # Update histories
        self.player_health_history.append(player_health)
        self.opponent_health_history.append(opponent_health)
        self.score_history.append(score)
        self.x_diff_history.append(x_diff)
        self.y_diff_history.append(y_diff)
        self.distance_history.append(distance)

        # Calculate momentum features
        player_health_momentum = self._calculate_momentum(self.player_health_history)
        opponent_health_momentum = self._calculate_momentum(
            self.opponent_health_history
        )
        score_momentum = self._calculate_momentum(self.score_history)

        # Relative position momentum (how the relative position is changing)
        x_diff_momentum = self._calculate_momentum(
            self.x_diff_history
        )  # Moving left/right relative to enemy
        y_diff_momentum = self._calculate_momentum(
            self.y_diff_history
        )  # Moving up/down relative to enemy

        # Distance momentum (approaching/retreating)
        distance_momentum = self._calculate_momentum(self.distance_history)

        # Calculate enhanced features
        health_advantage = player_health - opponent_health

        # Normalize relative positions (-1 to 1 range)
        normalized_x_diff = np.clip(
            x_diff / SCREEN_WIDTH, -1.0, 1.0
        )  # -1 = far left, +1 = far right
        normalized_y_diff = np.clip(
            y_diff / SCREEN_HEIGHT, -1.0, 1.0
        )  # -1 = above, +1 = below
        normalized_distance = min(
            distance / np.sqrt(SCREEN_WIDTH**2 + SCREEN_HEIGHT**2), 1.0
        )

        # Tactical positioning features based on relative position
        is_close_combat = 1.0 if normalized_distance < 0.15 else 0.0  # Very close
        is_medium_range = (
            1.0 if 0.15 <= normalized_distance < 0.4 else 0.0
        )  # Medium range
        is_long_range = 1.0 if normalized_distance >= 0.4 else 0.0  # Long range

        # Relative positioning tactical features
        agent_on_right = 1.0 if x_diff > 20 else 0.0  # Agent significantly to the right
        agent_on_left = 1.0 if x_diff < -20 else 0.0  # Agent significantly to the left
        agents_aligned_x = (
            1.0 if abs(x_diff) < 20 else 0.0
        )  # Agents roughly aligned horizontally

        # Vertical positioning (important for jumps/crouches)
        agent_above = 1.0 if y_diff < -10 else 0.0  # Agent is above enemy (jumping)
        agent_below = 1.0 if y_diff > 10 else 0.0  # Agent is below enemy

        # Movement patterns
        approaching = 1.0 if distance_momentum < -2 else 0.0  # Getting closer
        retreating = 1.0 if distance_momentum > 2 else 0.0  # Getting farther

        # Create enhanced 18-dimensional feature vector focused on relative positioning
        features = np.array(
            [
                # Health momentum (4 features)
                player_health_momentum,
                opponent_health_momentum,
                player_health / MAX_HEALTH,
                opponent_health / MAX_HEALTH,
                # Score and advantage (2 features)
                score_momentum,
                health_advantage / MAX_HEALTH,
                # Relative position momentum (2 features)
                x_diff_momentum,  # How relative X position is changing
                y_diff_momentum,  # How relative Y position is changing
                # Normalized relative positioning (3 features)
                normalized_x_diff,  # Current relative X position
                normalized_y_diff,  # Current relative Y position
                normalized_distance,  # Current distance
                # Distance momentum (1 feature)
                distance_momentum,  # How distance is changing
                # Tactical range positioning (3 features)
                is_close_combat,
                is_medium_range,
                is_long_range,
                # Relative tactical positioning (3 features)
                agent_on_right,
                agent_on_left,
                approaching,
                # History completeness (1 feature)
                len(self.player_health_history) / self.history_length,
            ],
            dtype=np.float32,
        )

        # Debug: Check actual feature count
        actual_count = len(features)
        expected_count = 19  # Count shows this should be 19, not 18
        if actual_count != expected_count:
            logger.warning(
                f"Momentum feature count mismatch: got {actual_count}, expected {expected_count}"
            )

        return features

    def _calculate_momentum(self, history):
        """Calculate momentum (rate of change)"""
        if len(history) < 2:
            return 0.0

        values = list(history)
        changes = [values[i] - values[i - 1] for i in range(1, len(values))]

        # Use recent changes (last 3) if available, otherwise all changes
        recent_changes = changes[-3:] if len(changes) >= 3 else changes
        return np.mean(recent_changes)


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
    """Simplified Vision Transformer - only attack and defend timing predictions"""

    def __init__(
        self, visual_dim=512, opencv_dim=2, momentum_dim=19, seq_length=8
    ):  # Changed from 18 to 19
        super().__init__()
        self.seq_length = seq_length

        # Combined input dimension: 512 + 2 + 19 = 533
        combined_dim = visual_dim + opencv_dim + momentum_dim

        # Debug: Print actual dimensions to catch mismatches
        print(f"🔍 SimplifiedVisionTransformer dimensions:")
        print(
            f"   Visual: {visual_dim}, OpenCV: {opencv_dim}, Momentum: {momentum_dim}"
        )
        print(f"   Expected combined: {combined_dim}")

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

        # Simplified tactical prediction head - only attack and defend timing
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
            expected_dim = 533  # 512 + 2 + 19

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
    """Simplified Street Fighter wrapper with tactical vision pipeline - attack/defend only"""

    def __init__(
        self,
        env,
        reset_round=True,
        rendering=False,
        max_episode_steps=5000,
        frame_stack=8,
        enable_vision_transformer=True,
        defend_action_indices=None,
    ):
        super().__init__(env)

        self.frame_stack = frame_stack
        self.enable_vision_transformer = enable_vision_transformer
        self.target_size = (128, 180)  # H, W - optimized for 180×128

        # Episode management
        self.max_episode_steps = max_episode_steps
        self.episode_steps = 0
        self.reset_round = reset_round

        # Health tracking
        self.full_hp = 176
        self.prev_player_health = self.full_hp
        self.prev_opponent_health = self.full_hp

        # Anti-spam defense tracking
        self.defend_action_indices = defend_action_indices or [4, 5, 6]
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

        # Initialize pipeline components
        self.opencv_detector = EnhancedOpenCVDetector()
        self.momentum_tracker = EnhancedMomentumTracker()

        # Frame and feature buffers
        self.frame_buffer = deque(maxlen=frame_stack)
        self.visual_features_history = deque(maxlen=frame_stack)
        self.opencv_features_history = deque(maxlen=frame_stack)
        self.momentum_features_history = deque(maxlen=frame_stack)

        # Vision transformer components
        self.cnn_extractor = None
        self.vision_transformer = None
        self.vision_ready = False

        # Current tactical predictions (simplified)
        self.current_attack_timing = 0.0
        self.current_defend_timing = 0.0
        self.recent_rewards = deque(maxlen=10)

        # Statistics (simplified)
        self.stats = {
            "motion_detected_count": 0,
            "predictions_made": 0,
            "vision_transformer_ready": False,
            "avg_attack_timing": 0.0,
            "avg_defend_timing": 0.0,
        }

        logger.info(
            f"🎮 Simplified Street Fighter Vision Pipeline Wrapper initialized:"
        )
        logger.info(f"   Resolution: {self.target_size[1]}×{self.target_size[0]}")
        logger.info(f"   Frame stack: {frame_stack} RGB frames (24 channels total)")
        logger.info(f"   OpenCV: Simple motion detection (2 features)")
        logger.info(
            f"   Enhanced Momentum: Health + Score + Relative Position (18 features)"
        )
        logger.info(f"   Tactical Predictions: Attack/Defend timing only")
        logger.info(
            f"   Position Analysis: Uses relative positioning (agent-enemy diffs)"
        )
        logger.info(
            f"   Defense anti-spam: {self.defense_cooldown_frames} frame cooldown"
        )
        logger.info(
            f"   Vision Transformer: {'Enabled' if enable_vision_transformer else 'Disabled'}"
        )

    def inject_feature_extractor(self, feature_extractor):
        """Inject CNN feature extractor and initialize simplified vision transformer"""
        if not self.enable_vision_transformer:
            logger.info("   🔧 Vision Transformer disabled")
            return

        try:
            self.cnn_extractor = feature_extractor
            actual_feature_dim = self.cnn_extractor.features_dim
            logger.info(f"   📏 Detected CNN feature dimension: {actual_feature_dim}")

            # Initialize simplified vision transformer with correct dimensions
            device = next(feature_extractor.parameters()).device
            self.vision_transformer = SimplifiedVisionTransformer(
                visual_dim=actual_feature_dim,
                opencv_dim=2,
                momentum_dim=19,  # Updated to 19 to match actual momentum features
                seq_length=self.frame_stack,
            ).to(device)
            self.vision_ready = True
            self.stats["vision_transformer_ready"] = True

            logger.info("   ✅ Simplified Vision Transformer initialized and ready!")

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

        # Reset tactical predictions (simplified)
        self.current_attack_timing = 0.0
        self.current_defend_timing = 0.0

        # Initialize frame buffer with processed frames
        processed_frame = self._preprocess_frame(observation)
        self.frame_buffer.clear()
        for _ in range(self.frame_stack):
            self.frame_buffer.append(processed_frame)

        # Clear feature history buffers
        self.visual_features_history.clear()
        self.opencv_features_history.clear()
        self.momentum_features_history.clear()

        stacked_obs = self._get_stacked_observation()
        return stacked_obs, info

    def step(self, action):
        """Execute action and process through simplified vision pipeline"""
        # Convert action if needed
        if isinstance(action, np.ndarray) and action.ndim == 0:
            binary_action = np.zeros(self.env.action_space.n, dtype=int)
            if 0 <= action < self.env.action_space.n:
                binary_action[action] = 1
            action = binary_action

        # Track defense actions for anti-spam
        is_defending = False
        if hasattr(action, "__iter__") and len(action) > max(
            self.defend_action_indices
        ):
            is_defending = any(action[i] > 0 for i in self.defend_action_indices)
        elif isinstance(action, int):
            is_defending = action in self.defend_action_indices

        if is_defending:
            self.last_defense_frame = self.episode_steps

        # Execute step
        observation, reward, done, truncated, info = self.env.step(action)

        # Extract enhanced game state information
        (
            curr_player_health,
            curr_opponent_health,
            score,
            player_x,
            player_y,
            opponent_x,
            opponent_y,
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

        # Process through simplified vision pipeline
        tactical_predictions = self._process_simplified_vision_pipeline(
            stacked_obs,
            curr_player_health,
            curr_opponent_health,
            score,
            player_x,
            player_y,
            opponent_x,
            opponent_y,
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
        """Extract enhanced game state including positions and score"""
        player_health = info.get("agent_hp", self.full_hp)
        opponent_health = info.get("enemy_hp", self.full_hp)
        score = info.get("score", 0)
        player_x = info.get("agent_x", 90)  # Default to center if not available
        player_y = info.get("agent_y", 64)
        opponent_x = info.get("enemy_x", 90)
        opponent_y = info.get("enemy_y", 64)

        return (
            player_health,
            opponent_health,
            score,
            player_x,
            player_y,
            opponent_x,
            opponent_y,
        )

    def _process_simplified_vision_pipeline(
        self,
        stacked_obs,
        player_health,
        opponent_health,
        score,
        player_x,
        player_y,
        opponent_x,
        opponent_y,
    ):
        """Process through simplified vision pipeline - attack/defend only"""
        try:
            # Step 1: OpenCV detection on latest frame
            latest_frame = self._extract_latest_frame_rgb(stacked_obs)
            opencv_detections = self.opencv_detector.detect_activity(latest_frame)
            opencv_features = self._opencv_to_features(opencv_detections)

            if opencv_detections.get("motion_detected", False):
                self.stats["motion_detected_count"] += 1

            # Step 2: Enhanced momentum tracking with relative positioning
            momentum_features = self.momentum_tracker.update(
                player_health,
                opponent_health,
                score,
                player_x,
                player_y,
                opponent_x,
                opponent_y,
            )

            # Step 3: CNN feature extraction
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
            self.opencv_features_history.append(opencv_features)
            self.momentum_features_history.append(momentum_features)

            # Step 4: Simplified vision transformer prediction
            if (
                self.vision_ready
                and len(self.visual_features_history) == self.frame_stack
            ):
                prediction = self._make_simplified_tactical_prediction()
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
            logger.error(f"Simplified vision pipeline processing error: {e}")
            return None

    def _make_simplified_tactical_prediction(self):
        """Make simplified tactical predictions - attack and defend timing only"""
        try:
            if (
                not self.vision_ready
                or len(self.visual_features_history) < self.frame_stack
            ):
                return None

            # Stack sequences
            visual_seq = np.stack(list(self.visual_features_history))  # [8, 512]
            opencv_seq = np.stack(list(self.opencv_features_history))  # [8, 2]
            momentum_seq = np.stack(list(self.momentum_features_history))  # [8, 18]

            # Combine features at each timestep
            combined_seq = np.concatenate(
                [visual_seq, opencv_seq, momentum_seq], axis=1
            )  # [8, 532]

            # Convert to tensor and add batch dimension
            device = next(self.vision_transformer.parameters()).device
            combined_tensor = (
                torch.from_numpy(combined_seq).float().unsqueeze(0).to(device)
            )  # [1, 8, 532]

            # Get simplified tactical predictions from vision transformer
            with torch.no_grad():
                predictions = self.vision_transformer(combined_tensor)

            return {
                "attack_timing": predictions["attack_timing"].cpu().item(),
                "defend_timing": predictions["defend_timing"].cpu().item(),
            }

        except Exception as e:
            logger.error(f"Simplified vision prediction error: {e}")
            return None

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

    def _extract_latest_frame_rgb(self, stacked_obs):
        """Extract latest RGB frame from 8-frame stack for OpenCV processing"""
        try:
            # stacked_obs shape: [24, 128, 180] = [8*3, H, W] for 8 RGB frames
            # Latest frame is last 3 channels: [21:24, :, :] (channels 21, 22, 23)
            latest_frame_chw = stacked_obs[-3:]  # [3, 128, 180] - last RGB frame
            latest_frame_hwc = latest_frame_chw.transpose(
                1, 2, 0
            )  # [128, 180, 3] - HWC format
            return latest_frame_hwc

        except Exception as e:
            logger.error(f"Frame extraction error: {e}")
            return np.zeros((128, 180, 3), dtype=np.uint8)

    def _opencv_to_features(self, opencv_detections):
        """Convert OpenCV detections to fixed-size feature vector (2 dims)"""
        try:
            features = np.zeros(2, dtype=np.float32)

            # Only motion features - keep it simple
            features[0] = (
                1.0 if opencv_detections.get("motion_detected", False) else 0.0
            )
            features[1] = opencv_detections.get("motion_intensity", 0.0)

            return features

        except Exception as e:
            logger.error(f"OpenCV feature conversion error: {e}")
            return np.zeros(2, dtype=np.float32)

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
