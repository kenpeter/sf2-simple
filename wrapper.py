#!/usr/bin/env python3
"""
wrapper.py - COMPLETE STREET FIGHTER AI WRAPPER
FEATURES: Advanced baiting, blocking detection, move-specific analysis, frame data
RESEARCH-BASED: Proper Street Fighter mechanics implementation
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
VECTOR_FEATURE_DIM = 42  # 21 core + 12 movement + 9 advanced tactics

# Frame data constants (60 FPS)
FRAMES_PER_SECOND = 60
BLOCK_DETECTION_WINDOW = 5  # frames to detect blocking
PUNISH_WINDOW_MIN = 2  # minimum frames for punish
PUNISH_WINDOW_MAX = 10  # maximum frames for punish


class AdvancedFeatureNormalizer:
    """Research-based feature normalizer for stable learning"""

    def __init__(self, feature_dim, clip_range=2.0, adaptive=True):
        self.feature_dim = feature_dim
        self.clip_range = clip_range
        self.adaptive = adaptive
        self.running_mean = np.zeros(feature_dim, dtype=np.float32)
        self.running_var = np.ones(feature_dim, dtype=np.float32)
        self.count = 0
        self.warmup_samples = 500

    def normalize(self, features):
        """Normalize features with research-based methods"""
        features = np.array(features, dtype=np.float32)

        # Ensure correct dimensions
        if len(features) != self.feature_dim:
            if len(features) < self.feature_dim:
                padded = np.zeros(self.feature_dim, dtype=np.float32)
                padded[: len(features)] = features
                features = padded
            else:
                features = features[: self.feature_dim]

        self.count += 1

        # Update statistics
        if self.count == 1:
            self.running_mean = features.copy()
            self.running_var = np.ones_like(features)
        else:
            # Welford's online algorithm for stable variance calculation
            delta = features - self.running_mean
            self.running_mean += delta / min(self.count, self.warmup_samples)
            delta2 = features - self.running_mean
            self.running_var += (delta * delta2 - self.running_var) / min(
                self.count, self.warmup_samples
            )

        # Apply normalization
        if self.count < self.warmup_samples:
            # During warmup: simple clipping
            normalized = np.clip(features, -self.clip_range, self.clip_range)
        else:
            # Z-score normalization with adaptive clipping
            std = np.sqrt(np.abs(self.running_var) + 1e-8)
            normalized = (features - self.running_mean) / std

            if self.adaptive:
                # Adaptive clipping based on data distribution
                clip_val = np.minimum(
                    self.clip_range, np.maximum(1.0, np.abs(normalized).mean() * 1.5)
                )
                normalized = np.clip(normalized, -clip_val, clip_val)
            else:
                normalized = np.clip(normalized, -self.clip_range, self.clip_range)

        return normalized.astype(np.float32)


class RewardNormalizer:
    """Stable reward normalization for consistent learning"""

    def __init__(self, clip_range=3.0, discount=0.99):
        self.clip_range = clip_range
        self.discount = discount
        self.running_mean = 0.0
        self.running_var = 1.0
        self.running_return = 0.0
        self.count = 0
        self.warmup_samples = 100

    def normalize(self, reward):
        """Normalize rewards for stable training"""
        self.count += 1
        self.running_return = self.running_return * self.discount + reward

        if self.count == 1:
            self.running_mean = reward
            self.running_var = 1.0
        else:
            # Update running statistics
            alpha = min(0.99, (self.count - 1) / self.count)
            delta = reward - self.running_mean
            self.running_mean += delta / min(self.count, self.warmup_samples)
            self.running_var = alpha * self.running_var + (1 - alpha) * delta**2

        # Normalize
        if self.count < self.warmup_samples:
            normalized_reward = np.clip(reward, -self.clip_range, self.clip_range)
        else:
            std = np.sqrt(self.running_var + 1e-8)
            normalized_reward = (reward - self.running_mean) / std
            normalized_reward = np.clip(
                normalized_reward, -self.clip_range, self.clip_range
            )

        return float(normalized_reward)

    def get_stats(self):
        return {
            "reward_mean": self.running_mean,
            "reward_std": np.sqrt(self.running_var),
            "running_return": self.running_return,
            "count": self.count,
        }


class StreetFighterMoveDatabase:
    """Database of Street Fighter move properties based on research"""

    def __init__(self):
        # Move ranges and frame data (estimated for SF2/similar games)
        self.move_data = {
            "psycho_crusher": {
                "startup_range": (
                    45,
                    85,
                ),  # Range where Psycho Crusher typically starts
                "active_range": (35, 75),  # Range where it's active/dangerous
                "recovery_range": (25, 65),  # Range during recovery (punishable)
                "block_advantage": -4,  # Negative on block (punishable)
                "whiff_recovery": 25,  # Frames of recovery if whiffed
                "chip_damage": 0.1,  # Chip damage ratio
            },
            "scissor_kick": {
                "startup_range": (35, 60),
                "active_range": (25, 50),
                "recovery_range": (15, 45),
                "block_advantage": -2,
                "whiff_recovery": 18,
                "chip_damage": 0.05,
            },
            "head_stomp": {
                "startup_range": (20, 45),
                "active_range": (15, 35),
                "recovery_range": (10, 30),
                "block_advantage": -6,
                "whiff_recovery": 22,
                "chip_damage": 0.08,
            },
            "heavy_punch": {
                "startup_range": (25, 50),
                "active_range": (20, 40),
                "recovery_range": (15, 35),
                "block_advantage": -3,
                "whiff_recovery": 15,
                "chip_damage": 0.02,
            },
        }

        # Baiting ranges for different moves
        self.bait_ranges = {
            "psycho_crusher": (50, 80),  # Optimal range to bait Psycho Crusher
            "scissor_kick": (40, 65),
            "head_stomp": (25, 50),
            "general": (35, 70),  # General baiting range
        }


class AdvancedBaitingSystem:
    """Research-based baiting system implementation"""

    def __init__(self, history_length=60):  # 1 second at 60 FPS
        self.history_length = history_length
        self.move_db = StreetFighterMoveDatabase()

        # Position and movement tracking
        self.position_history = deque(maxlen=history_length)
        self.velocity_history = deque(maxlen=history_length)
        self.distance_history = deque(maxlen=history_length)

        # Attack and damage tracking
        self.player_attack_history = deque(maxlen=history_length)
        self.opponent_attack_history = deque(maxlen=history_length)
        self.damage_events = deque(maxlen=history_length)

        # Baiting pattern detection
        self.bait_attempts = 0
        self.successful_baits = 0
        self.whiff_punishes = 0
        self.control_normal_uses = 0

        # Advanced pattern tracking
        self.retreat_patterns = []
        self.approach_patterns = []
        self.frame_count = 0

        # Timing tracking
        self.last_approach_frame = -100
        self.last_retreat_frame = -100
        self.last_opponent_attack_frame = -100
        self.last_whiff_punish_frame = -100

    def update(
        self,
        player_x,
        opponent_x,
        player_attacking,
        opponent_attacking,
        player_damage_taken,
        opponent_damage_taken,
    ):
        """Update baiting system with current frame data"""
        self.frame_count += 1
        distance = abs(player_x - opponent_x)

        # Calculate velocity (smoothed)
        if len(self.position_history) > 0:
            prev_player_x, prev_opponent_x, _ = self.position_history[-1]
            player_velocity = (player_x - prev_player_x) * 0.7 + (
                self.velocity_history[-1][0] * 0.3 if self.velocity_history else 0
            )
            opponent_velocity = (opponent_x - prev_opponent_x) * 0.7 + (
                self.velocity_history[-1][1] * 0.3 if self.velocity_history else 0
            )
        else:
            player_velocity = opponent_velocity = 0.0

        # Store history
        self.position_history.append((player_x, opponent_x, distance))
        self.velocity_history.append((player_velocity, opponent_velocity))
        self.distance_history.append(distance)
        self.player_attack_history.append(player_attacking)
        self.opponent_attack_history.append(opponent_attacking)
        self.damage_events.append((player_damage_taken, opponent_damage_taken))

        # Update attack frame tracking
        if opponent_attacking and not self._was_opponent_attacking_recently(3):
            self.last_opponent_attack_frame = self.frame_count

        # Detect baiting patterns
        self._detect_approach_retreat_patterns(
            player_x, opponent_x, distance, player_velocity
        )
        self._detect_control_normal_usage(player_attacking, distance)
        self._detect_whiff_punish_opportunities(
            player_attacking, opponent_attacking, distance
        )

        return self._get_baiting_features()

    def _was_opponent_attacking_recently(self, frames):
        """Check if opponent was attacking in recent frames"""
        if len(self.opponent_attack_history) < frames:
            return False
        # Convert deque to list for proper slicing
        recent_history = list(self.opponent_attack_history)
        return any(recent_history[-frames:-1]) if len(recent_history) > 1 else False

    def _detect_approach_retreat_patterns(
        self, player_x, opponent_x, distance, player_velocity
    ):
        """Detect classic approach->retreat baiting patterns"""
        if len(self.position_history) < 20:
            return

        # Determine if we're approaching or retreating
        approaching = (player_x < opponent_x and player_velocity > 0.5) or (
            player_x > opponent_x and player_velocity < -0.5
        )
        retreating = (player_x < opponent_x and player_velocity < -0.5) or (
            player_x > opponent_x and player_velocity > 0.5
        )

        # Check for baiting ranges
        in_bait_range = any(
            r[0] <= distance <= r[1] for r in self.move_db.bait_ranges.values()
        )

        if approaching and in_bait_range:
            self.last_approach_frame = self.frame_count

        elif retreating and in_bait_range:
            self.last_retreat_frame = self.frame_count

            # Check if this follows an approach (classic bait pattern)
            if self.frame_count - self.last_approach_frame <= 30:  # Within 0.5 seconds
                # Look for opponent attack during our retreat
                recent_attacks = (
                    list(self.opponent_attack_history)[-10:]
                    if len(self.opponent_attack_history) >= 10
                    else list(self.opponent_attack_history)
                )
                recent_opp_attacks = any(recent_attacks)
                if recent_opp_attacks:
                    self.bait_attempts += 1

                    # Check if opponent whiffed and we can punish
                    if self._detect_opponent_whiff():
                        self.successful_baits += 1

    def _detect_control_normal_usage(self, player_attacking, distance):
        """Detect usage of control normals for baiting"""
        # Control normals are attacks used at specific ranges to threaten opponent
        optimal_control_range = (30, 60)  # Range where control normals are effective

        if (
            player_attacking
            and optimal_control_range[0] <= distance <= optimal_control_range[1]
            and len(self.player_attack_history) >= 5
        ):

            # Check if this is a deliberate control normal (not part of combo)
            recent_attacks_list = list(self.player_attack_history)[-5:]
            recent_attacks = sum(recent_attacks_list)
            if recent_attacks <= 2:  # Isolated attack, likely a control normal
                self.control_normal_uses += 1

    def _detect_whiff_punish_opportunities(
        self, player_attacking, opponent_attacking, distance
    ):
        """Detect and track whiff punish attempts and successes"""
        if len(self.damage_events) < 10:
            return

        # Look for pattern: opponent attacked, we didn't take damage, we counter-attacked
        frames_since_opp_attack = self.frame_count - self.last_opponent_attack_frame

        if (
            2 <= frames_since_opp_attack <= 15  # Within punish window
            and player_attacking  # We're counter-attacking
            and not self._took_damage_recently(5)
        ):  # We didn't take damage (opponent whiffed)

            # Check if opponent was in recovery range for their move
            in_punish_range = any(
                r[0] <= distance <= r[1]
                for move_data in self.move_db.move_data.values()
                for r in [move_data["recovery_range"]]
            )

            if in_punish_range:
                self.whiff_punishes += 1
                self.last_whiff_punish_frame = self.frame_count

    def _detect_opponent_whiff(self):
        """Detect if opponent likely whiffed an attack"""
        if len(self.damage_events) < 5:
            return False

        # Check if opponent attacked but we took no damage
        frames_since_opp_attack = self.frame_count - self.last_opponent_attack_frame
        if frames_since_opp_attack <= 10:
            return not self._took_damage_recently(10)
        return False

    def _took_damage_recently(self, frames):
        """Check if we took damage in recent frames"""
        if len(self.damage_events) < frames:
            return False
        # Convert deque to list for proper slicing
        recent_events = list(self.damage_events)
        return (
            any(event[0] > 0 for event in recent_events[-frames:])
            if recent_events
            else False
        )

    def _get_baiting_features(self):
        """Generate 5 baiting-related features"""
        features = np.zeros(5, dtype=np.float32)

        # Feature 0: Bait success rate
        if self.bait_attempts > 0:
            features[0] = self.successful_baits / self.bait_attempts

        # Feature 1: Whiff punish frequency (per minute)
        minutes_played = max(1, self.frame_count / 3600)  # 60 FPS
        features[1] = np.clip(self.whiff_punishes / minutes_played, 0, 10) / 10

        # Feature 2: Control normal usage rate
        total_attacks = sum(self.player_attack_history) + 1
        features[2] = np.clip(self.control_normal_uses / total_attacks, 0, 1)

        # Feature 3: Recent baiting activity (sliding window)
        recent_window = 300  # 5 seconds
        if self.frame_count > recent_window:
            recent_baits = max(
                0, self.bait_attempts - (self.frame_count - recent_window) // 60
            )
            features[3] = np.clip(recent_baits / 5, 0, 1)

        # Feature 4: Approach-retreat rhythm quality
        approach_retreat_gap = abs(self.last_approach_frame - self.last_retreat_frame)
        if approach_retreat_gap > 0:
            rhythm_quality = np.exp(
                -approach_retreat_gap / 30
            )  # Prefer quick transitions
            features[4] = np.clip(rhythm_quality, 0, 1)

        return features


class AdvancedBlockingSystem:
    """Research-based blocking and punishment system"""

    def __init__(self, history_length=60):
        self.history_length = history_length
        self.move_db = StreetFighterMoveDatabase()

        # Blocking detection
        self.damage_history = deque(maxlen=history_length)
        self.attack_history = deque(maxlen=history_length)
        self.health_history = deque(maxlen=history_length)

        # Frame advantage tracking
        self.block_events = []
        self.punish_attempts = 0
        self.successful_punishes = 0
        self.frame_perfect_blocks = 0

        # Timing analysis
        self.last_block_frame = -100
        self.last_punish_frame = -100
        self.frame_count = 0

        # Block type classification
        self.safe_move_blocks = 0
        self.unsafe_move_blocks = 0

    def update(
        self,
        player_health,
        opponent_health,
        player_attacking,
        opponent_attacking,
        player_damage_taken,
        opponent_damage_taken,
        distance,
    ):
        """Update blocking system with frame data"""
        self.frame_count += 1

        # Store history
        self.damage_history.append((player_damage_taken, opponent_damage_taken))
        self.attack_history.append((player_attacking, opponent_attacking))
        self.health_history.append((player_health, opponent_health))

        # Detect blocking events
        self._detect_blocking_events(opponent_attacking, player_damage_taken, distance)

        # Detect punishment attempts and success
        self._detect_punishment_events(player_attacking, distance)

        return self._get_blocking_features()

    def _detect_blocking_events(
        self, opponent_attacking, player_damage_taken, distance
    ):
        """Detect when we successfully block opponent attacks"""
        if len(self.damage_history) < 5:
            return

        # Block detection: opponent attacking but we take little/no damage
        if opponent_attacking:
            # Check for different block scenarios
            if player_damage_taken == 0:
                # Perfect block (no damage)
                self._register_block_event("perfect", distance)
            elif 0 < player_damage_taken <= 5:  # Small chip damage
                # Chip damage block (partial block or chip damage)
                self._register_block_event("chip", distance)

    def _register_block_event(self, block_type, distance):
        """Register a blocking event with frame data analysis"""
        self.last_block_frame = self.frame_count

        # Classify the type of move blocked based on distance and game knowledge
        move_type = self._classify_blocked_move(distance)

        block_event = {
            "frame": self.frame_count,
            "type": block_type,
            "distance": distance,
            "move_type": move_type,
            "frame_advantage": self._calculate_frame_advantage(move_type),
        }

        self.block_events.append(block_event)

        # Update statistics
        if block_type == "perfect":
            self.frame_perfect_blocks += 1

        if move_type in ["psycho_crusher", "head_stomp"]:  # Typically unsafe moves
            self.unsafe_move_blocks += 1
        else:
            self.safe_move_blocks += 1

    def _classify_blocked_move(self, distance):
        """Classify what type of move was likely blocked based on distance"""
        # Use move database to classify based on active ranges
        for move_name, move_data in self.move_db.move_data.items():
            active_min, active_max = move_data["active_range"]
            if active_min <= distance <= active_max:
                return move_name
        return "unknown"

    def _calculate_frame_advantage(self, move_type):
        """Calculate frame advantage after blocking a specific move"""
        if move_type in self.move_db.move_data:
            return -self.move_db.move_data[move_type][
                "block_advantage"
            ]  # Negative becomes positive
        return 0

    def _detect_punishment_events(self, player_attacking, distance):
        """Detect punishment attempts after blocking"""
        frames_since_block = self.frame_count - self.last_block_frame

        # Check if we're attempting to punish within the optimal window
        if (
            PUNISH_WINDOW_MIN <= frames_since_block <= PUNISH_WINDOW_MAX
            and player_attacking
        ):

            self.punish_attempts += 1
            self.last_punish_frame = self.frame_count

            # Determine if this is likely a successful punish
            if self._is_successful_punish(distance, frames_since_block):
                self.successful_punishes += 1

    def _is_successful_punish(self, distance, frames_since_block):
        """Determine if a punish attempt is likely successful"""
        # Check recent damage events to see if we hit opponent
        if len(self.damage_history) >= 3:
            recent_events = list(self.damage_history)[-3:]
            recent_opponent_damage = [event[1] for event in recent_events]
            if any(damage > 0 for damage in recent_opponent_damage):
                return True

        # Also consider optimal punish timing and range
        if frames_since_block <= 6:  # Fast punish
            punish_range = (10, 50)  # Close range for fast punishes
        else:
            punish_range = (15, 60)  # Medium range for slower punishes

        return punish_range[0] <= distance <= punish_range[1]

    def _get_blocking_features(self):
        """Generate 4 blocking-related features"""
        features = np.zeros(4, dtype=np.float32)

        # Feature 0: Block success rate (perfect blocks vs total opponent attacks)
        total_opponent_attacks = sum(attack[1] for attack in self.attack_history) + 1
        features[0] = self.frame_perfect_blocks / total_opponent_attacks

        # Feature 1: Punish success rate after blocking
        if self.punish_attempts > 0:
            features[1] = self.successful_punishes / self.punish_attempts

        # Feature 2: Unsafe move punishment rate
        if self.unsafe_move_blocks > 0:
            # Count recent unsafe move punishments
            recent_unsafe_punishes = 0
            for event in self.block_events[-10:]:
                if event.get("move_type") in ["psycho_crusher", "head_stomp"]:
                    recent_unsafe_punishes += 1
            features[2] = min(
                1.0, recent_unsafe_punishes / max(1, self.unsafe_move_blocks)
            )

        # Feature 3: Frame advantage utilization
        if len(self.block_events) > 0:
            recent_events = self.block_events[-5:]
            avg_frame_advantage = np.mean(
                [event.get("frame_advantage", 0) for event in recent_events]
            )
            features[3] = np.clip(avg_frame_advantage / 10, 0, 1)  # Normalize to [0,1]

        return features


class EnhancedMovementTracker:
    """Enhanced movement and spacing analysis"""

    def __init__(self, history_length=30):
        self.history_length = history_length
        self.position_history = deque(maxlen=history_length)
        self.velocity_history = deque(maxlen=history_length)

        # Movement pattern detection
        self.oscillations = 0
        self.direction_changes = 0
        self.optimal_spacing_time = 0
        self.frame_count = 0

        # Spacing analysis
        self.OPTIMAL_RANGES = {"close": (15, 35), "mid": (35, 65), "far": (65, 100)}

    def update(self, player_x, opponent_x, player_attacking):
        """Update movement tracking"""
        self.frame_count += 1
        distance = abs(player_x - opponent_x)

        # Calculate velocity
        if len(self.position_history) > 0:
            prev_x, prev_opp_x, _ = self.position_history[-1]
            velocity = player_x - prev_x
        else:
            velocity = 0.0

        self.position_history.append((player_x, opponent_x, distance))
        self.velocity_history.append(velocity)

        # Detect direction changes (oscillation)
        if len(self.velocity_history) >= 3:
            recent_velocities = list(self.velocity_history)[-3:]
            if (recent_velocities[0] > 0 and recent_velocities[2] < 0) or (
                recent_velocities[0] < 0 and recent_velocities[2] > 0
            ):
                self.direction_changes += 1

        # Track optimal spacing
        if self.OPTIMAL_RANGES["mid"][0] <= distance <= self.OPTIMAL_RANGES["mid"][1]:
            self.optimal_spacing_time += 1

        return self._get_movement_features()

    def _get_movement_features(self):
        """Generate 12 movement features"""
        features = np.zeros(12, dtype=np.float32)

        if len(self.position_history) == 0:
            return features

        current_distance = self.position_history[-1][2]

        # Features 0-2: Range categorization
        features[0] = (
            1.0
            if self.OPTIMAL_RANGES["close"][0]
            <= current_distance
            <= self.OPTIMAL_RANGES["close"][1]
            else 0.0
        )
        features[1] = (
            1.0
            if self.OPTIMAL_RANGES["mid"][0]
            <= current_distance
            <= self.OPTIMAL_RANGES["mid"][1]
            else 0.0
        )
        features[2] = (
            1.0
            if self.OPTIMAL_RANGES["far"][0]
            <= current_distance
            <= self.OPTIMAL_RANGES["far"][1]
            else 0.0
        )

        # Features 3-4: Movement patterns
        if self.frame_count > 0:
            features[3] = (
                np.clip(self.direction_changes / (self.frame_count / 60), 0, 5) / 5
            )  # Direction changes per second
            features[4] = (
                self.optimal_spacing_time / self.frame_count
            )  # Time in optimal range

        # Features 5-8: Position and spacing analysis
        if len(self.position_history) >= 5:
            recent_distances = [pos[2] for pos in list(self.position_history)[-5:]]
            features[5] = np.mean(recent_distances) / SCREEN_WIDTH  # Average distance
            features[6] = np.std(recent_distances) / 20  # Distance variance
            features[7] = (
                max(recent_distances) - min(recent_distances)
            ) / SCREEN_WIDTH  # Range of movement

        # Features 8-11: Velocity and momentum
        if len(self.velocity_history) >= 3:
            recent_velocities = list(self.velocity_history)[-3:]
            features[8] = np.clip(
                np.mean(recent_velocities) / 5, -1, 1
            )  # Average velocity
            features[9] = np.clip(recent_velocities[-1] / 5, -1, 1)  # Current velocity
            features[10] = (
                1.0 if abs(recent_velocities[-1]) > 1.0 else 0.0
            )  # Active movement
            features[11] = np.clip(
                len([v for v in recent_velocities if abs(v) > 0.5]) / 3, 0, 1
            )  # Movement consistency

        return features


class StreetFighterDiscreteActions:
    """Optimized action space for Street Fighter"""

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

        # Core action combinations for effective gameplay
        self.action_combinations = [
            [],  # 0: No action
            [6],  # 1: LEFT
            [7],  # 2: RIGHT
            [4],  # 3: UP
            [5],  # 4: DOWN
            [0],  # 5: B (punch)
            [1],  # 6: Y (punch)
            [8],  # 7: A (kick)
            [9],  # 8: X (kick)
            [6, 0],  # 9: LEFT + B
            [7, 0],  # 10: RIGHT + B
            [5, 0],  # 11: DOWN + B (crouch punch)
            [6, 8],  # 12: LEFT + A
            [7, 8],  # 13: RIGHT + A
            [5, 8],  # 14: DOWN + A (crouch kick)
            [4, 0],  # 15: UP + B (jump punch)
            [4, 8],  # 16: UP + A (jump kick)
            [5, 7, 0],  # 17: DOWN + RIGHT + B (hadoken motion)
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


class ComprehensiveStrategicTracker:
    """Complete strategic tracking system with all advanced features"""

    def __init__(self, history_length=8):
        self.history_length = history_length

        # Core game state tracking
        self.player_health_history = deque(maxlen=history_length)
        self.opponent_health_history = deque(maxlen=history_length)
        self.score_history = deque(maxlen=history_length)
        self.score_change_history = deque(maxlen=history_length)

        # Combat tracking
        self.combo_counter = 0
        self.max_combo_this_round = 0
        self.last_score_increase_frame = -1
        self.current_frame = 0

        # Damage tracking
        self.player_damage_dealt_history = deque(maxlen=history_length)
        self.opponent_damage_dealt_history = deque(maxlen=history_length)

        # Button input tracking
        self.button_features_history = deque(maxlen=history_length)
        self.previous_button_features = np.zeros(12, dtype=np.float32)

        # Advanced systems
        self.baiting_system = AdvancedBaitingSystem(history_length=60)
        self.blocking_system = AdvancedBlockingSystem(history_length=60)
        self.movement_tracker = EnhancedMovementTracker(history_length=30)

        # Feature normalization
        self.feature_normalizer = AdvancedFeatureNormalizer(
            VECTOR_FEATURE_DIM, clip_range=2.0, adaptive=True
        )

        # Game constants
        self.DANGER_ZONE_HEALTH = MAX_HEALTH * 0.25
        self.CORNER_THRESHOLD = 30
        self.CLOSE_DISTANCE = 40
        self.OPTIMAL_SPACING_MIN = 35
        self.OPTIMAL_SPACING_MAX = 55
        self.COMBO_TIMEOUT_FRAMES = 60
        self.MIN_SCORE_INCREASE_FOR_HIT = 50

        # Previous frame state
        self.prev_player_health = None
        self.prev_opponent_health = None
        self.prev_score = None

        # Combat statistics
        self.close_combat_count = 0
        self.total_frames = 0

    def update(self, info: Dict, button_features: np.ndarray) -> np.ndarray:
        """Main update function that processes all game state"""
        self.current_frame += 1
        self.total_frames += 1

        # Extract game state
        player_health = info.get("agent_hp", MAX_HEALTH)
        opponent_health = info.get("enemy_hp", MAX_HEALTH)
        score = info.get("score", 0)
        player_x = info.get("agent_x", SCREEN_WIDTH / 2)
        opponent_x = info.get("enemy_x", SCREEN_WIDTH / 2)
        distance = abs(player_x - opponent_x)

        # Update histories
        self.player_health_history.append(player_health)
        self.opponent_health_history.append(opponent_health)
        self.score_history.append(score)
        self.button_features_history.append(self.previous_button_features.copy())
        self.previous_button_features = button_features.copy()

        # Calculate damage events
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

        # Update score tracking and combo detection
        score_change = score - self.prev_score if self.prev_score is not None else 0
        self.score_change_history.append(score_change)
        self._update_combo_tracking(score_change)

        # Detect attacks
        player_attacking = np.any(button_features[[0, 1, 8, 9, 10, 11]])
        opponent_attacking = self._detect_opponent_attacking(
            opponent_damage, player_damage
        )

        # Update advanced systems
        baiting_features = self.baiting_system.update(
            player_x,
            opponent_x,
            player_attacking,
            opponent_attacking,
            opponent_damage,
            player_damage,
        )

        blocking_features = self.blocking_system.update(
            player_health,
            opponent_health,
            player_attacking,
            opponent_attacking,
            opponent_damage,
            player_damage,
            distance,
        )

        movement_features = self.movement_tracker.update(
            player_x, opponent_x, player_attacking
        )

        # Update combat statistics
        if distance <= self.CLOSE_DISTANCE:
            self.close_combat_count += 1

        # Construct complete feature vector
        complete_features = self._construct_feature_vector(
            info, distance, baiting_features, blocking_features, movement_features
        )

        # Normalize features
        normalized_features = self.feature_normalizer.normalize(complete_features)

        # Update previous state
        self.prev_player_health = player_health
        self.prev_opponent_health = opponent_health
        self.prev_score = score

        return normalized_features

    def _update_combo_tracking(self, score_change):
        """Update combo counter based on score changes"""
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

    def _detect_opponent_attacking(self, opponent_damage, player_damage):
        """Infer if opponent is attacking based on damage patterns"""
        # Simple heuristic: if we took damage recently, opponent was likely attacking
        if len(self.opponent_damage_dealt_history) >= 3:
            recent_damage = list(self.opponent_damage_dealt_history)[-3:]
            return any(dmg > 0 for dmg in recent_damage)
        return False

    def _construct_feature_vector(
        self, info, distance, baiting_features, blocking_features, movement_features
    ):
        """Construct the complete 42-dimensional feature vector"""

        # 21 core features
        core_features = self._calculate_core_features(info, distance)

        # 12 movement features (from movement_tracker)
        # 5 baiting features (from baiting_system)
        # 4 blocking features (from blocking_system)

        # Concatenate all features: 21 + 12 + 5 + 4 = 42
        complete_features = np.concatenate(
            [
                core_features,  # 21 features
                movement_features,  # 12 features
                baiting_features,  # 5 features
                blocking_features,  # 4 features
            ]
        )

        # Verify dimension
        assert (
            len(complete_features) == VECTOR_FEATURE_DIM
        ), f"Feature dimension mismatch: {len(complete_features)} != {VECTOR_FEATURE_DIM}"

        return complete_features

    def _calculate_core_features(self, info, distance) -> np.ndarray:
        """Calculate 21 core game state features"""
        features = np.zeros(21, dtype=np.float32)

        player_health = info.get("agent_hp", MAX_HEALTH)
        opponent_health = info.get("enemy_hp", MAX_HEALTH)
        player_x = info.get("agent_x", SCREEN_WIDTH / 2)
        opponent_x = info.get("enemy_x", SCREEN_WIDTH / 2)

        # Health features (0-5)
        features[0] = player_health / MAX_HEALTH
        features[1] = opponent_health / MAX_HEALTH
        features[2] = 1.0 if player_health <= self.DANGER_ZONE_HEALTH else 0.0
        features[3] = 1.0 if opponent_health <= self.DANGER_ZONE_HEALTH else 0.0

        if opponent_health > 0:
            health_ratio = player_health / opponent_health
            features[4] = np.clip(health_ratio, 0.0, 3.0) / 3.0
        else:
            features[4] = 1.0

        features[5] = (player_health - opponent_health) / (2 * MAX_HEALTH)

        # Position features (6-12)
        features[6] = player_x / SCREEN_WIDTH
        features[7] = opponent_x / SCREEN_WIDTH
        features[8] = distance / SCREEN_WIDTH

        features[9] = (
            1.0
            if min(player_x, SCREEN_WIDTH - player_x) <= self.CORNER_THRESHOLD
            else 0.0
        )
        features[10] = (
            1.0
            if min(opponent_x, SCREEN_WIDTH - opponent_x) <= self.CORNER_THRESHOLD
            else 0.0
        )

        features[11] = np.sign(
            abs(opponent_x - SCREEN_WIDTH / 2) - abs(player_x - SCREEN_WIDTH / 2)
        )

        y_diff = info.get("agent_y", 64) - info.get("enemy_y", 64)
        features[12] = np.clip(y_diff / (SCREEN_HEIGHT / 2), -1.0, 1.0)

        # Tactical features (13-17)
        features[13] = (
            1.0
            if self.OPTIMAL_SPACING_MIN <= distance <= self.OPTIMAL_SPACING_MAX
            else 0.0
        )
        features[14] = self.close_combat_count / max(1, self.total_frames)
        features[15] = self._calculate_score_momentum() / 5.0

        status_diff = info.get("agent_status", 0) - info.get("enemy_status", 0)
        features[16] = np.clip(status_diff / 100.0, -1.0, 1.0)

        features[17] = self._calculate_momentum(self.player_damage_dealt_history) / 50.0

        # Combo and timing features (18-20)
        features[18] = np.clip(self.combo_counter / 10, 0, 1)
        features[19] = np.clip(self.max_combo_this_round / 15, 0, 1)

        # Recent performance
        if len(self.score_change_history) > 0:
            recent_score_changes = list(self.score_change_history)[-5:]
            features[20] = np.clip(
                np.mean([max(0, s) for s in recent_score_changes]) / 100, 0, 1
            )

        # Ensure all features are bounded
        features = np.clip(features, -2.0, 2.0)

        return features

    def _calculate_score_momentum(self) -> float:
        """Calculate score momentum with combo multiplier"""
        if len(self.score_change_history) < 2:
            return 0.0

        base_momentum = np.mean(
            [max(0, c) for c in list(self.score_change_history)[-5:]]
        )
        combo_multiplier = 1.0 + (self.combo_counter * 0.1)
        momentum = (base_momentum * combo_multiplier) / 100.0

        return np.clip(momentum, -2.0, 5.0)

    def _calculate_momentum(self, history):
        """Calculate damage momentum from history"""
        if len(history) < 2:
            return 0.0

        values = list(history)
        changes = [values[i] - values[i - 1] for i in range(1, len(values))]
        momentum = np.mean(changes[-3:]) if changes else 0.0

        return np.clip(momentum, -50.0, 50.0)

    def get_comprehensive_stats(self) -> Dict:
        """Get comprehensive statistics from all systems"""
        base_stats = {
            "current_combo": self.combo_counter,
            "max_combo_this_round": self.max_combo_this_round,
            "close_combat_ratio": self.close_combat_count / max(1, self.total_frames),
            "total_frames": self.total_frames,
        }

        # Add baiting system stats
        baiting_stats = {
            "bait_attempts": self.baiting_system.bait_attempts,
            "successful_baits": self.baiting_system.successful_baits,
            "whiff_punishes": self.baiting_system.whiff_punishes,
            "control_normal_uses": self.baiting_system.control_normal_uses,
        }

        # Add blocking system stats
        blocking_stats = {
            "frame_perfect_blocks": self.blocking_system.frame_perfect_blocks,
            "punish_attempts": self.blocking_system.punish_attempts,
            "successful_punishes": self.blocking_system.successful_punishes,
            "safe_move_blocks": self.blocking_system.safe_move_blocks,
            "unsafe_move_blocks": self.blocking_system.unsafe_move_blocks,
        }

        # Add movement stats
        movement_stats = {
            "direction_changes": self.movement_tracker.direction_changes,
            "optimal_spacing_time": self.movement_tracker.optimal_spacing_time,
            "optimal_spacing_ratio": self.movement_tracker.optimal_spacing_time
            / max(1, self.total_frames),
        }

        # Combine all stats
        combined_stats = {
            **base_stats,
            **baiting_stats,
            **blocking_stats,
            **movement_stats,
        }

        # Calculate advanced metrics
        if self.baiting_system.bait_attempts > 0:
            combined_stats["bait_success_rate"] = (
                self.baiting_system.successful_baits / self.baiting_system.bait_attempts
            )
        else:
            combined_stats["bait_success_rate"] = 0.0

        if self.blocking_system.punish_attempts > 0:
            combined_stats["punish_success_rate"] = (
                self.blocking_system.successful_punishes
                / self.blocking_system.punish_attempts
            )
        else:
            combined_stats["punish_success_rate"] = 0.0

        return combined_stats


class OptimizedStreetFighterCNN(BaseFeaturesExtractor):
    """Optimized CNN architecture for Street Fighter with advanced features"""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        visual_space = observation_space["visual_obs"]
        vector_space = observation_space["vector_obs"]
        n_input_channels = visual_space.shape[0]
        seq_length, vector_feature_count = vector_space.shape

        print(f"🥊 Advanced Street Fighter CNN Configuration:")
        print(f"   - Visual channels: {n_input_channels}")
        print(f"   - Vector sequence: {seq_length} x {vector_feature_count}")
        print(f"   - Output features: {features_dim}")

        # Visual processing with residual connections
        self.visual_cnn = nn.Sequential(
            # First block
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Second block
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Third block
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Adaptive pooling and flatten
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )

        # Calculate visual output size
        with torch.no_grad():
            dummy_visual = torch.zeros(
                1, n_input_channels, visual_space.shape[1], visual_space.shape[2]
            )
            visual_output_size = self.visual_cnn(dummy_visual).shape[1]

        # Advanced vector processing with attention mechanism
        self.vector_attention = nn.Sequential(
            nn.Linear(vector_feature_count, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, vector_feature_count),
            nn.Sigmoid(),
        )

        self.vector_processor = nn.Sequential(
            nn.Linear(vector_feature_count, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )

        # Temporal processing for vector sequences
        self.temporal_conv = nn.Conv1d(
            vector_feature_count, 32, kernel_size=3, padding=1
        )

        # Feature fusion
        fusion_input_size = (
            visual_output_size + 64 + 32
        )  # visual + processed vector + temporal
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, features_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(features_dim, features_dim),
            nn.ReLU(inplace=True),
        )

        # Initialize weights
        self.apply(self._init_weights)

        print(f"   - Visual output size: {visual_output_size}")
        print(f"   - Fusion input size: {fusion_input_size}")
        print(f"   ✅ Advanced CNN initialized with attention and temporal processing")

    def _init_weights(self, m):
        """Initialize weights with proper scaling"""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Advanced forward pass with attention and temporal processing"""
        visual_obs = observations["visual_obs"]
        vector_obs = observations["vector_obs"]

        device = next(self.parameters()).device
        visual_obs = visual_obs.float().to(device)
        vector_obs = vector_obs.float().to(device)

        # Normalize visual input
        visual_obs = torch.clamp(visual_obs / 255.0, 0.0, 1.0)

        # Process visual features
        visual_features = self.visual_cnn(visual_obs)

        # Process vector features with attention
        # Take the last timestep for main processing
        current_vector = vector_obs[:, -1, :]

        # Apply attention mechanism
        attention_weights = self.vector_attention(current_vector)
        attended_vector = current_vector * attention_weights

        # Process attended vector features
        vector_features = self.vector_processor(attended_vector)

        # Temporal processing across sequence
        # Transpose for Conv1d: (batch, features, time)
        vector_seq_transposed = vector_obs.transpose(1, 2)
        temporal_features = self.temporal_conv(vector_seq_transposed)
        temporal_features = F.adaptive_avg_pool1d(temporal_features, 1).squeeze(-1)

        # Fuse all features
        combined_features = torch.cat(
            [visual_features, vector_features, temporal_features], dim=1
        )
        output = self.fusion(combined_features)

        # Final clamping for stability
        output = torch.clamp(output, -5.0, 5.0)

        return output


class AdvancedStreetFighterPolicy(ActorCriticPolicy):
    """Advanced policy with proper architecture for Street Fighter"""

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
        kwargs["features_extractor_class"] = OptimizedStreetFighterCNN
        kwargs["features_extractor_kwargs"] = {"features_dim": 256}

        # Advanced network architecture
        if net_arch is None:
            net_arch = dict(
                pi=[128, 64], vf=[128, 64]  # Policy network  # Value network
            )

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            *args,
            **kwargs,
        )

        # Apply advanced initialization
        self.apply(self._advanced_init)
        print("✅ Advanced Street Fighter Policy initialized")

    def _advanced_init(self, m):
        """Advanced weight initialization"""
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class StreetFighterVisionWrapper(gym.Wrapper):
    """Complete Street Fighter wrapper with all advanced features"""

    def __init__(self, env, frame_stack=4, rendering=False):
        super().__init__(env)
        self.frame_stack = frame_stack
        self.rendering = rendering
        self.target_size = (128, 180)

        # Action space
        self.discrete_actions = StreetFighterDiscreteActions()
        self.action_space = spaces.Discrete(self.discrete_actions.num_actions)

        # Observation space
        self.observation_space = spaces.Dict(
            {
                "visual_obs": spaces.Box(
                    low=0,
                    high=255,
                    shape=(3 * frame_stack, *self.target_size),
                    dtype=np.uint8,
                ),
                "vector_obs": spaces.Box(
                    low=-5.0,
                    high=5.0,
                    shape=(frame_stack, VECTOR_FEATURE_DIM),
                    dtype=np.float32,
                ),
            }
        )

        # Frame buffers
        self.frame_buffer = deque(maxlen=frame_stack)
        self.vector_features_history = deque(maxlen=frame_stack)

        # Strategic tracking
        self.strategic_tracker = ComprehensiveStrategicTracker(
            history_length=frame_stack
        )

        # Game state
        self.full_hp = 176
        self.prev_player_health = self.full_hp
        self.prev_opponent_health = self.full_hp
        self.wins, self.losses, self.total_rounds = 0, 0, 0
        self.total_damage_dealt, self.total_damage_received = 0, 0

        # Reward normalization
        self.reward_normalizer = RewardNormalizer(clip_range=3.0, discount=0.99)

        # Episode management
        self.episode_steps = 0
        self.max_episode_steps = 10000
        self.episode_rewards = deque(maxlen=100)
        self.stats = {}

    def reset(self, **kwargs):
        """Reset environment with proper initialization"""
        obs, info = self.env.reset(**kwargs)

        # Reset game state
        self.prev_player_health = self.full_hp
        self.prev_opponent_health = self.full_hp
        self.episode_steps = 0

        # Process initial frame
        processed_frame = self._preprocess_frame(obs)
        initial_vector_features = self._create_initial_vector_features(info)

        # Initialize buffers
        self.frame_buffer.clear()
        self.vector_features_history.clear()

        for _ in range(self.frame_stack):
            self.frame_buffer.append(processed_frame)
            self.vector_features_history.append(initial_vector_features.copy())

        # Reset strategic tracker
        self.strategic_tracker = ComprehensiveStrategicTracker(
            history_length=self.frame_stack
        )
        initial_button_features = np.zeros(12, dtype=np.float32)
        self.strategic_tracker.update(info, initial_button_features)

        return self._get_observation(), info

    def step(self, discrete_action):
        """Execute action and return observation"""
        self.episode_steps += 1

        # Convert action
        multibinary_action = self.discrete_actions.discrete_to_multibinary(
            discrete_action
        )
        observation, reward, done, truncated, info = self.env.step(multibinary_action)

        if self.rendering:
            self.env.render()

        # Get current health values
        curr_player_health = info.get("agent_hp", self.full_hp)
        curr_opponent_health = info.get("enemy_hp", self.full_hp)

        # Calculate custom reward
        custom_reward, custom_done = self._calculate_advanced_reward(
            info, curr_player_health, curr_opponent_health, discrete_action
        )

        # Check episode termination
        if self.episode_steps >= self.max_episode_steps:
            truncated = True
        done = custom_done or done

        # Process frame
        processed_frame = self._preprocess_frame(observation)
        self.frame_buffer.append(processed_frame)

        # Update strategic tracking
        button_features = self.discrete_actions.get_button_features(discrete_action)
        vector_features = self.strategic_tracker.update(info, button_features)
        self.vector_features_history.append(vector_features)

        # Update statistics
        self._update_comprehensive_stats()
        info.update(self.stats)

        return self._get_observation(), custom_reward, done, truncated, info

    def _calculate_advanced_reward(
        self, info, curr_player_health, curr_opponent_health, action
    ):
        """Advanced reward calculation with all tactical considerations"""
        raw_reward = 0.0
        done = False

        # Calculate damage events
        damage_dealt = max(0, self.prev_opponent_health - curr_opponent_health)
        damage_received = max(0, self.prev_player_health - curr_player_health)

        # Core combat rewards
        raw_reward += damage_dealt * 0.05  # Damage reward
        raw_reward -= damage_received * 0.025  # Damage penalty

        # Health advantage bonus
        health_advantage = (curr_player_health - curr_opponent_health) / MAX_HEALTH
        raw_reward += health_advantage * 0.002

        # Advanced tactical rewards
        stats = self.strategic_tracker.get_comprehensive_stats()

        # Baiting rewards
        if stats.get("bait_success_rate", 0) > 0.3:
            raw_reward += 0.01  # Bonus for successful baiting

        # Blocking and punishment rewards
        if stats.get("punish_success_rate", 0) > 0.5:
            raw_reward += 0.015  # Bonus for good punishment

        if stats.get("frame_perfect_blocks", 0) > 0:
            raw_reward += 0.005  # Bonus for perfect blocks

        # Movement and spacing rewards
        if stats.get("optimal_spacing_ratio", 0) > 0.6:
            raw_reward += 0.005  # Bonus for good spacing

        # Combo rewards
        current_combo = stats.get("current_combo", 0)
        if current_combo > 1:
            raw_reward += current_combo * 0.02  # Exponential combo bonus

        # Win/Loss handling
        if curr_player_health <= 0 or curr_opponent_health <= 0:
            self.total_rounds += 1
            if curr_opponent_health <= 0 < curr_player_health:
                self.wins += 1
                health_bonus = curr_player_health / MAX_HEALTH
                raw_reward += 5.0 + health_bonus * 2.0  # Significant win bonus
                print(
                    f"🏆 AI WON! Total: {self.wins}W/{self.losses}L (Round {self.total_rounds})"
                )
            else:
                self.losses += 1
                raw_reward -= 2.0  # Loss penalty
                print(
                    f"💀 AI LOST! Total: {self.wins}W/{self.losses}L (Round {self.total_rounds})"
                )
            done = True

        # Small step penalty for efficiency
        raw_reward -= 0.0002

        # Normalize reward
        normalized_reward = self.reward_normalizer.normalize(raw_reward)

        # Update tracking
        self.prev_player_health, self.prev_opponent_health = (
            curr_player_health,
            curr_opponent_health,
        )
        self.total_damage_dealt += damage_dealt
        self.total_damage_received += damage_received

        if done:
            self.episode_rewards.append(normalized_reward)

        return normalized_reward, done

    def _create_initial_vector_features(self, info):
        """Create initial vector features for reset"""
        initial_button_features = np.zeros(12, dtype=np.float32)
        initial_features = self.strategic_tracker.update(info, initial_button_features)
        return np.clip(initial_features, -5.0, 5.0)

    def _get_observation(self):
        """Get current observation dictionary"""
        visual_obs = np.concatenate(list(self.frame_buffer), axis=2).transpose(2, 0, 1)
        vector_obs = np.stack(list(self.vector_features_history))
        return {"visual_obs": visual_obs, "vector_obs": vector_obs}

    def _preprocess_frame(self, frame):
        """Preprocess visual frame"""
        if frame is None:
            return np.zeros((*self.target_size, 3), dtype=np.uint8)
        return cv2.resize(frame, (self.target_size[1], self.target_size[0]))

    def _update_comprehensive_stats(self):
        """Update comprehensive statistics"""
        total_games = self.wins + self.losses
        win_rate = self.wins / total_games if total_games > 0 else 0.0

        # Basic stats
        avg_damage_per_round = self.total_damage_dealt / max(1, self.total_rounds)
        defensive_efficiency = self.total_damage_dealt / max(
            1, self.total_damage_dealt + self.total_damage_received
        )
        damage_ratio = self.total_damage_dealt / max(1, self.total_damage_received)

        # Get advanced stats from strategic tracker
        advanced_stats = self.strategic_tracker.get_comprehensive_stats()

        # Get normalization stats
        reward_stats = self.reward_normalizer.get_stats()

        # Combine all statistics
        self.stats.update(
            {
                # Basic performance
                "win_rate": win_rate,
                "wins": self.wins,
                "losses": self.losses,
                "total_games": total_games,
                "total_rounds": self.total_rounds,
                "avg_damage_per_round": avg_damage_per_round,
                "defensive_efficiency": defensive_efficiency,
                "damage_ratio": damage_ratio,
                "episode_steps": self.episode_steps,
                # Advanced tactical stats
                "bait_success_rate": advanced_stats.get("bait_success_rate", 0.0),
                "punish_success_rate": advanced_stats.get("punish_success_rate", 0.0),
                "whiff_punishes": advanced_stats.get("whiff_punishes", 0),
                "frame_perfect_blocks": advanced_stats.get("frame_perfect_blocks", 0),
                "optimal_spacing_ratio": advanced_stats.get(
                    "optimal_spacing_ratio", 0.0
                ),
                "max_combo": advanced_stats.get("max_combo_this_round", 0),
                # Movement stats
                "direction_changes": advanced_stats.get("direction_changes", 0),
                "close_combat_ratio": advanced_stats.get("close_combat_ratio", 0.0),
                # Normalization monitoring
                "reward_mean": reward_stats["reward_mean"],
                "reward_std": reward_stats["reward_std"],
                "running_return": reward_stats["running_return"],
                "normalization_samples": reward_stats["count"],
            }
        )


def verify_gradient_flow(model, env, device=None):
    """Comprehensive gradient flow verification"""
    print("\n🔬 Advanced Gradient Flow Verification")
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

    # Comprehensive feature analysis
    vector_obs = obs_tensor["vector_obs"]
    visual_obs = obs_tensor["visual_obs"]

    print(f"🔍 Observation Analysis:")
    print(f"   - Vector shape: {vector_obs.shape}")
    print(
        f"   - Vector range: {vector_obs.min().item():.3f} to {vector_obs.max().item():.3f}"
    )
    print(f"   - Visual shape: {visual_obs.shape}")
    print(
        f"   - Visual range: {visual_obs.min().item():.1f} to {visual_obs.max().item():.1f}"
    )

    # Check for problematic values
    vector_issues = []
    if vector_obs.abs().max() > 10.0:
        vector_issues.append("Large magnitudes detected")
    if torch.isnan(vector_obs).any():
        vector_issues.append("NaN values detected")
    if torch.isinf(vector_obs).any():
        vector_issues.append("Infinite values detected")

    if vector_issues:
        print(f"   ⚠️  Vector issues: {', '.join(vector_issues)}")
        return False
    else:
        print("   ✅ Vector features healthy")

    # Test forward pass
    model.policy.train()
    try:
        actions, values, log_probs = model.policy(obs_tensor)

        print(f"✅ Forward pass successful")
        print(f"   - Action shape: {actions.shape}")
        print(f"   - Value: {values.item():.3f}")
        print(f"   - Log prob: {log_probs.item():.3f}")

        # Check output ranges
        if abs(values.item()) > 50.0:
            print("   🚨 Value output extremely large!")
            return False
        elif abs(values.item()) > 20.0:
            print("   ⚠️  Value output large but manageable")
        else:
            print("   ✅ Value output in healthy range")

        # Test backward pass
        test_loss = values.mean() + log_probs.mean()
        test_loss.backward()

        # Check gradients
        grad_norms = []
        for name, param in model.policy.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                if grad_norm > 10.0:
                    print(f"   ⚠️  Large gradient in {name}: {grad_norm:.3f}")

        if grad_norms:
            avg_grad_norm = np.mean(grad_norms)
            max_grad_norm = np.max(grad_norms)
            print(f"   📊 Gradient analysis:")
            print(f"      - Average norm: {avg_grad_norm:.3f}")
            print(f"      - Maximum norm: {max_grad_norm:.3f}")

            if max_grad_norm > 100.0:
                print("   🚨 Gradient explosion detected!")
                return False
            elif max_grad_norm > 10.0:
                print("   ⚠️  Large gradients detected")
                return True
            else:
                print("   ✅ Gradients healthy")
                return True
        else:
            print("   ⚠️  No gradients found")
            return False

    except Exception as e:
        print(f"❌ Forward/backward pass failed: {e}")
        import traceback

        traceback.print_exc()
        return False


# Export all components
__all__ = [
    "StreetFighterVisionWrapper",
    "OptimizedStreetFighterCNN",
    "AdvancedStreetFighterPolicy",
    "verify_gradient_flow",
    "AdvancedBaitingSystem",
    "AdvancedBlockingSystem",
    "EnhancedMovementTracker",
    "ComprehensiveStrategicTracker",
    "StreetFighterDiscreteActions",
    "StreetFighterMoveDatabase",
]
