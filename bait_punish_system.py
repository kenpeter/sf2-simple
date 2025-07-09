#!/usr/bin/env python3
"""
bait_punish_system.py - Simplified and effective Bait->Block->Punish sequence detection
APPROACH: Event-driven pattern recognition with clear reward signals
LEARNS: Temporal sequences through immediate feedback and pattern recognition
FIXES: Simplified feature extraction, clearer phase detection, and proper reward scaling
"""

import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math


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


def normalize_value(value, min_val, max_val, target_min=-1.0, target_max=1.0):
    """Normalize a value to target range with safety bounds."""
    try:
        value = ensure_scalar(value, (min_val + max_val) / 2.0)

        # Clamp to expected range first
        value = max(min_val, min(max_val, value))

        # Normalize to [0, 1] then scale to target range
        if max_val > min_val:
            normalized = (value - min_val) / (max_val - min_val)
        else:
            normalized = 0.5

        # Scale to target range
        result = target_min + normalized * (target_max - target_min)

        # Final safety clamp
        return max(target_min, min(target_max, result))
    except:
        return (target_min + target_max) / 2.0
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


@dataclass
class GameState:
    """Represents a single frame's game state for sequence analysis."""

    player_x: float
    opponent_x: float
    player_health: int
    opponent_health: int
    player_attacking: bool
    opponent_attacking: bool
    player_blocking: bool
    distance: float
    player_damage_taken: int
    opponent_damage_taken: int
    frame_number: int


class BaitPunishDetector:
    """
    Simplified bait-punish detector focused on clear pattern recognition.
    Based on research showing that successful fighting game AI uses:
    1. Clear phase detection (bait, defend, punish)
    2. Immediate rewards for correct behaviors
    3. Simple, interpretable features
    """

    def __init__(self, history_length=60):
        self.history_length = history_length
        self.state_history = deque(maxlen=history_length)

        # Phase tracking
        self.current_phase = "neutral"
        self.phase_start_frame = 0
        self.phase_duration = 0

        # Sequence tracking
        self.bait_start_frame = -1
        self.defend_start_frame = -1
        self.punish_start_frame = -1

        # Pattern tracking
        self.opponent_attack_frames = deque(maxlen=30)  # Track when opponent attacks
        self.player_damage_frames = deque(maxlen=30)  # Track when player takes damage
        self.successful_punishes = 0
        self.total_punish_attempts = 0

        # Adaptive thresholds (from research)
        self.bait_distance_min = 50  # Minimum safe distance for baiting
        self.bait_distance_max = 90  # Maximum effective bait distance
        self.defend_window = 8  # Frames to respond after opponent attack
        self.punish_window = 10  # Frames to punish after opponent vulnerability

        # NORMALIZATION: Reward scaling and bounds
        self.base_reward_scale = 0.01
        self.sequence_bonus_scale = 0.05
        self.max_reward = 0.1

        # NORMALIZATION: Running statistics for reward normalization
        self.reward_history = deque(maxlen=1000)
        self.reward_mean = 0.0
        self.reward_std = 1.0

        # Success tracking for learning
        self.recent_outcomes = deque(maxlen=100)
        self.success_rate = 0.0

        # NORMALIZATION: Game state bounds for proper scaling
        self.game_bounds = {
            "screen_width": 320,
            "max_health": 176,
            "max_distance": 160,
            "max_damage": 100,
        }

    def update(
        self,
        info: Dict,
        button_features: np.ndarray,
        damage_dealt: int,
        damage_received: int,
    ) -> Dict:
        """Update the detector with new frame data."""

        try:
            # Create current game state with NORMALIZATION
            raw_player_x = ensure_scalar(info.get("agent_x", 160), 160.0)
            raw_opponent_x = ensure_scalar(info.get("enemy_x", 160), 160.0)
            raw_player_hp = ensure_scalar(info.get("agent_hp", 176), 176)
            raw_opponent_hp = ensure_scalar(info.get("enemy_hp", 176), 176)
            raw_damage_dealt = ensure_scalar(damage_dealt, 0)
            raw_damage_received = ensure_scalar(damage_received, 0)

            # NORMALIZE positions to [-1, 1]
            norm_player_x = normalize_value(
                raw_player_x, 0, self.game_bounds["screen_width"], -1, 1
            )
            norm_opponent_x = normalize_value(
                raw_opponent_x, 0, self.game_bounds["screen_width"], -1, 1
            )

            # NORMALIZE health to [0, 1]
            norm_player_hp = normalize_value(
                raw_player_hp, 0, self.game_bounds["max_health"], 0, 1
            )
            norm_opponent_hp = normalize_value(
                raw_opponent_hp, 0, self.game_bounds["max_health"], 0, 1
            )

            # NORMALIZE damage to [0, 1]
            norm_damage_dealt = normalize_value(
                raw_damage_dealt, 0, self.game_bounds["max_damage"], 0, 1
            )
            norm_damage_received = normalize_value(
                raw_damage_received, 0, self.game_bounds["max_damage"], 0, 1
            )

            # Calculate and normalize distance
            raw_distance = abs(raw_player_x - raw_opponent_x)
            norm_distance = normalize_value(
                raw_distance, 0, self.game_bounds["max_distance"], 0, 1
            )

            state = GameState(
                player_x=norm_player_x,
                opponent_x=norm_opponent_x,
                player_health=norm_player_hp,
                opponent_health=norm_opponent_hp,
                player_attacking=self._is_attacking(button_features),
                opponent_attacking=self._infer_opponent_attacking(
                    info, damage_received
                ),
                player_blocking=self._is_blocking(button_features, damage_received),
                distance=norm_distance,
                player_damage_taken=norm_damage_received,
                opponent_damage_taken=norm_damage_dealt,
                frame_number=len(self.state_history),
            )

            self.state_history.append(state)

            # Track opponent attacks and player damage (using raw values for logic)
            if state.opponent_attacking:
                self.opponent_attack_frames.append(state.frame_number)
            if raw_damage_received > 0:
                self.player_damage_frames.append(state.frame_number)

            # Detect phase transitions and calculate rewards
            reward, phase_info = self._analyze_current_frame(state, raw_distance)

            # NORMALIZE the final reward
            normalized_reward = self._normalize_reward(reward)

            # Update success tracking
            self._update_success_tracking(normalized_reward, phase_info)

            # Prepare return info with NORMALIZED values
            result = {
                "bait_punish_reward": np.clip(
                    normalized_reward, -self.max_reward, self.max_reward
                ),
                "sequence_phase": self.current_phase,
                "phase_duration": normalize_value(
                    self.phase_duration, 0, 60, 0, 1
                ),  # Normalize duration
                "success_rate": self.success_rate,  # Already normalized [0,1]
                **phase_info,
            }

            return result

        except Exception as e:
            print(f"BaitPunish update error: {e}")
            return {
                "bait_punish_reward": 0.0,
                "sequence_phase": "neutral",
                "phase_duration": 0,
                "success_rate": 0.0,
            }

    def _is_attacking(self, button_features: np.ndarray) -> bool:
        """Detect if player is attacking based on button inputs."""
        try:
            button_features = sanitize_array(button_features, 0.0)
            if len(button_features) < 12:
                return False
            # Attack buttons: B, Y, A, X, L, R
            attack_buttons = [0, 1, 8, 9, 10, 11]
            return bool(np.any(button_features[attack_buttons] > 0.5))
        except:
            return False

    def _normalize_reward(self, raw_reward: float) -> float:
        """Normalize rewards using running statistics to prevent training instability."""
        raw_reward = ensure_scalar(raw_reward, 0.0)

        # Update running statistics
        self.reward_history.append(raw_reward)

        if len(self.reward_history) > 10:
            # Update running mean and std
            recent_rewards = list(self.reward_history)[-100:]  # Use recent history
            self.reward_mean = np.mean(recent_rewards)
            self.reward_std = max(
                np.std(recent_rewards), 0.01
            )  # Prevent division by tiny std

        # Z-score normalization
        if self.reward_std > 0:
            normalized = (raw_reward - self.reward_mean) / self.reward_std
        else:
            normalized = 0.0

        # Scale and clip to reasonable range
        scaled = normalized * self.base_reward_scale

        # Final clipping to prevent explosions
        return max(-self.max_reward, min(self.max_reward, scaled))

    def _infer_opponent_attacking(self, info: Dict, damage_received: int) -> bool:
        """Infer if opponent is attacking based on damage received."""
        damage_received = ensure_scalar(damage_received, 0)

        # Direct indicator: damage received
        if damage_received > 0:
            return True

        # Indirect indicator: rapid approach (if available in info)
        if len(self.state_history) > 1:
            prev_state = self.state_history[-1]
            # Use raw values for logic decisions (convert back from normalized)
            current_distance = abs(
                ensure_scalar(info.get("agent_x", 160), 160.0)
                - ensure_scalar(info.get("enemy_x", 160), 160.0)
            )
            # Convert previous normalized distance back to raw for comparison
            prev_raw_distance = prev_state.distance * self.game_bounds["max_distance"]

            # Rapid approach (more than 5 units per frame)
            if prev_raw_distance - current_distance > 5:
                return True

        return False

    def _is_blocking(self, button_features: np.ndarray, damage_received: int) -> bool:
        """Detect if player is blocking."""
        try:
            button_features = sanitize_array(button_features, 0.0)
            if len(button_features) < 12:
                return False

            damage_received = ensure_scalar(damage_received, 0)

            # Check for defensive inputs (direction buttons)
            defensive_buttons = [4, 5, 6, 7]  # UP, DOWN, LEFT, RIGHT
            defensive_input = bool(np.any(button_features[defensive_buttons] > 0.5))

            # Check for damage reduction (successful block)
            if len(self.opponent_attack_frames) > 0:
                recent_opponent_attacks = sum(
                    1
                    for frame in self.opponent_attack_frames
                    if len(self.state_history) - frame <= 5
                )
                if recent_opponent_attacks > 0 and damage_received < 5:
                    return True

            return defensive_input
        except:
            return False

    def _analyze_current_frame(
        self, state: GameState, raw_distance: float
    ) -> Tuple[float, Dict]:
        """Analyze current frame for bait-punish patterns and calculate rewards."""

        reward = 0.0
        info = {}

        # Update phase duration
        self.phase_duration += 1

        # Detect phase based on current behavior (using raw distance for thresholds)
        new_phase = self._detect_current_phase(state, raw_distance)

        # Handle phase transitions
        if new_phase != self.current_phase:
            transition_reward = self._handle_phase_transition(
                self.current_phase, new_phase, state
            )
            reward += transition_reward

            self.current_phase = new_phase
            self.phase_start_frame = state.frame_number
            self.phase_duration = 0

            info["phase_transition"] = True
            info["transition_reward"] = normalize_value(transition_reward, 0, 5, 0, 1)
        else:
            info["phase_transition"] = False
            info["transition_reward"] = 0.0

        # Phase-specific rewards
        phase_reward = self._calculate_phase_reward(state, raw_distance)
        reward += phase_reward
        info["phase_reward"] = normalize_value(phase_reward, 0, 1, 0, 1)

        # Sequence completion bonus
        sequence_reward = self._check_sequence_completion(state)
        reward += sequence_reward
        info["sequence_reward"] = normalize_value(sequence_reward, 0, 5, 0, 1)

        # Quality assessment (normalized outputs)
        info["bait_quality"] = self._assess_bait_quality(state, raw_distance)
        info["defend_quality"] = self._assess_defend_quality(state)
        info["punish_quality"] = self._assess_punish_quality(state)

        return reward, info

    def _detect_current_phase(self, state: GameState, raw_distance: float) -> str:
        """Detect current phase based on player behavior and game state."""

        # Punishing: Player attacking after opponent stopped attacking
        if self._is_punishing(state):
            return "punishing"

        # Defending: Opponent attacking and player blocking/avoiding
        elif self._is_defending(state):
            return "defending"

        # Baiting: Player at optimal distance, not attacking, trying to induce opponent attack
        elif self._is_baiting(state, raw_distance):
            return "baiting"

        # Default to neutral
        else:
            return "neutral"

    def _is_baiting(self, state: GameState, raw_distance: float) -> bool:
        """Check if player is currently baiting."""
        # Player not attacking
        if state.player_attacking:
            return False

        # Player at optimal bait distance (use raw distance for thresholds)
        if not (self.bait_distance_min <= raw_distance <= self.bait_distance_max):
            return False

        # Player not taking damage (successful bait positioning)
        if state.player_damage_taken > 0:
            return False

        # Optional: Player moving (changing distance to bait)
        if len(self.state_history) > 1:
            prev_state = self.state_history[-1]
            # Convert normalized distances back to raw for comparison
            prev_raw_distance = prev_state.distance * self.game_bounds["max_distance"]
            distance_change = abs(raw_distance - prev_raw_distance)
            if distance_change > 1:  # Player is actively moving
                return True

        return True

    def _is_defending(self, state: GameState) -> bool:
        """Check if player is currently defending."""
        # Opponent must be attacking or recently attacked
        recent_opponent_attacks = len(
            [f for f in self.opponent_attack_frames if state.frame_number - f <= 5]
        )

        if recent_opponent_attacks == 0 and not state.opponent_attacking:
            return False

        # Player should be blocking or avoiding damage
        if state.player_blocking:
            return True

        # Player successfully avoiding damage during opponent attack
        if state.opponent_attacking and state.player_damage_taken == 0:
            return True

        return False

    def _is_punishing(self, state: GameState) -> bool:
        """Check if player is currently punishing."""
        # Player must be attacking
        if not state.player_attacking:
            return False

        # Check if this follows a defensive phase or opponent vulnerability
        if len(self.opponent_attack_frames) > 0:
            # Opponent recently stopped attacking (vulnerability window)
            last_opponent_attack = max(self.opponent_attack_frames)
            frames_since_attack = state.frame_number - last_opponent_attack

            if 1 <= frames_since_attack <= self.punish_window:
                return True

        # Check if player was recently defending
        if self.current_phase == "defending" and self.phase_duration <= 3:
            return True

        return False

    def _handle_phase_transition(
        self, old_phase: str, new_phase: str, state: GameState
    ) -> float:
        """Handle phase transitions and assign rewards."""

        # Positive transitions (following the bait-punish sequence)
        positive_transitions = {
            ("neutral", "baiting"): 0.5,
            ("baiting", "defending"): 1.0,  # Successfully baited opponent
            (
                "defending",
                "punishing",
            ): 2.0,  # Successfully defended and counter-attacking
            ("neutral", "defending"): 0.3,  # Direct defensive response
        }

        # Record important transitions
        if new_phase == "defending":
            self.defend_start_frame = state.frame_number
        elif new_phase == "punishing":
            self.punish_start_frame = state.frame_number
            self.total_punish_attempts += 1

        return positive_transitions.get((old_phase, new_phase), 0.0)

    def _calculate_phase_reward(self, state: GameState, raw_distance: float) -> float:
        """Calculate ongoing rewards for current phase behavior."""

        if self.current_phase == "baiting":
            # Reward for maintaining good bait position
            if (
                self.bait_distance_min <= raw_distance <= self.bait_distance_max
                and not state.player_attacking
                and state.player_damage_taken == 0
            ):
                return 0.1

        elif self.current_phase == "defending":
            # Reward for successful defense
            if state.opponent_attacking and state.player_damage_taken == 0:
                return 0.3
            elif (
                state.player_blocking and state.player_damage_taken < 0.3
            ):  # Normalized damage
                return 0.2  # Partial success

        elif self.current_phase == "punishing":
            # Reward for dealing damage during punish
            if state.opponent_damage_taken > 0:
                return 0.5

        return 0.0

    def _check_sequence_completion(self, state: GameState) -> float:
        """Check for complete bait->defend->punish sequences."""

        if self.current_phase == "punishing" and state.opponent_damage_taken > 0:

            # Check if this follows the full sequence
            sequence_bonus = 0.0

            if (
                self.defend_start_frame > 0
                and self.punish_start_frame - self.defend_start_frame <= 10
            ):
                # Successful defend->punish
                sequence_bonus += 2.0
                self.successful_punishes += 1

                # Additional bonus if this was part of bait->defend->punish
                if (
                    self.bait_start_frame > 0
                    and self.defend_start_frame - self.bait_start_frame <= 30
                ):
                    sequence_bonus += 3.0  # Full sequence bonus

            return sequence_bonus * self.sequence_bonus_scale

        return 0.0

    def _assess_bait_quality(self, state: GameState, raw_distance: float) -> float:
        """Assess quality of current baiting behavior."""
        if self.current_phase != "baiting":
            return 0.0

        quality = 0.0

        # Distance management
        if self.bait_distance_min <= raw_distance <= self.bait_distance_max:
            quality += 0.5

        # Not attacking (good bait discipline)
        if not state.player_attacking:
            quality += 0.3

        # Not taking damage (safe positioning)
        if state.player_damage_taken == 0:
            quality += 0.2

        return quality

    def _assess_defend_quality(self, state: GameState) -> float:
        """Assess quality of current defensive behavior."""
        if self.current_phase != "defending":
            return 0.0

        quality = 0.0

        # Active blocking
        if state.player_blocking:
            quality += 0.4

        # Damage avoidance (normalized damage values)
        if state.player_damage_taken == 0:
            quality += 0.4
        elif state.player_damage_taken < 0.3:  # 30/100 normalized
            quality += 0.2  # Partial success

        # Positioning (not too close during opponent attack)
        if state.opponent_attacking and state.distance > 0.1875:  # 30/160 normalized
            quality += 0.2

        return quality

    def _assess_punish_quality(self, state: GameState) -> float:
        """Assess quality of current punish behavior."""
        if self.current_phase != "punishing":
            return 0.0

        quality = 0.0

        # Dealing damage
        if state.opponent_damage_taken > 0:
            quality += 0.6

        # Attacking when opponent is vulnerable
        if state.player_attacking:
            quality += 0.2

        # Quick punish (within window)
        if (
            len(self.opponent_attack_frames) > 0
            and state.frame_number - max(self.opponent_attack_frames)
            <= self.punish_window
        ):
            quality += 0.2

        return quality

    def _update_success_tracking(self, reward: float, info: Dict):
        """Update success tracking for learning adaptation."""

        # Record significant events
        if info.get("sequence_reward", 0) > 0:
            self.recent_outcomes.append(1.0)  # Success
        elif reward < -0.01:  # Negative reward threshold
            self.recent_outcomes.append(0.0)  # Failure

        # Update success rate
        if len(self.recent_outcomes) > 10:
            self.success_rate = np.mean(list(self.recent_outcomes))

        # Adapt thresholds based on success rate (simple adaptive mechanism)
        if self.success_rate > 0.7:
            # Increase difficulty slightly
            self.bait_distance_min = min(60, self.bait_distance_min + 0.1)
            self.defend_window = max(5, self.defend_window - 0.1)
        elif self.success_rate < 0.3:
            # Decrease difficulty slightly
            self.bait_distance_min = max(40, self.bait_distance_min - 0.1)
            self.defend_window = min(12, self.defend_window + 0.1)

    def reset_sequence(self):
        """Reset sequence tracking (call on round/match end)."""
        self.current_phase = "neutral"
        self.phase_start_frame = 0
        self.phase_duration = 0
        self.bait_start_frame = -1
        self.defend_start_frame = -1
        self.punish_start_frame = -1

    def get_learning_stats(self) -> Dict:
        """Get statistics about the learning process."""
        punish_success_rate = self.successful_punishes / max(
            self.total_punish_attempts, 1
        )

        return {
            "current_phase": self.current_phase,
            "phase_duration": self.phase_duration,
            "success_rate": self.success_rate,
            "successful_punishes": self.successful_punishes,
            "total_punish_attempts": self.total_punish_attempts,
            "punish_success_rate": punish_success_rate,
            "bait_distance_range": [self.bait_distance_min, self.bait_distance_max],
            "adapt_thresholds": {
                "defend_window": self.defend_window,
                "punish_window": self.punish_window,
            },
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize detector
    detector = BaitPunishDetector()

    print("Testing Bait-Punish Detector...")

    # Simulate a bait-punish sequence
    for frame in range(150):
        # Mock game info
        info = {
            "agent_x": 160 + np.sin(frame * 0.05) * 15,  # Player moving slightly
            "enemy_x": 160 + np.cos(frame * 0.03) * 25,  # Enemy moving
            "agent_hp": 176,
            "enemy_hp": 176 - max(0, frame - 120) * 2,
        }

        # Mock button features
        buttons = np.zeros(12)

        # Simulate bait-punish sequence
        if 20 <= frame <= 40:
            # Baiting phase: no attacks, good positioning
            pass
        elif 41 <= frame <= 50:
            # Defending phase: opponent attacks, player blocks
            buttons[4] = 1.0  # DOWN (block)
            info["enemy_attacking"] = True
        elif 51 <= frame <= 65:
            # Punishing phase: player attacks
            buttons[0] = 1.0  # Attack button

        # Mock damage
        damage_dealt = 8 if 52 <= frame <= 60 else 0
        damage_received = 2 if frame == 45 else 0

        # Update detector
        result = detector.update(info, buttons, damage_dealt, damage_received)

        # Print interesting frames
        if frame % 10 == 0 or result["phase_transition"]:
            print(
                f"Frame {frame:3d}: Phase={result['sequence_phase']:>10s}, "
                f"Reward={result['bait_punish_reward']:6.3f}, "
                f"Success={result['success_rate']:5.2f}"
            )

    # Print final stats
    stats = detector.get_learning_stats()
    print(f"\nFinal Stats:")
    print(f"  Success Rate: {stats['success_rate']:.2f}")
    print(f"  Punish Success Rate: {stats['punish_success_rate']:.2f}")
    print(f"  Total Punish Attempts: {stats['total_punish_attempts']}")
    print(f"  Successful Punishes: {stats['successful_punishes']}")
