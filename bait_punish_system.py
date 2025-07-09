#!/usr/bin/env python3
"""
bait_punish_system.py - Learning-based Bait->Block->Punish sequence detection - DIMENSION SAFE
APPROACH: Data-driven pattern recognition instead of hard-coded algorithms
LEARNS: Temporal sequences, defensive patterns, counter-attack opportunities
NEW FIX: Proper dimension handling for integration with base feature system
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


@dataclass
class GameState:
    """Represents a single frame's game state for sequence analysis."""

    player_x: float
    opponent_x: float
    player_health: int
    opponent_health: int
    player_attacking: bool
    opponent_attacking: bool
    player_blocking: bool  # We'll infer this
    distance: float
    player_damage_taken: int
    opponent_damage_taken: int
    button_state: np.ndarray
    frame_number: int


class SequencePattern:
    """Learns and recognizes temporal patterns in fighting game sequences."""

    def __init__(self, pattern_length=30, confidence_threshold=0.7):
        self.pattern_length = pattern_length
        self.confidence_threshold = confidence_threshold

        # Pattern learning
        self.successful_sequences = []
        self.failed_sequences = []
        self.pattern_features = defaultdict(list)

        # Real-time sequence tracking
        self.current_sequence = deque(maxlen=pattern_length)
        self.sequence_rewards = deque(maxlen=1000)

    def add_frame(self, state: GameState):
        """Add a frame to the current sequence being tracked."""
        self.current_sequence.append(state)

    def extract_sequence_features(self, sequence: List[GameState]) -> np.ndarray:
        """Extract features from a sequence of game states."""
        if len(sequence) < 5:
            return np.zeros(20, dtype=np.float32)  # Return empty feature vector

        features = []

        # Movement patterns
        player_positions = [ensure_scalar(s.player_x, 160.0) for s in sequence]
        opponent_positions = [ensure_scalar(s.opponent_x, 160.0) for s in sequence]
        distances = [ensure_scalar(s.distance, 80.0) for s in sequence]

        # 1-4: Movement statistics
        features.extend(
            [
                np.std(player_positions) / 50.0,  # Player movement variance
                np.std(opponent_positions) / 50.0,  # Opponent movement variance
                np.mean(distances) / 100.0,  # Average distance
                np.std(distances) / 50.0,  # Distance variance
            ]
        )

        # 5-8: Attack patterns
        player_attack_frames = sum(1 for s in sequence if bool(s.player_attacking))
        opponent_attack_frames = sum(1 for s in sequence if bool(s.opponent_attacking))
        features.extend(
            [
                player_attack_frames / len(sequence),  # Player attack frequency
                opponent_attack_frames / len(sequence),  # Opponent attack frequency
                self._calculate_attack_clustering(
                    sequence, "player"
                ),  # Attack clustering
                self._calculate_attack_clustering(sequence, "opponent"),
            ]
        )

        # 9-12: Damage patterns
        total_player_damage = sum(
            ensure_scalar(s.player_damage_taken, 0) for s in sequence
        )
        total_opponent_damage = sum(
            ensure_scalar(s.opponent_damage_taken, 0) for s in sequence
        )
        features.extend(
            [
                total_player_damage / 100.0,  # Player damage taken
                total_opponent_damage / 100.0,  # Opponent damage taken
                self._calculate_damage_timing(
                    sequence, "player"
                ),  # When player took damage
                self._calculate_damage_timing(sequence, "opponent"),
            ]
        )

        # 13-16: Positional dynamics
        features.extend(
            [
                self._calculate_approach_retreat_ratio(sequence),
                self._calculate_range_control(sequence),
                self._calculate_positioning_advantage(sequence),
                self._calculate_momentum_shift(sequence),
            ]
        )

        # 17-20: Temporal patterns
        features.extend(
            [
                self._calculate_sequence_rhythm(sequence),
                self._calculate_defensive_windows(sequence),
                self._calculate_counter_attack_timing(sequence),
                self._calculate_pressure_relief(sequence),
            ]
        )

        features_array = np.array(features, dtype=np.float32)
        return sanitize_array(features_array, 0.0)

    def _calculate_attack_clustering(
        self, sequence: List[GameState], player: str
    ) -> float:
        """Calculate how clustered attacks are in time."""
        attacks = []
        for i, state in enumerate(sequence):
            attacking = (player == "player" and bool(state.player_attacking)) or (
                player == "opponent" and bool(state.opponent_attacking)
            )
            if attacking:
                attacks.append(i)

        if len(attacks) < 2:
            return 0.0

        # Calculate variance in attack timing
        attack_intervals = [
            attacks[i + 1] - attacks[i] for i in range(len(attacks) - 1)
        ]
        return (
            1.0 - (np.std(attack_intervals) / len(sequence))
            if attack_intervals
            else 0.0
        )

    def _calculate_damage_timing(self, sequence: List[GameState], player: str) -> float:
        """Calculate when in the sequence damage occurred (0=early, 1=late)."""
        damage_frames = []
        for i, state in enumerate(sequence):
            damage = (
                ensure_scalar(state.player_damage_taken, 0)
                if player == "player"
                else ensure_scalar(state.opponent_damage_taken, 0)
            )
            if damage > 0:
                damage_frames.append(i)

        if not damage_frames:
            return 0.5  # No damage, neutral timing

        avg_damage_frame = np.mean(damage_frames)
        return avg_damage_frame / len(sequence)

    def _calculate_approach_retreat_ratio(self, sequence: List[GameState]) -> float:
        """Calculate ratio of approaching vs retreating behavior."""
        approach_count = 0
        retreat_count = 0

        for i in range(1, len(sequence)):
            prev_dist = ensure_scalar(sequence[i - 1].distance, 80.0)
            curr_dist = ensure_scalar(sequence[i].distance, 80.0)

            if curr_dist < prev_dist - 1:  # Approaching
                approach_count += 1
            elif curr_dist > prev_dist + 1:  # Retreating
                retreat_count += 1

        total = approach_count + retreat_count
        return approach_count / total if total > 0 else 0.5

    def _calculate_range_control(self, sequence: List[GameState]) -> float:
        """Calculate how well the player controls the fighting range."""
        optimal_ranges = 0
        for state in sequence:
            distance = ensure_scalar(state.distance, 80.0)
            if 40 <= distance <= 80:  # Optimal fighting range
                optimal_ranges += 1
        return optimal_ranges / len(sequence)

    def _calculate_positioning_advantage(self, sequence: List[GameState]) -> float:
        """Calculate positional advantage based on screen control."""
        screen_center = 160  # Assuming 320 width
        player_advantage = 0

        for state in sequence:
            player_x = ensure_scalar(state.player_x, screen_center)
            opponent_x = ensure_scalar(state.opponent_x, screen_center)

            player_center_dist = abs(player_x - screen_center)
            opponent_center_dist = abs(opponent_x - screen_center)

            if opponent_center_dist > player_center_dist:
                player_advantage += 1

        return player_advantage / len(sequence)

    def _calculate_momentum_shift(self, sequence: List[GameState]) -> float:
        """Calculate if momentum is shifting toward the player."""
        early_half = sequence[: len(sequence) // 2]
        late_half = sequence[len(sequence) // 2 :]

        early_damage_ratio = self._get_damage_ratio(early_half)
        late_damage_ratio = self._get_damage_ratio(late_half)

        return late_damage_ratio - early_damage_ratio  # Positive if improving

    def _get_damage_ratio(self, states: List[GameState]) -> float:
        """Get damage ratio for a sequence segment."""
        player_damage = sum(ensure_scalar(s.player_damage_taken, 0) for s in states)
        opponent_damage = sum(ensure_scalar(s.opponent_damage_taken, 0) for s in states)
        total_damage = player_damage + opponent_damage

        if total_damage == 0:
            return 0.0

        return opponent_damage / total_damage  # Higher is better for player

    def _calculate_sequence_rhythm(self, sequence: List[GameState]) -> float:
        """Calculate the rhythm/tempo of the sequence."""
        action_frames = []
        for i, state in enumerate(sequence):
            if (
                bool(state.player_attacking)
                or bool(state.opponent_attacking)
                or ensure_scalar(state.player_damage_taken, 0) > 0
                or ensure_scalar(state.opponent_damage_taken, 0) > 0
            ):
                action_frames.append(i)

        if len(action_frames) < 2:
            return 0.0

        intervals = [
            action_frames[i + 1] - action_frames[i]
            for i in range(len(action_frames) - 1)
        ]
        return 1.0 / (1.0 + np.std(intervals))  # Consistent rhythm = higher score

    def _calculate_defensive_windows(self, sequence: List[GameState]) -> float:
        """Calculate how well defensive windows are utilized."""
        defensive_opportunities = 0
        successful_defenses = 0

        for i in range(1, len(sequence)):
            prev_state = sequence[i - 1]
            curr_state = sequence[i]

            # Opponent was attacking, now isn't (defensive window)
            if bool(prev_state.opponent_attacking) and not bool(
                curr_state.opponent_attacking
            ):
                defensive_opportunities += 1

                # Check if player took advantage (attacked or repositioned)
                player_x_prev = ensure_scalar(prev_state.player_x, 160.0)
                player_x_curr = ensure_scalar(curr_state.player_x, 160.0)
                distance_prev = ensure_scalar(prev_state.distance, 80.0)
                distance_curr = ensure_scalar(curr_state.distance, 80.0)

                if (
                    bool(curr_state.player_attacking)
                    or abs(distance_curr - distance_prev) > 5
                ):
                    successful_defenses += 1

        return (
            successful_defenses / defensive_opportunities
            if defensive_opportunities > 0
            else 0.0
        )

    def _calculate_counter_attack_timing(self, sequence: List[GameState]) -> float:
        """Calculate quality of counter-attack timing."""
        counter_attacks = 0
        total_opportunities = 0

        for i in range(2, len(sequence)):
            prev2 = sequence[i - 2]
            prev1 = sequence[i - 1]
            curr = sequence[i]

            # Pattern: Opponent attacks -> stops -> player attacks
            if (
                bool(prev2.opponent_attacking)
                and not bool(prev1.opponent_attacking)
                and bool(curr.player_attacking)
            ):
                counter_attacks += 1

            # Count opportunities (opponent attack endings)
            if bool(prev1.opponent_attacking) and not bool(curr.opponent_attacking):
                total_opportunities += 1

        return counter_attacks / total_opportunities if total_opportunities > 0 else 0.0

    def _calculate_pressure_relief(self, sequence: List[GameState]) -> float:
        """Calculate how well the player relieves opponent pressure."""
        pressure_frames = 0
        relief_frames = 0

        for i, state in enumerate(sequence):
            distance = ensure_scalar(state.distance, 80.0)
            # High pressure: close distance + opponent attacking
            if distance < 50 and bool(state.opponent_attacking):
                pressure_frames += 1

                # Check if pressure was relieved in next few frames
                for j in range(i + 1, min(i + 5, len(sequence))):
                    next_distance = ensure_scalar(sequence[j].distance, 80.0)
                    if next_distance > 60 or not bool(sequence[j].opponent_attacking):
                        relief_frames += 1
                        break

        return relief_frames / pressure_frames if pressure_frames > 0 else 1.0


class BaitPunishDetector:
    """Detects and rewards bait->block->punish sequences using learned patterns."""

    def __init__(self, history_length=60):
        self.history_length = history_length
        self.state_history = deque(maxlen=history_length)
        self.pattern_learner = SequencePattern()

        # Sequence state tracking
        self.current_phase = "neutral"  # neutral, baiting, defending, punishing
        self.sequence_start_frame = -1
        self.sequence_quality_score = 0.0

        # Learning data
        self.successful_sequences = []
        self.sequence_outcomes = deque(maxlen=500)

        # Adaptive thresholds (learned from data)
        self.bait_movement_threshold = 15.0  # Will adapt
        self.defense_damage_reduction = 0.7  # Will adapt
        self.punish_timing_window = 8  # Will adapt

        # Pattern recognition weights (learned)
        self.pattern_weights = np.ones(20, dtype=np.float32) * 0.05  # Will be updated

    def update(
        self,
        info: Dict,
        button_features: np.ndarray,
        damage_dealt: int,
        damage_received: int,
    ) -> Dict:
        """Update the detector with new frame data."""

        # Create current game state with safe scalar extraction
        state = GameState(
            player_x=ensure_scalar(info.get("agent_x", 160), 160.0),
            opponent_x=ensure_scalar(info.get("enemy_x", 160), 160.0),
            player_health=ensure_scalar(info.get("agent_hp", 176), 176),
            opponent_health=ensure_scalar(info.get("enemy_hp", 176), 176),
            player_attacking=self._is_attacking(button_features),
            opponent_attacking=self._infer_opponent_attacking(info, damage_received),
            player_blocking=self._is_blocking(button_features, damage_received),
            distance=abs(
                ensure_scalar(info.get("agent_x", 160), 160.0)
                - ensure_scalar(info.get("enemy_x", 160), 160.0)
            ),
            player_damage_taken=ensure_scalar(damage_received, 0),
            opponent_damage_taken=ensure_scalar(damage_dealt, 0),
            button_state=sanitize_array(button_features, 0.0),
            frame_number=len(self.state_history),
        )

        self.state_history.append(state)
        self.pattern_learner.add_frame(state)

        # Analyze current sequence
        sequence_reward, sequence_info = self._analyze_sequence(state)

        # Update learning
        self._update_pattern_learning()

        return {
            "bait_punish_reward": ensure_scalar(sequence_reward, 0.0),
            "sequence_phase": self.current_phase,
            "sequence_quality": ensure_scalar(self.sequence_quality_score, 0.0),
            **sequence_info,
        }

    def _is_attacking(self, button_features: np.ndarray) -> bool:
        """Detect if player is attacking based on button inputs."""
        try:
            button_features = sanitize_array(button_features, 0.0)
            attack_buttons = [0, 1, 8, 9, 10, 11]  # B, Y, A, X, L, R
            return bool(np.any(button_features[attack_buttons] > 0.5))
        except:
            return False

    def _infer_opponent_attacking(self, info: Dict, damage_received: int) -> bool:
        """Infer if opponent is attacking based on game state and damage."""
        # Multiple indicators of opponent attacking
        indicators = []

        # Direct damage received
        damage_received = ensure_scalar(damage_received, 0)
        if damage_received > 0:
            indicators.append(True)

        # Opponent status/animation changes (if available)
        if "enemy_status" in info:
            enemy_status = ensure_scalar(info["enemy_status"], 0)
            # Certain status values might indicate attacking
            indicators.append(enemy_status > 0)

        # Distance closing rapidly (rushing attack)
        if len(self.state_history) > 1:
            prev_distance = ensure_scalar(self.state_history[-1].distance, 80.0)
            curr_distance = abs(
                ensure_scalar(info.get("agent_x", 160), 160.0)
                - ensure_scalar(info.get("enemy_x", 160), 160.0)
            )
            if prev_distance - curr_distance > 3:  # Rapid approach
                indicators.append(True)

        # Use majority vote or any positive indicator
        return any(indicators) if indicators else False

    def _is_blocking(self, button_features: np.ndarray, damage_received: int) -> bool:
        """Detect if player is blocking."""
        try:
            button_features = sanitize_array(button_features, 0.0)
            damage_received = ensure_scalar(damage_received, 0)

            # Check for defensive button inputs
            defensive_buttons = [4, 5, 6, 7]  # UP, DOWN, LEFT, RIGHT
            defensive_input = bool(np.any(button_features[defensive_buttons] > 0.5))

            # If damage was received but less than expected, might be blocking
            expected_damage = 10  # Typical attack damage
            reduced_damage = damage_received < expected_damage and damage_received > 0

            return defensive_input or reduced_damage
        except:
            return False

    def _analyze_sequence(self, current_state: GameState) -> Tuple[float, Dict]:
        """Analyze the current sequence for bait->block->punish patterns."""

        if len(self.state_history) < 10:
            return 0.0, {}

        reward = 0.0
        info = {}

        # Get recent sequence for analysis
        recent_sequence = list(self.state_history)[-30:]

        # Phase detection and transition
        new_phase = self._detect_phase_transition(recent_sequence)

        if new_phase != self.current_phase:
            phase_reward = self._evaluate_phase_transition(
                self.current_phase, new_phase, recent_sequence
            )
            reward += phase_reward
            self.current_phase = new_phase

            if new_phase == "baiting":
                self.sequence_start_frame = current_state.frame_number

        # Sequence quality assessment
        if self.current_phase in ["defending", "punishing"]:
            quality = self._assess_sequence_quality(recent_sequence)
            self.sequence_quality_score = quality
            reward += quality * 0.1  # Small ongoing reward for good sequences

        # Complete sequence reward
        if self._is_sequence_complete(recent_sequence):
            completion_reward = self._evaluate_complete_sequence(recent_sequence)
            reward += completion_reward
            self._record_sequence_outcome(recent_sequence, completion_reward)

        info.update(
            {
                "phase_transition": new_phase != self.current_phase,
                "sequence_length": current_state.frame_number
                - self.sequence_start_frame,
                "bait_quality": ensure_scalar(
                    self._assess_bait_quality(recent_sequence), 0.0
                ),
                "defense_quality": ensure_scalar(
                    self._assess_defense_quality(recent_sequence), 0.0
                ),
                "punish_quality": ensure_scalar(
                    self._assess_punish_quality(recent_sequence), 0.0
                ),
            }
        )

        return ensure_scalar(reward, 0.0), info

    def _detect_phase_transition(self, sequence: List[GameState]) -> str:
        """Detect which phase of bait->block->punish we're in."""
        if len(sequence) < 5:
            return "neutral"

        recent_states = sequence[-8:]

        # Baiting detection: Movement without attacking, specific range control
        baiting_score = self._calculate_baiting_score(recent_states)

        # Defending detection: Blocking, damage reduction, defensive positioning
        defending_score = self._calculate_defending_score(recent_states)

        # Punishing detection: Counter-attacks after opponent attacks
        punishing_score = self._calculate_punishing_score(recent_states)

        # Determine phase based on highest score
        scores = {
            "baiting": baiting_score,
            "defending": defending_score,
            "punishing": punishing_score,
            "neutral": 0.3,  # Default baseline
        }

        return max(scores, key=scores.get)

    def _calculate_baiting_score(self, states: List[GameState]) -> float:
        """Calculate likelihood that player is baiting opponent."""
        if len(states) < 3:
            return 0.0

        score = 0.0

        # Movement in and out of range without attacking
        range_changes = 0
        attack_count = 0

        for i in range(1, len(states)):
            prev_dist = ensure_scalar(states[i - 1].distance, 80.0)
            curr_dist = ensure_scalar(states[i].distance, 80.0)

            # Check for range manipulation
            if abs(curr_dist - prev_dist) > 3:
                range_changes += 1

            # Count attacks
            if bool(states[i].player_attacking):
                attack_count += 1

        # High movement, low attacks = baiting
        movement_score = min(range_changes / len(states), 1.0)
        non_attack_score = 1.0 - (attack_count / len(states))

        score = movement_score * 0.6 + non_attack_score * 0.4

        # Bonus for optimal baiting range (just outside attack range)
        avg_distance = np.mean([ensure_scalar(s.distance, 80.0) for s in states])
        if 60 <= avg_distance <= 90:  # Sweet spot for baiting
            score += 0.2

        return min(score, 1.0)

    def _calculate_defending_score(self, states: List[GameState]) -> float:
        """Calculate likelihood that player is defending."""
        if len(states) < 3:
            return 0.0

        score = 0.0

        # Check for defensive actions
        blocking_frames = sum(1 for s in states if bool(s.player_blocking))
        opponent_attack_frames = sum(1 for s in states if bool(s.opponent_attacking))
        damage_taken = sum(ensure_scalar(s.player_damage_taken, 0) for s in states)

        # High blocking during opponent attacks
        if opponent_attack_frames > 0:
            defense_ratio = blocking_frames / len(states)
            score += defense_ratio * 0.5

            # Bonus for reduced damage (successful blocking)
            expected_damage = opponent_attack_frames * 8  # Rough estimate
            if damage_taken < expected_damage * 0.7:
                score += 0.3

        # Defensive positioning (maintaining distance)
        distances = [ensure_scalar(s.distance, 80.0) for s in states]
        if len(distances) > 1:
            mean_dist = np.mean(distances)
            if mean_dist > 0:
                distance_stability = 1.0 - (np.std(distances) / mean_dist)
                score += distance_stability * 0.2

        return min(score, 1.0)

    def _calculate_punishing_score(self, states: List[GameState]) -> float:
        """Calculate likelihood that player is punishing opponent."""
        if len(states) < 4:
            return 0.0

        score = 0.0

        # Look for counter-attack patterns
        for i in range(2, len(states)):
            prev2 = states[i - 2]
            prev1 = states[i - 1]
            curr = states[i]

            # Pattern: Opponent attacking -> not attacking -> player attacking
            if (
                bool(prev2.opponent_attacking)
                and not bool(prev1.opponent_attacking)
                and bool(curr.player_attacking)
            ):
                score += 0.4

            # Damage dealing after opponent vulnerability
            if (
                not bool(prev1.opponent_attacking)
                and ensure_scalar(curr.opponent_damage_taken, 0) > 0
            ):
                score += 0.3

        # Bonus for closing distance during punish
        if len(states) >= 4:
            early_dist = np.mean([ensure_scalar(s.distance, 80.0) for s in states[:2]])
            late_dist = np.mean([ensure_scalar(s.distance, 80.0) for s in states[-2:]])
            if early_dist > late_dist + 5:  # Moving in for punish
                score += 0.2

        return min(score, 1.0)

    def _evaluate_phase_transition(
        self, old_phase: str, new_phase: str, sequence: List[GameState]
    ) -> float:
        """Evaluate the quality of phase transitions."""
        transitions = {
            ("neutral", "baiting"): 0.02,
            ("baiting", "defending"): 0.05,
            ("defending", "punishing"): 0.1,
            ("punishing", "neutral"): 0.05,
            ("neutral", "defending"): 0.03,
            ("neutral", "punishing"): 0.08,
        }

        base_reward = transitions.get((old_phase, new_phase), 0.0)

        # Bonus for proper sequence flow
        if (old_phase, new_phase) in [
            ("baiting", "defending"),
            ("defending", "punishing"),
        ]:
            base_reward *= 1.5

        return base_reward

    def _assess_sequence_quality(self, sequence: List[GameState]) -> float:
        """Assess the quality of the current sequence."""
        if len(sequence) < 10:
            return 0.0

        # Extract features and use pattern weights
        features = self.pattern_learner.extract_sequence_features(sequence)
        quality = np.dot(features, self.pattern_weights)

        return max(0.0, min(1.0, quality))

    def _is_sequence_complete(self, sequence: List[GameState]) -> bool:
        """Check if a bait->block->punish sequence is complete."""
        if len(sequence) < 15:
            return False

        # Look for the full pattern in recent history
        phases = []
        for i in range(max(0, len(sequence) - 20), len(sequence), 5):
            segment = sequence[i : i + 5]
            if len(segment) >= 5:
                phase = self._detect_phase_transition(segment)
                phases.append(phase)

        # Check for bait->defend->punish pattern
        pattern_found = False
        for i in range(len(phases) - 2):
            if (
                phases[i] == "baiting"
                and phases[i + 1] == "defending"
                and phases[i + 2] == "punishing"
            ):
                pattern_found = True
                break

        return pattern_found

    def _evaluate_complete_sequence(self, sequence: List[GameState]) -> float:
        """Evaluate a complete bait->block->punish sequence."""
        if len(sequence) < 15:
            return 0.0

        # Base reward for completion
        base_reward = 0.5

        # Quality multipliers
        bait_quality = self._assess_bait_quality(sequence)
        defense_quality = self._assess_defense_quality(sequence)
        punish_quality = self._assess_punish_quality(sequence)

        # Timing bonus
        sequence_length = len(sequence)
        timing_bonus = 1.0 if 15 <= sequence_length <= 35 else 0.8

        # Damage efficiency bonus
        damage_dealt = sum(ensure_scalar(s.opponent_damage_taken, 0) for s in sequence)
        damage_received = sum(ensure_scalar(s.player_damage_taken, 0) for s in sequence)
        damage_ratio = damage_dealt / max(damage_received + 1, 1)
        damage_bonus = min(damage_ratio * 0.1, 0.3)

        total_reward = (
            base_reward * (bait_quality + defense_quality + punish_quality) / 3.0
        )
        total_reward = total_reward * timing_bonus + damage_bonus

        return ensure_scalar(total_reward, 0.0)

    def _assess_bait_quality(self, sequence: List[GameState]) -> float:
        """Assess the quality of baiting in the sequence."""
        if len(sequence) < 5:
            return 0.0

        bait_frames = 0
        quality_score = 0.0

        for i, state in enumerate(sequence):
            # Look for baiting indicators
            distance = ensure_scalar(state.distance, 80.0)
            player_attacking = bool(state.player_attacking)

            # Good baiting: optimal range, not attacking, movement
            if 60 <= distance <= 90 and not player_attacking:
                bait_frames += 1

                # Bonus for inducing opponent attacks
                if i < len(sequence) - 1:
                    next_state = sequence[i + 1]
                    if bool(next_state.opponent_attacking):
                        quality_score += 0.2

                # Bonus for range manipulation
                if i > 0:
                    prev_distance = ensure_scalar(sequence[i - 1].distance, 80.0)
                    if abs(distance - prev_distance) > 2:
                        quality_score += 0.1

        base_quality = bait_frames / len(sequence)
        return min(1.0, base_quality + quality_score)

    def _assess_defense_quality(self, sequence: List[GameState]) -> float:
        """Assess the quality of defensive play in the sequence."""
        if len(sequence) < 5:
            return 0.0

        defense_score = 0.0
        defensive_situations = 0

        for i, state in enumerate(sequence):
            if bool(state.opponent_attacking):
                defensive_situations += 1

                # Good defense: blocking or avoiding damage
                if bool(state.player_blocking):
                    defense_score += 0.3

                # Excellent defense: no damage taken during opponent attack
                damage_taken = ensure_scalar(state.player_damage_taken, 0)
                if damage_taken == 0:
                    defense_score += 0.4
                elif damage_taken < 5:  # Reduced damage (partial block)
                    defense_score += 0.2

                # Positioning defense: maintaining good distance
                distance = ensure_scalar(state.distance, 80.0)
                if distance > 40:  # Not too close
                    defense_score += 0.1

        return defense_score / max(defensive_situations, 1)

    def _assess_punish_quality(self, sequence: List[GameState]) -> float:
        """Assess the quality of punishing in the sequence."""
        if len(sequence) < 5:
            return 0.0

        punish_score = 0.0
        punish_opportunities = 0

        for i in range(1, len(sequence)):
            prev_state = sequence[i - 1]
            curr_state = sequence[i]

            # Punish opportunity: opponent stopped attacking
            if bool(prev_state.opponent_attacking) and not bool(
                curr_state.opponent_attacking
            ):
                punish_opportunities += 1

                # Check for immediate punish in next few frames
                for j in range(i, min(i + 5, len(sequence))):
                    punish_state = sequence[j]

                    # Player attacks after opponent vulnerability
                    if bool(punish_state.player_attacking):
                        punish_score += 0.4

                        # Bonus for damage dealt
                        damage_dealt = ensure_scalar(
                            punish_state.opponent_damage_taken, 0
                        )
                        if damage_dealt > 0:
                            punish_score += 0.3

                        # Bonus for quick punish
                        frames_to_punish = j - i
                        if frames_to_punish <= 2:
                            punish_score += 0.2

                        break

        return punish_score / max(punish_opportunities, 1)

    def _record_sequence_outcome(self, sequence: List[GameState], reward: float):
        """Record the outcome of a sequence for learning."""
        features = self.pattern_learner.extract_sequence_features(sequence)

        self.sequence_outcomes.append(
            {
                "features": features,
                "reward": ensure_scalar(reward, 0.0),
                "length": len(sequence),
                "success": reward > 0.3,
            }
        )

        # Update successful/failed sequence collections
        if reward > 0.3:
            self.successful_sequences.append(sequence)
            if len(self.successful_sequences) > 100:
                self.successful_sequences.pop(0)
        else:
            self.pattern_learner.failed_sequences.append(sequence)
            if len(self.pattern_learner.failed_sequences) > 100:
                self.pattern_learner.failed_sequences.pop(0)

    def _update_pattern_learning(self):
        """Update pattern recognition weights based on recent outcomes."""
        if len(self.sequence_outcomes) < 20:
            return

        # Get recent outcomes
        recent_outcomes = list(self.sequence_outcomes)[-50:]

        # Separate successful and failed sequences
        successful_features = []
        failed_features = []

        for outcome in recent_outcomes:
            if outcome["success"]:
                successful_features.append(outcome["features"])
            else:
                failed_features.append(outcome["features"])

        if len(successful_features) < 5 or len(failed_features) < 5:
            return

        # Calculate feature importance
        successful_mean = np.mean(successful_features, axis=0)
        failed_mean = np.mean(failed_features, axis=0)

        # Update weights based on feature discrimination
        feature_diff = successful_mean - failed_mean

        # Smooth weight updates
        learning_rate = 0.01
        self.pattern_weights = (
            1 - learning_rate
        ) * self.pattern_weights + learning_rate * feature_diff

        # Ensure weights stay reasonable
        self.pattern_weights = np.clip(self.pattern_weights, 0.0, 0.2)

    def get_pattern_confidence(self, sequence: List[GameState]) -> float:
        """Get confidence score for current pattern recognition."""
        if len(sequence) < 10:
            return 0.0

        features = self.pattern_learner.extract_sequence_features(sequence)
        confidence = np.dot(features, self.pattern_weights)

        return max(0.0, min(1.0, confidence))

    def reset_sequence(self):
        """Reset sequence tracking (e.g., on round end)."""
        self.current_phase = "neutral"
        self.sequence_start_frame = -1
        self.sequence_quality_score = 0.0

    def get_learning_stats(self) -> Dict:
        """Get statistics about the learning process."""
        return {
            "successful_sequences": len(self.successful_sequences),
            "failed_sequences": len(self.pattern_learner.failed_sequences),
            "total_outcomes": len(self.sequence_outcomes),
            "pattern_weights_sum": float(np.sum(self.pattern_weights)),
            "avg_pattern_weight": float(np.mean(self.pattern_weights)),
            "current_phase": self.current_phase,
            "sequence_quality": ensure_scalar(self.sequence_quality_score, 0.0),
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize detector
    detector = BaitPunishDetector()

    # Simulate some game frames
    for frame in range(100):
        # Mock game info
        info = {
            "agent_x": 160 + np.sin(frame * 0.1) * 20,
            "enemy_x": 160 + np.cos(frame * 0.1) * 30,
            "agent_hp": 176,
            "enemy_hp": 176 - max(0, frame - 80),
        }

        # Mock button features (12-dimensional)
        buttons = np.zeros(12)
        if frame % 20 < 3:  # Simulate periodic attacks
            buttons[0] = 1.0  # Attack button

        # Mock damage
        damage_dealt = 5 if frame % 25 == 0 else 0
        damage_received = 3 if frame % 30 == 0 else 0

        # Update detector
        result = detector.update(info, buttons, damage_dealt, damage_received)

        if frame % 20 == 0:
            print(
                f"Frame {frame}: Phase={result['sequence_phase']}, "
                f"Reward={result['bait_punish_reward']:.3f}, "
                f"Quality={result['sequence_quality']:.3f}"
            )

    # Print learning stats
    stats = detector.get_learning_stats()
    print(f"\nLearning Stats: {stats}")
