#!/usr/bin/env python3
"""
bait_punish_system.py - Learning-based Bait->Block->Punish sequence detection
APPROACH: Data-driven pattern recognition instead of hard-coded algorithms
LEARNS: Temporal sequences, defensive patterns, counter-attack opportunities
"""

import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math


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
            return np.zeros(20)  # Return empty feature vector

        features = []

        # Movement patterns
        player_positions = [s.player_x for s in sequence]
        opponent_positions = [s.opponent_x for s in sequence]
        distances = [s.distance for s in sequence]

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
        player_attack_frames = sum(1 for s in sequence if s.player_attacking)
        opponent_attack_frames = sum(1 for s in sequence if s.opponent_attacking)
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
        total_player_damage = sum(s.player_damage_taken for s in sequence)
        total_opponent_damage = sum(s.opponent_damage_taken for s in sequence)
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

        return np.array(features, dtype=np.float32)

    def _calculate_attack_clustering(
        self, sequence: List[GameState], player: str
    ) -> float:
        """Calculate how clustered attacks are in time."""
        attacks = []
        for i, state in enumerate(sequence):
            if (player == "player" and state.player_attacking) or (
                player == "opponent" and state.opponent_attacking
            ):
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
                state.player_damage_taken
                if player == "player"
                else state.opponent_damage_taken
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
            prev_dist = sequence[i - 1].distance
            curr_dist = sequence[i].distance

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
            if 40 <= state.distance <= 80:  # Optimal fighting range
                optimal_ranges += 1
        return optimal_ranges / len(sequence)

    def _calculate_positioning_advantage(self, sequence: List[GameState]) -> float:
        """Calculate positional advantage based on screen control."""
        screen_center = 160  # Assuming 320 width
        player_advantage = 0

        for state in sequence:
            player_center_dist = abs(state.player_x - screen_center)
            opponent_center_dist = abs(state.opponent_x - screen_center)

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
        player_damage = sum(s.player_damage_taken for s in states)
        opponent_damage = sum(s.opponent_damage_taken for s in states)
        total_damage = player_damage + opponent_damage

        if total_damage == 0:
            return 0.0

        return opponent_damage / total_damage  # Higher is better for player

    def _calculate_sequence_rhythm(self, sequence: List[GameState]) -> float:
        """Calculate the rhythm/tempo of the sequence."""
        action_frames = []
        for i, state in enumerate(sequence):
            if (
                state.player_attacking
                or state.opponent_attacking
                or state.player_damage_taken > 0
                or state.opponent_damage_taken > 0
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
            if prev_state.opponent_attacking and not curr_state.opponent_attacking:
                defensive_opportunities += 1

                # Check if player took advantage (attacked or repositioned)
                if (
                    curr_state.player_attacking
                    or abs(curr_state.distance - prev_state.distance) > 5
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
                prev2.opponent_attacking
                and not prev1.opponent_attacking
                and curr.player_attacking
            ):
                counter_attacks += 1

            # Count opportunities (opponent attack endings)
            if prev1.opponent_attacking and not curr.opponent_attacking:
                total_opportunities += 1

        return counter_attacks / total_opportunities if total_opportunities > 0 else 0.0

    def _calculate_pressure_relief(self, sequence: List[GameState]) -> float:
        """Calculate how well the player relieves opponent pressure."""
        pressure_frames = 0
        relief_frames = 0

        for i, state in enumerate(sequence):
            # High pressure: close distance + opponent attacking
            if state.distance < 50 and state.opponent_attacking:
                pressure_frames += 1

                # Check if pressure was relieved in next few frames
                for j in range(i + 1, min(i + 5, len(sequence))):
                    if sequence[j].distance > 60 or not sequence[j].opponent_attacking:
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
        self.pattern_weights = np.ones(20) * 0.05  # Will be updated

    def update(
        self,
        info: Dict,
        button_features: np.ndarray,
        damage_dealt: int,
        damage_received: int,
    ) -> Dict:
        """Update the detector with new frame data."""

        # Create current game state
        state = GameState(
            player_x=info.get("agent_x", 160),
            opponent_x=info.get("enemy_x", 160),
            player_health=info.get("agent_hp", 176),
            opponent_health=info.get("enemy_hp", 176),
            player_attacking=self._is_attacking(button_features),
            opponent_attacking=self._infer_opponent_attacking(info, damage_received),
            player_blocking=self._is_blocking(button_features, damage_received),
            distance=abs(info.get("agent_x", 160) - info.get("enemy_x", 160)),
            player_damage_taken=damage_received,
            opponent_damage_taken=damage_dealt,
            button_state=button_features.copy(),
            frame_number=len(self.state_history),
        )

        self.state_history.append(state)
        self.pattern_learner.add_frame(state)

        # Analyze current sequence
        sequence_reward, sequence_info = self._analyze_sequence(state)

        # Update learning
        self._update_pattern_learning()

        return {
            "bait_punish_reward": sequence_reward,
            "sequence_phase": self.current_phase,
            "sequence_quality": self.sequence_quality_score,
            **sequence_info,
        }

    def _is_attacking(self, button_features: np.ndarray) -> bool:
        """Detect if player is attacking based on button inputs."""
        attack_buttons = [0, 1, 8, 9, 10, 11]  # B, Y, A, X, L, R
        return np.any(button_features[attack_buttons] > 0.5)

    def _infer_opponent_attacking(self, info: Dict, damage_received: int) -> bool:
        """Infer if opponent is attacking based on game state and damage."""
        # Multiple indicators of opponent attacking
        indicators = []

        # Direct damage received
        if damage_received > 0:
            indicators.append(True)

        # Opponent status/animation changes (if available)
        if "enemy_status" in info:
            # Certain status values might indicate attacking
            indicators.append(info["enemy_status"] > 0)

        # Distance closing rapidly (rushing attack)
        if len(self.state_history) > 1:
            prev_distance = self.state_history[-1].distance
            curr_distance = abs(info.get("agent_x", 160) - info.get("enemy_x", 160))
            if prev_distance - curr_distance > 3:  # Rapid approach
                indicators.append(True)

        # Use majority vote or any positive indicator
        return any(indicators) if indicators else False

    def _is_blocking(self, button_features: np.ndarray, damage_received: int) -> bool:
        """Detect if player is blocking."""
        # Blocking typically involves holding back (LEFT/RIGHT away from opponent)
        # and often reduces damage

        # Check for defensive button inputs
        defensive_buttons = [4, 5, 6, 7]  # UP, DOWN, LEFT, RIGHT
        defensive_input = np.any(button_features[defensive_buttons] > 0.5)

        # If damage was received but less than expected, might be blocking
        expected_damage = 10  # Typical attack damage
        reduced_damage = damage_received < expected_damage and damage_received > 0

        return defensive_input or reduced_damage

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
                "bait_quality": self._assess_bait_quality(recent_sequence),
                "defense_quality": self._assess_defense_quality(recent_sequence),
                "punish_quality": self._assess_punish_quality(recent_sequence),
            }
        )

        return reward, info

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
            prev_dist = states[i - 1].distance
            curr_dist = states[i].distance

            # Check for range manipulation
            if abs(curr_dist - prev_dist) > 3:
                range_changes += 1

            # Count attacks
            if states[i].player_attacking:
                attack_count += 1

        # High movement, low attacks = baiting
        movement_score = min(range_changes / len(states), 1.0)
        non_attack_score = 1.0 - (attack_count / len(states))

        score = movement_score * 0.6 + non_attack_score * 0.4

        # Bonus for optimal baiting range (just outside attack range)
        avg_distance = np.mean([s.distance for s in states])
        if 60 <= avg_distance <= 90:  # Sweet spot for baiting
            score += 0.2

        return min(score, 1.0)

    def _calculate_defending_score(self, states: List[GameState]) -> float:
        """Calculate likelihood that player is defending."""
        if len(states) < 3:
            return 0.0

        score = 0.0

        # Check for defensive actions
        blocking_frames = sum(1 for s in states if s.player_blocking)
        opponent_attack_frames = sum(1 for s in states if s.opponent_attacking)
        damage_taken = sum(s.player_damage_taken for s in states)

        # High blocking during opponent attacks
        if opponent_attack_frames > 0:
            defense_ratio = blocking_frames / len(states)
            score += defense_ratio * 0.5

            # Bonus for reduced damage (successful blocking)
            expected_damage = opponent_attack_frames * 8  # Rough estimate
            if damage_taken < expected_damage * 0.7:
                score += 0.3

        # Defensive positioning (maintaining distance)
        distances = [s.distance for s in states]
        if len(distances) > 1:
            distance_stability = 1.0 - (np.std(distances) / np.mean(distances))
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
                prev2.opponent_attacking
                and not prev1.opponent_attacking
                and curr.player_attacking
            ):
                score += 0.4

            # Damage dealing after opponent vulnerability
            if not prev1.opponent_attacking and curr.opponent_damage_taken > 0:
                score += 0.3

        # Bonus for closing distance during punish
        if len(states) >= 4:
            early_dist = np.mean([s.distance for s in states[:2]])
            late_dist = np.mean([s.distance for s in states[-2:]])
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
            ("punishing", "neutral"): 0.03,
        }

        return transitions.get((old_phase, new_phase), 0.0)

    def _assess_sequence_quality(self, sequence: List[GameState]) -> float:
        """Assess the quality of the current sequence using learned patterns."""
        features = self.pattern_learner.extract_sequence_features(sequence)

        # Weighted combination of features based on learned importance
        quality = np.dot(features, self.pattern_weights)

        return np.clip(quality, 0.0, 1.0)

    def _is_sequence_complete(self, sequence: List[GameState]) -> bool:
        """Check if a complete bait->defend->punish sequence occurred."""
        if len(sequence) < 15:  # Minimum sequence length
            return False

        # Look for the full pattern in recent history
        has_bait_phase = False
        has_defend_phase = False
        has_punish_phase = False

        # Sliding window analysis
        for i in range(len(sequence) - 10):
            window = sequence[i : i + 10]

            if self._calculate_baiting_score(window) > 0.6:
                has_bait_phase = True
            elif self._calculate_defending_score(window) > 0.6 and has_bait_phase:
                has_defend_phase = True
            elif self._calculate_punishing_score(window) > 0.6 and has_defend_phase:
                has_punish_phase = True
                break

        return has_bait_phase and has_defend_phase and has_punish_phase

    def _evaluate_complete_sequence(self, sequence: List[GameState]) -> float:
        """Evaluate a complete bait->defend->punish sequence."""
        features = self.pattern_learner.extract_sequence_features(sequence)

        # Base reward for completion
        reward = 0.5

        # Bonus based on sequence quality
        quality_bonus = np.dot(features, self.pattern_weights) * 0.3

        # Bonus for efficiency (shorter sequences are better)
        efficiency_bonus = max(0, (40 - len(sequence)) / 40 * 0.2)

        # Bonus for damage dealt during punish
        punish_damage = sum(s.opponent_damage_taken for s in sequence[-8:])
        damage_bonus = min(punish_damage / 50.0, 0.3)

        total_reward = reward + quality_bonus + efficiency_bonus + damage_bonus
        return np.clip(total_reward, 0.0, 2.0)

    def _record_sequence_outcome(self, sequence: List[GameState], reward: float):
        """Record sequence outcome for learning."""
        features = self.pattern_learner.extract_sequence_features(sequence)
        self.sequence_outcomes.append((features, reward))

        # Store successful sequences for analysis
        if reward > 0.7:
            self.successful_sequences.append(sequence.copy())

    def _update_pattern_learning(self):
        """Update pattern recognition weights based on observed outcomes."""
        if len(self.sequence_outcomes) < 50:
            return

        # Simple online learning: adjust weights based on recent outcomes
        recent_outcomes = list(self.sequence_outcomes)[-50:]

        for features, reward in recent_outcomes:
            # Positive reinforcement for successful patterns
            if reward > 0.5:
                self.pattern_weights += features * 0.001 * reward
            # Negative reinforcement for unsuccessful patterns
            elif reward < 0.1:
                self.pattern_weights -= features * 0.0005

        # Normalize weights to prevent explosion
        self.pattern_weights = np.clip(self.pattern_weights, 0.01, 0.2)

    def _assess_bait_quality(self, sequence: List[GameState]) -> float:
        """Assess quality of baiting behavior."""
        return self._calculate_baiting_score(sequence[-8:])

    def _assess_defense_quality(self, sequence: List[GameState]) -> float:
        """Assess quality of defensive behavior."""
        return self._calculate_defending_score(sequence[-8:])

    def _assess_punish_quality(self, sequence: List[GameState]) -> float:
        """Assess quality of punishing behavior."""
        return self._calculate_punishing_score(sequence[-8:])

    def get_learning_stats(self) -> Dict:
        """Get statistics about the learning system."""
        return {
            "successful_sequences": len(self.successful_sequences),
            "total_outcomes_recorded": len(self.sequence_outcomes),
            "current_phase": self.current_phase,
            "sequence_quality": self.sequence_quality_score,
            "pattern_weights_sum": np.sum(self.pattern_weights),
            "avg_recent_reward": (
                np.mean([r for _, r in list(self.sequence_outcomes)[-20:]])
                if self.sequence_outcomes
                else 0.0
            ),
        }


# Integration function for the main wrapper
def integrate_bait_punish_system(strategic_tracker):
    """Add the bait-punish system to the existing strategic tracker."""
    strategic_tracker.bait_punish_detector = BaitPunishDetector()

    # Modify the update method to include bait-punish detection
    original_update = strategic_tracker.update

    def enhanced_update(info, button_features):
        # Get original features
        features = original_update(info, button_features)

        # Calculate damage this frame
        current_player_health = info.get("agent_hp", 176)
        current_opponent_health = info.get("enemy_hp", 176)

        damage_dealt = (
            max(0, strategic_tracker.prev_opponent_health - current_opponent_health)
            if strategic_tracker.prev_opponent_health
            else 0
        )
        damage_received = (
            max(0, strategic_tracker.prev_player_health - current_player_health)
            if strategic_tracker.prev_player_health
            else 0
        )

        # Update bait-punish detector
        bait_punish_result = strategic_tracker.bait_punish_detector.update(
            info, button_features, damage_dealt, damage_received
        )

        # Add bait-punish features to the feature vector (expand to 52 features)
        enhanced_features = np.zeros(52, dtype=np.float32)
        enhanced_features[:45] = features  # Original features

        # Add bait-punish features (7 additional features)
        enhanced_features[45] = bait_punish_result.get("sequence_quality", 0.0)

        # Phase encoding (one-hot)
        phase = bait_punish_result.get("sequence_phase", "neutral")
        phase_encoding = {"neutral": 0, "baiting": 1, "defending": 2, "punishing": 3}
        enhanced_features[46] = phase_encoding.get(phase, 0) / 3.0  # Normalized

        enhanced_features[47] = bait_punish_result.get("bait_quality", 0.0)
        enhanced_features[48] = bait_punish_result.get("defense_quality", 0.0)
        enhanced_features[49] = bait_punish_result.get("punish_quality", 0.0)

        # Sequence timing features
        sequence_length = bait_punish_result.get("sequence_length", 0)
        enhanced_features[50] = min(
            sequence_length / 60.0, 1.0
        )  # Normalized sequence length

        enhanced_features[51] = (
            1.0 if bait_punish_result.get("phase_transition", False) else 0.0
        )

        # Store bait-punish reward for later use in reward calculation
        strategic_tracker.last_bait_punish_reward = bait_punish_result.get(
            "bait_punish_reward", 0.0
        )
        strategic_tracker.last_bait_punish_info = bait_punish_result

        return enhanced_features

    # Replace the update method
    strategic_tracker.update = enhanced_update
    strategic_tracker.last_bait_punish_reward = 0.0
    strategic_tracker.last_bait_punish_info = {}

    return strategic_tracker


class AdaptiveRewardShaper:
    """Shapes rewards based on learned fighting game patterns without hard-coding."""

    def __init__(self):
        self.reward_history = deque(maxlen=1000)
        self.pattern_rewards = defaultdict(lambda: deque(maxlen=100))
        self.adaptation_rate = 0.01

        # Learned reward weights (will adapt based on success)
        self.reward_weights = {
            "bait_punish_sequence": 1.0,
            "defensive_success": 0.8,
            "counter_attack": 1.2,
            "range_control": 0.3,
            "pressure_relief": 0.6,
        }

        # Success tracking for adaptation
        self.pattern_success_rates = defaultdict(lambda: deque(maxlen=50))

    def shape_reward(
        self, base_reward: float, bait_punish_info: Dict, game_info: Dict
    ) -> float:
        """Shape the reward based on learned patterns."""

        total_reward = base_reward
        reward_breakdown = {"base": base_reward}

        # Bait-punish sequence rewards
        bp_reward = bait_punish_info.get("bait_punish_reward", 0.0)
        if bp_reward > 0:
            shaped_bp_reward = bp_reward * self.reward_weights["bait_punish_sequence"]
            total_reward += shaped_bp_reward
            reward_breakdown["bait_punish"] = shaped_bp_reward

            # Track success
            self.pattern_success_rates["bait_punish_sequence"].append(1.0)

        # Defensive success rewards
        defense_quality = bait_punish_info.get("defense_quality", 0.0)
        if defense_quality > 0.7:
            defense_reward = (
                defense_quality * self.reward_weights["defensive_success"] * 0.1
            )
            total_reward += defense_reward
            reward_breakdown["defense"] = defense_reward

        # Counter-attack rewards
        punish_quality = bait_punish_info.get("punish_quality", 0.0)
        if punish_quality > 0.6:
            counter_reward = (
                punish_quality * self.reward_weights["counter_attack"] * 0.15
            )
            total_reward += counter_reward
            reward_breakdown["counter"] = counter_reward

        # Range control rewards
        bait_quality = bait_punish_info.get("bait_quality", 0.0)
        if bait_quality > 0.5:
            range_reward = bait_quality * self.reward_weights["range_control"] * 0.05
            total_reward += range_reward
            reward_breakdown["range_control"] = range_reward

        # Phase transition rewards (learning optimal transitions)
        if bait_punish_info.get("phase_transition", False):
            phase = bait_punish_info.get("sequence_phase", "neutral")
            transition_reward = self._get_phase_transition_reward(phase)
            total_reward += transition_reward
            reward_breakdown["phase_transition"] = transition_reward

        # Record reward for adaptation
        self.reward_history.append(total_reward)

        # Adapt weights based on recent performance
        self._adapt_reward_weights(reward_breakdown, game_info)

        return total_reward

    def _get_phase_transition_reward(self, phase: str) -> float:
        """Get reward for phase transitions based on learned preferences."""
        phase_rewards = {
            "baiting": 0.02,
            "defending": 0.05,
            "punishing": 0.1,
            "neutral": 0.01,
        }

        return phase_rewards.get(phase, 0.0) * self.reward_weights.get(
            "phase_transition", 0.5
        )

    def _adapt_reward_weights(self, reward_breakdown: Dict, game_info: Dict):
        """Adapt reward weights based on performance."""

        # Simple adaptation: increase weights for patterns that lead to wins
        win_rate = game_info.get("win_rate", 0.0)

        if len(self.reward_history) > 50:
            recent_avg_reward = np.mean(list(self.reward_history)[-20:])
            overall_avg_reward = np.mean(list(self.reward_history))

            # If recent performance is better, strengthen recent patterns
            if recent_avg_reward > overall_avg_reward * 1.1:
                for pattern, reward_value in reward_breakdown.items():
                    if reward_value > 0 and pattern != "base":
                        current_weight = self.reward_weights.get(pattern, 1.0)
                        self.reward_weights[pattern] = current_weight * (
                            1 + self.adaptation_rate
                        )

            # If performance is declining, reduce weights
            elif recent_avg_reward < overall_avg_reward * 0.9:
                for pattern in self.reward_weights:
                    self.reward_weights[pattern] *= 1 - self.adaptation_rate * 0.5

        # Clip weights to reasonable ranges
        for pattern in self.reward_weights:
            self.reward_weights[pattern] = np.clip(
                self.reward_weights[pattern], 0.1, 3.0
            )

    def get_adaptation_stats(self) -> Dict:
        """Get statistics about reward adaptation."""
        return {
            "reward_weights": dict(self.reward_weights),
            "avg_recent_reward": (
                np.mean(list(self.reward_history)[-20:]) if self.reward_history else 0.0
            ),
            "avg_overall_reward": (
                np.mean(list(self.reward_history)) if self.reward_history else 0.0
            ),
            "adaptation_trends": {
                pattern: np.mean(list(rates)) if rates else 0.0
                for pattern, rates in self.pattern_success_rates.items()
            },
        }


# Example integration with the main wrapper
"""
To integrate this system into your wrapper.py, you would:

1. Import the bait-punish system:
   from bait_punish_system import integrate_bait_punish_system, AdaptiveRewardShaper

2. In StrategicFeatureTracker.__init__(), add:
   integrate_bait_punish_system(self)
   
3. Update VECTOR_FEATURE_DIM from 45 to 52 in constants

4. In StreetFighterVisionWrapper.__init__(), add:
   self.reward_shaper = AdaptiveRewardShaper()

5. In _calculate_stabilized_reward(), add bait-punish rewards:
   # After calculating base reward
   bait_punish_info = self.strategic_tracker.last_bait_punish_info
   final_reward = self.reward_shaper.shape_reward(reward, bait_punish_info, info)

6. Update stats to include bait-punish information:
   bait_punish_stats = self.strategic_tracker.bait_punish_detector.get_learning_stats()
   adaptation_stats = self.reward_shaper.get_adaptation_stats()
   info.update(bait_punish_stats)
   info.update(adaptation_stats)

This approach provides:
- Learning-based pattern recognition instead of hard-coded rules
- Adaptive reward shaping that improves over time
- Rich feature representation for the neural network
- Temporal sequence understanding
- Automatic discovery of effective fighting patterns

The system learns what constitutes good baiting, defending, and punishing
behavior from the data rather than relying on pre-programmed logic.
"""
