#!/usr/bin/env python3
"""
bait_punish_system.py - Learning-based Bait->Block->Punish sequence detection
APPROACH: Data-driven pattern recognition instead of hard-coded algorithms
LEARNS: Temporal sequences, defensive patterns, counter-attack opportunities
FIX: Added ROBUST sanitization to the main integration function (`enhanced_update`) and the detector itself, fully preventing ValueError from array-like inputs from vectorized environments.
"""

import numpy as np
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math


# <<< FIX: Helper function to sanitize the info dictionary >>>
def _sanitize_info_dict(info: Dict) -> Dict:
    """Converts array values from a vectorized env's info dict to scalars."""
    sanitized = {}
    for k, v in info.items():
        if isinstance(v, np.ndarray):
            # Extract scalar, provide a default for empty arrays
            sanitized[k] = v.item(0) if v.size > 0 else 0
        else:
            sanitized[k] = v
    return sanitized


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
        self.current_sequence = deque(maxlen=pattern_length)
        self.pattern_weights = np.ones(20) * 0.05

    def add_frame(self, state: GameState):
        self.current_sequence.append(state)

    def extract_sequence_features(self, sequence: List[GameState]) -> np.ndarray:
        if len(sequence) < 5:
            return np.zeros(20)
        features = []
        player_positions = [s.player_x for s in sequence]
        opponent_positions = [s.opponent_x for s in sequence]
        distances = [s.distance for s in sequence]
        features.extend(
            [
                np.std(player_positions) / 50.0,
                np.std(opponent_positions) / 50.0,
                np.mean(distances) / 100.0,
                np.std(distances) / 50.0,
            ]
        )
        player_attack_frames = sum(1 for s in sequence if s.player_attacking)
        opponent_attack_frames = sum(1 for s in sequence if s.opponent_attacking)
        features.extend(
            [
                player_attack_frames / len(sequence),
                opponent_attack_frames / len(sequence),
                self._calculate_attack_clustering(sequence, "player"),
                self._calculate_attack_clustering(sequence, "opponent"),
            ]
        )
        total_player_damage = sum(s.player_damage_taken for s in sequence)
        total_opponent_damage = sum(s.opponent_damage_taken for s in sequence)
        features.extend(
            [
                total_player_damage / 100.0,
                total_opponent_damage / 100.0,
                self._calculate_damage_timing(sequence, "player"),
                self._calculate_damage_timing(sequence, "opponent"),
            ]
        )
        features.extend(
            [
                self._calculate_approach_retreat_ratio(sequence),
                self._calculate_range_control(sequence),
                self._calculate_positioning_advantage(sequence),
                self._calculate_momentum_shift(sequence),
            ]
        )
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
        attacks = [
            i
            for i, state in enumerate(sequence)
            if (player == "player" and state.player_attacking)
            or (player == "opponent" and state.opponent_attacking)
        ]
        if len(attacks) < 2:
            return 0.0
        attack_intervals = [
            attacks[i + 1] - attacks[i] for i in range(len(attacks) - 1)
        ]
        return (
            1.0 - (np.std(attack_intervals) / len(sequence))
            if attack_intervals
            else 0.0
        )

    def _calculate_damage_timing(self, sequence: List[GameState], player: str) -> float:
        damage_frames = [
            i
            for i, state in enumerate(sequence)
            if (
                state.player_damage_taken > 0
                if player == "player"
                else state.opponent_damage_taken > 0
            )
        ]
        if not damage_frames:
            return 0.5
        return np.mean(damage_frames) / len(sequence)

    def _calculate_approach_retreat_ratio(self, sequence: List[GameState]) -> float:
        approach, retreat = 0, 0
        for i in range(1, len(sequence)):
            if sequence[i].distance < sequence[i - 1].distance - 1:
                approach += 1
            elif sequence[i].distance > sequence[i - 1].distance + 1:
                retreat += 1
        total = approach + retreat
        return approach / total if total > 0 else 0.5

    def _calculate_range_control(self, sequence: List[GameState]) -> float:
        return sum(1 for s in sequence if 40 <= s.distance <= 80) / len(sequence)

    def _calculate_positioning_advantage(self, sequence: List[GameState]) -> float:
        screen_center = 160
        return sum(
            1
            for s in sequence
            if abs(s.opponent_x - screen_center) > abs(s.player_x - screen_center)
        ) / len(sequence)

    def _calculate_momentum_shift(self, sequence: List[GameState]) -> float:
        if len(sequence) < 2:
            return 0.0
        mid = len(sequence) // 2
        early_ratio = self._get_damage_ratio(sequence[:mid])
        late_ratio = self._get_damage_ratio(sequence[mid:])
        return late_ratio - early_ratio

    def _get_damage_ratio(self, states: List[GameState]) -> float:
        if not states:
            return 0.0
        p_dmg = sum(s.player_damage_taken for s in states)
        o_dmg = sum(s.opponent_damage_taken for s in states)
        total = p_dmg + o_dmg
        return o_dmg / total if total > 0 else 0.0

    def _calculate_sequence_rhythm(self, sequence: List[GameState]) -> float:
        actions = [
            i
            for i, s in enumerate(sequence)
            if s.player_attacking
            or s.opponent_attacking
            or s.player_damage_taken > 0
            or s.opponent_damage_taken > 0
        ]
        if len(actions) < 2:
            return 0.0
        intervals = [actions[i + 1] - actions[i] for i in range(len(actions) - 1)]
        return 1.0 / (1.0 + np.std(intervals))

    def _calculate_defensive_windows(self, sequence: List[GameState]) -> float:
        ops, successes = 0, 0
        for i in range(1, len(sequence)):
            if (
                sequence[i - 1].opponent_attacking
                and not sequence[i].opponent_attacking
            ):
                ops += 1
                if (
                    sequence[i].player_attacking
                    or abs(sequence[i].distance - sequence[i - 1].distance) > 5
                ):
                    successes += 1
        return successes / ops if ops > 0 else 0.0

    def _calculate_counter_attack_timing(self, sequence: List[GameState]) -> float:
        counters, ops = 0, 0
        for i in range(2, len(sequence)):
            if (
                sequence[i - 2].opponent_attacking
                and not sequence[i - 1].opponent_attacking
                and sequence[i].player_attacking
            ):
                counters += 1
            if (
                sequence[i - 1].opponent_attacking
                and not sequence[i].opponent_attacking
            ):
                ops += 1
        return counters / ops if ops > 0 else 0.0

    def _calculate_pressure_relief(self, sequence: List[GameState]) -> float:
        pressure, relief = 0, 0
        for i, state in enumerate(sequence):
            if state.distance < 50 and state.opponent_attacking:
                pressure += 1
                for j in range(i + 1, min(i + 5, len(sequence))):
                    if sequence[j].distance > 60 or not sequence[j].opponent_attacking:
                        relief += 1
                        break
        return relief / pressure if pressure > 0 else 1.0


class BaitPunishDetector:
    def __init__(self, history_length=60):
        self.history_length = history_length
        self.state_history = deque(maxlen=history_length)
        self.pattern_learner = SequencePattern()
        self.current_phase = "neutral"
        self.sequence_start_frame = -1
        self.sequence_quality_score = 0.0
        self.successful_sequences = []
        self.sequence_outcomes = deque(maxlen=500)
        self.pattern_weights = np.ones(20) * 0.05

    def update(
        self,
        info: Dict,
        button_features: np.ndarray,
        damage_dealt: int,
        damage_received: int,
    ) -> Dict:
        s_info = _sanitize_info_dict(info)
        player_x, opponent_x = float(s_info.get("agent_x", 160.0)), float(
            s_info.get("enemy_x", 160.0)
        )
        state = GameState(
            player_x=player_x,
            opponent_x=opponent_x,
            player_health=int(s_info.get("agent_hp", 176)),
            opponent_health=int(s_info.get("enemy_hp", 176)),
            player_attacking=self._is_attacking(button_features),
            opponent_attacking=self._infer_opponent_attacking(s_info, damage_received),
            player_blocking=self._is_blocking(button_features, damage_received),
            distance=abs(player_x - opponent_x),
            player_damage_taken=damage_received,
            opponent_damage_taken=damage_dealt,
            button_state=button_features.copy(),
            frame_number=len(self.state_history),
        )
        self.state_history.append(state)
        self.pattern_learner.add_frame(state)
        sequence_reward, sequence_info = self._analyze_sequence(state)
        self._update_pattern_learning()
        return {
            "bait_punish_reward": sequence_reward,
            "sequence_phase": self.current_phase,
            "sequence_quality": self.sequence_quality_score,
            **sequence_info,
        }

    def _is_attacking(self, bf: np.ndarray) -> bool:
        return np.any(bf[[0, 1, 8, 9, 10, 11]] > 0.5)

    def _is_blocking(self, bf: np.ndarray, dmg_rec: int) -> bool:
        return np.any(bf[[4, 5, 6, 7]] > 0.5) or (0 < dmg_rec < 10)

    def _infer_opponent_attacking(self, s_info: Dict, dmg_rec: int) -> bool:
        indicators = [dmg_rec > 0, s_info.get("enemy_status", 0) > 0]
        if len(self.state_history) > 1:
            indicators.append(
                self.state_history[-1].distance
                - abs(s_info.get("agent_x", 160) - s_info.get("enemy_x", 160))
                > 3
            )
        return any(indicators)

    def _analyze_sequence(self, current_state: GameState) -> Tuple[float, Dict]:
        if len(self.state_history) < 10:
            return 0.0, {}
        reward, info = 0.0, {}
        recent_sequence = list(self.state_history)[-30:]
        new_phase = self._detect_phase_transition(recent_sequence)
        if new_phase != self.current_phase:
            reward += self._evaluate_phase_transition(self.current_phase, new_phase)
            self.current_phase = new_phase
            if new_phase == "baiting":
                self.sequence_start_frame = current_state.frame_number
        if self.current_phase in ["defending", "punishing"]:
            quality = self._assess_sequence_quality(recent_sequence)
            self.sequence_quality_score, reward = quality, reward + quality * 0.1
        if self._is_sequence_complete(recent_sequence):
            completion_reward = self._evaluate_complete_sequence(recent_sequence)
            reward += completion_reward
            self._record_sequence_outcome(recent_sequence, completion_reward)
        info.update(
            {
                "phase_transition": new_phase != self.current_phase,
                "sequence_length": current_state.frame_number
                - self.sequence_start_frame,
                "bait_quality": self._calculate_baiting_score(recent_sequence[-8:]),
                "defense_quality": self._calculate_defending_score(
                    recent_sequence[-8:]
                ),
                "punish_quality": self._calculate_punishing_score(recent_sequence[-8:]),
            }
        )
        return reward, info

    def _detect_phase_transition(self, sequence: List[GameState]) -> str:
        if len(sequence) < 5:
            return "neutral"
        scores = {
            "baiting": self._calculate_baiting_score(sequence[-8:]),
            "defending": self._calculate_defending_score(sequence[-8:]),
            "punishing": self._calculate_punishing_score(sequence[-8:]),
            "neutral": 0.3,
        }
        return max(scores, key=scores.get)

    def _calculate_baiting_score(self, states: List[GameState]) -> float:
        if len(states) < 3:
            return 0.0
        range_changes = sum(
            1
            for i in range(1, len(states))
            if abs(states[i].distance - states[i - 1].distance) > 3
        )
        attacks = sum(1 for s in states if s.player_attacking)
        score = (range_changes / len(states)) * 0.6 + (
            1.0 - (attacks / len(states))
        ) * 0.4
        if 60 <= np.mean([s.distance for s in states]) <= 90:
            score += 0.2
        return min(score, 1.0)

    def _calculate_defending_score(self, states: List[GameState]) -> float:
        if len(states) < 3:
            return 0.0
        score = 0.0
        blocks = sum(1 for s in states if s.player_blocking)
        opp_attacks = sum(1 for s in states if s.opponent_attacking)
        dmg_taken = sum(s.player_damage_taken for s in states)
        if opp_attacks > 0:
            score += (blocks / len(states)) * 0.5
            if dmg_taken < (opp_attacks * 8) * 0.7:
                score += 0.3
        dists = [s.distance for s in states]
        if len(dists) > 1 and np.mean(dists) > 1e-6:
            score += (1.0 - (np.std(dists) / np.mean(dists))) * 0.2
        return min(score, 1.0)

    def _calculate_punishing_score(self, states: List[GameState]) -> float:
        if len(states) < 4:
            return 0.0
        score = 0.0
        for i in range(2, len(states)):
            if (
                states[i - 2].opponent_attacking
                and not states[i - 1].opponent_attacking
                and states[i].player_attacking
            ):
                score += 0.4
            if (
                not states[i - 1].opponent_attacking
                and states[i].opponent_damage_taken > 0
            ):
                score += 0.3
        if (
            len(states) >= 4
            and np.mean([s.distance for s in states[:2]])
            > np.mean([s.distance for s in states[-2:]]) + 5
        ):
            score += 0.2
        return min(score, 1.0)

    def _evaluate_phase_transition(self, old_phase: str, new_phase: str) -> float:
        return {
            ("neutral", "baiting"): 0.02,
            ("baiting", "defending"): 0.05,
            ("defending", "punishing"): 0.1,
            ("punishing", "neutral"): 0.03,
        }.get((old_phase, new_phase), 0.0)

    def _assess_sequence_quality(self, sequence: List[GameState]) -> float:
        features = self.pattern_learner.extract_sequence_features(sequence)
        return np.clip(np.dot(features, self.pattern_weights), 0.0, 1.0)

    def _is_sequence_complete(self, sequence: List[GameState]) -> bool:
        if len(sequence) < 15:
            return False
        b, d, p = False, False, False
        for i in range(len(sequence) - 10):
            w = sequence[i : i + 10]
            if self._calculate_baiting_score(w) > 0.6:
                b = True
            elif self._calculate_defending_score(w) > 0.6 and b:
                d = True
            elif self._calculate_punishing_score(w) > 0.6 and d:
                p = True
                break
        return b and d and p

    def _evaluate_complete_sequence(self, sequence: List[GameState]) -> float:
        features = self.pattern_learner.extract_sequence_features(sequence)
        reward = 0.5 + np.dot(features, self.pattern_weights) * 0.3
        reward += max(0, (40 - len(sequence)) / 40 * 0.2)
        reward += min(sum(s.opponent_damage_taken for s in sequence[-8:]) / 50.0, 0.3)
        return np.clip(reward, 0.0, 2.0)

    def _record_sequence_outcome(self, sequence: List[GameState], reward: float):
        features = self.pattern_learner.extract_sequence_features(sequence)
        self.sequence_outcomes.append((features, reward))
        if reward > 0.7:
            self.successful_sequences.append(list(sequence))

    def _update_pattern_learning(self):
        if len(self.sequence_outcomes) < 50:
            return
        for features, reward in list(self.sequence_outcomes)[-50:]:
            if reward > 0.5:
                self.pattern_weights += features * 0.001 * reward
            elif reward < 0.1:
                self.pattern_weights -= features * 0.0005
        self.pattern_weights = np.clip(self.pattern_weights, 0.01, 0.2)
        if np.sum(self.pattern_weights) > 1e-6:
            self.pattern_weights /= np.sum(self.pattern_weights)

    def get_learning_stats(self) -> Dict:
        avg_reward = 0.0
        if self.sequence_outcomes:
            recent_rewards = [r for _, r in list(self.sequence_outcomes)[-20:]]
            if recent_rewards:
                avg_reward = np.mean(recent_rewards)
        return {
            "successful_sequences": len(self.successful_sequences),
            "total_outcomes_recorded": len(self.sequence_outcomes),
            "current_phase": self.current_phase,
            "sequence_quality": self.sequence_quality_score,
            "pattern_weights_sum": np.sum(self.pattern_weights),
            "avg_recent_reward": avg_reward,
        }


def integrate_bait_punish_system(strategic_tracker):
    """Add the bait-punish system to the existing strategic tracker."""
    strategic_tracker.bait_punish_detector = BaitPunishDetector()
    original_update = strategic_tracker.update

    def enhanced_update(info, button_features):
        # <<< DEFINITIVE FIX: Sanitize info at the top of the new method >>>
        s_info = _sanitize_info_dict(info)

        # Call original method with sanitized info
        features = original_update(s_info, button_features)

        # Calculate damage using sanitized info
        current_player_health = s_info.get("agent_hp", 176)
        current_opponent_health = s_info.get("enemy_hp", 176)

        damage_dealt = (
            max(0, strategic_tracker.prev_opponent_health - current_opponent_health)
            if strategic_tracker.prev_opponent_health is not None
            else 0
        )
        damage_received = (
            max(0, strategic_tracker.prev_player_health - current_player_health)
            if strategic_tracker.prev_player_health is not None
            else 0
        )

        # Update bait-punish detector with the original, raw info, as the detector now has its own sanitizer.
        bait_punish_result = strategic_tracker.bait_punish_detector.update(
            info, button_features, damage_dealt, damage_received
        )

        # Combine features
        enhanced_features = np.zeros(52, dtype=np.float32)
        enhanced_features[:45] = features
        enhanced_features[45] = bait_punish_result.get("sequence_quality", 0.0)
        phase_map = {"neutral": 0, "baiting": 1, "defending": 2, "punishing": 3}
        enhanced_features[46] = (
            phase_map.get(bait_punish_result.get("sequence_phase", "neutral"), 0) / 3.0
        )
        enhanced_features[47] = bait_punish_result.get("bait_quality", 0.0)
        enhanced_features[48] = bait_punish_result.get("defense_quality", 0.0)
        enhanced_features[49] = bait_punish_result.get("punish_quality", 0.0)
        enhanced_features[50] = min(
            bait_punish_result.get("sequence_length", 0) / 60.0, 1.0
        )
        enhanced_features[51] = (
            1.0 if bait_punish_result.get("phase_transition", False) else 0.0
        )

        strategic_tracker.last_bait_punish_info = bait_punish_result
        return enhanced_features

    strategic_tracker.update = enhanced_update
    strategic_tracker.last_bait_punish_info = {}
    return strategic_tracker


class AdaptiveRewardShaper:
    def __init__(self):
        self.reward_history = deque(maxlen=1000)
        self.adaptation_rate = 0.01
        self.reward_weights = {
            "bait_punish_sequence": 1.0,
            "defensive_success": 0.8,
            "counter_attack": 1.2,
            "range_control": 0.3,
            "pressure_relief": 0.6,
            "phase_transition": 0.5,
        }
        self.pattern_success_rates = defaultdict(lambda: deque(maxlen=50))

    def shape_reward(
        self, base_reward: float, bait_punish_info: Dict, game_info: Dict
    ) -> float:
        total_reward = base_reward
        reward_breakdown = {"base": base_reward}

        # Sanitize game_info defensively
        s_game_info = _sanitize_info_dict(game_info)

        bp_reward = bait_punish_info.get("bait_punish_reward", 0.0)
        if bp_reward > 0:
            shaped_bp = bp_reward * self.reward_weights["bait_punish_sequence"]
            total_reward += shaped_bp
            reward_breakdown["bait_punish"] = shaped_bp
            self.pattern_success_rates["bait_punish_sequence"].append(1.0)

        if bait_punish_info.get("defense_quality", 0.0) > 0.7:
            total_reward += (
                bait_punish_info["defense_quality"]
                * self.reward_weights["defensive_success"]
                * 0.1
            )
        if bait_punish_info.get("punish_quality", 0.0) > 0.6:
            total_reward += (
                bait_punish_info["punish_quality"]
                * self.reward_weights["counter_attack"]
                * 0.15
            )
        if bait_punish_info.get("bait_quality", 0.0) > 0.5:
            total_reward += (
                bait_punish_info["bait_quality"]
                * self.reward_weights["range_control"]
                * 0.05
            )
        if bait_punish_info.get("phase_transition", False):
            phase_rewards = {
                "baiting": 0.02,
                "defending": 0.05,
                "punishing": 0.1,
                "neutral": 0.01,
            }
            total_reward += (
                phase_rewards.get(bait_punish_info.get("sequence_phase"), 0.0)
                * self.reward_weights["phase_transition"]
            )

        self.reward_history.append(total_reward)
        self._adapt_reward_weights(reward_breakdown, s_game_info)
        return total_reward

    def _adapt_reward_weights(self, reward_breakdown: Dict, s_game_info: Dict):
        if len(self.reward_history) > 50:
            recent_avg = np.mean(list(self.reward_history)[-20:])
            overall_avg = np.mean(list(self.reward_history))
            if recent_avg > overall_avg * 1.1:
                for pattern, value in reward_breakdown.items():
                    if value > 0 and pattern in self.reward_weights:
                        self.reward_weights[pattern] *= 1 + self.adaptation_rate
            elif recent_avg < overall_avg * 0.9:
                for pattern in self.reward_weights:
                    self.reward_weights[pattern] *= 1 - self.adaptation_rate * 0.5
        for p in self.reward_weights:
            self.reward_weights[p] = np.clip(self.reward_weights[p], 0.1, 3.0)

    def get_adaptation_stats(self) -> Dict:
        return {
            "reward_weights": dict(self.reward_weights),
            "avg_recent_reward": (
                np.mean(list(self.reward_history)[-20:]) if self.reward_history else 0.0
            ),
            "adaptation_trends": {
                p: np.mean(list(r)) if r else 0.0
                for p, r in self.pattern_success_rates.items()
            },
        }
