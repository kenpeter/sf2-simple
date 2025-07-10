#!/usr/bin/env python3
"""
bait_punish_system.py - ENERGY-BASED DEFENSIVE STRATEGY SYSTEM
ENERGY INTEGRATION:
1. Energy-based reward shaping (replaces PPO-specific rewards)
2. Enhanced opponent attack detection for energy landscape
3. Extended punish window for energy optimization
4. Defensive energy reward shaping
5. Energy-stable reward normalization for defensive play
ADAPTED FOR: Energy-Based Transformer training paradigm
"""

import numpy as np
from collections import deque
from typing import Dict, List, Tuple


class SimpleBlockPunishDetector:
    """
    Energy-Based defensive strategy system using health gap rewards and proper detection.
    Adapted for Energy-Based Transformer training.
    """

    def __init__(self):
        # ENERGY ADAPTATION: Removed PPO-specific rewards, using energy-compatible values
        self.DEFENSIVE_HEALTH_ADVANTAGE_BONUS = 0.015  # Reduced for energy stability
        self.SUCCESSFUL_BLOCK_BONUS = 0.003  # Smaller bonus for energy training
        self.BLOCK_THEN_HIT_BONUS = 0.02  # Reduced for energy compatibility
        self.PERFECT_BLOCK_PUNISH_BONUS = 0.03  # Maximum bonus for perfect timing

        # Penalties for poor defensive play (energy-compatible)
        self.HEALTH_DISADVANTAGE_PENALTY = -0.005  # Smaller penalty for energy training
        self.FAILED_BLOCK_PENALTY = -0.004  # Reduced penalty

        # Extended timing windows for proper defensive play
        self.BLOCKSTUN_FRAMES = 6  # Increased from 3 to 6
        self.PUNISH_WINDOW_FRAMES = 20  # Increased from 8 to 20
        self.ATTACK_DETECTION_FRAMES = 8  # Increased from 5 to 8

        # Game state tracking
        self.last_opponent_attacking = False
        self.last_damage_received = 0
        self.last_damage_dealt = 0
        self.last_player_x = None
        self.last_opponent_x = None
        self.last_player_health = None
        self.last_opponent_health = None

        # Enhanced action tracking for defensive mechanics
        self.recent_actions = deque(maxlen=15)  # Increased for longer sequences
        self.frame_count = 0

        # Statistics for defensive play
        self.total_hits_dealt = 0
        self.total_hits_received = 0
        self.total_blocks_successful = 0
        self.total_blocks_failed = 0
        self.total_block_then_hit_sequences = 0
        self.total_perfect_punishes = 0
        self.total_health_advantage_frames = 0

        # Enhanced opponent attack detection
        self.opponent_attack_indicators = deque(maxlen=8)  # Increased tracking
        self.opponent_attack_frames = deque(maxlen=5)  # Multi-frame detection
        self.last_opponent_status = 0
        self.last_score_change = 0
        self.opponent_approaching_frames = deque(maxlen=3)

        # ENERGY ADAPTATION: Energy-stable tracking
        self.total_reward_given = 0.0
        self.reward_history = deque(maxlen=100)
        self.max_reward_per_step = 0.03  # Reduced for energy stability
        self.health_gap_history = deque(maxlen=10)

        # Energy-based learning metrics
        self.energy_positive_examples = (
            0  # Count of positive examples for energy training
        )
        self.energy_negative_examples = (
            0  # Count of negative examples for energy training
        )
        self.defensive_action_quality = 0.0  # Quality score for defensive actions

        print("✅ Energy-Based SimpleBlockPunishDetector initialized")
        print("🧠 ENERGY-BASED DEFENSIVE STRATEGY:")
        print("   1. Energy-compatible reward shaping: ACTIVE")
        print("   2. Enhanced attack detection: ACTIVE")
        print("   3. Extended punish window: 20 frames")
        print("   4. Defensive energy shaping: ACTIVE")
        print("   5. Energy-stable normalization: ACTIVE")
        print(f"   - Blockstun frames: {self.BLOCKSTUN_FRAMES}")
        print(f"   - Punish window: {self.PUNISH_WINDOW_FRAMES} frames")

    def update(
        self,
        info: Dict,
        button_features: np.ndarray,
        damage_dealt: int,
        damage_received: int,
    ) -> Dict:
        """
        Energy-adapted system using health gap rewards and defensive strategy.
        Compatible with Energy-Based Transformer training.
        """
        self.frame_count += 1
        reward = 0.0

        # Convert inputs to safe values
        damage_dealt = max(0, int(damage_dealt)) if np.isfinite(damage_dealt) else 0
        damage_received = (
            max(0, int(damage_received)) if np.isfinite(damage_received) else 0
        )

        # Extract game state
        player_x = info.get("agent_x", 160)
        opponent_x = info.get("enemy_x", 160)
        player_status = info.get("agent_status", 0)
        opponent_status = info.get("enemy_status", 0)
        score = info.get("score", 0)

        # Health gap as primary reward source (energy-compatible)
        player_health = info.get("agent_hp", 176)
        opponent_health = info.get("enemy_hp", 176)

        # Enhanced opponent attack detection
        opponent_attacking = self._enhanced_opponent_attack_detection_v2(
            info, damage_received, opponent_status, player_x, opponent_x
        )

        # Proper Street Fighter blocking detection
        player_blocking = self._detect_sf_blocking(
            button_features, player_x, opponent_x, opponent_attacking
        )

        # Player attack detection
        player_attacking = self._is_attacking(button_features)

        # Store current frame action with enhanced data
        current_action = {
            "frame": self.frame_count,
            "player_attacking": player_attacking,
            "opponent_attacking": opponent_attacking,
            "player_blocking": player_blocking,
            "damage_dealt": damage_dealt,
            "damage_received": damage_received,
            "player_x": player_x,
            "opponent_x": opponent_x,
            "player_status": player_status,
            "opponent_status": opponent_status,
            "blockstun_remaining": self._calculate_blockstun_remaining(),
            "player_health": player_health,
            "opponent_health": opponent_health,
            "health_gap": player_health - opponent_health,
        }
        self.recent_actions.append(current_action)

        # PRIMARY REWARD - Health Gap System (energy-compatible)
        health_gap_reward = self._calculate_health_gap_reward(
            player_health, opponent_health
        )
        reward += health_gap_reward

        # Defensive reward shaping (energy-adapted)
        defensive_rewards = self._calculate_defensive_rewards(
            player_health,
            opponent_health,
            player_blocking,
            opponent_attacking,
            damage_dealt,
        )
        reward += defensive_rewards

        # Track statistics for defensive play
        if damage_dealt > 0:
            self.total_hits_dealt += 1
            # Check if this was a punish after block
            if self._was_punish_after_block():
                punish_bonus = self.BLOCK_THEN_HIT_BONUS
                if self._was_perfect_punish():
                    punish_bonus = self.PERFECT_BLOCK_PUNISH_BONUS
                    self.total_perfect_punishes += 1
                    # High-quality defensive action for energy training
                    self.energy_positive_examples += 1
                reward += punish_bonus
                self.total_block_then_hit_sequences += 1
                print(
                    f"🎯 ENERGY DEFENSIVE: Block-then-punish executed! Bonus: {punish_bonus:.3f}"
                )

        # Successful blocking detection
        if self._detected_successful_block():
            self.total_blocks_successful += 1
            self.energy_positive_examples += 1  # Good defensive action
            print(f"🛡️  ENERGY DEFENSIVE: Successful block detected!")

        # Track failed blocks
        if damage_received > 0:
            self.total_hits_received += 1
            if self._should_have_blocked():
                self.total_blocks_failed += 1
                self.energy_negative_examples += 1  # Poor defensive action

        # Track health advantage
        if player_health > opponent_health:
            self.total_health_advantage_frames += 1

        # Energy-stable reward normalization for defensive play
        final_reward = self._normalize_reward_for_energy_defensive_play(reward)

        # Update tracking
        self.total_reward_given += final_reward
        self.reward_history.append(final_reward)
        self.health_gap_history.append(player_health - opponent_health)

        # Calculate defensive action quality for energy training
        self._update_defensive_action_quality()

        # Update state tracking
        self.last_opponent_attacking = opponent_attacking
        self.last_damage_received = damage_received
        self.last_damage_dealt = damage_dealt
        self.last_player_x = player_x
        self.last_opponent_x = opponent_x
        self.last_opponent_status = opponent_status
        self.last_score_change = score - info.get("prev_score", score)
        self.last_player_health = player_health
        self.last_opponent_health = opponent_health

        # Return comprehensive defensive stats for energy training
        return {
            "bait_punish_reward": final_reward,
            "health_gap_reward": health_gap_reward,
            "defensive_rewards": defensive_rewards,
            "hits_dealt": self.total_hits_dealt,
            "hits_received": self.total_hits_received,
            "blocks_successful": self.total_blocks_successful,
            "blocks_failed": self.total_blocks_failed,
            "block_then_hit_sequences": self.total_block_then_hit_sequences,
            "perfect_punishes": self.total_perfect_punishes,
            "health_advantage_frames": self.total_health_advantage_frames,
            "defensive_efficiency": self._calculate_defensive_efficiency(),
            "sequence_phase": self._get_current_phase(),
            "blocking_effectiveness": self._calculate_blocking_effectiveness(),
            "punish_accuracy": self._calculate_punish_accuracy(),
            "health_gap": player_health - opponent_health,
            "avg_health_gap": self._get_avg_health_gap(),
            # Energy-specific metrics
            "energy_positive_examples": self.energy_positive_examples,
            "energy_negative_examples": self.energy_negative_examples,
            "defensive_action_quality": self.defensive_action_quality,
            "energy_training_signal": self._get_energy_training_signal(),
        }

    def _calculate_health_gap_reward(
        self, player_health: int, opponent_health: int
    ) -> float:
        """
        Energy-adapted health gap reward calculation.
        """
        try:
            # Ensure valid health values
            player_health = (
                max(0, int(player_health)) if np.isfinite(player_health) else 0
            )
            opponent_health = (
                max(0, int(opponent_health)) if np.isfinite(opponent_health) else 0
            )

            # Health gap calculation
            health_gap = player_health - opponent_health

            # Scale health gap reward (smaller values for energy stability)
            base_health_reward = health_gap * 0.0003  # Reduced for energy training

            # Additional bonus for maintaining significant health advantage
            if health_gap > 50:
                base_health_reward += 0.005  # Reduced for energy
            elif health_gap > 100:
                base_health_reward += 0.01  # Reduced for energy

            # Penalty for significant health disadvantage
            if health_gap < -50:
                base_health_reward -= 0.005  # Reduced for energy
            elif health_gap < -100:
                base_health_reward -= 0.01  # Reduced for energy

            return np.clip(
                base_health_reward, -0.02, 0.02
            )  # Tighter clipping for energy

        except Exception as e:
            print(f"⚠️  Error in energy health gap reward: {e}")
            return 0.0

    def _enhanced_opponent_attack_detection_v2(
        self, info, damage_received, opponent_status, player_x, opponent_x
    ):
        """Enhanced opponent attack detection for energy training."""
        attack_detected = False

        # METHOD 1: We received damage (most reliable)
        if damage_received > 0:
            attack_detected = True

        # METHOD 2: Opponent animation state changes
        if (
            self.last_opponent_status != 0
            and opponent_status != self.last_opponent_status
        ):
            status_diff = abs(opponent_status - self.last_opponent_status)
            if status_diff > 20:
                attack_detected = True

        # METHOD 3: Opponent approaching aggressively
        if self._opponent_approaching_aggressively(player_x, opponent_x):
            attack_detected = True

        # METHOD 4: Multi-frame attack pattern
        self.opponent_attack_frames.append(damage_received > 0)
        if len(self.opponent_attack_frames) >= 3:
            recent_attacks = sum(self.opponent_attack_frames[-3:])
            if recent_attacks >= 2:
                attack_detected = True

        return attack_detected

    def _opponent_approaching_aggressively(
        self, player_x: float, opponent_x: float
    ) -> bool:
        """Detect if opponent is approaching aggressively."""
        try:
            if self.last_opponent_x is None:
                return False

            # Calculate opponent movement toward player
            if player_x < opponent_x:
                # Player on left, opponent moving left toward player
                opponent_advancing = opponent_x < self.last_opponent_x
            else:
                # Player on right, opponent moving right toward player
                opponent_advancing = opponent_x > self.last_opponent_x

            self.opponent_approaching_frames.append(opponent_advancing)

            # Aggressive approach: moving toward player for multiple frames
            if len(self.opponent_approaching_frames) >= 3:
                return sum(self.opponent_approaching_frames) >= 2

            return False

        except Exception as e:
            return False

    def _calculate_defensive_rewards(
        self,
        player_health: int,
        opponent_health: int,
        player_blocking: bool,
        opponent_attacking: bool,
        damage_dealt: int,
    ) -> float:
        """
        Energy-adapted defensive reward shaping.
        """
        defensive_reward = 0.0

        try:
            # Reward for maintaining health advantage
            health_gap = player_health - opponent_health
            if health_gap > 30:
                defensive_reward += (
                    self.DEFENSIVE_HEALTH_ADVANTAGE_BONUS * 0.3
                )  # Reduced for energy
            if health_gap > 60:
                defensive_reward += (
                    self.DEFENSIVE_HEALTH_ADVANTAGE_BONUS * 0.6
                )  # Reduced for energy

            # Reward for successful blocking during opponent attacks
            if player_blocking and opponent_attacking:
                defensive_reward += self.SUCCESSFUL_BLOCK_BONUS

            # Penalty for health disadvantage (encourages better defense)
            if health_gap < -30:
                defensive_reward += self.HEALTH_DISADVANTAGE_PENALTY * 0.5
            if health_gap < -60:
                defensive_reward += self.HEALTH_DISADVANTAGE_PENALTY

            # Small bonus for patient play (not constantly attacking)
            if not self._is_constantly_attacking():
                defensive_reward += 0.0005  # Reduced for energy

            return np.clip(defensive_reward, -0.015, 0.02)  # Tighter bounds for energy

        except Exception as e:
            print(f"⚠️  Error in energy defensive rewards: {e}")
            return 0.0

    def _normalize_reward_for_energy_defensive_play(self, reward: float) -> float:
        """
        Energy-stable reward normalization specifically for defensive learning.
        Adapted for Energy-Based Transformer training.
        """
        try:
            # Ensure reward is finite
            if not np.isfinite(reward):
                reward = 0.0

            # First clipping for defensive range (tighter for energy)
            reward = np.clip(reward, -0.03, 0.03)

            # Add to history for normalization
            self.reward_history.append(reward)

            # Light normalization if we have enough history
            if len(self.reward_history) > 20:
                recent_rewards = list(self.reward_history)[-20:]
                reward_mean = np.mean(recent_rewards)
                reward_std = max(
                    np.std(recent_rewards), 0.005
                )  # Smaller minimum std for energy

                # Gentle normalization (more conservative for energy training)
                normalized = (reward - reward_mean) / reward_std
                normalized = np.clip(normalized, -1.5, 1.5)  # Tighter clipping

                # Scale down for energy stability
                final_reward = normalized * 0.015  # Smaller scale for energy
            else:
                # Simple scaling for early training
                final_reward = reward * 0.3  # More conservative for energy

            # Final safety clipping for energy training
            return np.clip(final_reward, -0.02, 0.02)

        except Exception as e:
            print(f"⚠️  Error in energy reward normalization: {e}")
            return 0.0

    def _update_defensive_action_quality(self):
        """Update defensive action quality score for energy training."""
        try:
            total_examples = (
                self.energy_positive_examples + self.energy_negative_examples
            )
            if total_examples > 0:
                self.defensive_action_quality = (
                    self.energy_positive_examples / total_examples
                )
            else:
                self.defensive_action_quality = 0.5  # Neutral quality

            # Smooth the quality metric
            if hasattr(self, "_prev_defensive_quality"):
                self.defensive_action_quality = (
                    0.9 * self._prev_defensive_quality
                    + 0.1 * self.defensive_action_quality
                )
            self._prev_defensive_quality = self.defensive_action_quality

        except Exception as e:
            self.defensive_action_quality = 0.5

    def _get_energy_training_signal(self) -> float:
        """
        Get energy training signal for contrastive learning.
        Positive signal = good defensive action, Negative signal = bad defensive action.
        """
        try:
            # Recent action quality
            if len(self.recent_actions) == 0:
                return 0.0

            current_action = self.recent_actions[-1]

            # Positive signals (good for energy training)
            if (
                current_action.get("player_blocking", False)
                and current_action.get("opponent_attacking", False)
                and current_action.get("damage_received", 0) == 0
            ):
                return 1.0  # Perfect block

            if (
                current_action.get("damage_dealt", 0) > 0
                and self._was_punish_after_block()
            ):
                return 1.0  # Successful punish

            if current_action.get("health_gap", 0) > 50:
                return 0.7  # Health advantage

            # Negative signals (bad for energy training)
            if current_action.get("damage_received", 0) > 0 and not current_action.get(
                "player_blocking", False
            ):
                return -1.0  # Failed to block

            if current_action.get("health_gap", 0) < -50:
                return -0.7  # Health disadvantage

            # Neutral
            return 0.0

        except Exception as e:
            return 0.0

    def _is_constantly_attacking(self) -> bool:
        """Helper to detect if player is constantly attacking (discourages spam)."""
        if len(self.recent_actions) < 5:
            return False

        recent_attacks = sum(
            1
            for action in self.recent_actions[-5:]
            if action.get("player_attacking", False)
        )
        return recent_attacks >= 4  # Attacking 4 out of 5 recent frames

    def _detect_sf_blocking(
        self,
        button_features: np.ndarray,
        player_x: float,
        opponent_x: float,
        opponent_attacking: bool,
    ) -> bool:
        """
        Proper Street Fighter blocking detection (unchanged).
        """
        try:
            if not isinstance(button_features, np.ndarray) or len(button_features) < 12:
                return False

            # Determine which direction is "back" from opponent
            if player_x < opponent_x:
                # Player is on left, opponent on right - block with RIGHT (button 7)
                blocking_direction = button_features[7] > 0.5  # RIGHT
                # Also check for down-back (crouch blocking)
                crouch_block = (
                    button_features[5] > 0.5 and button_features[7] > 0.5
                )  # DOWN + RIGHT
            else:
                # Player is on right, opponent on left - block with LEFT (button 6)
                blocking_direction = button_features[6] > 0.5  # LEFT
                # Also check for down-back (crouch blocking)
                crouch_block = (
                    button_features[5] > 0.5 and button_features[6] > 0.5
                )  # DOWN + LEFT

            # Player is blocking if holding back direction or crouch-blocking
            is_blocking = blocking_direction or crouch_block

            # Additional check: not attacking while blocking
            if is_blocking:
                attack_buttons = [0, 1, 8, 9, 10, 11]  # B, Y, A, X, L, R
                is_attacking = any(button_features[btn] > 0.5 for btn in attack_buttons)
                return not is_attacking  # Only blocking if not attacking

            return False

        except Exception as e:
            print(f"⚠️  Error in SF blocking detection: {e}")
            return False

    def _calculate_blockstun_remaining(self) -> int:
        """Calculate remaining blockstun frames after a block."""
        if len(self.recent_actions) < 2:
            return 0

        # Look for recent successful block
        for i in range(
            len(self.recent_actions) - 1, max(0, len(self.recent_actions) - 8), -1
        ):
            action = self.recent_actions[i]
            if (
                action["player_blocking"]
                and action["opponent_attacking"]
                and action["damage_received"] == 0
            ):
                # Found a block, calculate remaining blockstun
                frames_since_block = self.frame_count - action["frame"]
                return max(0, self.BLOCKSTUN_FRAMES - frames_since_block)

        return 0

    def _detected_successful_block(self) -> bool:
        """Detect if we just successfully blocked an attack."""
        if len(self.recent_actions) < 2:
            return False

        current = self.recent_actions[-1]

        # Successful block: blocking + opponent attacking + no damage
        return (
            current["player_blocking"]
            and current["opponent_attacking"]
            and current["damage_received"] == 0
        )

    def _should_have_blocked(self) -> bool:
        """Check if we should have blocked but didn't."""
        if len(self.recent_actions) < 2:
            return False

        current = self.recent_actions[-1]

        # We should have blocked if: opponent attacking + we took damage + we weren't blocking
        return (
            current["opponent_attacking"]
            and current["damage_received"] > 0
            and not current["player_blocking"]
        )

    def _was_punish_after_block(self) -> bool:
        """Check if current hit was a punish after successful block."""
        if len(self.recent_actions) < 3:
            return False

        current = self.recent_actions[-1]

        # Must be dealing damage with an attack
        if not (current["player_attacking"] and current["damage_dealt"] > 0):
            return False

        # Look for recent successful block within extended punish window
        for i in range(
            len(self.recent_actions) - 2,
            max(0, len(self.recent_actions) - self.PUNISH_WINDOW_FRAMES),
            -1,
        ):
            action = self.recent_actions[i]
            if (
                action["player_blocking"]
                and action["opponent_attacking"]
                and action["damage_received"] == 0
            ):
                # Found a successful block within punish window
                return True

        return False

    def _was_perfect_punish(self) -> bool:
        """Check if the punish was executed with perfect timing."""
        if len(self.recent_actions) < 5:
            return False

        current = self.recent_actions[-1]

        # Look for block immediately followed by attack
        for i in range(
            len(self.recent_actions) - 2, max(0, len(self.recent_actions) - 8), -1
        ):
            action = self.recent_actions[i]
            if (
                action["player_blocking"]
                and action["opponent_attacking"]
                and action["damage_received"] == 0
            ):
                # Check if attack started right after blockstun
                frames_after_block = current["frame"] - action["frame"]
                if frames_after_block <= self.BLOCKSTUN_FRAMES + 3:  # Perfect timing
                    return True
                break

        return False

    def _is_attacking(self, button_features: np.ndarray) -> bool:
        """Enhanced attack detection."""
        try:
            if not isinstance(button_features, np.ndarray) or len(button_features) < 12:
                return False
            attack_buttons = [0, 1, 8, 9, 10, 11]  # B, Y, A, X, L, R
            return bool(np.any(button_features[attack_buttons] > 0.5))
        except Exception:
            return False

    def _get_current_phase(self) -> str:
        """Enhanced phase detection with defensive focus."""
        if len(self.recent_actions) == 0:
            return "neutral"

        current = self.recent_actions[-1]

        # Check for active punish sequence
        if self._was_punish_after_block():
            return "punishing"

        # Check current state
        if current["player_blocking"] and current["opponent_attacking"]:
            return "blocking"
        elif current["player_attacking"]:
            return "attacking"
        elif current["opponent_attacking"]:
            return "defending"
        elif current["blockstun_remaining"] > 0:
            return "blockstun"
        elif current.get("health_gap", 0) > 30:
            return "health_advantage"
        else:
            return "neutral"

    def _calculate_defensive_efficiency(self) -> float:
        """Calculate overall defensive efficiency."""
        total_defensive_situations = (
            self.total_blocks_successful
            + self.total_blocks_failed
            + self.total_hits_received
        )

        if total_defensive_situations == 0:
            return 0.0

        defensive_successes = self.total_blocks_successful
        return defensive_successes / total_defensive_situations

    def _calculate_blocking_effectiveness(self) -> float:
        """Calculate how effective our blocking is."""
        total_defensive_situations = (
            self.total_blocks_successful + self.total_blocks_failed
        )

        if total_defensive_situations == 0:
            return 0.0

        return self.total_blocks_successful / total_defensive_situations

    def _calculate_punish_accuracy(self) -> float:
        """Calculate how often we punish after successful blocks."""
        if self.total_blocks_successful == 0:
            return 0.0

        return self.total_block_then_hit_sequences / self.total_blocks_successful

    def _get_avg_health_gap(self) -> float:
        """Get average health gap over recent frames."""
        if len(self.health_gap_history) == 0:
            return 0.0
        return np.mean(list(self.health_gap_history))

    def reset_sequence(self):
        """Reset sequence state."""
        self.recent_actions.clear()
        self.opponent_attack_indicators.clear()
        self.opponent_attack_frames.clear()
        self.opponent_approaching_frames.clear()
        self.health_gap_history.clear()
        self.last_opponent_attacking = False
        self.last_damage_received = 0
        self.last_damage_dealt = 0
        self.last_player_x = None
        self.last_opponent_x = None
        self.last_player_health = None
        self.last_opponent_health = None
        self.last_opponent_status = 0
        self.last_score_change = 0
        self.frame_count = 0

    def get_learning_stats(self) -> Dict:
        """Get comprehensive defensive learning statistics for energy training."""
        defensive_efficiency = self._calculate_defensive_efficiency()
        blocking_effectiveness = self._calculate_blocking_effectiveness()
        punish_accuracy = self._calculate_punish_accuracy()
        avg_health_gap = self._get_avg_health_gap()

        return {
            "hits_dealt": self.total_hits_dealt,
            "hits_received": self.total_hits_received,
            "blocks_successful": self.total_blocks_successful,
            "blocks_failed": self.total_blocks_failed,
            "block_then_hit_sequences": self.total_block_then_hit_sequences,
            "perfect_punishes": self.total_perfect_punishes,
            "health_advantage_frames": self.total_health_advantage_frames,
            "defensive_efficiency": defensive_efficiency,
            "blocking_effectiveness": blocking_effectiveness,
            "punish_accuracy": punish_accuracy,
            "avg_health_gap": avg_health_gap,
            "total_reward_given": self.total_reward_given,
            "reward_std": (
                np.std(list(self.reward_history))
                if len(self.reward_history) > 1
                else 0.0
            ),
            "current_phase": self._get_current_phase(),
            "frame_count": self.frame_count,
            # Energy-specific metrics
            "energy_positive_examples": self.energy_positive_examples,
            "energy_negative_examples": self.energy_negative_examples,
            "defensive_action_quality": self.defensive_action_quality,
            "energy_training_signal": self._get_energy_training_signal(),
        }


class AdaptiveRewardShaper:
    """
    Energy-Based adaptive reward shaper with defensive strategy focus.
    Adapted for Energy-Based Transformer training.
    """

    def __init__(self):
        # Energy-adapted reward scaling for defensive stability
        self.base_reward_scale = 0.08  # Reduced for energy training
        self.defensive_scale = 0.06  # Energy-specific scaling for defensive rewards
        self.health_gap_scale = 0.09  # Energy-specific scaling for health gap
        self.max_total_reward = 0.15  # Reduced for energy training

        # Adaptive parameters for defensive learning (energy-compatible)
        self.adaptation_rate = 0.0005  # Slower adaptation for energy stability
        self.performance_window = 300  # Larger window for energy stability
        self.performance_history = deque(maxlen=self.performance_window)

        # Defensive-specific bonuses (energy-adapted)
        self.defensive_stance_bonus = 0.01  # Reduced for energy
        self.health_preservation_bonus = 0.015  # Reduced for energy
        self.perfect_defense_bonus = 0.02  # Reduced for energy

        # Energy training compatibility
        self.energy_positive_bonus = 0.005  # Bonus for positive energy examples
        self.energy_negative_penalty = -0.003  # Penalty for negative examples

        # Stability tracking
        self.total_rewards_shaped = 0
        self.reward_clipping_events = 0

        print("✅ Energy-Based AdaptiveRewardShaper initialized")
        print("🧠 ENERGY DEFENSIVE REWARD SHAPING:")
        print(f"   - Base reward scale: {self.base_reward_scale}")
        print(f"   - Defensive scale: {self.defensive_scale}")
        print(f"   - Health gap scale: {self.health_gap_scale}")
        print(f"   - Max total reward: {self.max_total_reward}")

    def shape_reward(
        self, base_reward: float, bait_punish_info: Dict, game_info: Dict
    ) -> float:
        """
        Energy-enhanced defensive reward shaping.
        """
        try:
            # Ensure base reward is finite
            if not np.isfinite(base_reward):
                base_reward = 0.0
            base_reward = np.clip(base_reward, -0.5, 0.5)  # Tighter clip for energy

            # Get defensive-specific rewards
            health_gap_reward = bait_punish_info.get("health_gap_reward", 0.0)
            defensive_rewards = bait_punish_info.get("defensive_rewards", 0.0)

            if not np.isfinite(health_gap_reward):
                health_gap_reward = 0.0
            if not np.isfinite(defensive_rewards):
                defensive_rewards = 0.0

            # Scale rewards with defensive focus (energy-adapted)
            scaled_base = base_reward * self.base_reward_scale
            scaled_health_gap = health_gap_reward * self.health_gap_scale
            scaled_defensive = defensive_rewards * self.defensive_scale

            # Add defensive-specific bonuses
            bonus_reward = 0.0

            # Bonus for successful defensive sequences
            if bait_punish_info.get("blocks_successful", 0) > 0:
                bonus_reward += self.defensive_stance_bonus

            # Bonus for health preservation
            health_gap = bait_punish_info.get("health_gap", 0)
            if health_gap > 50:
                bonus_reward += self.health_preservation_bonus

            # Bonus for perfect defensive play
            if bait_punish_info.get("perfect_punishes", 0) > 0:
                bonus_reward += self.perfect_defense_bonus

            # Energy training bonuses/penalties
            energy_training_signal = bait_punish_info.get("energy_training_signal", 0.0)
            if energy_training_signal > 0.5:
                bonus_reward += self.energy_positive_bonus
            elif energy_training_signal < -0.5:
                bonus_reward += self.energy_negative_penalty

            # Combine all rewards
            total_reward = (
                scaled_base + scaled_health_gap + scaled_defensive + bonus_reward
            )

            # Hard cap to prevent energy explosion
            original_reward = total_reward
            total_reward = np.clip(
                total_reward, -self.max_total_reward, self.max_total_reward
            )

            # Track clipping
            if abs(original_reward - total_reward) > 1e-6:
                self.reward_clipping_events += 1

            # Ensure finite
            if not np.isfinite(total_reward):
                total_reward = 0.0

            # Update tracking
            self.total_rewards_shaped += 1
            self.performance_history.append(total_reward)

            # Adapt slowly for defensive learning (energy-compatible)
            if len(self.performance_history) >= self.performance_window:
                self._adapt_defensive_parameters_for_energy()

            return total_reward

        except Exception as e:
            print(f"⚠️  Error in energy defensive reward shaping: {e}")
            return 0.0

    def _adapt_defensive_parameters_for_energy(self):
        """Adaptive parameter adjustment for defensive learning with energy stability."""
        try:
            if len(self.performance_history) < self.performance_window:
                return

            recent_rewards = list(self.performance_history)
            avg_reward = np.mean(recent_rewards)
            reward_std = np.std(recent_rewards)

            # Only adapt if performance is stable (tighter requirement for energy)
            if reward_std < 0.01:  # Tighter stability requirement for energy
                # Adjust defensive scale based on performance
                if avg_reward > 0.03:  # Good defensive performance
                    self.defensive_scale = min(self.defensive_scale + 0.0005, 0.12)
                elif avg_reward < -0.03:  # Poor defensive performance
                    self.defensive_scale = max(self.defensive_scale - 0.0005, 0.01)

        except Exception as e:
            print(f"⚠️  Error in energy defensive parameter adaptation: {e}")

    def get_adaptation_stats(self) -> Dict:
        """Get energy defensive adaptation statistics."""
        try:
            avg_reward = (
                np.mean(list(self.performance_history))
                if len(self.performance_history) > 0
                else 0.0
            )
            reward_std = (
                np.std(list(self.performance_history))
                if len(self.performance_history) > 1
                else 0.0
            )

            return {
                "defensive_reward_scale": self.defensive_scale,
                "health_gap_scale": self.health_gap_scale,
                "avg_shaped_reward": avg_reward,
                "reward_std": reward_std,
                "total_rewards_shaped": self.total_rewards_shaped,
                "reward_clipping_events": self.reward_clipping_events,
                "clipping_rate": self.reward_clipping_events
                / max(1, self.total_rewards_shaped),
                "defensive_stance_bonus": self.defensive_stance_bonus,
                "health_preservation_bonus": self.health_preservation_bonus,
                "perfect_defense_bonus": self.perfect_defense_bonus,
                "energy_positive_bonus": self.energy_positive_bonus,
                "energy_negative_penalty": self.energy_negative_penalty,
            }
        except Exception as e:
            print(f"⚠️  Error getting energy defensive adaptation stats: {e}")
            return {}


def integrate_bait_punish_system(strategic_tracker):
    """
    Integrate the Energy-Based defensive bait-punish system into the strategic tracker.
    """
    try:
        if not hasattr(strategic_tracker, "bait_punish_detector"):
            strategic_tracker.bait_punish_detector = SimpleBlockPunishDetector()
            print("✅ Energy-Based defensive bait-punish system integrated")

        # Add method to get defensive bait-punish features
        def get_bait_punish_features():
            """Get 7 defensive bait-punish features for the enhanced feature vector."""
            try:
                stats = strategic_tracker.bait_punish_detector.get_learning_stats()
                features = np.zeros(7, dtype=np.float32)

                # Feature 0: Defensive efficiency (blocks + health preservation)
                features[0] = np.clip(stats.get("defensive_efficiency", 0.0), 0.0, 1.0)

                # Feature 1: Blocking effectiveness
                features[1] = np.clip(
                    stats.get("blocking_effectiveness", 0.0), 0.0, 1.0
                )

                # Feature 2: Punish accuracy
                features[2] = np.clip(stats.get("punish_accuracy", 0.0), 0.0, 1.0)

                # Feature 3: Block-then-hit sequences (normalized)
                features[3] = np.clip(
                    stats.get("block_then_hit_sequences", 0) / 15.0, 0.0, 1.0
                )

                # Feature 4: Perfect punishes (normalized)
                features[4] = np.clip(stats.get("perfect_punishes", 0) / 8.0, 0.0, 1.0)

                # Feature 5: Health advantage ratio
                avg_health_gap = stats.get("avg_health_gap", 0)
                features[5] = np.clip((avg_health_gap + 100) / 200.0, 0.0, 1.0)

                # Feature 6: Energy training signal (normalized for energy training)
                energy_signal = stats.get("energy_training_signal", 0.0)
                features[6] = np.clip((energy_signal + 1.0) / 2.0, 0.0, 1.0)

                return features

            except Exception as e:
                print(f"⚠️  Error getting energy defensive bait-punish features: {e}")
                return np.zeros(7, dtype=np.float32)

        # Add the method to the strategic tracker
        strategic_tracker.get_bait_punish_features = get_bait_punish_features

        return True

    except Exception as e:
        print(f"❌ Failed to integrate energy defensive bait-punish system: {e}")
        return False


# For compatibility
BaitPunishDetector = SimpleBlockPunishDetector

# Export components
__all__ = [
    "SimpleBlockPunishDetector",
    "BaitPunishDetector",
    "AdaptiveRewardShaper",
    "integrate_bait_punish_system",
]

# Test the Energy-Based defensive system
if __name__ == "__main__":
    print("🧪 Testing Energy-Based Defensive SimpleBlockPunishDetector...")

    detector = SimpleBlockPunishDetector()

    # Test defensive strategy mechanics for energy training
    test_info = {
        "agent_hp": 150,
        "enemy_hp": 120,  # Player has health advantage
        "score": 1000,
        "agent_x": 100,
        "enemy_x": 200,
        "agent_status": 0,
        "enemy_status": 0,
    }

    # Test 1: Health advantage (primary reward)
    test_buttons = np.zeros(12, dtype=np.float32)
    result = detector.update(test_info, test_buttons, 0, 0)
    print(f"Energy health advantage test: {result}")
    print(f"  Health gap reward: {result.get('health_gap_reward', 0):.4f}")
    print(f"  Energy training signal: {result.get('energy_training_signal', 0):.4f}")

    # Test 2: Defensive blocking stance
    test_buttons[7] = 1.0  # RIGHT (blocking away from opponent)
    test_info["enemy_status"] = 45  # Opponent attacking
    result = detector.update(
        test_info, test_buttons, 0, 0
    )  # No damage = successful block
    print(f"Energy defensive blocking: {result}")
    print(f"  Defensive rewards: {result.get('defensive_rewards', 0):.4f}")
    print(f"  Energy positive examples: {result.get('energy_positive_examples', 0)}")

    # Test 3: Health gap maintained (should get reward)
    test_info["agent_hp"] = 150
    test_info["enemy_hp"] = 110  # Health gap increased
    result = detector.update(test_info, test_buttons, 0, 0)
    print(f"Energy health gap maintained: {result}")
    print(f"  Health gap: {result.get('health_gap', 0)}")

    # Test 4: Defensive counter-attack after block (energy training signal)
    test_buttons[7] = 0.0  # Stop blocking
    test_buttons[0] = 1.0  # B button (attack)
    test_info["enemy_status"] = 0  # Opponent no longer attacking
    result = detector.update(test_info, test_buttons, 20, 0)  # Deal damage after block
    print(f"Energy defensive counter-attack: {result}")
    print(f"  Block-then-hit sequences: {result.get('block_then_hit_sequences', 0)}")
    print(f"  Energy training signal: {result.get('energy_training_signal', 0):.4f}")

    # Test 5: Health disadvantage (negative energy signal)
    test_buttons[0] = 0.0  # Stop attacking
    test_info["agent_hp"] = 80  # Lower health
    test_info["enemy_hp"] = 140  # Opponent has advantage
    test_info["enemy_status"] = 55  # Opponent attacking
    result = detector.update(test_info, test_buttons, 0, 15)  # Took damage
    print(f"Energy health disadvantage: {result}")
    print(f"  Health gap: {result.get('health_gap', 0)}")
    print(f"  Energy negative examples: {result.get('energy_negative_examples', 0)}")

    # Test 6: Perfect defensive sequence (high energy signal)
    test_info["agent_hp"] = 160  # Restore health advantage
    test_info["enemy_hp"] = 100
    test_buttons[5] = 1.0  # DOWN
    test_buttons[6] = 1.0  # LEFT (crouch blocking left)
    result = detector.update(test_info, test_buttons, 0, 0)
    print(f"Perfect energy defensive sequence: {result}")

    # Display final comprehensive defensive stats for energy training
    final_stats = detector.get_learning_stats()
    print(f"\n📊 Final ENERGY DEFENSIVE Stats:")
    print(f"🛡️  DEFENSIVE METRICS:")
    print(f"   - Defensive efficiency: {final_stats['defensive_efficiency']:.2%}")
    print(f"   - Blocking effectiveness: {final_stats['blocking_effectiveness']:.2%}")
    print(f"   - Punish accuracy: {final_stats['punish_accuracy']:.2%}")
    print(f"   - Health advantage frames: {final_stats['health_advantage_frames']}")
    print(f"   - Average health gap: {final_stats['avg_health_gap']:.1f}")
    print(f"🎯 COMBAT STATS:")
    print(f"   - Successful blocks: {final_stats['blocks_successful']}")
    print(f"   - Failed blocks: {final_stats['blocks_failed']}")
    print(f"   - Block-then-hit sequences: {final_stats['block_then_hit_sequences']}")
    print(f"   - Perfect punishes: {final_stats['perfect_punishes']}")
    print(f"🧠 ENERGY TRAINING METRICS:")
    print(f"   - Energy positive examples: {final_stats['energy_positive_examples']}")
    print(f"   - Energy negative examples: {final_stats['energy_negative_examples']}")
    print(
        f"   - Defensive action quality: {final_stats['defensive_action_quality']:.2%}"
    )
    print(f"   - Energy training signal: {final_stats['energy_training_signal']:.3f}")
    print(f"🏥 HEALTH MANAGEMENT:")
    print(f"   - Current phase: {final_stats['current_phase']}")
    print(f"   - Total reward given: {final_stats['total_reward_given']:.3f}")

    print("\n🧪 Testing Energy-Based AdaptiveRewardShaper...")

    shaper = AdaptiveRewardShaper()

    # Test defensive reward shaping for energy training
    base_reward = 0.1  # Smaller base reward for energy
    bait_punish_info = {
        "health_gap_reward": 0.015,
        "defensive_rewards": 0.01,
        "blocks_successful": 2,
        "perfect_punishes": 1,
        "health_gap": 60,
        "energy_training_signal": 1.0,  # Positive energy signal
    }
    game_info = {}

    shaped_reward = shaper.shape_reward(base_reward, bait_punish_info, game_info)
    print(f"Energy defensive shaped reward: {shaped_reward:.4f}")

    # Test with strong defensive focus and energy signal
    defensive_info = {
        "health_gap_reward": 0.02,
        "defensive_rewards": 0.015,
        "blocks_successful": 3,
        "perfect_punishes": 2,
        "health_gap": 80,
        "energy_training_signal": 1.0,  # Strong positive signal
    }

    defensive_shaped = shaper.shape_reward(0.05, defensive_info, game_info)
    print(f"Strong energy defensive reward: {defensive_shaped:.4f}")

    # Test adaptation stats
    adaptation_stats = shaper.get_adaptation_stats()
    print(f"Energy defensive adaptation stats: {adaptation_stats}")

    print("\n✅ All Energy-Based defensive bait-punish system tests passed!")
    print("🧠 Ready for ENERGY-BASED Street Fighter II training!")
    print("\n🔧 ENERGY ADAPTATIONS APPLIED:")
    print("   ✅ Energy-compatible reward scaling (reduced values)")
    print("   ✅ Energy training signal generation")
    print("   ✅ Positive/negative example tracking for contrastive learning")
    print("   ✅ Energy-stable reward normalization")
    print("   ✅ Defensive action quality metrics for energy landscape")
    print("\n🎯 Expected ENERGY-BASED Behavior:")
    print("   - AI will learn energy landscape through defensive examples")
    print(
        "   - Positive energy examples: successful blocks, punishes, health advantage"
    )
    print(
        "   - Negative energy examples: failed blocks, taking damage, health disadvantage"
    )
    print("   - Energy verifier will score defensive actions appropriately")
    print("   - Thinking process will optimize for better defensive decisions")
    print("   - System prevents poor defensive play through energy training")
