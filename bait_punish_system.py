#!/usr/bin/env python3
"""
bait_punish_system.py - STABLE BLOCK-PUNISH SYSTEM
Research-proven approach for fighting game AI with PPO stability focus:
- Small, fixed reward values (prevent value explosion)
- Simple sequence detection
- No complex normalization
- Immediate rewards only
"""

import numpy as np
from collections import deque
from typing import Dict, List, Tuple


class SimpleBlockPunishDetector:
    """
    Research-proven simple approach with TRUE block-then-hit sequence detection:
    - Hit opponent = small positive reward
    - Block opponent attack = small positive reward
    - Get hit = small negative reward
    - BLOCK THEN HIT SEQUENCE = bonus reward

    CRITICAL: All rewards are small and bounded to prevent PPO value loss explosion.
    """

    def __init__(self):
        # CRITICAL: Small, fixed rewards to prevent PPO value explosion
        self.HIT_REWARD = 0.01  # Small positive when hitting opponent
        self.BLOCK_REWARD = 0.005  # Small positive when blocking
        self.HIT_PENALTY = -0.01  # Small negative when taking damage
        self.BLOCK_FAIL_PENALTY = -0.005  # Small negative when failing to block

        # NEW: Block-then-hit sequence reward (slightly larger but still small)
        self.BLOCK_THEN_HIT_BONUS = 0.02  # Bonus for successful block->hit sequence

        # Simple state tracking (no complex history)
        self.last_opponent_attacking = False
        self.last_damage_received = 0
        self.last_damage_dealt = 0

        # NEW: Track recent actions for sequence detection (limited history)
        self.recent_actions = deque(maxlen=5)  # Only last 5 frames

        # Simple statistics (no running normalization that could cause instability)
        self.total_hits_dealt = 0
        self.total_hits_received = 0
        self.total_blocks_successful = 0
        self.total_blocks_failed = 0
        self.total_block_then_hit_sequences = 0  # NEW: Track sequences

        # PPO stability tracking
        self.total_reward_given = 0.0
        self.reward_history = deque(maxlen=100)
        self.max_reward_per_step = 0.05  # Cap total reward per step

        print("✅ SimpleBlockPunishDetector initialized with PPO stability focus")
        print(f"   - Hit reward: {self.HIT_REWARD}")
        print(f"   - Block reward: {self.BLOCK_REWARD}")
        print(f"   - Block-then-hit bonus: {self.BLOCK_THEN_HIT_BONUS}")
        print(f"   - Max reward per step: {self.max_reward_per_step}")

    def update(
        self,
        info: Dict,
        button_features: np.ndarray,
        damage_dealt: int,
        damage_received: int,
    ) -> Dict:
        """
        Simple update with immediate rewards + block-then-hit sequence detection.
        CRITICAL: All rewards are small and capped to prevent PPO value explosion.
        """

        reward = 0.0

        # Convert inputs to simple values
        damage_dealt = max(0, int(damage_dealt)) if np.isfinite(damage_dealt) else 0
        damage_received = (
            max(0, int(damage_received)) if np.isfinite(damage_received) else 0
        )

        # Simple checks
        player_attacking = self._is_attacking(button_features)
        opponent_attacking = self._infer_opponent_attacking(info, damage_received)
        player_blocking = self._is_blocking(button_features)

        # Store current frame action
        current_action = {
            "player_attacking": player_attacking,
            "opponent_attacking": opponent_attacking,
            "player_blocking": player_blocking,
            "damage_dealt": damage_dealt,
            "damage_received": damage_received,
            "frame": len(self.recent_actions),
        }
        self.recent_actions.append(current_action)

        # REWARD 1: Hit opponent (immediate reward)
        if damage_dealt > 0:
            reward += self.HIT_REWARD
            self.total_hits_dealt += 1

        # REWARD 2: Successfully blocked (immediate reward)
        if self.last_opponent_attacking and player_blocking and damage_received == 0:
            reward += self.BLOCK_REWARD
            self.total_blocks_successful += 1

        # PENALTY 1: Got hit (immediate penalty)
        if damage_received > 0:
            reward += self.HIT_PENALTY
            self.total_hits_received += 1

        # PENALTY 2: Failed to block (immediate penalty)
        if self.last_opponent_attacking and not player_blocking and damage_received > 0:
            reward += self.BLOCK_FAIL_PENALTY
            self.total_blocks_failed += 1

        # NEW: BONUS REWARD for block-then-hit sequence
        if self._detect_block_then_hit_sequence():
            reward += self.BLOCK_THEN_HIT_BONUS
            self.total_block_then_hit_sequences += 1

        # CRITICAL: Cap reward to prevent PPO value explosion
        reward = np.clip(reward, -self.max_reward_per_step, self.max_reward_per_step)

        # Ensure reward is finite
        if not np.isfinite(reward):
            reward = 0.0

        # Update tracking
        self.total_reward_given += reward
        self.reward_history.append(reward)

        # Update simple state
        self.last_opponent_attacking = opponent_attacking
        self.last_damage_received = damage_received
        self.last_damage_dealt = damage_dealt

        # Return with simple stats
        return {
            "bait_punish_reward": reward,  # Keep same key name for compatibility
            "hits_dealt": self.total_hits_dealt,
            "hits_received": self.total_hits_received,
            "blocks_successful": self.total_blocks_successful,
            "blocks_failed": self.total_blocks_failed,
            "block_then_hit_sequences": self.total_block_then_hit_sequences,  # NEW
            "success_rate": self._calculate_success_rate(),
            "sequence_phase": self._get_current_phase(),  # NEW: Show current phase
            "phase_duration": 0,  # For compatibility
            "total_reward_given": self.total_reward_given,  # PPO stability tracking
            "avg_reward_per_step": self._get_avg_reward_per_step(),
        }

    def _detect_block_then_hit_sequence(self) -> bool:
        """
        Detect if player successfully blocked then immediately hit opponent.
        Looks for pattern: Block (successful) -> Hit (successful) within 3 frames.
        """
        if len(self.recent_actions) < 2:
            return False

        # Check last few frames for block-then-hit pattern
        for i in range(
            max(0, len(self.recent_actions) - 3), len(self.recent_actions) - 1
        ):
            block_frame = self.recent_actions[i]

            # Check if this frame was a successful block
            if (
                block_frame["player_blocking"]
                and block_frame["opponent_attacking"]
                and block_frame["damage_received"] == 0
            ):

                # Check next 1-2 frames for successful hit
                for j in range(i + 1, min(i + 3, len(self.recent_actions))):
                    hit_frame = self.recent_actions[j]

                    if hit_frame["player_attacking"] and hit_frame["damage_dealt"] > 0:
                        return True

        return False

    def _get_current_phase(self) -> str:
        """
        Determine current phase based on recent actions.
        """
        if len(self.recent_actions) == 0:
            return "neutral"

        current = self.recent_actions[-1]

        # Check if we just completed a block-then-hit sequence
        if self._detect_block_then_hit_sequence():
            return "punishing"

        # Check current action
        if current["player_blocking"] and current["opponent_attacking"]:
            return "blocking"
        elif current["player_attacking"]:
            return "attacking"
        elif current["opponent_attacking"]:
            return "defending"
        else:
            return "neutral"

    def _is_attacking(self, button_features: np.ndarray) -> bool:
        """Simple attack detection."""
        try:
            if not isinstance(button_features, np.ndarray) or len(button_features) < 12:
                return False
            attack_buttons = [0, 1, 8, 9, 10, 11]  # B, Y, A, X, L, R
            return bool(np.any(button_features[attack_buttons] > 0.5))
        except Exception:
            return False

    def _is_blocking(self, button_features: np.ndarray) -> bool:
        """Simple block detection."""
        try:
            if not isinstance(button_features, np.ndarray) or len(button_features) < 12:
                return False
            # In Street Fighter, blocking is typically done by holding back (left/right)
            # when opponent is attacking
            defensive_buttons = [6, 7]  # LEFT, RIGHT (back from opponent)
            return bool(np.any(button_features[defensive_buttons] > 0.5))
        except Exception:
            return False

    def _infer_opponent_attacking(self, info: Dict, damage_received: int) -> bool:
        """Simple opponent attack inference."""
        # Simple heuristic: if we received damage, opponent was probably attacking
        return damage_received > 0

    def _calculate_success_rate(self) -> float:
        """Simple success rate calculation."""
        total_actions = self.total_hits_dealt + self.total_blocks_successful
        total_attempts = (
            total_actions + self.total_hits_received + self.total_blocks_failed
        )

        if total_attempts == 0:
            return 0.0

        return total_actions / total_attempts

    def _get_avg_reward_per_step(self) -> float:
        """Get average reward per step for PPO stability monitoring."""
        if len(self.reward_history) == 0:
            return 0.0

        return np.mean(list(self.reward_history))

    def reset_sequence(self):
        """Reset method for compatibility."""
        # Clear recent actions but keep statistics
        self.recent_actions.clear()
        self.last_opponent_attacking = False
        self.last_damage_received = 0
        self.last_damage_dealt = 0

    def get_learning_stats(self) -> Dict:
        """Get statistics for compatibility and monitoring."""
        avg_reward = self._get_avg_reward_per_step()

        return {
            "hits_dealt": self.total_hits_dealt,
            "hits_received": self.total_hits_received,
            "blocks_successful": self.total_blocks_successful,
            "blocks_failed": self.total_blocks_failed,
            "block_then_hit_sequences": self.total_block_then_hit_sequences,
            "success_rate": self._calculate_success_rate(),
            "total_reward_given": self.total_reward_given,
            "avg_reward_per_step": avg_reward,
            "reward_std": (
                np.std(list(self.reward_history))
                if len(self.reward_history) > 1
                else 0.0
            ),
            "current_phase": self._get_current_phase(),
        }


def integrate_bait_punish_system(strategic_tracker):
    """
    Integrate the bait-punish system into the strategic tracker.
    CRITICAL: This function is called from the wrapper to add bait-punish features.
    """
    try:
        if not hasattr(strategic_tracker, "bait_punish_detector"):
            strategic_tracker.bait_punish_detector = SimpleBlockPunishDetector()
            print("✅ Bait-punish system integrated into strategic tracker")

        # Add method to get bait-punish features
        def get_bait_punish_features():
            """Get 7 bait-punish features for the enhanced feature vector."""
            try:
                stats = strategic_tracker.bait_punish_detector.get_learning_stats()
                features = np.zeros(7, dtype=np.float32)

                # Feature 0: Success rate
                features[0] = np.clip(stats.get("success_rate", 0.0), 0.0, 1.0)

                # Feature 1: Hits dealt (normalized)
                features[1] = np.clip(stats.get("hits_dealt", 0) / 100.0, 0.0, 1.0)

                # Feature 2: Hits received (normalized)
                features[2] = np.clip(stats.get("hits_received", 0) / 100.0, 0.0, 1.0)

                # Feature 3: Blocks successful (normalized)
                features[3] = np.clip(
                    stats.get("blocks_successful", 0) / 100.0, 0.0, 1.0
                )

                # Feature 4: Blocks failed (normalized)
                features[4] = np.clip(stats.get("blocks_failed", 0) / 100.0, 0.0, 1.0)

                # Feature 5: Block-then-hit sequences (normalized)
                features[5] = np.clip(
                    stats.get("block_then_hit_sequences", 0) / 10.0, 0.0, 1.0
                )

                # Feature 6: Current phase (one-hot encoded)
                current_phase = stats.get("current_phase", "neutral")
                if current_phase == "punishing":
                    features[6] = 1.0
                elif current_phase == "blocking":
                    features[6] = 0.5
                elif current_phase == "attacking":
                    features[6] = 0.3
                else:
                    features[6] = 0.0

                return features

            except Exception as e:
                print(f"⚠️  Error getting bait-punish features: {e}")
                return np.zeros(7, dtype=np.float32)

        # Add the method to the strategic tracker
        strategic_tracker.get_bait_punish_features = get_bait_punish_features

        return True

    except Exception as e:
        print(f"❌ Failed to integrate bait-punish system: {e}")
        return False


class AdaptiveRewardShaper:
    """
    OPTIONAL: Adaptive reward shaper for bait-punish system.
    CRITICAL: Designed to prevent PPO value explosion with careful reward scaling.
    """

    def __init__(self):
        # CRITICAL: Very conservative reward scaling to prevent PPO explosion
        self.base_reward_scale = 0.1
        self.bait_punish_scale = 0.05
        self.max_total_reward = 0.2  # Hard cap on total reward

        # Adaptive parameters (very conservative)
        self.adaptation_rate = 0.001  # Very slow adaptation
        self.performance_window = 100
        self.performance_history = deque(maxlen=self.performance_window)

        # Stability tracking
        self.total_rewards_shaped = 0
        self.reward_clipping_events = 0

        print("✅ AdaptiveRewardShaper initialized with PPO stability focus")
        print(f"   - Base reward scale: {self.base_reward_scale}")
        print(f"   - Bait-punish scale: {self.bait_punish_scale}")
        print(f"   - Max total reward: {self.max_total_reward}")

    def shape_reward(
        self, base_reward: float, bait_punish_info: Dict, game_info: Dict
    ) -> float:
        """
        Shape reward with PPO stability as top priority.
        CRITICAL: All rewards are heavily clipped to prevent value explosion.
        """
        try:
            # Ensure base reward is finite and reasonable
            if not np.isfinite(base_reward):
                base_reward = 0.0

            base_reward = np.clip(base_reward, -1.0, 1.0)

            # Get bait-punish reward (should already be small)
            bait_punish_reward = bait_punish_info.get("bait_punish_reward", 0.0)

            if not np.isfinite(bait_punish_reward):
                bait_punish_reward = 0.0

            # Scale rewards conservatively
            scaled_base = base_reward * self.base_reward_scale
            scaled_bait_punish = bait_punish_reward * self.bait_punish_scale

            # Combine rewards
            total_reward = scaled_base + scaled_bait_punish

            # CRITICAL: Hard cap to prevent PPO value explosion
            original_reward = total_reward
            total_reward = np.clip(
                total_reward, -self.max_total_reward, self.max_total_reward
            )

            # Track clipping events
            if abs(original_reward - total_reward) > 1e-6:
                self.reward_clipping_events += 1

            # Ensure final reward is finite
            if not np.isfinite(total_reward):
                total_reward = 0.0

            # Update tracking
            self.total_rewards_shaped += 1
            self.performance_history.append(total_reward)

            # Optional: Very slow adaptation based on performance
            if len(self.performance_history) >= self.performance_window:
                self._adapt_slowly()

            return total_reward

        except Exception as e:
            print(f"⚠️  Error in reward shaping: {e}")
            return 0.0

    def _adapt_slowly(self):
        """
        Very slow adaptation to prevent PPO instability.
        """
        try:
            if len(self.performance_history) < self.performance_window:
                return

            # Calculate performance metrics
            recent_rewards = list(self.performance_history)
            avg_reward = np.mean(recent_rewards)
            reward_std = np.std(recent_rewards)

            # Only adapt if performance is stable (low std)
            if reward_std < 0.01:  # Very conservative threshold
                # Tiny adjustments
                if avg_reward > 0.05:  # Performing well
                    self.bait_punish_scale = min(self.bait_punish_scale + 0.001, 0.1)
                elif avg_reward < -0.05:  # Performing poorly
                    self.bait_punish_scale = max(self.bait_punish_scale - 0.001, 0.01)

        except Exception as e:
            print(f"⚠️  Error in adaptation: {e}")

    def get_adaptation_stats(self) -> Dict:
        """Get adaptation statistics for monitoring."""
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
                "reward_shaper_scale": self.bait_punish_scale,
                "avg_shaped_reward": avg_reward,
                "reward_std": reward_std,
                "total_rewards_shaped": self.total_rewards_shaped,
                "reward_clipping_events": self.reward_clipping_events,
                "clipping_rate": self.reward_clipping_events
                / max(1, self.total_rewards_shaped),
            }
        except Exception as e:
            print(f"⚠️  Error getting adaptation stats: {e}")
            return {}


# For compatibility with the main system
BaitPunishDetector = SimpleBlockPunishDetector

# Export key components
__all__ = [
    "SimpleBlockPunishDetector",
    "BaitPunishDetector",
    "AdaptiveRewardShaper",
    "integrate_bait_punish_system",
]

# Test the system
if __name__ == "__main__":
    print("🧪 Testing SimpleBlockPunishDetector...")

    detector = SimpleBlockPunishDetector()

    # Test basic functionality
    test_info = {"agent_hp": 100, "enemy_hp": 100, "score": 1000}
    test_buttons = np.zeros(12, dtype=np.float32)

    # Test normal step
    result = detector.update(test_info, test_buttons, 0, 0)
    print(f"Normal step result: {result}")

    # Test hit
    result = detector.update(test_info, test_buttons, 10, 0)
    print(f"Hit result: {result}")

    # Test block setup
    test_buttons[6] = 1.0  # LEFT button (blocking)
    result = detector.update(test_info, test_buttons, 0, 0)  # Setup block

    # Test successful block
    result = detector.update(test_info, test_buttons, 0, 0)  # Block successful
    print(f"Block result: {result}")

    # Test block-then-hit sequence
    test_buttons[6] = 0.0  # Stop blocking
    test_buttons[0] = 1.0  # B button (attack)
    result = detector.update(test_info, test_buttons, 15, 0)  # Hit after block
    print(f"Block-then-hit result: {result}")

    # Test final stats
    stats = detector.get_learning_stats()
    print(f"Final stats: {stats}")

    print("✅ SimpleBlockPunishDetector test completed")

    # Test reward shaper
    print("\n🧪 Testing AdaptiveRewardShaper...")

    shaper = AdaptiveRewardShaper()

    # Test reward shaping
    base_reward = 0.5
    bait_punish_info = {"bait_punish_reward": 0.02}
    game_info = {}

    shaped_reward = shaper.shape_reward(base_reward, bait_punish_info, game_info)
    print(f"Shaped reward: {shaped_reward}")

    # Test extreme reward (should be clipped)
    extreme_reward = 10.0
    shaped_extreme = shaper.shape_reward(extreme_reward, bait_punish_info, game_info)
    print(f"Extreme reward shaped: {shaped_extreme} (should be clipped)")

    # Test stats
    adaptation_stats = shaper.get_adaptation_stats()
    print(f"Adaptation stats: {adaptation_stats}")

    print("✅ AdaptiveRewardShaper test completed")
    print("🎉 All bait-punish system tests passed!")
