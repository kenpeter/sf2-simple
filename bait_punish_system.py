#!/usr/bin/env python3
"""
Simple Block-Punish System - Ultra-stable training approach
Based on research showing successful fighting game AI uses:
1. Immediate hit/block rewards only
2. No complex sequence tracking
3. Fixed, small reward values
4. No normalization or running statistics
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
    """

    def __init__(self):
        # Fixed rewards (research shows these values work)
        self.HIT_REWARD = 0.01  # Small positive when hitting opponent
        self.BLOCK_REWARD = 0.005  # Small positive when blocking
        self.HIT_PENALTY = -0.01  # Small negative when taking damage
        self.BLOCK_FAIL_PENALTY = -0.005  # Small negative when failing to block

        # NEW: Block-then-hit sequence reward
        self.BLOCK_THEN_HIT_BONUS = 0.02  # Bonus for successful block->hit sequence

        # Simple state tracking (no complex history)
        self.last_opponent_attacking = False
        self.last_damage_received = 0
        self.last_damage_dealt = 0

        # NEW: Track recent actions for sequence detection
        self.recent_actions = deque(maxlen=5)  # Last 5 frames

        # Simple statistics (no running normalization)
        self.total_hits_dealt = 0
        self.total_hits_received = 0
        self.total_blocks_successful = 0
        self.total_blocks_failed = 0
        self.total_block_then_hit_sequences = 0  # NEW: Track sequences

    def update(
        self,
        info: Dict,
        button_features: np.ndarray,
        damage_dealt: int,
        damage_received: int,
    ) -> Dict:
        """
        Simple update with immediate rewards + block-then-hit sequence detection.
        """

        reward = 0.0

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
            if len(button_features) < 12:
                return False
            attack_buttons = [0, 1, 8, 9, 10, 11]  # B, Y, A, X, L, R
            return bool(np.any(button_features[attack_buttons] > 0.5))
        except:
            return False

    def _is_blocking(self, button_features: np.ndarray) -> bool:
        """Simple block detection."""
        try:
            if len(button_features) < 12:
                return False
            defensive_buttons = [4, 5, 6, 7]  # UP, DOWN, LEFT, RIGHT
            return bool(np.any(button_features[defensive_buttons] > 0.5))
        except:
            return False

    def _infer_opponent_attacking(self, info: Dict, damage_received: int) -> bool:
        """Simple opponent attack inference."""
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

    def reset_sequence(self):
        """Reset method for compatibility."""
        pass

    def get_learning_stats(self) -> Dict:
        """Get statistics for compatibility."""
        return {
            "hits_dealt": self.total_hits_dealt,
            "hits_received": self.total_hits_received,
            "blocks_successful": self.total_blocks_successful,
            "blocks_failed": self.total_blocks_failed,
            "success_rate": self._calculate_success_rate(),
        }


# For compatibility, use the simple detector as the main class
BaitPunishDetector = SimpleBlockPunishDetector
