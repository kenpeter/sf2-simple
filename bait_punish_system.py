#!/usr/bin/env python3
"""
bait_punish_system.py - Mock Implementation for Testing
This is a simplified mock version to avoid integration issues during testing.
"""

import numpy as np
from typing import Dict, Any


class AdaptiveRewardShaper:
    """Mock reward shaper that doesn't change the base reward."""

    def __init__(self):
        self.adaptation_count = 0

    def shape_reward(
        self, base_reward: float, bait_punish_info: Dict, game_info: Dict
    ) -> float:
        """Simply return the base reward without modification."""
        return base_reward

    def get_adaptation_stats(self) -> Dict[str, Any]:
        """Return mock adaptation statistics."""
        return {
            "reward_shaper_adaptations": self.adaptation_count,
            "bait_punish_bonus": 0.0,
            "reward_shaping_active": False,
        }


def integrate_bait_punish_system(strategic_tracker):
    """Mock integration function that adds minimal bait-punish features."""

    # Add a simple mock bait-punish detector
    class MockBaitPunishDetector:
        def __init__(self):
            self.learning_count = 0

        def get_learning_stats(self) -> Dict[str, Any]:
            return {
                "bait_punish_detections": 0,
                "successful_punishes": 0,
                "bait_attempts": 0,
                "punish_success_rate": 0.0,
                "learning_progress": 0.0,
            }

    # Add the mock detector to the strategic tracker
    strategic_tracker.bait_punish_detector = MockBaitPunishDetector()

    # Add mock bait-punish features (7 additional features)
    strategic_tracker.last_bait_punish_features = np.zeros(7, dtype=np.float32)

    print("✅ Mock bait-punish system integrated successfully")
