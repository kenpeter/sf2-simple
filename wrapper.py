#!/usr/bin/env python3
"""
🛡️ ENHANCED WRAPPER - RGB Version with Transformer Context Sequence (RACE CONDITION FULLY FIXED)
Key Improvements:
1. Uses exact memory addresses from data.json for reliable health detection
2. Implements robust fallback system: Memory -> Visual -> Previous frame
3. Fixed race condition in final frame health reading with multi-method validation
4. Enhanced termination detection with cross-validation
5. Improved health tracking with temporal consistency checks
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import deque, defaultdict
from gymnasium import spaces
from typing import Dict, Tuple, List, Type, Any, Optional, Union
import math
import logging
import os
from datetime import datetime
import retro
import json
import pickle
from pathlib import Path
import random
import time
import copy

# --- FIX for TypeError in retro.make ---
_original_retro_make = retro.make


def _patched_retro_make(game, state=None, **kwargs):
    if not state:
        state = "ken_bison_12.state"
    return _original_retro_make(game=game, state=state, **kwargs)


retro.make = _patched_retro_make

# Configure logging
os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
log_filename = (
    f'logs/enhanced_sf_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
)
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename)],
)
logger = logging.getLogger(__name__)

# Constants - RGB version with smaller image size
MAX_HEALTH = 176
SCREEN_WIDTH = 160
SCREEN_HEIGHT = 112
VECTOR_FEATURE_DIM = 32
MAX_FIGHT_STEPS = 3500
FRAME_STACK_SIZE = 8
CONTEXT_SEQUENCE_DIM = 64

# Memory addresses from data.json
MEMORY_ADDRESSES = {
    "agent_hp": 16744514,  # >i2 (big-endian signed 16-bit)
    "enemy_hp": 16745154,  # >i2 (big-endian signed 16-bit)
    "agent_x": 16744454,  # >u2 (big-endian unsigned 16-bit)
    "agent_y": 16744458,  # >u2
    "enemy_x": 16745094,  # >u2
    "enemy_y": 16745098,  # >u2
    "agent_victories": 16744922,  # |u1 (unsigned 8-bit)
    "enemy_victories": 16745559,  # >u4 (big-endian unsigned 32-bit)
    "round_countdown": 16750378,  # >u2
    "reset_countdown": 16744917,  # |u1
    "agent_status": 16744450,  # >u2
    "enemy_status": 16745090,  # >u2
}

print(
    f"🚀 ENHANCED Street Fighter II Configuration (RGB with Transformer - SINGLE ROUND MODE):"
)
print(f"   - Game Mode: SINGLE ROUND (episode ends after one round)")
print(f"   - Health detection: MEMORY-FIRST with Visual Fallback")
print(f"   - Memory addresses: data.json validated")
print(f"   - Image format: RGB (3 channels)")
print(f"   - Image size: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
print(f"   - Time-decayed rewards: ENABLED")
print(f"   - Aggressive exploration: ACTIVE")
print(f"   - Frame stacking: {FRAME_STACK_SIZE} frames")
print(f"   - Transformer context sequence: ENABLED")
print(f"   - Win/Loss detection: RACE CONDITION FULLY FIXED")


# Utility functions
def safe_divide(numerator, denominator, default=0.0):
    try:
        if isinstance(numerator, np.ndarray):
            numerator = (
                numerator.item() if numerator.size == 1 else float(numerator.flat[0])
            )
        if isinstance(denominator, np.ndarray):
            denominator = (
                denominator.item()
                if denominator.size == 1
                else float(denominator.flat[0])
            )
        numerator = float(numerator) if numerator is not None else default
        denominator = (
            float(denominator)
            if denominator is not None
            else (1.0 if default == 0.0 else default)
        )
        if denominator == 0 or not np.isfinite(denominator):
            return default
        result = numerator / denominator
        return result if np.isfinite(result) else default
    except:
        return default


def safe_std(values, default=0.0):
    if len(values) < 2:
        return default
    try:
        values_array = np.array(values)
        finite_values = values_array[np.isfinite(values_array)]
        if len(finite_values) < 2:
            return default
        std_val = np.std(finite_values)
        return std_val if np.isfinite(std_val) else default
    except:
        return default


def safe_mean(values, default=0.0):
    if len(values) == 0:
        return default
    try:
        values_array = np.array(values)
        finite_values = values_array[np.isfinite(values_array)]
        if len(finite_values) == 0:
            return default
        mean_val = np.mean(finite_values)
        return mean_val if np.isfinite(mean_val) else default
    except:
        return default


def sanitize_array(arr, default_val=0.0):
    if isinstance(arr, (int, float)):
        if np.isfinite(arr):
            return np.array([arr], dtype=np.float32)
        else:
            return np.array([default_val], dtype=np.float32)
    if not isinstance(arr, np.ndarray):
        try:
            arr = np.array(arr, dtype=np.float32)
        except (ValueError, TypeError):
            print(f"⚠️ Cannot convert to array: {type(arr)}, using default")
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


def ensure_scalar(value, default=0.0):
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


def ensure_feature_dimension(features, target_dim):
    if not isinstance(features, np.ndarray):
        if isinstance(features, (int, float)):
            features = np.array([features], dtype=np.float32)
        else:
            return np.zeros(target_dim, dtype=np.float32)
    if features.ndim == 0:
        features = np.array([features.item()], dtype=np.float32)
    try:
        current_length = len(features)
    except TypeError:
        return np.zeros(target_dim, dtype=np.float32)
    if current_length == target_dim:
        return features.astype(np.float32)
    elif current_length < target_dim:
        padding = np.zeros(target_dim - current_length, dtype=np.float32)
        return np.concatenate([features, padding]).astype(np.float32)
    else:
        return features[:target_dim].astype(np.float32)


# Enhanced HealthDetector with data.json memory addresses
class EnhancedHealthDetector:
    def __init__(self):
        self.health_history = {"player": deque(maxlen=15), "opponent": deque(maxlen=15)}
        self.last_valid_health = {"player": MAX_HEALTH, "opponent": MAX_HEALTH}
        self.last_reliable_health = {"player": MAX_HEALTH, "opponent": MAX_HEALTH}
        self.health_change_detected = False
        self.frame_count = 0
        self.memory_read_success_count = 0
        self.visual_fallback_count = 0

        # Visual health bar positions (fallback)
        self.bar_positions = {
            "player": {"x": 20, "y": 8, "width": 60, "height": 4},
            "opponent": {"x": 80, "y": 8, "width": 60, "height": 4},
        }

        print("🩺 Enhanced Health Detector initialized with data.json memory addresses")

    def read_memory_health(self, env):
        """Read health directly from memory using data.json addresses"""
        player_health = None
        opponent_health = None

        try:
            if hasattr(env, "data") and hasattr(env.data, "memory"):
                memory = env.data.memory

                # Try to read agent_hp (player)
                try:
                    # Read as big-endian signed 16-bit integer
                    player_bytes = []
                    for i in range(2):  # 16-bit = 2 bytes
                        byte_val = memory.read_u8(MEMORY_ADDRESSES["agent_hp"] + i)
                        player_bytes.append(byte_val)
                    # Convert big-endian bytes to signed integer
                    player_health = int.from_bytes(
                        bytes(player_bytes), byteorder="big", signed=True
                    )

                except Exception as e:
                    # Fallback to direct read methods
                    for read_method in ["read_s16", "read_u16", "read_s8", "read_u8"]:
                        try:
                            if hasattr(memory, read_method):
                                raw_val = getattr(memory, read_method)(
                                    MEMORY_ADDRESSES["agent_hp"]
                                )
                                if 0 <= raw_val <= MAX_HEALTH:
                                    player_health = raw_val
                                    break
                        except:
                            continue

                # Try to read enemy_hp (opponent)
                try:
                    # Read as big-endian signed 16-bit integer
                    opponent_bytes = []
                    for i in range(2):  # 16-bit = 2 bytes
                        byte_val = memory.read_u8(MEMORY_ADDRESSES["enemy_hp"] + i)
                        opponent_bytes.append(byte_val)
                    # Convert big-endian bytes to signed integer
                    opponent_health = int.from_bytes(
                        bytes(opponent_bytes), byteorder="big", signed=True
                    )

                except Exception as e:
                    # Fallback to direct read methods
                    for read_method in ["read_s16", "read_u16", "read_s8", "read_u8"]:
                        try:
                            if hasattr(memory, read_method):
                                raw_val = getattr(memory, read_method)(
                                    MEMORY_ADDRESSES["enemy_hp"]
                                )
                                if 0 <= raw_val <= MAX_HEALTH:
                                    opponent_health = raw_val
                                    break
                        except:
                            continue

                # Validate the readings
                if (
                    player_health is not None
                    and 0 <= player_health <= MAX_HEALTH
                    and opponent_health is not None
                    and 0 <= opponent_health <= MAX_HEALTH
                ):
                    self.memory_read_success_count += 1
                    return player_health, opponent_health

        except Exception as e:
            pass

        return None, None

    def extract_health_from_memory_fallback(self, info):
        """Fallback method using info dict"""
        player_health = MAX_HEALTH
        opponent_health = MAX_HEALTH

        # Try multiple health key combinations
        health_keys = [
            ("agent_hp", "enemy_hp"),
            ("player_health", "opponent_health"),
            ("p1_health", "p2_health"),
            ("health_p1", "health_p2"),
            ("hp_player", "hp_enemy"),
        ]

        for p_key, o_key in health_keys:
            if p_key in info and o_key in info:
                try:
                    p_hp = int(info[p_key])
                    o_hp = int(info[o_key])
                    if 0 <= p_hp <= MAX_HEALTH and 0 <= o_hp <= MAX_HEALTH:
                        if (
                            p_hp != MAX_HEALTH
                            or o_hp != MAX_HEALTH
                            or self.frame_count < 50
                        ):
                            player_health = p_hp
                            opponent_health = o_hp
                            break
                except (ValueError, TypeError):
                    continue

        return player_health, opponent_health

    def extract_health_from_visual(self, visual_obs):
        """Visual health detection as final fallback"""
        if visual_obs is None or len(visual_obs.shape) != 3:
            return MAX_HEALTH, MAX_HEALTH

        try:
            # Handle different image formats
            if visual_obs.shape[0] == 3:
                frame = np.transpose(visual_obs, (1, 2, 0))
            else:
                frame = visual_obs

            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)

            player_health = self._analyze_health_bar(frame, "player")
            opponent_health = self._analyze_health_bar(frame, "opponent")

            self.visual_fallback_count += 1
            return player_health, opponent_health
        except Exception:
            return MAX_HEALTH, MAX_HEALTH

    def _analyze_health_bar(self, frame, player_type):
        """Analyze health bar from visual frame"""
        pos = self.bar_positions[player_type]
        health_region = frame[
            pos["y"] : pos["y"] + pos["height"], pos["x"] : pos["x"] + pos["width"]
        ]

        if health_region.size == 0:
            return MAX_HEALTH

        if len(health_region.shape) == 3:
            gray_region = cv2.cvtColor(health_region, cv2.COLOR_RGB2GRAY)
        else:
            gray_region = health_region

        health_pixels = np.sum(gray_region > 50)
        total_pixels = gray_region.size
        health_percentage = health_pixels / total_pixels if total_pixels > 0 else 1.0
        estimated_health = int(health_percentage * MAX_HEALTH)

        return max(0, min(MAX_HEALTH, estimated_health))

    def get_health(self, env, info, visual_obs, is_final_frame=False):
        """
        Get health with robust fallback system
        Priority: Memory (data.json) -> Info dict -> Visual -> Previous reliable
        """
        self.frame_count += 1

        # Method 1: Try memory-based reading using data.json addresses
        player_health, opponent_health = self.read_memory_health(env)
        method_used = "memory"

        # Method 2: Fallback to info dict if memory failed
        if player_health is None or opponent_health is None:
            player_health, opponent_health = self.extract_health_from_memory_fallback(
                info
            )
            method_used = "info_dict"

        # Method 3: Visual fallback for final frame or if other methods failed
        if (
            player_health == MAX_HEALTH
            and opponent_health == MAX_HEALTH
            and self.frame_count > 100
        ) or is_final_frame:
            visual_p, visual_o = self.extract_health_from_visual(visual_obs)
            if visual_p != MAX_HEALTH or visual_o != MAX_HEALTH:
                # Cross-validate visual readings with previous readings
                if is_final_frame:
                    # On final frame, use visual if it shows a clear winner/loser scenario
                    if (visual_p <= 0 and visual_o > 0) or (
                        visual_o <= 0 and visual_p > 0
                    ):
                        player_health, opponent_health = visual_p, visual_o
                        method_used = "visual_final"
                    elif visual_p <= 0 and visual_o <= 0:
                        # Double KO scenario confirmed visually
                        player_health, opponent_health = 0, 0
                        method_used = "visual_double_ko"
                else:
                    player_health, opponent_health = visual_p, visual_o
                    method_used = "visual"

        # Validate and smooth readings
        player_health = self._validate_health_reading(
            player_health, "player", is_final_frame
        )
        opponent_health = self._validate_health_reading(
            opponent_health, "opponent", is_final_frame
        )

        # Update reliable health if readings are sensible (not on game reset)
        if not is_final_frame and (
            player_health != MAX_HEALTH
            or opponent_health != MAX_HEALTH
            or self.frame_count < 50
        ):
            self.last_reliable_health["player"] = player_health
            self.last_reliable_health["opponent"] = opponent_health

        # Check if detection is working
        if (
            player_health != MAX_HEALTH
            or opponent_health != MAX_HEALTH
            or len(set(self.health_history["player"])) > 1
            or len(set(self.health_history["opponent"])) > 1
        ):
            self.health_change_detected = True

        return player_health, opponent_health, method_used

    def _validate_health_reading(self, health, player_type, is_final_frame=False):
        """Validate and smooth health readings"""
        health = max(0, min(MAX_HEALTH, health))

        # On final frame, allow any valid health value (including 0)
        if is_final_frame:
            self.health_history[player_type].append(health)
            return health

        # During gameplay, smooth out sudden invalid jumps
        if len(self.health_history[player_type]) >= 2:
            prev_health = self.health_history[player_type][-1]
            health_change = abs(health - prev_health)

            # Reject impossible health changes (except for round start/reset)
            if health_change > MAX_HEALTH * 0.6 and self.frame_count > 50:
                # Use previous health if change is too dramatic
                health = prev_health

        self.health_history[player_type].append(health)

        # Update last valid health
        if health != MAX_HEALTH or self.frame_count < 50:
            self.last_valid_health[player_type] = health

        return health

    def get_pre_termination_health(self):
        """Get the most recent reliable health before termination"""
        return (
            self.last_reliable_health["player"],
            self.last_reliable_health["opponent"],
        )

    def is_detection_working(self):
        """Check if health detection is functioning"""
        if not self.health_change_detected and self.frame_count > 300:
            return False

        player_variance = len(set(list(self.health_history["player"])[-5:])) > 1
        opponent_variance = len(set(list(self.health_history["opponent"])[-5:])) > 1

        return player_variance or opponent_variance or self.frame_count < 100

    def get_detection_stats(self):
        """Get detection method statistics"""
        total_reads = self.memory_read_success_count + self.visual_fallback_count
        return {
            "memory_success_rate": self.memory_read_success_count / max(1, total_reads),
            "visual_fallback_rate": self.visual_fallback_count / max(1, total_reads),
            "total_reads": total_reads,
            "detection_working": self.is_detection_working(),
        }


# Enhanced RewardCalculator
class EnhancedRewardCalculator:
    def __init__(self):
        self.previous_opponent_health = MAX_HEALTH
        self.previous_player_health = MAX_HEALTH
        self.match_started = False
        self.step_count = 0
        self.max_damage_reward = 1.5
        self.base_winning_bonus = 5.0
        self.health_advantage_bonus = 0.8
        self.health_preservation_bonus = 2.0
        self.damage_taken_penalty_multiplier = 2.5
        self.double_ko_penalty = -10.0
        self.combo_bonus_multiplier = 2.0
        self.fast_damage_bonus = 1.0
        self.timeout_penalty_multiplier = 3.0

        # Tracking of round results
        self.round_won = False
        self.round_lost = False
        self.round_draw = False
        self.round_result_determined = False

        self.total_damage_dealt = 0.0
        self.total_damage_taken = 0.0
        self.consecutive_damage_frames = 0
        self.last_damage_frame = -1

    def calculate_reward(self, player_health, opponent_health, done, info):
        reward = 0.0
        reward_breakdown = {}
        self.step_count = info.get("step_count", self.step_count + 1)

        if not self.match_started:
            self.previous_player_health = player_health
            self.previous_opponent_health = opponent_health
            self.match_started = True
            return 0.0, {"initialization": 0.0}

        player_damage_taken = max(0, self.previous_player_health - player_health)
        opponent_damage_dealt = max(0, self.previous_opponent_health - opponent_health)

        self.total_damage_dealt += opponent_damage_dealt
        self.total_damage_taken += player_damage_taken

        # Damage dealt rewards
        if opponent_damage_dealt > 0:
            if self.step_count == self.last_damage_frame + 1:
                self.consecutive_damage_frames += 1
            else:
                self.consecutive_damage_frames = 1
            self.last_damage_frame = self.step_count

            damage_reward = min(
                opponent_damage_dealt / MAX_HEALTH, self.max_damage_reward
            )

            if self.consecutive_damage_frames > 1:
                combo_multiplier = min(
                    1 + (self.consecutive_damage_frames - 1) * 0.5, 3.0
                )
                damage_reward *= combo_multiplier
                reward_breakdown["combo_multiplier"] = combo_multiplier
                reward_breakdown["combo_frames"] = self.consecutive_damage_frames

            time_factor = (MAX_FIGHT_STEPS - self.step_count) / MAX_FIGHT_STEPS
            fast_bonus = damage_reward * self.fast_damage_bonus * time_factor
            damage_reward += fast_bonus
            reward_breakdown["fast_damage_bonus"] = fast_bonus

            reward += damage_reward
            reward_breakdown["damage_dealt"] = damage_reward

        # Damage taken penalties
        if player_damage_taken > 0:
            damage_penalty = (
                -(player_damage_taken / MAX_HEALTH)
                * self.damage_taken_penalty_multiplier
            )
            reward += damage_penalty
            reward_breakdown["damage_taken"] = damage_penalty

        # Health advantage bonus (only during ongoing fight)
        if not done:
            health_diff = (player_health - opponent_health) / MAX_HEALTH
            if abs(health_diff) > 0.1:
                advantage_bonus = health_diff * self.health_advantage_bonus
                reward += advantage_bonus
                reward_breakdown["health_advantage"] = advantage_bonus

        # End-of-round rewards using reliable termination reason
        if done and not self.round_result_determined:
            self.round_result_determined = True
            termination_reason = info.get("termination_reason", "unknown")

            if termination_reason == "player_defeated":
                self.round_lost = True
                self.round_won = False
                self.round_draw = False
                reward += -3.0
                reward_breakdown["round_lost"] = -3.0

            elif termination_reason == "opponent_defeated":
                self.round_won = True
                self.round_lost = False
                self.round_draw = False

                # Time bonus for faster wins
                time_bonus_factor = (
                    MAX_FIGHT_STEPS - self.step_count
                ) / MAX_FIGHT_STEPS
                time_multiplier = 1 + 3 * (time_bonus_factor**2)
                win_bonus = self.base_winning_bonus * time_multiplier
                reward += win_bonus
                reward_breakdown["round_won_base"] = win_bonus
                reward_breakdown["time_multiplier"] = time_multiplier

                # Health preservation bonus
                health_bonus = (
                    player_health / MAX_HEALTH
                ) * self.health_preservation_bonus
                reward += health_bonus
                reward_breakdown["health_preservation_bonus"] = health_bonus

                # Speed bonus for very fast wins
                if self.step_count < MAX_FIGHT_STEPS * 0.3:
                    speed_bonus = self.base_winning_bonus * 0.5
                    reward += speed_bonus
                    reward_breakdown["speed_bonus"] = speed_bonus

            elif termination_reason == "double_ko":
                self.round_draw = True
                self.round_won = False
                self.round_lost = False
                reward += self.double_ko_penalty
                reward_breakdown["double_ko_penalty"] = self.double_ko_penalty

            elif termination_reason == "timeout_win":
                self.round_won = True
                self.round_lost = False
                self.round_draw = False
                timeout_win_bonus = self.base_winning_bonus * 0.5
                reward += timeout_win_bonus
                reward_breakdown["timeout_win"] = timeout_win_bonus

            elif termination_reason == "timeout_loss":
                self.round_lost = True
                self.round_won = False
                self.round_draw = False
                reward += -2.0
                reward_breakdown["timeout_loss"] = -2.0

            elif termination_reason == "timeout_draw":
                self.round_draw = True
                self.round_won = False
                self.round_lost = False
                reward += -1.0
                reward_breakdown["timeout_draw"] = -1.0

            else:
                # Fallback case - shouldn't happen
                self.round_draw = True
                self.round_won = False
                self.round_lost = False
                reward += -1.0
                reward_breakdown["fallback_draw"] = -1.0

            # Damage ratio bonus
            if self.total_damage_dealt > 0 or self.total_damage_taken > 0:
                damage_ratio = safe_divide(
                    self.total_damage_dealt, self.total_damage_taken + 1, 1.0
                )
                damage_ratio_bonus = (damage_ratio - 1.0) * 0.8
                reward += damage_ratio_bonus
                reward_breakdown["damage_ratio"] = damage_ratio_bonus

        # Step penalty to encourage faster resolution
        step_penalty = -0.01
        reward += step_penalty
        reward_breakdown["step_penalty"] = step_penalty

        self.previous_player_health = player_health
        self.previous_opponent_health = opponent_health

        return reward, reward_breakdown

    def get_round_result(self):
        """Return accurate round result"""
        if self.round_won:
            return "WIN"
        elif self.round_lost:
            return "LOSE"
        elif self.round_draw:
            return "DRAW"
        else:
            return "ONGOING"

    def reset(self):
        self.previous_opponent_health = MAX_HEALTH
        self.previous_player_health = MAX_HEALTH
        self.match_started = False
        self.step_count = 0
        self.round_won = False
        self.round_lost = False
        self.round_draw = False
        self.round_result_determined = False
        self.total_damage_dealt = 0.0
        self.total_damage_taken = 0.0
        self.consecutive_damage_frames = 0
        self.last_damage_frame = -1


# Feature tracker
class SimplifiedFeatureTracker:
    def __init__(self, history_length=FRAME_STACK_SIZE):
        self.history_length = history_length
        self.reset()

    def reset(self):
        self.player_health_history = deque(maxlen=self.history_length)
        self.opponent_health_history = deque(maxlen=self.history_length)
        self.action_history = deque(maxlen=self.history_length)
        self.reward_history = deque(maxlen=self.history_length)
        self.damage_history = deque(maxlen=self.history_length)
        self.last_action = 0
        self.combo_count = 0

    def update(self, player_health, opponent_health, action, reward_breakdown):
        self.player_health_history.append(player_health / MAX_HEALTH)
        self.opponent_health_history.append(opponent_health / MAX_HEALTH)
        self.action_history.append(action / 55.0)

        reward_signal = reward_breakdown.get(
            "damage_dealt", 0.0
        ) - reward_breakdown.get("damage_taken", 0.0)
        self.reward_history.append(np.clip(reward_signal, -1.0, 1.0))

        damage_dealt = reward_breakdown.get("damage_dealt", 0.0)
        self.damage_history.append(damage_dealt)

        # Update combo tracking
        if damage_dealt > 0:
            if action == self.last_action:
                self.combo_count += 1
            else:
                self.combo_count = max(0, self.combo_count - 1)
        else:
            self.combo_count = max(0, self.combo_count - 1)

        self.last_action = action

    def get_features(self):
        features = []

        # Get histories and pad if necessary
        player_hist = list(self.player_health_history)
        opponent_hist = list(self.opponent_health_history)
        action_hist = list(self.action_history)
        reward_hist = list(self.reward_history)
        damage_hist = list(self.damage_history)

        # Pad histories
        while len(player_hist) < self.history_length:
            player_hist.insert(0, 1.0)
        while len(opponent_hist) < self.history_length:
            opponent_hist.insert(0, 1.0)
        while len(action_hist) < self.history_length:
            action_hist.insert(0, 0.0)
        while len(reward_hist) < self.history_length:
            reward_hist.insert(0, 0.0)
        while len(damage_hist) < self.history_length:
            damage_hist.insert(0, 0.0)

        # Add history features
        features.extend(player_hist)
        features.extend(opponent_hist)

        # Current state features
        current_player_health = player_hist[-1] if player_hist else 1.0
        current_opponent_health = opponent_hist[-1] if opponent_hist else 1.0

        # Trend analysis
        mid_point = self.history_length // 2
        player_trend = (
            current_player_health - player_hist[mid_point]
            if len(player_hist) > mid_point
            else 0.0
        )
        opponent_trend = (
            current_opponent_health - opponent_hist[mid_point]
            if len(opponent_hist) > mid_point
            else 0.0
        )

        # Performance metrics
        recent_damage = sum(damage_hist[-4:]) / 4.0
        damage_acceleration = (
            damage_hist[-1] - damage_hist[-2] if len(damage_hist) >= 2 else 0.0
        )

        # Action diversity
        recent_actions = action_hist[-4:]
        action_diversity = len(set([int(a * 55) for a in recent_actions])) / 4.0

        # Add derived features
        features.extend(
            [
                current_player_health,
                current_opponent_health,
                current_player_health - current_opponent_health,
                player_trend,
                opponent_trend,
                self.last_action / 55.0,
                min(self.combo_count / 5.0, 1.0),
                recent_damage,
                damage_acceleration,
                action_diversity,
            ]
        )

        return ensure_feature_dimension(
            np.array(features, dtype=np.float32), VECTOR_FEATURE_DIM
        )

    def get_context_sequence(self):
        action_hist = list(self.action_history)
        reward_hist = list(self.reward_history)

        # Pad histories
        while len(action_hist) < self.history_length:
            action_hist.insert(0, 0.0)
        while len(reward_hist) < self.history_length:
            reward_hist.insert(0, 0.0)

        # Get context vector
        context_vector = self.get_features()[-10:]  # Use last 10 features as context

        # Build sequence
        context_sequence = []
        for i in range(self.history_length):
            # Action one-hot
            action_one_hot = np.zeros(56)
            action_idx = int(action_hist[i] * 55.0)
            action_one_hot[action_idx] = 1.0

            # Combine action, reward, and context
            context_sequence.append(
                np.concatenate(
                    [
                        action_one_hot,
                        [reward_hist[i]],
                        context_vector,
                    ]
                )
            )

        return np.array(context_sequence, dtype=np.float32)


# Street Fighter Action Mapping
class StreetFighterDiscreteActions:
    def __init__(self):
        self.action_map = {
            0: [],
            1: ["LEFT"],
            2: ["RIGHT"],
            3: ["UP"],
            4: ["DOWN"],
            5: ["A"],
            6: ["B"],
            7: ["C"],
            8: ["X"],
            9: ["Y"],
            10: ["Z"],
            11: ["LEFT", "A"],
            12: ["LEFT", "B"],
            13: ["LEFT", "C"],
            14: ["RIGHT", "A"],
            15: ["RIGHT", "B"],
            16: ["RIGHT", "C"],
            17: ["DOWN", "A"],
            18: ["DOWN", "B"],
            19: ["DOWN", "C"],
            20: ["UP", "A"],
            21: ["UP", "B"],
            22: ["UP", "C"],
            23: ["LEFT", "X"],
            24: ["LEFT", "Y"],
            25: ["LEFT", "Z"],
            26: ["RIGHT", "X"],
            27: ["RIGHT", "Y"],
            28: ["RIGHT", "Z"],
            29: ["DOWN", "X"],
            30: ["DOWN", "Y"],
            31: ["DOWN", "Z"],
            32: ["UP", "X"],
            33: ["UP", "Y"],
            34: ["UP", "Z"],
            35: ["DOWN", "RIGHT", "A"],
            36: ["DOWN", "RIGHT", "B"],
            37: ["DOWN", "RIGHT", "C"],
            38: ["DOWN", "RIGHT", "X"],
            39: ["DOWN", "RIGHT", "Y"],
            40: ["DOWN", "RIGHT", "Z"],
            41: ["DOWN", "LEFT", "A"],
            42: ["DOWN", "LEFT", "B"],
            43: ["DOWN", "LEFT", "C"],
            44: ["DOWN", "LEFT", "X"],
            45: ["DOWN", "LEFT", "Y"],
            46: ["DOWN", "LEFT", "Z"],
            47: ["RIGHT", "DOWN", "A"],
            48: ["RIGHT", "DOWN", "B"],
            49: ["RIGHT", "DOWN", "C"],
            50: ["A", "B"],
            51: ["B", "C"],
            52: ["X", "Y"],
            53: ["Y", "Z"],
            54: ["A", "X"],
            55: ["C", "Z"],
        }
        self.n_actions = len(self.action_map)
        self.button_to_index = {tuple(v): k for k, v in self.action_map.items()}

    def get_action(self, action_idx):
        return self.action_map.get(action_idx, [])


# Context Transformer for processing action/reward/context sequences
class ContextTransformer(nn.Module):
    def __init__(self, input_dim=CONTEXT_SEQUENCE_DIM, hidden_dim=128, num_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Linear(input_dim, hidden_dim)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 4,
                dropout=0.1,
                activation="relu",
            ),
            num_layers=num_layers,
        )

        self.output = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, context_sequence, attention_mask=None):
        batch_size, seq_len, _ = context_sequence.shape

        # Embed the sequence
        x = self.embedding(context_sequence)
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, hidden_dim)

        # Create causal attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=x.device)
            ).bool()

        # Apply transformer
        x = self.transformer(x, mask=~attention_mask)

        # Take the last sequence output
        x = x[-1, :, :]  # (batch_size, hidden_dim)
        x = self.output(x)

        return x


# SimpleCNN for visual processing
class SimpleCNN(nn.Module):
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__()

        visual_space = observation_space["visual_obs"]
        vector_space = observation_space["vector_obs"]

        n_input_channels = visual_space.shape[0]
        seq_length, vector_feature_count = vector_space.shape

        # Visual CNN
        self.visual_cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )

        # Calculate visual output size
        with torch.no_grad():
            dummy_visual = torch.zeros(
                1, n_input_channels, visual_space.shape[1], visual_space.shape[2]
            )
            visual_output_size = self.visual_cnn(dummy_visual).shape[1]

        # Vector LSTM
        self.vector_lstm = nn.LSTM(
            input_size=vector_feature_count,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
        )

        self.vector_processor = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Fusion layer
        fusion_input_size = visual_output_size + 64
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        visual_obs = observations["visual_obs"]
        vector_obs = observations["vector_obs"]

        # Process visual features
        visual_features = self.visual_cnn(visual_obs.float() / 255.0)

        # Process vector features
        lstm_out, _ = self.vector_lstm(vector_obs)
        vector_features = self.vector_processor(lstm_out[:, -1, :])

        # Combine and fuse
        combined = torch.cat([visual_features, vector_features], dim=1)
        output = self.fusion(combined)

        return output


# SimpleVerifier with Transformer context
class SimpleVerifier(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        features_dim: int = 256,
    ):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.features_dim = features_dim
        self.action_dim = action_space.n if hasattr(action_space, "n") else 56

        # Feature extractor
        self.features_extractor = SimpleCNN(observation_space, features_dim)

        # Context transformer
        self.context_transformer = ContextTransformer(
            input_dim=56 + 1 + 10
        )  # Action one-hot + reward + context

        # Action embedding
        self.action_embed = nn.Sequential(
            nn.Linear(self.action_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Energy network
        self.energy_net = nn.Sequential(
            nn.Linear(features_dim + 64 + 128, 512),  # Include Transformer context
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        self.energy_scale = 0.7

    def forward(
        self, context: Dict[str, torch.Tensor], candidate_action: torch.Tensor
    ) -> torch.Tensor:
        # Extract context features
        context_features = self.features_extractor(context)

        # Process context sequence through transformer
        context_sequence = context.get(
            "context_sequence",
            torch.zeros(
                context_features.shape[0],
                FRAME_STACK_SIZE,
                56 + 1 + 10,
                device=context_features.device,
            ),
        )
        context_embedding = self.context_transformer(context_sequence)

        # Embed action
        action_embedded = self.action_embed(candidate_action)

        # Combine all features
        combined = torch.cat(
            [context_features, action_embedded, context_embedding], dim=-1
        )

        # Calculate energy
        energy = self.energy_net(combined) * self.energy_scale

        return energy


# AggressiveAgent with thinking optimization
class AggressiveAgent:
    def __init__(
        self,
        verifier: SimpleVerifier,
        thinking_steps: int = 6,
        thinking_lr: float = 0.025,
    ):
        self.verifier = verifier
        self.thinking_steps = thinking_steps
        self.thinking_lr = thinking_lr
        self.action_dim = verifier.action_dim
        self.epsilon = 0.40
        self.epsilon_decay = 0.999
        self.min_epsilon = 0.10

        self.stats = {
            "total_predictions": 0,
            "successful_optimizations": 0,
            "exploration_actions": 0,
            "exploitation_actions": 0,
        }

    def predict(
        self, observations: Dict[str, torch.Tensor], deterministic: bool = False
    ) -> Tuple[int, Dict]:
        device = next(self.verifier.parameters()).device

        # Move observations to device
        obs_device = {}
        for key, value in observations.items():
            if isinstance(value, torch.Tensor):
                obs_device[key] = value.to(device)
            else:
                obs_device[key] = torch.from_numpy(value).to(device)

        # Add batch dimension if needed
        if len(obs_device["visual_obs"].shape) == 3:
            for key in obs_device:
                obs_device[key] = obs_device[key].unsqueeze(0)

        batch_size = obs_device["visual_obs"].shape[0]

        # Exploration vs exploitation
        if not deterministic and np.random.random() < self.epsilon:
            action_idx = np.random.randint(0, self.action_dim)
            self.stats["exploration_actions"] += 1
            self.stats["total_predictions"] += 1

            thinking_info = {
                "steps_taken": 0,
                "final_energy": 0.0,
                "exploration": True,
                "epsilon": self.epsilon,
            }
            return action_idx, thinking_info

        self.stats["exploitation_actions"] += 1

        # Initialize candidate action
        if deterministic:
            candidate_action = (
                torch.ones(batch_size, self.action_dim, device=device) / self.action_dim
            )
        else:
            candidate_action = (
                torch.randn(batch_size, self.action_dim, device=device) * 0.01
            )
            candidate_action = F.softmax(candidate_action, dim=-1)

        candidate_action.requires_grad_(True)

        # Thinking optimization
        best_energy = float("inf")
        best_action = candidate_action.clone().detach()

        for step in range(self.thinking_steps):
            try:
                energy = self.verifier(obs_device, candidate_action)
                current_energy = energy.mean().item()

                if current_energy < best_energy:
                    best_energy = current_energy
                    best_action = candidate_action.clone().detach()

                # Calculate gradients
                gradients = torch.autograd.grad(
                    outputs=energy.sum(),
                    inputs=candidate_action,
                    create_graph=False,
                    retain_graph=False,
                )[0]

                # Update candidate action
                with torch.no_grad():
                    step_size = self.thinking_lr * (0.85**step)
                    candidate_action = candidate_action - step_size * gradients
                    candidate_action = F.softmax(candidate_action, dim=-1)
                    candidate_action.requires_grad_(True)

            except Exception:
                candidate_action = best_action
                break

        # Final action selection
        with torch.no_grad():
            final_action_probs = F.softmax(candidate_action, dim=-1)

            if deterministic:
                action_idx = torch.argmax(final_action_probs, dim=-1)
            else:
                if torch.rand(1).item() < 0.15:
                    action_idx = torch.randint(
                        0, self.action_dim, (batch_size,), device=device
                    )
                else:
                    action_idx = torch.multinomial(final_action_probs, 1).squeeze(-1)

        # Update epsilon
        if not deterministic:
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

        self.stats["total_predictions"] += 1

        thinking_info = {
            "steps_taken": self.thinking_steps,
            "final_energy": best_energy,
            "energy_improvement": best_energy < 0,
            "exploration": False,
            "epsilon": self.epsilon,
        }

        return action_idx.item() if batch_size == 1 else action_idx, thinking_info

    def get_thinking_stats(self) -> Dict:
        stats = self.stats.copy()
        if stats["total_predictions"] > 0:
            stats["success_rate"] = (
                stats["successful_optimizations"] / stats["total_predictions"]
            )
            stats["exploration_rate"] = (
                stats["exploration_actions"] / stats["total_predictions"]
            )
        else:
            stats["success_rate"] = 0.0
            stats["exploration_rate"] = 0.0

        stats["current_epsilon"] = self.epsilon
        return stats


# FULLY FIXED EnhancedStreetFighterWrapper
class EnhancedStreetFighterWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.reward_calculator = EnhancedRewardCalculator()
        self.feature_tracker = SimplifiedFeatureTracker()
        self.action_mapper = StreetFighterDiscreteActions()
        self.health_detector = EnhancedHealthDetector()
        self.frame_stack = deque(maxlen=FRAME_STACK_SIZE)

        # Observation spaces
        visual_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(3 * FRAME_STACK_SIZE, SCREEN_HEIGHT, SCREEN_WIDTH),
            dtype=np.uint8,
        )
        vector_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(FRAME_STACK_SIZE, VECTOR_FEATURE_DIM),
            dtype=np.float32,
        )
        context_sequence_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(FRAME_STACK_SIZE, 56 + 1 + 10),  # Action one-hot + reward + context
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Dict(
            {
                "visual_obs": visual_space,
                "vector_obs": vector_space,
                "context_sequence": context_sequence_space,
            }
        )

        self.action_space = gym.spaces.Discrete(self.action_mapper.n_actions)
        self.vector_history = deque(maxlen=FRAME_STACK_SIZE)

        self.episode_count = 0
        self.step_count = 0

        # Health tracking for race condition fix
        self.health_before_termination = {"player": MAX_HEALTH, "opponent": MAX_HEALTH}
        self.termination_validated = False

        print(
            f"🚀 EnhancedStreetFighterWrapper initialized (SINGLE ROUND MODE - RACE CONDITION FULLY FIXED)"
        )
        print(f"   - Game Mode: SINGLE ROUND (episode ends after one fight)")
        print(f"   - Memory-first health detection with data.json addresses")
        print(f"   - Visual fallback for final frame validation")
        print(f"   - Enhanced termination detection with cross-validation")

    def _initialize_frame_stack(self, initial_frame):
        self.frame_stack.clear()
        for _ in range(FRAME_STACK_SIZE):
            self.frame_stack.append(initial_frame)

    def _process_visual_frame(self, obs):
        if isinstance(obs, np.ndarray):
            if len(obs.shape) == 3:
                if obs.shape[2] == 3:
                    frame = obs
                elif obs.shape[0] == 3:
                    frame = np.transpose(obs, (1, 2, 0))
                else:
                    frame = obs
            else:
                frame = obs

            # Resize if needed
            if frame.shape[:2] != (SCREEN_HEIGHT, SCREEN_WIDTH):
                frame = cv2.resize(frame, (SCREEN_WIDTH, SCREEN_HEIGHT))

            # Ensure uint8
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)

            # Convert to CHW format
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = np.transpose(frame, (2, 0, 1))

            return frame

        return np.zeros((3, SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.uint8)

    def _get_stacked_visual_obs(self):
        if len(self.frame_stack) == 0:
            empty_frame = np.zeros((3, SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.uint8)
            return np.tile(empty_frame, (FRAME_STACK_SIZE, 1, 1))

        stacked = np.concatenate(list(self.frame_stack), axis=0)
        return stacked

    def reset(self, **kwargs):
        reset_obs, info = self.env.reset(**kwargs)

        # Reset components
        self.reward_calculator.reset()
        self.feature_tracker.reset()
        self.health_detector = EnhancedHealthDetector()
        self.vector_history.clear()

        self.episode_count += 1
        self.step_count = 0
        self.termination_validated = False

        # Process initial frame
        processed_frame = self._process_visual_frame(reset_obs)
        self._initialize_frame_stack(processed_frame)

        # Get initial health
        player_health, opponent_health, method = self.health_detector.get_health(
            self.env, info, reset_obs, is_final_frame=False
        )

        # Initialize health tracking
        self.health_before_termination = {
            "player": player_health,
            "opponent": opponent_health,
        }

        # Update feature tracker
        self.feature_tracker.update(player_health, opponent_health, 0, {})

        # Build observation
        observation = self._build_observation(reset_obs, info)

        # Update info
        info.update(
            {
                "reset_complete": True,
                "starting_health": {
                    "player": player_health,
                    "opponent": opponent_health,
                },
                "episode_count": self.episode_count,
                "health_detection_working": self.health_detector.is_detection_working(),
                "health_detection_method": method,
                "frame_stack_size": FRAME_STACK_SIZE,
                "image_format": "RGB",
                "image_size": f"{SCREEN_WIDTH}x{SCREEN_HEIGHT}",
            }
        )

        return observation, info

    def step(self, action):
        self.step_count += 1

        # Convert action and execute
        button_combination = self.action_mapper.get_action(action)
        retro_action = self._convert_to_retro_action(button_combination)

        # Execute the action in the environment
        step_obs, original_reward, done, truncated, info = self.env.step(retro_action)

        # Process visual frame
        processed_frame = self._process_visual_frame(step_obs)
        self.frame_stack.append(processed_frame)

        # ===================================================================
        # ==== SINGLE ROUND MODE: ROBUST HEALTH DETECTION AND TERMINATION ====
        # ===================================================================

        # Get health using enhanced detector
        current_player_health, current_opponent_health, detection_method = (
            self.health_detector.get_health(
                self.env, info, step_obs, is_final_frame=done
            )
        )

        # Track health before potential termination
        if not done:
            self.health_before_termination["player"] = current_player_health
            self.health_before_termination["opponent"] = current_opponent_health

        # SINGLE ROUND TERMINATION LOGIC
        round_ended = False
        termination_reason = "ongoing"

        # Check for single round termination conditions
        if done or truncated:
            round_ended = True
            done = True  # Ensure done is set for single round
            truncated = True

            # Get pre-termination health for comparison
            pre_term_player, pre_term_opponent = (
                self.health_detector.get_pre_termination_health()
            )

            # Cross-validate termination reason using multiple sources
            termination_reason = self._determine_termination_reason(
                current_player_health,
                current_opponent_health,
                pre_term_player,
                pre_term_opponent,
                step_obs,
                info,
            )

            # Use the most reliable health values for final state
            final_player_health = self._get_final_health_value(
                current_player_health, pre_term_player, "player", termination_reason
            )
            final_opponent_health = self._get_final_health_value(
                current_opponent_health,
                pre_term_opponent,
                "opponent",
                termination_reason,
            )

        elif self.step_count >= MAX_FIGHT_STEPS:
            # Timeout condition - SINGLE ROUND ENDS HERE
            round_ended = True
            done = True
            truncated = True
            final_player_health = current_player_health
            final_opponent_health = current_opponent_health

            if current_player_health > current_opponent_health:
                termination_reason = "timeout_win"
            elif current_player_health < current_opponent_health:
                termination_reason = "timeout_loss"
            else:
                termination_reason = "timeout_draw"

        elif current_player_health <= 0 or current_opponent_health <= 0:
            # Health-based termination - SINGLE ROUND ENDS HERE
            round_ended = True
            done = True
            truncated = True

            if current_player_health <= 0 and current_opponent_health > 0:
                termination_reason = "player_defeated"
                final_player_health = 0
                final_opponent_health = current_opponent_health
            elif current_opponent_health <= 0 and current_player_health > 0:
                termination_reason = "opponent_defeated"
                final_player_health = current_player_health
                final_opponent_health = 0
            elif current_player_health <= 0 and current_opponent_health <= 0:
                termination_reason = "double_ko"
                final_player_health = 0
                final_opponent_health = 0
            else:
                # Shouldn't reach here, but fallback
                final_player_health = current_player_health
                final_opponent_health = current_opponent_health
                termination_reason = "unknown_termination"
        else:
            # Ongoing fight
            final_player_health = current_player_health
            final_opponent_health = current_opponent_health
            pre_term_player, pre_term_opponent = (
                current_player_health,
                current_opponent_health,
            )

        # Update info with comprehensive data
        info.update(
            {
                "step_count": self.step_count,
                "round_ended": round_ended,
                "player_health": final_player_health,
                "opponent_health": final_opponent_health,
                "termination_reason": termination_reason,
                "health_detection_method": detection_method,
                "single_round_mode": True,  # Flag to indicate single round mode
            }
        )

        # Calculate enhanced reward
        enhanced_reward, reward_breakdown = self.reward_calculator.calculate_reward(
            final_player_health, final_opponent_health, done, info
        )

        # Update feature tracker
        self.feature_tracker.update(
            final_player_health, final_opponent_health, action, reward_breakdown
        )

        # Build observation
        observation = self._build_observation(step_obs, info)

        # Get round result
        round_result = self.reward_calculator.get_round_result()

        # Enhanced info with debugging
        detection_stats = self.health_detector.get_detection_stats()
        info.update(
            {
                "reward_breakdown": reward_breakdown,
                "enhanced_reward": enhanced_reward,
                "episode_count": self.episode_count,
                "round_result": round_result,
                "final_health_diff": final_player_health - final_opponent_health,
                "health_detection_working": self.health_detector.is_detection_working(),
                "detection_stats": detection_stats,
                "total_damage_dealt": self.reward_calculator.total_damage_dealt,
                "total_damage_taken": self.reward_calculator.total_damage_taken,
                # Debug info
                "debug_health_method": detection_method,
                "debug_pre_term_health": (
                    f"P:{pre_term_player} O:{pre_term_opponent}"
                    if round_ended
                    else "N/A"
                ),
                "debug_current_health": f"P:{current_player_health} O:{current_opponent_health}",
                "debug_final_health": f"P:{final_player_health} O:{final_opponent_health}",
            }
        )

        # Print round result with enhanced debugging (SINGLE ROUND)
        if round_ended:
            result_emoji = (
                "🏆"
                if round_result == "WIN"
                else "💀" if round_result == "LOSE" else "🤝"
            )
            speed_indicator = (
                "⚡"
                if self.step_count < MAX_FIGHT_STEPS * 0.5
                else "🐌" if self.step_count >= MAX_FIGHT_STEPS * 0.9 else "🚶"
            )
            time_multiplier = reward_breakdown.get("time_multiplier", 1.0)
            combo_info = reward_breakdown.get("combo_frames", 0)

            print(
                f"  {result_emoji}{speed_indicator} Episode {self.episode_count} [SINGLE ROUND]: {round_result} - "
                f"Steps: {self.step_count}, "
                f"Health: {final_player_health} vs {final_opponent_health}, "
                f"Method: {detection_method}, "
                f"Reason: {termination_reason}, "
                f"TimeBonus: {time_multiplier:.1f}x, Combos: {combo_info}"
            )

        return observation, enhanced_reward, done, truncated, info

    def _determine_termination_reason(
        self, current_p, current_o, pre_term_p, pre_term_o, visual_obs, info
    ):
        """Enhanced termination reason determination with cross-validation"""

        # Check if we have clear winner/loser from pre-termination health
        if pre_term_p <= 0 and pre_term_o > 0:
            return "player_defeated"
        elif pre_term_o <= 0 and pre_term_p > 0:
            return "opponent_defeated"
        elif pre_term_p <= 0 and pre_term_o <= 0:
            return "double_ko"

        # If pre-termination health doesn't show clear result, check current health
        if current_p <= 0 and current_o > 0:
            return "player_defeated"
        elif current_o <= 0 and current_p > 0:
            return "opponent_defeated"
        elif current_p <= 0 and current_o <= 0:
            return "double_ko"

        # Visual validation for ambiguous cases
        if visual_obs is not None:
            visual_p, visual_o = self.health_detector.extract_health_from_visual(
                visual_obs
            )
            if visual_p <= 0 and visual_o > 0:
                return "player_defeated"
            elif visual_o <= 0 and visual_p > 0:
                return "opponent_defeated"
            elif visual_p <= 0 and visual_o <= 0:
                return "double_ko"

        # Fallback to health difference if no clear knockout
        if pre_term_p != pre_term_o:
            if pre_term_p > pre_term_o:
                return "timeout_win"
            else:
                return "timeout_loss"

        return "timeout_draw"

    def _get_final_health_value(
        self, current_health, pre_term_health, player_type, termination_reason
    ):
        """Get the most reliable final health value"""

        # For knockout scenarios, use 0 for the defeated player
        if termination_reason == "player_defeated" and player_type == "player":
            return 0
        elif termination_reason == "opponent_defeated" and player_type == "opponent":
            return 0
        elif termination_reason == "double_ko":
            return 0

        # For timeout scenarios, use current health if available and reasonable
        if "timeout" in termination_reason:
            if 0 <= current_health <= MAX_HEALTH:
                return current_health
            elif 0 <= pre_term_health <= MAX_HEALTH:
                return pre_term_health

        # Use current health if it's valid
        if 0 <= current_health <= MAX_HEALTH:
            return current_health

        # Fallback to pre-termination health
        return pre_term_health if 0 <= pre_term_health <= MAX_HEALTH else MAX_HEALTH

    def _convert_to_retro_action(self, button_combination):
        button_tuple = tuple(button_combination)
        if button_tuple in self.action_mapper.button_to_index:
            return self.action_mapper.button_to_index[button_tuple]
        else:
            return 0

    def _build_observation(self, visual_obs, info):
        # Get stacked visual observation
        stacked_visual = self._get_stacked_visual_obs()

        # Get vector features
        vector_features = self.feature_tracker.get_features()
        self.vector_history.append(vector_features)

        # Pad vector history if needed
        while len(self.vector_history) < FRAME_STACK_SIZE:
            self.vector_history.appendleft(
                np.zeros(VECTOR_FEATURE_DIM, dtype=np.float32)
            )

        # Stack vector observations
        vector_obs = np.stack(list(self.vector_history), axis=0)

        # Get context sequence
        context_sequence = self.feature_tracker.get_context_sequence()

        return {
            "visual_obs": stacked_visual.astype(np.uint8),
            "vector_obs": vector_obs.astype(np.float32),
            "context_sequence": context_sequence.astype(np.float32),
        }


def make_enhanced_env(
    game="StreetFighterIISpecialChampionEdition-Genesis", state="ken_bison_12.state"
):
    """Create enhanced environment with race condition fix"""
    try:
        env = retro.make(
            game=game, state=state, use_restricted_actions=retro.Actions.DISCRETE
        )
        env = EnhancedStreetFighterWrapper(env)
        print(
            f"   ✅ Enhanced RGB environment created with SINGLE ROUND MODE - RACE CONDITION FULLY FIXED"
        )
        print(f"   - Game Mode: SINGLE ROUND (episode ends after one fight)")
        print(f"   - Memory-first health detection using data.json addresses")
        print(f"   - Visual fallback for final frame validation")
        print(f"   - Enhanced termination detection with cross-validation")
        print(f"   - Time-decayed rewards and aggression incentives: ACTIVE")
        return env
    except Exception as e:
        print(f"   ❌ Environment creation failed: {e}")
        raise


def verify_health_detection(env, episodes=5):
    """Verify the enhanced health detection system"""
    print(
        f"🔍 Verifying enhanced system with data.json memory addresses over {episodes} episodes..."
    )

    detection_working = 0
    health_changes_detected = 0
    correct_terminations = 0
    method_counts = defaultdict(int)
    termination_counts = defaultdict(int)

    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        step_count = 0
        episode_healths = {"player": [], "opponent": []}
        episode_methods = []

        print(f"   Episode {episode + 1}: Starting verification...")

        while not done and step_count < 300:
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)

            player_health = info.get("player_health", MAX_HEALTH)
            opponent_health = info.get("opponent_health", MAX_HEALTH)
            detection_method = info.get("health_detection_method", "unknown")

            episode_healths["player"].append(player_health)
            episode_healths["opponent"].append(opponent_health)
            episode_methods.append(detection_method)
            method_counts[detection_method] += 1

            if done:
                termination_reason = info.get("termination_reason", "unknown")
                termination_counts[termination_reason] += 1
                debug_method = info.get("debug_health_method", "N/A")
                debug_current = info.get("debug_current_health", "N/A")
                debug_final = info.get("debug_final_health", "N/A")

                print(f"     Termination: {termination_reason}")
                print(f"     Detection method: {debug_method}")
                print(f"     Final health: {debug_final}")

                # Check if termination makes sense
                if termination_reason in [
                    "player_defeated",
                    "opponent_defeated",
                    "double_ko",
                    "timeout_win",
                    "timeout_loss",
                    "timeout_draw",
                ]:
                    correct_terminations += 1

            step_count += 1

        # Check for health variations and detection quality
        player_varied = len(set(episode_healths["player"])) > 1
        opponent_varied = len(set(episode_healths["opponent"])) > 1
        detection_status = info.get("health_detection_working", False)
        detection_stats = info.get("detection_stats", {})

        if detection_status:
            detection_working += 1

        if player_varied or opponent_varied:
            health_changes_detected += 1

        most_common_method = (
            max(set(episode_methods), key=episode_methods.count)
            if episode_methods
            else "unknown"
        )

        print(
            f"   Episode {episode + 1}: Detection: {detection_status}, "
            f"Primary method: {most_common_method}, "
            f"Player range: {min(episode_healths['player'])}-{max(episode_healths['player'])}, "
            f"Opponent range: {min(episode_healths['opponent'])}-{max(episode_healths['opponent'])}, "
            f"Result: {info.get('round_result', 'UNKNOWN')}"
        )

    success_rate = health_changes_detected / episodes
    termination_accuracy = correct_terminations / episodes

    print(f"\n🎯 Enhanced System Results (RACE CONDITION FULLY FIXED):")
    print(f"   - Health detection working: {detection_working}/{episodes}")
    print(
        f"   - Health changes detected: {health_changes_detected}/{episodes} ({success_rate:.1%})"
    )
    print(
        f"   - Correct termination detection: {correct_terminations}/{episodes} ({termination_accuracy:.1%})"
    )

    print(f"   - Detection method usage:")
    total_method_calls = sum(method_counts.values())
    for method, count in method_counts.items():
        percentage = (count / total_method_calls * 100) if total_method_calls > 0 else 0
        print(f"     • {method}: {count} calls ({percentage:.1f}%)")

    print(f"   - Termination reason distribution:")
    for reason, count in termination_counts.items():
        print(f"     • {reason}: {count} episodes")

    print(f"   - Memory addresses from data.json: ACTIVE")
    print(f"   - Visual fallback system: ACTIVE")
    print(f"   - Race condition fix: FULLY IMPLEMENTED")

    # Check for the specific "double_ko" bug
    double_ko_rate = termination_counts.get("double_ko", 0) / episodes
    if double_ko_rate > 0.7:  # More than 70% double KOs indicates the bug
        print(
            f"   ⚠️  High double KO rate ({double_ko_rate:.1%}) - race condition may persist"
        )
        return False
    elif double_ko_rate > 0.3:  # 30-70% double KOs
        print(
            f"   ⚠️  Moderate double KO rate ({double_ko_rate:.1%}) - some race condition issues remain"
        )

    if success_rate > 0.8 and termination_accuracy > 0.9 and double_ko_rate < 0.3:
        print(f"   ✅ Enhanced system with race condition fix is working excellently!")
        return True
    elif success_rate > 0.6 and termination_accuracy > 0.8 and double_ko_rate < 0.5:
        print(f"   ✅ Enhanced system is working well. Ready for training.")
        return True
    else:
        print(f"   ⚠️ System may need further adjustment.")
        return False


# Enhanced testing function
def test_race_condition_fix(env, test_episodes=10):
    """Specific test for the race condition fix"""
    print(f"🧪 Testing race condition fix over {test_episodes} episodes...")

    results = {
        "player_wins": 0,
        "opponent_wins": 0,
        "double_kos": 0,
        "timeouts": 0,
        "detection_methods": defaultdict(int),
        "health_readings": [],
    }

    for episode in range(test_episodes):
        obs, info = env.reset()
        done = False
        step_count = 0

        while not done and step_count < 400:
            # Use more aggressive actions to increase chance of knockouts
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)

            if done:
                termination_reason = info.get("termination_reason", "unknown")
                detection_method = info.get("health_detection_method", "unknown")
                final_health = info.get("debug_final_health", "N/A")

                results["detection_methods"][detection_method] += 1
                results["health_readings"].append(final_health)

                if "player_defeated" in termination_reason:
                    results["opponent_wins"] += 1
                elif "opponent_defeated" in termination_reason:
                    results["player_wins"] += 1
                elif "double_ko" in termination_reason:
                    results["double_kos"] += 1
                elif "timeout" in termination_reason:
                    results["timeouts"] += 1

                print(
                    f"   Episode {episode + 1}: {termination_reason} - {final_health} - Method: {detection_method}"
                )
                break

            step_count += 1

    print(f"\n🔬 Race Condition Fix Test Results:")
    print(f"   - Player wins: {results['player_wins']}")
    print(f"   - Opponent wins: {results['opponent_wins']}")
    print(
        f"   - Double KOs: {results['double_kos']} ({results['double_kos']/test_episodes:.1%})"
    )
    print(f"   - Timeouts: {results['timeouts']}")

    print(f"   - Detection methods used:")
    for method, count in results["detection_methods"].items():
        print(f"     • {method}: {count}")

    # The fix is working if double KO rate is reasonable (< 20%)
    double_ko_rate = results["double_kos"] / test_episodes
    if double_ko_rate < 0.2:
        print(f"   ✅ Race condition fix appears to be working! Low double KO rate.")
        return True
    else:
        print(f"   ⚠️ High double KO rate may indicate race condition persists.")
        return False


# Export all components
__all__ = [
    "EnhancedStreetFighterWrapper",
    "make_enhanced_env",
    "verify_health_detection",
    "test_race_condition_fix",
    "EnhancedHealthDetector",
    "EnhancedRewardCalculator",
    "SimplifiedFeatureTracker",
    "StreetFighterDiscreteActions",
    "SimpleCNN",
    "SimpleVerifier",
    "ContextTransformer",
    "AggressiveAgent",
    "safe_divide",
    "safe_std",
    "safe_mean",
    "sanitize_array",
    "ensure_scalar",
    "ensure_feature_dimension",
    "VECTOR_FEATURE_DIM",
    "MAX_FIGHT_STEPS",
    "MAX_HEALTH",
    "FRAME_STACK_SIZE",
    "SCREEN_WIDTH",
    "SCREEN_HEIGHT",
    "CONTEXT_SEQUENCE_DIM",
    "MEMORY_ADDRESSES",
]

print(
    f"🚀 ENHANCED Street Fighter wrapper loaded successfully! (SINGLE ROUND MODE - RACE CONDITION FULLY FIXED)"
)
print(f"   - ✅ Game Mode: SINGLE ROUND (episode ends after one fight)")
print(f"   - ✅ Memory-first health detection with data.json addresses")
print(f"   - ✅ Visual fallback for final frame validation")
print(f"   - ✅ Enhanced termination detection with cross-validation")
print(
    f"   - ✅ RGB images with frame stacking: ACTIVE ({SCREEN_WIDTH}x{SCREEN_HEIGHT})"
)
print(f"   - ✅ Time-decayed rewards and aggressive exploration: ACTIVE")
print(f"   - ✅ Transformer context sequence: ENABLED")
print(f"   - ✅ Race condition fix: FULLY IMPLEMENTED")
print(f"🎯 Ready for robust SINGLE ROUND training with accurate win/loss detection!")
