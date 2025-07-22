#!/usr/bin/env python3
"""
🥊 Energy-Based Transformer Street Fighter Wrapper
Based on energy-based transformers: https://energy-based-transformers.github.io/
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

# Configure logging - SINGLE LOG FOLDER
os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
log_filename = (
    f'logs/energy_transformer_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
)
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename)],
)
logger = logging.getLogger(__name__)

# Constants
MAX_HEALTH = 176
SCREEN_WIDTH = 320
SCREEN_HEIGHT = 224
VECTOR_FEATURE_DIM = 32
MAX_FIGHT_STEPS = 1200
TRANSFORMER_DIM = 256
TRANSFORMER_HEADS = 8
TRANSFORMER_LAYERS = 4

print(f"🚀 Energy-Based Transformer Configuration:")
print(f"   - Using Energy-Based Transformers")
print(f"   - Transformer dim: {TRANSFORMER_DIM}")
print(f"   - Attention heads: {TRANSFORMER_HEADS}")
print(f"   - Transformer layers: {TRANSFORMER_LAYERS}")


# Safe operations
def safe_divide(numerator, denominator, default=0.0):
    """Safe division that prevents NaN and handles edge cases."""
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
    """Safe standard deviation calculation."""
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
    """Safe mean calculation."""
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
            print(f"⚠️  Cannot convert to array: {type(arr)}, using default")
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


class IntelligentRewardCalculator:
    """🎯 Reward calculator for Street Fighter."""

    def __init__(self):
        self.previous_opponent_health = MAX_HEALTH
        self.previous_player_health = MAX_HEALTH
        self.match_started = False

        self.max_damage_reward = 0.8
        self.winning_bonus = 2.0
        self.health_advantage_bonus = 0.3

        self.round_won = False
        self.round_lost = False

    def calculate_reward(self, player_health, opponent_health, done, info):
        """Calculate reward based on game state."""
        reward = 0.0
        reward_breakdown = {}

        if not self.match_started:
            self.previous_player_health = player_health
            self.previous_opponent_health = opponent_health
            self.match_started = True
            return 0.0, {"initialization": 0.0}

        player_damage_taken = max(0, self.previous_player_health - player_health)
        opponent_damage_dealt = max(0, self.previous_opponent_health - opponent_health)

        if opponent_damage_dealt > 0:
            damage_reward = min(
                opponent_damage_dealt / MAX_HEALTH, self.max_damage_reward
            )
            reward += damage_reward
            reward_breakdown["damage_dealt"] = damage_reward

        if player_damage_taken > 0:
            damage_penalty = -(player_damage_taken / MAX_HEALTH) * 0.5
            reward += damage_penalty
            reward_breakdown["damage_taken"] = damage_penalty

        if done:
            if player_health > opponent_health:
                win_bonus = self.winning_bonus
                reward += win_bonus
                reward_breakdown["round_won"] = win_bonus
                self.round_won = True
            elif opponent_health > player_health:
                loss_penalty = -1.0
                reward += loss_penalty
                reward_breakdown["round_lost"] = loss_penalty
                self.round_lost = True
            else:
                draw_penalty = -0.3
                reward += draw_penalty
                reward_breakdown["draw"] = draw_penalty
        else:
            health_diff = (player_health - opponent_health) / MAX_HEALTH
            if abs(health_diff) > 0.1:
                advantage_bonus = health_diff * self.health_advantage_bonus
                reward += advantage_bonus
                reward_breakdown["health_advantage"] = advantage_bonus

        step_penalty = -0.01
        reward += step_penalty
        reward_breakdown["step_penalty"] = step_penalty

        self.previous_player_health = player_health
        self.previous_opponent_health = opponent_health

        return reward, reward_breakdown

    def reset(self):
        """Reset for new episode."""
        self.previous_opponent_health = MAX_HEALTH
        self.previous_player_health = MAX_HEALTH
        self.match_started = False
        self.round_won = False
        self.round_lost = False


class TransformerFeatureTracker:
    """📊 Feature tracker for transformer input."""

    def __init__(self, history_length=10):
        self.history_length = history_length
        self.reset()

    def reset(self):
        """Reset all tracking for new episode."""
        self.player_health_history = deque(maxlen=self.history_length)
        self.opponent_health_history = deque(maxlen=self.history_length)
        self.action_history = deque(maxlen=self.history_length)
        self.reward_history = deque(maxlen=self.history_length)
        self.last_action = 0
        self.combo_count = 0

    def update(self, player_health, opponent_health, action, reward, reward_breakdown):
        """Update tracking with current state."""
        self.player_health_history.append(player_health / MAX_HEALTH)
        self.opponent_health_history.append(opponent_health / MAX_HEALTH)
        self.action_history.append(action)
        self.reward_history.append(reward)

        if "damage_dealt" in reward_breakdown and reward_breakdown["damage_dealt"] > 0:
            if action == self.last_action:
                self.combo_count += 1
            else:
                self.combo_count = 0

        self.last_action = action

    def get_sequence_features(self):
        """Get sequential features for transformer."""
        seq_features = []

        # Pad sequences to history length
        player_hist = list(self.player_health_history)
        opponent_hist = list(self.opponent_health_history)
        action_hist = list(self.action_history)
        reward_hist = list(self.reward_history)

        while len(player_hist) < self.history_length:
            player_hist.insert(0, 1.0)
        while len(opponent_hist) < self.history_length:
            opponent_hist.insert(0, 1.0)
        while len(action_hist) < self.history_length:
            action_hist.insert(0, 0)
        while len(reward_hist) < self.history_length:
            reward_hist.insert(0, 0.0)

        for i in range(self.history_length):
            step_features = [
                player_hist[i],
                opponent_hist[i],
                action_hist[i] / 55.0,  # Normalize action
                reward_hist[i],
                player_hist[i] - opponent_hist[i],  # Health difference
            ]
            seq_features.append(step_features)

        return np.array(seq_features, dtype=np.float32)


class StreetFighterDiscreteActions:
    """🎮 Action mapping for Street Fighter."""

    def __init__(self):
        self.action_map = {
            0: [],  # No action
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
            # Combinations
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
            # Special moves
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
        """Convert action index to button combination."""
        return self.action_map.get(action_idx, [])


class StreetFighterTransformerWrapper(gym.Wrapper):
    """🥊 Street Fighter wrapper for Energy-Based Transformers."""

    def __init__(self, env, render=False):
        super().__init__(env)

        self.render_enabled = render

        self.reward_calculator = IntelligentRewardCalculator()
        self.feature_tracker = TransformerFeatureTracker()
        self.action_mapper = StreetFighterDiscreteActions()

        # Observation space: visual + sequential features
        visual_space = gym.spaces.Box(
            low=0, high=255, shape=(3, SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.uint8
        )
        sequence_space = gym.spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(10, 5),
            dtype=np.float32,  # 10 timesteps, 5 features each
        )

        self.observation_space = gym.spaces.Dict(
            {"visual_obs": visual_space, "sequence_obs": sequence_space}
        )

        self.action_space = gym.spaces.Discrete(self.action_mapper.n_actions)

        self.episode_count = 0
        self.step_count = 0
        self.previous_player_health = MAX_HEALTH
        self.previous_opponent_health = MAX_HEALTH

        print(f"🥊 StreetFighterTransformerWrapper initialized (render: {render})")

    def reset(self, **kwargs):
        """Reset environment."""
        obs, info = self.env.reset(**kwargs)

        self.reward_calculator.reset()
        self.feature_tracker.reset()
        self.previous_player_health = MAX_HEALTH
        self.previous_opponent_health = MAX_HEALTH
        self.episode_count += 1
        self.step_count = 0

        player_health, opponent_health = self._extract_health(info)
        self.feature_tracker.update(player_health, opponent_health, 0, 0.0, {})

        observation = self._build_observation(obs, info)
        return observation, info

    def step(self, action):
        """Execute action in environment."""
        self.step_count += 1

        button_combination = self.action_mapper.get_action(action)
        retro_action = self._convert_to_retro_action(button_combination)
        obs, reward, done, truncated, info = self.env.step(retro_action)

        # Render if enabled
        if self.render_enabled:
            try:
                self.env.render()
            except:
                pass  # Ignore render errors

        player_health, opponent_health = self._extract_health(info)

        # Check for fight end conditions
        if (self.previous_player_health > 0 and player_health <= 0) or (
            self.previous_opponent_health > 0 and opponent_health <= 0
        ):
            done = True
        elif self.step_count >= MAX_FIGHT_STEPS:
            done = True

        self.previous_player_health = player_health
        self.previous_opponent_health = opponent_health

        intelligent_reward, reward_breakdown = self.reward_calculator.calculate_reward(
            player_health, opponent_health, done, info
        )

        self.feature_tracker.update(
            player_health, opponent_health, action, intelligent_reward, reward_breakdown
        )

        observation = self._build_observation(obs, info)

        info.update(
            {
                "player_health": player_health,
                "opponent_health": opponent_health,
                "reward_breakdown": reward_breakdown,
                "intelligent_reward": intelligent_reward,
                "episode_count": self.episode_count,
                "step_count": self.step_count,
            }
        )

        return observation, intelligent_reward, done, truncated, info

    def _extract_health(self, info):
        """Extract health from game info."""
        player_health = info.get("player_health", MAX_HEALTH)
        opponent_health = info.get("opponent_health", MAX_HEALTH)

        if hasattr(self.env, "data") and hasattr(self.env.data, "memory"):
            try:
                player_health = self.env.data.memory.read_byte(0x8004)
                opponent_health = self.env.data.memory.read_byte(0x8008)
            except:
                pass

        return player_health, opponent_health

    def _convert_to_retro_action(self, button_combination):
        """Convert button combination to retro action."""
        button_tuple = tuple(button_combination)
        if button_tuple in self.action_mapper.button_to_index:
            return self.action_mapper.button_to_index[button_tuple]
        else:
            return 0

    def _build_observation(self, visual_obs, info):
        """Build observation dictionary."""
        if isinstance(visual_obs, np.ndarray):
            if len(visual_obs.shape) == 3 and visual_obs.shape[2] == 3:
                visual_obs = np.transpose(visual_obs, (2, 0, 1))

            if visual_obs.shape[-2:] != (SCREEN_HEIGHT, SCREEN_WIDTH):
                visual_obs = cv2.resize(
                    visual_obs.transpose(1, 2, 0), (SCREEN_WIDTH, SCREEN_HEIGHT)
                ).transpose(2, 0, 1)

        sequence_features = self.feature_tracker.get_sequence_features()

        return {
            "visual_obs": visual_obs.astype(np.uint8),
            "sequence_obs": sequence_features.astype(np.float32),
        }


class EnergyBasedTransformerCNN(nn.Module):
    """🧠 CNN feature extractor for visual input."""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__()

        visual_space = observation_space["visual_obs"]
        n_input_channels = visual_space.shape[0]

        self.visual_cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((4, 5)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 5, features_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.3)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, visual_obs: torch.Tensor) -> torch.Tensor:
        visual_obs = visual_obs.float() / 255.0
        visual_obs = torch.clamp(visual_obs, 0.0, 1.0)
        return self.visual_cnn(visual_obs)


class EnergyBasedTransformer(nn.Module):
    """🚀 Energy-Based Transformer for action prediction."""

    def __init__(self, observation_space: spaces.Dict, action_space: spaces.Space):
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space
        self.action_dim = action_space.n

        # Visual feature extractor
        self.visual_encoder = EnergyBasedTransformerCNN(
            observation_space, TRANSFORMER_DIM
        )

        # Sequence encoder
        sequence_space = observation_space["sequence_obs"]
        seq_length, seq_features = sequence_space.shape

        self.sequence_embedding = nn.Linear(seq_features, TRANSFORMER_DIM)
        self.positional_encoding = nn.Parameter(
            torch.randn(seq_length, TRANSFORMER_DIM)
        )

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=TRANSFORMER_DIM,
            nhead=TRANSFORMER_HEADS,
            dim_feedforward=TRANSFORMER_DIM * 4,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=TRANSFORMER_LAYERS
        )

        # Action embedding for energy calculation
        self.action_embedding = nn.Embedding(self.action_dim, TRANSFORMER_DIM)

        # Energy network
        self.energy_net = nn.Sequential(
            nn.Linear(TRANSFORMER_DIM * 2, TRANSFORMER_DIM),
            nn.ReLU(),
            nn.LayerNorm(TRANSFORMER_DIM),
            nn.Dropout(0.1),
            nn.Linear(TRANSFORMER_DIM, TRANSFORMER_DIM // 2),
            nn.ReLU(),
            nn.LayerNorm(TRANSFORMER_DIM // 2),
            nn.Dropout(0.1),
            nn.Linear(TRANSFORMER_DIM // 2, 1),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0, std=0.02)

    def forward(
        self, observations: Dict[str, torch.Tensor], actions: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass for energy calculation.
        If actions is None, returns context features for all actions.
        If actions is provided, returns energy for those specific actions.
        """
        device = next(self.parameters()).device

        visual_obs = observations["visual_obs"].to(device)
        sequence_obs = observations["sequence_obs"].to(device)

        batch_size = visual_obs.shape[0]

        # Visual encoding
        visual_features = self.visual_encoder(visual_obs)  # [batch, transformer_dim]

        # Sequence encoding
        seq_embedded = self.sequence_embedding(
            sequence_obs
        )  # [batch, seq_len, transformer_dim]
        seq_embedded = seq_embedded + self.positional_encoding.unsqueeze(0)

        # Transformer encoding
        context_features = self.transformer(
            seq_embedded
        )  # [batch, seq_len, transformer_dim]
        context_features = context_features.mean(
            dim=1
        )  # [batch, transformer_dim] - global average pooling

        # Combine visual and sequential context
        combined_context = (
            visual_features + context_features
        )  # [batch, transformer_dim]

        if actions is None:
            # Return context for all possible actions
            return combined_context
        else:
            # Calculate energy for specific actions
            if actions.dtype != torch.long:
                actions = actions.long()

            action_features = self.action_embedding(actions)  # [batch, transformer_dim]

            # Combine context and action
            energy_input = torch.cat(
                [combined_context, action_features], dim=-1
            )  # [batch, transformer_dim * 2]

            # Calculate energy
            energy = self.energy_net(energy_input)  # [batch, 1]

            return energy.squeeze(-1)

    def predict_action(
        self,
        observations: Dict[str, torch.Tensor],
        temperature=1.0,
        deterministic=False,
    ):
        """Predict action using energy-based sampling."""
        device = next(self.parameters()).device
        batch_size = observations["visual_obs"].shape[0]

        # Get context features
        context = self.forward(observations, actions=None)

        # Calculate energies for all actions
        all_actions = (
            torch.arange(self.action_dim, device=device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        context_expanded = context.unsqueeze(1).expand(-1, self.action_dim, -1)

        energies = []
        for i in range(self.action_dim):
            action_batch = all_actions[:, i]
            energy = self.forward(observations, actions=action_batch)
            energies.append(energy)

        energies = torch.stack(energies, dim=1)  # [batch, action_dim]

        if deterministic:
            # Choose action with lowest energy
            actions = torch.argmin(energies, dim=1)
            return actions
        else:
            # Sample using Boltzmann distribution
            probs = F.softmax(-energies / temperature, dim=1)
            actions = torch.multinomial(probs, 1).squeeze(-1)
            return actions


class QualityBasedExperienceBuffer:
    """🎯 Experience buffer for energy-based transformer training."""

    def __init__(self, capacity=30000, quality_threshold=0.3):
        self.capacity = capacity
        self.quality_threshold = quality_threshold

        self.good_experiences = deque(maxlen=capacity // 2)
        self.bad_experiences = deque(maxlen=capacity // 2)

        self.quality_scores = deque(maxlen=1000)
        self.total_added = 0

        print(f"🎯 Quality-Based Experience Buffer initialized")
        print(f"   - Quality threshold: {quality_threshold}")

    def add_experience(self, experience, reward, reward_breakdown, quality_score):
        """Add experience with quality scoring."""
        self.total_added += 1
        self.quality_scores.append(quality_score)

        if quality_score >= self.quality_threshold:
            self.good_experiences.append(experience)
        else:
            self.bad_experiences.append(experience)

    def sample_balanced_batch(self, batch_size):
        """Sample balanced batch of good and bad experiences."""
        if (
            len(self.good_experiences) < batch_size // 4
            or len(self.bad_experiences) < batch_size // 4
        ):
            return None, None

        good_count = batch_size // 2
        bad_count = batch_size // 2

        good_indices = np.random.choice(
            len(self.good_experiences), good_count, replace=False
        )
        good_batch = [self.good_experiences[i] for i in good_indices]

        bad_indices = np.random.choice(
            len(self.bad_experiences), bad_count, replace=False
        )
        bad_batch = [self.bad_experiences[i] for i in bad_indices]

        return good_batch, bad_batch

    def get_stats(self):
        """Get buffer statistics."""
        total_size = len(self.good_experiences) + len(self.bad_experiences)

        if len(self.quality_scores) > 0:
            avg_quality = np.mean(list(self.quality_scores))
        else:
            avg_quality = 0.0

        return {
            "total_size": total_size,
            "good_count": len(self.good_experiences),
            "bad_count": len(self.bad_experiences),
            "good_ratio": len(self.good_experiences) / max(1, total_size),
            "quality_threshold": self.quality_threshold,
            "avg_quality_score": avg_quality,
            "total_added": self.total_added,
        }


class CheckpointManager:
    """💾 Checkpoint manager."""

    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.best_checkpoint_path = None
        self.best_win_rate = 0.0

    def save_checkpoint(self, model, episode, win_rate, loss=0.0):
        """Save model checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"transformer_ep{episode}_wr{win_rate:.3f}_loss{loss:.3f}_{timestamp}.pt"
        )
        checkpoint_path = self.checkpoint_dir / filename

        checkpoint_data = {
            "episode": episode,
            "win_rate": win_rate,
            "loss": loss,
            "model_state_dict": model.state_dict(),
            "timestamp": timestamp,
        }

        try:
            torch.save(checkpoint_data, checkpoint_path)

            if win_rate > self.best_win_rate:
                self.best_checkpoint_path = checkpoint_path
                self.best_win_rate = win_rate

            return checkpoint_path
        except Exception as e:
            print(f"Failed to save checkpoint: {e}")
            return None

    def load_checkpoint(self, checkpoint_path, model):
        """Load model checkpoint."""
        try:
            checkpoint_data = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            model.load_state_dict(checkpoint_data["model_state_dict"])
            return checkpoint_data
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return None


# Utility functions
def make_env(
    game="StreetFighterIISpecialChampionEdition-Genesis",
    state="ken_bison_12.state",
    render=False,
):
    """Create Street Fighter environment."""
    try:
        env = retro.make(
            game=game,
            state=state,
            use_restricted_actions=retro.Actions.DISCRETE,
            render_mode="human" if render else None,
        )
        env = StreetFighterTransformerWrapper(env, render=render)
        print(f"✅ Environment created successfully (render: {render})")
        return env
    except Exception as e:
        print(f"❌ Environment creation failed: {e}")
        raise


def verify_transformer_flow(model, observation_space, action_space):
    """Verify transformer model works correctly."""
    try:
        device = next(model.parameters()).device

        # Create dummy observations
        visual_obs = torch.zeros(1, 3, SCREEN_HEIGHT, SCREEN_WIDTH, device=device)
        sequence_obs = torch.zeros(1, 10, 5, device=device)
        dummy_obs = {"visual_obs": visual_obs, "sequence_obs": sequence_obs}

        # Test context extraction
        context = model.forward(dummy_obs, actions=None)
        print(f"✅ Context extraction successful, shape: {context.shape}")

        # Test energy calculation
        dummy_action = torch.tensor([0], device=device)
        energy = model.forward(dummy_obs, actions=dummy_action)
        print(f"✅ Energy calculation successful, energy: {energy.item():.4f}")

        # Test action prediction
        action = model.predict_action(dummy_obs, deterministic=True)
        print(f"✅ Action prediction successful, action: {action.item()}")

        return True
    except Exception as e:
        print(f"❌ Transformer flow verification failed: {e}")
        return False


# Export components
__all__ = [
    "StreetFighterTransformerWrapper",
    "EnergyBasedTransformerCNN",
    "EnergyBasedTransformer",
    "TransformerFeatureTracker",
    "StreetFighterDiscreteActions",
    "IntelligentRewardCalculator",
    "QualityBasedExperienceBuffer",
    "CheckpointManager",
    "make_env",
    "verify_transformer_flow",
    "safe_divide",
    "safe_std",
    "safe_mean",
    "sanitize_array",
    "ensure_scalar",
    "MAX_FIGHT_STEPS",
    "TRANSFORMER_DIM",
    "TRANSFORMER_HEADS",
    "TRANSFORMER_LAYERS",
]

print(f"🚀 Energy-Based Transformer wrapper loaded successfully!")
print(f"   - Transformer architecture ready")
print(f"   - Single log folder: logs/")
print(f"   - Single checkpoint folder: checkpoints/")
print(f"🎯 Ready for energy-based transformer training!")
