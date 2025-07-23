#!/usr/bin/env python3
"""
🚀 ENHANCED ENERGY-BASED TRANSFORMER WRAPPER
Combines existing energy-based thinking with Energy-Based Transformers
Both systems work together synergistically for superior performance
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
    f'logs/ebt_enhanced_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
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
EBT_SEQUENCE_LENGTH = 16  # Sequence length for EBT processing
EBT_HIDDEN_DIM = 256
EBT_NUM_HEADS = 8
EBT_NUM_LAYERS = 6

print(f"🚀 ENHANCED ENERGY-BASED TRANSFORMER Configuration:")
print(f"   - Energy-Based Thinking: ENABLED")
print(f"   - Energy-Based Transformers: ENABLED")
print(f"   - EBT Sequence Length: {EBT_SEQUENCE_LENGTH}")
print(f"   - EBT Hidden Dim: {EBT_HIDDEN_DIM}")
print(f"   - Synergistic Integration: ACTIVE")


# Keep all your original safe operations (unchanged)
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


def ensure_feature_dimension(features, target_dim):
    """Ensure features match target dimension exactly."""
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


def safe_comparison(value1, value2, operator="==", default=False):
    """Safe comparison that handles arrays."""
    try:
        val1 = ensure_scalar(value1)
        val2 = ensure_scalar(value2)

        if operator == "==":
            return val1 == val2
        elif operator == "!=":
            return val1 != val2
        elif operator == "<":
            return val1 < val2
        elif operator == "<=":
            return val1 <= val2
        elif operator == ">":
            return val1 > val2
        elif operator == ">=":
            return val1 >= val2
        else:
            return default
    except:
        return default


# NEW: Positional Encoding for Energy-Based Transformers
class EBTPositionalEncoding(nn.Module):
    """Positional encoding for Energy-Based Transformers."""

    def __init__(self, d_model, max_seq_length=512):
        super().__init__()
        self.d_model = d_model

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0).transpose(0, 1))

    def forward(self, x):
        """Add positional encoding to input."""
        seq_len = x.size(1)
        return x + self.pe[:seq_len, :].transpose(0, 1)


# NEW: Energy-Based Multi-Head Attention
class EBTMultiHeadAttention(nn.Module):
    """Energy-Based Multi-Head Attention for handling predicted tokens."""

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        """Compute scaled dot-product attention with energy considerations."""
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context = torch.matmul(attention_weights, V)
        return context, attention_weights

    def forward(self, query, key, value, mask=None):
        """Forward pass with residual connection."""
        batch_size = query.size(0)
        seq_length = query.size(1)

        # Residual connection input
        residual = query

        # Linear transformations and reshape
        Q = (
            self.w_q(query)
            .view(batch_size, seq_length, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        K = (
            self.w_k(key)
            .view(batch_size, seq_length, self.num_heads, self.d_k)
            .transpose(1, 2)
        )
        V = (
            self.w_v(value)
            .view(batch_size, seq_length, self.num_heads, self.d_k)
            .transpose(1, 2)
        )

        # Apply attention
        context, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads and put through final linear layer
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_length, self.d_model)
        )

        output = self.w_o(context)

        # Residual connection and layer norm
        output = self.layer_norm(output + residual)

        return output, attention_weights


# NEW: Energy-Based Transformer Block
class EBTTransformerBlock(nn.Module):
    """Energy-Based Transformer block with energy-aware processing."""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.attention = EBTMultiHeadAttention(d_model, num_heads, dropout)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """Forward pass through transformer block."""
        # Self-attention with residual connection
        attn_output, attention_weights = self.attention(x, x, x, mask)

        # Feed-forward with residual connection
        ff_input = attn_output
        ff_output = self.feed_forward(ff_input)
        output = self.layer_norm(ff_output + ff_input)

        return output, attention_weights


# NEW: Core Energy-Based Transformer
class EnergyBasedTransformer(nn.Module):
    """
    🚀 Energy-Based Transformer for Street Fighter RL
    Integrates with existing energy-based thinking system
    """

    def __init__(
        self,
        input_dim,
        d_model=EBT_HIDDEN_DIM,
        num_heads=EBT_NUM_HEADS,
        num_layers=EBT_NUM_LAYERS,
        max_seq_length=EBT_SEQUENCE_LENGTH,
        dropout=0.1,
    ):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.max_seq_length = max_seq_length

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoding = EBTPositionalEncoding(d_model, max_seq_length)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                EBTTransformerBlock(d_model, num_heads, d_model * 4, dropout)
                for _ in range(num_layers)
            ]
        )

        # Energy function head
        self.energy_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1),
        )

        # Context aggregation for sequence-level energy
        self.context_aggregator = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights for stable training."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def create_causal_mask(self, seq_length, device):
        """Create causal mask for autoregressive attention."""
        mask = torch.tril(torch.ones(seq_length, seq_length, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_length, seq_length]

    def forward(self, sequence_features, use_causal_mask=True):
        """
        Forward pass through Energy-Based Transformer.

        Args:
            sequence_features: [batch_size, seq_length, input_dim]
            use_causal_mask: Whether to use causal masking

        Returns:
            sequence_energies: [batch_size, seq_length, 1] - per-token energies
            sequence_energy: [batch_size, 1] - aggregated sequence energy
            sequence_representations: [batch_size, seq_length, d_model]
        """
        batch_size, seq_length, _ = sequence_features.shape
        device = sequence_features.device

        # Input projection and positional encoding
        x = self.input_projection(
            sequence_features
        )  # [batch_size, seq_length, d_model]
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Create attention mask if needed
        mask = None
        if use_causal_mask:
            mask = self.create_causal_mask(seq_length, device)

        # Pass through transformer blocks
        attention_weights = []
        for transformer_block in self.transformer_blocks:
            x, attn_weights = transformer_block(x, mask)
            attention_weights.append(attn_weights)

        # Store sequence representations
        sequence_representations = x

        # Calculate per-token energies
        sequence_energies = self.energy_head(x)  # [batch_size, seq_length, 1]

        # Aggregate sequence-level energy
        # Use attention-weighted aggregation
        context = self.context_aggregator(x)  # [batch_size, seq_length, d_model]

        # Compute attention weights for aggregation
        attention_scores = torch.matmul(context, context.transpose(-2, -1)) / math.sqrt(
            self.d_model
        )
        if mask is not None:
            attention_scores = attention_scores.masked_fill(
                mask.squeeze(0).squeeze(0) == 0, -1e9
            )

        attention_weights_agg = F.softmax(attention_scores, dim=-1)

        # Weighted sum for sequence energy
        weighted_energies = torch.matmul(
            attention_weights_agg, sequence_energies
        )  # [batch_size, seq_length, 1]
        sequence_energy = weighted_energies.mean(dim=1)  # [batch_size, 1]

        return {
            "sequence_energies": sequence_energies,
            "sequence_energy": sequence_energy,
            "sequence_representations": sequence_representations,
            "attention_weights": attention_weights,
            "aggregation_weights": attention_weights_agg,
        }


# ENHANCED: Sequence History Tracker for EBT
class EBTSequenceTracker:
    """🎯 Enhanced sequence tracker for Energy-Based Transformers."""

    def __init__(
        self, sequence_length=EBT_SEQUENCE_LENGTH, feature_dim=VECTOR_FEATURE_DIM
    ):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.reset()

    def reset(self):
        """Reset sequence tracking for new episode."""
        self.state_sequence = deque(maxlen=self.sequence_length)
        self.action_sequence = deque(maxlen=self.sequence_length)
        self.reward_sequence = deque(maxlen=self.sequence_length)
        self.feature_sequence = deque(maxlen=self.sequence_length)
        self.energy_sequence = deque(maxlen=self.sequence_length)
        self.step_count = 0

    def add_step(self, state_features, action, reward, energy_score=0.0):
        """Add a step to the sequence history."""
        self.state_sequence.append(state_features)
        self.action_sequence.append(action)
        self.reward_sequence.append(reward)
        self.energy_sequence.append(energy_score)

        # Create combined feature representation
        action_one_hot = np.zeros(56, dtype=np.float32)  # 56 actions
        action_one_hot[action] = 1.0

        combined_features = np.concatenate(
            [
                state_features.flatten()[
                    : self.feature_dim - 1
                ],  # State features (truncated if needed)
                [reward],  # Reward as feature
            ]
        )

        # Ensure proper dimension
        combined_features = ensure_feature_dimension(
            combined_features, self.feature_dim
        )
        self.feature_sequence.append(combined_features)

        self.step_count += 1

    def get_sequence_tensor(self, device="cpu"):
        """Get current sequence as tensors for EBT processing."""
        if len(self.feature_sequence) == 0:
            # Return zero tensor if no history
            return torch.zeros(1, self.sequence_length, self.feature_dim, device=device)

        # Pad sequence if needed
        feature_list = list(self.feature_sequence)
        while len(feature_list) < self.sequence_length:
            feature_list.insert(0, np.zeros(self.feature_dim, dtype=np.float32))

        # Convert to tensor
        sequence_tensor = (
            torch.from_numpy(np.stack(feature_list)).float().unsqueeze(0).to(device)
        )
        return sequence_tensor

    def get_sequence_info(self):
        """Get information about current sequence."""
        return {
            "length": len(self.feature_sequence),
            "max_length": self.sequence_length,
            "step_count": self.step_count,
            "recent_rewards": list(self.reward_sequence)[-5:],
            "recent_energies": list(self.energy_sequence)[-5:],
        }


# Keep your original classes but enhance them
class IntelligentRewardCalculator:
    """🎯 Your original reward calculator - UNCHANGED."""

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
        """Your original reward calculation - UNCHANGED."""
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


# ENHANCED: Feature Tracker with EBT Integration
class EBTEnhancedFeatureTracker:
    """📊 Enhanced feature tracker with EBT sequence modeling."""

    def __init__(self, history_length=5):
        self.history_length = history_length
        self.ebt_tracker = EBTSequenceTracker()
        self.reset()

    def reset(self):
        """Reset all tracking for new episode."""
        self.player_health_history = deque(maxlen=self.history_length)
        self.opponent_health_history = deque(maxlen=self.history_length)
        self.last_action = 0
        self.combo_count = 0
        self.ebt_tracker.reset()

    def update(
        self, player_health, opponent_health, action, reward_breakdown, energy_score=0.0
    ):
        """Update tracking with current state."""
        self.player_health_history.append(player_health / MAX_HEALTH)
        self.opponent_health_history.append(opponent_health / MAX_HEALTH)

        if "damage_dealt" in reward_breakdown and reward_breakdown["damage_dealt"] > 0:
            if action == self.last_action:
                self.combo_count += 1
            else:
                self.combo_count = 0

        self.last_action = action

        # Update EBT sequence tracker
        state_features = self.get_features()
        reward = sum(reward_breakdown.values()) if reward_breakdown else 0.0
        self.ebt_tracker.add_step(state_features, action, reward, energy_score)

    def get_features(self):
        """Get current feature vector."""
        features = []

        player_hist = list(self.player_health_history)
        opponent_hist = list(self.opponent_health_history)

        while len(player_hist) < self.history_length:
            player_hist.insert(0, 1.0)
        while len(opponent_hist) < self.history_length:
            opponent_hist.insert(0, 1.0)

        features.extend(player_hist)
        features.extend(opponent_hist)

        current_player_health = player_hist[-1] if player_hist else 1.0
        current_opponent_health = opponent_hist[-1] if opponent_hist else 1.0

        features.extend(
            [
                current_player_health,
                current_opponent_health,
                current_player_health - current_opponent_health,
                self.last_action / 55.0,
                min(self.combo_count / 5.0, 1.0),
            ]
        )

        return ensure_feature_dimension(
            np.array(features, dtype=np.float32), VECTOR_FEATURE_DIM
        )

    def get_ebt_sequence(self, device="cpu"):
        """Get EBT sequence tensor."""
        return self.ebt_tracker.get_sequence_tensor(device)

    def get_sequence_info(self):
        """Get EBT sequence information."""
        return self.ebt_tracker.get_sequence_info()


class StreetFighterDiscreteActions:
    """🎮 Your original action mapping - UNCHANGED."""

    def __init__(self):
        self.action_map = {
            0: [],  # No action
            1: ["LEFT"],
            2: ["RIGHT"],
            3: ["UP"],
            4: ["DOWN"],
            5: ["A"],  # Light punch
            6: ["B"],  # Medium punch
            7: ["C"],  # Heavy punch
            8: ["X"],  # Light kick
            9: ["Y"],  # Medium kick
            10: ["Z"],  # Heavy kick
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
            # Quarter circle back
            41: ["DOWN", "LEFT", "A"],
            42: ["DOWN", "LEFT", "B"],
            43: ["DOWN", "LEFT", "C"],
            44: ["DOWN", "LEFT", "X"],
            45: ["DOWN", "LEFT", "Y"],
            46: ["DOWN", "LEFT", "Z"],
            # Dragon punch motion
            47: ["RIGHT", "DOWN", "A"],
            48: ["RIGHT", "DOWN", "B"],
            49: ["RIGHT", "DOWN", "C"],
            # Additional combinations
            50: ["A", "B"],
            51: ["B", "C"],
            52: ["X", "Y"],
            53: ["Y", "Z"],
            54: ["A", "X"],
            55: ["C", "Z"],
        }

        self.n_actions = len(self.action_map)
        # Reverse mapping for button combinations to indices
        self.button_to_index = {tuple(v): k for k, v in self.action_map.items()}

    def get_action(self, action_idx):
        """Convert action index to button combination."""
        return self.action_map.get(action_idx, [])


class StreetFighterVisionWrapper(gym.Wrapper):
    """🥊 Enhanced wrapper with EBT integration."""

    def __init__(self, env):
        super().__init__(env)

        self.reward_calculator = IntelligentRewardCalculator()
        self.feature_tracker = EBTEnhancedFeatureTracker()  # Enhanced with EBT
        self.action_mapper = StreetFighterDiscreteActions()

        visual_space = gym.spaces.Box(
            low=0, high=255, shape=(3, SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.uint8
        )
        vector_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(5, VECTOR_FEATURE_DIM), dtype=np.float32
        )

        self.observation_space = gym.spaces.Dict(
            {"visual_obs": visual_space, "vector_obs": vector_space}
        )

        self.action_space = gym.spaces.Discrete(self.action_mapper.n_actions)
        self.vector_history = deque(maxlen=5)

        self.episode_count = 0
        self.step_count = 0

        # Your original health tracking
        self.previous_player_health = MAX_HEALTH
        self.previous_opponent_health = MAX_HEALTH

        print(f"🚀 Enhanced StreetFighterVisionWrapper with EBT initialized")

    def reset(self, **kwargs):
        """Enhanced reset with EBT sequence tracking."""
        obs, info = self.env.reset(**kwargs)

        self.reward_calculator.reset()
        self.feature_tracker.reset()  # This now resets EBT tracker too
        self.vector_history.clear()

        self.previous_player_health = MAX_HEALTH
        self.previous_opponent_health = MAX_HEALTH

        self.episode_count += 1
        self.step_count = 0

        player_health, opponent_health = self._extract_health(info)
        self.feature_tracker.update(player_health, opponent_health, 0, {})

        observation = self._build_observation(obs, info)

        return observation, info

    def step(self, action):
        """Enhanced step with EBT sequence tracking."""
        self.step_count += 1

        button_combination = self.action_mapper.get_action(action)
        retro_action = self._convert_to_retro_action(button_combination)
        obs, reward, done, truncated, info = self.env.step(retro_action)

        player_health, opponent_health = self._extract_health(info)

        # Timeout for single fights
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

        # Update feature tracker with EBT sequence tracking
        # Note: energy_score will be filled in by the agent later
        self.feature_tracker.update(
            player_health, opponent_health, action, reward_breakdown, energy_score=0.0
        )

        observation = self._build_observation(obs, info)

        # Enhanced info with EBT sequence information
        info.update(
            {
                "player_health": player_health,
                "opponent_health": opponent_health,
                "reward_breakdown": reward_breakdown,
                "intelligent_reward": intelligent_reward,
                "episode_count": self.episode_count,
                "step_count": self.step_count,
                "ebt_sequence_info": self.feature_tracker.get_sequence_info(),
            }
        )

        return observation, intelligent_reward, done, truncated, info

    def _extract_health(self, info):
        """Your original health extraction - UNCHANGED."""
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
        """Your original action conversion - UNCHANGED."""
        button_tuple = tuple(button_combination)
        if button_tuple in self.action_mapper.button_to_index:
            return self.action_mapper.button_to_index[button_tuple]
        else:
            return 0

    def _build_observation(self, visual_obs, info):
        """Enhanced observation building with EBT support."""
        if isinstance(visual_obs, np.ndarray):
            if len(visual_obs.shape) == 3 and visual_obs.shape[2] == 3:
                visual_obs = np.transpose(visual_obs, (2, 0, 1))

            if visual_obs.shape[-2:] != (SCREEN_HEIGHT, SCREEN_WIDTH):
                visual_obs = cv2.resize(
                    visual_obs.transpose(1, 2, 0), (SCREEN_WIDTH, SCREEN_HEIGHT)
                ).transpose(2, 0, 1)

        vector_features = self.feature_tracker.get_features()
        self.vector_history.append(vector_features)

        while len(self.vector_history) < 5:
            self.vector_history.appendleft(
                np.zeros(VECTOR_FEATURE_DIM, dtype=np.float32)
            )

        vector_obs = np.stack(list(self.vector_history), axis=0)

        return {
            "visual_obs": visual_obs.astype(np.uint8),
            "vector_obs": vector_obs.astype(np.float32),
        }

    def get_ebt_sequence(self, device="cpu"):
        """Get EBT sequence tensor for current episode."""
        return self.feature_tracker.get_ebt_sequence(device)


class GoldenExperienceBuffer:
    """🏆 Golden buffer with enhanced EBT integration."""

    def __init__(self, capacity=1000):
        self.capacity = capacity
        self.experiences = deque(maxlen=capacity)
        self.min_quality_for_golden = 0.6
        self.peak_win_rate_threshold = 0.3
        self.current_win_rate = 0.0

    def update_win_rate(self, win_rate):
        self.current_win_rate = win_rate

    def add_experience(self, experience, quality_score):
        if (
            self.current_win_rate >= self.peak_win_rate_threshold
            and quality_score >= self.min_quality_for_golden
        ):
            golden_experience = experience.copy()
            golden_experience["is_golden"] = True
            golden_experience["golden_win_rate"] = self.current_win_rate
            golden_experience["golden_quality"] = quality_score
            self.experiences.append(golden_experience)

    def sample_golden_batch(self, batch_size):
        if len(self.experiences) < batch_size:
            return list(self.experiences)
        indices = np.random.choice(len(self.experiences), batch_size, replace=False)
        return [self.experiences[i] for i in indices]

    def get_stats(self):
        if len(self.experiences) > 0:
            qualities = [exp.get("golden_quality", 0.0) for exp in self.experiences]
            win_rates = [exp.get("golden_win_rate", 0.0) for exp in self.experiences]
            avg_quality = np.mean(qualities)
            avg_win_rate = np.mean(win_rates)
            utilization = len(self.experiences) / self.capacity
        else:
            avg_quality = 0.0
            avg_win_rate = 0.0
            utilization = 0.0

        return {
            "size": len(self.experiences),
            "capacity": self.capacity,
            "utilization": utilization,
            "avg_quality": avg_quality,
            "avg_win_rate": avg_win_rate,
            "min_quality_threshold": self.min_quality_for_golden,
            "peak_win_rate_threshold": self.peak_win_rate_threshold,
        }


class EBTEnhancedExperienceBuffer:
    """🎯 Enhanced experience buffer with EBT sequence modeling."""

    def __init__(
        self, capacity=30000, quality_threshold=0.3, golden_buffer_capacity=1000
    ):
        self.capacity = capacity
        self.quality_threshold = quality_threshold

        self.good_experiences = deque(maxlen=capacity // 2)
        self.bad_experiences = deque(maxlen=capacity // 2)

        # NEW: EBT-specific experience storage
        self.sequence_experiences = deque(maxlen=capacity // 4)

        self.golden_buffer = GoldenExperienceBuffer(capacity=golden_buffer_capacity)

        self.quality_scores = deque(maxlen=1000)
        self.total_added = 0

        self.adjustment_rate = 0.05
        self.threshold_adjustment_frequency = 25
        self.threshold_adjustments = 0

        print(f"🚀 EBT-Enhanced Experience Buffer initialized")
        print(f"   - Quality threshold: {quality_threshold}")
        print(f"   - EBT sequence support: ENABLED")

    def add_experience(
        self, experience, reward, reward_breakdown, quality_score, ebt_sequence=None
    ):
        """Enhanced experience addition with EBT sequence support."""
        self.total_added += 1
        self.quality_scores.append(quality_score)

        # Store EBT sequence if provided
        if ebt_sequence is not None:
            experience["ebt_sequence"] = ebt_sequence

        if quality_score >= self.quality_threshold:
            self.good_experiences.append(experience)
            self.golden_buffer.add_experience(experience, quality_score)

            # Also store in sequence experiences if EBT sequence is available
            if ebt_sequence is not None:
                self.sequence_experiences.append(experience)
        else:
            self.bad_experiences.append(experience)

    def update_win_rate(self, win_rate):
        self.golden_buffer.update_win_rate(win_rate)

    def sample_enhanced_balanced_batch(
        self, batch_size, golden_ratio=0.15, sequence_ratio=0.1
    ):
        """Enhanced batch sampling with EBT sequence experiences."""
        if (
            len(self.good_experiences) < batch_size // 4
            or len(self.bad_experiences) < batch_size // 4
        ):
            return None, None, None, None

        golden_count = int(batch_size * golden_ratio)
        sequence_count = int(batch_size * sequence_ratio)
        remaining_good = (batch_size // 2) - golden_count - sequence_count
        bad_count = batch_size // 2

        # Sample different types of experiences
        golden_batch = (
            self.golden_buffer.sample_golden_batch(golden_count)
            if golden_count > 0
            else []
        )

        sequence_batch = []
        if sequence_count > 0 and len(self.sequence_experiences) > 0:
            seq_indices = np.random.choice(
                len(self.sequence_experiences),
                min(sequence_count, len(self.sequence_experiences)),
                replace=False,
            )
            sequence_batch = [self.sequence_experiences[i] for i in seq_indices]

        good_indices = np.random.choice(
            len(self.good_experiences), remaining_good, replace=False
        )
        good_batch = [self.good_experiences[i] for i in good_indices]

        bad_indices = np.random.choice(
            len(self.bad_experiences), bad_count, replace=False
        )
        bad_batch = [self.bad_experiences[i] for i in bad_indices]

        combined_good_batch = good_batch + golden_batch + sequence_batch
        return combined_good_batch, bad_batch, golden_batch, sequence_batch

    def get_stats(self):
        total_size = len(self.good_experiences) + len(self.bad_experiences)

        if len(self.quality_scores) > 0:
            avg_quality = np.mean(list(self.quality_scores))
            quality_std = safe_std(list(self.quality_scores), 0.0)
        else:
            avg_quality = 0.0
            quality_std = 0.0

        golden_stats = self.golden_buffer.get_stats()

        return {
            "total_size": total_size,
            "good_count": len(self.good_experiences),
            "bad_count": len(self.bad_experiences),
            "sequence_count": len(self.sequence_experiences),
            "good_ratio": len(self.good_experiences) / max(1, total_size),
            "quality_threshold": self.quality_threshold,
            "avg_quality_score": avg_quality,
            "quality_std": quality_std,
            "total_added": self.total_added,
            "acceptance_rate": total_size / max(1, self.total_added),
            "threshold_adjustments": self.threshold_adjustments,
            "golden_buffer": golden_stats,
        }

    def adjust_threshold(self, target_good_ratio=0.5, episode_number=0):
        if episode_number % self.threshold_adjustment_frequency != 0:
            return

        if len(self.quality_scores) < 100:
            return

        total_size = len(self.good_experiences) + len(self.bad_experiences)
        current_good_ratio = len(self.good_experiences) / max(1, total_size)

        if current_good_ratio < target_good_ratio - 0.15:
            self.quality_threshold *= 1 - self.adjustment_rate
            self.threshold_adjustments += 1
            print(f"📉 Lowered quality threshold to {self.quality_threshold:.3f}")
        elif current_good_ratio > target_good_ratio + 0.15:
            self.quality_threshold *= 1 + self.adjustment_rate
            self.threshold_adjustments += 1
            print(f"📈 Raised quality threshold to {self.quality_threshold:.3f}")

        self.quality_threshold = max(0.4, min(0.75, self.quality_threshold))


# Keep your original PolicyMemoryManager class unchanged
class PolicyMemoryManager:
    """🧠 Your original policy memory manager - UNCHANGED."""

    def __init__(self, performance_drop_threshold=0.05, averaging_weight=0.7):
        self.performance_drop_threshold = performance_drop_threshold
        self.averaging_weight = averaging_weight

        self.peak_win_rate = 0.0
        self.peak_checkpoint_state = None
        self.episodes_since_peak = 0
        self.performance_drop_detected = False

        self.peak_lr = None
        self.lr_reduction_factor = 0.5
        self.min_lr = 1e-7

        self.averaging_performed = 0
        self.last_averaging_episode = -1

    def update_performance(
        self, current_win_rate, current_episode, model_state_dict, current_lr
    ):
        performance_improved = False
        performance_drop = False

        if current_win_rate > self.peak_win_rate:
            self.peak_win_rate = current_win_rate
            self.peak_checkpoint_state = copy.deepcopy(model_state_dict)
            self.peak_lr = current_lr
            self.episodes_since_peak = 0
            self.performance_drop_detected = False
            performance_improved = True
        else:
            self.episodes_since_peak += 1

            if (
                current_win_rate < self.peak_win_rate - self.performance_drop_threshold
                and not self.performance_drop_detected
                and self.episodes_since_peak > 10
            ):
                self.performance_drop_detected = True
                performance_drop = True

        return performance_improved, performance_drop

    def should_perform_averaging(self, current_episode):
        return (
            self.performance_drop_detected
            and self.peak_checkpoint_state is not None
            and current_episode - self.last_averaging_episode > 5
        )

    def perform_checkpoint_averaging(self, current_model):
        if self.peak_checkpoint_state is None:
            return False

        try:
            current_state = current_model.state_dict()
            averaged_state = {}

            for name, param in current_state.items():
                if name in self.peak_checkpoint_state:
                    peak_param = self.peak_checkpoint_state[name]
                    peak_param = peak_param.to(param.device)
                    averaged_state[name] = (
                        self.averaging_weight * peak_param
                        + (1 - self.averaging_weight) * param
                    )
                else:
                    averaged_state[name] = param

            current_model.load_state_dict(averaged_state)

            self.averaging_performed += 1
            self.last_averaging_episode = self.episodes_since_peak
            self.performance_drop_detected = False

            return True

        except Exception as e:
            return False

    def should_reduce_lr(self):
        return self.performance_drop_detected and self.peak_lr is not None

    def get_reduced_lr(self, current_lr):
        new_lr = max(current_lr * self.lr_reduction_factor, self.min_lr)
        return new_lr

    def get_stats(self):
        return {
            "peak_win_rate": self.peak_win_rate,
            "episodes_since_peak": self.episodes_since_peak,
            "performance_drop_detected": self.performance_drop_detected,
            "averaging_performed": self.averaging_performed,
            "has_peak_checkpoint": self.peak_checkpoint_state is not None,
            "peak_lr": self.peak_lr,
        }


# ENHANCED: CNN with EBT Integration
class EBTEnhancedStreetFighterCNN(nn.Module):
    """🛡️ Enhanced CNN with EBT preprocessing capabilities."""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__()

        visual_space = observation_space["visual_obs"]
        vector_space = observation_space["vector_obs"]
        n_input_channels = visual_space.shape[0]
        seq_length, vector_feature_count = vector_space.shape

        # Your original CNN architecture
        self.visual_cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.15),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.15),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.1),
            nn.AdaptiveAvgPool2d((3, 4)),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy_visual = torch.zeros(
                1, n_input_channels, visual_space.shape[1], visual_space.shape[2]
            )
            visual_output_size = self.visual_cnn(dummy_visual).shape[1]

        # Enhanced vector processing for EBT compatibility
        self.vector_embed = nn.Linear(vector_feature_count, 64)
        self.vector_norm = nn.LayerNorm(64)
        self.vector_dropout = nn.Dropout(0.2)
        self.vector_gru = nn.GRU(64, 64, batch_first=True, dropout=0.15)
        self.vector_final = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
        )

        # Enhanced fusion layer for EBT integration
        fusion_input_size = visual_output_size + 32
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, 512),
            nn.ReLU(inplace=True),
            nn.LayerNorm(512),
            nn.Dropout(0.2),
            nn.Linear(512, features_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

        # NEW: EBT preparation layer
        self.ebt_projection = nn.Linear(features_dim, EBT_HIDDEN_DIM)

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
        elif isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param, gain=0.3)
                elif "bias" in name:
                    nn.init.constant_(param, 0)

    def forward(
        self, observations: Dict[str, torch.Tensor], return_ebt_features=False
    ) -> torch.Tensor:
        visual_obs = observations["visual_obs"]
        vector_obs = observations["vector_obs"]
        device = next(self.parameters()).device

        visual_obs = visual_obs.float().to(device)
        vector_obs = vector_obs.float().to(device)

        # NaN handling
        visual_nan_mask = ~torch.isfinite(visual_obs)
        vector_nan_mask = ~torch.isfinite(vector_obs)

        if torch.any(visual_nan_mask):
            visual_obs = torch.where(
                visual_nan_mask, torch.zeros_like(visual_obs), visual_obs
            )

        if torch.any(vector_nan_mask):
            vector_obs = torch.where(
                vector_nan_mask, torch.zeros_like(vector_obs), vector_obs
            )

        # Normalization
        visual_obs = torch.clamp(visual_obs / 255.0, 0.0, 1.0)
        vector_obs = torch.clamp(vector_obs, -8.0, 8.0)

        # Visual processing
        visual_features = self.visual_cnn(visual_obs)

        if torch.any(torch.abs(visual_features) > 50.0):
            visual_features = torch.clamp(visual_features, -50.0, 50.0)

        # Vector processing
        batch_size, seq_len, feature_dim = vector_obs.shape
        vector_embedded = self.vector_embed(vector_obs)
        vector_embedded = self.vector_norm(vector_embedded)
        vector_embedded = self.vector_dropout(vector_embedded)

        gru_output, _ = self.vector_gru(vector_embedded)
        vector_features = gru_output[:, -1, :]
        vector_features = self.vector_final(vector_features)

        # Fusion
        combined_features = torch.cat([visual_features, vector_features], dim=1)
        output = self.fusion(combined_features)

        # Safety checks
        if torch.any(~torch.isfinite(output)):
            output = torch.where(
                ~torch.isfinite(output), torch.zeros_like(output), output
            )

        output = torch.clamp(output, -15.0, 15.0)

        # NEW: EBT feature preparation
        if return_ebt_features:
            ebt_features = self.ebt_projection(output)
            ebt_features = torch.clamp(ebt_features, -10.0, 10.0)
            return output, ebt_features

        return output


# ENHANCED: Verifier with EBT Integration
class EBTEnhancedStreetFighterVerifier(nn.Module):
    """🛡️ Enhanced verifier with Energy-Based Transformer integration."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        features_dim: int = 256,
        use_ebt: bool = True,
    ):
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space
        self.features_dim = features_dim
        self.action_dim = action_space.n if hasattr(action_space, "n") else 56
        self.use_ebt = use_ebt

        # Enhanced feature extractor
        self.features_extractor = EBTEnhancedStreetFighterCNN(
            observation_space, features_dim
        )

        # Action embedding
        self.action_embed = nn.Sequential(
            nn.Linear(self.action_dim, 64),
            nn.ReLU(inplace=True),
            nn.LayerNorm(64),
            nn.Dropout(0.15),
        )

        # NEW: Energy-Based Transformer
        if self.use_ebt:
            # FIX 1: The EBT processes sequences of low-level features from the tracker.
            ebt_input_dim = VECTOR_FEATURE_DIM  # Corrected dimension
            self.ebt = EnergyBasedTransformer(
                input_dim=ebt_input_dim,
                d_model=EBT_HIDDEN_DIM,
                num_heads=EBT_NUM_HEADS,
                num_layers=EBT_NUM_LAYERS,
                max_seq_length=EBT_SEQUENCE_LENGTH,
            )

            # EBT-enhanced energy network
            self.energy_net = nn.Sequential(
                nn.Linear(EBT_HIDDEN_DIM + features_dim + 64, 256),
                nn.ReLU(inplace=True),
                nn.LayerNorm(256),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.LayerNorm(128),
                nn.Dropout(0.15),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.LayerNorm(64),
                nn.Dropout(0.1),
                nn.Linear(64, 1),
            )
        else:
            # Original energy network
            self.energy_net = nn.Sequential(
                nn.Linear(features_dim + 64, 256),
                nn.ReLU(inplace=True),
                nn.LayerNorm(256),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.LayerNorm(128),
                nn.Dropout(0.15),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.LayerNorm(64),
                nn.Dropout(0.1),
                nn.Linear(64, 1),
            )

        self.energy_scale = 0.6
        self.energy_clamp_min = -6.0
        self.energy_clamp_max = 6.0

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.005)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(
        self,
        context: torch.Tensor,
        candidate_action: torch.Tensor,
        sequence_context: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Enhanced forward pass with EBT integration.

        Args:
            context: State context (dict or tensor)
            candidate_action: Action candidates
            sequence_context: Optional EBT sequence context
        """
        device = next(self.parameters()).device

        # Extract features from context
        if isinstance(context, dict):
            if self.use_ebt:
                context_features, ebt_features = self.features_extractor(
                    context, return_ebt_features=True
                )
            else:
                context_features = self.features_extractor(context)
                ebt_features = None
        else:
            context_features = context
            ebt_features = None

        context_features = context_features.to(device)
        candidate_action = candidate_action.to(device)

        # Safety checks
        if torch.any(~torch.isfinite(context_features)):
            context_features = torch.where(
                ~torch.isfinite(context_features),
                torch.zeros_like(context_features),
                context_features,
            )

        if torch.any(~torch.isfinite(candidate_action)):
            candidate_action = torch.where(
                ~torch.isfinite(candidate_action),
                torch.zeros_like(candidate_action),
                candidate_action,
            )

        context_features = torch.clamp(context_features, -15.0, 15.0)
        candidate_action = torch.clamp(candidate_action, 0.0, 1.0)

        # Action embedding
        action_embedded = self.action_embed(candidate_action)

        if torch.any(~torch.isfinite(action_embedded)):
            action_embedded = torch.where(
                ~torch.isfinite(action_embedded),
                torch.zeros_like(action_embedded),
                action_embedded,
            )

        # NEW: EBT Processing
        ebt_output = None
        if self.use_ebt and ebt_features is not None:
            batch_size = ebt_features.shape[0]

            # FIX 2: Correctly handle the EBT input sequence.
            # Use the provided sequence_context (from tracker) or a safe fallback.
            if sequence_context is not None and sequence_context.shape[1] > 1:
                ebt_input = sequence_context.to(device)
            else:
                # Fallback: if no sequence context, create a dummy zero sequence.
                # This prevents a crash and ensures the model can run.
                ebt_input = torch.zeros(
                    batch_size, EBT_SEQUENCE_LENGTH, VECTOR_FEATURE_DIM, device=device
                )

            # Safety check for EBT input
            if torch.any(~torch.isfinite(ebt_input)):
                ebt_input = torch.where(
                    ~torch.isfinite(ebt_input), torch.zeros_like(ebt_input), ebt_input
                )

            # Process through EBT
            try:
                ebt_result = self.ebt(ebt_input, use_causal_mask=True)

                # Extract sequence-level energy and representations
                ebt_sequence_energy = ebt_result["sequence_energy"]  # [batch_size, 1]
                ebt_representations = ebt_result[
                    "sequence_representations"
                ]  # [batch_size, seq_len, d_model]

                # Use the last token's representation for current decision
                ebt_output = ebt_representations[:, -1, :]  # [batch_size, d_model]

                # Combine with EBT sequence energy
                ebt_energy_contribution = ebt_sequence_energy.squeeze(
                    -1
                )  # [batch_size]

            except Exception as e:
                print(f"⚠️ EBT processing failed: {e}")
                ebt_output = torch.zeros(batch_size, EBT_HIDDEN_DIM, device=device)
                ebt_energy_contribution = torch.zeros(batch_size, device=device)

        # Combine inputs for energy calculation
        if ebt_output is not None:
            # Enhanced input with EBT features
            combined_input = torch.cat(
                [context_features, action_embedded, ebt_output], dim=-1
            )
        else:
            # Original input without EBT
            combined_input = torch.cat([context_features, action_embedded], dim=-1)

        # Calculate base energy
        raw_energy = self.energy_net(combined_input)

        # Apply energy scaling
        energy = raw_energy * self.energy_scale

        # Add EBT energy contribution if available
        if self.use_ebt and "ebt_energy_contribution" in locals():
            # Combine energies with learned weighting
            ebt_weight = 0.3  # Balance between base energy and EBT energy
            energy = energy.squeeze(-1) + ebt_weight * ebt_energy_contribution
            energy = energy.unsqueeze(-1)

        # Final clamping
        energy = torch.clamp(energy, self.energy_clamp_min, self.energy_clamp_max)

        # Final safety check
        if torch.any(~torch.isfinite(energy)):
            energy = torch.where(
                ~torch.isfinite(energy), torch.zeros_like(energy), energy
            )

        return energy

    def get_energy_stats(self):
        return {
            "energy_mean": 0.0,
            "energy_std": 0.0,
            "nan_count": 0,
            "explosion_count": 0,
            "ebt_enabled": self.use_ebt,
        }


# ENHANCED: Agent with EBT Integration
class EBTEnhancedEnergyBasedAgent:
    """🛡️ Enhanced agent with Energy-Based Transformer integration."""

    def __init__(
        self,
        verifier: EBTEnhancedStreetFighterVerifier,
        thinking_steps: int = 3,
        thinking_lr: float = 0.06,
        noise_scale: float = 0.02,
        use_ebt_thinking: bool = True,
    ):
        self.verifier = verifier
        self.initial_thinking_steps = thinking_steps
        self.current_thinking_steps = thinking_steps
        self.initial_thinking_lr = thinking_lr
        self.current_thinking_lr = thinking_lr
        self.noise_scale = noise_scale
        self.action_dim = verifier.action_dim
        self.use_ebt_thinking = use_ebt_thinking

        self.gradient_clip = 0.3
        self.early_stop_patience = 2
        self.min_energy_improvement = 8e-4

        self.max_thinking_steps = 5
        self.min_thinking_steps = 1

        # Enhanced thinking stats with EBT metrics
        self.thinking_stats = {
            "total_predictions": 0,
            "avg_thinking_steps": 0.0,
            "avg_energy_improvement": 0.0,
            "early_stops": 0,
            "energy_explosions": 0,
            "gradient_explosions": 0,
            "successful_optimizations": 0,
            "failed_optimizations": 0,
            "ebt_successes": 0,
            "ebt_failures": 0,
        }

        self.recent_performance = deque(maxlen=100)

    def predict(
        self,
        observations: Dict[str, torch.Tensor],
        deterministic: bool = False,
        sequence_context: torch.Tensor = None,
    ) -> Tuple[int, Dict]:
        """Enhanced prediction with EBT integration."""
        device = next(self.verifier.parameters()).device

        obs_device = {}
        for key, value in observations.items():
            if isinstance(value, torch.Tensor):
                obs_device[key] = value.to(device)
            else:
                obs_device[key] = torch.from_numpy(value).to(device)

        if len(obs_device["visual_obs"].shape) == 3:
            for key in obs_device:
                obs_device[key] = obs_device[key].unsqueeze(0)

        batch_size = obs_device["visual_obs"].shape[0]

        # Initialize candidate action
        if deterministic:
            candidate_action = (
                torch.ones(batch_size, self.action_dim, device=device) / self.action_dim
            )
        else:
            candidate_action = (
                torch.randn(batch_size, self.action_dim, device=device)
                * self.noise_scale
            )
            candidate_action = F.softmax(candidate_action, dim=-1)

        candidate_action.requires_grad_(True)

        # Thinking process tracking
        energy_history = []
        steps_taken = 0
        early_stopped = False
        energy_explosion = False
        gradient_explosion = False
        optimization_successful = False
        ebt_success = True

        # Initial energy evaluation
        with torch.no_grad():
            try:
                if self.use_ebt_thinking and sequence_context is not None:
                    initial_energy = self.verifier(
                        obs_device, candidate_action, sequence_context
                    )
                else:
                    initial_energy = self.verifier(obs_device, candidate_action)
                energy_history.append(initial_energy.mean().item())
            except Exception as e:
                print(f"⚠️ Initial energy evaluation failed: {e}")
                return 0, {"error": "initial_energy_failed", "ebt_error": str(e)}

        # Thinking loop with EBT integration
        for step in range(self.current_thinking_steps):
            try:
                # Enhanced energy calculation with EBT
                if self.use_ebt_thinking and sequence_context is not None:
                    energy = self.verifier(
                        obs_device, candidate_action, sequence_context
                    )
                else:
                    energy = self.verifier(obs_device, candidate_action)

                # Energy explosion check
                if torch.any(torch.abs(energy) > 8.0):
                    energy_explosion = True
                    break

                # Gradient calculation
                gradients = torch.autograd.grad(
                    outputs=energy.sum(),
                    inputs=candidate_action,
                    create_graph=False,
                    retain_graph=False,
                )[0]

                gradient_norm = torch.norm(gradients).item()

                # Gradient explosion check
                if gradient_norm > self.gradient_clip:
                    gradient_explosion = True
                    gradients = gradients * (self.gradient_clip / gradient_norm)

                if torch.any(~torch.isfinite(gradients)):
                    break

                # Update candidate action
                with torch.no_grad():
                    candidate_action = (
                        candidate_action - self.current_thinking_lr * gradients
                    )
                    candidate_action = F.softmax(candidate_action, dim=-1)
                    candidate_action.requires_grad_(True)

                # Energy evaluation
                with torch.no_grad():
                    if self.use_ebt_thinking and sequence_context is not None:
                        new_energy = self.verifier(
                            obs_device, candidate_action, sequence_context
                        )
                    else:
                        new_energy = self.verifier(obs_device, candidate_action)
                    energy_history.append(new_energy.mean().item())

                # Early stopping check
                if len(energy_history) >= self.early_stop_patience + 1:
                    recent_improvement = (
                        energy_history[-self.early_stop_patience - 1]
                        - energy_history[-1]
                    )
                    if abs(recent_improvement) < self.min_energy_improvement:
                        early_stopped = True
                        break

                steps_taken += 1

            except Exception as e:
                print(f"⚠️ EBT thinking step failed: {e}")
                ebt_success = False
                break

        # Optimization success evaluation
        if len(energy_history) > 1:
            total_improvement = abs(energy_history[0] - energy_history[-1])
            optimization_successful = (
                total_improvement > self.min_energy_improvement
                and not energy_explosion
                and not gradient_explosion
                and ebt_success
            )

        # Final action selection
        with torch.no_grad():
            try:
                final_action_probs = F.softmax(candidate_action, dim=-1)
                final_action_probs = final_action_probs + 1e-8
                final_action_probs = final_action_probs / final_action_probs.sum(
                    dim=-1, keepdim=True
                )

                if deterministic:
                    action_idx = torch.argmax(final_action_probs, dim=-1)
                else:
                    action_idx = torch.multinomial(final_action_probs, 1).squeeze(-1)
            except Exception as e:
                print(f"⚠️ Action selection failed: {e}")
                return 0, {"error": "action_selection_failed", "ebt_error": str(e)}

        # Update statistics
        self.thinking_stats["total_predictions"] += 1
        self.thinking_stats["avg_thinking_steps"] = (
            self.thinking_stats["avg_thinking_steps"] * 0.9 + steps_taken * 0.1
        )

        if len(energy_history) > 1:
            energy_improvement = abs(energy_history[0] - energy_history[-1])
            self.thinking_stats["avg_energy_improvement"] = (
                self.thinking_stats["avg_energy_improvement"] * 0.9
                + energy_improvement * 0.1
            )

        # Update counters
        if early_stopped:
            self.thinking_stats["early_stops"] += 1
        if energy_explosion:
            self.thinking_stats["energy_explosions"] += 1
        if gradient_explosion:
            self.thinking_stats["gradient_explosions"] += 1
        if optimization_successful:
            self.thinking_stats["successful_optimizations"] += 1
        else:
            self.thinking_stats["failed_optimizations"] += 1

        # EBT-specific stats
        if ebt_success:
            self.thinking_stats["ebt_successes"] += 1
        else:
            self.thinking_stats["ebt_failures"] += 1

        self.recent_performance.append(1.0 if optimization_successful else 0.0)

        # Enhanced thinking info
        thinking_info = {
            "energy_history": energy_history,
            "steps_taken": steps_taken,
            "early_stopped": early_stopped,
            "energy_explosion": energy_explosion,
            "gradient_explosion": gradient_explosion,
            "optimization_successful": optimization_successful,
            "ebt_success": ebt_success,
            "energy_improvement": (
                abs(energy_history[0] - energy_history[-1])
                if len(energy_history) > 1
                else 0.0
            ),
            "final_energy": energy_history[-1] if energy_history else 0.0,
            "current_thinking_steps": self.current_thinking_steps,
            "current_thinking_lr": self.current_thinking_lr,
            "used_ebt_thinking": self.use_ebt_thinking and sequence_context is not None,
        }

        return action_idx.item() if batch_size == 1 else action_idx, thinking_info

    def get_thinking_stats(self) -> Dict:
        """Enhanced thinking stats with EBT metrics."""
        stats = self.thinking_stats.copy()
        if stats["total_predictions"] > 0:
            stats["success_rate"] = safe_divide(
                stats["successful_optimizations"], stats["total_predictions"], 0.0
            )
            stats["early_stop_rate"] = safe_divide(
                stats["early_stops"], stats["total_predictions"], 0.0
            )
            stats["explosion_rate"] = safe_divide(
                stats["energy_explosions"] + stats["gradient_explosions"],
                stats["total_predictions"],
                0.0,
            )
            stats["ebt_success_rate"] = safe_divide(
                stats["ebt_successes"], stats["total_predictions"], 0.0
            )
        return stats


# Keep remaining classes unchanged but add EBT integration points
class EnhancedEnergyStabilityManager:
    """🛡️ Enhanced stability manager with EBT monitoring."""

    def __init__(self, initial_lr=3e-5, thinking_lr=0.06, policy_memory_manager=None):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.thinking_lr = thinking_lr
        self.current_thinking_lr = thinking_lr
        self.policy_memory_manager = policy_memory_manager

        self.win_rate_window = deque(maxlen=25)
        self.energy_quality_window = deque(maxlen=25)
        self.energy_separation_window = deque(maxlen=25)

        # NEW: EBT-specific monitoring
        self.ebt_success_window = deque(maxlen=25)

        self.min_win_rate = 0.35
        self.min_energy_quality = 2.5
        self.min_energy_separation = 0.015
        self.max_early_stop_rate = 0.75
        self.min_ebt_success_rate = 0.7  # NEW: EBT success rate threshold

        self.lr_decay_factor = 0.85
        self.lr_recovery_factor = 1.02
        self.max_lr_reductions = 2
        self.lr_reductions = 0

        self.last_reset_episode = 0
        self.consecutive_poor_episodes = 0
        self.best_model_state = None
        self.best_win_rate = 0.0
        self.emergency_mode = False

    def update_metrics(
        self,
        win_rate,
        energy_quality,
        energy_separation,
        early_stop_rate,
        ebt_success_rate=1.0,
    ):
        """Enhanced metrics update with EBT monitoring."""
        self.win_rate_window.append(win_rate)
        self.energy_quality_window.append(energy_quality)
        self.energy_separation_window.append(energy_separation)
        self.ebt_success_window.append(ebt_success_rate)  # NEW

        avg_win_rate = safe_mean(list(self.win_rate_window), 0.5)
        avg_energy_quality = safe_mean(list(self.energy_quality_window), 10.0)
        avg_energy_separation = safe_mean(list(self.energy_separation_window), 0.1)
        avg_ebt_success = safe_mean(list(self.ebt_success_window), 1.0)  # NEW

        collapse_indicators = 0

        if avg_win_rate < self.min_win_rate:
            collapse_indicators += 1

        if avg_energy_quality < self.min_energy_quality:
            collapse_indicators += 1

        if abs(avg_energy_separation) < self.min_energy_separation:
            collapse_indicators += 1

        if early_stop_rate > self.max_early_stop_rate:
            collapse_indicators += 1

        # NEW: EBT success rate check
        if avg_ebt_success < self.min_ebt_success_rate:
            collapse_indicators += 1
            print(
                f"⚠️ EBT success rate too low: {avg_ebt_success:.3f} < {self.min_ebt_success_rate}"
            )

        if collapse_indicators >= 2:
            self.consecutive_poor_episodes += 1
            if self.consecutive_poor_episodes >= 2:
                return self._trigger_emergency_protocol()
        else:
            self.consecutive_poor_episodes = 0
            self.emergency_mode = False

        return False

    def _trigger_emergency_protocol(self):
        emergency_triggered = False

        if not self.emergency_mode:
            if (
                self.policy_memory_manager
                and self.policy_memory_manager.should_perform_averaging(0)
            ):
                emergency_triggered = True

            if self.lr_reductions < self.max_lr_reductions:
                self.current_lr *= self.lr_decay_factor
                self.current_thinking_lr *= self.lr_decay_factor
                self.lr_reductions += 1
                emergency_triggered = True

            self.emergency_mode = True
            self.consecutive_poor_episodes = 0

        return emergency_triggered

    def get_current_lrs(self):
        return self.current_lr, self.current_thinking_lr


class EBTEnhancedCheckpointManager:
    """💾 Enhanced checkpoint manager with EBT state tracking."""

    def __init__(self, checkpoint_dir="checkpoints_ebt_enhanced"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.best_checkpoint_path = None
        self.best_win_rate = 0.0
        self.emergency_checkpoint_path = None

        self.peak_checkpoint_path = None
        self.averaging_checkpoints = []

    def save_checkpoint(
        self,
        verifier,
        agent,
        episode,
        win_rate,
        energy_quality,
        is_emergency=False,
        is_peak=False,
        policy_memory_stats=None,
        ebt_stats=None,
    ):
        """Enhanced checkpoint saving with EBT state."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if is_emergency:
            filename = f"emergency_ebt_ep{episode}_{timestamp}.pt"
        elif is_peak:
            filename = f"peak_ebt_ep{episode}_wr{win_rate:.3f}_{timestamp}.pt"
        else:
            filename = f"checkpoint_ebt_ep{episode}_wr{win_rate:.3f}_eq{energy_quality:.1f}_{timestamp}.pt"

        checkpoint_path = self.checkpoint_dir / filename

        checkpoint_data = {
            "episode": episode,
            "win_rate": win_rate,
            "energy_quality": energy_quality,
            "verifier_state_dict": verifier.state_dict(),
            "agent_thinking_stats": agent.get_thinking_stats(),
            "agent_current_thinking_steps": agent.current_thinking_steps,
            "agent_current_thinking_lr": agent.current_thinking_lr,
            "energy_scale": verifier.energy_scale,
            "timestamp": timestamp,
            "is_emergency": is_emergency,
            "is_peak": is_peak,
            "policy_memory_stats": policy_memory_stats or {},
            "ebt_stats": ebt_stats or {},  # NEW: EBT statistics
            "ebt_enabled": verifier.use_ebt,  # NEW: EBT configuration
        }

        try:
            torch.save(checkpoint_data, checkpoint_path)

            if is_emergency:
                self.emergency_checkpoint_path = checkpoint_path
            elif is_peak:
                self.peak_checkpoint_path = checkpoint_path
            elif win_rate > self.best_win_rate:
                self.best_checkpoint_path = checkpoint_path
                self.best_win_rate = win_rate

            return checkpoint_path

        except Exception as e:
            print(f"⚠️ Checkpoint save failed: {e}")
            return None

    def load_checkpoint(self, checkpoint_path, verifier, agent):
        """Enhanced checkpoint loading with EBT state."""
        try:
            checkpoint_data = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )

            verifier.load_state_dict(checkpoint_data["verifier_state_dict"])

            agent.current_thinking_steps = checkpoint_data.get(
                "agent_current_thinking_steps", agent.initial_thinking_steps
            )
            agent.current_thinking_lr = checkpoint_data.get(
                "agent_current_thinking_lr", agent.initial_thinking_lr
            )

            if "energy_scale" in checkpoint_data:
                verifier.energy_scale = checkpoint_data["energy_scale"]

            # NEW: Load EBT configuration
            if "ebt_enabled" in checkpoint_data:
                ebt_enabled = checkpoint_data["ebt_enabled"]
                print(
                    f"🚀 Loaded checkpoint with EBT: {'ENABLED' if ebt_enabled else 'DISABLED'}"
                )

            return checkpoint_data

        except Exception as e:
            print(f"⚠️ Checkpoint load failed: {e}")
            return None


# Utility functions
def verify_ebt_energy_flow(verifier, observation_space, action_space):
    """Verify EBT-enhanced energy flow works correctly."""
    try:
        visual_obs = torch.zeros(1, 3, SCREEN_HEIGHT, SCREEN_WIDTH)
        vector_obs = torch.zeros(1, 5, VECTOR_FEATURE_DIM)
        dummy_obs = {"visual_obs": visual_obs, "vector_obs": vector_obs}
        dummy_action = torch.ones(1, action_space.n) / action_space.n

        # Test without sequence context
        energy1 = verifier(dummy_obs, dummy_action)

        # Test with sequence context
        dummy_sequence = torch.randn(1, EBT_SEQUENCE_LENGTH, VECTOR_FEATURE_DIM)
        energy2 = verifier(dummy_obs, dummy_action, dummy_sequence)

        print(f"   ✅ EBT-Enhanced energy calculation successful")
        print(f"   - Base energy shape: {energy1.shape}, value: {energy1.item():.4f}")
        print(f"   - EBT energy shape: {energy2.shape}, value: {energy2.item():.4f}")
        print(f"   - EBT enabled: {verifier.use_ebt}")

        return True

    except Exception as e:
        print(f"   ❌ EBT energy flow verification failed: {e}")
        return False


def make_ebt_enhanced_env(
    game="StreetFighterIISpecialChampionEdition-Genesis", state="ken_bison_12.state"
):
    """Create EBT-enhanced Street Fighter environment."""
    try:
        env = retro.make(
            game=game, state=state, use_restricted_actions=retro.Actions.DISCRETE
        )
        env = StreetFighterVisionWrapper(env)

        print(f"   ✅ EBT-Enhanced environment created")
        return env

    except Exception as e:
        print(f"   ❌ Environment creation failed: {e}")
        raise


# Export components
__all__ = [
    "StreetFighterVisionWrapper",
    "EBTEnhancedStreetFighterCNN",
    "EBTEnhancedStreetFighterVerifier",
    "EBTEnhancedEnergyBasedAgent",
    "EBTEnhancedFeatureTracker",
    "StreetFighterDiscreteActions",
    "IntelligentRewardCalculator",
    "EBTEnhancedExperienceBuffer",
    "GoldenExperienceBuffer",
    "PolicyMemoryManager",
    "EnhancedEnergyStabilityManager",
    "EBTEnhancedCheckpointManager",
    "EnergyBasedTransformer",
    "EBTSequenceTracker",
    "verify_ebt_energy_flow",
    "make_ebt_enhanced_env",
    "safe_divide",
    "safe_std",
    "safe_mean",
    "sanitize_array",
    "ensure_scalar",
    "safe_comparison",
    "ensure_feature_dimension",
    "VECTOR_FEATURE_DIM",
    "MAX_FIGHT_STEPS",
    "EBT_SEQUENCE_LENGTH",
    "EBT_HIDDEN_DIM",
]

print(f"🚀 EBT-ENHANCED wrapper.py loaded successfully!")
print(f"   - Energy-Based Transformers: INTEGRATED")
print(f"   - Synergistic dual-energy system: ACTIVE")
print(f"   - Sequence modeling: {EBT_SEQUENCE_LENGTH} steps")
print(f"   - Enhanced thinking capabilities: ENABLED")
print(f"🎯 Ready for advanced energy-based learning!")
