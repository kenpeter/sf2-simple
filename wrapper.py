#!/usr/bin/env python3
"""
🚀 OPTIMIZED ENERGY-BASED TRANSFORMER WRAPPER
High-performance Street Fighter RL with maintained rendering capability
Fixes: Slow UI, 0% win rate, action space complexity, reward signal issues
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
    # Remove render_mode if present to avoid conflicts
    if "render_mode" in kwargs:
        del kwargs["render_mode"]
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

# Constants - OPTIMIZED
MAX_HEALTH = 176
SCREEN_WIDTH = 84  # Reduced from 320 for performance
SCREEN_HEIGHT = 84  # Reduced from 224 for performance
VECTOR_FEATURE_DIM = 24  # Reduced from 32 for efficiency
MAX_FIGHT_STEPS = 600  # Reduced from 1200 for faster episodes
EBT_SEQUENCE_LENGTH = 8  # Reduced from 16 for memory efficiency
EBT_HIDDEN_DIM = 128  # Reduced from 256 for performance
EBT_NUM_HEADS = 4  # Reduced from 8 for efficiency
EBT_NUM_LAYERS = 3  # Reduced from 6 for speed

# OPTIMIZED ACTION SPACE - Fixes major training bottleneck
OPTIMIZED_ACTIONS = {
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
    # Essential combinations only
    11: ["DOWN", "RIGHT", "A"],  # Hadoken motion
    12: ["DOWN", "LEFT", "A"],  # Reverse hadoken
    13: ["RIGHT", "DOWN", "A"],  # Dragon punch motion
    14: ["LEFT", "A"],  # Walking punch
    15: ["RIGHT", "A"],  # Walking punch
    16: ["DOWN", "A"],  # Crouching punch
    17: ["UP", "A"],  # Jumping punch
    18: ["LEFT", "X"],  # Walking kick
    19: ["RIGHT", "X"],  # Walking kick
    20: ["DOWN", "X"],  # Crouching kick
}

print(f"🚀 OPTIMIZED ENERGY-BASED TRANSFORMER Configuration:")
print(f"   - Energy-Based Thinking: ENABLED")
print(f"   - Energy-Based Transformers: ENABLED")
print(f"   - Screen Size: {SCREEN_WIDTH}x{SCREEN_HEIGHT} (optimized)")
print(f"   - Action Space: {len(OPTIMIZED_ACTIONS)} (reduced from 4096)")
print(f"   - EBT Sequence Length: {EBT_SEQUENCE_LENGTH}")
print(f"   - Performance Optimizations: ACTIVE")


# Keep all safe operations (unchanged)
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


# OPTIMIZED: Memory-Efficient Frame Processor
class OptimizedFrameProcessor:
    """High-performance frame processing with pre-allocated buffers."""

    def __init__(self, target_size=(SCREEN_WIDTH, SCREEN_HEIGHT)):
        self.target_size = target_size
        # Pre-allocate working memory for zero-copy operations
        self.gray_frame = np.empty((224, 320), dtype=np.uint8)  # Original size
        self.resized_frame = np.empty(target_size, dtype=np.uint8)

    def process(self, frame):
        """Ultra-fast frame processing with memory reuse."""
        # Convert to grayscale first (reduces data by 3x)
        cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY, dst=self.gray_frame)

        # Use INTER_AREA for highest quality downscaling
        cv2.resize(
            self.gray_frame,
            self.target_size,
            dst=self.resized_frame,
            interpolation=cv2.INTER_AREA,
        )

        return self.resized_frame.copy()


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
    """🚀 Energy-Based Transformer for Street Fighter RL"""

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
                EBTTransformerBlock(d_model, num_heads, d_model * 2, dropout)
                for _ in range(num_layers)
            ]
        )

        # Energy function head
        self.energy_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        # Context aggregation for sequence-level energy
        self.context_aggregator = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )

        self.dropout = nn.Dropout(dropout)
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
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, sequence_features, use_causal_mask=True):
        """Forward pass through Energy-Based Transformer."""
        batch_size, seq_length, _ = sequence_features.shape
        device = sequence_features.device

        # Input projection and positional encoding
        x = self.input_projection(sequence_features)
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
        sequence_energies = self.energy_head(x)

        # Aggregate sequence-level energy
        context = self.context_aggregator(x)
        attention_scores = torch.matmul(context, context.transpose(-2, -1)) / math.sqrt(
            self.d_model
        )

        if mask is not None:
            attention_scores = attention_scores.masked_fill(
                mask.squeeze(0).squeeze(0) == 0, -1e9
            )

        attention_weights_agg = F.softmax(attention_scores, dim=-1)
        weighted_energies = torch.matmul(attention_weights_agg, sequence_energies)
        sequence_energy = weighted_energies.mean(dim=1)

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
        action_one_hot = np.zeros(len(OPTIMIZED_ACTIONS), dtype=np.float32)
        action_one_hot[action] = 1.0

        combined_features = np.concatenate(
            [
                state_features.flatten()[: self.feature_dim - 1],
                [reward],
            ]
        )

        combined_features = ensure_feature_dimension(
            combined_features, self.feature_dim
        )
        self.feature_sequence.append(combined_features)
        self.step_count += 1

    def get_sequence_tensor(self, device="cpu"):
        """Get current sequence as tensors for EBT processing."""
        if len(self.feature_sequence) == 0:
            return torch.zeros(1, self.sequence_length, self.feature_dim, device=device)

        feature_list = list(self.feature_sequence)
        while len(feature_list) < self.sequence_length:
            feature_list.insert(0, np.zeros(self.feature_dim, dtype=np.float32))

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


# OPTIMIZED: Dense Reward Calculator - Fixes 0% win rate
class OptimizedRewardCalculator:
    """🎯 Dense reward calculator optimized for learning signals."""

    def __init__(self):
        self.previous_opponent_health = MAX_HEALTH
        self.previous_player_health = MAX_HEALTH
        self.match_started = False

        # Reward weights optimized for learning
        self.damage_reward_scale = 2.0  # Increased for stronger signal
        self.health_penalty_scale = 1.5  # Balanced penalty
        self.winning_bonus = 10.0  # Increased win bonus
        self.time_penalty = 0.05  # Reduced time penalty

        # Combat encouragement
        self.action_bonus_scale = 0.1
        self.last_action = 0
        self.idle_penalty = 0.02

        self.round_won = False
        self.round_lost = False

    def calculate_reward(self, player_health, opponent_health, done, info, action=0):
        """Dense reward calculation optimized for Street Fighter learning."""
        reward = 0.0
        reward_breakdown = {}

        if not self.match_started:
            self.previous_player_health = player_health
            self.previous_opponent_health = opponent_health
            self.match_started = True
            return 0.0, {"initialization": 0.0}

        # Primary reward: Health differential changes
        player_damage_taken = max(0, self.previous_player_health - player_health)
        opponent_damage_dealt = max(0, self.previous_opponent_health - opponent_health)

        # Reward for dealing damage (primary learning signal)
        if opponent_damage_dealt > 0:
            damage_reward = opponent_damage_dealt * self.damage_reward_scale
            reward += damage_reward
            reward_breakdown["damage_dealt"] = damage_reward

        # Penalty for taking damage
        if player_damage_taken > 0:
            damage_penalty = -player_damage_taken * self.health_penalty_scale
            reward += damage_penalty
            reward_breakdown["damage_taken"] = damage_penalty

        # Health advantage bonus (encourages maintaining health lead)
        health_diff = (player_health - opponent_health) / MAX_HEALTH
        if abs(health_diff) > 0.1:
            advantage_bonus = health_diff * 0.5
            reward += advantage_bonus
            reward_breakdown["health_advantage"] = advantage_bonus

        # Combat action encouragement (prevents passive play)
        if action in [5, 6, 7, 8, 9, 10, 11, 12, 13]:  # Attack actions
            action_bonus = self.action_bonus_scale
            reward += action_bonus
            reward_breakdown["action_bonus"] = action_bonus
        elif action == 0:  # Idle action
            idle_penalty = -self.idle_penalty
            reward += idle_penalty
            reward_breakdown["idle_penalty"] = idle_penalty

        # Round completion rewards
        if done:
            if player_health > opponent_health:
                win_bonus = self.winning_bonus
                reward += win_bonus
                reward_breakdown["round_won"] = win_bonus
                self.round_won = True
            elif opponent_health > player_health:
                loss_penalty = -5.0
                reward += loss_penalty
                reward_breakdown["round_lost"] = loss_penalty
                self.round_lost = True
            else:
                draw_penalty = -2.0
                reward += draw_penalty
                reward_breakdown["draw"] = draw_penalty

        # Small time penalty to encourage action
        time_penalty = -self.time_penalty
        reward += time_penalty
        reward_breakdown["time_penalty"] = time_penalty

        # Update previous states
        self.previous_player_health = player_health
        self.previous_opponent_health = opponent_health
        self.last_action = action

        return reward, reward_breakdown

    def reset(self):
        """Reset for new episode."""
        self.previous_opponent_health = MAX_HEALTH
        self.previous_player_health = MAX_HEALTH
        self.match_started = False
        self.round_won = False
        self.round_lost = False
        self.last_action = 0


# ENHANCED: Feature Tracker with EBT Integration
class EBTEnhancedFeatureTracker:
    """📊 Enhanced feature tracker with EBT sequence modeling."""

    def __init__(self, history_length=3):  # Reduced for efficiency
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
        """Get optimized feature vector."""
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
                self.last_action / (len(OPTIMIZED_ACTIONS) - 1),
                min(self.combo_count / 3.0, 1.0),
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


# OPTIMIZED: Action Mapper with Reduced Action Space
class OptimizedStreetFighterActions:
    """🎮 Optimized action mapping with reduced complexity."""

    def __init__(self):
        self.action_map = OPTIMIZED_ACTIONS
        self.n_actions = len(self.action_map)
        # Reverse mapping for button combinations to indices
        self.button_to_index = {tuple(v): k for k, v in self.action_map.items()}

    def get_action(self, action_idx):
        """Convert action index to button combination."""
        return self.action_map.get(action_idx, [])


# OPTIMIZED: Vision Wrapper with Performance Enhancements
class OptimizedStreetFighterVisionWrapper(gym.Wrapper):
    """🥊 High-performance wrapper with rendering capability maintained."""

    def __init__(
        self, env, render_frequency=30
    ):  # Render every 30 steps for performance
        super().__init__(env)

        self.reward_calculator = OptimizedRewardCalculator()
        self.feature_tracker = EBTEnhancedFeatureTracker()
        self.action_mapper = OptimizedStreetFighterActions()
        self.frame_processor = OptimizedFrameProcessor()

        self.render_frequency = render_frequency
        self.step_count = 0

        # Optimized observation space
        visual_space = gym.spaces.Box(
            low=0, high=255, shape=(1, SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.uint8
        )
        vector_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(3, VECTOR_FEATURE_DIM), dtype=np.float32
        )

        self.observation_space = gym.spaces.Dict(
            {"visual_obs": visual_space, "vector_obs": vector_space}
        )

        self.action_space = gym.spaces.Discrete(self.action_mapper.n_actions)
        self.vector_history = deque(maxlen=3)  # Reduced for efficiency

        self.episode_count = 0

        # Health tracking
        self.previous_player_health = MAX_HEALTH
        self.previous_opponent_health = MAX_HEALTH

        print(f"🚀 Optimized StreetFighterVisionWrapper initialized")
        print(f"   - Render frequency: Every {render_frequency} steps")
        print(f"   - Action space size: {self.action_mapper.n_actions}")
        print(f"   - Screen resolution: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")

    def reset(self, **kwargs):
        """Enhanced reset with EBT sequence tracking."""
        obs, info = self.env.reset(**kwargs)

        self.reward_calculator.reset()
        self.feature_tracker.reset()
        self.vector_history.clear()

        # Ensure step_count is properly initialized as integer
        self.step_count = 0

        self.previous_player_health = MAX_HEALTH
        self.previous_opponent_health = MAX_HEALTH

        self.episode_count += 1

        player_health, opponent_health = self._extract_health(info)
        self.feature_tracker.update(player_health, opponent_health, 0, {})

        observation = self._build_observation(obs, info)

        return observation, info

    def step(self, action):
        """Enhanced step with optimized rendering and dense rewards."""
        # Ensure step_count is initialized and is an integer
        if not hasattr(self, "step_count") or not isinstance(self.step_count, int):
            self.step_count = 0

        self.step_count += 1

        button_combination = self.action_mapper.get_action(action)
        retro_action = self._convert_to_retro_action(button_combination)
        obs, reward, done, truncated, info = self.env.step(retro_action)

        # Selective rendering for performance
        if (
            hasattr(self, "step_count")
            and isinstance(self.step_count, int)
            and self.step_count % self.render_frequency == 0
        ):
            try:
                self.env.render()
            except:
                pass  # Continue if rendering fails

        player_health, opponent_health = self._extract_health(info)

        # Episode termination logic
        if (self.previous_player_health > 0 and player_health <= 0) or (
            self.previous_opponent_health > 0 and opponent_health <= 0
        ):
            done = True
        elif self.step_count >= MAX_FIGHT_STEPS:
            done = True

        self.previous_player_health = player_health
        self.previous_opponent_health = opponent_health

        # Dense reward calculation with action information
        intelligent_reward, reward_breakdown = self.reward_calculator.calculate_reward(
            player_health, opponent_health, done, info, action
        )

        # Update feature tracker with EBT sequence tracking
        self.feature_tracker.update(
            player_health, opponent_health, action, reward_breakdown, energy_score=0.0
        )

        observation = self._build_observation(obs, info)

        # Enhanced info with performance metrics
        info.update(
            {
                "player_health": player_health,
                "opponent_health": opponent_health,
                "reward_breakdown": reward_breakdown,
                "intelligent_reward": intelligent_reward,
                "episode_count": self.episode_count,
                "step_count": self.step_count,
                "ebt_sequence_info": self.feature_tracker.get_sequence_info(),
                "action_taken": action,
                "button_combination": button_combination,
            }
        )

        return observation, intelligent_reward, done, truncated, info

    def _extract_health(self, info):
        """Extract health values from game state."""
        player_health = info.get("player_health", MAX_HEALTH)
        opponent_health = info.get("opponent_health", MAX_HEALTH)

        # Try to read from memory if available
        if hasattr(self.env, "data") and hasattr(self.env.data, "memory"):
            try:
                player_health = self.env.data.memory.read_byte(0x8004)
                opponent_health = self.env.data.memory.read_byte(0x8008)
            except:
                pass

        return player_health, opponent_health

    def _convert_to_retro_action(self, button_combination):
        """Convert button combination to retro action."""
        # Create 12-element binary array for SNES controller
        action_array = [0] * 12

        # Map buttons to SNES controller positions
        button_mapping = {
            "B": 0,
            "Y": 1,
            "SELECT": 2,
            "START": 3,
            "UP": 4,
            "DOWN": 5,
            "LEFT": 6,
            "RIGHT": 7,
            "A": 8,
            "X": 9,
            "L": 10,
            "R": 11,
            # Street Fighter specific mappings
            "C": 11,
            "Z": 10,  # Heavy punch/kick
        }

        # Ensure button_combination is a list
        if not isinstance(button_combination, (list, tuple)):
            button_combination = [button_combination] if button_combination else []

        for button in button_combination:
            if button in button_mapping:
                action_array[button_mapping[button]] = 1

        return action_array

    def _build_observation(self, visual_obs, info):
        """Build optimized observation with enhanced visual processing."""
        # Process visual observation with optimized pipeline
        if isinstance(visual_obs, np.ndarray):
            if len(visual_obs.shape) == 3 and visual_obs.shape[2] == 3:
                # RGB image - process efficiently
                processed_frame = self.frame_processor.process(visual_obs)
                visual_obs = processed_frame.reshape(1, SCREEN_HEIGHT, SCREEN_WIDTH)
            elif len(visual_obs.shape) == 3:
                # Already processed
                visual_obs = visual_obs[0:1]  # Take first channel if multiple
            else:
                # Unexpected format - create dummy
                visual_obs = np.zeros((1, SCREEN_HEIGHT, SCREEN_WIDTH), dtype=np.uint8)

        # Build vector observation
        vector_features = self.feature_tracker.get_features()
        self.vector_history.append(vector_features)

        while len(self.vector_history) < 3:
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


# Keep remaining classes with optimizations


class GoldenExperienceBuffer:
    """🏆 Golden buffer with enhanced EBT integration."""

    def __init__(self, capacity=500):  # Reduced capacity for efficiency
        self.capacity = capacity
        self.experiences = deque(maxlen=capacity)
        self.min_quality_for_golden = 0.7  # Increased threshold
        self.peak_win_rate_threshold = 0.2  # Lowered threshold
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
        self, capacity=20000, quality_threshold=0.4, golden_buffer_capacity=500
    ):
        self.capacity = capacity
        self.quality_threshold = quality_threshold

        self.good_experiences = deque(maxlen=capacity // 2)
        self.bad_experiences = deque(maxlen=capacity // 2)
        self.sequence_experiences = deque(maxlen=capacity // 4)

        self.golden_buffer = GoldenExperienceBuffer(capacity=golden_buffer_capacity)

        self.quality_scores = deque(maxlen=500)
        self.total_added = 0

        self.adjustment_rate = 0.03
        self.threshold_adjustment_frequency = 20
        self.threshold_adjustments = 0

        print(f"🚀 EBT-Enhanced Experience Buffer initialized")
        print(f"   - Quality threshold: {quality_threshold}")
        print(f"   - Total capacity: {capacity:,}")

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
        self, batch_size, golden_ratio=0.1, sequence_ratio=0.1
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

        if len(self.quality_scores) < 50:
            return

        total_size = len(self.good_experiences) + len(self.bad_experiences)
        current_good_ratio = len(self.good_experiences) / max(1, total_size)

        if current_good_ratio < target_good_ratio - 0.1:
            self.quality_threshold *= 1 - self.adjustment_rate
            self.threshold_adjustments += 1
            print(f"📉 Lowered quality threshold to {self.quality_threshold:.3f}")
        elif current_good_ratio > target_good_ratio + 0.1:
            self.quality_threshold *= 1 + self.adjustment_rate
            self.threshold_adjustments += 1
            print(f"📈 Raised quality threshold to {self.quality_threshold:.3f}")

        self.quality_threshold = max(0.3, min(0.8, self.quality_threshold))


# Keep PolicyMemoryManager unchanged but add performance optimizations
class PolicyMemoryManager:
    """🧠 Optimized policy memory manager."""

    def __init__(self, performance_drop_threshold=0.1, averaging_weight=0.7):
        self.performance_drop_threshold = performance_drop_threshold
        self.averaging_weight = averaging_weight

        self.peak_win_rate = 0.0
        self.peak_checkpoint_state = None
        self.episodes_since_peak = 0
        self.performance_drop_detected = False

        self.peak_lr = None
        self.lr_reduction_factor = 0.8
        self.min_lr = 1e-6

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
                and self.episodes_since_peak > 8  # Reduced from 10
            ):
                self.performance_drop_detected = True
                performance_drop = True

        return performance_improved, performance_drop

    def should_perform_averaging(self, current_episode):
        return (
            self.performance_drop_detected
            and self.peak_checkpoint_state is not None
            and current_episode - self.last_averaging_episode > 3  # Reduced from 5
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


# OPTIMIZED: CNN with Performance Enhancements
class OptimizedStreetFighterCNN(nn.Module):
    """🛡️ High-performance CNN optimized for Street Fighter RL."""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 128):
        super().__init__()

        visual_space = observation_space["visual_obs"]
        vector_space = observation_space["vector_obs"]
        n_input_channels = visual_space.shape[0]
        seq_length, vector_feature_count = vector_space.shape

        # Optimized CNN architecture for smaller inputs
        self.visual_cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy_visual = torch.zeros(
                1, n_input_channels, visual_space.shape[1], visual_space.shape[2]
            )
            visual_output_size = self.visual_cnn(dummy_visual).shape[1]

        # Optimized vector processing
        self.vector_embed = nn.Linear(vector_feature_count, 32)
        self.vector_norm = nn.LayerNorm(32)
        self.vector_dropout = nn.Dropout(0.1)
        self.vector_gru = nn.GRU(32, 32, batch_first=True, dropout=0.1)
        self.vector_final = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
        )

        # Optimized fusion layer
        fusion_input_size = visual_output_size + 16
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Dropout(0.1),
            nn.Linear(256, features_dim),
            nn.ReLU(inplace=True),
        )

        # EBT preparation layer
        self.ebt_projection = nn.Linear(features_dim, EBT_HIDDEN_DIM)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(
        self, observations: Dict[str, torch.Tensor], return_ebt_features=False
    ) -> torch.Tensor:
        visual_obs = observations["visual_obs"]
        vector_obs = observations["vector_obs"]
        device = next(self.parameters()).device

        visual_obs = visual_obs.float().to(device)
        vector_obs = vector_obs.float().to(device)

        # Input validation and normalization
        visual_obs = torch.clamp(visual_obs / 255.0, 0.0, 1.0)
        vector_obs = torch.clamp(vector_obs, -5.0, 5.0)

        # Visual processing
        visual_features = self.visual_cnn(visual_obs)

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
        output = torch.clamp(output, -10.0, 10.0)

        if return_ebt_features:
            ebt_features = self.ebt_projection(output)
            ebt_features = torch.clamp(ebt_features, -8.0, 8.0)
            return output, ebt_features

        return output


# OPTIMIZED: Verifier with EBT Integration
class OptimizedStreetFighterVerifier(nn.Module):
    """🛡️ High-performance verifier with Energy-Based Transformer integration."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        features_dim: int = 128,
        use_ebt: bool = True,
    ):
        super().__init__()

        self.observation_space = observation_space
        self.action_space = action_space
        self.features_dim = features_dim
        self.action_dim = (
            action_space.n if hasattr(action_space, "n") else len(OPTIMIZED_ACTIONS)
        )
        self.use_ebt = use_ebt

        # Optimized feature extractor
        self.features_extractor = OptimizedStreetFighterCNN(
            observation_space, features_dim
        )

        # Action embedding
        self.action_embed = nn.Sequential(
            nn.Linear(self.action_dim, 32),
            nn.ReLU(inplace=True),
            nn.LayerNorm(32),
        )

        # Energy-Based Transformer
        if self.use_ebt:
            ebt_input_dim = VECTOR_FEATURE_DIM
            self.ebt = EnergyBasedTransformer(
                input_dim=ebt_input_dim,
                d_model=EBT_HIDDEN_DIM,
                num_heads=EBT_NUM_HEADS,
                num_layers=EBT_NUM_LAYERS,
                max_seq_length=EBT_SEQUENCE_LENGTH,
            )

            # EBT-enhanced energy network
            self.energy_net = nn.Sequential(
                nn.Linear(EBT_HIDDEN_DIM + features_dim + 32, 128),
                nn.ReLU(inplace=True),
                nn.LayerNorm(128),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
            )
        else:
            # Standard energy network
            self.energy_net = nn.Sequential(
                nn.Linear(features_dim + 32, 128),
                nn.ReLU(inplace=True),
                nn.LayerNorm(128),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
            )

        self.energy_scale = 0.5
        self.energy_clamp_min = -5.0
        self.energy_clamp_max = 5.0

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(
        self,
        context: torch.Tensor,
        candidate_action: torch.Tensor,
        sequence_context: torch.Tensor = None,
    ) -> torch.Tensor:
        """Enhanced forward pass with EBT integration."""
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

        # Input validation
        context_features = torch.clamp(context_features, -10.0, 10.0)
        candidate_action = torch.clamp(candidate_action, 0.0, 1.0)

        # Action embedding
        action_embedded = self.action_embed(candidate_action)

        # EBT Processing
        ebt_output = None
        if self.use_ebt and ebt_features is not None:
            batch_size = ebt_features.shape[0]

            if sequence_context is not None and sequence_context.shape[1] > 1:
                ebt_input = sequence_context.to(device)
            else:
                ebt_input = torch.zeros(
                    batch_size, EBT_SEQUENCE_LENGTH, VECTOR_FEATURE_DIM, device=device
                )

            try:
                ebt_result = self.ebt(ebt_input, use_causal_mask=True)
                ebt_sequence_energy = ebt_result["sequence_energy"]
                ebt_representations = ebt_result["sequence_representations"]
                ebt_output = ebt_representations[:, -1, :]
                ebt_energy_contribution = ebt_sequence_energy.squeeze(-1)
            except Exception as e:
                ebt_output = torch.zeros(batch_size, EBT_HIDDEN_DIM, device=device)
                ebt_energy_contribution = torch.zeros(batch_size, device=device)

        # Combine inputs for energy calculation
        if ebt_output is not None:
            combined_input = torch.cat(
                [context_features, action_embedded, ebt_output], dim=-1
            )
        else:
            combined_input = torch.cat([context_features, action_embedded], dim=-1)

        # Calculate energy
        raw_energy = self.energy_net(combined_input)
        energy = raw_energy * self.energy_scale

        # Add EBT energy contribution if available
        if self.use_ebt and "ebt_energy_contribution" in locals():
            ebt_weight = 0.2
            energy = energy.squeeze(-1) + ebt_weight * ebt_energy_contribution
            energy = energy.unsqueeze(-1)

        # Final clamping
        energy = torch.clamp(energy, self.energy_clamp_min, self.energy_clamp_max)

        return energy


# OPTIMIZED: Agent with EBT Integration
class OptimizedEnergyBasedAgent:
    """🛡️ High-performance agent with Energy-Based Transformer integration."""

    def __init__(
        self,
        verifier: OptimizedStreetFighterVerifier,
        thinking_steps: int = 2,  # Reduced for performance
        thinking_lr: float = 0.03,
        noise_scale: float = 0.01,
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

        self.gradient_clip = 0.5
        self.early_stop_patience = 1  # Reduced for performance
        self.min_energy_improvement = 5e-4

        self.max_thinking_steps = 3
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

        self.recent_performance = deque(maxlen=50)

    def predict(
        self,
        observations: Dict[str, torch.Tensor],
        deterministic: bool = False,
        sequence_context: torch.Tensor = None,
    ) -> Tuple[int, Dict]:
        """Optimized prediction with EBT integration."""
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
                return 0, {"error": "initial_energy_failed", "ebt_error": str(e)}

        # Optimized thinking loop
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
                if torch.any(torch.abs(energy) > 6.0):
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
                    recent_improvement = abs(
                        energy_history[-self.early_stop_patience - 1]
                        - energy_history[-1]
                    )
                    if recent_improvement < self.min_energy_improvement:
                        early_stopped = True
                        break

                steps_taken += 1

            except Exception as e:
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


# Keep remaining stability and checkpoint managers with optimizations
class EnhancedEnergyStabilityManager:
    """🛡️ Enhanced stability manager with EBT monitoring."""

    def __init__(self, initial_lr=3e-4, thinking_lr=0.03, policy_memory_manager=None):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.thinking_lr = thinking_lr
        self.current_thinking_lr = thinking_lr
        self.policy_memory_manager = policy_memory_manager

        self.win_rate_window = deque(maxlen=15)  # Reduced window
        self.energy_quality_window = deque(maxlen=15)
        self.energy_separation_window = deque(maxlen=15)
        self.ebt_success_window = deque(maxlen=15)

        self.min_win_rate = 0.2  # Lowered threshold
        self.min_energy_quality = 1.5  # Lowered threshold
        self.min_energy_separation = 0.01
        self.max_early_stop_rate = 0.8
        self.min_ebt_success_rate = 0.6

        self.lr_decay_factor = 0.9
        self.lr_recovery_factor = 1.01
        self.max_lr_reductions = 3
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
        self.ebt_success_window.append(ebt_success_rate)

        avg_win_rate = safe_mean(list(self.win_rate_window), 0.5)
        avg_energy_quality = safe_mean(list(self.energy_quality_window), 10.0)
        avg_energy_separation = safe_mean(list(self.energy_separation_window), 0.1)
        avg_ebt_success = safe_mean(list(self.ebt_success_window), 1.0)

        collapse_indicators = 0

        if avg_win_rate < self.min_win_rate:
            collapse_indicators += 1

        if avg_energy_quality < self.min_energy_quality:
            collapse_indicators += 1

        if abs(avg_energy_separation) < self.min_energy_separation:
            collapse_indicators += 1

        if early_stop_rate > self.max_early_stop_rate:
            collapse_indicators += 1

        if avg_ebt_success < self.min_ebt_success_rate:
            collapse_indicators += 1

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

    def __init__(self, checkpoint_dir="checkpoints_optimized"):
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
            "ebt_stats": ebt_stats or {},
            "ebt_enabled": verifier.use_ebt,
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
        visual_obs = torch.zeros(1, 1, SCREEN_HEIGHT, SCREEN_WIDTH)
        vector_obs = torch.zeros(1, 3, VECTOR_FEATURE_DIM)
        dummy_obs = {"visual_obs": visual_obs, "vector_obs": vector_obs}
        dummy_action = torch.ones(1, action_space.n) / action_space.n

        # Test without sequence context
        energy1 = verifier(dummy_obs, dummy_action)

        # Test with sequence context
        dummy_sequence = torch.randn(1, EBT_SEQUENCE_LENGTH, VECTOR_FEATURE_DIM)
        energy2 = verifier(dummy_obs, dummy_action, dummy_sequence)

        print(f"   ✅ Optimized EBT energy calculation successful")
        print(f"   - Base energy shape: {energy1.shape}, value: {energy1.item():.4f}")
        print(f"   - EBT energy shape: {energy2.shape}, value: {energy2.item():.4f}")
        print(f"   - EBT enabled: {verifier.use_ebt}")

        return True

    except Exception as e:
        print(f"   ❌ EBT energy flow verification failed: {e}")
        return False


def make_optimized_sf_env(
    game="StreetFighterIISpecialChampionEdition-Genesis", state="ken_bison_12.state"
):
    """Create optimized Street Fighter environment."""
    try:
        env = retro.make(
            game=game, state=state, use_restricted_actions=retro.Actions.DISCRETE
        )
        env = OptimizedStreetFighterVisionWrapper(env, render_frequency=30)

        print(f"   ✅ Optimized environment created")
        print(f"   - Render frequency: Every 30 steps")
        print(f"   - Action space: {env.action_space.n} actions")
        print(f"   - Visual obs: {env.observation_space['visual_obs'].shape}")

        return env

    except Exception as e:
        print(f"   ❌ Environment creation failed: {e}")
        raise


# Export components
__all__ = [
    "OptimizedStreetFighterVisionWrapper",
    "OptimizedStreetFighterCNN",
    "OptimizedStreetFighterVerifier",
    "OptimizedEnergyBasedAgent",
    "EBTEnhancedFeatureTracker",
    "OptimizedStreetFighterActions",
    "OptimizedRewardCalculator",
    "EBTEnhancedExperienceBuffer",
    "GoldenExperienceBuffer",
    "PolicyMemoryManager",
    "EnhancedEnergyStabilityManager",
    "EBTEnhancedCheckpointManager",
    "EnergyBasedTransformer",
    "EBTSequenceTracker",
    "OptimizedFrameProcessor",
    "verify_ebt_energy_flow",
    "make_optimized_sf_env",
    "safe_divide",
    "safe_std",
    "safe_mean",
    "sanitize_array",
    "ensure_scalar",
    "ensure_feature_dimension",
    "VECTOR_FEATURE_DIM",
    "MAX_FIGHT_STEPS",
    "EBT_SEQUENCE_LENGTH",
    "EBT_HIDDEN_DIM",
    "OPTIMIZED_ACTIONS",
    "SCREEN_WIDTH",
    "SCREEN_HEIGHT",
]

print(f"🚀 OPTIMIZED wrapper.py loaded successfully!")
print(f"   - Performance optimizations: ACTIVE")
print(f"   - Dense rewards: ENABLED")
print(f"   - Reduced action space: {len(OPTIMIZED_ACTIONS)} actions")
print(f"   - Optimized rendering: Every 30 steps")
print(f"   - Memory efficiency: ENHANCED")
print(f"🎯 Ready for high-performance Street Fighter RL training!")
