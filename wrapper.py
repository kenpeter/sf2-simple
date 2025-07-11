#!/usr/bin/env python3
"""
Clean Multi-Round Street Fighter Wrapper
Handles up to 3 rounds per match with proper win detection
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, SGD
import numpy as np
import gymnasium as gym
from collections import deque
from gymnasium import spaces
from typing import Dict, Tuple, List, Any
import logging
import os
from datetime import datetime
import retro
from pathlib import Path
import zipfile

# Constants
MAX_HEALTH = 176
SCREEN_WIDTH = 320
SCREEN_HEIGHT = 224
VECTOR_FEATURE_DIM = 45  # Base features without bait.py

print("🧠 Street Fighter AI Wrapper Configuration:")
print(f"   - Vector features: {VECTOR_FEATURE_DIM}")
print("   - Training paradigm: Multi-Round (max 3 rounds)")

# Fix for retro.make TypeError
_original_retro_make = retro.make


def _patched_retro_make(game, state=None, **kwargs):
    if not state:
        state = "ken_bison_12.state"
    return _original_retro_make(game=game, state=state, **kwargs)


retro.make = _patched_retro_make

# Configure logging
os.makedirs("logs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
log_filename = f'logs/sf_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename)],
)
logger = logging.getLogger(__name__)


# Safe utility functions
def safe_divide(numerator, denominator, default=0.0):
    try:
        if denominator == 0 or not np.isfinite(denominator):
            return default
        result = numerator / denominator
        return result if np.isfinite(result) else default
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


def ensure_scalar(value, default=0.0):
    if value is None:
        return default
    try:
        if isinstance(value, np.ndarray):
            return float(value.item()) if value.size == 1 else float(value.flat[0])
        elif isinstance(value, (list, tuple)):
            return float(value[0]) if len(value) > 0 else default
        else:
            return float(value)
    except:
        return default


class StreetFighterDiscreteActions:
    """Handles discrete action space conversion for Street Fighter."""

    def __init__(self):
        self.num_buttons = 12
        self.action_combinations = [
            [],  # 0: No action
            [6],  # 1: Light Punch
            [7],  # 2: Medium Punch
            [4],  # 3: Light Kick
            [5],  # 4: Medium Kick
            [6, 4],  # 5: Light Punch + Light Kick
            [7, 4],  # 6: Medium Punch + Light Kick
            [6, 5],  # 7: Light Punch + Medium Kick
            [7, 5],  # 8: Medium Punch + Medium Kick
            [1],  # 9: Down
            [0],  # 10: Up
            [1, 6],  # 11: Down + Light Punch
            [1, 7],  # 12: Down + Medium Punch
            [0, 6],  # 13: Up + Light Punch
            [0, 7],  # 14: Up + Medium Punch
            [9],  # 15: Right
            [8],  # 16: Left
            [9, 6],  # 17: Right + Light Punch
            [9, 7],  # 18: Right + Medium Punch
            [8, 6],  # 19: Left + Light Punch
            [8, 7],  # 20: Left + Medium Punch
            [10],  # 21: Heavy Punch
            [11],  # 22: Heavy Kick
            [10, 6],  # 23: Heavy Punch + Light Punch
            [10, 7],  # 24: Heavy Punch + Medium Punch
            [11, 6],  # 25: Heavy Kick + Light Punch
            [11, 7],  # 26: Heavy Kick + Medium Punch
            [1, 4],  # 27: Down + Light Kick
            [0, 4],  # 28: Up + Light Kick
            [9, 4],  # 29: Right + Light Kick
            [8, 4],  # 30: Left + Light Kick
            [10, 4],  # 31: Heavy Punch + Light Kick
            [11, 4],  # 32: Heavy Kick + Light Kick
            [1, 5],  # 33: Down + Medium Kick
            [0, 5],  # 34: Up + Medium Kick
            [9, 5],  # 35: Right + Medium Kick
            [8, 5],  # 36: Left + Medium Kick
            [10, 5],  # 37: Heavy Punch + Medium Kick
            [11, 5],  # 38: Heavy Kick + Medium Kick
            [5, 7],  # 39: Medium Kick + Medium Punch
            [5, 6],  # 40: Medium Kick + Light Punch
            [5, 7, 1],  # 41: Medium Kick + Medium Punch + Down
            [5, 7, 9],  # 42: Medium Kick + Medium Punch + Right
            [5, 7, 10],  # 43: Medium Kick + Medium Punch + Heavy Punch
            [5, 6, 1],  # 44: Medium Kick + Light Punch + Down
            [5, 6, 9],  # 45: Medium Kick + Light Punch + Right
            [5, 6, 10],  # 46: Medium Kick + Light Punch + Heavy Punch
            [5, 7, 0],  # 47: Medium Kick + Medium Punch + Up
            [5, 7, 8],  # 48: Medium Kick + Medium Punch + Left
            [5, 7, 11],  # 49: Medium Kick + Medium Punch + Heavy Kick
            [4, 5],  # 50: Light Kick + Medium Kick
            [7, 1],  # 51: Medium Punch + Down
            [7, 9],  # 52: Medium Punch + Right
            [7, 10],  # 53: Medium Punch + Heavy Punch
            [6],  # 54: Light Punch (duplicate)
            [6, 5],  # 55: Light Punch + Medium Kick
            [4, 6],  # 56: Light Kick + Light Punch
        ]
        self.num_actions = len(self.action_combinations)

    def discrete_to_multibinary(self, action_index: int):
        action = np.zeros(self.num_buttons, dtype=np.uint8)
        if 0 <= action_index < self.num_actions:
            for button_idx in self.action_combinations[action_index]:
                action[button_idx] = 1
        return action

    def get_button_features(self, action_index: int):
        return self.discrete_to_multibinary(action_index).astype(np.float32)


class StrategicFeatureTracker:
    """Basic strategic feature tracker."""

    def __init__(self, history_length=8):
        self.history_length = history_length
        self.player_health_history = deque(maxlen=history_length)
        self.opponent_health_history = deque(maxlen=history_length)

    def update(self, info, button_features):
        player_health = info.get("health", info.get("player_health", 176))
        opponent_health = info.get("enemy_health", info.get("opponent_health", 176))

        self.player_health_history.append(player_health)
        self.opponent_health_history.append(opponent_health)

        features = np.zeros(VECTOR_FEATURE_DIM, dtype=np.float32)
        features[0] = player_health / 176.0
        features[1] = opponent_health / 176.0
        features[2] = (player_health - opponent_health) / 176.0

        if len(button_features) >= 12:
            features[3:15] = button_features[:12]

        return features


class MultiRoundStreetFighterWrapper(gym.Wrapper):
    """Street Fighter wrapper for multi-round matches (max 3 rounds)."""

    def __init__(self, env, frame_stack=8, rendering=False, max_rounds=3):
        super().__init__(env)
        self.frame_stack = frame_stack
        self.rendering = rendering
        self.max_rounds = max_rounds

        obs, _ = env.reset()
        self.target_size = obs.shape[:2]
        self.discrete_actions = StreetFighterDiscreteActions()
        self.action_space = spaces.Discrete(self.discrete_actions.num_actions)
        self.observation_space = spaces.Dict(
            {
                "visual_obs": spaces.Box(
                    low=0,
                    high=255,
                    shape=(3 * frame_stack, *self.target_size),
                    dtype=np.uint8,
                ),
                "vector_obs": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(frame_stack, VECTOR_FEATURE_DIM),
                    dtype=np.float32,
                ),
            }
        )

        self.frame_buffer = deque(maxlen=frame_stack)
        self.vector_features_history = deque(maxlen=frame_stack)
        self.strategic_tracker = StrategicFeatureTracker(history_length=frame_stack)
        self.full_hp = 176
        self.max_episode_steps = 18000

        self.reset_match_state()

    def reset_match_state(self):
        """Reset all match tracking."""
        self.match_state = {
            "current_round": 1,
            "player_rounds_won": 0,
            "opponent_rounds_won": 0,
            "match_winner": None,
            "match_finished": False,
            "round_results": [],
            "total_steps": 0,
            "total_reward": 0.0,
        }

        self.round_state = {
            "round_winner": None,
            "round_finished": False,
            "initial_player_health": self.full_hp,
            "initial_opponent_health": self.full_hp,
            "round_reward": 0.0,
            "round_steps": 0,
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.reset_match_state()

        self.round_state["initial_player_health"] = info.get("health", self.full_hp)
        self.round_state["initial_opponent_health"] = info.get(
            "enemy_health", self.full_hp
        )

        processed_frame = self._preprocess_frame(obs)
        initial_vector_features = np.zeros(VECTOR_FEATURE_DIM, dtype=np.float32)
        self.frame_buffer.clear()
        self.vector_features_history.clear()
        for _ in range(self.frame_stack):
            self.frame_buffer.append(processed_frame)
            self.vector_features_history.append(initial_vector_features)

        info.update(self._get_match_info())
        return self._get_observation(), info

    def step(self, discrete_action):
        multibinary_action = self.discrete_actions.discrete_to_multibinary(
            discrete_action
        )
        obs, reward, done, truncated, info = self.env.step(multibinary_action)

        if self.rendering:
            self.env.render()

        self.round_state["round_steps"] += 1
        self.round_state["round_reward"] += reward
        self.match_state["total_steps"] += 1
        self.match_state["total_reward"] += reward

        current_player_health = info.get("health", self.full_hp)
        current_opponent_health = info.get("enemy_health", self.full_hp)

        # Check for round completion
        if not self.round_state["round_finished"]:
            round_winner = self._check_round_winner(
                current_player_health,
                current_opponent_health,
                done,
                truncated,
                reward,
                info,
            )

            if round_winner is not None:
                self._complete_round(round_winner)

        # Check for match completion
        if not self.match_state["match_finished"]:
            self._check_match_completion(done, truncated)

        self.frame_buffer.append(self._preprocess_frame(obs))
        button_features = self.discrete_actions.get_button_features(discrete_action)
        vector_features = self.strategic_tracker.update(info, button_features)
        self.vector_features_history.append(vector_features)

        info.update(
            {
                "player_health": current_player_health,
                "opponent_health": current_opponent_health,
                **self._get_match_info(),
            }
        )

        match_done = self.match_state["match_finished"]
        return self._get_observation(), reward, match_done, truncated, info

    def _check_round_winner(
        self, player_health, opponent_health, done, truncated, reward, info
    ):
        """Determine round winner."""
        # Direct KO detection
        if player_health <= 0 and opponent_health <= 0:
            return "draw"
        elif player_health <= 0:
            return "opponent"
        elif opponent_health <= 0:
            return "player"

        # Check game state for round wins
        player_wins = info.get("matches_won", 0)
        opponent_wins = info.get("enemy_matches_won", 0)

        total_tracked_rounds = len(self.match_state["round_results"])
        total_game_rounds = player_wins + opponent_wins

        if total_game_rounds > total_tracked_rounds:
            if player_wins > self.match_state["player_rounds_won"]:
                return "player"
            elif opponent_wins > self.match_state["opponent_rounds_won"]:
                return "opponent"

        # Episode termination with health comparison
        if done or truncated:
            health_diff = player_health - opponent_health
            if abs(health_diff) <= 5:
                return "draw"
            elif health_diff > 0:
                return "player"
            else:
                return "opponent"

        return None

    def _complete_round(self, round_winner):
        """Complete the current round."""
        self.round_state["round_winner"] = round_winner
        self.round_state["round_finished"] = True
        self.match_state["round_results"].append(round_winner)

        if round_winner == "player":
            self.match_state["player_rounds_won"] += 1
        elif round_winner == "opponent":
            self.match_state["opponent_rounds_won"] += 1

    def _check_match_completion(self, done, truncated):
        """Check if match is complete."""
        player_wins = self.match_state["player_rounds_won"]
        opponent_wins = self.match_state["opponent_rounds_won"]
        rounds_played = len(self.match_state["round_results"])

        if player_wins >= 2:
            self.match_state["match_winner"] = "player"
            self.match_state["match_finished"] = True
        elif opponent_wins >= 2:
            self.match_state["match_winner"] = "opponent"
            self.match_state["match_finished"] = True
        elif rounds_played >= self.max_rounds:
            if player_wins > opponent_wins:
                self.match_state["match_winner"] = "player"
            elif opponent_wins > player_wins:
                self.match_state["match_winner"] = "opponent"
            else:
                self.match_state["match_winner"] = "draw"
            self.match_state["match_finished"] = True

    def _get_match_info(self):
        """Get match information."""
        return {
            "current_round": self.match_state["current_round"],
            "player_rounds_won": self.match_state["player_rounds_won"],
            "opponent_rounds_won": self.match_state["opponent_rounds_won"],
            "match_winner": self.match_state["match_winner"],
            "match_finished": self.match_state["match_finished"],
            "round_results": self.match_state["round_results"].copy(),
            "total_steps": self.match_state["total_steps"],
            "total_reward": self.match_state["total_reward"],
            "round_winner": self.round_state["round_winner"],
            "round_finished": self.round_state["round_finished"],
            "round_steps": self.round_state["round_steps"],
            "round_reward": self.round_state["round_reward"],
            "max_rounds": self.max_rounds,
        }

    def _get_observation(self):
        visual_obs = np.concatenate(list(self.frame_buffer), axis=2).transpose(2, 0, 1)
        vector_obs = np.stack(list(self.vector_features_history))
        return {"visual_obs": visual_obs, "vector_obs": vector_obs}

    def _preprocess_frame(self, frame):
        if frame.shape[:2] != self.target_size:
            return cv2.resize(frame, (self.target_size[1], self.target_size[0]))
        return frame


# Neural Network Models
class EnergyBasedStreetFighterCNN(nn.Module):
    """CNN feature extractor for Energy-Based Transformer."""

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__()
        visual_space = observation_space["visual_obs"]
        vector_space = observation_space["vector_obs"]
        n_input_channels = visual_space.shape[0]

        if len(vector_space.shape) == 2:
            _, vector_feature_count = vector_space.shape
        else:
            vector_feature_count = vector_space.shape[0]

        self.visual_cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout2d(0.05),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.05),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((3, 4)),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy_visual = torch.zeros(1, n_input_channels, *visual_space.shape[1:])
            visual_output_size = self.visual_cnn(dummy_visual).shape[1]

        self.vector_embed = nn.Linear(vector_feature_count, 64)
        self.vector_norm = nn.LayerNorm(64)
        self.vector_gru = nn.GRU(64, 64, batch_first=True)
        self.vector_final = nn.Linear(64, 32)

        fusion_input_size = visual_output_size + 32
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_size, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        device = next(self.parameters()).device
        visual_obs = torch.nan_to_num(
            observations["visual_obs"].float().to(device) / 255.0
        )
        vector_obs = torch.nan_to_num(observations["vector_obs"].float().to(device))

        visual_features = self.visual_cnn(visual_obs)

        if vector_obs.dim() == 4:
            batch_size, seq_len = vector_obs.shape[:2]
            vector_obs = vector_obs.view(batch_size, seq_len, -1)
        elif vector_obs.dim() == 3:
            pass
        elif vector_obs.dim() == 2:
            vector_obs = vector_obs.unsqueeze(1)
        else:
            raise ValueError(f"Unexpected vector_obs dimensions: {vector_obs.shape}")

        if vector_obs.shape[-1] != self.vector_embed.in_features:
            expected_features = self.vector_embed.in_features
            current_features = vector_obs.shape[-1]
            if current_features < expected_features:
                padding = torch.zeros(
                    *vector_obs.shape[:-1],
                    expected_features - current_features,
                    device=device,
                )
                vector_obs = torch.cat([vector_obs, padding], dim=-1)
            elif current_features > expected_features:
                vector_obs = vector_obs[..., :expected_features]

        vector_embedded = F.relu(self.vector_norm(self.vector_embed(vector_obs)))
        gru_output, _ = self.vector_gru(vector_embedded)
        vector_features = F.relu(self.vector_final(gru_output[:, -1, :]))

        combined_features = torch.cat([visual_features, vector_features], dim=1)
        return self.fusion(combined_features)


class EnergyBasedStreetFighterVerifier(nn.Module):
    """Energy-Based Transformer Verifier."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        features_dim: int = 256,
    ):
        super().__init__()
        self.action_dim = action_space.n
        self.features_extractor = EnergyBasedStreetFighterCNN(
            observation_space, features_dim
        )
        self.action_embed = nn.Sequential(
            nn.Linear(self.action_dim, 64), nn.ReLU(), nn.LayerNorm(64)
        )
        self.energy_net = nn.Sequential(
            nn.Linear(features_dim + 64, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 1),
        )
        self.energy_scale = 0.01
        self.energy_clamp_min = -2.0
        self.energy_clamp_max = 2.0

    def forward(
        self, context: Dict[str, torch.Tensor], candidate_action: torch.Tensor
    ) -> torch.Tensor:
        context_features = self.features_extractor(context)
        action_embedded = self.action_embed(candidate_action)
        combined = torch.cat([context_features, action_embedded], dim=-1)
        energy = self.energy_net(combined)
        return torch.clamp(
            energy * self.energy_scale, self.energy_clamp_min, self.energy_clamp_max
        )


class StabilizedEnergyBasedAgent:
    """Energy-Based Agent with thinking process."""

    def __init__(
        self,
        verifier,
        thinking_steps: int = 2,
        thinking_lr: float = 0.05,
        noise_scale: float = 0.01,
    ):
        self.verifier = verifier
        self.current_thinking_steps = thinking_steps
        self.current_thinking_lr = thinking_lr
        self.noise_scale = noise_scale
        self.action_dim = verifier.action_dim
        self.gradient_clip = 0.5

    def predict(
        self, observations: Dict[str, Any], deterministic: bool = False
    ) -> Tuple[int, Dict]:
        device = next(self.verifier.parameters()).device

        obs_device = {}
        for k, v in observations.items():
            if isinstance(v, torch.Tensor):
                tensor = v
            else:
                tensor = torch.from_numpy(v)

            tensor = tensor.to(device)
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            elif tensor.dim() == 2:
                tensor = tensor.unsqueeze(0)

            obs_device[k] = tensor

        action_logits = (
            torch.randn(1, self.action_dim, device=device) * self.noise_scale
        )
        action_logits.requires_grad_(True)
        optimizer = SGD([action_logits], lr=self.current_thinking_lr)

        for step in range(self.current_thinking_steps):
            optimizer.zero_grad()
            candidate_action_probs = F.softmax(action_logits, dim=-1)
            try:
                energy = self.verifier(obs_device, candidate_action_probs)
                energy.sum().backward()
                torch.nn.utils.clip_grad_norm_([action_logits], self.gradient_clip)
                optimizer.step()
            except Exception as e:
                break

        with torch.no_grad():
            final_probs = F.softmax(action_logits, dim=-1)
            action_idx = (
                torch.argmax(final_probs, dim=-1)
                if deterministic
                else torch.multinomial(final_probs, 1)
            )

        return action_idx.item(), {}


# Training Components
class EnergyStabilityManager:
    """Manages training stability."""

    def __init__(self, initial_lr=1e-4, thinking_lr=0.1):
        self.initial_lr, self.current_lr = initial_lr, initial_lr
        self.thinking_lr, self.current_thinking_lr = thinking_lr, thinking_lr
        self.win_rate_window = deque(maxlen=20)
        self.energy_quality_window = deque(maxlen=20)
        self.energy_separation_window = deque(maxlen=20)
        self.min_win_rate, self.min_energy_quality, self.min_energy_separation = (
            0.15,
            5.0,
            0.01,
        )
        self.max_early_stop_rate = 0.8
        self.lr_decay_factor, self.max_lr_reductions, self.lr_reductions = 0.7, 5, 0
        self.consecutive_poor_episodes = 0
        self.emergency_mode = False

    def update_metrics(
        self, win_rate, energy_quality, energy_separation, early_stop_rate
    ):
        self.win_rate_window.append(win_rate)
        self.energy_quality_window.append(energy_quality)
        self.energy_separation_window.append(energy_separation)

        avg_win_rate = safe_mean(list(self.win_rate_window), 0.5)
        avg_energy_quality = safe_mean(list(self.energy_quality_window), 50.0)
        avg_energy_separation = safe_mean(list(self.energy_separation_window), 1.0)

        collapse_indicators = sum(
            [
                avg_win_rate < self.min_win_rate,
                avg_energy_quality < self.min_energy_quality,
                avg_energy_separation < self.min_energy_separation,
                early_stop_rate > self.max_early_stop_rate,
            ]
        )

        if collapse_indicators >= 2:
            self.consecutive_poor_episodes += 1
            if self.consecutive_poor_episodes >= 5:
                return self._trigger_emergency_protocol()
        else:
            self.consecutive_poor_episodes = 0
            self.emergency_mode = False
        return False

    def _trigger_emergency_protocol(self):
        if not self.emergency_mode:
            self.emergency_mode = True
            self.consecutive_poor_episodes = 0
            if self.lr_reductions < self.max_lr_reductions:
                self.current_lr *= self.lr_decay_factor
                self.current_thinking_lr *= self.lr_decay_factor
                self.lr_reductions += 1
            return True
        return False

    def get_current_lrs(self):
        return self.current_lr, self.current_thinking_lr


class ExperienceBuffer:
    """Experience replay buffer."""

    def __init__(self, capacity=50000, quality_threshold=0.6):
        self.capacity = capacity
        self.quality_threshold = quality_threshold
        self.buffer = deque(maxlen=capacity)
        self.quality_scores = deque(maxlen=capacity)
        self.total_added, self.total_rejected = 0, 0
        self.min_quality_threshold, self.max_quality_threshold = 0.2, 0.9
        self.adaptation_rate = 0.01

    def add_experience(self, experience, quality_score):
        self.total_added += 1
        if quality_score >= self.quality_threshold:
            self.buffer.append(experience)
            self.quality_scores.append(quality_score)
        else:
            self.total_rejected += 1

    def adapt_quality_threshold(self, acceptance_rate):
        if (
            acceptance_rate < 0.3
            and self.quality_threshold > self.min_quality_threshold
        ):
            self.quality_threshold -= self.adaptation_rate
        elif (
            acceptance_rate > 0.8
            and self.quality_threshold < self.max_quality_threshold
        ):
            self.quality_threshold += self.adaptation_rate

    def sample_batch(self, batch_size):
        if len(self.buffer) < batch_size:
            return []
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def emergency_purge(self, keep_ratio=0.2):
        if not self.buffer:
            return
        sorted_indices = np.argsort(np.array(list(self.quality_scores)))[::-1]
        keep_count = max(1, int(len(self.buffer) * keep_ratio))
        keep_indices = sorted_indices[:keep_count]
        self.buffer = deque(
            (self.buffer[i] for i in keep_indices), maxlen=self.capacity
        )
        self.quality_scores = deque(
            (self.quality_scores[i] for i in keep_indices), maxlen=self.capacity
        )

    def get_stats(self):
        return {
            "size": len(self.buffer),
            "acceptance_rate": safe_divide(
                self.total_added - self.total_rejected, self.total_added
            ),
        }


class EnergyBasedTrainer:
    """Handles training loop and updates."""

    def __init__(self, verifier, agent, device, lr=3e-5, batch_size=16):
        self.verifier, self.device, self.batch_size = verifier, device, batch_size
        self.optimizer = AdamW(self.verifier.parameters(), lr=lr, weight_decay=1e-4)
        self.positive_margin, self.negative_margin = 0.1, 1.0

    def update_learning_rate(self, new_lr):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

    def train_step(self, experience_batch):
        obs_list, good_actions_list, bad_actions_list = zip(*experience_batch)
        context = {
            "visual_obs": torch.stack(
                [torch.from_numpy(obs["visual_obs"]) for obs in obs_list]
            ).to(self.device),
            "vector_obs": torch.stack(
                [torch.from_numpy(obs["vector_obs"]) for obs in obs_list]
            ).to(self.device),
        }
        good_actions = (
            F.one_hot(
                torch.from_numpy(np.array(good_actions_list)),
                num_classes=self.verifier.action_dim,
            )
            .float()
            .to(self.device)
        )
        bad_actions = (
            F.one_hot(
                torch.from_numpy(np.array(bad_actions_list)),
                num_classes=self.verifier.action_dim,
            )
            .float()
            .to(self.device)
        )

        good_energies = self.verifier(context, good_actions)
        bad_energies = self.verifier(context, bad_actions)
        loss = (
            F.relu(good_energies - self.positive_margin).mean()
            + F.relu(self.negative_margin - bad_energies).mean()
        )

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.verifier.parameters(), 1.0)
        self.optimizer.step()

        with torch.no_grad():
            energy_sep = (bad_energies.mean() - good_energies.mean()).item()
            energy_qual = (1 / torch.clamp(good_energies.mean(), min=1e-6)).item()
        return loss.item(), energy_sep, energy_qual


class CheckpointManager:
    """Model checkpoint management."""

    def __init__(self, checkpoint_dir="checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

    def save_checkpoint(self, verifier, agent, episode, win_rate, energy_quality):
        path = self.checkpoint_dir / f"model_ep{episode}.zip"
        temp_model_path = self.checkpoint_dir / f"temp_model_ep{episode}.pt"
        temp_info_path = self.checkpoint_dir / f"temp_info_ep{episode}.txt"

        torch.save(verifier.state_dict(), temp_model_path)

        with open(temp_info_path, "w") as f:
            f.write(f"episode={episode}\n")
            f.write(f"win_rate={win_rate:.6f}\n")

        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(temp_model_path, f"model_ep{episode}.pt")
            zipf.write(temp_info_path, f"info_ep{episode}.txt")

        temp_model_path.unlink()
        temp_info_path.unlink()
        print(f"💾 Model saved as {path} (Win Rate: {win_rate:.3f})")

    def load_checkpoint(self, checkpoint_path, verifier, agent):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        with zipfile.ZipFile(checkpoint_path, "r") as zipf:
            pt_files = [f for f in zipf.namelist() if f.endswith(".pt")]
            txt_files = [f for f in zipf.namelist() if f.endswith(".txt")]

            if not pt_files:
                raise ValueError(f"No .pt file found in {checkpoint_path}")

            temp_dir = self.checkpoint_dir / "temp"
            temp_dir.mkdir(exist_ok=True)

            zipf.extract(pt_files[0], temp_dir)
            temp_pt_path = temp_dir / pt_files[0]
            state_dict = torch.load(temp_pt_path, map_location="cpu")
            verifier.load_state_dict(state_dict)

            episode = 0
            win_rate = 0.0
            if txt_files:
                zipf.extract(txt_files[0], temp_dir)
                temp_txt_path = temp_dir / txt_files[0]

                try:
                    with open(temp_txt_path, "r") as f:
                        for line in f:
                            if line.startswith("episode="):
                                episode = int(line.strip().split("=")[1])
                            elif line.startswith("win_rate="):
                                win_rate = float(line.strip().split("=")[1])
                except Exception as e:
                    print(f"⚠️ Could not read info file: {e}")

                temp_txt_path.unlink()

            temp_pt_path.unlink()
            if temp_dir.exists():
                temp_dir.rmdir()

        return {"episode": episode, "win_rate": win_rate}


# Helper functions
def make_multi_round_env(
    game="StreetFighterIISpecialChampionEdition-Genesis",
    state="ken_bison_12.state",
    render_mode=None,
    max_rounds=3,
):
    """Create a multi-round Street Fighter environment."""
    env = retro.make(game=game, state=state, render_mode=render_mode)
    return MultiRoundStreetFighterWrapper(
        env, rendering=(render_mode is not None), max_rounds=max_rounds
    )


def calculate_match_win_rate(match_winner):
    """Convert match winner to win rate."""
    if match_winner == "player":
        return 1.0
    elif match_winner == "opponent":
        return 0.0
    elif match_winner == "draw":
        return 0.5
    else:
        return 0.0


def format_match_result(match_info):
    """Format match result for display."""
    winner = match_info["match_winner"]
    player_rounds = match_info["player_rounds_won"]
    opponent_rounds = match_info["opponent_rounds_won"]
    round_results = match_info["round_results"]

    if winner == "player":
        emoji = "🏆"
    elif winner == "opponent":
        emoji = "💀"
    elif winner == "draw":
        emoji = "🤝"
    else:
        emoji = "❓"

    rounds_str = " | ".join(
        [
            "🏆" if r == "player" else "💀" if r == "opponent" else "🤝"
            for r in round_results
        ]
    )

    return f"{emoji} {winner.upper()} WINS {player_rounds}-{opponent_rounds} [{rounds_str}]"


def calculate_experience_quality(match_info, match_reward):
    """Calculate experience quality based on match performance."""
    quality_score = 0.5  # Base quality

    match_winner = match_info.get("match_winner", None)
    if match_winner == "player":
        quality_score += 0.4
    elif match_winner == "opponent":
        quality_score -= 0.2
    elif match_winner == "draw":
        quality_score += 0.1

    if match_reward > 0:
        quality_score += min(match_reward / 200.0, 0.2)
    elif match_reward < 0:
        quality_score -= min(abs(match_reward) / 200.0, 0.1)

    player_rounds = match_info.get("player_rounds_won", 0)
    opponent_rounds = match_info.get("opponent_rounds_won", 0)

    if player_rounds > opponent_rounds:
        quality_score += 0.1 * (player_rounds - opponent_rounds)

    return max(0.0, min(1.0, quality_score))


def create_experience_tuple(obs, good_action, bad_action):
    """Create experience tuple."""
    return (obs, good_action, bad_action)


print("\n🎉 Multi-Round Street Fighter Wrapper loaded successfully!")
print("   - Maximum 3 rounds per match")
print("   - First to win 2 rounds wins the match")
print("   - All components ready for training")
print("🛡️  Ready for Multi-Round Street Fighter AI training!")
