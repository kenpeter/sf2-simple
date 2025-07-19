#!/usr/bin/env python3
"""
🎯 STABILIZED TRAINING SYSTEM FOR 60% WIN RATE
Fixed version compatible with older PyTorch versions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import time
from pathlib import Path
from tqdm import tqdm
import signal
import sys
from collections import deque
import math

# Import wrapper components
from wrapper import (
    make_fixed_env,
    FixedEnergyBasedStreetFighterVerifier,
    FixedStabilizedEnergyBasedAgent,
    FixedEnergyStabilityManager,
    CheckpointManager,
    verify_fixed_energy_flow,
    safe_mean,
    safe_std,
    VECTOR_FEATURE_DIM,
)


class IntelligentRewardShaper:
    """
    🧠 Intelligent reward shaping that provides dense, meaningful feedback
    for every single action, directly integrated into the training loop.
    """

    def __init__(self, history_length=30):
        self.history_length = history_length

        # Combat state tracking
        self.health_history = deque(maxlen=history_length)
        self.opponent_health_history = deque(maxlen=history_length)
        self.score_history = deque(maxlen=history_length)
        self.position_history = deque(maxlen=history_length)
        self.opponent_position_history = deque(maxlen=history_length)

        # Strategic metrics
        self.combo_count = 0
        self.last_hit_frame = -1
        self.current_frame = 0
        self.knockdown_bonus_frames = 0
        self.pressure_frames = 0
        self.defensive_frames = 0

        # Advanced combat analysis
        self.momentum_tracker = deque(maxlen=10)
        self.spacing_quality_history = deque(maxlen=20)
        self.aggression_history = deque(maxlen=15)

        # Reward components weights (tuned for winning)
        self.weights = {
            "damage_dealt": 15.0,  # High value for dealing damage
            "damage_avoided": 8.0,  # Reward avoiding damage
            "health_advantage": 5.0,  # Reward maintaining health lead
            "combo_progression": 12.0,  # Big bonus for combo building
            "positioning": 6.0,  # Reward good spacing
            "pressure": 4.0,  # Reward maintaining offensive pressure
            "momentum": 7.0,  # Reward positive momentum shifts
            "defensive": 3.0,  # Moderate reward for defense
            "win_progress": 10.0,  # Reward progress toward winning
            "frame_efficiency": 2.0,  # Small bonus for not wasting time
        }

        # Normalization factors
        self.max_health = 176
        self.screen_width = 320
        self.optimal_distance_min = 60
        self.optimal_distance_max = 100

        print("🧠 Intelligent Reward Shaper initialized")
        print(f"   - Dense rewards every frame")
        print(f"   - {len(self.weights)} reward components")
        print(f"   - Optimized for winning behavior")

    def calculate_intelligent_reward(self, info: dict, prev_info: dict = None):
        """
        Calculate intelligent, dense reward for current action.
        Returns (reward, breakdown) for transparency.
        """
        self.current_frame += 1
        reward_breakdown = {}
        total_reward = 0.0

        # Extract current state
        player_health = info.get("agent_hp", self.max_health)
        opponent_health = info.get("enemy_hp", self.max_health)
        player_x = info.get("agent_x", self.screen_width // 2)
        opponent_x = info.get("enemy_x", self.screen_width // 2)
        score = info.get("score", 0)

        # Store history
        self.health_history.append(player_health)
        self.opponent_health_history.append(opponent_health)
        self.score_history.append(score)
        self.position_history.append(player_x)
        self.opponent_position_history.append(opponent_x)

        # 1. DAMAGE AND HEALTH REWARDS
        damage_reward = self._calculate_damage_rewards(
            player_health, opponent_health, prev_info
        )
        total_reward += damage_reward
        reward_breakdown["damage"] = damage_reward

        # 2. COMBO AND MOMENTUM REWARDS
        combo_reward = self._calculate_combo_rewards(score, prev_info)
        total_reward += combo_reward
        reward_breakdown["combo"] = combo_reward

        # 3. POSITIONING AND SPACING REWARDS
        spacing_reward = self._calculate_spacing_rewards(player_x, opponent_x)
        total_reward += spacing_reward
        reward_breakdown["spacing"] = spacing_reward

        # 4. TACTICAL PRESSURE REWARDS
        pressure_reward = self._calculate_pressure_rewards(
            player_x, opponent_x, player_health, opponent_health
        )
        total_reward += pressure_reward
        reward_breakdown["pressure"] = pressure_reward

        # 5. MOMENTUM AND ADAPTATION REWARDS
        momentum_reward = self._calculate_momentum_rewards()
        total_reward += momentum_reward
        reward_breakdown["momentum"] = momentum_reward

        # 6. WIN CONDITION PROGRESS REWARDS
        win_progress_reward = self._calculate_win_progress_rewards(
            player_health, opponent_health
        )
        total_reward += win_progress_reward
        reward_breakdown["win_progress"] = win_progress_reward

        # 7. FRAME EFFICIENCY (small time penalty)
        efficiency_reward = -0.001 * self.weights["frame_efficiency"]
        total_reward += efficiency_reward
        reward_breakdown["efficiency"] = efficiency_reward

        # Apply intelligent scaling based on game state
        total_reward = self._apply_context_scaling(
            total_reward, player_health, opponent_health
        )

        # Normalize to reasonable range
        total_reward = np.clip(total_reward, -0.5, 2.0)

        return total_reward, reward_breakdown

    def _calculate_damage_rewards(self, player_health, opponent_health, prev_info):
        """Calculate rewards for damage dealt and damage avoided."""
        reward = 0.0

        if prev_info is not None:
            prev_player_health = prev_info.get("agent_hp", player_health)
            prev_opponent_health = prev_info.get("enemy_hp", opponent_health)

            # Damage dealt reward (high priority)
            damage_dealt = max(0, prev_opponent_health - opponent_health)
            if damage_dealt > 0:
                # Scale reward based on opponent's remaining health (more valuable when they're low)
                health_factor = 1.0 + (1.0 - opponent_health / self.max_health) * 0.5
                reward += (
                    damage_dealt * self.weights["damage_dealt"] * health_factor * 0.01
                )

            # Damage avoided reward
            damage_received = max(0, prev_player_health - player_health)
            if damage_received == 0 and len(self.health_history) > 5:
                # Bonus for avoiding damage over multiple frames
                # Convert deque to list for safe iteration
                health_list = list(self.health_history)
                recent_damage = sum(
                    max(0, health_list[i] - health_list[i + 1])
                    for i in range(max(0, len(health_list) - 5), len(health_list) - 1)
                )
                if recent_damage == 0:
                    reward += (
                        self.weights["damage_avoided"] * 0.001
                    )  # Small but consistent

            # Health advantage reward
            health_diff = player_health - opponent_health
            prev_health_diff = prev_player_health - prev_opponent_health
            if health_diff > prev_health_diff:
                reward += (
                    (health_diff - prev_health_diff)
                    * self.weights["health_advantage"]
                    * 0.005
                )

        return reward

    def _calculate_combo_rewards(self, score, prev_info):
        """Calculate rewards for combo building and score improvement."""
        reward = 0.0

        if prev_info is not None:
            prev_score = prev_info.get("score", 0)
            score_increase = score - prev_score

            if score_increase > 0:
                # Detect combo continuation
                if self.current_frame - self.last_hit_frame <= 60:  # 1 second window
                    self.combo_count += 1
                    # Exponential bonus for longer combos
                    combo_multiplier = 1.0 + (self.combo_count * 0.2)
                    reward += (
                        score_increase
                        * self.weights["combo_progression"]
                        * combo_multiplier
                        * 0.001
                    )
                else:
                    self.combo_count = 1
                    reward += score_increase * self.weights["combo_progression"] * 0.001

                self.last_hit_frame = self.current_frame
            else:
                # Combo dropped after 1 second
                if self.current_frame - self.last_hit_frame > 60:
                    self.combo_count = 0

        return reward

    def _calculate_spacing_rewards(self, player_x, opponent_x):
        """Calculate rewards for optimal spacing and positioning."""
        reward = 0.0
        distance = abs(player_x - opponent_x)

        # Optimal spacing reward
        if self.optimal_distance_min <= distance <= self.optimal_distance_max:
            reward += self.weights["positioning"] * 0.002
            self.spacing_quality_history.append(1.0)
        else:
            self.spacing_quality_history.append(0.0)

        # Consistent good spacing bonus
        if len(self.spacing_quality_history) >= 10:
            # Convert deque to list for slicing
            spacing_list = list(self.spacing_quality_history)
            recent_quality = sum(spacing_list[-10:]) / 10
            if recent_quality > 0.7:  # 70% good spacing
                reward += self.weights["positioning"] * 0.001

        # Corner awareness (slight penalty for being cornered)
        player_corner_distance = min(player_x, self.screen_width - player_x)
        if player_corner_distance < 50:  # Too close to corner
            reward -= self.weights["positioning"] * 0.001

        # Center control bonus
        screen_center = self.screen_width // 2
        if abs(player_x - screen_center) < abs(opponent_x - screen_center):
            reward += self.weights["positioning"] * 0.0005

        return reward

    def _calculate_pressure_rewards(
        self, player_x, opponent_x, player_health, opponent_health
    ):
        """Calculate rewards for maintaining offensive pressure."""
        reward = 0.0
        distance = abs(player_x - opponent_x)

        # Close-range pressure (when we have health advantage)
        if player_health > opponent_health and distance < 80:
            self.pressure_frames += 1
            reward += self.weights["pressure"] * 0.001
        else:
            self.pressure_frames = max(0, self.pressure_frames - 1)

        # Sustained pressure bonus
        if self.pressure_frames > 30:  # Half second of pressure
            reward += self.weights["pressure"] * 0.002

        # Defensive positioning when behind
        if player_health < opponent_health and distance > 100:
            self.defensive_frames += 1
            reward += self.weights["defensive"] * 0.0005
        else:
            self.defensive_frames = max(0, self.defensive_frames - 1)

        return reward

    def _calculate_momentum_rewards(self):
        """Calculate rewards for positive momentum shifts."""
        reward = 0.0

        if len(self.health_history) >= 5 and len(self.opponent_health_history) >= 5:
            # Calculate health momentum over last 5 frames
            player_momentum = (self.health_history[-1] - self.health_history[-5]) / 5
            opponent_momentum = (
                self.opponent_health_history[-1] - self.opponent_health_history[-5]
            ) / 5

            # Positive momentum = we're losing less health than opponent
            relative_momentum = opponent_momentum - player_momentum

            self.momentum_tracker.append(relative_momentum)

            # Reward positive momentum
            if relative_momentum > 0:
                reward += relative_momentum * self.weights["momentum"] * 0.01

            # Bonus for sustained positive momentum
            if len(self.momentum_tracker) >= 5:
                # Convert deque to list for slicing
                momentum_list = list(self.momentum_tracker)
                recent_momentum = sum(momentum_list[-5:]) / 5
                if recent_momentum > 0.5:  # Consistent positive momentum
                    reward += self.weights["momentum"] * 0.002

        return reward

    def _calculate_win_progress_rewards(self, player_health, opponent_health):
        """Calculate rewards for progress toward winning condition."""
        reward = 0.0

        # Health percentage rewards
        player_health_pct = player_health / self.max_health
        opponent_health_pct = opponent_health / self.max_health

        # Reward for maintaining high health
        if player_health_pct > 0.7:
            reward += self.weights["win_progress"] * 0.001

        # Big bonus for getting opponent to critical health
        if opponent_health_pct < 0.3:
            reward += self.weights["win_progress"] * 0.005

        if opponent_health_pct < 0.1:
            reward += self.weights["win_progress"] * 0.01  # Close to victory!

        # Health advantage scaling
        health_advantage = player_health_pct - opponent_health_pct
        if health_advantage > 0:
            reward += health_advantage * self.weights["win_progress"] * 0.002

        return reward

    def _apply_context_scaling(self, base_reward, player_health, opponent_health):
        """Apply intelligent scaling based on game context."""
        # Scale up rewards when the game is close (more critical decisions)
        health_diff = abs(player_health - opponent_health)
        if health_diff < 30:  # Very close game
            base_reward *= 1.3
        elif health_diff < 60:  # Somewhat close
            base_reward *= 1.1

        # Scale up rewards in critical health situations
        if player_health < 40 or opponent_health < 40:
            base_reward *= 1.2

        return base_reward

    def calculate_win_reward(
        self, won: bool, player_health: int, opponent_health: int, episode_length: int
    ) -> float:
        """Calculate final win/loss reward with intelligent scaling."""
        if won:
            # Base win reward
            base_win_reward = 1.0

            # Health bonus (win with more health = better)
            health_bonus = (player_health / self.max_health) * 0.3

            # Speed bonus (faster win = better, but not too aggressive)
            if episode_length < 1000:
                speed_bonus = 0.2
            elif episode_length < 2000:
                speed_bonus = 0.1
            else:
                speed_bonus = 0.0

            total_win_reward = base_win_reward + health_bonus + speed_bonus
            return total_win_reward
        else:
            # Small loss penalty, but not too harsh
            return -0.1

    def get_experience_quality_score(
        self, reward: float, reward_breakdown: dict
    ) -> float:
        """
        Calculate quality score for experience buffer labeling.
        This replaces the broken percentile-based system.
        """
        # Base quality from total reward
        base_quality = math.tanh(reward * 2.0) * 0.5 + 0.5  # Normalize to [0, 1]

        # Bonus for key strategic actions
        strategic_bonus = 0.0

        if reward_breakdown.get("damage", 0) > 0.01:  # Dealing damage
            strategic_bonus += 0.2

        if reward_breakdown.get("combo", 0) > 0.005:  # Good combo
            strategic_bonus += 0.15

        if reward_breakdown.get("spacing", 0) > 0.001:  # Good positioning
            strategic_bonus += 0.1

        if reward_breakdown.get("momentum", 0) > 0.001:  # Positive momentum
            strategic_bonus += 0.1

        quality_score = min(1.0, base_quality + strategic_bonus)
        return quality_score

    def reset_episode(self):
        """Reset tracking for new episode."""
        self.health_history.clear()
        self.opponent_health_history.clear()
        self.score_history.clear()
        self.position_history.clear()
        self.opponent_position_history.clear()
        self.momentum_tracker.clear()
        self.spacing_quality_history.clear()
        self.aggression_history.clear()

        self.combo_count = 0
        self.last_hit_frame = -1
        self.current_frame = 0
        self.knockdown_bonus_frames = 0
        self.pressure_frames = 0
        self.defensive_frames = 0

    def get_stats(self) -> dict:
        """Get current shaper statistics."""
        return {
            "combo_count": self.combo_count,
            "pressure_frames": self.pressure_frames,
            "defensive_frames": self.defensive_frames,
            "avg_spacing_quality": (
                np.mean(list(self.spacing_quality_history))
                if self.spacing_quality_history
                else 0.0
            ),
            "avg_momentum": (
                np.mean(list(self.momentum_tracker)) if self.momentum_tracker else 0.0
            ),
        }


class StabilizedExperienceBuffer:
    """
    🎯 Stabilized experience buffer using intelligent quality scoring
    instead of broken percentile-based labeling.
    """

    def __init__(self, capacity=40000, quality_threshold=0.6):
        self.capacity = capacity
        self.quality_threshold = quality_threshold

        # Simple good/bad storage
        self.good_experiences = deque(maxlen=capacity // 2)
        self.bad_experiences = deque(maxlen=capacity // 2)

        # Quality tracking
        self.total_added = 0
        self.total_good = 0
        self.total_bad = 0

        print(f"🎯 Stabilized Experience Buffer initialized")
        print(f"   - Quality threshold: {quality_threshold}")
        print(f"   - Good/bad split based on intelligent scoring")

    def add_experience(self, experience, quality_score):
        """Add single experience with quality-based labeling."""
        self.total_added += 1

        # Label based on quality threshold
        if quality_score >= self.quality_threshold:
            experience["is_good"] = True
            experience["quality_score"] = quality_score
            self.good_experiences.append(experience)
            self.total_good += 1
        else:
            experience["is_good"] = False
            experience["quality_score"] = quality_score
            self.bad_experiences.append(experience)
            self.total_bad += 1

    def sample_balanced_batch(self, batch_size):
        """Sample balanced batch of good and bad experiences."""
        target_per_class = batch_size // 2

        if (
            len(self.good_experiences) < target_per_class
            or len(self.bad_experiences) < target_per_class
        ):
            return None, None

        # Sample randomly from each class
        good_indices = np.random.choice(
            len(self.good_experiences), target_per_class, replace=False
        )
        bad_indices = np.random.choice(
            len(self.bad_experiences), target_per_class, replace=False
        )

        good_batch = [self.good_experiences[i] for i in good_indices]
        bad_batch = [self.bad_experiences[i] for i in bad_indices]

        return good_batch, bad_batch

    def get_stats(self):
        """Get buffer statistics."""
        total_size = len(self.good_experiences) + len(self.bad_experiences)
        good_ratio = len(self.good_experiences) / max(1, total_size)

        return {
            "total_size": total_size,
            "good_count": len(self.good_experiences),
            "bad_count": len(self.bad_experiences),
            "good_ratio": good_ratio,
            "total_added": self.total_added,
            "acceptance_rate": total_size / max(1, self.total_added),
        }


class IntelligentStreetFighterWrapper:
    """
    🧠 Enhanced wrapper that integrates intelligent reward shaping
    directly into the environment step process.
    """

    def __init__(self, base_env):
        self.base_env = base_env
        self.reward_shaper = IntelligentRewardShaper()

        # Previous info for reward calculation
        self.prev_info = None
        self.episode_steps = 0
        self.episode_rewards = []
        self.episode_reward_breakdowns = []

        # Episode tracking for experience buffer
        self.current_episode_experiences = []
        self.experience_callback = None

        print("🧠 Intelligent Street Fighter Wrapper initialized")

    def reset(self, **kwargs):
        """Reset environment and reward shaper."""
        obs, info = self.base_env.reset(**kwargs)

        # Process previous episode if exists
        if self.current_episode_experiences:
            self._finalize_episode()

        # Reset tracking
        self.reward_shaper.reset_episode()
        self.prev_info = info.copy()
        self.episode_steps = 0
        self.episode_rewards = []
        self.episode_reward_breakdowns = []
        self.current_episode_experiences = []

        return obs, info

    def step(self, action):
        """Enhanced step with intelligent reward shaping."""
        # Take step in base environment
        obs, base_reward, done, truncated, info = self.base_env.step(action)
        self.episode_steps += 1

        # Calculate intelligent reward
        intelligent_reward, reward_breakdown = (
            self.reward_shaper.calculate_intelligent_reward(info, self.prev_info)
        )

        # Handle episode completion
        if done or truncated:
            # Add final win/loss reward
            won = info.get("wins", 0) > info.get("losses", 0)
            final_reward = self.reward_shaper.calculate_win_reward(
                won,
                info.get("agent_hp", 0),
                info.get("enemy_hp", 0),
                self.episode_steps,
            )
            intelligent_reward += final_reward
            reward_breakdown["final"] = final_reward

        # Store episode data
        self.episode_rewards.append(intelligent_reward)
        self.episode_reward_breakdowns.append(reward_breakdown)

        # Calculate quality score for experience buffer
        quality_score = self.reward_shaper.get_experience_quality_score(
            intelligent_reward, reward_breakdown
        )

        # Store experience data
        experience = {
            "observations": obs.copy() if isinstance(obs, dict) else obs,
            "action": action,
            "reward": intelligent_reward,
            "quality_score": quality_score,
            "reward_breakdown": reward_breakdown,
            "step_number": self.episode_steps,
            "info": info.copy(),
        }
        self.current_episode_experiences.append(experience)

        # Update previous info
        self.prev_info = info.copy()

        # Add shaper stats to info
        info.update(self.reward_shaper.get_stats())
        info["intelligent_reward"] = intelligent_reward
        info["quality_score"] = quality_score

        return obs, intelligent_reward, done, truncated, info

    def _finalize_episode(self):
        """Process completed episode for experience buffer."""
        if self.experience_callback:
            self.experience_callback(self.current_episode_experiences)

    def set_experience_callback(self, callback):
        """Set callback for experience processing."""
        self.experience_callback = callback

    # Delegate all other attributes to base environment
    def __getattr__(self, name):
        return getattr(self.base_env, name)


class StabilizedEnergyTrainer:
    """
    🎯 Stabilized trainer for achieving 60% win rate using intelligent rewards.
    """

    def __init__(self, args):
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
        )

        # Initialize components
        self.env = None
        self.verifier = None
        self.agent = None
        self.optimizer = None
        self.scheduler = None
        self.stability_manager = None
        self.experience_buffer = None
        self.checkpoint_manager = None

        # Training state
        self.episode = 0
        self.total_steps = 0
        self.training_active = True

        # Performance tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_wins = deque(maxlen=100)
        self.energy_separations = deque(maxlen=50)
        self.training_losses = deque(maxlen=50)

        # Target tracking
        self.target_win_rate = 0.6
        self.best_win_rate = 0.0
        self.win_rate_history = deque(maxlen=20)

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        print(f"🎯 Stabilized Energy Trainer initialized")
        print(f"   - Target win rate: {self.target_win_rate:.1%}")
        print(f"   - Device: {self.device}")

    def _signal_handler(self, signum, frame):
        """Handle graceful shutdown."""
        print(f"\n🛑 Graceful shutdown initiated")
        self.training_active = False

        if self.checkpoint_manager and self.verifier and self.agent:
            self._save_emergency_checkpoint()

        sys.exit(0)

    def setup_training(self):
        """Initialize all training components with stabilized parameters."""
        print("🔧 Setting up stabilized training components...")

        # Create base environment
        render_mode = "human" if self.args.render else None
        base_env = make_fixed_env(render_mode=render_mode)

        # Wrap with intelligent reward system
        self.env = IntelligentStreetFighterWrapper(base_env)

        # Create verifier with conservative initialization
        self.verifier = FixedEnergyBasedStreetFighterVerifier(
            self.env.observation_space, self.env.action_space, features_dim=256
        ).to(self.device)

        # Create agent with stabilized parameters
        self.agent = FixedStabilizedEnergyBasedAgent(
            self.verifier,
            thinking_steps=self.args.thinking_steps,
            thinking_lr=self.args.thinking_lr,
            noise_scale=0.05,  # Reduced noise for stability
        )

        # Create optimizer with lower learning rate
        self.optimizer = optim.Adam(
            self.verifier.parameters(), lr=self.args.lr, weight_decay=1e-5, eps=1e-8
        )

        # Add learning rate scheduler (fixed version)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.8, patience=50
        )

        # Create stability manager
        self.stability_manager = FixedEnergyStabilityManager(
            initial_lr=self.args.lr, thinking_lr=self.args.thinking_lr
        )

        # Create stabilized experience buffer
        self.experience_buffer = StabilizedExperienceBuffer(
            capacity=30000, quality_threshold=0.55  # Balanced threshold
        )

        # Set up experience callback
        self.env.set_experience_callback(self._process_episode_experiences)

        # Create checkpoint manager
        self.checkpoint_manager = CheckpointManager("checkpoints_stabilized")

        print("✅ Stabilized training components initialized")

    def _process_episode_experiences(self, experiences):
        """Process completed episode experiences."""
        for exp in experiences:
            quality_score = exp["quality_score"]
            self.experience_buffer.add_experience(exp, quality_score)

    def verify_setup(self):
        """Verify energy flow before training."""
        print("🔬 Verifying energy flow...")

        if verify_fixed_energy_flow(self.verifier, self.env, self.device):
            print("   ✅ Energy flow verified - ready for stabilized training!")
            return True
        else:
            print("   ❌ Energy flow verification failed!")
            return False

    def calculate_contrastive_loss(self, good_batch, bad_batch):
        """Calculate contrastive loss with stabilized margin."""
        if not good_batch or not bad_batch:
            return torch.tensor(0.0, device=self.device), torch.tensor(
                0.0, device=self.device
            )

        good_energies = []
        bad_energies = []

        # Process good experiences
        for exp in good_batch:
            obs = exp["observations"]
            action = exp["action"]

            # Convert observations to tensors
            obs_tensor = {}
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    obs_tensor[key] = (
                        torch.from_numpy(value).unsqueeze(0).float().to(self.device)
                    )
                else:
                    obs_tensor[key] = (
                        torch.tensor(value).unsqueeze(0).float().to(self.device)
                    )

            # Create action tensor (one-hot)
            action_tensor = torch.zeros(1, self.verifier.action_dim, device=self.device)
            action_tensor[0, action] = 1.0

            energy = self.verifier(obs_tensor, action_tensor)
            good_energies.append(energy)

        # Process bad experiences
        for exp in bad_batch:
            obs = exp["observations"]
            action = exp["action"]

            # Convert observations to tensors
            obs_tensor = {}
            for key, value in obs.items():
                if isinstance(value, np.ndarray):
                    obs_tensor[key] = (
                        torch.from_numpy(value).unsqueeze(0).float().to(self.device)
                    )
                else:
                    obs_tensor[key] = (
                        torch.tensor(value).unsqueeze(0).float().to(self.device)
                    )

            # Create action tensor (one-hot)
            action_tensor = torch.zeros(1, self.verifier.action_dim, device=self.device)
            action_tensor[0, action] = 1.0

            energy = self.verifier(obs_tensor, action_tensor)
            bad_energies.append(energy)

        if not good_energies or not bad_energies:
            return torch.tensor(0.0, device=self.device), torch.tensor(
                0.0, device=self.device
            )

        # Stack energies
        good_energy_tensor = torch.cat(good_energies)
        bad_energy_tensor = torch.cat(bad_energies)

        # Calculate contrastive loss (good should have lower energy than bad)
        good_mean = good_energy_tensor.mean()
        bad_mean = bad_energy_tensor.mean()

        # Energy separation (bad - good should be positive and > margin)
        energy_separation = bad_mean - good_mean
        margin = self.args.contrastive_margin
        contrastive_loss = torch.clamp(margin - energy_separation, min=0.0)

        return contrastive_loss, energy_separation

    def train_step(self):
        """Single stabilized training step."""
        # Sample balanced batch
        good_batch, bad_batch = self.experience_buffer.sample_balanced_batch(
            self.args.batch_size
        )

        if good_batch is None or bad_batch is None:
            return 0.0, 0.0, {"message": "insufficient_data"}

        # Zero gradients
        self.optimizer.zero_grad()

        # Calculate contrastive loss
        contrastive_loss, energy_separation = self.calculate_contrastive_loss(
            good_batch, bad_batch
        )

        # Backward pass
        if contrastive_loss.requires_grad and contrastive_loss.item() > 0:
            contrastive_loss.backward()

            # Stabilized gradient clipping
            torch.nn.utils.clip_grad_norm_(self.verifier.parameters(), max_norm=1.0)

            self.optimizer.step()

        return (
            contrastive_loss.item(),
            energy_separation.item(),
            {
                "good_quality": safe_mean(
                    [exp["quality_score"] for exp in good_batch], 0.0
                ),
                "bad_quality": safe_mean(
                    [exp["quality_score"] for exp in bad_batch], 0.0
                ),
            },
        )

    def run_episode(self):
        """Run single episode with intelligent reward tracking."""
        obs, info = self.env.reset()
        total_intelligent_reward = 0.0
        episode_reward_breakdowns = []
        steps = 0
        won = False

        thinking_successes = 0
        thinking_attempts = 0

        while True:
            # Get action from agent
            action, thinking_info = self.agent.predict(obs, deterministic=False)

            # Track thinking process
            thinking_attempts += 1
            if thinking_info.get("optimization_successful", False):
                thinking_successes += 1

            # Take step
            obs, reward, done, truncated, info = self.env.step(action)
            total_intelligent_reward += reward
            steps += 1
            self.total_steps += 1

            # Store reward breakdown
            if "reward_breakdown" in info:
                episode_reward_breakdowns.append(info["reward_breakdown"])

            if done or truncated:
                won = info.get("wins", 0) > info.get("losses", 0)
                break

        thinking_success_rate = thinking_successes / max(thinking_attempts, 1)

        return {
            "total_reward": total_intelligent_reward,
            "steps": steps,
            "won": won,
            "thinking_success_rate": thinking_success_rate,
            "reward_breakdowns": episode_reward_breakdowns,
            "info": info,
        }

    def _save_emergency_checkpoint(self):
        """Save emergency checkpoint."""
        try:
            current_win_rate = safe_mean(list(self.episode_wins), 0.0)
            energy_quality = safe_mean(list(self.energy_separations), 0.0)

            path = self.checkpoint_manager.save_checkpoint(
                self.verifier,
                self.agent,
                self.episode,
                current_win_rate,
                energy_quality,
                is_emergency=True,
            )
            if path:
                print(f"💾 Emergency checkpoint saved: {path.name}")
        except Exception as e:
            print(f"❌ Failed to save emergency checkpoint: {e}")

    def train(self):
        """Main stabilized training loop focused on 60% win rate."""
        if not self.verify_setup():
            print("❌ Setup verification failed!")
            return

        print(f"\n🎯 Starting STABILIZED Energy-Based Training")
        print(f"🏆 Target: {self.target_win_rate:.1%} win rate")
        print(f"📊 Using intelligent reward shaping")

        # Training metrics
        consecutive_good_episodes = 0
        episodes_since_improvement = 0

        # Progress tracking
        pbar = tqdm(range(self.args.total_episodes), desc="🎯 Stabilized Training")

        for episode in pbar:
            if not self.training_active:
                break

            self.episode = episode

            # Run episode
            episode_result = self.run_episode()

            # Track performance
            self.episode_rewards.append(episode_result["total_reward"])
            self.episode_wins.append(1.0 if episode_result["won"] else 0.0)

            # Training step (if we have enough data)
            buffer_stats = self.experience_buffer.get_stats()
            train_info = {"message": "collecting_data"}

            if buffer_stats["total_size"] >= self.args.batch_size * 2:
                loss, separation, train_info = self.train_step()

                if loss > 0:  # Only track meaningful losses
                    self.training_losses.append(loss)
                    self.energy_separations.append(abs(separation))

                # Calculate current performance metrics
                current_win_rate = safe_mean(list(self.episode_wins), 0.0)
                self.win_rate_history.append(current_win_rate)

                # Update learning rate based on win rate
                self.scheduler.step(current_win_rate)

                # Check for improvement
                if current_win_rate > self.best_win_rate:
                    self.best_win_rate = current_win_rate
                    episodes_since_improvement = 0

                    # Save checkpoint for new best
                    if episode > 50:  # Don't save too early
                        energy_quality = safe_mean(list(self.energy_separations), 0.0)
                        self.checkpoint_manager.save_checkpoint(
                            self.verifier,
                            self.agent,
                            episode,
                            current_win_rate,
                            energy_quality,
                        )
                else:
                    episodes_since_improvement += 1

                # Track consecutive good episodes
                if current_win_rate > 0.5:
                    consecutive_good_episodes += 1
                else:
                    consecutive_good_episodes = 0

                # Update stability manager
                avg_energy_quality = safe_mean(list(self.energy_separations), 0.0)
                avg_energy_separation = safe_mean(list(self.energy_separations), 0.0)
                thinking_stats = self.agent.get_thinking_stats()
                early_stop_rate = thinking_stats.get("early_stop_rate", 0.0)

                emergency_triggered = self.stability_manager.update_metrics(
                    current_win_rate,
                    avg_energy_quality,
                    avg_energy_separation,
                    early_stop_rate,
                )

                if emergency_triggered:
                    print(f"\n🚨 Stability intervention at episode {episode}")
                    # Reduce learning rates
                    new_lr, new_thinking_lr = self.stability_manager.get_current_lrs()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = new_lr
                    self.agent.current_thinking_lr = new_thinking_lr

                # Progress bar update
                pbar.set_postfix(
                    {
                        "WR": f"{current_win_rate:.2f}",
                        "Best": f"{self.best_win_rate:.2f}",
                        "Reward": f"{episode_result['total_reward']:.1f}",
                        "Sep": f"{avg_energy_separation:.3f}",
                        "Buf": f"{buffer_stats['total_size']}",
                        "Good": f"{buffer_stats['good_count']}",
                        "Bad": f"{buffer_stats['bad_count']}",
                    }
                )

                # Detailed reporting every 25 episodes
                if episode % 25 == 0 and episode > 0:
                    self._print_training_report(
                        episode, current_win_rate, episode_result, buffer_stats
                    )

                # Early success check
                if current_win_rate >= self.target_win_rate and episode > 100:
                    print(f"\n🎉 TARGET ACHIEVED! Win rate: {current_win_rate:.1%}")
                    print(
                        f"🏆 Reached {self.target_win_rate:.1%} target at episode {episode}"
                    )
                    break

            else:
                # Still collecting data
                current_win_rate = safe_mean(list(self.episode_wins), 0.0)
                pbar.set_postfix(
                    {
                        "WR": f"{current_win_rate:.2f}",
                        "Reward": f"{episode_result['total_reward']:.1f}",
                        "Collecting": f"{buffer_stats['total_size']}/{self.args.batch_size * 2}",
                        "Good": f"{buffer_stats['good_count']}",
                        "Bad": f"{buffer_stats['bad_count']}",
                    }
                )

        pbar.close()

        # Final results
        final_win_rate = safe_mean(list(self.episode_wins), 0.0)
        final_energy_quality = safe_mean(list(self.energy_separations), 0.0)

        print(f"\n📊 FINAL STABILIZED TRAINING RESULTS:")
        print(f"   🎯 Final Win Rate: {final_win_rate:.1%}")
        print(f"   🏆 Best Win Rate: {self.best_win_rate:.1%}")
        print(f"   ⚡ Energy Quality: {final_energy_quality:.3f}")
        print(f"   📈 Total Episodes: {len(self.episode_rewards)}")
        print(f"   💰 Average Reward: {safe_mean(list(self.episode_rewards), 0.0):.2f}")

        # Success evaluation
        if final_win_rate >= self.target_win_rate:
            print(f"   🎉 SUCCESS: {self.target_win_rate:.1%} TARGET ACHIEVED!")
        elif final_win_rate >= 0.5:
            print(f"   ⭐ STRONG PROGRESS: {final_win_rate:.1%} - Close to target!")
        else:
            print(f"   📈 GOOD FOUNDATION: Continue training to reach target")

        # Save final checkpoint
        final_path = self.checkpoint_manager.save_checkpoint(
            self.verifier, self.agent, episode, final_win_rate, final_energy_quality
        )
        if final_path:
            print(f"💾 Final checkpoint saved: {final_path.name}")

        return {
            "final_win_rate": final_win_rate,
            "best_win_rate": self.best_win_rate,
            "energy_quality": final_energy_quality,
            "total_episodes": len(self.episode_rewards),
        }

    def _print_training_report(
        self, episode, current_win_rate, episode_result, buffer_stats
    ):
        """Print detailed training report."""
        print(f"\n🎯 Episode {episode} - Stabilized Training Report:")
        print(
            f"   📊 Win Rate: {current_win_rate:.1%} (Best: {self.best_win_rate:.1%})"
        )
        print(f"   🏆 Last Episode: {'WON' if episode_result['won'] else 'LOST'}")
        print(f"   💰 Intelligent Reward: {episode_result['total_reward']:.2f}")
        print(f"   🧠 Thinking Success: {episode_result['thinking_success_rate']:.1%}")
        print(
            f"   📚 Buffer: {buffer_stats['total_size']} total ({buffer_stats['good_count']} good, {buffer_stats['bad_count']} bad)"
        )

        # Show recent reward breakdown if available
        if episode_result["reward_breakdowns"]:
            last_breakdown = episode_result["reward_breakdowns"][-1]
            print(f"   🔍 Last Action Rewards: {last_breakdown}")

        # Progress toward target
        progress = current_win_rate / self.target_win_rate
        print(
            f"   🎯 Target Progress: {progress:.1%} toward {self.target_win_rate:.1%}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Stabilized Energy-Based Training for 60% Win Rate"
    )

    # Training parameters (stabilized)
    parser.add_argument(
        "--total-episodes", type=int, default=400, help="Total episodes to train"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate (stabilized)"
    )
    parser.add_argument(
        "--thinking-lr",
        type=float,
        default=0.15,
        help="Thinking learning rate (stabilized)",
    )
    parser.add_argument(
        "--thinking-steps", type=int, default=3, help="Number of thinking steps"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (smaller for stability)"
    )
    parser.add_argument(
        "--contrastive-margin",
        type=float,
        default=1.5,
        help="Contrastive loss margin (stabilized)",
    )

    # System parameters
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument("--render", action="store_true", help="Render environment")
    parser.add_argument("--save-freq", type=int, default=50, help="Save frequency")
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )

    args = parser.parse_args()

    print(f"🎯 STABILIZED Energy-Based Training for 60% Win Rate")
    print(f"💡 Using INTELLIGENT REWARD SHAPING")
    print(f"   - Episodes: {args.total_episodes}")
    print(f"   - Learning rate: {args.lr} (stabilized)")
    print(f"   - Thinking LR: {args.thinking_lr} (stabilized)")
    print(f"   - Contrastive margin: {args.contrastive_margin} (balanced)")
    print(f"   - Batch size: {args.batch_size} (smaller)")
    print(f"   - Device: {args.device}")

    # Create and run trainer
    trainer = StabilizedEnergyTrainer(args)
    trainer.setup_training()

    # Resume from checkpoint if specified
    if args.resume:
        checkpoint_path = Path(args.resume)
        if checkpoint_path.exists():
            trainer.checkpoint_manager.load_checkpoint(
                checkpoint_path, trainer.verifier, trainer.agent
            )
        else:
            print(f"⚠️  Checkpoint not found: {args.resume}")

    # Start stabilized training
    results = trainer.train()

    # Final success evaluation
    if results["final_win_rate"] >= 0.6:
        print(f"\n🎉🏆 MISSION ACCOMPLISHED! 🏆🎉")
        print(f"✅ Successfully achieved 60% win rate target!")
        print(f"🥇 Final win rate: {results['final_win_rate']:.1%}")
        print(f"🔥 Best win rate: {results['best_win_rate']:.1%}")
    else:
        print(f"\n📈 Strong foundation built! Continue training to reach 60% target.")
        print(
            f"Current: {results['final_win_rate']:.1%} | Best: {results['best_win_rate']:.1%}"
        )
        print(
            f"💡 The intelligent reward system is working - just needs more episodes!"
        )


if __name__ == "__main__":
    main()
