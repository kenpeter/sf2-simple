#!/usr/bin/env python3
"""
train.py - ENERGY-BASED TRANSFORMER TRAINING FOR STREET FIGHTER
TRAINING APPROACH:
- Energy landscape learning (replaces PPO entirely)
- Contrastive energy training
- System 2 thinking optimization
- Verifier-based action scoring
IMPLEMENTS: Pure Energy-Based Transformer methodology from the paper
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from datetime import datetime
from collections import deque, defaultdict
import logging
import time
import json
import pickle
from tqdm import tqdm  # Add progress bar

# Import energy-based components
from wrapper import (
    EnergyBasedStreetFighterVerifier,
    EnergyBasedAgent,
    StreetFighterVisionWrapper,
    verify_energy_flow,
    ensure_scalar,
    safe_bool_check,
    sanitize_array,
    make_env,
    VECTOR_FEATURE_DIM,
    BASE_VECTOR_FEATURE_DIM,
    ENHANCED_VECTOR_FEATURE_DIM,
    BAIT_PUNISH_AVAILABLE,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnergyTrainingCallback:
    """
    Callback for monitoring Energy-Based Transformer training.
    Tracks energy landscape learning and thinking optimization.
    """

    def __init__(self, save_freq=10000, save_path="./models/", verbose=1):
        self.save_freq = save_freq
        self.save_path = save_path
        self.verbose = verbose
        os.makedirs(save_path, exist_ok=True)

        # Energy training metrics
        self.energy_losses = []
        self.contrastive_margins = []
        self.positive_energies = []
        self.negative_energies = []
        self.gradient_norms = []

        # Thinking process metrics
        self.thinking_steps_taken = []
        self.energy_improvements = []
        self.early_stops = 0
        self.energy_explosions = 0
        self.thinking_episodes = 0

        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.win_rates = []

        # Best model tracking
        self.best_energy_loss = float("inf")
        self.best_win_rate = 0.0
        self.best_thinking_improvement = 0.0

        # Training start time
        self.training_start_time = None

        # Feature dimension tracking
        self.current_feature_dim = VECTOR_FEATURE_DIM
        self.feature_dimension_changes = 0

        # Energy landscape statistics
        self.energy_landscape_quality = 0.0
        self.verifier_accuracy = 0.0
        self.action_diversity = 0.0

    def on_training_start(self):
        """Initialize Energy-Based Transformer training."""
        self.training_start_time = datetime.now()

        print(f"🧠 ENERGY-BASED TRANSFORMER TRAINING STARTED")
        print(f"🎯 ENERGY TRAINING OBJECTIVES:")
        print(f"   - Energy Loss: Minimize contrastive energy")
        print(f"   - Thinking Quality: Improve optimization steps")
        print(f"   - Action Diversity: Explore energy landscape")
        print(f"   - Win Rate: Maximize game performance")
        print(f"🧠 Feature System:")
        print(f"   - Current dimension: {VECTOR_FEATURE_DIM}")
        print(
            f"   - Bait-punish: {'Available' if BAIT_PUNISH_AVAILABLE else 'Not available'}"
        )

    def on_episode_end(self, episode_num: int, episode_data: dict):
        """Process end of episode data."""

        # Extract episode metrics
        episode_reward = episode_data.get("total_reward", 0.0)
        episode_length = episode_data.get("episode_length", 0)
        win = episode_data.get("win", False)

        # Energy metrics
        energy_loss = episode_data.get("energy_loss", 0.0)
        positive_energy = episode_data.get("avg_positive_energy", 0.0)
        negative_energy = episode_data.get("avg_negative_energy", 0.0)
        gradient_norm = episode_data.get("avg_gradient_norm", 0.0)

        # Thinking metrics
        thinking_data = episode_data.get("thinking_stats", {})
        avg_thinking_steps = thinking_data.get("avg_thinking_steps", 0.0)
        avg_energy_improvement = thinking_data.get("avg_energy_improvement", 0.0)
        episode_early_stops = thinking_data.get("early_stops", 0)
        episode_energy_explosions = thinking_data.get("energy_explosions", 0)

        # Update tracking
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.win_rates.append(1.0 if win else 0.0)

        self.energy_losses.append(energy_loss)
        self.positive_energies.append(positive_energy)
        self.negative_energies.append(negative_energy)
        self.gradient_norms.append(gradient_norm)

        self.thinking_steps_taken.append(avg_thinking_steps)
        self.energy_improvements.append(avg_energy_improvement)
        self.early_stops += episode_early_stops
        self.energy_explosions += episode_energy_explosions
        self.thinking_episodes += 1

        # Keep recent history only
        max_history = 1000
        for history_list in [
            self.episode_rewards,
            self.episode_lengths,
            self.win_rates,
            self.energy_losses,
            self.positive_energies,
            self.negative_energies,
            self.gradient_norms,
            self.thinking_steps_taken,
            self.energy_improvements,
        ]:
            if len(history_list) > max_history:
                history_list.pop(0)

        # Calculate energy landscape quality
        self._calculate_energy_landscape_quality()

        # Detailed reporting
        if episode_num % 100 == 0:
            self._log_energy_training_report(episode_num)

        # Save checkpoints
        if episode_num > 0 and episode_num % self.save_freq == 0:
            self._save_energy_checkpoint(episode_num)

    def _calculate_energy_landscape_quality(self):
        """Calculate the quality of the learned energy landscape."""
        if len(self.positive_energies) < 10 or len(self.negative_energies) < 10:
            self.energy_landscape_quality = 0.0
            return

        # Calculate separation between positive and negative energies
        recent_positive = np.mean(self.positive_energies[-10:])
        recent_negative = np.mean(self.negative_energies[-10:])

        # Good energy landscape: positive energies < negative energies
        energy_separation = recent_negative - recent_positive

        # Calculate consistency (lower variance = better)
        positive_consistency = 1.0 / (1.0 + np.std(self.positive_energies[-10:]))
        negative_consistency = 1.0 / (1.0 + np.std(self.negative_energies[-10:]))

        # Combine metrics
        self.energy_landscape_quality = (
            energy_separation * 0.5
            + positive_consistency * 0.25
            + negative_consistency * 0.25
        )

        # Normalize to 0-100 scale
        self.energy_landscape_quality = max(
            0, min(100, self.energy_landscape_quality * 20)
        )

    def _log_energy_training_report(self, episode_num: int):
        """Log detailed Energy-Based Transformer training report."""
        print(f"\n📊 ENERGY-BASED TRANSFORMER REPORT - Episode {episode_num:,}")
        print("=" * 70)

        # Training time
        if self.training_start_time:
            elapsed = datetime.now() - self.training_start_time
            hours = elapsed.total_seconds() / 3600
            print(f"⏱️  Training Time: {hours:.1f} hours")

        # Feature system
        print(f"🧠 Feature System:")
        print(f"   - Current dimension: {self.current_feature_dim}")
        print(f"   - Dimension changes: {self.feature_dimension_changes}")

        # Energy landscape metrics
        if self.energy_losses:
            recent_energy_loss = np.mean(self.energy_losses[-20:])
            print(f"⚡ Energy Loss: {recent_energy_loss:.6f}")

            if recent_energy_loss < 0.1:
                print(f"   ✅ EXCELLENT - Energy landscape well-learned!")
            elif recent_energy_loss < 0.5:
                print(f"   👍 GOOD - Energy landscape learning")
            elif recent_energy_loss < 1.0:
                print(f"   ⚠️  FAIR - Energy landscape developing")
            else:
                print(f"   🚨 POOR - Energy landscape needs work")

        if self.positive_energies and self.negative_energies:
            recent_positive = np.mean(self.positive_energies[-20:])
            recent_negative = np.mean(self.negative_energies[-20:])
            energy_separation = recent_negative - recent_positive

            print(f"🎯 Energy Separation: {energy_separation:.6f}")
            print(f"   - Positive (good actions): {recent_positive:.6f}")
            print(f"   - Negative (bad actions): {recent_negative:.6f}")

            if energy_separation > 1.0:
                print(f"   ✅ EXCELLENT - Clear energy distinction!")
            elif energy_separation > 0.5:
                print(f"   👍 GOOD - Energy landscape forming")
            elif energy_separation > 0.0:
                print(f"   ⚠️  FAIR - Weak energy separation")
            else:
                print(f"   🚨 POOR - No energy separation")

        print(f"🏔️  Energy Landscape Quality: {self.energy_landscape_quality:.1f}/100")

        # Thinking process metrics
        if self.thinking_steps_taken:
            recent_thinking_steps = np.mean(self.thinking_steps_taken[-20:])
            print(f"🤔 Thinking Process:")
            print(f"   - Average steps: {recent_thinking_steps:.2f}")

            if recent_thinking_steps > 3.0:
                print(f"   🧠 DEEP thinking - thorough optimization")
            elif recent_thinking_steps > 1.5:
                print(f"   👍 GOOD thinking - moderate optimization")
            else:
                print(f"   ⚡ FAST thinking - quick decisions")

        if self.energy_improvements:
            recent_improvement = np.mean(self.energy_improvements[-20:])
            print(f"   - Energy improvement: {recent_improvement:.6f}")

            if recent_improvement > 0.1:
                print(f"   ✅ EXCELLENT - Thinking very effective!")
            elif recent_improvement > 0.01:
                print(f"   👍 GOOD - Thinking helping")
            else:
                print(f"   ⚠️  MINIMAL - Thinking needs improvement")

        # Training stability
        print(f"🛡️  Training Stability:")
        if self.thinking_episodes > 0:
            early_stop_rate = self.early_stops / self.thinking_episodes
            explosion_rate = self.energy_explosions / self.thinking_episodes

            print(f"   - Early stops: {early_stop_rate:.1%}")
            print(f"   - Energy explosions: {explosion_rate:.1%}")

            if explosion_rate < 0.01:
                print(f"   ✅ STABLE - No energy explosions")
            elif explosion_rate < 0.05:
                print(f"   👍 MOSTLY STABLE - Few explosions")
            else:
                print(f"   ⚠️  UNSTABLE - Frequent explosions")

        # Performance metrics
        if self.episode_rewards:
            recent_rewards = self.episode_rewards[-10:]
            avg_reward = np.mean(recent_rewards)
            reward_std = np.std(recent_rewards)
            print(f"🎮 Performance (last 10 episodes):")
            print(f"   - Average reward: {avg_reward:.2f} ± {reward_std:.2f}")

        if self.win_rates:
            recent_win_rate = np.mean(self.win_rates[-10:])
            print(f"🏆 Win Rate: {recent_win_rate:.1%}")

            if recent_win_rate > 0.7:
                print(f"   🔥 DOMINATING!")
            elif recent_win_rate > 0.5:
                print(f"   👍 COMPETITIVE!")
            elif recent_win_rate > 0.3:
                print(f"   💪 LEARNING!")
            else:
                print(f"   🎯 DEVELOPING...")

        print()

    def _save_energy_checkpoint(self, episode_num: int):
        """Save Energy-Based Transformer checkpoint with model files."""
        current_energy_loss = (
            np.mean(self.energy_losses[-10:]) if self.energy_losses else float("inf")
        )
        current_win_rate = np.mean(self.win_rates[-10:]) if self.win_rates else 0.0
        current_thinking_improvement = (
            np.mean(self.energy_improvements[-10:]) if self.energy_improvements else 0.0
        )

        # Basic checkpoint info
        feature_suffix = (
            "enhanced"
            if self.current_feature_dim == ENHANCED_VECTOR_FEATURE_DIM
            else "base"
        )
        checkpoint_name = f"energy_transformer_{episode_num}_{feature_suffix}"

        print(f"💾 Energy Checkpoint: {checkpoint_name}")

        # Track best models
        if current_energy_loss < self.best_energy_loss:
            self.best_energy_loss = current_energy_loss
            print(f"   🎯 NEW BEST energy loss: {current_energy_loss:.6f}")

        if current_win_rate > self.best_win_rate:
            self.best_win_rate = current_win_rate
            print(f"   🎯 NEW BEST win rate: {current_win_rate:.1%}")

        if current_thinking_improvement > self.best_thinking_improvement:
            self.best_thinking_improvement = current_thinking_improvement
            print(
                f"   🎯 NEW BEST thinking improvement: {current_thinking_improvement:.6f}"
            )

        # Current metrics
        print(
            f"   📊 Current Energy Landscape Quality: {self.energy_landscape_quality:.1f}/100"
        )

        # Return checkpoint info for trainer to save actual model
        return {
            "checkpoint_name": checkpoint_name,
            "feature_suffix": feature_suffix,
            "episode_num": episode_num,
            "energy_loss": current_energy_loss,
            "win_rate": current_win_rate,
            "thinking_improvement": current_thinking_improvement,
            "is_best_energy": current_energy_loss < self.best_energy_loss,
            "is_best_win_rate": current_win_rate > self.best_win_rate,
            "is_best_thinking": current_thinking_improvement
            > self.best_thinking_improvement,
        }

    def get_training_stats(self) -> dict:
        """Get comprehensive training statistics."""
        return {
            "energy_loss": (
                np.mean(self.energy_losses[-10:]) if self.energy_losses else 0.0
            ),
            "energy_landscape_quality": self.energy_landscape_quality,
            "avg_thinking_steps": (
                np.mean(self.thinking_steps_taken[-10:])
                if self.thinking_steps_taken
                else 0.0
            ),
            "avg_energy_improvement": (
                np.mean(self.energy_improvements[-10:])
                if self.energy_improvements
                else 0.0
            ),
            "win_rate": np.mean(self.win_rates[-10:]) if self.win_rates else 0.0,
            "early_stop_rate": self.early_stops / max(1, self.thinking_episodes),
            "explosion_rate": self.energy_explosions / max(1, self.thinking_episodes),
            "best_energy_loss": self.best_energy_loss,
            "best_win_rate": self.best_win_rate,
            "best_thinking_improvement": self.best_thinking_improvement,
        }


class EnergyBasedTrainer:
    """
    Energy-Based Transformer trainer for Street Fighter.
    Implements the core EBT training loop with contrastive learning.
    """

    def __init__(
        self,
        verifier: EnergyBasedStreetFighterVerifier,
        agent: EnergyBasedAgent,
        learning_rate: float = 1e-4,
        contrastive_margin: float = 1.0,
        batch_size: int = 32,
        device: str = "auto",
    ):

        self.verifier = verifier
        self.agent = agent
        self.learning_rate = learning_rate
        self.contrastive_margin = contrastive_margin
        self.batch_size = batch_size

        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.verifier.to(self.device)

        # Optimizer for energy landscape learning
        self.optimizer = optim.Adam(
            self.verifier.parameters(),
            lr=learning_rate,
            weight_decay=1e-5,  # Small regularization
            eps=1e-8,
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=1000
        )

        # Experience buffer for contrastive learning
        self.experience_buffer = deque(maxlen=10000)

        # Training statistics
        self.training_stats = {
            "total_updates": 0,
            "energy_loss": 0.0,
            "positive_energy": 0.0,
            "negative_energy": 0.0,
            "gradient_norm": 0.0,
            "learning_rate": learning_rate,
        }

        print(f"✅ EnergyBasedTrainer initialized")
        print(f"   - Device: {self.device}")
        print(f"   - Learning rate: {learning_rate}")
        print(f"   - Contrastive margin: {contrastive_margin}")
        print(f"   - Batch size: {batch_size}")

    def add_experience(
        self, observations: dict, action: int, reward: float, done: bool
    ):
        """Add experience to the buffer for contrastive learning."""

        # Convert observations to tensors
        obs_tensor = {}
        for key, value in observations.items():
            if isinstance(value, np.ndarray):
                obs_tensor[key] = torch.from_numpy(value).float()
            else:
                obs_tensor[key] = torch.tensor(value).float()

        # Create one-hot action representation
        action_onehot = torch.zeros(self.verifier.action_dim)
        action_onehot[action] = 1.0

        # Classify experience as good or bad based on reward
        is_good = reward > 0.0  # Simple threshold - can be made more sophisticated

        experience = {
            "observations": obs_tensor,
            "action": action_onehot,
            "reward": reward,
            "is_good": is_good,
            "done": done,
        }

        self.experience_buffer.append(experience)

    def train_step(self) -> dict:
        """
        Perform one training step using contrastive energy learning.
        This is the core of EBT training.
        """

        if len(self.experience_buffer) < self.batch_size * 2:
            return {"energy_loss": 0.0, "updated": False}

        # Sample good and bad experiences
        good_experiences = [exp for exp in self.experience_buffer if exp["is_good"]]
        bad_experiences = [exp for exp in self.experience_buffer if not exp["is_good"]]

        if (
            len(good_experiences) < self.batch_size // 2
            or len(bad_experiences) < self.batch_size // 2
        ):
            return {"energy_loss": 0.0, "updated": False}

        # Sample balanced batch
        good_batch = np.random.choice(
            good_experiences, size=self.batch_size // 2, replace=True
        )
        bad_batch = np.random.choice(
            bad_experiences, size=self.batch_size // 2, replace=True
        )

        # Prepare batch data
        good_obs_batch = self._prepare_observation_batch(
            [exp["observations"] for exp in good_batch]
        )
        good_actions_batch = torch.stack([exp["action"] for exp in good_batch]).to(
            self.device
        )

        bad_obs_batch = self._prepare_observation_batch(
            [exp["observations"] for exp in bad_batch]
        )
        bad_actions_batch = torch.stack([exp["action"] for exp in bad_batch]).to(
            self.device
        )

        # Calculate energies
        good_energies = self.verifier(good_obs_batch, good_actions_batch)
        bad_energies = self.verifier(bad_obs_batch, bad_actions_batch)

        # Contrastive loss: good actions should have lower energy than bad actions
        # Loss = max(0, positive_energy - negative_energy + margin)
        contrastive_loss = F.relu(
            good_energies - bad_energies + self.contrastive_margin
        )
        energy_loss = contrastive_loss.mean()

        # Additional regularization
        energy_regularization = 0.01 * (
            good_energies.pow(2).mean() + bad_energies.pow(2).mean()
        )
        total_loss = energy_loss + energy_regularization

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping for stability
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.verifier.parameters(), max_norm=1.0
        )

        self.optimizer.step()
        self.scheduler.step(energy_loss)

        # Update statistics
        self.training_stats.update(
            {
                "total_updates": self.training_stats["total_updates"] + 1,
                "energy_loss": energy_loss.item(),
                "positive_energy": good_energies.mean().item(),
                "negative_energy": bad_energies.mean().item(),
                "gradient_norm": gradient_norm.item(),
                "learning_rate": self.optimizer.param_groups[0]["lr"],
            }
        )

        return {
            "energy_loss": energy_loss.item(),
            "positive_energy": good_energies.mean().item(),
            "negative_energy": bad_energies.mean().item(),
            "gradient_norm": gradient_norm.item(),
            "updated": True,
        }

    def _prepare_observation_batch(self, obs_list: list) -> dict:
        """Prepare a batch of observations for the verifier."""
        batch_obs = {}

        # Stack observations
        for key in obs_list[0].keys():
            batch_obs[key] = torch.stack([obs[key] for obs in obs_list]).to(self.device)

        return batch_obs

    def save_model(self, filepath: str):
        """Save the trained verifier model."""
        checkpoint = {
            "verifier_state_dict": self.verifier.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "training_stats": self.training_stats,
            "contrastive_margin": self.contrastive_margin,
        }

        torch.save(checkpoint, filepath)
        print(f"💾 Model saved: {filepath}")

    def load_model(self, filepath: str):
        """Load a trained verifier model."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.verifier.load_state_dict(checkpoint["verifier_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.training_stats = checkpoint.get("training_stats", self.training_stats)
        self.contrastive_margin = checkpoint.get(
            "contrastive_margin", self.contrastive_margin
        )

        print(f"📂 Model loaded: {filepath}")

    def get_training_stats(self) -> dict:
        """Get current training statistics."""
        return self.training_stats.copy()


def run_energy_training_episode(
    env,
    agent: EnergyBasedAgent,
    trainer: EnergyBasedTrainer,
    max_steps: int = 18000,
    train_freq: int = 10,
) -> dict:
    """
    Run one training episode using Energy-Based Transformer.
    """

    obs, info = env.reset()
    total_reward = 0.0
    episode_length = 0
    done = False
    truncated = False

    # Episode tracking
    actions_taken = []
    rewards_received = []
    thinking_stats_history = []

    step_count = 0
    last_train_step = 0

    while not (done or truncated) and episode_length < max_steps:
        episode_length += 1
        step_count += 1

        # Convert observations to tensors
        obs_tensor = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                obs_tensor[key] = torch.from_numpy(value).float()
            else:
                obs_tensor[key] = torch.tensor(value).float()

        # Agent predicts action using energy-based thinking
        action, thinking_info = agent.predict(obs_tensor, deterministic=False)
        actions_taken.append(action)
        thinking_stats_history.append(thinking_info)

        # Environment step
        next_obs, reward, done, truncated, next_info = env.step(action)

        total_reward += reward
        rewards_received.append(reward)

        # Add experience to trainer's buffer
        trainer.add_experience(obs, action, reward, done or truncated)

        # Training step (periodically)
        train_result = {"updated": False}
        if step_count - last_train_step >= train_freq:
            train_result = trainer.train_step()
            last_train_step = step_count

        obs = next_obs
        info = next_info

    # Episode statistics
    win = (
        info.get("wins", 0) > info.get("losses", 0)
        if "wins" in info and "losses" in info
        else False
    )

    # Aggregate thinking statistics
    avg_thinking_stats = {}
    if thinking_stats_history:
        avg_thinking_stats = {
            "avg_thinking_steps": np.mean(
                [stats["steps_taken"] for stats in thinking_stats_history]
            ),
            "avg_energy_improvement": np.mean(
                [stats["energy_improvement"] for stats in thinking_stats_history]
            ),
            "early_stops": sum(
                [stats["early_stopped"] for stats in thinking_stats_history]
            ),
            "energy_explosions": sum(
                [stats["energy_explosion"] for stats in thinking_stats_history]
            ),
        }

    # Get final training statistics
    final_train_stats = trainer.get_training_stats()

    episode_data = {
        "total_reward": total_reward,
        "episode_length": episode_length,
        "win": win,
        "energy_loss": final_train_stats.get("energy_loss", 0.0),
        "avg_positive_energy": final_train_stats.get("positive_energy", 0.0),
        "avg_negative_energy": final_train_stats.get("negative_energy", 0.0),
        "avg_gradient_norm": final_train_stats.get("gradient_norm", 0.0),
        "thinking_stats": avg_thinking_stats,
        "actions_taken": len(set(actions_taken)),  # Action diversity
        "final_info": info,
    }

    return episode_data


def main():
    parser = argparse.ArgumentParser(
        description="Energy-Based Transformer Training - Street Fighter"
    )
    parser.add_argument(
        "--total-episodes", type=int, default=50000, help="Total training episodes"
    )
    parser.add_argument(
        "--save-freq", type=int, default=1000, help="Save frequency (episodes)"
    )
    parser.add_argument("--resume", type=str, default=None, help="Path to resume from")
    parser.add_argument("--render", action="store_true", help="Render during training")
    parser.add_argument(
        "--game", type=str, default="StreetFighterIISpecialChampionEdition-Genesis"
    )
    parser.add_argument("--state", type=str, default="ken_bison_12.state")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU usage")
    parser.add_argument(
        "--test-energy", action="store_true", help="Test energy flow before training"
    )

    # Energy-Based Transformer hyperparameters
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate for energy training",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for contrastive learning"
    )
    parser.add_argument(
        "--contrastive-margin", type=float, default=1.0, help="Contrastive loss margin"
    )
    parser.add_argument(
        "--thinking-steps", type=int, default=5, help="Number of thinking steps"
    )
    parser.add_argument(
        "--thinking-lr",
        type=float,
        default=0.1,
        help="Learning rate for thinking optimization",
    )
    parser.add_argument(
        "--train-freq", type=int, default=10, help="Training frequency (steps)"
    )

    args = parser.parse_args()

    print("🧠 ENERGY-BASED TRANSFORMER TRAINING - STREET FIGHTER")
    print("=" * 70)
    print("🎯 TRAINING APPROACH:")
    print("   - Energy landscape learning (NOT PPO)")
    print("   - Contrastive energy training")
    print("   - System 2 thinking optimization")
    print("   - Verifier-based action scoring")
    print()
    print("🛠️  ENERGY HYPERPARAMETERS:")
    print(f"   - Learning Rate: {args.learning_rate}")
    print(f"   - Batch Size: {args.batch_size}")
    print(f"   - Contrastive Margin: {args.contrastive_margin}")
    print(f"   - Thinking Steps: {args.thinking_steps}")
    print(f"   - Thinking LR: {args.thinking_lr}")
    print(f"   - Training Frequency: {args.train_freq} steps")
    print()
    print("🧠 FEATURE SYSTEM:")
    print(f"   - Base features: {BASE_VECTOR_FEATURE_DIM}")
    print(f"   - Enhanced features: {ENHANCED_VECTOR_FEATURE_DIM}")
    print(
        f"   - Current mode: {VECTOR_FEATURE_DIM} ({'Enhanced' if BAIT_PUNISH_AVAILABLE else 'Base'})"
    )
    print(
        f"   - Bait-punish: {'Available' if BAIT_PUNISH_AVAILABLE else 'Not available'}"
    )
    print()

    # Device selection
    if args.force_cpu:
        device = torch.device("cpu")
        print(f"🔧 Device: CPU (forced)")
    elif args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Device: {device} (auto-detected)")
    else:
        device = torch.device(args.device)
        print(f"🔧 Device: {device} (specified)")

    # Create environment
    render_mode = "human" if args.render else None
    env = make_env(args.game, args.state, render_mode)

    # Test environment
    print("🧪 Testing Energy-Based environment...")
    obs, info = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    visual_shape = obs["visual_obs"].shape
    vector_shape = obs["vector_obs"].shape

    print(f"   ✅ Visual obs shape: {visual_shape}")
    print(f"   ✅ Vector obs shape: {vector_shape}")
    print(f"   🧠 Feature dimension: {vector_shape[-1]}")

    if vector_shape[-1] != VECTOR_FEATURE_DIM:
        print(f"   ⚠️  WARNING: Feature dimension mismatch!")
        print(f"       Expected: {VECTOR_FEATURE_DIM}, Got: {vector_shape[-1]}")

    # Create Energy-Based Transformer components
    verifier = EnergyBasedStreetFighterVerifier(
        observation_space=env.observation_space,
        action_space=env.action_space,
        features_dim=256,
    )

    agent = EnergyBasedAgent(
        verifier=verifier,
        thinking_steps=args.thinking_steps,
        thinking_lr=args.thinking_lr,
        noise_scale=0.1,
    )

    trainer = EnergyBasedTrainer(
        verifier=verifier,
        agent=agent,
        learning_rate=args.learning_rate,
        contrastive_margin=args.contrastive_margin,
        batch_size=args.batch_size,
        device=device,
    )

    # Load model if resuming
    if args.resume and os.path.exists(args.resume):
        print(f"📂 Resuming from {args.resume}")
        try:
            trainer.load_model(args.resume)
            print(f"   ✅ Model loaded successfully")
        except Exception as e:
            print(f"   ❌ Failed to load model: {e}")
            print("   🆕 Starting fresh training...")

    # Verify energy flow if requested
    if args.test_energy:
        print("\n🔬 Testing Energy-Based Transformer flow...")
        stable = verify_energy_flow(verifier, env, device)
        if not stable:
            print("⚠️  Energy flow issues detected but proceeding with monitoring")
        else:
            print("✅ Energy flow verified - ready for Energy-Based training!")

    # Create training callback
    callback = EnergyTrainingCallback(save_freq=args.save_freq, save_path="./models/")
    callback.on_training_start()

    print("\n🚀 STARTING ENERGY-BASED TRANSFORMER TRAINING...")
    print("📊 Real-time monitoring of:")
    print("   - Energy Loss (contrastive learning)")
    print("   - Thinking Quality (optimization steps)")
    print("   - Energy Landscape Formation")
    print("   - Action Diversity & Performance")
    print("💾 Auto-saving best energy models")
    print()

    try:
        for episode in range(args.total_episodes):
            # Run training episode
            episode_data = run_energy_training_episode(
                env=env,
                agent=agent,
                trainer=trainer,
                max_steps=18000,
                train_freq=args.train_freq,
            )

            # Process episode with callback
            callback.on_episode_end(episode + 1, episode_data)

            # Save model periodically
            if (episode + 1) % args.save_freq == 0:
                feature_suffix = "enhanced" if BAIT_PUNISH_AVAILABLE else "base"
                model_path = (
                    f"./models/energy_transformer_{episode + 1}_{feature_suffix}.pt"
                )
                trainer.save_model(model_path)

        # Save final model
        feature_suffix = "enhanced" if BAIT_PUNISH_AVAILABLE else "base"
        final_path = f"./models/final_energy_transformer_{feature_suffix}.pt"
        trainer.save_model(final_path)

        print(f"🎉 Energy-Based Transformer training completed!")
        print(f"💾 Final model saved: {final_path}")

        # Final training report
        final_stats = callback.get_training_stats()
        print(f"\n📊 FINAL ENERGY-BASED TRANSFORMER RESULTS:")
        print(f"⚡ Final Energy Loss: {final_stats['energy_loss']:.6f}")
        print(
            f"🏔️  Energy Landscape Quality: {final_stats['energy_landscape_quality']:.1f}/100"
        )
        print(f"🤔 Average Thinking Steps: {final_stats['avg_thinking_steps']:.2f}")
        print(f"📈 Energy Improvement: {final_stats['avg_energy_improvement']:.6f}")
        print(f"🏆 Best Win Rate: {final_stats['best_win_rate']:.1%}")
        print(f"🛡️  Explosion Rate: {final_stats['explosion_rate']:.1%}")

        if final_stats["energy_landscape_quality"] >= 70:
            print("🎉 ENERGY LANDSCAPE LEARNING SUCCESSFUL!")
        elif final_stats["energy_landscape_quality"] >= 50:
            print("📈 PARTIAL ENERGY LANDSCAPE LEARNING - Continue training")
        else:
            print("⚠️  ENERGY LANDSCAPE NEEDS MORE TRAINING")

    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        if "pbar" in locals():
            pbar.close()
        feature_suffix = "enhanced" if BAIT_PUNISH_AVAILABLE else "base"
        interrupted_path = (
            f"./models/interrupted_energy_transformer_{feature_suffix}.pt"
        )
        trainer.save_model(interrupted_path)
        print(f"💾 Model saved: {interrupted_path}")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        if "pbar" in locals():
            pbar.close()
        import traceback

        traceback.print_exc()

        # Check for specific energy issues
        if "energy" in str(e).lower() and "explosion" in str(e).lower():
            print("🚨 CRITICAL: Energy explosion during training!")
            print("   - Try reducing learning rate")
            print("   - Try reducing thinking learning rate")
            print("   - Try increasing contrastive margin")

        if "gradient" in str(e).lower() or "nan" in str(e).lower():
            print("🚨 CRITICAL: Gradient issues detected!")
            print("   - Try reducing learning rates")
            print("   - Try stronger gradient clipping")

        feature_suffix = "enhanced" if BAIT_PUNISH_AVAILABLE else "base"
        error_path = f"./models/error_energy_transformer_{feature_suffix}.pt"
        try:
            trainer.save_model(error_path)
            print(f"💾 Model saved: {error_path}")
        except Exception as save_error:
            print(f"❌ Could not save model: {save_error}")
        raise

    finally:
        env.close()
        print("🔚 Energy-Based Transformer training session ended")


if __name__ == "__main__":
    main()
