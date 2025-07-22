#!/usr/bin/env python3
"""
🚀 Energy-Based Transformer Training for Street Fighter
Based on energy-based transformers: https://energy-based-transformers.github.io/
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import time
import os
from collections import deque
from pathlib import Path
import logging
from datetime import datetime

# Import energy-based transformer components
from wrapper import (
    make_env,
    verify_transformer_flow,
    EnergyBasedTransformer,
    QualityBasedExperienceBuffer,
    CheckpointManager,
    safe_mean,
    safe_std,
    safe_divide,
    MAX_FIGHT_STEPS,
    TRANSFORMER_DIM,
    TRANSFORMER_HEADS,
    TRANSFORMER_LAYERS,
)


class EnergyBasedTransformerTrainer:
    """🚀 Energy-Based Transformer trainer for Street Fighter."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize environment with render flag
        print(f"🎮 Initializing environment...")
        print(f"   - Render enabled: {args.render}")
        self.env = make_env(render=args.render)

        # Initialize energy-based transformer
        self.model = EnergyBasedTransformer(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
        ).to(self.device)

        # Verify model works
        if not verify_transformer_flow(
            self.model, self.env.observation_space, self.env.action_space
        ):
            raise RuntimeError("Transformer flow verification failed!")

        # Initialize experience buffer
        self.experience_buffer = QualityBasedExperienceBuffer(
            capacity=args.buffer_capacity,
            quality_threshold=args.quality_threshold,
        )

        # Initialize checkpoint manager (SINGLE FOLDER)
        self.checkpoint_manager = CheckpointManager(checkpoint_dir="checkpoints")

        # Initialize optimizer with different learning rates for different components
        self.optimizer = optim.AdamW(
            [
                {
                    "params": self.model.visual_encoder.parameters(),
                    "lr": args.visual_lr,
                },
                {
                    "params": self.model.transformer.parameters(),
                    "lr": args.transformer_lr,
                },
                {"params": self.model.energy_net.parameters(), "lr": args.energy_lr},
                {
                    "params": [
                        self.model.sequence_embedding.weight,
                        self.model.positional_encoding,
                    ],
                    "lr": args.transformer_lr,
                },
                {
                    "params": self.model.action_embedding.parameters(),
                    "lr": args.energy_lr,
                },
            ],
            weight_decay=args.weight_decay,
            eps=1e-8,
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.max_episodes, eta_min=args.min_lr
        )

        # Training state
        self.episode = 0
        self.total_steps = 0
        self.best_win_rate = 0.0

        # Performance tracking
        self.win_rate_history = deque(maxlen=args.win_rate_window)
        self.loss_history = deque(maxlen=100)
        self.energy_history = deque(maxlen=100)

        # Enhanced logging (SINGLE LOG FOLDER)
        self.setup_logging()

        print(f"🚀 Energy-Based Transformer Trainer initialized")
        print(f"   - Device: {self.device}")
        print(f"   - Transformer dim: {TRANSFORMER_DIM}")
        print(f"   - Attention heads: {TRANSFORMER_HEADS}")
        print(f"   - Transformer layers: {TRANSFORMER_LAYERS}")
        print(f"   - Visual LR: {args.visual_lr:.2e}")
        print(f"   - Transformer LR: {args.transformer_lr:.2e}")
        print(f"   - Energy LR: {args.energy_lr:.2e}")

    def setup_logging(self):
        """Setup logging system (SINGLE LOG FOLDER)."""
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"energy_transformer_training_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def calculate_experience_quality(self, reward, reward_breakdown, episode_stats):
        """Calculate quality score for experience."""
        base_quality = 0.5

        # Reward component
        reward_component = min(max(reward, -1.0), 2.0) * 0.3

        # Win/loss component
        if "round_won" in reward_breakdown:
            win_component = 0.4
        elif "round_lost" in reward_breakdown:
            win_component = -0.3
        else:
            win_component = 0.0

        # Health advantage component
        health_component = reward_breakdown.get("health_advantage", 0.0) * 0.1

        # Damage component
        damage_component = min(reward_breakdown.get("damage_dealt", 0.0), 0.2)

        # Episode performance component
        if episode_stats:
            episode_component = 0.1 if episode_stats.get("won", False) else -0.1
        else:
            episode_component = 0.0

        quality_score = (
            base_quality
            + reward_component
            + win_component
            + health_component
            + damage_component
            + episode_component
        )

        return max(0.0, min(1.0, quality_score))

    def run_episode(self):
        """Run single episode and collect experiences."""
        obs, info = self.env.reset()
        done = False
        truncated = False

        episode_reward = 0.0
        episode_steps = 0
        episode_experiences = []

        # Episode-level tracking
        damage_dealt_total = 0.0
        damage_taken_total = 0.0
        round_won = False

        while (
            not done and not truncated and episode_steps < self.args.max_episode_steps
        ):
            # Convert observations to tensors
            obs_tensor = {
                key: (
                    torch.from_numpy(val).unsqueeze(0).to(self.device)
                    if isinstance(val, np.ndarray)
                    else val.unsqueeze(0).to(self.device)
                )
                for key, val in obs.items()
            }

            # Predict action using energy-based transformer
            with torch.no_grad():
                action = self.model.predict_action(
                    obs_tensor, temperature=self.args.temperature, deterministic=False
                ).item()

            # Execute action
            next_obs, reward, done, truncated, info = self.env.step(action)

            episode_reward += reward
            episode_steps += 1
            self.total_steps += 1

            # Track episode stats
            reward_breakdown = info.get("reward_breakdown", {})
            damage_dealt_total += reward_breakdown.get("damage_dealt", 0.0)
            damage_taken_total += abs(reward_breakdown.get("damage_taken", 0.0))

            if "round_won" in reward_breakdown:
                round_won = True

            # Calculate experience quality
            episode_stats = {
                "won": round_won,
                "damage_ratio": safe_divide(
                    damage_dealt_total, damage_taken_total + 1e-6, 1.0
                ),
            }
            quality_score = self.calculate_experience_quality(
                reward, reward_breakdown, episode_stats
            )

            # Store experience
            experience = {
                "obs": obs,
                "action": action,
                "reward": reward,
                "next_obs": next_obs,
                "done": done,
                "episode": self.episode,
                "step": episode_steps,
            }

            episode_experiences.append((experience, quality_score))
            obs = next_obs

        # Episode completed - process experiences
        episode_stats_final = {
            "won": round_won,
            "damage_ratio": safe_divide(
                damage_dealt_total, damage_taken_total + 1e-6, 1.0
            ),
            "reward": episode_reward,
            "steps": episode_steps,
        }

        # Add experiences to buffer
        for experience, quality_score in episode_experiences:
            reward_breakdown = experience.get("reward_breakdown", {})
            self.experience_buffer.add_experience(
                experience, experience["reward"], reward_breakdown, quality_score
            )

        return episode_stats_final

    def calculate_energy_contrastive_loss(self, good_batch, bad_batch, margin=2.0):
        """Calculate contrastive loss for energy-based transformer."""
        device = self.device

        def process_batch(batch):
            if not batch:
                return None, None

            obs_batch = []
            action_batch = []

            for exp in batch:
                obs = exp["obs"]
                action = exp["action"]

                # Convert observations to tensors
                if isinstance(obs, dict):
                    obs_tensor = {
                        key: (
                            torch.from_numpy(val).float()
                            if isinstance(val, np.ndarray)
                            else val.float()
                        )
                        for key, val in obs.items()
                    }
                else:
                    obs_tensor = torch.from_numpy(obs).float()

                obs_batch.append(obs_tensor)
                action_batch.append(action)

            return obs_batch, action_batch

        # Process batches
        good_obs, good_actions = process_batch(good_batch)
        bad_obs, bad_actions = process_batch(bad_batch)

        if good_obs is None or bad_obs is None:
            return torch.tensor(0.0, device=device), {}

        # Stack observations
        def stack_obs_dict(obs_list):
            stacked = {}
            for key in obs_list[0].keys():
                stacked[key] = torch.stack([obs[key] for obs in obs_list]).to(device)
            return stacked

        good_obs_stacked = stack_obs_dict(good_obs)
        bad_obs_stacked = stack_obs_dict(bad_obs)
        good_actions_tensor = torch.tensor(good_actions, device=device)
        bad_actions_tensor = torch.tensor(bad_actions, device=device)

        # Calculate energies
        good_energies = self.model(good_obs_stacked, good_actions_tensor)
        bad_energies = self.model(bad_obs_stacked, bad_actions_tensor)

        # Contrastive loss: good experiences should have lower energy than bad ones
        good_energy_mean = good_energies.mean()
        bad_energy_mean = bad_energies.mean()

        energy_separation = bad_energy_mean - good_energy_mean
        contrastive_loss = torch.clamp(margin - energy_separation, min=0.0)

        # Add regularization
        energy_reg = 0.01 * (good_energies.pow(2).mean() + bad_energies.pow(2).mean())

        # Add diversity loss to encourage exploration
        good_energy_std = (
            good_energies.std()
            if len(good_energies) > 1
            else torch.tensor(0.0, device=device)
        )
        bad_energy_std = (
            bad_energies.std()
            if len(bad_energies) > 1
            else torch.tensor(0.0, device=device)
        )
        diversity_loss = 0.001 * torch.clamp(
            1.0 - (good_energy_std + bad_energy_std), min=0.0
        )

        total_loss = contrastive_loss + energy_reg + diversity_loss

        loss_info = {
            "contrastive_loss": contrastive_loss.item(),
            "energy_reg": energy_reg.item(),
            "diversity_loss": diversity_loss.item(),
            "good_energy_mean": good_energy_mean.item(),
            "bad_energy_mean": bad_energy_mean.item(),
            "energy_separation": energy_separation.item(),
            "good_energy_std": good_energy_std.item(),
            "bad_energy_std": bad_energy_std.item(),
        }

        return total_loss, loss_info

    def train_step(self):
        """Perform single training step."""
        # Sample balanced batch
        good_batch, bad_batch = self.experience_buffer.sample_balanced_batch(
            self.args.batch_size
        )

        if good_batch is None or bad_batch is None:
            return None  # Not enough experiences yet

        # Calculate loss
        loss, loss_info = self.calculate_energy_contrastive_loss(
            good_batch, bad_batch, margin=self.args.contrastive_margin
        )

        # Backward pass with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=1.0
        )

        # Check for gradient explosion
        if grad_norm > 10.0:
            print(f"⚠️ Large gradient norm detected: {grad_norm:.2f}")
            return None

        self.optimizer.step()

        # Add gradient norm to loss info
        loss_info["grad_norm"] = grad_norm.item()
        loss_info["learning_rate"] = self.optimizer.param_groups[0]["lr"]

        return loss_info

    def evaluate_performance(self):
        """Evaluate current model performance."""
        eval_episodes = min(5, max(1, self.episode // 100))

        wins = 0
        total_reward = 0.0
        total_steps = 0

        self.model.eval()

        for _ in range(eval_episodes):
            obs, info = self.env.reset()
            done = False
            truncated = False
            episode_reward = 0.0
            episode_steps = 0

            while (
                not done
                and not truncated
                and episode_steps < self.args.max_episode_steps
            ):
                # Convert observations to tensors
                obs_tensor = {
                    key: (
                        torch.from_numpy(val).unsqueeze(0).to(self.device)
                        if isinstance(val, np.ndarray)
                        else val.unsqueeze(0).to(self.device)
                    )
                    for key, val in obs.items()
                }

                with torch.no_grad():
                    action = self.model.predict_action(
                        obs_tensor, deterministic=True
                    ).item()

                obs, reward, done, truncated, info = self.env.step(action)
                episode_reward += reward
                episode_steps += 1

                # Check for win
                reward_breakdown = info.get("reward_breakdown", {})
                if "round_won" in reward_breakdown:
                    wins += 1
                    break

            total_reward += episode_reward
            total_steps += episode_steps

        self.model.train()

        win_rate = wins / eval_episodes
        avg_reward = total_reward / eval_episodes
        avg_steps = total_steps / eval_episodes

        return {
            "win_rate": win_rate,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
            "eval_episodes": eval_episodes,
        }

    def train(self):
        """Main training loop."""
        print(f"🚀 Starting Energy-Based Transformer Training")
        print(f"   - Target episodes: {self.args.max_episodes}")
        print(f"   - Quality threshold: {self.args.quality_threshold}")
        print(f"   - Temperature: {self.args.temperature}")

        # Training metrics
        episode_rewards = deque(maxlen=100)
        recent_losses = deque(maxlen=50)
        training_start_time = time.time()

        for episode in range(self.args.max_episodes):
            self.episode = episode
            episode_start_time = time.time()

            # Run episode
            episode_stats = self.run_episode()
            episode_rewards.append(episode_stats["reward"])

            # Training step if we have enough experiences
            if (
                len(self.experience_buffer.good_experiences)
                >= self.args.batch_size // 4
            ):
                train_stats = self.train_step()
                if train_stats:
                    recent_losses.append(train_stats.get("contrastive_loss", 0.0))
                    self.loss_history.append(train_stats.get("contrastive_loss", 0.0))
                    self.energy_history.append(
                        train_stats.get("energy_separation", 0.0)
                    )
            else:
                train_stats = {}

            # Update learning rate
            self.scheduler.step()

            # Periodic evaluation
            if episode % self.args.eval_frequency == 0:
                # Performance evaluation
                performance_stats = self.evaluate_performance()
                self.win_rate_history.append(performance_stats["win_rate"])

                # Buffer stats
                buffer_stats = self.experience_buffer.get_stats()

                # Energy stats
                avg_energy_separation = (
                    safe_mean(list(self.energy_history)[-10:], 0.0)
                    if self.energy_history
                    else 0.0
                )
                avg_loss = (
                    safe_mean(list(recent_losses)[-10:], 0.0) if recent_losses else 0.0
                )

                # Checkpoint saving
                if (
                    episode % self.args.checkpoint_frequency == 0
                    or performance_stats["win_rate"] > self.best_win_rate
                ):

                    self.checkpoint_manager.save_checkpoint(
                        self.model, episode, performance_stats["win_rate"], avg_loss
                    )

                    if performance_stats["win_rate"] > self.best_win_rate:
                        self.best_win_rate = performance_stats["win_rate"]
                        print(f"🏆 NEW BEST WIN RATE: {self.best_win_rate:.3f}")

                # Progress logging
                episode_time = time.time() - episode_start_time
                total_time = time.time() - training_start_time

                print(
                    f"\n🚀 Episode {episode} ({episode_time:.1f}s, total: {total_time/60:.1f}m)"
                )
                print(
                    f"   Win Rate: {performance_stats['win_rate']:.1%} (best: {self.best_win_rate:.1%})"
                )
                print(f"   Avg Reward: {performance_stats['avg_reward']:.2f}")
                print(f"   Avg Loss: {avg_loss:.4f}")
                print(f"   Energy Separation: {avg_energy_separation:.4f}")
                print(
                    f"   Buffer: {buffer_stats['good_count']} good, {buffer_stats['bad_count']} bad"
                )
                print(f"   Quality: {buffer_stats['avg_quality_score']:.3f}")

                if train_stats:
                    print(f"   LR: {train_stats.get('learning_rate', 0):.2e}")
                    print(f"   Grad Norm: {train_stats.get('grad_norm', 0):.3f}")

                # Log to file
                self.logger.info(
                    f"Ep {episode}: WinRate={performance_stats['win_rate']:.3f}, "
                    f"Loss={avg_loss:.4f}, Good={buffer_stats['good_count']}, "
                    f"Bad={buffer_stats['bad_count']}, Quality={buffer_stats['avg_quality_score']:.3f}"
                )

                # Check for early stopping
                if len(self.win_rate_history) >= 20:
                    recent_win_rate = safe_mean(list(self.win_rate_history)[-10:], 0.0)
                    if recent_win_rate >= self.args.target_win_rate:
                        print(
                            f"🎯 Target win rate {self.args.target_win_rate:.1%} achieved!"
                        )
                        break

        # Training completed
        final_performance = self.evaluate_performance()
        print(f"\n🏁 Energy-Based Transformer Training Completed!")
        print(f"   - Total episodes: {self.episode + 1}")
        print(f"   - Final win rate: {final_performance['win_rate']:.1%}")
        print(f"   - Best win rate: {self.best_win_rate:.1%}")

        # Save final checkpoint
        self.checkpoint_manager.save_checkpoint(
            self.model,
            self.episode,
            final_performance["win_rate"],
            safe_mean(list(recent_losses)[-10:], 0.0) if recent_losses else 0.0,
        )


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Energy-Based Transformer Training")

    # Environment arguments
    parser.add_argument(
        "--max-episodes", type=int, default=1000, help="Maximum training episodes"
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=1200, help="Maximum steps per episode"
    )
    parser.add_argument(
        "--eval-frequency", type=int, default=5, help="Evaluate every N episodes"
    )
    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=25,
        help="Save checkpoint every N episodes",
    )

    # Model arguments
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )

    # Training arguments
    parser.add_argument(
        "--visual-lr", type=float, default=1e-4, help="Learning rate for visual encoder"
    )
    parser.add_argument(
        "--transformer-lr",
        type=float,
        default=5e-5,
        help="Learning rate for transformer",
    )
    parser.add_argument(
        "--energy-lr", type=float, default=3e-4, help="Learning rate for energy network"
    )
    parser.add_argument(
        "--min-lr", type=float, default=1e-6, help="Minimum learning rate"
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Training batch size"
    )
    parser.add_argument(
        "--contrastive-margin", type=float, default=2.0, help="Contrastive loss margin"
    )

    # Experience buffer arguments
    parser.add_argument(
        "--buffer-capacity", type=int, default=30000, help="Experience buffer capacity"
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.3,
        help="Quality threshold for experiences",
    )

    # Evaluation arguments
    parser.add_argument(
        "--target-win-rate",
        type=float,
        default=0.60,
        help="Target win rate for early stopping",
    )
    parser.add_argument(
        "--win-rate-window",
        type=int,
        default=50,
        help="Window size for win rate calculation",
    )

    # Checkpoint arguments
    parser.add_argument(
        "--load-checkpoint", type=str, default=None, help="Path to checkpoint to load"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint (alias for --load-checkpoint)",
    )

    # Rendering arguments
    parser.add_argument("--render", action="store_true", help="Render the environment")

    args = parser.parse_args()

    # Handle resume alias
    if args.resume and not args.load_checkpoint:
        args.load_checkpoint = args.resume

    # Print configuration
    print(f"🚀 Energy-Based Transformer Training Configuration:")
    print(f"   Max Episodes: {args.max_episodes:,}")
    print(f"   Visual LR: {args.visual_lr:.2e}")
    print(f"   Transformer LR: {args.transformer_lr:.2e}")
    print(f"   Energy LR: {args.energy_lr:.2e}")
    print(f"   Quality Threshold: {args.quality_threshold}")
    print(f"   Temperature: {args.temperature}")
    print(f"   Target Win Rate: {args.target_win_rate:.1%}")
    print(f"   Max Fight Steps: {MAX_FIGHT_STEPS}")
    print(f"   Render: {args.render}")
    print(f"   Resume from: {args.load_checkpoint}")

    # Initialize and run trainer
    try:
        trainer = EnergyBasedTransformerTrainer(args)

        # Load checkpoint if specified
        if args.load_checkpoint:
            print(f"📂 Loading checkpoint: {args.load_checkpoint}")
            trainer.checkpoint_manager.load_checkpoint(
                Path(args.load_checkpoint), trainer.model
            )

        # Start training
        trainer.train()

    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
