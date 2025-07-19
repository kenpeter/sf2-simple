#!/usr/bin/env python3
"""
🛡️ FIXED ENERGY-BASED TRANSFORMER TRAINING SCRIPT
Uses the fixed wrapper components with proper scaling and realistic thresholds.
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

# Import the FIXED wrapper components
from wrapper import (
    make_fixed_env,
    FixedEnergyBasedStreetFighterVerifier,
    FixedStabilizedEnergyBasedAgent,
    FixedEnergyStabilityManager,
    DiversityExperienceBuffer,
    CheckpointManager,
    verify_fixed_energy_flow,
    safe_mean,
    safe_std,
    VECTOR_FEATURE_DIM,
)


class FixedEnergyBasedTrainer:
    """
    🎯 FIXED Energy-Based Transformer Trainer
    Complete training pipeline with proper scaling and realistic thresholds.
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
        self.stability_manager = None
        self.experience_buffer = None
        self.checkpoint_manager = None

        # Training state
        self.episode = 0
        self.total_steps = 0
        self.training_active = True

        # Performance tracking
        self.episode_rewards = []
        self.episode_wins = []
        self.energy_qualities = []

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        print(f"🧠 FIXED Energy-Based Transformer Trainer initialized")
        print(f"   - Device: {self.device}")
        print(f"   - Max episodes: {args.total_episodes}")
        print(f"   - Learning rate: {args.lr}")
        print(f"   - Thinking steps: {args.thinking_steps}")

    def _signal_handler(self, signum, frame):
        """Handle graceful shutdown."""
        print(f"\n🛑 Received shutdown signal ({signum})")
        print("🔄 Saving checkpoint and exiting...")
        self.training_active = False

        if self.checkpoint_manager and self.verifier and self.agent:
            emergency_path = self.checkpoint_manager.save_checkpoint(
                self.verifier,
                self.agent,
                self.episode,
                safe_mean(self.episode_wins, 0.0),
                safe_mean(self.energy_qualities, 0.0),
                is_emergency=True,
            )
            if emergency_path:
                print(f"✅ Emergency checkpoint saved: {emergency_path.name}")

        sys.exit(0)

    def setup_training(self):
        """Initialize all training components with FIXED parameters."""
        print("🔧 Setting up FIXED training components...")

        # Create environment
        render_mode = "human" if self.args.render else None
        self.env = make_fixed_env(render_mode=render_mode)

        # Create FIXED verifier
        self.verifier = FixedEnergyBasedStreetFighterVerifier(
            self.env.observation_space, self.env.action_space, features_dim=256
        ).to(self.device)

        # Create FIXED agent
        self.agent = FixedStabilizedEnergyBasedAgent(
            self.verifier,
            thinking_steps=self.args.thinking_steps,
            thinking_lr=self.args.thinking_lr,
            noise_scale=0.1,
        )

        # Create optimizer
        self.optimizer = optim.Adam(
            self.verifier.parameters(), lr=self.args.lr, weight_decay=1e-5
        )

        # Create FIXED stability manager
        self.stability_manager = FixedEnergyStabilityManager(
            initial_lr=self.args.lr, thinking_lr=self.args.thinking_lr
        )

        # Create experience buffer
        self.experience_buffer = DiversityExperienceBuffer(
            capacity=50000, quality_threshold=0.5
        )

        # Create checkpoint manager
        self.checkpoint_manager = CheckpointManager("checkpoints")

        # Set episode callback for experience buffer
        self.env.set_episode_callback(self.experience_buffer.add_episode_experiences)

        print("✅ FIXED training components initialized")

    def verify_setup(self):
        """Verify the FIXED energy flow before training."""
        print("🔬 Verifying FIXED energy flow...")

        if verify_fixed_energy_flow(self.verifier, self.env, self.device):
            print("   ✅ Energy flow is PROPERLY SCALED.")
            return True
        else:
            print("   ❌ Energy flow verification failed!")
            return False

    def calculate_contrastive_loss(self, good_batch, bad_batch, margin=1.0):
        """Calculate contrastive loss with FIXED energy scaling."""
        if not good_batch or not bad_batch:
            return torch.tensor(0.0, device=self.device)

        good_energies = []
        bad_energies = []

        # Process good experiences
        for exp in good_batch:
            obs = exp["observations"]
            action = exp["action"]

            # Convert to tensors
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

            # Convert to tensors
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
            return torch.tensor(0.0, device=self.device)

        # Stack energies
        good_energy_tensor = torch.cat(good_energies)
        bad_energy_tensor = torch.cat(bad_energies)

        # Calculate contrastive loss (good should have lower energy than bad)
        good_mean = good_energy_tensor.mean()
        bad_mean = bad_energy_tensor.mean()

        # Energy separation (bad - good should be positive and > margin)
        # energy sepration
        energy_separation = bad_mean - good_mean
        # contrastive loss
        contrastive_loss = torch.clamp(margin - energy_separation, min=0.0)

        return contrastive_loss, energy_separation

    def train_step(self):
        """Single training step with FIXED scaling."""
        # Sample balanced batch from experience buffer
        good_batch, bad_batch = self.experience_buffer.sample_balanced_batch(
            self.args.batch_size, prioritize_diversity=True
        )

        if good_batch is None or bad_batch is None:
            return 0.0, 0.0, {"message": "insufficient_data"}

        self.optimizer.zero_grad()

        # Calculate contrastive loss with FIXED scaling
        contrastive_loss, energy_separation = self.calculate_contrastive_loss(
            good_batch, bad_batch, margin=self.args.contrastive_margin
        )

        # Backward pass
        if contrastive_loss.requires_grad:
            contrastive_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.verifier.parameters(), max_norm=1.0)

            self.optimizer.step()

        return (
            contrastive_loss.item(),
            energy_separation.item(),
            {
                "good_energy": safe_mean([exp["reward"] for exp in good_batch], 0.0),
                "bad_energy": safe_mean([exp["reward"] for exp in bad_batch], 0.0),
            },
        )

    def run_episode(self):
        """Run single episode with FIXED agent."""
        obs, info = self.env.reset()
        total_reward = 0.0
        steps = 0
        won = False

        thinking_successes = 0
        thinking_attempts = 0

        while True:
            # Get action from FIXED agent
            action, thinking_info = self.agent.predict(obs, deterministic=False)

            # Track thinking process
            thinking_attempts += 1
            if thinking_info.get("optimization_successful", False):
                thinking_successes += 1

            # Take step
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            steps += 1
            self.total_steps += 1

            if done or truncated:
                won = info.get("wins", 0) > 0
                break

        thinking_success_rate = safe_mean(
            [thinking_successes / max(thinking_attempts, 1)], 0.0
        )

        return {
            "reward": total_reward,
            "steps": steps,
            "won": won,
            "thinking_success_rate": thinking_success_rate,
            "info": info,
        }

    def train(self):
        """Main training loop with FIXED components."""
        if not self.verify_setup():
            print("❌ Setup verification failed!")
            return

        print("\n🚀 Starting FIXED Energy-Based Training...")
        print("🎯 Target: Proper energy scaling and realistic thresholds")

        # Training metrics
        recent_rewards = []
        recent_wins = []
        training_losses = []

        # Progress bar
        pbar = tqdm(range(self.args.total_episodes), desc="Episodes")

        for episode in pbar:
            if not self.training_active:
                break

            self.episode = episode

            # Run episode
            episode_result = self.run_episode()

            # Track performance
            recent_rewards.append(episode_result["reward"])
            recent_wins.append(1.0 if episode_result["won"] else 0.0)
            self.episode_rewards.append(episode_result["reward"])
            self.episode_wins.append(1.0 if episode_result["won"] else 0.0)

            # Training step (if we have enough data)
            buffer_stats = self.experience_buffer.get_stats()
            if buffer_stats["total_size"] >= self.args.batch_size * 2:
                loss, separation, train_info = self.train_step()
                training_losses.append(loss)

                # Check energy separation quality
                energy_quality = abs(separation) * 100  # Convert to percentage
                self.energy_qualities.append(energy_quality)

                # Update stability manager
                win_rate = safe_mean(recent_wins[-20:], 0.0)
                avg_energy_quality = safe_mean(self.energy_qualities[-10:], 0.0)
                avg_energy_separation = abs(safe_mean([separation], 0.0))
                early_stop_rate = self.agent.get_thinking_stats().get(
                    "early_stop_rate", 0.0
                )

                emergency_triggered = self.stability_manager.update_metrics(
                    win_rate, avg_energy_quality, avg_energy_separation, early_stop_rate
                )

                if emergency_triggered:
                    print("🚨 EMERGENCY PROTOCOL TRIGGERED!")

                    # Emergency: Purge low-quality experiences
                    self.experience_buffer.emergency_purge(keep_ratio=0.3)

                    # Update learning rate
                    new_lr, new_thinking_lr = self.stability_manager.get_current_lrs()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = new_lr
                    self.agent.current_thinking_lr = new_thinking_lr

                # Update progress bar with FIXED metrics
                pbar.set_postfix(
                    {
                        "WR": f"{win_rate:.2f}",
                        "EQ": f"{avg_energy_quality:.1f}",
                        "Sep": f"{avg_energy_separation:.3f}",
                        "Loss": f"{loss:.3f}",
                        "LR": f"{self.optimizer.param_groups[0]['lr']:.1e}",
                        "Buf": buffer_stats["total_size"],
                        "G/B": f"{buffer_stats['good_count']}/{buffer_stats['bad_count']}",
                        "Mode": (
                            "Emergency"
                            if self.stability_manager.emergency_mode
                            else "Normal"
                        ),
                    }
                )

                # Show energy separation warnings (less frequently)
                if episode % 10 == 0 and abs(separation) < 0.1:
                    print(f"⚠️  Energy separation low: {separation:.6f}")
                    print(
                        f"   Good energy: {train_info.get('good_energy', 0):.6f}, "
                        f"Bad energy: {train_info.get('bad_energy', 0):.6f}"
                    )
                    print(f"   Contrastive loss: {loss:.6f}")

            else:
                # Not enough data for training yet
                pbar.set_postfix(
                    {
                        "WR": f"{safe_mean(recent_wins[-10:], 0.0):.2f}",
                        "EQ": "0.0",
                        "Sep": "0.000",
                        "Loss": "0.000",
                        "LR": f"{self.optimizer.param_groups[0]['lr']:.1e}",
                        "Buf": buffer_stats["total_size"],
                        "G/B": f"{buffer_stats['good_count']}/{buffer_stats['bad_count']}",
                        "Mode": "Collecting",
                    }
                )

            # Save checkpoints
            if episode > 0 and episode % self.args.save_freq == 0:
                win_rate = safe_mean(self.episode_wins[-100:], 0.0)
                energy_quality = safe_mean(self.energy_qualities[-20:], 0.0)

                self.checkpoint_manager.save_checkpoint(
                    self.verifier, self.agent, episode, win_rate, energy_quality
                )

        pbar.close()
        print("\n✅ Training completed!")

        # Final statistics
        final_win_rate = safe_mean(self.episode_wins[-100:], 0.0)
        final_energy_quality = safe_mean(self.energy_qualities[-20:], 0.0)

        print(f"📊 Final Results:")
        print(f"   - Win rate: {final_win_rate:.3f}")
        print(f"   - Energy quality: {final_energy_quality:.1f}")
        print(f"   - Total episodes: {len(self.episode_rewards)}")
        print(
            f"   - Emergency activations: {self.stability_manager.consecutive_poor_episodes}"
        )

        # Save final checkpoint
        final_path = self.checkpoint_manager.save_checkpoint(
            self.verifier, self.agent, episode, final_win_rate, final_energy_quality
        )
        if final_path:
            print(f"💾 Final checkpoint saved: {final_path.name}")

    def safe_reset_energy_network(self):
        """Safe energy network reset that avoids tensor size mismatches."""
        print("🛡️  SAFE ENERGY NETWORK RESET")

        # Store current settings
        current_lr = self.optimizer.param_groups[0]["lr"]
        current_thinking_lr = self.agent.current_thinking_lr
        energy_scale = self.verifier.energy_scale

        # Create new verifier with same architecture
        print("🔄 Creating new verifier instance")
        self.verifier = FixedEnergyBasedStreetFighterVerifier(
            self.env.observation_space, self.env.action_space, features_dim=256
        ).to(self.device)

        # Restore energy scale
        self.verifier.energy_scale = energy_scale

        # Create new optimizer
        self.optimizer = optim.Adam(
            self.verifier.parameters(), lr=current_lr, weight_decay=1e-5
        )

        # Update agent
        self.agent.verifier = self.verifier
        self.agent.current_thinking_lr = current_thinking_lr

        print(
            f"✅ Safe reset complete with LR: {current_lr:.2e}, Energy scale: {energy_scale:.3f}"
        )

        # Clear monitoring history
        self.stability_manager.consecutive_poor_episodes = 0


def main():
    parser = argparse.ArgumentParser(
        description="FIXED Energy-Based Transformer Training"
    )

    # Training parameters
    parser.add_argument(
        "--total-episodes", type=int, default=1000, help="Total episodes to train"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--thinking-lr", type=float, default=0.2, help="Thinking learning rate"
    )
    parser.add_argument(
        "--thinking-steps", type=int, default=3, help="Number of thinking steps"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--contrastive-margin", type=float, default=1.0, help="Contrastive loss margin"
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

    print(f"🚀 Starting FIXED Energy-Based Transformer Training")
    print(f"   - Episodes: {args.total_episodes}")
    print(f"   - Learning rate: {args.lr}")
    print(f"   - Thinking LR: {args.thinking_lr}")
    print(f"   - Thinking steps: {args.thinking_steps}")
    print(f"   - Device: {args.device}")

    # Create and run trainer
    trainer = FixedEnergyBasedTrainer(args)
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

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
