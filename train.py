#!/usr/bin/env python3
"""
🏆 AGGRESSIVE ENERGY-BASED TRANSFORMER TRAINING SCRIPT
Focuses on winning with massive reward bonuses while maintaining energy-based learning.
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
import matplotlib.pyplot as plt

# Import the wrapper components (assuming they're in wrapper.py)
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


class AggressiveRewardManager:
    """
    🏆 Aggressive reward management system focused on winning.
    """

    def __init__(self):
        self.win_history = deque(maxlen=50)
        self.streak_counter = 0
        self.best_streak = 0
        self.performance_level = "learning"  # learning, improving, dominating

        # Aggressive reward scales
        self.base_win_reward = 20.0
        self.health_bonus_scale = 10.0
        self.speed_bonus_scale = 15.0
        self.streak_bonus_scale = 5.0
        self.damage_bonus_scale = 0.5
        self.combo_bonus_scale = 2.0

        print("🏆 Aggressive Reward Manager initialized")
        print(f"   - Base win reward: {self.base_win_reward}")
        print(f"   - Maximum possible reward: ~70.0 per win")

    def calculate_aggressive_reward(self, base_reward, info, episode_steps, won):
        """Calculate aggressive rewards focused on winning."""
        aggressive_reward = base_reward
        bonus_breakdown = {}

        if won:
            # Base win bonus
            win_bonus = self.base_win_reward
            aggressive_reward += win_bonus
            bonus_breakdown["base_win"] = win_bonus

            # Health bonus (win with more health = bigger bonus)
            player_health = info.get("agent_hp", 0)
            if player_health > 140:  # Near full health
                health_bonus = self.health_bonus_scale
                aggressive_reward += health_bonus
                bonus_breakdown["health_bonus"] = health_bonus
            elif player_health > 100:
                health_bonus = self.health_bonus_scale * 0.6
                aggressive_reward += health_bonus
                bonus_breakdown["health_bonus"] = health_bonus

            # Speed bonus (faster wins = bigger bonus)
            if episode_steps < 1000:  # Very fast win
                speed_bonus = self.speed_bonus_scale
                aggressive_reward += speed_bonus
                bonus_breakdown["speed_bonus"] = speed_bonus
            elif episode_steps < 2000:  # Fast win
                speed_bonus = self.speed_bonus_scale * 0.6
                aggressive_reward += speed_bonus
                bonus_breakdown["speed_bonus"] = speed_bonus

            # Win streak bonus
            self.streak_counter += 1
            self.best_streak = max(self.best_streak, self.streak_counter)
            if self.streak_counter >= 3:
                streak_bonus = min(self.streak_bonus_scale * self.streak_counter, 25.0)
                aggressive_reward += streak_bonus
                bonus_breakdown["streak_bonus"] = streak_bonus

            # Update win history
            self.win_history.append(1)

        else:
            # Loss - reset streak but don't punish too hard
            self.streak_counter = 0
            aggressive_reward -= 1.0  # Small penalty
            self.win_history.append(0)
            bonus_breakdown["loss_penalty"] = -1.0

        # Strategic bonuses (win or lose)
        damage_dealt = info.get("total_damage_dealt", 0)
        if damage_dealt > 0:
            damage_bonus = min(damage_dealt * self.damage_bonus_scale, 20.0)
            aggressive_reward += damage_bonus
            bonus_breakdown["damage_bonus"] = damage_bonus

        # Combo bonus
        max_combo = info.get("max_combo", 0)
        if max_combo > 2:
            combo_bonus = min(max_combo * self.combo_bonus_scale, 15.0)
            aggressive_reward += combo_bonus
            bonus_breakdown["combo_bonus"] = combo_bonus

        # Performance scaling
        current_win_rate = safe_mean(list(self.win_history), 0.0)

        if current_win_rate < 0.4:  # Struggling
            self.performance_level = "learning"
            aggressive_reward *= 1.5  # Boost rewards to encourage learning
            bonus_breakdown["learning_boost"] = 0.5
        elif current_win_rate > 0.6:  # Doing well
            self.performance_level = "dominating"
            aggressive_reward *= 0.8  # Slight reduction to prevent overconfidence
            bonus_breakdown["excellence_factor"] = 0.8
        else:
            self.performance_level = "improving"

        return aggressive_reward, bonus_breakdown

    def get_stats(self):
        """Get reward manager statistics."""
        current_win_rate = safe_mean(list(self.win_history), 0.0)
        return {
            "current_win_rate": current_win_rate,
            "current_streak": self.streak_counter,
            "best_streak": self.best_streak,
            "performance_level": self.performance_level,
            "games_played": len(self.win_history),
        }


class AggressiveEnergyBasedTrainer:
    """
    🚀 Aggressive Energy-Based Transformer Trainer focused on winning.
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
        self.reward_manager = None

        # Training state
        self.episode = 0
        self.total_steps = 0
        self.training_active = True

        # Performance tracking
        self.episode_rewards = []
        self.episode_wins = []
        self.energy_qualities = []
        self.reward_breakdowns = []

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        print(f"🚀 Aggressive Energy-Based Transformer Trainer initialized")
        print(f"   - Device: {self.device}")
        print(f"   - Max episodes: {args.total_episodes}")
        print(f"   - Focus: WINNING with massive rewards")

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
        """Initialize all training components with aggressive parameters."""
        print("🔧 Setting up aggressive training components...")

        # Create environment
        render_mode = "human" if self.args.render else None
        self.env = make_fixed_env(render_mode=render_mode)

        # Create verifier
        self.verifier = FixedEnergyBasedStreetFighterVerifier(
            self.env.observation_space, self.env.action_space, features_dim=256
        ).to(self.device)

        # Create agent with more aggressive thinking
        self.agent = FixedStabilizedEnergyBasedAgent(
            self.verifier,
            thinking_steps=self.args.thinking_steps,
            thinking_lr=self.args.thinking_lr,
            noise_scale=0.15,  # Slightly more exploration
        )

        # Create optimizer with higher learning rate
        self.optimizer = optim.Adam(
            self.verifier.parameters(),
            lr=self.args.lr,
            weight_decay=5e-6,  # Less regularization
        )

        # Create stability manager with more aggressive parameters
        self.stability_manager = FixedEnergyStabilityManager(
            initial_lr=self.args.lr, thinking_lr=self.args.thinking_lr
        )
        # More lenient thresholds for aggressive training
        self.stability_manager.min_win_rate = 0.15  # Lower threshold
        self.stability_manager.min_energy_quality = 3.0  # Lower threshold

        # Create experience buffer
        self.experience_buffer = DiversityExperienceBuffer(
            capacity=40000, quality_threshold=0.4  # Lower threshold for more data
        )

        # Create checkpoint manager
        checkpoint_dir = "checkpoints_aggressive"
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)

        # Create aggressive reward manager
        self.reward_manager = AggressiveRewardManager()

        print("✅ Aggressive training components initialized")

    def verify_setup(self):
        """Verify the energy flow before training."""
        print("🔬 Verifying energy flow...")

        if verify_fixed_energy_flow(self.verifier, self.env, self.device):
            print("   ✅ Energy flow verified - ready for aggressive training!")
            return True
        else:
            print("   ❌ Energy flow verification failed!")
            return False

    def calculate_contrastive_loss(self, good_batch, bad_batch, margin=2.0):
        """Calculate contrastive loss with higher margin for aggressive training."""
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
        contrastive_loss = torch.clamp(margin - energy_separation, min=0.0)

        return contrastive_loss, energy_separation

    def train_step(self):
        """Single training step with aggressive parameters."""
        # Sample balanced batch from experience buffer
        good_batch, bad_batch = self.experience_buffer.sample_balanced_batch(
            self.args.batch_size, prioritize_diversity=True
        )

        if good_batch is None or bad_batch is None:
            return 0.0, 0.0, {"message": "insufficient_data"}

        self.optimizer.zero_grad()

        # Calculate contrastive loss with higher margin
        contrastive_loss, energy_separation = self.calculate_contrastive_loss(
            good_batch, bad_batch, margin=self.args.contrastive_margin
        )

        # Backward pass
        if contrastive_loss.requires_grad:
            contrastive_loss.backward()

            # Gradient clipping (less aggressive)
            torch.nn.utils.clip_grad_norm_(self.verifier.parameters(), max_norm=2.0)

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
        """Run single episode with aggressive reward tracking."""
        obs, info = self.env.reset()
        total_reward = 0.0
        base_reward = 0.0
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
            base_reward += reward
            steps += 1
            self.total_steps += 1

            if done or truncated:
                won = info.get("wins", 0) > 0 or info.get("win_rate", 0.0) > 0.5
                break

        # Calculate aggressive reward
        aggressive_reward, bonus_breakdown = (
            self.reward_manager.calculate_aggressive_reward(
                base_reward, info, steps, won
            )
        )

        thinking_success_rate = thinking_successes / max(thinking_attempts, 1)

        return {
            "base_reward": base_reward,
            "aggressive_reward": aggressive_reward,
            "bonus_breakdown": bonus_breakdown,
            "steps": steps,
            "won": won,
            "thinking_success_rate": thinking_success_rate,
            "info": info,
        }

    def train(self):
        """Main training loop with aggressive focus on winning."""
        if not self.verify_setup():
            print("❌ Setup verification failed!")
            return

        print("\n🚀 Starting AGGRESSIVE Energy-Based Training...")
        print("🏆 Target: 60%+ win rate with massive reward bonuses")

        # Training metrics
        recent_rewards = []
        recent_wins = []
        training_losses = []
        win_streaks = []

        # Progress bar
        pbar = tqdm(range(self.args.total_episodes), desc="🏆 Aggressive Training")

        for episode in pbar:
            if not self.training_active:
                break

            self.episode = episode

            # Run episode
            episode_result = self.run_episode()

            # Track performance
            recent_rewards.append(episode_result["aggressive_reward"])
            recent_wins.append(1.0 if episode_result["won"] else 0.0)
            self.episode_rewards.append(episode_result["aggressive_reward"])
            self.episode_wins.append(1.0 if episode_result["won"] else 0.0)
            self.reward_breakdowns.append(episode_result["bonus_breakdown"])

            # Training step (if we have enough data)
            buffer_stats = self.experience_buffer.get_stats()
            if buffer_stats["total_size"] >= self.args.batch_size * 2:
                loss, separation, train_info = self.train_step()
                training_losses.append(loss)

                # Check energy separation quality
                energy_quality = abs(separation) * 100
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
                    self.experience_buffer.emergency_purge(keep_ratio=0.3)

                    # Update learning rate
                    new_lr, new_thinking_lr = self.stability_manager.get_current_lrs()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = new_lr
                    self.agent.current_thinking_lr = new_thinking_lr

                # Get reward manager stats
                reward_stats = self.reward_manager.get_stats()

                # Update progress bar with aggressive metrics
                pbar.set_postfix(
                    {
                        "WR": f"{win_rate:.2f}",
                        "Streak": f"{reward_stats['current_streak']}",
                        "Best": f"{reward_stats['best_streak']}",
                        "AR": f"{episode_result['aggressive_reward']:.1f}",
                        "Sep": f"{avg_energy_separation:.3f}",
                        "Mode": reward_stats["performance_level"][:4],
                        "Buf": buffer_stats["total_size"],
                    }
                )

                # Show detailed breakdown every 25 episodes
                if episode % 25 == 0 and episode > 0:
                    print(f"\n🏆 Episode {episode} - Aggressive Training Report:")
                    print(
                        f"   Win Rate: {win_rate:.1%} | Current Streak: {reward_stats['current_streak']} | Best Streak: {reward_stats['best_streak']}"
                    )
                    print(
                        f"   Performance Level: {reward_stats['performance_level'].upper()}"
                    )
                    if episode_result["bonus_breakdown"]:
                        print(
                            f"   Reward Breakdown: {episode_result['bonus_breakdown']}"
                        )
                    print(
                        f"   Energy Quality: {avg_energy_quality:.1f} | Separation: {avg_energy_separation:.3f}"
                    )

            else:
                # Not enough data for training yet
                reward_stats = self.reward_manager.get_stats()
                pbar.set_postfix(
                    {
                        "WR": f"{safe_mean(recent_wins[-10:], 0.0):.2f}",
                        "Streak": f"{reward_stats['current_streak']}",
                        "AR": f"{episode_result['aggressive_reward']:.1f}",
                        "Mode": "Collecting",
                        "Buf": buffer_stats["total_size"],
                    }
                )

            # Save checkpoints more frequently for aggressive training
            if episode > 0 and episode % max(25, self.args.save_freq // 2) == 0:
                win_rate = safe_mean(self.episode_wins[-50:], 0.0)
                energy_quality = safe_mean(self.energy_qualities[-20:], 0.0)

                self.checkpoint_manager.save_checkpoint(
                    self.verifier, self.agent, episode, win_rate, energy_quality
                )

        pbar.close()
        print("\n✅ Aggressive training completed!")

        # Final comprehensive statistics
        final_win_rate = safe_mean(self.episode_wins[-100:], 0.0)
        final_energy_quality = safe_mean(self.energy_qualities[-20:], 0.0)
        final_reward_stats = self.reward_manager.get_stats()

        print(f"\n📊 FINAL AGGRESSIVE TRAINING RESULTS:")
        print(f"   🏆 Final Win Rate: {final_win_rate:.1%}")
        print(f"   🔥 Best Win Streak: {final_reward_stats['best_streak']}")
        print(f"   ⚡ Energy Quality: {final_energy_quality:.1f}")
        print(
            f"   🎯 Performance Level: {final_reward_stats['performance_level'].upper()}"
        )
        print(f"   📈 Total Episodes: {len(self.episode_rewards)}")
        print(
            f"   💰 Average Aggressive Reward: {safe_mean(self.episode_rewards[-50:], 0.0):.1f}"
        )

        # Achievement analysis
        if final_win_rate >= 0.6:
            print(f"   🎉 SUCCESS: Target 60% win rate ACHIEVED!")
        elif final_win_rate >= 0.5:
            print(f"   ⭐ GOOD: Strong performance, approaching target!")
        else:
            print(f"   📈 PROGRESS: Keep training to reach 60% target!")

        # Save final checkpoint
        final_path = self.checkpoint_manager.save_checkpoint(
            self.verifier, self.agent, episode, final_win_rate, final_energy_quality
        )
        if final_path:
            print(f"💾 Final checkpoint saved: {final_path.name}")

        return {
            "final_win_rate": final_win_rate,
            "best_streak": final_reward_stats["best_streak"],
            "energy_quality": final_energy_quality,
            "total_episodes": len(self.episode_rewards),
        }


def main():
    parser = argparse.ArgumentParser(
        description="Aggressive Energy-Based Transformer Training"
    )

    # Training parameters
    parser.add_argument(
        "--total-episodes", type=int, default=300, help="Total episodes to train"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="Learning rate (higher for aggressive training)",
    )
    parser.add_argument(
        "--thinking-lr",
        type=float,
        default=0.8,
        help="Thinking learning rate (higher for aggressive training)",
    )
    parser.add_argument(
        "--thinking-steps", type=int, default=4, help="Number of thinking steps"
    )
    parser.add_argument("--batch-size", type=int, default=24, help="Batch size")
    parser.add_argument(
        "--contrastive-margin",
        type=float,
        default=3.0,
        help="Contrastive loss margin (higher for aggressive training)",
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

    print(f"🚀 Starting AGGRESSIVE Energy-Based Transformer Training")
    print(f"🏆 FOCUS: MASSIVE rewards for WINNING")
    print(f"   - Episodes: {args.total_episodes}")
    print(f"   - Learning rate: {args.lr} (aggressive)")
    print(f"   - Thinking LR: {args.thinking_lr} (aggressive)")
    print(f"   - Contrastive margin: {args.contrastive_margin} (high)")
    print(f"   - Device: {args.device}")
    print(f"   - Target: 60%+ win rate")

    # Create and run trainer
    trainer = AggressiveEnergyBasedTrainer(args)
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

    # Start aggressive training
    results = trainer.train()

    # Final success message
    if results["final_win_rate"] >= 0.6:
        print(f"\n🎉🏆 MISSION ACCOMPLISHED! 🏆🎉")
        print(f"🥇 Win rate: {results['final_win_rate']:.1%}")
        print(f"🔥 Best streak: {results['best_streak']}")
        print(f"⚡ Energy quality: {results['energy_quality']:.1f}")
    else:
        print(f"\n📈 Good progress! Continue training to reach 60% target.")
        print(
            f"Current: {results['final_win_rate']:.1%} | Best streak: {results['best_streak']}"
        )


if __name__ == "__main__":
    main()
