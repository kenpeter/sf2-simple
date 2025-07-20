#!/usr/bin/env python3
"""
🥊 COMPLETE TRAINING SCRIPT FOR SINGLE ROUND FIGHTS
Integrates intelligent reward shaping with quality-based labeling for 60% win rate in single rounds.
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
    QualityBasedExperienceBuffer,
    CheckpointManager,
    verify_fixed_energy_flow,
    safe_mean,
    safe_std,
    VECTOR_FEATURE_DIM,
)


class SingleRoundEnergyTrainer:
    """
    🥊 Single round trainer for achieving 60% win rate using intelligent rewards.
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

        # Performance tracking for single rounds
        self.episode_rewards = deque(maxlen=100)
        self.single_round_wins = deque(maxlen=100)  # Track single round wins
        self.energy_separations = deque(maxlen=50)
        self.training_losses = deque(maxlen=50)
        self.quality_scores = deque(maxlen=200)
        self.knockout_wins = deque(maxlen=50)  # Track knockout vs timeout wins
        self.episode_lengths = deque(maxlen=100)  # Track fight duration

        # Target tracking for single rounds
        self.target_win_rate = 0.6
        self.best_win_rate = 0.0
        self.win_rate_history = deque(maxlen=20)

        # Single round specific tracking
        self.consecutive_wins = 0
        self.max_consecutive_wins = 0
        self.quick_knockout_count = 0  # Wins in < 800 steps

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        print(f"🥊 Single Round Energy Trainer initialized")
        print(f"   - Target win rate: {self.target_win_rate:.1%}")
        print(f"   - Device: {self.device}")
        print(f"   - Fight mode: Single decisive rounds")
        print(f"   - Quality-based experience labeling: ✅ ACTIVE")

    def _signal_handler(self, signum, frame):
        """Handle graceful shutdown."""
        print(f"\n🛑 Graceful shutdown initiated")
        self.training_active = False

        if self.checkpoint_manager and self.verifier and self.agent:
            self._save_emergency_checkpoint()

        sys.exit(0)

    def setup_training(self):
        """Initialize all training components with single round parameters."""
        print("🔧 Setting up single round training components...")

        # Create environment
        render_mode = "human" if self.args.render else None
        self.env = make_fixed_env(render_mode=render_mode)

        # Create verifier
        self.verifier = FixedEnergyBasedStreetFighterVerifier(
            self.env.observation_space, self.env.action_space, features_dim=256
        ).to(self.device)

        # Create agent with parameters optimized for single rounds
        self.agent = FixedStabilizedEnergyBasedAgent(
            self.verifier,
            thinking_steps=self.args.thinking_steps,
            thinking_lr=self.args.thinking_lr,
            noise_scale=0.03,  # Reduced noise for more decisive actions
        )

        # Create optimizer
        self.optimizer = optim.Adam(
            self.verifier.parameters(), lr=self.args.lr, weight_decay=1e-5, eps=1e-8
        )

        # Learning rate scheduler optimized for single rounds
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.8, patience=40  # Faster adaptation
        )

        # Create stability manager
        self.stability_manager = FixedEnergyStabilityManager(
            initial_lr=self.args.lr, thinking_lr=self.args.thinking_lr
        )

        # Create quality-based experience buffer (smaller for single rounds)
        self.experience_buffer = QualityBasedExperienceBuffer(
            capacity=25000,  # Reduced capacity for single rounds
            quality_threshold=0.58,  # Slightly higher threshold for decisive actions
        )

        # Connect experience buffer to environment
        self.env.set_experience_buffer(self.experience_buffer)

        # Create checkpoint manager
        self.checkpoint_manager = CheckpointManager("checkpoints_single_round")

        print("✅ Single round training components initialized")

    def verify_setup(self):
        """Verify energy flow before training."""
        print("🔬 Verifying energy flow for single round fights...")

        if verify_fixed_energy_flow(self.verifier, self.env, self.device):
            print("   ✅ Energy flow verified - ready for single round training!")
            return True
        else:
            print("   ❌ Energy flow verification failed!")
            return False

    def calculate_contrastive_loss(self, good_batch, bad_batch):
        """Calculate contrastive loss with quality-based experiences for single rounds."""
        if not good_batch or not bad_batch:
            return torch.tensor(0.0, device=self.device), torch.tensor(
                0.0, device=self.device
            )

        good_energies = []
        bad_energies = []

        # Process good experiences (winning actions in single rounds)
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

        # Process bad experiences (losing actions in single rounds)
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
        """Single training step optimized for single round learning."""
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

            # Gradient clipping (slightly more aggressive for single rounds)
            torch.nn.utils.clip_grad_norm_(self.verifier.parameters(), max_norm=0.8)

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

    def run_single_round(self):
        """Run single decisive round."""
        obs, info = self.env.reset()
        total_intelligent_reward = 0.0
        steps = 0
        won = False
        knockout_win = False

        thinking_successes = 0
        thinking_attempts = 0
        episode_quality_scores = []

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

            # Track quality scores
            if "quality_score" in info:
                episode_quality_scores.append(info["quality_score"])

            # Check if single round ended
            if done or truncated:
                won = info.get("single_round_won", False)

                # Determine if it was a knockout win (quick decisive victory)
                if won and steps < 800:  # Quick knockout
                    knockout_win = True
                    self.quick_knockout_count += 1

                break

        thinking_success_rate = thinking_successes / max(thinking_attempts, 1)
        avg_quality_score = safe_mean(episode_quality_scores, 0.5)

        return {
            "total_reward": total_intelligent_reward,
            "steps": steps,
            "won": won,
            "knockout_win": knockout_win,
            "thinking_success_rate": thinking_success_rate,
            "avg_quality_score": avg_quality_score,
            "info": info,
        }

    def _save_emergency_checkpoint(self):
        """Save emergency checkpoint."""
        try:
            current_win_rate = safe_mean(list(self.single_round_wins), 0.0)
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
        """Main single round training loop focused on 60% win rate."""
        if not self.verify_setup():
            print("❌ Setup verification failed!")
            return

        print(f"\n🥊 Starting SINGLE ROUND Energy-Based Training")
        print(f"🏆 Target: {self.target_win_rate:.1%} win rate in single rounds")
        print(f"📊 Using intelligent reward shaping with quality-based labeling")

        # Training metrics
        consecutive_good_rounds = 0
        episodes_since_improvement = 0

        # Progress tracking
        pbar = tqdm(range(self.args.total_episodes), desc="🥊 Single Round Training")

        for episode in pbar:
            if not self.training_active:
                break

            self.episode = episode

            # Run single round
            round_result = self.run_single_round()

            # Track performance
            self.episode_rewards.append(round_result["total_reward"])
            self.single_round_wins.append(1.0 if round_result["won"] else 0.0)
            self.quality_scores.append(round_result["avg_quality_score"])
            self.episode_lengths.append(round_result["steps"])

            if round_result["knockout_win"]:
                self.knockout_wins.append(1.0)
            else:
                self.knockout_wins.append(0.0)

            # Track consecutive wins
            if round_result["won"]:
                self.consecutive_wins += 1
                self.max_consecutive_wins = max(
                    self.max_consecutive_wins, self.consecutive_wins
                )
            else:
                self.consecutive_wins = 0

            # Training step (if we have enough data)
            buffer_stats = self.experience_buffer.get_stats()
            train_info = {"message": "collecting_data"}

            if buffer_stats["total_size"] >= self.args.batch_size * 2:
                loss, separation, train_info = self.train_step()

                if loss > 0:  # Only track meaningful losses
                    self.training_losses.append(loss)
                    self.energy_separations.append(abs(separation))

                # Calculate current performance metrics
                current_win_rate = safe_mean(list(self.single_round_wins), 0.0)
                self.win_rate_history.append(current_win_rate)

                # Update learning rate based on win rate
                self.scheduler.step(current_win_rate)

                # Check for improvement
                if current_win_rate > self.best_win_rate:
                    self.best_win_rate = current_win_rate
                    episodes_since_improvement = 0

                    # Save checkpoint for new best
                    if episode > 30:  # Save more frequently for single rounds
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

                # --- ADDED FIX: Save periodically based on save_freq ---
                if episode > 0 and episode % self.args.save_freq == 0:
                    energy_quality = safe_mean(list(self.energy_separations), 0.0)
                    print(f"\n💾 Saving periodic checkpoint at episode {episode}...")
                    self.checkpoint_manager.save_checkpoint(
                        self.verifier,
                        self.agent,
                        episode,
                        current_win_rate,
                        energy_quality,
                    )

                # Track consecutive good rounds
                if current_win_rate > 0.5:
                    consecutive_good_rounds += 1
                else:
                    consecutive_good_rounds = 0

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

                    # Emergency buffer purge
                    self.experience_buffer.emergency_purge(keep_ratio=0.4)

                # Adjust quality threshold periodically
                if (
                    episode % 40 == 0 and episode > 80
                ):  # More frequent for single rounds
                    self.experience_buffer.adjust_threshold(target_good_ratio=0.5)

                # Calculate additional metrics
                avg_episode_length = safe_mean(list(self.episode_lengths), 1000)
                knockout_rate = safe_mean(list(self.knockout_wins), 0.0)

                # Progress bar update
                pbar.set_postfix(
                    {
                        "WR": f"{current_win_rate:.2f}",
                        "Best": f"{self.best_win_rate:.2f}",
                        "Consecutive": f"{self.consecutive_wins}",
                        "KO%": f"{knockout_rate:.2f}",
                        "Steps": f"{avg_episode_length:.0f}",
                        "Quality": f"{round_result['avg_quality_score']:.2f}",
                        "Good": f"{buffer_stats['good_count']}",
                        "Bad": f"{buffer_stats['bad_count']}",
                    }
                )

                # Detailed reporting every 20 episodes (more frequent for single rounds)
                if episode % 20 == 0 and episode > 0:
                    self._print_single_round_report(
                        episode, current_win_rate, round_result, buffer_stats
                    )

                # Early success check
                if current_win_rate >= self.target_win_rate and episode > 80:
                    print(
                        f"\n🎉 TARGET ACHIEVED! Single Round Win Rate: {current_win_rate:.1%}"
                    )
                    print(
                        f"🏆 Reached {self.target_win_rate:.1%} target at episode {episode}"
                    )
                    print(f"🥊 Max consecutive wins: {self.max_consecutive_wins}")
                    print(f"⚡ Quick knockouts: {self.quick_knockout_count}")
                    break

            else:
                # Still collecting data
                current_win_rate = safe_mean(list(self.single_round_wins), 0.0)
                pbar.set_postfix(
                    {
                        "WR": f"{current_win_rate:.2f}",
                        "Won": "✅" if round_result["won"] else "❌",
                        "Steps": f"{round_result['steps']}",
                        "Quality": f"{round_result['avg_quality_score']:.2f}",
                        "Collecting": f"{buffer_stats['total_size']}/{self.args.batch_size * 2}",
                        "Good": f"{buffer_stats['good_count']}",
                        "Bad": f"{buffer_stats['bad_count']}",
                    }
                )

        pbar.close()

        # Final results
        final_win_rate = safe_mean(list(self.single_round_wins), 0.0)
        final_energy_quality = safe_mean(list(self.energy_separations), 0.0)
        final_avg_quality = safe_mean(list(self.quality_scores), 0.0)
        avg_episode_length = safe_mean(list(self.episode_lengths), 1000)
        knockout_rate = safe_mean(list(self.knockout_wins), 0.0)

        print(f"\n📊 FINAL SINGLE ROUND TRAINING RESULTS:")
        print(f"   🥊 Final Single Round Win Rate: {final_win_rate:.1%}")
        print(f"   🏆 Best Single Round Win Rate: {self.best_win_rate:.1%}")
        print(f"   🔥 Max Consecutive Wins: {self.max_consecutive_wins}")
        print(f"   ⚡ Quick Knockout Rate: {knockout_rate:.1%}")
        print(f"   ⏱️  Average Fight Duration: {avg_episode_length:.0f} steps")
        print(f"   ⚡ Energy Quality: {final_energy_quality:.3f}")
        print(f"   🎲 Average Quality Score: {final_avg_quality:.3f}")
        print(f"   📈 Total Single Rounds: {len(self.episode_rewards)}")
        print(f"   💰 Average Reward: {safe_mean(list(self.episode_rewards), 0.0):.2f}")

        # Buffer statistics
        buffer_stats = self.experience_buffer.get_stats()
        print(f"   📚 Buffer Stats:")
        print(f"      - Total experiences: {buffer_stats['total_size']}")
        print(f"      - Good experiences: {buffer_stats['good_count']}")
        print(f"      - Bad experiences: {buffer_stats['bad_count']}")
        print(f"      - Quality threshold: {buffer_stats['quality_threshold']:.3f}")

        # Success evaluation for single rounds
        if final_win_rate >= self.target_win_rate:
            print(
                f"   🎉 SUCCESS: {self.target_win_rate:.1%} SINGLE ROUND TARGET ACHIEVED!"
            )
            print(f"   🥊 AI has mastered single round combat!")
        elif final_win_rate >= 0.5:
            print(
                f"   ⭐ STRONG PROGRESS: {final_win_rate:.1%} - Close to single round mastery!"
            )
        else:
            print(
                f"   📈 GOOD FOUNDATION: Continue training for single round dominance"
            )

        # Save final checkpoint
        final_path = self.checkpoint_manager.save_checkpoint(
            self.verifier, self.agent, episode, final_win_rate, final_energy_quality
        )
        if final_path:
            print(f"💾 Final checkpoint saved: {final_path.name}")

        return {
            "final_win_rate": final_win_rate,
            "best_win_rate": self.best_win_rate,
            "max_consecutive_wins": self.max_consecutive_wins,
            "knockout_rate": knockout_rate,
            "avg_episode_length": avg_episode_length,
            "energy_quality": final_energy_quality,
            "avg_quality_score": final_avg_quality,
            "total_episodes": len(self.episode_rewards),
            "buffer_stats": buffer_stats,
        }

    def _print_single_round_report(
        self, episode, current_win_rate, round_result, buffer_stats
    ):
        """Print detailed single round training report."""
        avg_episode_length = safe_mean(list(self.episode_lengths), 1000)
        knockout_rate = safe_mean(list(self.knockout_wins), 0.0)

        print(f"\n🥊 Episode {episode} - Single Round Training Report:")
        print(
            f"   📊 Single Round Win Rate: {current_win_rate:.1%} (Best: {self.best_win_rate:.1%})"
        )
        print(f"   🏆 Last Round: {'WON' if round_result['won'] else 'LOST'}")
        print(
            f"   🔥 Consecutive Wins: {self.consecutive_wins} (Max: {self.max_consecutive_wins})"
        )
        print(f"   ⚡ Knockout Rate: {knockout_rate:.1%}")
        print(f"   ⏱️  Average Fight Duration: {avg_episode_length:.0f} steps")
        print(f"   💰 Intelligent Reward: {round_result['total_reward']:.2f}")
        print(f"   🎲 Round Quality Score: {round_result['avg_quality_score']:.3f}")
        print(f"   🧠 Thinking Success: {round_result['thinking_success_rate']:.1%}")
        print(
            f"   📚 Buffer: {buffer_stats['total_size']} total ({buffer_stats['good_count']} good, {buffer_stats['bad_count']} bad)"
        )
        print(f"   🎯 Quality Threshold: {buffer_stats['quality_threshold']:.3f}")
        print(f"   📈 Average Buffer Quality: {buffer_stats['avg_quality_score']:.3f}")

        # Progress toward target
        progress = current_win_rate / self.target_win_rate
        print(
            f"   🎯 Target Progress: {progress:.1%} toward {self.target_win_rate:.1%}"
        )

        # Fight analysis
        if round_result["won"]:
            if round_result["knockout_win"]:
                print(f"   🥊 KNOCKOUT VICTORY in {round_result['steps']} steps!")
            else:
                print(f"   🏆 Victory by decision in {round_result['steps']} steps")
        else:
            print(
                f"   💀 Defeated in {round_result['steps']} steps - analyzing for improvement"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Single Round Energy-Based Training for 60% Win Rate"
    )

    # Training parameters (optimized for single round learning)
    parser.add_argument(
        "--total-episodes", type=int, default=300, help="Total single rounds to train"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.2e-4,
        help="Learning rate (optimized for single rounds)",
    )
    parser.add_argument(
        "--thinking-lr",
        type=float,
        default=0.18,
        help="Thinking learning rate (optimized for decisive actions)",
    )
    parser.add_argument(
        "--thinking-steps", type=int, default=3, help="Number of thinking steps"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=14,
        help="Batch size (optimized for single rounds)",
    )
    parser.add_argument(
        "--contrastive-margin",
        type=float,
        default=1.3,
        help="Contrastive loss margin (optimized for decisive learning)",
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
    parser.add_argument(
        "--save-freq",
        type=int,
        default=30,
        help="Save frequency (more frequent for single rounds)",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )

    args = parser.parse_args()

    print(f"🥊 SINGLE ROUND Energy-Based Training for 60% Win Rate")
    print(f"💡 Using INTELLIGENT REWARD SHAPING + QUALITY-BASED LABELING")
    print(f"   - Single rounds: {args.total_episodes}")
    print(f"   - Learning rate: {args.lr} (optimized for single rounds)")
    print(f"   - Thinking LR: {args.thinking_lr} (optimized for decisive actions)")
    print(f"   - Contrastive margin: {args.contrastive_margin} (optimized)")
    print(f"   - Batch size: {args.batch_size} (optimized)")
    print(f"   - Device: {args.device}")
    print(f"   🎲 Quality threshold: 0.58 (optimized for decisive actions)")

    # Create and run trainer
    trainer = SingleRoundEnergyTrainer(args)
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

    # Start single round training
    results = trainer.train()

    # Final success evaluation
    if results["final_win_rate"] >= 0.6:
        print(f"\n🎉🏆 SINGLE ROUND MASTERY ACHIEVED! 🏆🎉")
        print(f"✅ Successfully achieved 60% single round win rate target!")
        print(f"🥇 Final single round win rate: {results['final_win_rate']:.1%}")
        print(f"🔥 Best single round win rate: {results['best_win_rate']:.1%}")
        print(f"🏆 Max consecutive wins: {results['max_consecutive_wins']}")
        print(f"⚡ Knockout rate: {results['knockout_rate']:.1%}")
        print(f"⏱️  Average fight duration: {results['avg_episode_length']:.0f} steps")
        print(f"🎲 Quality-based labeling worked perfectly for single rounds!")
        print(f"📊 Buffer had {results['buffer_stats']['total_size']} experiences")
        print(f"⚡ Energy separation: {results['energy_quality']:.3f}")

        # Performance analysis
        if results["knockout_rate"] > 0.3:
            print(f"🥊 DOMINANT FIGHTER: High knockout rate shows aggressive mastery!")
        if results["avg_episode_length"] < 1000:
            print(f"⚡ EFFICIENT FIGHTER: Quick decisive victories!")
        if results["max_consecutive_wins"] >= 5:
            print(f"🔥 CONSISTENT CHAMPION: Excellent win streaks!")

    else:
        print(f"\n📈 Strong single round foundation built!")
        print(
            f"Current: {results['final_win_rate']:.1%} | Best: {results['best_win_rate']:.1%}"
        )
        print(f"🥊 Single round fight stats:")
        print(f"   - Max consecutive wins: {results['max_consecutive_wins']}")
        print(f"   - Knockout rate: {results['knockout_rate']:.1%}")
        print(f"   - Average fight duration: {results['avg_episode_length']:.0f} steps")
        print(f"   - Average quality score: {results['avg_quality_score']:.3f}")
        print(f"   - Buffer experiences: {results['buffer_stats']['total_size']}")
        print(f"   - Good/bad ratio: {results['buffer_stats']['good_ratio']:.2f}")
        print(f"💡 The single round system is working - just needs more episodes!")


if __name__ == "__main__":
    main()
