#!/usr/bin/env python3
"""
🛡️ Enhanced Energy-Based Training with Policy Memory
Prevents catastrophic forgetting through checkpoint averaging and golden experience buffer.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import time
import os
from collections import deque
from pathlib import Path
import logging
from datetime import datetime

# Import our enhanced wrapper components
from wrapper import (
    make_fixed_env,
    verify_fixed_energy_flow,
    EnhancedFixedEnergyBasedStreetFighterVerifier,
    EnhancedFixedStabilizedEnergyBasedAgent,
    EnhancedQualityBasedExperienceBuffer,
    PolicyMemoryManager,
    EnhancedEnergyStabilityManager,
    EnhancedCheckpointManager,
    safe_mean,
    safe_std,
    safe_divide,
)


class EnhancedPolicyMemoryTrainer:
    """🛡️ Enhanced trainer with policy memory and golden experience buffer."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize environment
        print(f"🎮 Initializing enhanced training environment...")
        self.env = make_fixed_env()

        # Initialize verifier and agent
        self.verifier = EnhancedFixedEnergyBasedStreetFighterVerifier(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            features_dim=args.features_dim,
        ).to(self.device)

        # Verify energy flow
        if not verify_fixed_energy_flow(
            self.verifier, self.env.observation_space, self.env.action_space
        ):
            raise RuntimeError("Energy flow verification failed!")

        self.agent = EnhancedFixedStabilizedEnergyBasedAgent(
            verifier=self.verifier,
            thinking_steps=args.thinking_steps,
            thinking_lr=args.thinking_lr,
            noise_scale=args.noise_scale,
        )

        # Initialize Policy Memory Manager (KEY COMPONENT)
        self.policy_memory = PolicyMemoryManager(
            performance_drop_threshold=args.performance_drop_threshold,
            averaging_weight=args.averaging_weight,
        )

        # Initialize enhanced experience buffer with golden buffer
        self.experience_buffer = EnhancedQualityBasedExperienceBuffer(
            capacity=args.buffer_capacity,
            quality_threshold=args.quality_threshold,
            golden_buffer_capacity=args.golden_buffer_capacity,
        )

        # Initialize stability manager with policy memory integration
        self.stability_manager = EnhancedEnergyStabilityManager(
            initial_lr=args.learning_rate,
            thinking_lr=args.thinking_lr,
            policy_memory_manager=self.policy_memory,
        )

        # Initialize enhanced checkpoint manager
        self.checkpoint_manager = EnhancedCheckpointManager(
            checkpoint_dir=args.checkpoint_dir
        )

        # Initialize optimizer with enhanced parameters
        self.optimizer = optim.Adam(
            self.verifier.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,  # Increased for better regularization
            eps=1e-8,
            betas=(0.9, 0.999),
        )

        # Training state
        self.episode = 0
        self.total_steps = 0
        self.best_win_rate = 0.0

        # Performance tracking for policy memory
        self.win_rate_history = deque(maxlen=args.win_rate_window)
        self.energy_quality_history = deque(maxlen=50)
        self.last_checkpoint_episode = 0

        # Enhanced logging
        self.setup_logging()

        print(f"🛡️ Enhanced Policy Memory Trainer initialized")
        print(f"   - Device: {self.device}")
        print(f"   - Learning rate: {args.learning_rate:.2e}")
        print(f"   - Weight decay: {args.weight_decay:.2e}")
        print(f"   - Performance drop threshold: {args.performance_drop_threshold}")
        print(f"   - Checkpoint averaging weight: {args.averaging_weight}")
        print(f"   - Golden buffer capacity: {args.golden_buffer_capacity}")

    def setup_logging(self):
        """Setup enhanced logging system."""
        log_dir = Path("logs_enhanced")
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"enhanced_training_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def calculate_experience_quality(self, reward, reward_breakdown, episode_stats):
        """Calculate quality score for experience (improved formula)."""
        base_quality = 0.5  # Neutral starting point

        # Reward component (capped to prevent exploitation)
        reward_component = min(max(reward, -1.0), 2.0) * 0.3

        # Win/loss component (most important)
        if "round_won" in reward_breakdown:
            win_component = 0.4  # Strong positive signal
        elif "round_lost" in reward_breakdown:
            win_component = -0.3  # Negative signal
        else:
            win_component = 0.0

        # Health advantage component
        health_component = reward_breakdown.get("health_advantage", 0.0) * 0.1

        # Damage dealing component (capped)
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

        # Clamp to reasonable range
        return max(0.0, min(1.0, quality_score))

    def run_episode(self):
        """Run a single episode with enhanced tracking."""
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
            # Agent prediction with enhanced thinking
            action, thinking_info = self.agent.predict(obs, deterministic=False)

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
                "thinking_info": thinking_info,
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

    def calculate_contrastive_loss(self, good_batch, bad_batch, margin=2.0):
        """Calculate enhanced contrastive loss with tighter bounds."""
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

                # Convert action to one-hot
                action_one_hot = torch.zeros(self.env.action_space.n)
                action_one_hot[action] = 1.0

                obs_batch.append(obs_tensor)
                action_batch.append(action_one_hot)

            return obs_batch, action_batch

        # Process batches
        good_obs, good_actions = process_batch(good_batch)
        bad_obs, bad_actions = process_batch(bad_batch)

        if good_obs is None or bad_obs is None:
            return torch.tensor(0.0, device=device)

        # Stack observations and actions
        def stack_obs_dict(obs_list):
            stacked = {}
            for key in obs_list[0].keys():
                stacked[key] = torch.stack([obs[key] for obs in obs_list]).to(device)
            return stacked

        good_obs_stacked = stack_obs_dict(good_obs)
        bad_obs_stacked = stack_obs_dict(bad_obs)
        good_actions_stacked = torch.stack(good_actions).to(device)
        bad_actions_stacked = torch.stack(bad_actions).to(device)

        # Calculate energies
        good_energies = self.verifier(good_obs_stacked, good_actions_stacked)
        bad_energies = self.verifier(bad_obs_stacked, bad_actions_stacked)

        # Enhanced contrastive loss with tighter bounds
        good_energy_mean = good_energies.mean()
        bad_energy_mean = bad_energies.mean()

        # We want good energies to be lower (more negative) than bad energies
        energy_diff = bad_energy_mean - good_energy_mean
        contrastive_loss = torch.clamp(margin - energy_diff, min=0.0)

        # Add regularization to prevent energy explosion
        energy_reg = 0.01 * (good_energies.pow(2).mean() + bad_energies.pow(2).mean())

        total_loss = contrastive_loss + energy_reg

        return total_loss, {
            "contrastive_loss": contrastive_loss.item(),
            "energy_reg": energy_reg.item(),
            "good_energy_mean": good_energy_mean.item(),
            "bad_energy_mean": bad_energy_mean.item(),
            "energy_separation": energy_diff.item(),
        }

    def train_step(self):
        """Enhanced training step with policy memory integration."""
        # Sample balanced batch with golden experiences
        good_batch, bad_batch, golden_batch = (
            self.experience_buffer.sample_enhanced_balanced_batch(
                self.args.batch_size, golden_ratio=0.15
            )
        )

        if good_batch is None or bad_batch is None:
            return None  # Not enough experiences yet

        # Calculate loss
        loss, loss_info = self.calculate_contrastive_loss(
            good_batch, bad_batch, margin=self.args.contrastive_margin
        )

        # Get current learning rate from stability manager
        current_lr, current_thinking_lr = self.stability_manager.get_current_lrs()

        # Update optimizer learning rate if changed
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr

        # Update agent thinking learning rate
        self.agent.current_thinking_lr = current_thinking_lr

        # Backward pass with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()

        # Enhanced gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.verifier.parameters(), max_norm=1.0
        )

        # Check for gradient explosion
        if grad_norm > 10.0:
            print(f"⚠️ Large gradient norm detected: {grad_norm:.2f}")
            return None

        self.optimizer.step()

        # Add gradient norm to loss info
        loss_info["grad_norm"] = grad_norm.item()
        loss_info["learning_rate"] = current_lr
        loss_info["thinking_lr"] = current_thinking_lr

        return loss_info

    def evaluate_performance(self):
        """Evaluate current performance for policy memory decisions."""
        eval_episodes = min(5, max(1, self.episode // 100))  # Adaptive evaluation count

        wins = 0
        total_reward = 0.0
        total_steps = 0

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
                action, _ = self.agent.predict(
                    obs, deterministic=True
                )  # Deterministic for evaluation
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

        win_rate = wins / eval_episodes
        avg_reward = total_reward / eval_episodes
        avg_steps = total_steps / eval_episodes

        return {
            "win_rate": win_rate,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
            "eval_episodes": eval_episodes,
        }

    def handle_policy_memory_operations(self, performance_stats, train_stats):
        """Handle policy memory operations - KEY METHOD."""

    def handle_policy_memory_operations(self, performance_stats, train_stats):
        """Handle policy memory operations - KEY METHOD."""
        current_win_rate = performance_stats["win_rate"]
        current_lr = self.optimizer.param_groups[0]["lr"]

        # Update policy memory with current performance
        performance_improved, performance_drop = self.policy_memory.update_performance(
            current_win_rate, self.episode, self.verifier.state_dict(), current_lr
        )

        # Update experience buffer win rate for golden buffer filtering
        self.experience_buffer.update_win_rate(current_win_rate)

        policy_memory_action_taken = False

        # Handle performance improvement
        if performance_improved:
            print(f"🏆 NEW PEAK PERFORMANCE DETECTED!")
            # Save peak checkpoint
            self.checkpoint_manager.save_checkpoint(
                self.verifier,
                self.agent,
                self.episode,
                current_win_rate,
                train_stats.get("energy_separation", 0.0),
                is_peak=True,
                policy_memory_stats=self.policy_memory.get_stats(),
            )
            policy_memory_action_taken = True

        # Handle performance drop - CRITICAL POLICY MEMORY LOGIC
        elif performance_drop:
            print(f"📉 PERFORMANCE DROP DETECTED - Activating Policy Memory!")

            # Attempt checkpoint averaging if we should
            if self.policy_memory.should_perform_averaging(self.episode):
                print(f"🔄 Performing checkpoint averaging...")
                averaging_success = self.policy_memory.perform_checkpoint_averaging(
                    self.verifier
                )

                if averaging_success:
                    print(f"✅ Checkpoint averaging completed successfully")
                    policy_memory_action_taken = True

                    # Also reduce learning rate
                    if self.policy_memory.should_reduce_lr():
                        new_lr = self.policy_memory.get_reduced_lr(current_lr)
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = new_lr

                        # Update stability manager
                        self.stability_manager.current_lr = new_lr
                        print(
                            f"📉 Learning rate reduced to {new_lr:.2e} after policy memory recovery"
                        )
                else:
                    print(f"❌ Checkpoint averaging failed")

            # Save emergency checkpoint
            self.checkpoint_manager.save_checkpoint(
                self.verifier,
                self.agent,
                self.episode,
                current_win_rate,
                train_stats.get("energy_separation", 0.0),
                is_emergency=True,
                policy_memory_stats=self.policy_memory.get_stats(),
            )

        return policy_memory_action_taken

    def train(self):
        """Main enhanced training loop with policy memory."""
        print(f"🛡️ Starting Enhanced Policy Memory Training")
        print(f"   - Target episodes: {self.args.max_episodes}")
        print(f"   - Batch size: {self.args.batch_size}")
        print(f"   - Contrastive margin: {self.args.contrastive_margin}")
        print(f"   - Policy memory enabled: ✅")

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
                >= self.args.batch_size // 2
            ):
                train_stats = self.train_step()
                if train_stats:
                    recent_losses.append(train_stats.get("contrastive_loss", 0.0))
            else:
                train_stats = {}

            # Periodic evaluation and policy memory operations
            if episode % self.args.eval_frequency == 0:
                # Performance evaluation
                performance_stats = self.evaluate_performance()
                self.win_rate_history.append(performance_stats["win_rate"])

                # Calculate energy quality metrics
                energy_separation = train_stats.get("energy_separation", 0.0)
                energy_quality = abs(energy_separation) * 10.0  # Scale for readability
                self.energy_quality_history.append(energy_quality)

                # Policy memory operations (CRITICAL SECTION)
                policy_memory_action = self.handle_policy_memory_operations(
                    performance_stats, train_stats
                )

                # Update stability manager
                early_stop_rate = safe_divide(
                    self.agent.thinking_stats.get("early_stops", 0),
                    self.agent.thinking_stats.get("total_predictions", 1),
                    0.0,
                )

                stability_emergency = self.stability_manager.update_metrics(
                    performance_stats["win_rate"],
                    energy_quality,
                    energy_separation,
                    early_stop_rate,
                )

                # Handle stability emergency
                if stability_emergency and not policy_memory_action:
                    print(f"🚨 Stability emergency triggered!")
                    # Update learning rates from stability manager
                    new_lr, new_thinking_lr = self.stability_manager.get_current_lrs()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = new_lr
                    self.agent.current_thinking_lr = new_thinking_lr

                # Regular checkpoint saving
                if (
                    episode - self.last_checkpoint_episode
                    >= self.args.checkpoint_frequency
                    or performance_stats["win_rate"] > self.best_win_rate
                ):

                    self.checkpoint_manager.save_checkpoint(
                        self.verifier,
                        self.agent,
                        episode,
                        performance_stats["win_rate"],
                        energy_quality,
                        policy_memory_stats=self.policy_memory.get_stats(),
                    )
                    self.last_checkpoint_episode = episode

                    if performance_stats["win_rate"] > self.best_win_rate:
                        self.best_win_rate = performance_stats["win_rate"]

                # Adjust experience buffer threshold
                self.experience_buffer.adjust_threshold(episode_number=episode)

                # Comprehensive logging
                self.log_training_progress(
                    episode,
                    episode_stats,
                    performance_stats,
                    train_stats,
                    episode_start_time,
                    training_start_time,
                )

            # Early stopping check
            if len(self.win_rate_history) >= 20:
                recent_win_rate = safe_mean(list(self.win_rate_history)[-10:], 0.0)
                if recent_win_rate >= self.args.target_win_rate:
                    print(
                        f"🎯 Target win rate {self.args.target_win_rate:.1%} achieved!"
                    )
                    print(f"   Current win rate: {recent_win_rate:.1%}")
                    break

        # Training completed
        final_performance = self.evaluate_performance()
        print(f"\n🏁 Enhanced Policy Memory Training Completed!")
        print(f"   - Total episodes: {self.episode + 1}")
        print(f"   - Final win rate: {final_performance['win_rate']:.1%}")
        print(f"   - Best win rate: {self.best_win_rate:.1%}")
        print(
            f"   - Policy memory activations: {self.policy_memory.averaging_performed}"
        )
        print(f"   - Training time: {time.time() - training_start_time:.1f}s")

        # Save final checkpoint
        self.checkpoint_manager.save_checkpoint(
            self.verifier,
            self.agent,
            self.episode,
            final_performance["win_rate"],
            self.energy_quality_history[-1] if self.energy_quality_history else 0.0,
            policy_memory_stats=self.policy_memory.get_stats(),
        )

    def log_training_progress(
        self,
        episode,
        episode_stats,
        performance_stats,
        train_stats,
        episode_start_time,
        training_start_time,
    ):
        """Enhanced logging with policy memory metrics."""
        episode_time = time.time() - episode_start_time
        total_time = time.time() - training_start_time

        # Buffer statistics
        buffer_stats = self.experience_buffer.get_stats()
        golden_stats = buffer_stats["golden_buffer"]

        # Policy memory statistics
        policy_memory_stats = self.policy_memory.get_stats()

        # Agent thinking statistics
        thinking_stats = self.agent.get_thinking_stats()

        # Energy statistics
        energy_stats = self.verifier.get_energy_stats()

        print(f"\n{'='*80}")
        print(f"🥊 Episode {episode:,} | Policy Memory Training")
        print(f"{'='*80}")

        # Performance metrics
        print(f"📊 Performance Metrics:")
        print(
            f"   Win Rate: {performance_stats['win_rate']:.1%} | "
            f"Avg Reward: {performance_stats['avg_reward']:.2f} | "
            f"Episode Reward: {episode_stats['reward']:.2f}"
        )

        if len(self.win_rate_history) > 1:
            win_rate_trend = safe_mean(list(self.win_rate_history)[-5:], 0.0)
            print(f"   Win Rate Trend (5-ep): {win_rate_trend:.1%}")

        # Policy memory status
        print(f"🧠 Policy Memory Status:")
        print(
            f"   Peak Win Rate: {policy_memory_stats['peak_win_rate']:.1%} | "
            f"Episodes Since Peak: {policy_memory_stats['episodes_since_peak']}"
        )
        print(
            f"   Performance Drop: {'🚨 YES' if policy_memory_stats['performance_drop_detected'] else '✅ NO'} | "
            f"Averaging Performed: {policy_memory_stats['averaging_performed']}"
        )

        # Experience buffer metrics
        print(f"🎯 Experience Buffer:")
        print(
            f"   Total: {buffer_stats['total_size']:,} | "
            f"Good: {buffer_stats['good_count']:,} ({buffer_stats['good_ratio']:.1%}) | "
            f"Bad: {buffer_stats['bad_count']:,}"
        )
        print(
            f"   Quality Threshold: {buffer_stats['quality_threshold']:.3f} | "
            f"Avg Quality: {buffer_stats['avg_quality_score']:.3f}"
        )
        print(
            f"   🏆 Golden Buffer: {golden_stats['size']}/{golden_stats['capacity']} "
            f"({golden_stats['utilization']:.1%}) | "
            f"Avg Quality: {golden_stats['avg_quality']:.3f}"
        )

        # Training metrics
        if train_stats:
            print(f"🔧 Training Metrics:")
            print(
                f"   Contrastive Loss: {train_stats.get('contrastive_loss', 0.0):.4f} | "
                f"Energy Reg: {train_stats.get('energy_reg', 0.0):.4f}"
            )
            print(
                f"   Energy Separation: {train_stats.get('energy_separation', 0.0):.4f} | "
                f"Learning Rate: {train_stats.get('learning_rate', 0.0):.2e}"
            )
            print(
                f"   Good Energy: {train_stats.get('good_energy_mean', 0.0):.3f} | "
                f"Bad Energy: {train_stats.get('bad_energy_mean', 0.0):.3f}"
            )

        # Agent thinking metrics
        print(f"🤔 Agent Thinking:")
        print(
            f"   Success Rate: {thinking_stats.get('success_rate', 0.0):.1%} | "
            f"Avg Steps: {thinking_stats.get('avg_thinking_steps', 0.0):.1f} | "
            f"Early Stop Rate: {thinking_stats.get('early_stop_rate', 0.0):.1%}"
        )
        print(
            f"   Energy Explosions: {thinking_stats.get('energy_explosions', 0)} | "
            f"Gradient Explosions: {thinking_stats.get('gradient_explosions', 0)}"
        )

        # Energy monitoring
        print(f"⚡ Energy Monitoring:")
        print(
            f"   Mean Energy: {energy_stats.get('energy_mean', 0.0):.3f} | "
            f"Energy Std: {energy_stats.get('energy_std', 0.0):.3f}"
        )
        print(
            f"   NaN Count: {energy_stats.get('nan_count', 0)} | "
            f"Explosion Count: {energy_stats.get('explosion_count', 0)}"
        )

        # Timing
        print(f"⏱️  Timing:")
        print(
            f"   Episode: {episode_time:.1f}s | "
            f"Total: {total_time/60:.1f}m | "
            f"ETA: {self._estimate_time_remaining(episode, total_time)}"
        )

        print(f"{'='*80}\n")

        # Log to file
        self.logger.info(
            f"Ep {episode}: WinRate={performance_stats['win_rate']:.3f}, "
            f"Reward={episode_stats['reward']:.2f}, "
            f"PeakWR={policy_memory_stats['peak_win_rate']:.3f}, "
            f"MemoryAvg={policy_memory_stats['averaging_performed']}, "
            f"GoldenBuf={golden_stats['size']}"
        )

    def _estimate_time_remaining(self, current_episode, elapsed_time):
        """Estimate time remaining for training."""
        if current_episode == 0:
            return "Unknown"

        avg_time_per_episode = elapsed_time / (current_episode + 1)
        remaining_episodes = self.args.max_episodes - current_episode - 1
        remaining_time = remaining_episodes * avg_time_per_episode

        if remaining_time < 3600:
            return f"{remaining_time/60:.1f}m"
        else:
            return f"{remaining_time/3600:.1f}h"


def main():
    """Main training function with enhanced argument parsing."""
    parser = argparse.ArgumentParser(
        description="Enhanced Energy-Based Training with Policy Memory"
    )

    # Environment arguments (updated to match your existing args)
    parser.add_argument(
        "--total-episodes",
        type=int,
        default=2000,
        help="Maximum number of training episodes",
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=1000, help="Maximum steps per episode"
    )
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=10,
        help="Evaluate performance every N episodes",
    )
    parser.add_argument(
        "--save-freq", type=int, default=50, help="Save checkpoint every N episodes"
    )

    # Model arguments (updated to match your existing args)
    parser.add_argument(
        "--features-dim", type=int, default=256, help="Feature dimension for verifier"
    )
    parser.add_argument(
        "--thinking-steps",
        type=int,
        default=3,
        help="Number of thinking steps for agent",
    )
    parser.add_argument(
        "--thinking-lr",
        type=float,
        default=0.06,
        help="Learning rate for thinking process",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.02,
        help="Noise scale for action initialization",
    )

    # Training arguments (updated to match your existing args)
    parser.add_argument(
        "--lr", type=float, default=3e-5, help="Learning rate for verifier"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=2e-4,
        help="Weight decay for regularization",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Training batch size"
    )
    parser.add_argument(
        "--contrastive-margin",
        type=float,
        default=2.0,
        help="Margin for contrastive loss",
    )

    # Device and rendering (matching your existing args)
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for training"
    )
    parser.add_argument("--render", action="store_true", help="Render the environment")

    # Experience buffer arguments
    parser.add_argument(
        "--buffer-capacity", type=int, default=30000, help="Experience buffer capacity"
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.6,
        help="Quality threshold for experience labeling",
    )
    parser.add_argument(
        "--golden-buffer-capacity",
        type=int,
        default=1000,
        help="Golden experience buffer capacity",
    )

    # Policy memory arguments
    parser.add_argument(
        "--performance-drop-threshold",
        type=float,
        default=0.05,
        help="Performance drop threshold for policy memory",
    )
    parser.add_argument(
        "--averaging-weight",
        type=float,
        default=0.7,
        help="Weight for best checkpoint in averaging",
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
        "--checkpoint-dir",
        type=str,
        default="checkpoints_enhanced",
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--load-checkpoint", type=str, default=None, help="Path to checkpoint to load"
    )

    args = parser.parse_args()

    # Map your argument names to the internal structure
    args.max_episodes = args.total_episodes
    args.learning_rate = args.lr
    args.checkpoint_frequency = args.save_freq

    # Print configuration
    print(f"🛡️ Enhanced Policy Memory Training Configuration:")
    print(f"   Max Episodes: {args.max_episodes:,}")
    print(f"   Learning Rate: {args.learning_rate:.2e}")
    print(f"   Weight Decay: {args.weight_decay:.2e}")
    print(f"   Contrastive Margin: {args.contrastive_margin}")
    print(f"   Performance Drop Threshold: {args.performance_drop_threshold}")
    print(f"   Checkpoint Averaging Weight: {args.averaging_weight}")
    print(f"   Golden Buffer Capacity: {args.golden_buffer_capacity:,}")
    print(f"   Target Win Rate: {args.target_win_rate:.1%}")
    print(f"   Device: {args.device}")
    print(f"   Render: {args.render}")

    # Initialize and run trainer
    trainer = EnhancedPolicyMemoryTrainer(args)

    # Load checkpoint if specified
    if args.load_checkpoint:
        print(f"📂 Loading checkpoint: {args.load_checkpoint}")
        trainer.checkpoint_manager.load_checkpoint(
            Path(args.load_checkpoint), trainer.verifier, trainer.agent
        )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
