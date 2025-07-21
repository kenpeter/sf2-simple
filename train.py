#!/usr/bin/env python3
"""
🛡️ FIXED Energy-Based Training with Policy Memory
FIXES: Quality thresholds, health detection, gradient stability, single fights
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

# Import our FIXED wrapper components
from wrapper import (
    make_fixed_env,
    verify_fixed_energy_flow,
    FixedEnergyBasedStreetFighterVerifier,
    FixedStabilizedEnergyBasedAgent,
    FixedQualityBasedExperienceBuffer,
    PolicyMemoryManager,
    FixedEnergyStabilityManager,
    FixedCheckpointManager,
    safe_mean,
    safe_std,
    safe_divide,
    MAX_FIGHT_STEPS,
)


class FixedPolicyMemoryTrainer:
    """🛡️ FIXED trainer with proper thresholds and stability."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize environment
        print(f"🎮 Initializing FIXED training environment...")
        self.env = make_fixed_env()

        # Initialize verifier and agent
        self.verifier = FixedEnergyBasedStreetFighterVerifier(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            features_dim=args.features_dim,
        ).to(self.device)

        # Verify energy flow
        if not verify_fixed_energy_flow(
            self.verifier, self.env.observation_space, self.env.action_space
        ):
            raise RuntimeError("Energy flow verification failed!")

        self.agent = FixedStabilizedEnergyBasedAgent(
            verifier=self.verifier,
            thinking_steps=args.thinking_steps,
            thinking_lr=args.thinking_lr,
            noise_scale=args.noise_scale,
        )

        # Initialize Policy Memory Manager
        self.policy_memory = PolicyMemoryManager(
            performance_drop_threshold=args.performance_drop_threshold,
            averaging_weight=args.averaging_weight,
        )

        # FIXED: Initialize experience buffer with LOWERED quality threshold
        self.experience_buffer = FixedQualityBasedExperienceBuffer(
            capacity=args.buffer_capacity,
            quality_threshold=args.quality_threshold,  # This should be 0.3 now
            golden_buffer_capacity=args.golden_buffer_capacity,
        )

        # Initialize stability manager
        self.stability_manager = FixedEnergyStabilityManager(
            initial_lr=args.learning_rate,
            thinking_lr=args.thinking_lr,
            policy_memory_manager=self.policy_memory,
        )

        # Initialize checkpoint manager
        self.checkpoint_manager = FixedCheckpointManager(
            checkpoint_dir=args.checkpoint_dir
        )

        # FIXED: Initialize optimizer with more conservative parameters
        self.optimizer = optim.Adam(
            self.verifier.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            eps=1e-8,
            betas=(0.9, 0.999),
        )

        # Training state
        self.episode = 0
        self.total_steps = 0
        self.best_win_rate = 0.0

        # Performance tracking
        self.win_rate_history = deque(maxlen=args.win_rate_window)
        self.energy_quality_history = deque(maxlen=50)
        self.last_checkpoint_episode = 0

        # FIXED: Track actual wins/losses for better debugging
        self.recent_outcomes = deque(maxlen=20)
        self.total_wins = 0
        self.total_fights = 0

        # Enhanced logging
        self.setup_logging()

        print(f"🛡️ FIXED Policy Memory Trainer initialized")
        print(f"   - Device: {self.device}")
        print(f"   - Learning rate: {args.learning_rate:.2e}")
        print(f"   - Quality threshold: {args.quality_threshold} (LOWERED)")
        print(f"   - Max fight steps: {MAX_FIGHT_STEPS}")

    def setup_logging(self):
        """Setup logging system."""
        log_dir = Path("logs_fixed")
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"fixed_training_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def calculate_experience_quality_fixed(
        self, reward, reward_breakdown, episode_stats, steps_taken
    ):
        """FIXED: Much simpler and more lenient quality calculation."""

        # FIXED: Start with reasonable base
        base_quality = 0.4

        # FIXED: Strong signals for clear outcomes
        if "round_won" in reward_breakdown:
            quality = 0.9  # Very high quality for wins
            print(f"🏆 WIN! Quality: {quality:.3f}")
            return quality
        elif "round_lost" in reward_breakdown:
            quality = 0.2  # Low but not terrible for losses
            print(f"💀 LOSS! Quality: {quality:.3f}")
            return quality

        # FIXED: Reward-based quality (more lenient)
        if reward > 0:
            quality = 0.7  # Good quality for positive rewards
        elif reward > -1.0:
            quality = 0.5  # Neutral quality for small negative rewards
        elif reward > -2.0:
            quality = 0.4  # Below average for medium negative
        else:
            quality = 0.3  # Low quality for very negative

        # FIXED: Bonus for damage dealing
        if "damage_dealt" in reward_breakdown and reward_breakdown["damage_dealt"] > 0:
            quality += 0.1

        # FIXED: Small bonus for longer fights (shows engagement)
        if steps_taken > 100:
            quality += 0.05

        # FIXED: Ensure reasonable range
        quality = max(0.1, min(0.95, quality))

        return quality

    def run_episode(self):
        """Run a single episode with FIXED tracking."""
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
        round_lost = False

        # FIXED: Track episode start
        episode_start_time = time.time()

        print(f"🥊 Starting episode {self.episode + 1}")

        while not done and not truncated and episode_steps < MAX_FIGHT_STEPS:
            # Agent prediction
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
            elif "round_lost" in reward_breakdown:
                round_lost = True

            # FIXED: Calculate experience quality with new method
            episode_stats_partial = {
                "won": round_won,
                "lost": round_lost,
                "damage_ratio": safe_divide(
                    damage_dealt_total, damage_taken_total + 1e-6, 1.0
                ),
            }

            quality_score = self.calculate_experience_quality_fixed(
                reward, reward_breakdown, episode_stats_partial, episode_steps
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
                "quality_score": quality_score,
            }

            episode_experiences.append((experience, quality_score))
            obs = next_obs

        # FIXED: Episode completed - determine outcome
        fight_outcome = "unknown"
        if round_won:
            fight_outcome = "won"
            self.total_wins += 1
        elif round_lost:
            fight_outcome = "lost"
        elif episode_steps >= MAX_FIGHT_STEPS:
            fight_outcome = "timeout"

        self.total_fights += 1
        self.recent_outcomes.append(fight_outcome)

        episode_time = time.time() - episode_start_time

        # FIXED: Enhanced episode logging
        print(f"📊 Episode {self.episode + 1} completed:")
        print(f"   - Outcome: {fight_outcome.upper()}")
        print(f"   - Steps: {episode_steps}/{MAX_FIGHT_STEPS}")
        print(f"   - Reward: {episode_reward:.2f}")
        print(f"   - Time: {episode_time:.1f}s")
        print(f"   - Experiences: {len(episode_experiences)}")
        print(f"   - Damage dealt: {damage_dealt_total:.2f}")
        print(f"   - Total wins: {self.total_wins}/{self.total_fights}")

        # Add experiences to buffer
        experiences_added = 0
        good_experiences = 0
        for experience, quality_score in episode_experiences:
            self.experience_buffer.add_experience(
                experience,
                experience["reward"],
                experience.get("reward_breakdown", {}),
                quality_score,
            )
            experiences_added += 1
            if quality_score >= self.experience_buffer.quality_threshold:
                good_experiences += 1

        print(f"   - Added {experiences_added} experiences ({good_experiences} good)")

        # Episode stats
        episode_stats_final = {
            "won": round_won,
            "lost": round_lost,
            "outcome": fight_outcome,
            "reward": episode_reward,
            "steps": episode_steps,
            "time": episode_time,
            "damage_dealt": damage_dealt_total,
            "damage_taken": damage_taken_total,
            "damage_ratio": safe_divide(
                damage_dealt_total, damage_taken_total + 1e-6, 1.0
            ),
            "experiences_added": experiences_added,
            "good_experiences": good_experiences,
        }

        return episode_stats_final

    def calculate_contrastive_loss(
        self, good_batch, bad_batch, margin=1.5
    ):  # FIXED: Lower margin
        """FIXED: More stable contrastive loss calculation."""
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
            return torch.tensor(0.0, device=device), {}

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

        # FIXED: More stable contrastive loss
        good_energy_mean = good_energies.mean()
        bad_energy_mean = bad_energies.mean()

        # We want good energies to be lower (more negative) than bad energies
        energy_diff = bad_energy_mean - good_energy_mean
        contrastive_loss = torch.clamp(margin - energy_diff, min=0.0)

        # FIXED: Reduced regularization
        energy_reg = 0.005 * (good_energies.pow(2).mean() + bad_energies.pow(2).mean())

        total_loss = contrastive_loss + energy_reg

        return total_loss, {
            "contrastive_loss": contrastive_loss.item(),
            "energy_reg": energy_reg.item(),
            "good_energy_mean": good_energy_mean.item(),
            "bad_energy_mean": bad_energy_mean.item(),
            "energy_separation": energy_diff.item(),
        }

    def train_step(self):
        """FIXED: Training step with better batch sampling."""
        # Sample balanced batch
        good_batch, bad_batch, golden_batch = (
            self.experience_buffer.sample_enhanced_balanced_batch(
                self.args.batch_size,
                golden_ratio=0.1,  # FIXED: Lower golden ratio initially
            )
        )

        if good_batch is None or bad_batch is None:
            buffer_stats = self.experience_buffer.get_stats()
            print(
                f"⚠️ Cannot sample batch: Good={buffer_stats['good_count']}, Bad={buffer_stats['bad_count']}"
            )
            return None

        print(
            f"🎯 Training batch: {len(good_batch)} good, {len(bad_batch)} bad, {len(golden_batch) if golden_batch else 0} golden"
        )

        # Calculate loss
        loss, loss_info = self.calculate_contrastive_loss(
            good_batch, bad_batch, margin=self.args.contrastive_margin
        )

        # Get current learning rates
        current_lr, current_thinking_lr = self.stability_manager.get_current_lrs()

        # Update optimizer learning rate if changed
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr

        # Update agent thinking learning rate
        self.agent.current_thinking_lr = current_thinking_lr

        # FIXED: More conservative gradient handling
        self.optimizer.zero_grad()

        try:
            loss.backward()

            # FIXED: Better gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.verifier.parameters(), max_norm=0.5
            )  # Lower max norm

            # Check for gradient explosion
            if grad_norm > 2.0:  # Lower threshold
                print(f"⚠️ Large gradient norm: {grad_norm:.2f}, skipping update")
                return None

            self.optimizer.step()

        except Exception as e:
            print(f"❌ Training step failed: {e}")
            return None

        # Add gradient norm to loss info
        loss_info["grad_norm"] = grad_norm.item()
        loss_info["learning_rate"] = current_lr
        loss_info["thinking_lr"] = current_thinking_lr

        return loss_info

    def evaluate_performance(self):
        """Evaluate current performance with FIXED metrics."""
        eval_episodes = 3  # FIXED: Fewer evaluation episodes for faster feedback

        wins = 0
        total_reward = 0.0
        total_steps = 0
        outcomes = []

        print(f"🔍 Evaluating performance over {eval_episodes} episodes...")

        for eval_ep in range(eval_episodes):
            obs, info = self.env.reset()
            done = False
            truncated = False
            episode_reward = 0.0
            episode_steps = 0
            won_fight = False

            while not done and not truncated and episode_steps < MAX_FIGHT_STEPS:
                action, _ = self.agent.predict(
                    obs, deterministic=True
                )  # Deterministic for evaluation
                obs, reward, done, truncated, info = self.env.step(action)

                episode_reward += reward
                episode_steps += 1

                # Check for win
                reward_breakdown = info.get("reward_breakdown", {})
                if "round_won" in reward_breakdown:
                    won_fight = True
                    wins += 1
                    break

            outcomes.append("win" if won_fight else "loss")
            total_reward += episode_reward
            total_steps += episode_steps

        win_rate = wins / eval_episodes
        avg_reward = total_reward / eval_episodes
        avg_steps = total_steps / eval_episodes

        print(f"📊 Evaluation results: {wins}/{eval_episodes} wins ({win_rate:.1%})")
        print(f"   - Outcomes: {outcomes}")

        return {
            "win_rate": win_rate,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
            "eval_episodes": eval_episodes,
            "outcomes": outcomes,
        }

    def handle_policy_memory_operations(self, performance_stats, train_stats):
        """Handle policy memory operations."""
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

        # Handle performance drop
        elif performance_drop:
            print(f"📉 PERFORMANCE DROP DETECTED - Activating Policy Memory!")

            if self.policy_memory.should_perform_averaging(self.episode):
                print(f"🔄 Performing checkpoint averaging...")
                averaging_success = self.policy_memory.perform_checkpoint_averaging(
                    self.verifier
                )

                if averaging_success:
                    print(f"✅ Checkpoint averaging completed successfully")
                    policy_memory_action_taken = True

                    if self.policy_memory.should_reduce_lr():
                        new_lr = self.policy_memory.get_reduced_lr(current_lr)
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = new_lr
                        self.stability_manager.current_lr = new_lr
                        print(f"📉 Learning rate reduced to {new_lr:.2e}")
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
        """FIXED: Main training loop."""
        print(f"🛡️ Starting FIXED Policy Memory Training")
        print(f"   - Target episodes: {self.args.max_episodes}")
        print(f"   - Target win rate: {self.args.target_win_rate:.1%}")
        print(f"   - Quality threshold: {self.args.quality_threshold}")
        print(f"   - Max fight steps: {MAX_FIGHT_STEPS}")

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
            buffer_stats = self.experience_buffer.get_stats()
            if (
                buffer_stats["good_count"] >= 4 and buffer_stats["bad_count"] >= 4
            ):  # FIXED: Lower requirements
                train_stats = self.train_step()
                if train_stats:
                    recent_losses.append(train_stats.get("contrastive_loss", 0.0))
            else:
                train_stats = {}

            # FIXED: More frequent evaluation for faster feedback
            if episode % self.args.eval_frequency == 0:
                # Performance evaluation
                performance_stats = self.evaluate_performance()
                self.win_rate_history.append(performance_stats["win_rate"])

                # Calculate energy quality metrics
                energy_separation = train_stats.get("energy_separation", 0.0)
                energy_quality = abs(energy_separation) * 10.0
                self.energy_quality_history.append(energy_quality)

                # Policy memory operations
                policy_memory_action = self.handle_policy_memory_operations(
                    performance_stats, train_stats
                )

                # Update stability manager
                early_stop_rate = 0.0  # FIXED: Simplified for now
                stability_emergency = self.stability_manager.update_metrics(
                    performance_stats["win_rate"],
                    energy_quality,
                    energy_separation,
                    early_stop_rate,
                )

                # Handle stability emergency
                if stability_emergency and not policy_memory_action:
                    print(f"🚨 Stability emergency triggered!")
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

                # FIXED: Comprehensive logging
                self.log_training_progress_fixed(
                    episode,
                    episode_stats,
                    performance_stats,
                    train_stats,
                    episode_start_time,
                    training_start_time,
                )

            # Early stopping check
            if len(self.win_rate_history) >= 10:
                recent_win_rate = safe_mean(list(self.win_rate_history)[-5:], 0.0)
                if recent_win_rate >= self.args.target_win_rate:
                    print(
                        f"🎯 Target win rate {self.args.target_win_rate:.1%} achieved!"
                    )
                    break

        # Training completed
        final_performance = self.evaluate_performance()
        total_time = time.time() - training_start_time

        print(f"\n🏁 FIXED Policy Memory Training Completed!")
        print(f"   - Total episodes: {self.episode + 1}")
        print(f"   - Final win rate: {final_performance['win_rate']:.1%}")
        print(f"   - Best win rate: {self.best_win_rate:.1%}")
        print(f"   - Total wins: {self.total_wins}/{self.total_fights}")
        print(
            f"   - Policy memory activations: {self.policy_memory.averaging_performed}"
        )
        print(f"   - Training time: {total_time/60:.1f} minutes")

        # Save final checkpoint
        self.checkpoint_manager.save_checkpoint(
            self.verifier,
            self.agent,
            self.episode,
            final_performance["win_rate"],
            self.energy_quality_history[-1] if self.energy_quality_history else 0.0,
            policy_memory_stats=self.policy_memory.get_stats(),
        )

    def log_training_progress_fixed(
        self,
        episode,
        episode_stats,
        performance_stats,
        train_stats,
        episode_start_time,
        training_start_time,
    ):
        """FIXED: Enhanced logging with better metrics."""
        episode_time = time.time() - episode_start_time
        total_time = time.time() - training_start_time

        # Buffer statistics
        buffer_stats = self.experience_buffer.get_stats()
        golden_stats = buffer_stats["golden_buffer"]

        # Policy memory statistics
        policy_memory_stats = self.policy_memory.get_stats()

        print(f"\n{'='*80}")
        print(f"🥊 Episode {episode:,} | FIXED Policy Memory Training")
        print(f"{'='*80}")

        # FIXED: Performance metrics with better tracking
        print(f"📊 Performance Metrics:")
        print(
            f"   Win Rate: {performance_stats['win_rate']:.1%} | "
            f"Best: {self.best_win_rate:.1%} | "
            f"Total Wins: {self.total_wins}/{self.total_fights} ({self.total_wins/max(1,self.total_fights):.1%})"
        )
        print(
            f"   Episode: {episode_stats['outcome'].upper()} | "
            f"Reward: {episode_stats['reward']:.2f} | "
            f"Steps: {episode_stats['steps']}"
        )

        # Recent outcomes
        if len(self.recent_outcomes) > 0:
            recent_wins = sum(1 for outcome in self.recent_outcomes if outcome == "won")
            recent_win_rate = recent_wins / len(self.recent_outcomes)
            print(
                f"   Recent ({len(self.recent_outcomes)} fights): {recent_wins} wins ({recent_win_rate:.1%})"
            )

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

        # FIXED: Experience buffer metrics with better visibility
        print(f"🎯 Experience Buffer (FIXED):")
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
            f"({golden_stats['utilization']:.1%})"
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
        else:
            print(f"🔧 Training Metrics: Not enough experiences for training yet")

        # Agent thinking metrics
        thinking_stats = self.agent.get_thinking_stats()
        print(f"🤔 Agent Thinking:")
        print(
            f"   Success Rate: {thinking_stats.get('success_rate', 0.0):.1%} | "
            f"Total Predictions: {thinking_stats.get('total_predictions', 0)}"
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
            f"Outcome={episode_stats['outcome']}, "
            f"Reward={episode_stats['reward']:.2f}, "
            f"GoodExp={buffer_stats['good_count']}, "
            f"TotalWins={self.total_wins}/{self.total_fights}"
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
    """Main training function with FIXED argument parsing."""
    parser = argparse.ArgumentParser(
        description="FIXED Energy-Based Training with Policy Memory"
    )

    # Environment arguments
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=1000,
        help="Maximum number of training episodes",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=1200,
        help="Maximum steps per episode (auto-set)",
    )
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=5,
        help="Evaluate performance every N episodes",
    )
    parser.add_argument(
        "--checkpoint-frequency",
        type=int,
        default=25,
        help="Save checkpoint every N episodes",
    )

    # Model arguments
    parser.add_argument(
        "--features-dim", type=int, default=256, help="Feature dimension for verifier"
    )
    parser.add_argument(
        "--thinking-steps",
        type=int,
        default=2,
        help="Number of thinking steps for agent",
    )
    parser.add_argument(
        "--thinking-lr",
        type=float,
        default=0.03,
        help="Learning rate for thinking process",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.01,
        help="Noise scale for action initialization",
    )

    # FIXED: Training arguments with better defaults
    parser.add_argument(
        "--learning-rate", type=float, default=1e-5, help="Learning rate for verifier"
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for regularization",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--contrastive-margin",
        type=float,
        default=1.5,
        help="Margin for contrastive loss",
    )

    # Device and rendering
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for training"
    )
    parser.add_argument("--render", action="store_true", help="Render the environment")

    # FIXED: Experience buffer arguments with lowered thresholds
    parser.add_argument(
        "--buffer-capacity", type=int, default=20000, help="Experience buffer capacity"
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.3,
        help="Quality threshold (LOWERED)",
    )
    parser.add_argument(
        "--golden-buffer-capacity",
        type=int,
        default=500,
        help="Golden experience buffer capacity",
    )

    # Policy memory arguments
    parser.add_argument(
        "--performance-drop-threshold",
        type=float,
        default=0.08,
        help="Performance drop threshold",
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
        default=0.4,
        help="Target win rate for early stopping",
    )
    parser.add_argument(
        "--win-rate-window",
        type=int,
        default=20,
        help="Window size for win rate calculation",
    )

    # Checkpoint arguments
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints_fixed",
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--load-checkpoint", type=str, default=None, help="Path to checkpoint to load"
    )

    args = parser.parse_args()

    # FIXED: Print configuration with emphasis on fixes
    print(f"🛡️ FIXED Policy Memory Training Configuration:")
    print(f"   Max Episodes: {args.max_episodes:,}")
    print(f"   Learning Rate: {args.learning_rate:.2e} (LOWERED)")
    print(f"   Quality Threshold: {args.quality_threshold} (FIXED - LOWERED)")
    print(f"   Target Win Rate: {args.target_win_rate:.1%}")
    print(f"   Max Fight Steps: {MAX_FIGHT_STEPS} (SINGLE FIGHTS)")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Thinking Steps: {args.thinking_steps} (REDUCED)")
    print(f"   Contrastive Margin: {args.contrastive_margin} (LOWERED)")
    print(f"   Device: {args.device}")

    # Validation
    if args.quality_threshold > 0.5:
        print(
            f"⚠️  WARNING: Quality threshold {args.quality_threshold} is high. Consider using 0.3 or lower."
        )

    # Initialize and run trainer
    try:
        trainer = FixedPolicyMemoryTrainer(args)

        # Load checkpoint if specified
        if args.load_checkpoint:
            print(f"📂 Loading checkpoint: {args.load_checkpoint}")
            trainer.checkpoint_manager.load_checkpoint(
                Path(args.load_checkpoint), trainer.verifier, trainer.agent
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
