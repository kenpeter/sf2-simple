#!/usr/bin/env python3
"""
🚀 ENHANCED EBT TRAINING - Energy-Based Thinking + Energy-Based Transformers
Synergistic integration of dual energy systems for superior performance
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

# Import the EBT-enhanced wrapper components
from wrapper import (
    make_ebt_enhanced_env,
    verify_ebt_energy_flow,
    EBTEnhancedStreetFighterVerifier,
    EBTEnhancedEnergyBasedAgent,
    EBTEnhancedExperienceBuffer,
    PolicyMemoryManager,
    EnhancedEnergyStabilityManager,
    EBTEnhancedCheckpointManager,
    safe_mean,
    safe_std,
    safe_divide,
    MAX_FIGHT_STEPS,
    EBT_SEQUENCE_LENGTH,
    EBT_HIDDEN_DIM,
)


class EBTEnhancedTrainer:
    """🚀 Enhanced trainer with Energy-Based Transformers integration."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize environment
        print(f"🎮 Initializing EBT-enhanced environment...")
        self.env = make_ebt_enhanced_env()

        # Initialize EBT-enhanced verifier and agent
        self.verifier = EBTEnhancedStreetFighterVerifier(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            features_dim=args.features_dim,
            use_ebt=args.use_ebt,
        ).to(self.device)

        # Verify EBT energy flow
        if not verify_ebt_energy_flow(
            self.verifier, self.env.observation_space, self.env.action_space
        ):
            raise RuntimeError("EBT energy flow verification failed!")

        self.agent = EBTEnhancedEnergyBasedAgent(
            verifier=self.verifier,
            thinking_steps=args.thinking_steps,
            thinking_lr=args.thinking_lr,
            noise_scale=args.noise_scale,
            use_ebt_thinking=args.use_ebt_thinking,
        )

        # Initialize Policy Memory Manager
        self.policy_memory = PolicyMemoryManager(
            performance_drop_threshold=args.performance_drop_threshold,
            averaging_weight=args.averaging_weight,
        )

        # Enhanced experience buffer with EBT support
        self.experience_buffer = EBTEnhancedExperienceBuffer(
            capacity=args.buffer_capacity,
            quality_threshold=args.quality_threshold,
            golden_buffer_capacity=args.golden_buffer_capacity,
        )

        # Initialize stability manager
        self.stability_manager = EnhancedEnergyStabilityManager(
            initial_lr=args.learning_rate,
            thinking_lr=args.thinking_lr,
            policy_memory_manager=self.policy_memory,
        )

        # Initialize checkpoint manager
        self.checkpoint_manager = EBTEnhancedCheckpointManager(
            checkpoint_dir=args.checkpoint_dir
        )

        # Enhanced optimizer with EBT parameters
        optimizer_params = []

        # Add verifier parameters
        for name, param in self.verifier.named_parameters():
            if "ebt" in name and args.use_ebt:
                # Use different learning rate for EBT components
                optimizer_params.append(
                    {
                        "params": param,
                        "lr": args.learning_rate * args.ebt_lr_multiplier,
                        "weight_decay": args.weight_decay
                        * 0.5,  # Less regularization for EBT
                    }
                )
            else:
                # Standard parameters
                optimizer_params.append(
                    {
                        "params": param,
                        "lr": args.learning_rate,
                        "weight_decay": args.weight_decay,
                    }
                )

        self.optimizer = optim.AdamW(
            optimizer_params,
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
        self.ebt_performance_history = deque(maxlen=50)  # NEW: EBT performance tracking
        self.last_checkpoint_episode = 0

        # Enhanced logging
        self.setup_logging()

        print(f"🚀 EBT-Enhanced Trainer initialized")
        print(f"   - Device: {self.device}")
        print(f"   - Learning rate: {args.learning_rate:.2e}")
        print(f"   - EBT enabled: {args.use_ebt}")
        print(f"   - EBT thinking: {args.use_ebt_thinking}")
        print(f"   - EBT LR multiplier: {args.ebt_lr_multiplier}")
        print(f"   - Quality threshold: {args.quality_threshold}")

    def setup_logging(self):
        """Setup enhanced logging system."""
        log_dir = Path("logs_ebt_enhanced")
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"ebt_enhanced_training_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def calculate_experience_quality(
        self, reward, reward_breakdown, episode_stats, thinking_info=None
    ):
        """Enhanced quality calculation with EBT factors."""
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

        # NEW: EBT-specific quality factors
        ebt_component = 0.0
        if thinking_info and self.args.use_ebt:
            # Reward successful EBT optimization
            if thinking_info.get("optimization_successful", False):
                ebt_component += 0.05

            # Reward EBT success
            if thinking_info.get("ebt_success", False):
                ebt_component += 0.03

            # Penalize EBT failures
            if not thinking_info.get("ebt_success", True):
                ebt_component -= 0.02

            # Reward energy improvement
            energy_improvement = thinking_info.get("energy_improvement", 0.0)
            if energy_improvement > 0.01:
                ebt_component += min(energy_improvement * 5.0, 0.05)

        quality_score = (
            base_quality
            + reward_component
            + win_component
            + health_component
            + damage_component
            + episode_component
            + ebt_component
        )

        # Clamp to reasonable range
        return max(0.0, min(1.0, quality_score))

    def run_episode(self):
        """Enhanced episode running with EBT sequence tracking."""
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

        # EBT-specific tracking
        ebt_successes = 0
        ebt_failures = 0
        total_energy_improvement = 0.0

        while (
            not done and not truncated and episode_steps < self.args.max_episode_steps
        ):
            # Get EBT sequence context from environment
            sequence_context = None
            if self.args.use_ebt_thinking:
                try:
                    sequence_context = self.env.get_ebt_sequence(self.device)
                except Exception as e:
                    print(f"⚠️ Failed to get EBT sequence: {e}")
                    sequence_context = None

            # Enhanced agent prediction with EBT
            action, thinking_info = self.agent.predict(
                obs, deterministic=False, sequence_context=sequence_context
            )

            # Execute action
            next_obs, reward, done, truncated, info = self.env.step(action)

            # Update EBT sequence tracker with energy score
            if hasattr(self.env, "feature_tracker") and thinking_info:
                energy_score = thinking_info.get("final_energy", 0.0)
                # Update the last step with the actual energy score
                if self.env.feature_tracker.ebt_tracker.step_count > 0:
                    self.env.feature_tracker.ebt_tracker.energy_sequence[-1] = (
                        energy_score
                    )

            episode_reward += reward
            episode_steps += 1
            self.total_steps += 1

            # Track episode stats
            reward_breakdown = info.get("reward_breakdown", {})
            damage_dealt_total += reward_breakdown.get("damage_dealt", 0.0)
            damage_taken_total += abs(reward_breakdown.get("damage_taken", 0.0))

            if "round_won" in reward_breakdown:
                round_won = True

            # Track EBT performance
            if thinking_info.get("ebt_success", True):
                ebt_successes += 1
            else:
                ebt_failures += 1

            total_energy_improvement += thinking_info.get("energy_improvement", 0.0)

            # Calculate experience quality with EBT factors
            episode_stats = {
                "won": round_won,
                "damage_ratio": safe_divide(
                    damage_dealt_total, damage_taken_total + 1e-6, 1.0
                ),
            }
            quality_score = self.calculate_experience_quality(
                reward, reward_breakdown, episode_stats, thinking_info
            )

            # Store experience with EBT sequence if available
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

            # Add EBT sequence to experience if available
            ebt_sequence = None
            if sequence_context is not None:
                ebt_sequence = sequence_context.detach().cpu().numpy()

            episode_experiences.append((experience, quality_score, ebt_sequence))
            obs = next_obs

        # Episode completed - process experiences
        episode_stats_final = {
            "won": round_won,
            "damage_ratio": safe_divide(
                damage_dealt_total, damage_taken_total + 1e-6, 1.0
            ),
            "reward": episode_reward,
            "steps": episode_steps,
            "ebt_successes": ebt_successes,
            "ebt_failures": ebt_failures,
            "ebt_success_rate": safe_divide(
                ebt_successes, ebt_successes + ebt_failures, 1.0
            ),
            "avg_energy_improvement": safe_divide(
                total_energy_improvement, episode_steps, 0.0
            ),
        }

        # Add experiences to enhanced buffer
        for experience, quality_score, ebt_sequence in episode_experiences:
            reward_breakdown = experience.get("reward_breakdown", {})
            self.experience_buffer.add_experience(
                experience,
                experience["reward"],
                reward_breakdown,
                quality_score,
                ebt_sequence,
            )

        return episode_stats_final

    def calculate_ebt_enhanced_contrastive_loss(
        self, good_batch, bad_batch, sequence_batch=None, margin=2.0
    ):
        """Enhanced contrastive loss with EBT sequence modeling."""
        device = self.device

        def process_batch_with_ebt(batch):
            if not batch:
                return None, None, None

            obs_batch = []
            action_batch = []
            sequence_batch_processed = []

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

                # Handle EBT sequence
                ebt_sequence = exp.get("ebt_sequence", None)
                if ebt_sequence is not None:
                    sequence_tensor = torch.from_numpy(ebt_sequence).float()
                else:
                    # Create dummy sequence if not available
                    sequence_tensor = torch.zeros(
                        1, EBT_SEQUENCE_LENGTH, EBT_HIDDEN_DIM + 64
                    )

                obs_batch.append(obs_tensor)
                action_batch.append(action_one_hot)
                sequence_batch_processed.append(sequence_tensor)

            return obs_batch, action_batch, sequence_batch_processed

        # Process batches with EBT support
        good_obs, good_actions, good_sequences = process_batch_with_ebt(good_batch)
        bad_obs, bad_actions, bad_sequences = process_batch_with_ebt(bad_batch)

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

        # Stack sequences for EBT processing
        good_sequences_stacked = None
        bad_sequences_stacked = None
        if self.args.use_ebt and good_sequences[0] is not None:
            good_sequences_stacked = torch.stack(
                [seq.squeeze(0) for seq in good_sequences]
            ).to(device)
            bad_sequences_stacked = torch.stack(
                [seq.squeeze(0) for seq in bad_sequences]
            ).to(device)

        # Calculate energies with EBT integration
        good_energies = self.verifier(
            good_obs_stacked, good_actions_stacked, good_sequences_stacked
        )
        bad_energies = self.verifier(
            bad_obs_stacked, bad_actions_stacked, bad_sequences_stacked
        )

        # Enhanced contrastive loss
        good_energy_mean = good_energies.mean()
        bad_energy_mean = bad_energies.mean()

        # We want good energies to be lower (more negative) than bad energies
        energy_diff = bad_energy_mean - good_energy_mean
        contrastive_loss = torch.clamp(margin - energy_diff, min=0.0)

        # Enhanced regularization with EBT considerations
        energy_reg = 0.01 * (good_energies.pow(2).mean() + bad_energies.pow(2).mean())

        # NEW: EBT-specific regularization
        ebt_reg = torch.tensor(0.0, device=device)
        if (
            self.args.use_ebt
            and hasattr(self.verifier, "ebt")
            and self.verifier.use_ebt
        ):
            # Encourage stable EBT behavior
            try:
                # Get EBT internal energies for regularization
                with torch.no_grad():
                    if good_sequences_stacked is not None:
                        good_ebt_result = self.verifier.ebt(good_sequences_stacked)
                        bad_ebt_result = self.verifier.ebt(bad_sequences_stacked)

                        good_ebt_energies = good_ebt_result["sequence_energies"]
                        bad_ebt_energies = bad_ebt_result["sequence_energies"]

                        # Regularize EBT sequence energies
                        ebt_reg = 0.005 * (
                            good_ebt_energies.pow(2).mean()
                            + bad_ebt_energies.pow(2).mean()
                        )
            except Exception as e:
                print(f"⚠️ EBT regularization failed: {e}")

        total_loss = contrastive_loss + energy_reg + ebt_reg

        loss_info = {
            "contrastive_loss": contrastive_loss.item(),
            "energy_reg": energy_reg.item(),
            "ebt_reg": ebt_reg.item(),
            "good_energy_mean": good_energy_mean.item(),
            "bad_energy_mean": bad_energy_mean.item(),
            "energy_separation": energy_diff.item(),
            "used_ebt_sequences": good_sequences_stacked is not None,
        }

        return total_loss, loss_info

    def train_step(self):
        """Enhanced training step with EBT sequence support."""
        # Sample enhanced balanced batch with EBT sequences
        batch_result = self.experience_buffer.sample_enhanced_balanced_batch(
            self.args.batch_size, golden_ratio=0.15, sequence_ratio=0.1
        )

        if batch_result[0] is None or batch_result[1] is None:
            return None  # Not enough experiences yet

        good_batch, bad_batch, golden_batch, sequence_batch = batch_result

        # Calculate enhanced loss with EBT integration
        loss, loss_info = self.calculate_ebt_enhanced_contrastive_loss(
            good_batch, bad_batch, sequence_batch, margin=self.args.contrastive_margin
        )

        # Get current learning rates
        current_lr, current_thinking_lr = self.stability_manager.get_current_lrs()

        # Update optimizer learning rates
        for param_group in self.optimizer.param_groups:
            if "lr" not in param_group:
                param_group["lr"] = current_lr
            # EBT parameters might have different base LR
            base_lr = param_group.get("lr", current_lr)
            if base_lr != current_lr:
                # This is an EBT parameter group, scale accordingly
                param_group["lr"] = current_lr * self.args.ebt_lr_multiplier

        # Update agent thinking learning rate
        self.agent.current_thinking_lr = current_thinking_lr

        # Backward pass with enhanced gradient clipping
        self.optimizer.zero_grad()
        loss.backward()

        # Enhanced gradient clipping with EBT considerations
        total_grad_norm = 0.0
        ebt_grad_norm = 0.0

        for name, param in self.verifier.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_grad_norm += param_norm.item() ** 2

                # Track EBT gradient norms separately
                if "ebt" in name and self.args.use_ebt:
                    ebt_grad_norm += param_norm.item() ** 2

        total_grad_norm = total_grad_norm ** (1.0 / 2)
        ebt_grad_norm = ebt_grad_norm ** (1.0 / 2)

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.verifier.parameters(), max_norm=1.0)

        # Check for gradient explosion
        if total_grad_norm > 10.0:
            print(f"⚠️ Large total gradient norm detected: {total_grad_norm:.2f}")
            if ebt_grad_norm > 5.0:
                print(f"⚠️ Large EBT gradient norm detected: {ebt_grad_norm:.2f}")
            return None

        self.optimizer.step()

        # Enhanced loss info with EBT metrics
        loss_info.update(
            {
                "total_grad_norm": total_grad_norm,
                "ebt_grad_norm": ebt_grad_norm,
                "learning_rate": current_lr,
                "thinking_lr": current_thinking_lr,
                "ebt_lr": (
                    current_lr * self.args.ebt_lr_multiplier
                    if self.args.use_ebt
                    else 0.0
                ),
                "sequence_batch_size": len(sequence_batch) if sequence_batch else 0,
            }
        )

        return loss_info

    def evaluate_performance(self):
        """Enhanced evaluation with EBT metrics."""
        eval_episodes = min(5, max(1, self.episode // 100))

        wins = 0
        total_reward = 0.0
        total_steps = 0

        # EBT-specific metrics
        total_ebt_successes = 0
        total_ebt_attempts = 0
        total_energy_improvements = 0.0

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
                # Get EBT sequence context
                sequence_context = None
                if self.args.use_ebt_thinking:
                    try:
                        sequence_context = self.env.get_ebt_sequence(self.device)
                    except:
                        sequence_context = None

                # Enhanced prediction with EBT
                action, thinking_info = self.agent.predict(
                    obs, deterministic=True, sequence_context=sequence_context
                )

                obs, reward, done, truncated, info = self.env.step(action)

                episode_reward += reward
                episode_steps += 1

                # Track EBT performance
                if thinking_info:
                    total_ebt_attempts += 1
                    if thinking_info.get("ebt_success", True):
                        total_ebt_successes += 1
                    total_energy_improvements += thinking_info.get(
                        "energy_improvement", 0.0
                    )

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

        # EBT performance metrics
        ebt_success_rate = safe_divide(total_ebt_successes, total_ebt_attempts, 1.0)
        avg_energy_improvement = safe_divide(
            total_energy_improvements, total_ebt_attempts, 0.0
        )

        return {
            "win_rate": win_rate,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
            "eval_episodes": eval_episodes,
            "ebt_success_rate": ebt_success_rate,
            "avg_energy_improvement": avg_energy_improvement,
            "ebt_attempts": total_ebt_attempts,
        }

    def handle_policy_memory_operations(self, performance_stats, train_stats):
        """Enhanced policy memory operations with EBT awareness."""
        current_win_rate = performance_stats["win_rate"]
        current_lr = None

        # Find the base learning rate (not EBT-specific)
        for param_group in self.optimizer.param_groups:
            if (
                param_group.get("lr", 0) <= self.args.learning_rate * 1.1
            ):  # Base LR group
                current_lr = param_group["lr"]
                break

        if current_lr is None:
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
            # Save peak checkpoint with EBT stats
            ebt_stats = {
                "ebt_success_rate": performance_stats.get("ebt_success_rate", 1.0),
                "avg_energy_improvement": performance_stats.get(
                    "avg_energy_improvement", 0.0
                ),
                "ebt_enabled": self.args.use_ebt,
                "ebt_thinking_enabled": self.args.use_ebt_thinking,
            }

            self.checkpoint_manager.save_checkpoint(
                self.verifier,
                self.agent,
                self.episode,
                current_win_rate,
                train_stats.get("energy_separation", 0.0),
                is_peak=True,
                policy_memory_stats=self.policy_memory.get_stats(),
                ebt_stats=ebt_stats,
            )
            policy_memory_action_taken = True

        # Handle performance drop
        elif performance_drop:
            print(f"📉 PERFORMANCE DROP DETECTED - Activating Policy Memory!")

            # Attempt checkpoint averaging
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

                        # Update all parameter groups proportionally
                        for param_group in self.optimizer.param_groups:
                            old_lr = param_group["lr"]
                            if "ebt" in str(param_group.get("params", [""])[0]):
                                # EBT parameter group
                                param_group["lr"] = new_lr * self.args.ebt_lr_multiplier
                            else:
                                # Base parameter group
                                param_group["lr"] = new_lr

                        # Update stability manager
                        self.stability_manager.current_lr = new_lr
                        print(
                            f"📉 Learning rates reduced - Base: {new_lr:.2e}, EBT: {new_lr * self.args.ebt_lr_multiplier:.2e}"
                        )
                else:
                    print(f"❌ Checkpoint averaging failed")

            # Save emergency checkpoint with EBT stats
            ebt_stats = {
                "ebt_success_rate": performance_stats.get("ebt_success_rate", 1.0),
                "avg_energy_improvement": performance_stats.get(
                    "avg_energy_improvement", 0.0
                ),
                "ebt_enabled": self.args.use_ebt,
                "ebt_thinking_enabled": self.args.use_ebt_thinking,
                "emergency_trigger": "performance_drop",
            }

            self.checkpoint_manager.save_checkpoint(
                self.verifier,
                self.agent,
                self.episode,
                current_win_rate,
                train_stats.get("energy_separation", 0.0),
                is_emergency=True,
                policy_memory_stats=self.policy_memory.get_stats(),
                ebt_stats=ebt_stats,
            )

        return policy_memory_action_taken

    def train(self):
        """Enhanced main training loop with EBT integration."""
        print(f"🚀 Starting EBT-Enhanced Training")
        print(f"   - Target episodes: {self.args.max_episodes}")
        print(
            f"   - Energy-Based Transformers: {'ENABLED' if self.args.use_ebt else 'DISABLED'}"
        )
        print(
            f"   - EBT Thinking: {'ENABLED' if self.args.use_ebt_thinking else 'DISABLED'}"
        )
        print(f"   - Quality threshold: {self.args.quality_threshold}")

        # Training metrics
        episode_rewards = deque(maxlen=100)
        recent_losses = deque(maxlen=50)
        training_start_time = time.time()

        for episode in range(self.args.max_episodes):
            self.episode = episode
            episode_start_time = time.time()

            # Run episode with EBT integration
            episode_stats = self.run_episode()
            episode_rewards.append(episode_stats["reward"])

            # Track EBT performance
            ebt_success_rate = episode_stats.get("ebt_success_rate", 1.0)
            self.ebt_performance_history.append(ebt_success_rate)

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
                # Enhanced performance evaluation
                performance_stats = self.evaluate_performance()
                self.win_rate_history.append(performance_stats["win_rate"])

                # Calculate energy quality metrics
                if train_stats:
                    energy_separation = train_stats.get("energy_separation", 0.0)
                    energy_quality = abs(energy_separation) * 10.0
                else:
                    energy_separation = 0.0
                    energy_quality = 0.0
                self.energy_quality_history.append(energy_quality)

                # Policy memory operations with EBT awareness
                policy_memory_action = self.handle_policy_memory_operations(
                    performance_stats, train_stats or {}
                )

                # Enhanced stability manager with EBT metrics
                early_stop_rate = safe_divide(
                    self.agent.thinking_stats.get("early_stops", 0),
                    self.agent.thinking_stats.get("total_predictions", 1),
                    0.0,
                )

                avg_ebt_success = safe_mean(list(self.ebt_performance_history), 1.0)

                stability_emergency = self.stability_manager.update_metrics(
                    performance_stats["win_rate"],
                    energy_quality,
                    energy_separation,
                    early_stop_rate,
                    avg_ebt_success,  # NEW: EBT success rate
                )

                # Handle stability emergency
                if stability_emergency and not policy_memory_action:
                    print(f"🚨 Stability emergency triggered!")
                    # Update learning rates from stability manager
                    new_lr, new_thinking_lr = self.stability_manager.get_current_lrs()

                    for param_group in self.optimizer.param_groups:
                        if "ebt" in str(param_group.get("params", [""])[0]):
                            param_group["lr"] = new_lr * self.args.ebt_lr_multiplier
                        else:
                            param_group["lr"] = new_lr

                    self.agent.current_thinking_lr = new_thinking_lr

                # Regular checkpoint saving with EBT stats
                if (
                    episode - self.last_checkpoint_episode
                    >= self.args.checkpoint_frequency
                    or performance_stats["win_rate"] > self.best_win_rate
                ):
                    ebt_stats = {
                        "ebt_success_rate": performance_stats.get(
                            "ebt_success_rate", 1.0
                        ),
                        "avg_energy_improvement": performance_stats.get(
                            "avg_energy_improvement", 0.0
                        ),
                        "ebt_enabled": self.args.use_ebt,
                        "ebt_thinking_enabled": self.args.use_ebt_thinking,
                        "episode": episode,
                    }

                    self.checkpoint_manager.save_checkpoint(
                        self.verifier,
                        self.agent,
                        episode,
                        performance_stats["win_rate"],
                        energy_quality,
                        policy_memory_stats=self.policy_memory.get_stats(),
                        ebt_stats=ebt_stats,
                    )
                    self.last_checkpoint_episode = episode

                    if performance_stats["win_rate"] > self.best_win_rate:
                        self.best_win_rate = performance_stats["win_rate"]

                # Adjust experience buffer threshold
                self.experience_buffer.adjust_threshold(episode_number=episode)

                # Enhanced status reporting with EBT metrics
                buffer_stats = self.experience_buffer.get_stats()
                thinking_stats = self.agent.get_thinking_stats()

                print(f"\n🚀 EBT-ENHANCED STATUS (Episode {episode}):")
                print(f"   🎯 Performance:")
                print(f"      - Win rate: {performance_stats['win_rate']:.1%}")
                print(f"      - Avg reward: {performance_stats['avg_reward']:.2f}")
                print(f"      - Energy separation: {energy_separation:.4f}")

                print(f"   🤖 Experience Buffer:")
                print(f"      - Good: {buffer_stats['good_count']:,}")
                print(f"      - Bad: {buffer_stats['bad_count']:,}")
                print(f"      - Sequences: {buffer_stats['sequence_count']:,}")
                print(f"      - Golden: {buffer_stats['golden_count']:,}")
                print(
                    f"      - Quality threshold: {self.experience_buffer.current_threshold:.3f}"
                )

                print(f"   🧠 EBT Performance:")
                print(
                    f"      - EBT success rate: {performance_stats.get('ebt_success_rate', 1.0):.1%}"
                )
                print(
                    f"      - Avg energy improvement: {performance_stats.get('avg_energy_improvement', 0.0):.4f}"
                )
                print(
                    f"      - EBT attempts: {performance_stats.get('ebt_attempts', 0):,}"
                )

                print(f"   🔧 Training Metrics:")
                print(
                    f"      - Thinking success: {thinking_stats.get('success_rate', 1.0):.1%}"
                )
                print(f"      - Early stop rate: {early_stop_rate:.1%}")
                print(f"      - Learning rate: {current_lr:.2e}")
                if self.args.use_ebt:
                    print(
                        f"      - EBT learning rate: {current_lr * self.args.ebt_lr_multiplier:.2e}"
                    )

                if train_stats:
                    print(f"   📊 Loss Information:")
                    print(
                        f"      - Contrastive loss: {train_stats.get('contrastive_loss', 0.0):.4f}"
                    )
                    print(
                        f"      - Energy reg: {train_stats.get('energy_reg', 0.0):.4f}"
                    )
                    print(f"      - EBT reg: {train_stats.get('ebt_reg', 0.0):.4f}")
                    print(
                        f"      - Good energy: {train_stats.get('good_energy_mean', 0.0):.4f}"
                    )
                    print(
                        f"      - Bad energy: {train_stats.get('bad_energy_mean', 0.0):.4f}"
                    )
                    print(
                        f"      - Used EBT sequences: {train_stats.get('used_ebt_sequences', False)}"
                    )

                # Log enhanced metrics
                self.logger.info(
                    f"Episode {episode}: Win={performance_stats['win_rate']:.1%}, "
                    f"Reward={performance_stats['avg_reward']:.2f}, "
                    f"Energy_Sep={energy_separation:.4f}, "
                    f"EBT_Success={performance_stats.get('ebt_success_rate', 1.0):.1%}, "
                    f"Energy_Imp={performance_stats.get('avg_energy_improvement', 0.0):.4f}"
                )

        # Training completed
        training_time = time.time() - training_start_time

        print(f"\n🎉 EBT-Enhanced Training Completed!")
        print(f"   - Total episodes: {self.args.max_episodes}")
        print(f"   - Training time: {training_time/3600:.1f} hours")
        print(f"   - Best win rate: {self.best_win_rate:.1%}")
        print(
            f"   - Final EBT success rate: {safe_mean(list(self.ebt_performance_history), 1.0):.1%}"
        )

        # Save final checkpoint with comprehensive EBT stats
        final_performance = self.evaluate_performance()
        final_ebt_stats = {
            "ebt_success_rate": final_performance.get("ebt_success_rate", 1.0),
            "avg_energy_improvement": final_performance.get(
                "avg_energy_improvement", 0.0
            ),
            "ebt_enabled": self.args.use_ebt,
            "ebt_thinking_enabled": self.args.use_ebt_thinking,
            "final_episode": self.args.max_episodes,
            "training_time_hours": training_time / 3600,
            "best_win_rate": self.best_win_rate,
        }

        self.checkpoint_manager.save_checkpoint(
            self.verifier,
            self.agent,
            self.args.max_episodes,
            final_performance["win_rate"],
            energy_quality,
            is_final=True,
            policy_memory_stats=self.policy_memory.get_stats(),
            ebt_stats=final_ebt_stats,
        )

        self.logger.info(
            f"Training completed. Final win rate: {final_performance['win_rate']:.1%}"
        )


def parse_arguments():
    """Enhanced argument parsing with EBT-specific parameters."""
    parser = argparse.ArgumentParser(
        description="🚀 EBT-Enhanced Energy-Based Training"
    )

    # Environment and basic training parameters
    parser.add_argument(
        "--max_episodes", type=int, default=10000, help="Maximum training episodes"
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=MAX_FIGHT_STEPS,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--eval_frequency", type=int, default=25, help="Evaluation frequency"
    )
    parser.add_argument(
        "--checkpoint_frequency",
        type=int,
        default=100,
        help="Checkpoint saving frequency",
    )

    # Model architecture
    parser.add_argument(
        "--features_dim", type=int, default=256, help="Feature dimension"
    )
    parser.add_argument(
        "--thinking_steps", type=int, default=16, help="Energy-based thinking steps"
    )

    # NEW: EBT-specific parameters
    parser.add_argument(
        "--use_ebt", action="store_true", help="Enable Energy-Based Transformers"
    )
    parser.add_argument(
        "--use_ebt_thinking", action="store_true", help="Enable EBT-enhanced thinking"
    )
    parser.add_argument(
        "--ebt_lr_multiplier",
        type=float,
        default=0.5,
        help="EBT learning rate multiplier",
    )

    # Learning parameters
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument(
        "--thinking_lr", type=float, default=1e-3, help="Thinking learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--contrastive_margin", type=float, default=2.0, help="Contrastive loss margin"
    )

    # Experience buffer parameters
    parser.add_argument(
        "--buffer_capacity", type=int, default=50000, help="Experience buffer capacity"
    )
    parser.add_argument(
        "--golden_buffer_capacity",
        type=int,
        default=2000,
        help="Golden buffer capacity",
    )
    parser.add_argument(
        "--quality_threshold",
        type=float,
        default=0.6,
        help="Experience quality threshold",
    )

    # Energy-based parameters
    parser.add_argument(
        "--noise_scale", type=float, default=0.1, help="Energy noise scale"
    )

    # Policy memory parameters
    parser.add_argument(
        "--performance_drop_threshold",
        type=float,
        default=0.15,
        help="Performance drop threshold",
    )
    parser.add_argument(
        "--averaging_weight",
        type=float,
        default=0.7,
        help="Checkpoint averaging weight",
    )
    parser.add_argument(
        "--win_rate_window", type=int, default=50, help="Win rate averaging window"
    )

    # Paths
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints_ebt_enhanced",
        help="Checkpoint directory",
    )

    return parser.parse_args()


def main():
    """Enhanced main function with EBT support."""
    # Parse arguments
    args = parse_arguments()

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")

    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Enhanced configuration display
    print(f"\n🚀 EBT-ENHANCED TRAINING CONFIGURATION:")
    print(f"   🎮 Environment: Street Fighter (Retro)")
    print(f"   🏆 Target Episodes: {args.max_episodes:,}")
    print(f"   🧠 Features Dimension: {args.features_dim}")
    print(f"   🔄 Thinking Steps: {args.thinking_steps}")
    print(
        f"   ⚡ Energy-Based Transformers: {'ENABLED' if args.use_ebt else 'DISABLED'}"
    )
    print(
        f"   🎯 EBT-Enhanced Thinking: {'ENABLED' if args.use_ebt_thinking else 'DISABLED'}"
    )
    print(f"   📚 Learning Rate: {args.learning_rate:.2e}")
    if args.use_ebt:
        print(
            f"   📚 EBT Learning Rate: {args.learning_rate * args.ebt_lr_multiplier:.2e}"
        )
    print(f"   🎲 Batch Size: {args.batch_size}")
    print(f"   📊 Quality Threshold: {args.quality_threshold}")
    print(f"   💾 Buffer Capacity: {args.buffer_capacity:,}")
    print(f"   🏅 Golden Buffer: {args.golden_buffer_capacity:,}")

    try:
        # Initialize and run enhanced trainer
        trainer = EBTEnhancedTrainer(args)
        trainer.train()

        print(f"\n✅ EBT-Enhanced training completed successfully!")

    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print(f"🔚 EBT-Enhanced training session ended")


if __name__ == "__main__":
    main()
