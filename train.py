def run_enhanced_episode(self):
    """Enhanced episode running with detailed win/lose tracking."""
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
    round_draw = False
    termination_reason = "ongoing"
    victory_type = "none"
    defeat_type = "none"

    # Enhanced tracking for transformer analysis
    transformer_energy_total = 0.0

    while not done and not truncated and episode_steps < self.args.max_episode_steps:
        # Enhanced agent prediction with transformer awareness
        action, thinking_info = self.agent.predict(obs, deterministic=False)

        # Analyze attention patterns
        self.analyze_attention_patterns(thinking_info)

        # Execute action (FAST - no changes to game speed)
        next_obs, reward, done, truncated, info = self.env.step(action)

        episode_reward += reward
        episode_steps += 1
        self.total_steps += 1

        # Track round result from info
        if info.get("round_ended", False):
            termination_reason = info.get("termination_reason", "unknown")
            round_result = info.get("round_result", "ONGOING")
            victory_type = info.get("victory_type", "none")
            defeat_type = info.get("defeat_type", "none")

            if round_result == "WIN":
                round_won = True
                round_lost = False
                round_draw = False
            elif round_result == "LOSE":
                round_won = False
                round_lost = True
                round_draw = False
            elif round_result == "DRAW":
                round_won = False
                round_lost = False
                round_draw = True

        # Track episode stats
        reward_breakdown = info.get("reward_breakdown", {})
        damage_dealt_total += reward_breakdown.get("damage_dealt", 0.0)
        damage_taken_total += abs(reward_breakdown.get("damage_taken", 0.0))

        # Legacy round_won detection for compatibility
        if "round_won" in reward_breakdown:
            round_won = True

        # Enhanced tracking
        if "avg_transformer_energy" in thinking_info:
            transformer_energy_total += abs(thinking_info["avg_transformer_energy"])

        # Calculate enhanced experience quality
        episode_stats = {
            "won": round_won,
            "damage_ratio": safe_divide(
                damage_dealt_total, damage_taken_total + 1e-6, 1.0
            ),
        }
        quality_score = self.calculate_enhanced_experience_quality(
            reward, reward_breakdown, episode_stats
        )

        # Store enhanced experience
        experience = {
            "obs": obs,
            "action": action,
            "reward": reward,
            "next_obs": next_obs,
            "done": done,
            "thinking_info": thinking_info,
            "episode": self.episode,
            "step": episode_steps,
            "transformer_energy": thinking_info.get("avg_transformer_energy", 0.0),
        }

        episode_experiences.append((experience, quality_score))
        obs = next_obs

    # Update win/lose statistics#!/usr/bin/env python3


"""
🛡️ ENHANCED TRAINING - Energy-Based Transformers + Current Energy Thinking
Keeps your original training logic and adds EBT architecture for enhanced learning
Maintains fast game speed and single round fight logic
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

# Import the ENHANCED wrapper components
from wrapper import (
    make_enhanced_env,
    verify_enhanced_energy_flow,
    EnhancedEnergyBasedVerifier,
    EnhancedEnergyBasedAgent,
    EnhancedQualityBasedExperienceBuffer,
    PolicyMemoryManager,
    EnhancedEnergyStabilityManager,
    EnhancedCheckpointManager,
    safe_mean,
    safe_std,
    safe_divide,
    MAX_FIGHT_STEPS,
)


class EnhancedEBTTrainer:
    """🛡️ Enhanced trainer with Energy-Based Transformers."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize environment
        print(f"🎮 Initializing enhanced environment...")
        self.env = make_enhanced_env()

        # Initialize enhanced verifier with EBT
        self.verifier = EnhancedEnergyBasedVerifier(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            features_dim=args.features_dim,
        ).to(self.device)

        # Verify enhanced energy flow
        if not verify_enhanced_energy_flow(
            self.verifier, self.env.observation_space, self.env.action_space
        ):
            raise RuntimeError("Enhanced energy flow verification failed!")

        # Initialize enhanced agent with transformer-aware thinking
        self.agent = EnhancedEnergyBasedAgent(
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

        # Enhanced experience buffer with FIXED quality threshold
        self.experience_buffer = EnhancedQualityBasedExperienceBuffer(
            capacity=args.buffer_capacity,
            quality_threshold=args.quality_threshold,  # FIXED (0.3 instead of 0.6)
            golden_buffer_capacity=args.golden_buffer_capacity,
        )

        # Initialize stability manager
        self.stability_manager = EnhancedEnergyStabilityManager(
            initial_lr=args.learning_rate,
            thinking_lr=args.thinking_lr,
            policy_memory_manager=self.policy_memory,
        )

        # Initialize checkpoint manager
        self.checkpoint_manager = EnhancedCheckpointManager(
            checkpoint_dir=args.checkpoint_dir
        )

        # Enhanced optimizer with transformer parameters
        self.optimizer = optim.AdamW(  # Using AdamW for better transformer training
            self.verifier.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            eps=1e-8,
            betas=(0.9, 0.999),
        )

        # Learning rate scheduler for transformer training
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=50, T_mult=2, eta_min=args.learning_rate * 0.1
        )

        # Training state
        self.episode = 0
        self.total_steps = 0
        self.best_win_rate = 0.0

        # Enhanced performance tracking with detailed win/lose stats
        self.win_rate_history = deque(maxlen=args.win_rate_window)
        self.energy_quality_history = deque(maxlen=50)
        self.transformer_energy_history = deque(maxlen=50)
        self.attention_analysis = deque(maxlen=20)
        self.last_checkpoint_episode = 0

        # NEW: Detailed win/lose tracking
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.recent_results = deque(maxlen=20)  # Track last 20 results
        self.win_types = {
            "knockout": 0,
            "technical": 0,
            "decision": 0,
            "health_advantage": 0,
        }
        self.lose_types = {
            "knockout": 0,
            "technical": 0,
            "decision": 0,
            "health_disadvantage": 0,
        }
        self.termination_reasons = deque(maxlen=50)

        # Enhanced logging
        self.setup_logging()

        print(f"🛡️ Enhanced EBT Trainer initialized")
        print(f"   - Device: {self.device}")
        print(f"   - Learning rate: {args.learning_rate:.2e}")
        print(f"   - Quality threshold: {args.quality_threshold} (FIXED)")
        print(f"   - Transformer layers: Energy-Based attention")
        print(f"   - Fast game UI: MAINTAINED")
        print(f"   - Win/Lose tracking: ENABLED")

    def setup_logging(self):
        """Setup enhanced logging system."""
        log_dir = Path("logs_enhanced_ebt")
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"enhanced_ebt_training_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def calculate_enhanced_experience_quality(
        self, reward, reward_breakdown, episode_stats
    ):
        """Enhanced quality calculation with transformer awareness."""
        base_quality = 0.5  # Neutral starting point

        # Reward component (capped to prevent exploitation)
        reward_component = min(max(reward, -1.0), 2.0) * 0.25  # Reduced weight

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
        damage_component = min(reward_breakdown.get("damage_dealt", 0.0), 0.15)

        # Episode performance component
        if episode_stats:
            episode_component = 0.1 if episode_stats.get("won", False) else -0.1
        else:
            episode_component = 0.0

        # NEW: Transformer energy component (encourages meaningful attention patterns)
        transformer_component = 0.0
        if hasattr(self, "last_transformer_energy"):
            # Reward moderate transformer energy usage (not too high, not too low)
            energy_magnitude = abs(self.last_transformer_energy)
            if 0.1 <= energy_magnitude <= 2.0:  # Sweet spot
                transformer_component = 0.05
            elif energy_magnitude > 3.0:  # Too high
                transformer_component = -0.05

        quality_score = (
            base_quality
            + reward_component
            + win_component
            + health_component
            + damage_component
            + episode_component
            + transformer_component
        )

        # Clamp to reasonable range
        return max(0.0, min(1.0, quality_score))

    def analyze_attention_patterns(self, thinking_info):
        """Analyze transformer attention patterns for insights."""
        if "transformer_energies" in thinking_info:
            transformer_energies = thinking_info["transformer_energies"]
            if transformer_energies:
                self.last_transformer_energy = transformer_energies[-1]
                self.transformer_energy_history.append(
                    abs(self.last_transformer_energy)
                )

                # Store attention analysis
                attention_analysis = {
                    "avg_transformer_energy": sum(transformer_energies)
                    / len(transformer_energies),
                    "energy_variance": (
                        np.var(transformer_energies)
                        if len(transformer_energies) > 1
                        else 0.0
                    ),
                    "energy_trend": (
                        transformer_energies[-1] - transformer_energies[0]
                        if len(transformer_energies) > 1
                        else 0.0
                    ),
                }
                self.attention_analysis.append(attention_analysis)

    def run_enhanced_episode(self):
        """Enhanced episode running with round termination tracking."""
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
        termination_reason = "ongoing"

        # Enhanced tracking for transformer analysis
        transformer_energy_total = 0.0

        while (
            not done and not truncated and episode_steps < self.args.max_episode_steps
        ):
            # Enhanced agent prediction with transformer awareness
            action, thinking_info = self.agent.predict(obs, deterministic=False)

            # Analyze attention patterns
            self.analyze_attention_patterns(thinking_info)

            # Execute action (FAST - no changes to game speed)
            next_obs, reward, done, truncated, info = self.env.step(action)

            episode_reward += reward
            episode_steps += 1
            self.total_steps += 1

            # Track termination reason
            if info.get("round_ended", False):
                termination_reason = info.get("termination_reason", "unknown")
                if termination_reason in ["player_ko", "opponent_ko", "technical_ko"]:
                    round_won = termination_reason == "opponent_ko"

            # Track episode stats
            reward_breakdown = info.get("reward_breakdown", {})
            damage_dealt_total += reward_breakdown.get("damage_dealt", 0.0)
            damage_taken_total += abs(reward_breakdown.get("damage_taken", 0.0))

            if "round_won" in reward_breakdown:
                round_won = True

            # Enhanced tracking
            if "avg_transformer_energy" in thinking_info:
                transformer_energy_total += abs(thinking_info["avg_transformer_energy"])

            # Calculate enhanced experience quality
            episode_stats = {
                "won": round_won,
                "damage_ratio": safe_divide(
                    damage_dealt_total, damage_taken_total + 1e-6, 1.0
                ),
            }
            quality_score = self.calculate_enhanced_experience_quality(
                reward, reward_breakdown, episode_stats
            )

            # Store enhanced experience
            experience = {
                "obs": obs,
                "action": action,
                "reward": reward,
                "next_obs": next_obs,
                "done": done,
                "thinking_info": thinking_info,
                "episode": self.episode,
                "step": episode_steps,
                "transformer_energy": thinking_info.get("avg_transformer_energy", 0.0),
            }

            episode_experiences.append((experience, quality_score))
            obs = next_obs

        # Update win/lose statistics
        if round_won:
            self.wins += 1
            self.recent_results.append("WIN")
            if victory_type in self.win_types:
                self.win_types[victory_type] += 1
        elif round_lost:
            self.losses += 1
            self.recent_results.append("LOSE")
            if defeat_type in self.lose_types:
                self.lose_types[defeat_type] += 1
        elif round_draw:
            self.draws += 1
            self.recent_results.append("DRAW")
        else:
            self.recent_results.append("UNKNOWN")

        # Track termination reason
        self.termination_reasons.append(termination_reason)

        # Episode completed - process experiences
        episode_stats_final = {
            "won": round_won,
            "lost": round_lost,
            "draw": round_draw,
            "damage_ratio": safe_divide(
                damage_dealt_total, damage_taken_total + 1e-6, 1.0
            ),
            "reward": episode_reward,
            "steps": episode_steps,
            "avg_transformer_energy": safe_divide(
                transformer_energy_total, episode_steps, 0.0
            ),
            "termination_reason": termination_reason,
            "victory_type": victory_type,
            "defeat_type": defeat_type,
        }

        # Add experiences to buffer
        for experience, quality_score in episode_experiences:
            reward_breakdown = experience.get("reward_breakdown", {})
            self.experience_buffer.add_experience(
                experience, experience["reward"], reward_breakdown, quality_score
            )

        return episode_stats_final

    def get_win_lose_stats(self):
        """Get detailed win/lose statistics."""
        total_games = self.wins + self.losses + self.draws

        if total_games == 0:
            return {
                "total_games": 0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "win_rate": 0.0,
                "recent_win_rate": 0.0,
                "win_types": self.win_types.copy(),
                "lose_types": self.lose_types.copy(),
                "recent_results": list(self.recent_results),
                "recent_terminations": {},
            }

        # Calculate recent win rate (last 10 games)
        recent_wins = sum(
            1 for result in list(self.recent_results)[-10:] if result == "WIN"
        )
        recent_games = min(len(self.recent_results), 10)
        recent_win_rate = recent_wins / recent_games if recent_games > 0 else 0.0

        # Recent termination reasons
        recent_terminations = {}
        for reason in list(self.termination_reasons)[-10:]:
            recent_terminations[reason] = recent_terminations.get(reason, 0) + 1

        return {
            "total_games": total_games,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": self.wins / total_games,
            "recent_win_rate": recent_win_rate,
            "win_types": self.win_types.copy(),
            "lose_types": self.lose_types.copy(),
            "recent_results": list(self.recent_results),
            "recent_terminations": recent_terminations,
        }

    def calculate_enhanced_contrastive_loss(self, good_batch, bad_batch, margin=2.0):
        """Enhanced contrastive loss with transformer awareness."""
        device = self.device

        def process_enhanced_batch(batch):
            if not batch:
                return None, None, None

            obs_batch = []
            action_batch = []
            transformer_energies = []

            for exp in batch:
                obs = exp["obs"]
                action = exp["action"]
                transformer_energy = exp.get("transformer_energy", 0.0)

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
                transformer_energies.append(transformer_energy)

            return obs_batch, action_batch, transformer_energies

        # Process batches with enhanced information
        good_obs, good_actions, good_transformer_energies = process_enhanced_batch(
            good_batch
        )
        bad_obs, bad_actions, bad_transformer_energies = process_enhanced_batch(
            bad_batch
        )

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

        # Calculate energies with enhanced verifier
        good_energies = self.verifier(good_obs_stacked, good_actions_stacked)
        bad_energies = self.verifier(bad_obs_stacked, bad_actions_stacked)

        # Enhanced contrastive loss
        good_energy_mean = good_energies.mean()
        bad_energy_mean = bad_energies.mean()

        # We want good energies to be lower (more negative) than bad energies
        energy_diff = bad_energy_mean - good_energy_mean
        contrastive_loss = torch.clamp(margin - energy_diff, min=0.0)

        # Enhanced regularization with transformer awareness
        energy_reg = 0.01 * (good_energies.pow(2).mean() + bad_energies.pow(2).mean())

        # Transformer energy regularization (encourage meaningful but not excessive usage)
        transformer_reg = 0.0
        if good_transformer_energies and bad_transformer_energies:
            good_transformer_mean = np.mean([abs(e) for e in good_transformer_energies])
            bad_transformer_mean = np.mean([abs(e) for e in bad_transformer_energies])

            # Encourage moderate transformer usage in good experiences
            if good_transformer_mean > 0.1:  # Some usage is good
                transformer_reg += 0.001 * abs(
                    good_transformer_mean - 1.0
                )  # Target around 1.0

            transformer_reg = torch.tensor(transformer_reg, device=device)

        total_loss = contrastive_loss + energy_reg + transformer_reg

        return total_loss, {
            "contrastive_loss": contrastive_loss.item(),
            "energy_reg": energy_reg.item(),
            "transformer_reg": (
                transformer_reg.item()
                if isinstance(transformer_reg, torch.Tensor)
                else transformer_reg
            ),
            "good_energy_mean": good_energy_mean.item(),
            "bad_energy_mean": bad_energy_mean.item(),
            "energy_separation": energy_diff.item(),
            "good_transformer_energy": (
                np.mean([abs(e) for e in good_transformer_energies])
                if good_transformer_energies
                else 0.0
            ),
            "bad_transformer_energy": (
                np.mean([abs(e) for e in bad_transformer_energies])
                if bad_transformer_energies
                else 0.0
            ),
        }

    def train_step(self):
        """Enhanced training step with transformer-aware optimization."""
        # Sample balanced batch
        good_batch, bad_batch, golden_batch = (
            self.experience_buffer.sample_enhanced_balanced_batch(
                self.args.batch_size, golden_ratio=0.15
            )
        )

        if good_batch is None or bad_batch is None:
            return None  # Not enough experiences yet

        # Calculate enhanced loss
        loss, loss_info = self.calculate_enhanced_contrastive_loss(
            good_batch, bad_batch, margin=self.args.contrastive_margin
        )

        # Get current learning rates
        current_lr, current_thinking_lr = self.stability_manager.get_current_lrs()

        # Update optimizer learning rate if changed
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr

        # Update agent thinking learning rate
        self.agent.current_thinking_lr = current_thinking_lr

        # Enhanced backward pass with gradient clipping for transformers
        self.optimizer.zero_grad()
        loss.backward()

        # Enhanced gradient clipping (important for transformers)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.verifier.parameters(), max_norm=1.0
        )

        # Check for gradient explosion (more sensitive for transformers)
        if grad_norm > 5.0:  # Lower threshold
            print(f"⚠️ Large gradient norm detected: {grad_norm:.2f}")
            return None

        self.optimizer.step()

        # Update learning rate scheduler
        if self.episode % 10 == 0:  # Update every 10 episodes
            self.scheduler.step()

        # Add enhanced info to loss info
        loss_info["grad_norm"] = grad_norm.item()
        loss_info["learning_rate"] = current_lr
        loss_info["thinking_lr"] = current_thinking_lr
        loss_info["scheduler_lr"] = self.scheduler.get_last_lr()[0]

        return loss_info

    def evaluate_performance(self):
        """Enhanced evaluation with transformer analysis."""
        eval_episodes = min(5, max(1, self.episode // 100))

        wins = 0
        total_reward = 0.0
        total_steps = 0
        total_transformer_energy = 0.0
        transformer_usage_count = 0

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
                action, thinking_info = self.agent.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.env.step(action)

                episode_reward += reward
                episode_steps += 1

                # Track transformer usage in evaluation
                if "avg_transformer_energy" in thinking_info:
                    total_transformer_energy += abs(
                        thinking_info["avg_transformer_energy"]
                    )
                    transformer_usage_count += 1

                # Check for win (SINGLE ROUND LOGIC PRESERVED)
                reward_breakdown = info.get("reward_breakdown", {})
                if "round_won" in reward_breakdown:
                    wins += 1
                    break

            total_reward += episode_reward
            total_steps += episode_steps

        win_rate = wins / eval_episodes
        avg_reward = total_reward / eval_episodes
        avg_steps = total_steps / eval_episodes
        avg_transformer_energy = safe_divide(
            total_transformer_energy, transformer_usage_count, 0.0
        )

        return {
            "win_rate": win_rate,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
            "avg_transformer_energy": avg_transformer_energy,
            "eval_episodes": eval_episodes,
        }

    def handle_enhanced_policy_memory_operations(self, performance_stats, train_stats):
        """Enhanced policy memory operations with transformer awareness."""
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
            # Save peak checkpoint with enhanced info
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
            print(f"📉 PERFORMANCE DROP DETECTED - Activating Enhanced Policy Memory!")

            # Attempt checkpoint averaging
            if self.policy_memory.should_perform_averaging(self.episode):
                print(f"🔄 Performing enhanced checkpoint averaging...")
                averaging_success = self.policy_memory.perform_checkpoint_averaging(
                    self.verifier
                )

                if averaging_success:
                    print(f"✅ Enhanced checkpoint averaging completed successfully")
                    policy_memory_action_taken = True

                    # Reset learning rate scheduler
                    self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                        self.optimizer,
                        T_0=50,
                        T_mult=2,
                        eta_min=self.args.learning_rate * 0.1,
                    )

                    # Also reduce learning rate
                    if self.policy_memory.should_reduce_lr():
                        new_lr = self.policy_memory.get_reduced_lr(current_lr)
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = new_lr

                        # Update stability manager
                        self.stability_manager.current_lr = new_lr
                        print(f"📉 Learning rate reduced to {new_lr:.2e}")
                else:
                    print(f"❌ Enhanced checkpoint averaging failed")

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

    def analyze_enhanced_performance_trends(self):
        """Analyze performance trends with transformer insights."""
        if len(self.transformer_energy_history) < 10:
            return {}

        recent_transformer_energy = list(self.transformer_energy_history)[-10:]
        recent_attention = (
            list(self.attention_analysis)[-5:] if self.attention_analysis else []
        )

        analysis = {
            "transformer_energy_trend": np.mean(recent_transformer_energy),
            "transformer_energy_stability": 1.0
            / (np.std(recent_transformer_energy) + 1e-6),
            "attention_complexity": 0.0,
            "attention_variance": 0.0,
        }

        if recent_attention:
            analysis["attention_complexity"] = np.mean(
                [a["energy_variance"] for a in recent_attention]
            )
            analysis["attention_variance"] = np.std(
                [a["avg_transformer_energy"] for a in recent_attention]
            )

        return analysis

    def train(self):
        """Enhanced main training loop with EBT integration."""
        print(f"🛡️ Starting Enhanced EBT Training")
        print(f"   - Target episodes: {self.args.max_episodes}")
        print(f"   - Quality threshold: {self.args.quality_threshold} (FIXED)")
        print(f"   - Fast game UI: MAINTAINED")
        print(f"   - Single round logic: PRESERVED")

        # Training metrics
        episode_rewards = deque(maxlen=100)
        recent_losses = deque(maxlen=50)
        training_start_time = time.time()

        for episode in range(self.args.max_episodes):
            self.episode = episode
            episode_start_time = time.time()

            # Run enhanced episode
            episode_stats = self.run_enhanced_episode()
            episode_rewards.append(episode_stats["reward"])

            # Enhanced training step if we have enough experiences
            if (
                len(self.experience_buffer.good_experiences)
                >= self.args.batch_size // 2
            ):
                train_stats = self.train_step()
                if train_stats:
                    recent_losses.append(train_stats.get("contrastive_loss", 0.0))
            else:
                train_stats = {}

            # Periodic evaluation and enhanced analysis
            if episode % self.args.eval_frequency == 0:
                # Enhanced performance evaluation
                performance_stats = self.evaluate_performance()
                self.win_rate_history.append(performance_stats["win_rate"])

                # Calculate enhanced energy quality metrics
                if train_stats:
                    energy_separation = train_stats.get("energy_separation", 0.0)
                    energy_quality = abs(energy_separation) * 10.0
                else:
                    energy_separation = 0.0
                    energy_quality = 0.0
                self.energy_quality_history.append(energy_quality)

                # Enhanced policy memory operations
                policy_memory_action = self.handle_enhanced_policy_memory_operations(
                    performance_stats, train_stats or {}
                )

                # Update stability manager with enhanced metrics
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
                    print(f"🚨 Enhanced stability emergency triggered!")
                    # Update learning rates from stability manager
                    new_lr, new_thinking_lr = self.stability_manager.get_current_lrs()
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = new_lr
                    self.agent.current_thinking_lr = new_thinking_lr

                # Enhanced checkpoint saving
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

                # Enhanced analysis and logging with detailed win/lose stats
                buffer_stats = self.experience_buffer.get_stats()
                performance_trends = self.analyze_enhanced_performance_trends()
                win_lose_stats = self.get_win_lose_stats()

                print(f"\n🎯 ENHANCED STATUS (Episode {episode}):")
                print(f"   - Good experiences: {buffer_stats['good_count']:,}")
                print(f"   - Bad experiences: {buffer_stats['bad_count']:,}")
                print(
                    f"   - Quality threshold: {buffer_stats['quality_threshold']:.3f}"
                )

                # Detailed win/lose statistics
                print(f"\n🏆 WIN/LOSE STATISTICS:")
                print(f"   - Total games: {win_lose_stats['total_games']}")
                print(
                    f"   - Wins: {win_lose_stats['wins']} | Losses: {win_lose_stats['losses']} | Draws: {win_lose_stats['draws']}"
                )
                print(f"   - Overall win rate: {win_lose_stats['win_rate']:.1%}")
                print(
                    f"   - Recent win rate (last 10): {win_lose_stats['recent_win_rate']:.1%}"
                )

                # Win/lose type breakdown
                if win_lose_stats["wins"] > 0:
                    print(f"   - Win types: {dict(win_lose_stats['win_types'])}")
                if win_lose_stats["losses"] > 0:
                    print(f"   - Lose types: {dict(win_lose_stats['lose_types'])}")

                # Recent results pattern
                recent_results_str = "".join(
                    [
                        (
                            "🏆"
                            if r == "WIN"
                            else "💀" if r == "LOSE" else "🤝" if r == "DRAW" else "❓"
                        )
                        for r in win_lose_stats["recent_results"][-10:]
                    ]
                )
                print(f"   - Recent pattern: {recent_results_str}")

                # Recent termination reasons
                if win_lose_stats["recent_terminations"]:
                    print(
                        f"   - Recent terminations: {dict(win_lose_stats['recent_terminations'])}"
                    )

                print(f"\n🧠 TRANSFORMER STATUS:")
                print(
                    f"   - Avg transformer energy: {performance_stats.get('avg_transformer_energy', 0.0):.3f}"
                )

                if performance_trends:
                    print(
                        f"   - Transformer trend: {performance_trends.get('transformer_energy_trend', 0.0):.3f}"
                    )
                    print(
                        f"   - Attention complexity: {performance_trends.get('attention_complexity', 0.0):.3f}"
                    )

                # Enhanced success checking
                if buffer_stats["good_count"] > 0:
                    print(
                        f"\n✅ ENHANCED LEARNING: Both EBT and quality fixes working!"
                    )
                else:
                    print(f"\n❌ ISSUE: Quality threshold still needs adjustment")

                # Show if single-round logic is working
                if "round_2_prevention" in win_lose_stats.get(
                    "recent_terminations", {}
                ):
                    print(
                        f"⚠️  WARNING: Round 2 prevention activated {win_lose_stats['recent_terminations']['round_2_prevention']} times"
                    )
                else:
                    print(f"✅ SINGLE-ROUND: No round 2 detection in recent games")

                # Log enhanced metrics with win/lose details
                thinking_stats = self.agent.get_thinking_stats()
                self.logger.info(
                    f"Ep {episode}: "
                    f"Wins={win_lose_stats['wins']}, "
                    f"Losses={win_lose_stats['losses']}, "
                    f"WinRate={win_lose_stats['win_rate']:.3f}, "
                    f"RecentWinRate={win_lose_stats['recent_win_rate']:.3f}, "
                    f"Good={buffer_stats['good_count']}, "
                    f"TransformerEnergy={performance_stats.get('avg_transformer_energy', 0.0):.3f}, "
                    f"ThinkingSuccess={thinking_stats.get('success_rate', 0.0):.3f}, "
                    f"TransformerUsage={thinking_stats.get('transformer_energy_usage', 0.0):.3f}, "
                    f"LastTermination={episode_stats.get('termination_reason', 'unknown')}"
                )

                # Enhanced training info display
                if train_stats:
                    print(
                        f"\n🧠 Training: Energy sep={train_stats.get('energy_separation', 0.0):.3f}, "
                        f"Transformer reg={train_stats.get('transformer_reg', 0.0):.4f}"
                    )

            # Track termination reasons for analysis (moved outside eval check)
            if not hasattr(self, "recent_termination_reasons"):
                self.recent_termination_reasons = deque(maxlen=20)
            self.recent_termination_reasons.append(
                episode_stats.get("termination_reason", "unknown")
            )

            # Early stopping check with win rate
            if len(self.win_rate_history) >= 20:
                recent_win_rate = safe_mean(list(self.win_rate_history)[-10:], 0.0)
                if recent_win_rate >= self.args.target_win_rate:
                    print(
                        f"🎯 Target win rate {self.args.target_win_rate:.1%} achieved with Enhanced EBT!"
                    )
                    break

        # Training completed
        final_performance = self.evaluate_performance()
        final_trends = self.analyze_enhanced_performance_trends()
        final_win_lose_stats = self.get_win_lose_stats()

        print(f"\n🏁 ENHANCED EBT Training Completed!")
        print(f"   - Total episodes: {self.episode + 1}")
        print(f"   - Final evaluation win rate: {final_performance['win_rate']:.1%}")
        print(f"   - Best win rate: {self.best_win_rate:.1%}")
        print(
            f"   - Avg transformer energy: {final_performance.get('avg_transformer_energy', 0.0):.3f}"
        )

        # Show final detailed win/lose stats
        print(f"\n🏆 FINAL WIN/LOSE STATISTICS:")
        print(f"   - Total training games: {final_win_lose_stats['total_games']}")
        print(
            f"   - Final record: {final_win_lose_stats['wins']}W - {final_win_lose_stats['losses']}L - {final_win_lose_stats['draws']}D"
        )
        print(f"   - Overall training win rate: {final_win_lose_stats['win_rate']:.1%}")
        print(f"   - Recent win rate: {final_win_lose_stats['recent_win_rate']:.1%}")

        if final_win_lose_stats["wins"] > 0:
            print(f"   - Victory methods: {dict(final_win_lose_stats['win_types'])}")
        if final_win_lose_stats["losses"] > 0:
            print(f"   - Defeat methods: {dict(final_win_lose_stats['lose_types'])}")

        # Show enhanced final stats
        final_buffer_stats = self.experience_buffer.get_stats()
        final_thinking_stats = self.agent.get_thinking_stats()

        print(f"\n🎯 FINAL ENHANCED STATS:")
        print(f"   - Good experiences: {final_buffer_stats['good_count']:,}")
        print(f"   - Bad experiences: {final_buffer_stats['bad_count']:,}")
        print(f"   - Good ratio: {final_buffer_stats['good_ratio']:.1%}")
        print(
            f"   - Thinking success rate: {final_thinking_stats.get('success_rate', 0.0):.1%}"
        )
        print(
            f"   - Transformer energy usage: {final_thinking_stats.get('transformer_energy_usage', 0.0):.3f}"
        )

        if final_trends:
            print(
                f"   - Final transformer trend: {final_trends.get('transformer_energy_trend', 0.0):.3f}"
            )
            print(
                f"   - Final attention complexity: {final_trends.get('attention_complexity', 0.0):.3f}"
            )

        # Final termination analysis
        if self.termination_reasons:
            termination_counts = {}
            for reason in self.termination_reasons:
                termination_counts[reason] = termination_counts.get(reason, 0) + 1
            print(f"   - Termination breakdown: {dict(termination_counts)}")

            if "round_2_prevention" in termination_counts:
                print(
                    f"   ⚠️  Round 2 prevention triggered {termination_counts['round_2_prevention']} times"
                )
            else:
                print(f"   ✅ Perfect single-round operation - no round 2 detected!")

        # Success assessment
        if (
            final_buffer_stats["good_count"] > 100
            and final_win_lose_stats["win_rate"] > 0.3
        ):
            print(f"\n🎉 SUCCESS: Enhanced EBT training worked excellently!")
            print(f"   - Quality threshold fix: ✅ Working")
            print(f"   - Single-round logic: ✅ Working")
            print(f"   - Transformer learning: ✅ Working")
            print(
                f"   - Win rate improvement: ✅ {final_win_lose_stats['win_rate']:.1%}"
            )
        else:
            print(f"\n🔧 NEEDS TUNING: Some components need adjustment")
            if final_buffer_stats["good_count"] <= 100:
                print(f"   - Quality threshold: ❌ Too few good experiences")
            if final_win_lose_stats["win_rate"] <= 0.3:
                print(
                    f"   - Win rate: ❌ Still learning ({final_win_lose_stats['win_rate']:.1%})"
                )

        # Save enhanced final checkpoint
        self.checkpoint_manager.save_checkpoint(
            self.verifier,
            self.agent,
            self.episode,
            final_performance["win_rate"],
            self.energy_quality_history[-1] if self.energy_quality_history else 0.0,
            policy_memory_stats=self.policy_memory.get_stats(),
        )


def main():
    """Enhanced main training function with EBT support."""
    parser = argparse.ArgumentParser(
        description="Enhanced EBT Training - Energy-Based Transformers + Energy Thinking"
    )

    # Environment arguments
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=1500,  # Increased for transformer training
        help="Maximum number of training episodes",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=1200,
        help="Maximum steps per episode (single round preserved)",
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

    # Enhanced model arguments
    parser.add_argument(
        "--features-dim",
        type=int,
        default=256,
        help="Feature dimension for enhanced verifier",
    )
    parser.add_argument(
        "--thinking-steps",
        type=int,
        default=5,  # Increased for transformer
        help="Number of thinking steps for enhanced agent",
    )
    parser.add_argument(
        "--thinking-lr",
        type=float,
        default=0.05,  # Adjusted for transformer
        help="Learning rate for enhanced thinking process",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.02,
        help="Noise scale for action initialization",
    )

    # Enhanced training arguments
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,  # Reduced for transformer stability
        help="Learning rate for enhanced verifier",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,  # Reduced for transformer
        help="Weight decay for regularization",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Training batch size"
    )
    parser.add_argument(
        "--contrastive-margin",
        type=float,
        default=2.5,  # Increased for better separation
        help="Margin for enhanced contrastive loss",
    )

    # Device and rendering
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use for training"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment (fast UI maintained)",
    )

    # Enhanced experience buffer arguments with FIXED quality threshold
    parser.add_argument(
        "--buffer-capacity",
        type=int,
        default=40000,  # Increased for transformer
        help="Enhanced experience buffer capacity",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.3,  # FIXED THRESHOLD
        help="Quality threshold (THE MAIN FIX)",
    )
    parser.add_argument(
        "--golden-buffer-capacity",
        type=int,
        default=1500,  # Increased
        help="Golden experience buffer capacity",
    )

    # Policy memory arguments
    parser.add_argument(
        "--performance-drop-threshold",
        type=float,
        default=0.05,
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
        default=0.65,  # Slightly increased target
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
        default="checkpoints_enhanced_ebt",
        help="Directory for saving enhanced checkpoints",
    )
    parser.add_argument(
        "--load-checkpoint", type=str, default=None, help="Path to checkpoint to load"
    )

    args = parser.parse_args()

    # Print enhanced configuration
    print(f"🛡️ Enhanced EBT Training Configuration:")
    print(f"   Max Episodes: {args.max_episodes:,}")
    print(f"   Learning Rate: {args.learning_rate:.2e}")
    print(f"   🎯 Quality Threshold: {args.quality_threshold} (FIXED - was 0.6)")
    print(f"   Target Win Rate: {args.target_win_rate:.1%}")
    print(f"   Max Fight Steps: {MAX_FIGHT_STEPS} (single round preserved)")
    print(f"   Device: {args.device}")
    print(f"   ⚡ Fast Game UI: MAINTAINED")
    print(f"   🧠 Energy-Based Transformers: ENABLED")
    print(f"   🔄 Enhanced Thinking Steps: {args.thinking_steps}")

    # Warn if threshold is still too high
    if args.quality_threshold > 0.5:
        print(
            f"   ⚠️  WARNING: Quality threshold {args.quality_threshold} might still be too high"
        )
        print(f"      Consider using 0.3 or lower for better learning")

    # Initialize and run enhanced trainer
    try:
        trainer = EnhancedEBTTrainer(args)

        # Load checkpoint if specified
        if args.load_checkpoint:
            print(f"📂 Loading enhanced checkpoint: {args.load_checkpoint}")
            trainer.checkpoint_manager.load_checkpoint(
                Path(args.load_checkpoint), trainer.verifier, trainer.agent
            )

        # Start enhanced training
        trainer.train()

    except KeyboardInterrupt:
        print(f"\n⚠️  Enhanced training interrupted by user")
    except Exception as e:
        print(f"\n❌ Enhanced training failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
