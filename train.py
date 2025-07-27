#!/usr/bin/env python3
"""
🚀 ENHANCED TRAINING - Breaks Learning Plateaus with Multiple Strategies
Key Improvements:
1. Time-decayed winning bonuses (fast wins >>> slow wins)
2. Aggressive epsilon-greedy exploration (25% random actions)
3. Reservoir sampling for experience diversity
4. Learning rate reboots when performance stagnates
5. Enhanced reward shaping against timeout strategies
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
import random

# Import the ENHANCED wrapper components
try:
    from wrapper import (
        make_enhanced_env,
        verify_health_detection,
        SimpleVerifier,
        AggressiveAgent,
        SimpleCNN,
        safe_mean,
        safe_std,
        safe_divide,
        MAX_FIGHT_STEPS,
        MAX_HEALTH,
        FRAME_STACK_SIZE,
    )

    print("✅ Successfully imported enhanced wrapper components")
    print(f"✅ Aggressive exploration and time-decayed rewards: ACTIVE")
except ImportError as e:
    print(f"❌ Failed to import enhanced wrapper: {e}")
    print("Make sure enhanced_wrapper.py is in the same directory")
    exit(1)


class ReservoirExperienceBuffer:
    """🎯 Enhanced experience buffer with reservoir sampling for maximum diversity."""

    def __init__(self, capacity=30000):
        self.capacity = capacity
        self.good_experiences = []
        self.bad_experiences = []
        self.total_added = 0

        # NEW: Reservoir sampling state
        self.good_reservoir_size = capacity // 2
        self.bad_reservoir_size = capacity // 2

        # Diversity tracking
        self.sequence_quality_tracker = deque(maxlen=1000)
        self.diversity_scores = deque(maxlen=500)

    def add_experience(self, experience, reward, win_result):
        """Add experience using reservoir sampling for maximum diversity."""
        self.total_added += 1

        # Enhanced classification with temporal context
        temporal_quality = self._assess_temporal_quality(experience)
        diversity_score = self._assess_diversity(experience)

        # Store diversity metrics
        self.sequence_quality_tracker.append(temporal_quality)
        self.diversity_scores.append(diversity_score)

        # Enhanced classification - prioritize diverse and high-quality experiences
        is_good_experience = (
            win_result == "WIN"
            or (reward > 0.2 and temporal_quality > 0.4)
            or (
                diversity_score > 0.7 and temporal_quality > 0.3
            )  # High diversity experiences
        )

        # Add experience using reservoir sampling
        if is_good_experience:
            self._reservoir_add(
                self.good_experiences, experience, self.good_reservoir_size
            )
            experience["sequence_quality"] = "good"
        else:
            self._reservoir_add(
                self.bad_experiences, experience, self.bad_reservoir_size
            )
            experience["sequence_quality"] = "bad"

    def _reservoir_add(self, reservoir, experience, max_size):
        """Implement reservoir sampling for maintaining diversity."""
        if len(reservoir) < max_size:
            # Still filling the reservoir
            reservoir.append(experience)
        else:
            # Reservoir is full - randomly replace an existing experience
            # This gives ALL experiences (including early discoveries) a chance to survive
            random_index = random.randint(0, len(reservoir) - 1)
            reservoir[random_index] = experience

    def _assess_temporal_quality(self, experience):
        """Assess the temporal quality of an experience."""
        reward = experience.get("reward", 0.0)
        thinking_info = experience.get("thinking_info", {})

        # Consider exploration vs exploitation
        is_exploration = thinking_info.get("exploration", False)
        energy_improvement = thinking_info.get("energy_improvement", False)
        final_energy = thinking_info.get("final_energy", 0.0)

        quality = 0.5  # Base quality

        # Reward-based quality
        if reward > 0:
            quality += min(reward * 0.4, 0.4)
        elif reward < 0:
            quality -= min(abs(reward) * 0.3, 0.3)

        # Energy-based quality
        if energy_improvement:
            quality += 0.2
        if final_energy < 0:
            quality += 0.1

        # Exploration bonus - exploration experiences are valuable for diversity
        if is_exploration:
            quality += 0.1

        return np.clip(quality, 0.0, 1.0)

    def _assess_diversity(self, experience):
        """Assess how diverse/unique this experience is."""
        action = experience.get("action", 0)
        reward = experience.get("reward", 0.0)
        thinking_info = experience.get("thinking_info", {})

        # Base diversity from action uniqueness
        diversity = 0.5

        # Action diversity (rarer actions get higher scores)
        if hasattr(self, "action_counts"):
            total_actions = sum(self.action_counts.values())
            action_frequency = self.action_counts.get(action, 1) / max(total_actions, 1)
            # Rare actions get higher diversity scores
            diversity += (1.0 - action_frequency) * 0.3
        else:
            self.action_counts = {}

        # Update action counts
        self.action_counts[action] = self.action_counts.get(action, 0) + 1

        # Exploration experiences are inherently diverse
        if thinking_info.get("exploration", False):
            diversity += 0.2

        # Unique reward patterns increase diversity
        if abs(reward) > 1.0:  # Extreme rewards are diverse
            diversity += 0.1

        return np.clip(diversity, 0.0, 1.0)

    def sample_balanced_batch(self, batch_size):
        """Sample balanced batch with enhanced diversity consideration."""
        if (
            len(self.good_experiences) < batch_size // 4
            or len(self.bad_experiences) < batch_size // 4
        ):
            return None, None

        good_count = batch_size // 2
        bad_count = batch_size // 2

        # Sample with diversity bias - prefer more diverse experiences
        good_batch = self._sample_with_diversity_bias(self.good_experiences, good_count)
        bad_batch = self._sample_with_diversity_bias(self.bad_experiences, bad_count)

        return good_batch, bad_batch

    def _sample_with_diversity_bias(self, experience_list, count):
        """Sample with bias toward diverse experiences."""
        if len(experience_list) <= count:
            return experience_list[:]

        # Calculate diversity weights for each experience
        weights = []
        for exp in experience_list:
            action = exp.get("action", 0)
            thinking_info = exp.get("thinking_info", {})

            weight = 1.0  # Base weight

            # Boost rare actions
            if hasattr(self, "action_counts"):
                total_actions = sum(self.action_counts.values())
                action_frequency = self.action_counts.get(action, 1) / max(
                    total_actions, 1
                )
                weight *= 2.0 - action_frequency  # Rare actions get up to 2x weight

            # Boost exploration experiences
            if thinking_info.get("exploration", False):
                weight *= 1.5

            weights.append(weight)

        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()

        # Sample without replacement
        indices = np.random.choice(
            len(experience_list), size=count, replace=False, p=weights
        )
        return [experience_list[i] for i in indices]

    def get_stats(self):
        """Get comprehensive buffer statistics."""
        total_size = len(self.good_experiences) + len(self.bad_experiences)
        avg_sequence_quality = safe_mean(list(self.sequence_quality_tracker), 0.5)
        avg_diversity = safe_mean(list(self.diversity_scores), 0.5)

        action_diversity = 0.0
        if hasattr(self, "action_counts") and self.action_counts:
            unique_actions = len(self.action_counts)
            total_actions = sum(self.action_counts.values())
            action_diversity = unique_actions / max(total_actions, 1)

        return {
            "total_size": total_size,
            "good_count": len(self.good_experiences),
            "bad_count": len(self.bad_experiences),
            "good_ratio": len(self.good_experiences) / max(1, total_size),
            "total_added": self.total_added,
            "avg_sequence_quality": avg_sequence_quality,
            "avg_diversity": avg_diversity,
            "action_diversity": action_diversity,
            "frame_stack_size": FRAME_STACK_SIZE,
        }


class EnhancedTrainer:
    """🚀 Enhanced trainer with learning rate reboots and plateau detection."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup directories
        self._setup_directories()

        # Initialize ENHANCED environment
        print(f"🚀 Initializing ENHANCED environment with aggressive exploration...")
        self.env = make_enhanced_env()

        # Verify enhanced system
        if args.verify_health:
            if not verify_health_detection(self.env):
                print("⚠️  System verification failed, but continuing anyway...")

        # Initialize enhanced models
        print(f"🧠 Initializing enhanced models with aggressive training...")
        self.verifier = SimpleVerifier(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            features_dim=args.features_dim,
        ).to(self.device)

        self.agent = AggressiveAgent(
            verifier=self.verifier,
            thinking_steps=args.thinking_steps,
            thinking_lr=args.thinking_lr,
        )

        # Enhanced experience buffer with reservoir sampling
        self.experience_buffer = ReservoirExperienceBuffer(
            capacity=args.buffer_capacity
        )

        # Enhanced optimizer with learning rate management
        self.initial_learning_rate = args.learning_rate
        self.optimizer = optim.Adam(
            self.verifier.parameters(),
            lr=self.initial_learning_rate,
            weight_decay=args.weight_decay,
            eps=1e-8,
        )

        # NEW: Learning rate scheduler with plateau detection
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.7, patience=8
        )

        # NEW: Learning rate reboot system
        self.lr_reboot_threshold = 0.02  # Reboot when performance stagnates
        self.performance_history = deque(maxlen=20)
        self.last_reboot_episode = 0
        self.reboot_count = 0

        # Training state
        self.episode = 0
        self.total_steps = 0
        self.best_win_rate = 0.0

        # Enhanced performance tracking
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.recent_results = deque(maxlen=40)  # Increased for better tracking
        self.win_rate_history = deque(maxlen=100)
        self.recent_losses = deque(maxlen=100)

        # NEW: Plateau detection metrics
        self.timeout_wins = 0
        self.fast_wins = 0
        self.combo_count_history = deque(maxlen=50)
        self.speed_history = deque(maxlen=50)

        # Termination tracking
        self.termination_reasons = deque(maxlen=200)

        # Setup logging
        self.setup_logging()

        # Load checkpoint if provided
        self.load_checkpoint()

        print(f"🚀 Enhanced Trainer initialized")
        print(f"   - Device: {self.device}")
        print(f"   - Learning rate: {args.learning_rate:.2e} (with reboots)")
        print(f"   - Aggressive exploration: {self.agent.epsilon:.1%}")
        print(f"   - Reservoir sampling: ENABLED")
        print(f"   - Plateau detection: ACTIVE")

    def _setup_directories(self):
        """Create necessary directories."""
        self.log_dir = Path("logs_enhanced")
        self.checkpoint_dir = Path("checkpoints_enhanced")
        self.log_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)

    def setup_logging(self):
        """Setup enhanced logging."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"enhanced_training_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def detect_learning_plateau(self):
        """Detect if learning has plateaued and needs a reboot."""
        if len(self.performance_history) < 15:
            return False

        # Check recent performance variance
        recent_performance = list(self.performance_history)[-10:]
        performance_std = safe_std(recent_performance, 0.0)

        # Check for stagnation
        is_stagnant = performance_std < self.lr_reboot_threshold

        # Check for timeout strategy dominance
        recent_terminations = list(self.termination_reasons)[-20:]
        timeout_ratio = sum(
            1 for term in recent_terminations if "timeout" in term
        ) / max(1, len(recent_terminations))
        timeout_dominance = timeout_ratio > 0.7

        # Check for lack of improvement
        early_performance = safe_mean(recent_performance[:5], 0.0)
        late_performance = safe_mean(recent_performance[-5:], 0.0)
        no_improvement = late_performance <= early_performance + 0.01

        should_reboot = (
            is_stagnant
            and (timeout_dominance or no_improvement)
            and self.episode - self.last_reboot_episode > 50
        )

        if should_reboot:
            self.logger.info(
                f"🔄 Plateau detected: std={performance_std:.4f}, "
                f"timeout_ratio={timeout_ratio:.2f}, improvement={late_performance-early_performance:.4f}"
            )

        return should_reboot

    def reboot_learning_rate(self):
        """Reboot learning rate and exploration to break out of plateaus."""
        self.reboot_count += 1
        self.last_reboot_episode = self.episode

        # Reset learning rate to initial value (or slightly higher for shock)
        new_lr = self.initial_learning_rate * (
            1.2**self.reboot_count
        )  # Gradually increase
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr

        # Reset scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.7, patience=8
        )

        # Boost exploration temporarily
        self.agent.epsilon = min(0.4, self.agent.epsilon * 2.0)  # Double exploration

        self.logger.info(f"🚀 LEARNING RATE REBOOT #{self.reboot_count}")
        self.logger.info(f"   - New LR: {new_lr:.2e}")
        self.logger.info(f"   - Boosted exploration: {self.agent.epsilon:.1%}")

        print(f"🚀 LEARNING RATE REBOOT #{self.reboot_count}!")
        print(f"   - Learning rate reset to: {new_lr:.2e}")
        print(f"   - Exploration boosted to: {self.agent.epsilon:.1%}")
        print(f"   - Breaking out of plateau!")

    def save_checkpoint(self, episode):
        """Save enhanced checkpoint."""
        if not self.args.save_frequency > 0:
            return

        filename = self.checkpoint_dir / f"enhanced_checkpoint_ep_{episode}.pth"
        state = {
            "episode": episode,
            "verifier_state_dict": self.verifier.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "total_steps": self.total_steps,
            "best_win_rate": self.best_win_rate,
            "frame_stack_size": FRAME_STACK_SIZE,
            "reboot_count": self.reboot_count,
            "last_reboot_episode": self.last_reboot_episode,
            "timeout_wins": self.timeout_wins,
            "fast_wins": self.fast_wins,
            "agent_epsilon": self.agent.epsilon,
            "args": self.args,
        }
        torch.save(state, filename)
        self.logger.info(f"💾 Enhanced checkpoint saved to {filename}")

    def load_checkpoint(self):
        """Load enhanced checkpoint."""
        if self.args.load_checkpoint:
            checkpoint_path = self.args.load_checkpoint
            if os.path.exists(checkpoint_path):
                self.logger.info(
                    f"🔄 Loading enhanced checkpoint from {checkpoint_path}..."
                )
                checkpoint = torch.load(checkpoint_path, map_location=self.device)

                self.verifier.load_state_dict(checkpoint["verifier_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

                if "scheduler_state_dict" in checkpoint:
                    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

                # Load enhanced training state
                self.episode = checkpoint["episode"] + 1
                self.wins = checkpoint.get("wins", 0)
                self.losses = checkpoint.get("losses", 0)
                self.draws = checkpoint.get("draws", 0)
                self.total_steps = checkpoint.get("total_steps", 0)
                self.best_win_rate = checkpoint.get("best_win_rate", 0.0)
                self.reboot_count = checkpoint.get("reboot_count", 0)
                self.last_reboot_episode = checkpoint.get("last_reboot_episode", 0)
                self.timeout_wins = checkpoint.get("timeout_wins", 0)
                self.fast_wins = checkpoint.get("fast_wins", 0)

                # Restore agent state
                if "agent_epsilon" in checkpoint:
                    self.agent.epsilon = checkpoint["agent_epsilon"]

                self.logger.info(
                    f"✅ Enhanced checkpoint loaded. Resuming from episode {self.episode}."
                )
            else:
                self.logger.warning(f"⚠️ Checkpoint file not found. Starting fresh.")

    def run_episode(self):
        """Run enhanced episode with aggressive exploration."""
        obs, info = self.env.reset()
        done = False
        truncated = False

        episode_reward = 0.0
        episode_steps = 0
        episode_experiences = []

        # Episode tracking
        round_won = False
        round_lost = False
        round_draw = False
        termination_reason = "ongoing"

        # NEW: Enhanced tracking
        max_combo_length = 0
        total_damage_dealt = 0.0
        is_fast_win = False

        while (
            not done and not truncated and episode_steps < self.args.max_episode_steps
        ):
            # Enhanced agent prediction with aggressive exploration
            action, thinking_info = self.agent.predict(obs, deterministic=False)

            # Execute action
            next_obs, reward, done, truncated, info = self.env.step(action)

            episode_reward += reward
            episode_steps += 1
            self.total_steps += 1

            # Enhanced tracking
            reward_breakdown = info.get("reward_breakdown", {})
            combo_frames = reward_breakdown.get("combo_frames", 0)
            damage_dealt = reward_breakdown.get("damage_dealt", 0.0)

            max_combo_length = max(max_combo_length, combo_frames)
            total_damage_dealt += damage_dealt

            # Track round result
            if info.get("round_ended", False):
                termination_reason = info.get("termination_reason", "unknown")
                round_result = info.get("round_result", "ONGOING")

                if round_result == "WIN":
                    round_won = True
                    # Check if it's a fast win
                    if episode_steps < MAX_FIGHT_STEPS * 0.5:
                        is_fast_win = True
                        self.fast_wins += 1
                    # Check for timeout win (bad strategy)
                    if "timeout" in termination_reason:
                        self.timeout_wins += 1
                elif round_result == "LOSE":
                    round_lost = True
                elif round_result == "DRAW":
                    round_draw = True

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
                "frame_stack_size": FRAME_STACK_SIZE,
                "enhanced_context": {
                    "combo_length": combo_frames,
                    "damage_dealt": damage_dealt,
                    "is_exploration": thinking_info.get("exploration", False),
                    "episode_progress": episode_steps / MAX_FIGHT_STEPS,
                },
            }

            episode_experiences.append(experience)
            obs = next_obs

        # Update enhanced statistics
        if round_won:
            self.wins += 1
            self.recent_results.append("WIN")
            win_result = "WIN"
        elif round_lost:
            self.losses += 1
            self.recent_results.append("LOSE")
            win_result = "LOSE"
        elif round_draw:
            self.draws += 1
            self.recent_results.append("DRAW")
            win_result = "DRAW"
        else:
            self.recent_results.append("UNKNOWN")
            win_result = "UNKNOWN"

        # Track termination and enhanced metrics
        self.termination_reasons.append(termination_reason)
        self.combo_count_history.append(max_combo_length)
        self.speed_history.append(episode_steps)

        # Add experiences to enhanced buffer
        for experience in episode_experiences:
            self.experience_buffer.add_experience(
                experience, experience["reward"], win_result
            )

        # Enhanced episode stats
        episode_stats = {
            "won": round_won,
            "lost": round_lost,
            "draw": round_draw,
            "reward": episode_reward,
            "steps": episode_steps,
            "termination_reason": termination_reason,
            "player_health": info.get("player_health", MAX_HEALTH),
            "opponent_health": info.get("opponent_health", MAX_HEALTH),
            "health_detection_working": info.get("health_detection_working", False),
            "max_combo_length": max_combo_length,
            "total_damage_dealt": total_damage_dealt,
            "is_fast_win": is_fast_win,
            "exploration_rate": self.agent.epsilon,
            "frame_stack_size": FRAME_STACK_SIZE,
        }

        return episode_stats

    def calculate_contrastive_loss(self, good_batch, bad_batch, margin=3.0):
        """Enhanced contrastive loss with better separation."""
        device = self.device

        def process_batch(batch):
            if not batch:
                return None, None

            obs_batch = []
            action_batch = []

            for exp in batch:
                obs = exp["obs"]
                action = exp["action"]

                # Convert observations
                if isinstance(obs, dict):
                    obs_tensor = {}
                    for key, val in obs.items():
                        if isinstance(val, np.ndarray):
                            obs_tensor[key] = torch.from_numpy(val).float()
                        else:
                            obs_tensor[key] = val.float()
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

        # Stack observations
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

        # Enhanced contrastive loss
        good_energy_mean = good_energies.mean()
        bad_energy_mean = bad_energies.mean()

        energy_diff = bad_energy_mean - good_energy_mean
        contrastive_loss = torch.clamp(margin - energy_diff, min=0.0)

        # Enhanced regularization
        energy_reg = 0.01 * (good_energies.pow(2).mean() + bad_energies.pow(2).mean())

        # NEW: Diversity regularization to encourage variety
        good_energy_var = good_energies.var()
        bad_energy_var = bad_energies.var()
        diversity_reg = 0.005 * (good_energy_var + bad_energy_var)

        # NEW: Exploration bonus regularization
        exploration_bonus = 0.0
        for exp in good_batch + bad_batch:
            if exp.get("thinking_info", {}).get("exploration", False):
                exploration_bonus += 0.001

        total_loss = contrastive_loss + energy_reg + diversity_reg + exploration_bonus

        return total_loss, {
            "contrastive_loss": contrastive_loss.item(),
            "energy_reg": energy_reg.item(),
            "diversity_reg": diversity_reg.item(),
            "exploration_bonus": exploration_bonus,
            "good_energy_mean": good_energy_mean.item(),
            "bad_energy_mean": bad_energy_mean.item(),
            "energy_separation": energy_diff.item(),
        }

    def train_step(self):
        """Enhanced training step."""
        # Sample diverse batch
        good_batch, bad_batch = self.experience_buffer.sample_balanced_batch(
            self.args.batch_size
        )

        if good_batch is None or bad_batch is None:
            return None

        # Calculate enhanced loss
        loss, loss_info = self.calculate_contrastive_loss(
            good_batch, bad_batch, margin=self.args.contrastive_margin
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Enhanced gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.verifier.parameters(), max_norm=1.5
        )

        if grad_norm > 15.0:
            print(f"⚠️ Large gradient detected: {grad_norm:.2f}")
            return None

        self.optimizer.step()

        # Update scheduler
        self.scheduler.step(loss.item())

        loss_info["grad_norm"] = grad_norm.item()
        loss_info["learning_rate"] = self.optimizer.param_groups[0]["lr"]
        return loss_info

    def evaluate_performance(self):
        """Enhanced performance evaluation."""
        eval_episodes = 6

        wins = 0
        total_reward = 0.0
        total_steps = 0
        health_changes_detected = 0
        fast_wins = 0
        timeout_wins = 0
        combo_counts = []

        for _ in range(eval_episodes):
            obs, info = self.env.reset()
            done = False
            truncated = False
            episode_reward = 0.0
            episode_steps = 0
            max_combo = 0

            initial_player_health = info.get("player_health", MAX_HEALTH)
            initial_opponent_health = info.get("opponent_health", MAX_HEALTH)

            while (
                not done
                and not truncated
                and episode_steps < self.args.max_episode_steps
            ):
                action, thinking_info = self.agent.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.env.step(action)

                episode_reward += reward
                episode_steps += 1

                # Track combos
                reward_breakdown = info.get("reward_breakdown", {})
                combo_frames = reward_breakdown.get("combo_frames", 0)
                max_combo = max(max_combo, combo_frames)

                # Check for win
                round_result = info.get("round_result", "ONGOING")
                termination_reason = info.get("termination_reason", "ongoing")

                if round_result == "WIN":
                    wins += 1
                    if episode_steps < MAX_FIGHT_STEPS * 0.5:
                        fast_wins += 1
                    if "timeout" in termination_reason:
                        timeout_wins += 1
                    break

            # Health detection check
            final_player_health = info.get("player_health", MAX_HEALTH)
            final_opponent_health = info.get("opponent_health", MAX_HEALTH)

            if (
                final_player_health != initial_player_health
                or final_opponent_health != initial_opponent_health
            ):
                health_changes_detected += 1

            total_reward += episode_reward
            total_steps += episode_steps
            combo_counts.append(max_combo)

        win_rate = wins / eval_episodes
        avg_reward = total_reward / eval_episodes
        avg_steps = total_steps / eval_episodes
        health_detection_rate = health_changes_detected / eval_episodes
        avg_combo_length = safe_mean(combo_counts, 0.0)
        fast_win_rate = fast_wins / eval_episodes
        timeout_win_rate = timeout_wins / eval_episodes

        return {
            "win_rate": win_rate,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
            "health_detection_rate": health_detection_rate,
            "avg_combo_length": avg_combo_length,
            "fast_win_rate": fast_win_rate,
            "timeout_win_rate": timeout_win_rate,
            "eval_episodes": eval_episodes,
            "frame_stack_size": FRAME_STACK_SIZE,
        }

    def get_enhanced_stats(self):
        """Get comprehensive enhanced statistics."""
        total_games = self.wins + self.losses + self.draws

        if total_games == 0:
            return {
                "total_games": 0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "win_rate": 0.0,
                "recent_win_rate": 0.0,
                "draw_rate": 0.0,
                "timeout_strategy_rate": 0.0,
                "fast_win_rate": 0.0,
                "avg_combo_length": 0.0,
                "exploration_rate": self.agent.epsilon,
                "reboot_count": self.reboot_count,
                "recent_results": list(self.recent_results),
                "recent_terminations": {},
                "avg_speed": 0.0,
            }

        # Calculate enhanced rates
        win_rate = self.wins / total_games
        draw_rate = self.draws / total_games
        timeout_strategy_rate = self.timeout_wins / max(1, self.wins)
        fast_win_rate = self.fast_wins / max(1, self.wins)

        # Recent performance
        recent_wins = sum(
            1 for result in list(self.recent_results)[-20:] if result == "WIN"
        )
        recent_games = min(len(self.recent_results), 20)
        recent_win_rate = recent_wins / recent_games if recent_games > 0 else 0.0

        # Enhanced metrics
        avg_combo_length = safe_mean(list(self.combo_count_history), 0.0)
        avg_speed = safe_mean(list(self.speed_history), MAX_FIGHT_STEPS)

        # Recent termination analysis
        recent_terminations = {}
        for reason in list(self.termination_reasons)[-20:]:
            recent_terminations[reason] = recent_terminations.get(reason, 0) + 1

        return {
            "total_games": total_games,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": win_rate,
            "draw_rate": draw_rate,
            "recent_win_rate": recent_win_rate,
            "timeout_strategy_rate": timeout_strategy_rate,
            "fast_win_rate": fast_win_rate,
            "avg_combo_length": avg_combo_length,
            "avg_speed": avg_speed,
            "exploration_rate": self.agent.epsilon,
            "recent_results": list(self.recent_results),
            "recent_terminations": recent_terminations,
            "reboot_count": self.reboot_count,
        }

    def train(self):
        """Enhanced main training loop with plateau detection and learning rate reboots."""
        print(f"🚀 Starting ENHANCED Training with Aggressive Exploration")
        print(f"   - Target episodes: {self.args.max_episodes}")
        print(f"   - Starting from episode: {self.episode}")
        print(f"   - Aggressive exploration: {self.agent.epsilon:.1%}")
        print(f"   - Reservoir sampling: ACTIVE")
        print(f"   - Learning rate reboots: ENABLED")
        print(f"   - Focus: BREAK PLATEAUS & ELIMINATE TIMEOUT STRATEGIES")

        training_start_time = time.time()

        for episode in range(self.episode, self.args.max_episodes):
            self.episode = episode
            episode_start_time = time.time()

            # Run enhanced episode
            episode_stats = self.run_episode()

            # Enhanced training step
            if (
                len(self.experience_buffer.good_experiences)
                >= self.args.batch_size // 4
            ):
                train_stats = self.train_step()
                if train_stats:
                    self.recent_losses.append(train_stats.get("contrastive_loss", 0.0))
            else:
                train_stats = {}

            # Enhanced periodic evaluation and plateau detection
            if episode % self.args.eval_frequency == 0:
                # Performance evaluation
                performance_stats = self.evaluate_performance()
                self.win_rate_history.append(performance_stats["win_rate"])

                # Add to performance history for plateau detection
                composite_performance = (
                    performance_stats["win_rate"] * 0.4
                    + (1 - performance_stats.get("timeout_win_rate", 0.0)) * 0.3
                    + performance_stats.get("fast_win_rate", 0.0) * 0.3
                )
                self.performance_history.append(composite_performance)

                # PLATEAU DETECTION AND LEARNING RATE REBOOT
                if self.detect_learning_plateau():
                    self.reboot_learning_rate()

                # Get comprehensive stats
                buffer_stats = self.experience_buffer.get_stats()
                enhanced_stats = self.get_enhanced_stats()

                # Enhanced status display
                print(f"\n🚀 ENHANCED STATUS (Episode {episode}):")
                print(f"   - Good experiences: {buffer_stats['good_count']:,}")
                print(f"   - Bad experiences: {buffer_stats['bad_count']:,}")
                print(
                    f"   - Experience diversity: {buffer_stats.get('action_diversity', 0.0):.3f}"
                )
                print(
                    f"   - Sequence quality: {buffer_stats.get('avg_sequence_quality', 0.5):.3f}"
                )

                print(f"\n🏆 ENHANCED WIN/LOSE STATISTICS:")
                print(f"   - Total games: {enhanced_stats['total_games']}")
                print(
                    f"   - Record: {enhanced_stats['wins']}W - {enhanced_stats['losses']}L - {enhanced_stats['draws']}D"
                )
                print(f"   - Overall win rate: {enhanced_stats['win_rate']:.1%}")
                print(
                    f"   - Recent win rate (last 20): {enhanced_stats['recent_win_rate']:.1%}"
                )
                print(f"   - Draw rate: {enhanced_stats['draw_rate']:.1%}")

                print(f"\n⚡ AGGRESSION & SPEED METRICS:")
                print(f"   - Fast win rate: {enhanced_stats['fast_win_rate']:.1%}")
                print(
                    f"   - Timeout strategy rate: {enhanced_stats['timeout_strategy_rate']:.1%}"
                )
                print(
                    f"   - Average combo length: {enhanced_stats['avg_combo_length']:.1f}"
                )
                print(
                    f"   - Average fight duration: {enhanced_stats['avg_speed']:.0f} steps"
                )

                print(f"\n🎯 EXPLORATION & LEARNING:")
                print(
                    f"   - Current exploration rate: {enhanced_stats['exploration_rate']:.1%}"
                )
                print(f"   - Learning rate reboots: {enhanced_stats['reboot_count']}")
                print(f"   - Agent thinking steps: {self.agent.thinking_steps}")

                # Enhanced results pattern
                recent_results_str = "".join(
                    [
                        (
                            "🏆"
                            if r == "WIN"
                            else "💀" if r == "LOSE" else "🤝" if r == "DRAW" else "❓"
                        )
                        for r in enhanced_stats["recent_results"][-20:]
                    ]
                )
                print(f"   - Recent pattern: {recent_results_str}")

                # Termination analysis
                if enhanced_stats["recent_terminations"]:
                    timeout_terms = sum(
                        1
                        for term in enhanced_stats["recent_terminations"]
                        if "timeout" in term
                    )
                    ko_terms = sum(
                        1
                        for term in enhanced_stats["recent_terminations"]
                        if "ko" in term
                    )
                    print(
                        f"   - Recent terminations: {dict(enhanced_stats['recent_terminations'])}"
                    )
                    print(f"   - KO vs Timeout ratio: {ko_terms}:{timeout_terms}")

                print(f"\n🔍 HEALTH DETECTION:")
                print(
                    f"   - Detection rate: {performance_stats.get('health_detection_rate', 0.0):.1%}"
                )
                print(
                    f"   - Last episode health: P{episode_stats.get('player_health', MAX_HEALTH)} vs O{episode_stats.get('opponent_health', MAX_HEALTH)}"
                )

                # Enhanced plateau status
                if len(self.performance_history) >= 10:
                    recent_perf_std = safe_std(
                        list(self.performance_history)[-10:], 0.0
                    )
                    if recent_perf_std < self.lr_reboot_threshold:
                        print(f"\n⚠️  PLATEAU WARNING:")
                        print(
                            f"   - Performance variance: {recent_perf_std:.4f} (threshold: {self.lr_reboot_threshold})"
                        )
                        print(
                            f"   - Episodes since last reboot: {episode - self.last_reboot_episode}"
                        )
                    else:
                        print(f"\n✅ LEARNING ACTIVE:")
                        print(f"   - Performance variance: {recent_perf_std:.4f}")

                # Enhanced success assessment
                timeout_problem = enhanced_stats["timeout_strategy_rate"] > 0.6
                draw_problem = enhanced_stats["draw_rate"] > 0.3
                aggression_problem = enhanced_stats["avg_combo_length"] < 2.0

                print(f"\n🎯 PROBLEM ASSESSMENT:")
                if timeout_problem:
                    print(
                        f"   🚨 TIMEOUT STRATEGY PROBLEM: {enhanced_stats['timeout_strategy_rate']:.1%} of wins are timeouts"
                    )
                else:
                    print(
                        f"   ✅ Timeout strategy under control: {enhanced_stats['timeout_strategy_rate']:.1%}"
                    )

                if draw_problem:
                    print(
                        f"   🚨 DRAW RATE TOO HIGH: {enhanced_stats['draw_rate']:.1%}"
                    )
                else:
                    print(
                        f"   ✅ Draw rate acceptable: {enhanced_stats['draw_rate']:.1%}"
                    )

                if aggression_problem:
                    print(
                        f"   🚨 LOW AGGRESSION: Average combos {enhanced_stats['avg_combo_length']:.1f}"
                    )
                else:
                    print(
                        f"   ✅ Good aggression: Average combos {enhanced_stats['avg_combo_length']:.1f}"
                    )

                # Training progress
                if train_stats:
                    print(f"\n🧠 Training Progress:")
                    print(
                        f"   - Energy separation: {train_stats.get('energy_separation', 0.0):.3f}"
                    )
                    print(
                        f"   - Learning rate: {train_stats.get('learning_rate', 0.0):.2e}"
                    )
                    print(
                        f"   - Exploration bonus: {train_stats.get('exploration_bonus', 0.0):.4f}"
                    )

                # Enhanced success criteria
                overall_success = (
                    enhanced_stats["win_rate"] > 0.3
                    and enhanced_stats["draw_rate"] < 0.3
                    and enhanced_stats["timeout_strategy_rate"] < 0.4
                    and enhanced_stats["avg_combo_length"] > 1.5
                )

                if overall_success:
                    print(f"\n🎉 EXCELLENT PROGRESS:")
                    print(f"   - ✅ Good win rate ({enhanced_stats['win_rate']:.1%})")
                    print(f"   - ✅ Low draw rate ({enhanced_stats['draw_rate']:.1%})")
                    print(f"   - ✅ Aggressive play style")
                    print(f"   - ✅ Avoiding timeout strategies")
                else:
                    print(f"\n🔧 AREAS FOR IMPROVEMENT:")
                    if enhanced_stats["win_rate"] <= 0.3:
                        print(
                            f"   - 📈 Need higher win rate (current: {enhanced_stats['win_rate']:.1%})"
                        )
                    if enhanced_stats["draw_rate"] >= 0.3:
                        print(
                            f"   - 📉 Need lower draw rate (current: {enhanced_stats['draw_rate']:.1%})"
                        )
                    if enhanced_stats["timeout_strategy_rate"] >= 0.4:
                        print(
                            f"   - ⚡ Need more aggressive play (timeout rate: {enhanced_stats['timeout_strategy_rate']:.1%})"
                        )

                # Enhanced logging
                self.logger.info(
                    f"Ep {episode}: "
                    f"WinRate={enhanced_stats['win_rate']:.3f}, "
                    f"DrawRate={enhanced_stats['draw_rate']:.3f}, "
                    f"TimeoutRate={enhanced_stats['timeout_strategy_rate']:.3f}, "
                    f"FastWinRate={enhanced_stats['fast_win_rate']:.3f}, "
                    f"ComboAvg={enhanced_stats['avg_combo_length']:.1f}, "
                    f"Exploration={enhanced_stats['exploration_rate']:.3f}, "
                    f"Reboots={enhanced_stats['reboot_count']}, "
                    f"HealthDetection={performance_stats.get('health_detection_rate', 0.0):.3f}"
                )

            # Save checkpoint
            if (
                self.args.save_frequency > 0
                and episode > 0
                and episode % self.args.save_frequency == 0
            ):
                self.save_checkpoint(episode)

            # Enhanced early stopping
            if len(self.win_rate_history) >= 15:
                recent_win_rate = safe_mean(list(self.win_rate_history)[-8:], 0.0)
                recent_stats = self.get_enhanced_stats()

                success_criteria = (
                    recent_win_rate >= self.args.target_win_rate
                    and recent_stats["draw_rate"] < 0.2
                    and recent_stats["timeout_strategy_rate"] < 0.3
                )

                if success_criteria:
                    print(f"🎯 Enhanced targets achieved!")
                    print(
                        f"   - Win rate: {recent_win_rate:.1%} ≥ {self.args.target_win_rate:.1%}"
                    )
                    print(f"   - Draw rate: {recent_stats['draw_rate']:.1%} < 20%")
                    print(
                        f"   - Timeout strategy rate: {recent_stats['timeout_strategy_rate']:.1%} < 30%"
                    )
                    break

        # Training completed
        self.logger.info("💾 Saving final enhanced model...")
        self.save_checkpoint(self.episode)

        final_performance = self.evaluate_performance()
        final_stats = self.get_enhanced_stats()

        print(f"\n🏁 ENHANCED Training Completed!")
        print(f"   - Total episodes: {self.episode + 1}")
        print(f"   - Final win rate: {final_performance['win_rate']:.1%}")
        print(f"   - Final draw rate: {final_stats['draw_rate']:.1%}")
        print(f"   - Timeout strategy rate: {final_stats['timeout_strategy_rate']:.1%}")
        print(f"   - Fast win rate: {final_stats['fast_win_rate']:.1%}")
        print(f"   - Average combo length: {final_stats['avg_combo_length']:.1f}")
        print(f"   - Learning rate reboots used: {final_stats['reboot_count']}")

        print(f"\n🎯 FINAL ENHANCED ASSESSMENT:")

        # Enhanced success metrics
        timeout_eliminated = final_stats["timeout_strategy_rate"] < 0.3
        draws_controlled = final_stats["draw_rate"] < 0.3
        aggression_achieved = final_stats["avg_combo_length"] > 1.5
        wins_achieved = final_stats["win_rate"] > 0.25

        overall_success = (
            timeout_eliminated
            and draws_controlled
            and aggression_achieved
            and wins_achieved
        )

        print(
            f"   - Timeout strategy eliminated: {'✅' if timeout_eliminated else '❌'} ({final_stats['timeout_strategy_rate']:.1%} < 30%)"
        )
        print(
            f"   - Draw rate controlled: {'✅' if draws_controlled else '❌'} ({final_stats['draw_rate']:.1%} < 30%)"
        )
        print(
            f"   - Aggressive play achieved: {'✅' if aggression_achieved else '❌'} (combos: {final_stats['avg_combo_length']:.1f} > 1.5)"
        )
        print(
            f"   - Win rate achieved: {'✅' if wins_achieved else '❌'} ({final_stats['win_rate']:.1%} > 25%)"
        )

        print(
            f"   - Overall enhanced success: {'🎉 EXCELLENT' if overall_success else '🔧 PARTIAL SUCCESS'}"
        )

        # Show final record with enhanced breakdown
        print(f"\n📊 FINAL ENHANCED RECORD:")
        print(f"   - Total games: {final_stats['total_games']}")
        print(
            f"   - Record: {final_stats['wins']}W - {final_stats['losses']}L - {final_stats['draws']}D"
        )
        print(f"   - Win breakdown:")
        print(
            f"     • Fast wins: {self.fast_wins} ({final_stats['fast_win_rate']:.1%} of wins)"
        )
        print(
            f"     • Timeout wins: {self.timeout_wins} ({final_stats['timeout_strategy_rate']:.1%} of wins)"
        )
        print(f"   - Learning rate reboots: {final_stats['reboot_count']}")
        print(f"   - Final exploration rate: {final_stats['exploration_rate']:.1%}")

        return overall_success


def main():
    """Enhanced main training function."""
    parser = argparse.ArgumentParser(
        description="Enhanced Street Fighter Training - Break Plateaus with Aggressive Exploration"
    )

    # Enhanced environment arguments
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=1000,
        help="Maximum training episodes (increased for plateau breaking)",
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=4000, help="Maximum steps per episode"
    )
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=12,
        help="Evaluate every N episodes (more frequent for plateau detection)",
    )
    parser.add_argument(
        "--save-frequency",
        type=int,
        default=30,
        help="Save checkpoint every N episodes",
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training",
    )

    # Enhanced model arguments
    parser.add_argument(
        "--features-dim",
        type=int,
        default=512,
        help="Feature dimension for enhanced models",
    )
    parser.add_argument(
        "--thinking-steps",
        type=int,
        default=6,
        help="Thinking steps (increased for better optimization)",
    )
    parser.add_argument(
        "--thinking-lr", type=float, default=0.025, help="Thinking learning rate"
    )

    # Enhanced training arguments
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Initial learning rate (higher for aggressive training)",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=28,
        help="Batch size (increased for better diversity)",
    )
    parser.add_argument(
        "--contrastive-margin",
        type=float,
        default=3.0,
        help="Contrastive margin (increased for better separation)",
    )

    # Enhanced buffer arguments
    parser.add_argument(
        "--buffer-capacity",
        type=int,
        default=30000,
        help="Buffer capacity (increased for reservoir sampling)",
    )

    # Enhanced evaluation arguments
    parser.add_argument(
        "--target-win-rate", type=float, default=0.55, help="Target win rate"
    )
    parser.add_argument(
        "--verify-health", action="store_true", help="Verify system at start"
    )
    parser.add_argument(
        "--render", action="store_true", help="Enable rendering during training"
    )

    args = parser.parse_args()

    # Enhanced configuration display
    print(f"🚀 Enhanced Street Fighter Training Configuration:")
    print(f"   Max Episodes: {args.max_episodes:,}")
    print(f"   Learning Rate: {args.learning_rate:.2e} (with reboots)")
    print(f"   Batch Size: {args.batch_size}")
    print(f"   Buffer Capacity: {args.buffer_capacity:,} (reservoir sampling)")
    print(f"   Thinking Steps: {args.thinking_steps}")
    if args.load_checkpoint:
        print(f"   Resuming from: {args.load_checkpoint}")
    print(f"   Target Win Rate: {args.target_win_rate:.1%}")
    print(f"   🎯 PRIMARY GOALS:")
    print(f"     - Break learning plateaus with LR reboots")
    print(f"     - Eliminate timeout strategies with time-decayed rewards")
    print(f"     - Maintain diversity with reservoir sampling")
    print(f"     - Aggressive exploration to discover new strategies")

    # Run enhanced training
    try:
        trainer = EnhancedTrainer(args)
        success = trainer.train()

        if success:
            print(f"\n🎉 MISSION ACCOMPLISHED!")
            print(f"   Learning plateaus have been broken!")
            print(f"   Timeout strategies have been eliminated!")
            print(f"   Aggressive, fast-paced play has been achieved!")
        else:
            print(f"\n🔧 PARTIAL SUCCESS")
            print(
                f"   Significant improvements made, but some goals may need more time"
            )

    except KeyboardInterrupt:
        print(f"\n⚠️  Enhanced training interrupted by user")
    except Exception as e:
        print(f"\n❌ Enhanced training failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
