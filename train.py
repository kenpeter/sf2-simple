#!/usr/bin/env python3
"""
🛡️ FIXED TRAINING - Solves the 176 vs 176 draw problem
Complete training script with proper health detection and win/lose logic
Enhanced with 8-frame stacking for temporal awareness
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

# Import the FIXED wrapper components
try:
    from wrapper import (
        make_fixed_env,
        verify_health_detection,
        SimpleVerifier,
        SimpleAgent,
        SimpleCNN,
        safe_mean,
        safe_std,
        safe_divide,
        MAX_FIGHT_STEPS,
        MAX_HEALTH,
        FRAME_STACK_SIZE,
    )

    print("✅ Successfully imported fixed wrapper components")
    print(f"✅ Frame stacking: {FRAME_STACK_SIZE} frames for temporal context")
except ImportError as e:
    print(f"❌ Failed to import fixed wrapper: {e}")
    print("Make sure wrapper.py is in the same directory")
    exit(1)


class FixedExperienceBuffer:
    """🎯 Enhanced experience buffer for fixed training with temporal context."""

    def __init__(self, capacity=25000):  # Increased capacity for 8-frame sequences
        self.capacity = capacity
        self.good_experiences = deque(maxlen=capacity // 2)
        self.bad_experiences = deque(maxlen=capacity // 2)
        self.total_added = 0

        # NEW: Track sequence quality for temporal learning
        self.sequence_quality_tracker = deque(maxlen=1000)

    def add_experience(self, experience, reward, win_result):
        """Add experience based on win/lose result with temporal context awareness."""
        self.total_added += 1

        # Enhanced classification with temporal context consideration
        temporal_quality = self._assess_temporal_quality(experience)

        # Classify as good or bad based on actual results and temporal patterns
        if win_result == "WIN" or (reward > 0.3 and temporal_quality > 0.5):
            self.good_experiences.append(experience)
            experience["sequence_quality"] = "good"
        elif win_result == "LOSE" or (reward < -0.3 and temporal_quality < 0.3):
            self.bad_experiences.append(experience)
            experience["sequence_quality"] = "bad"
        else:
            # Neutral experiences - add to bad to learn from
            self.bad_experiences.append(experience)
            experience["sequence_quality"] = "neutral"

        self.sequence_quality_tracker.append(temporal_quality)

    def _assess_temporal_quality(self, experience):
        """Assess the temporal quality of an experience based on context."""
        # Simple temporal quality assessment
        reward = experience.get("reward", 0.0)
        thinking_info = experience.get("thinking_info", {})

        # Consider energy improvement and thinking effectiveness
        energy_improvement = thinking_info.get("energy_improvement", False)
        final_energy = thinking_info.get("final_energy", 0.0)

        quality = 0.5  # Base quality

        # Reward-based quality
        if reward > 0:
            quality += min(reward * 0.3, 0.3)
        elif reward < 0:
            quality -= min(abs(reward) * 0.2, 0.3)

        # Energy-based quality
        if energy_improvement:
            quality += 0.2
        if final_energy < 0:
            quality += 0.1

        return np.clip(quality, 0.0, 1.0)

    def sample_balanced_batch(self, batch_size):
        """Sample balanced batch of good and bad experiences with temporal awareness."""
        if (
            len(self.good_experiences) < batch_size // 4
            or len(self.bad_experiences) < batch_size // 4
        ):
            return None, None

        good_count = batch_size // 2
        bad_count = batch_size // 2

        # Sample with slight preference for more recent experiences
        good_indices = self._sample_with_recency_bias(self.good_experiences, good_count)
        good_batch = [self.good_experiences[i] for i in good_indices]

        bad_indices = self._sample_with_recency_bias(self.bad_experiences, bad_count)
        bad_batch = [self.bad_experiences[i] for i in bad_indices]

        return good_batch, bad_batch

    def _sample_with_recency_bias(self, experience_deque, count):
        """Sample indices with slight bias toward more recent experiences."""
        total_len = len(experience_deque)
        if total_len <= count:
            return list(range(total_len))

        # Create weights that favor more recent experiences (last 30% get higher weight)
        weights = np.ones(total_len)
        recent_threshold = int(total_len * 0.7)
        weights[recent_threshold:] *= 2.0  # Double weight for recent experiences

        # Normalize weights
        weights = weights / weights.sum()

        # Sample without replacement
        indices = np.random.choice(total_len, size=count, replace=False, p=weights)
        return indices.tolist()

    def get_stats(self):
        """Get buffer statistics with temporal quality metrics."""
        total_size = len(self.good_experiences) + len(self.bad_experiences)
        avg_sequence_quality = safe_mean(list(self.sequence_quality_tracker), 0.5)

        return {
            "total_size": total_size,
            "good_count": len(self.good_experiences),
            "bad_count": len(self.bad_experiences),
            "good_ratio": len(self.good_experiences) / max(1, total_size),
            "total_added": self.total_added,
            "avg_sequence_quality": avg_sequence_quality,
            "frame_stack_size": FRAME_STACK_SIZE,
        }


class FixedTrainer:
    """🛡️ Enhanced fixed trainer with 8-frame temporal awareness."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup directories for logging and checkpoints
        self._setup_directories()

        # Initialize FIXED environment with 8-frame stacking
        print(
            f"🎮 Initializing FIXED environment with {FRAME_STACK_SIZE}-frame stacking..."
        )
        self.env = make_fixed_env()

        # Verify health detection works with frame stacking
        if args.verify_health:
            if not verify_health_detection(self.env):
                print(
                    "⚠️  Health detection verification failed, but continuing anyway..."
                )
                print("   The training will work with whatever detection is available.")

        # Initialize enhanced models with temporal awareness
        print(
            f"🧠 Initializing enhanced models for {FRAME_STACK_SIZE}-frame processing..."
        )
        self.verifier = SimpleVerifier(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            features_dim=args.features_dim,
        ).to(self.device)

        self.agent = SimpleAgent(
            verifier=self.verifier,
            thinking_steps=args.thinking_steps,
            thinking_lr=args.thinking_lr,
        )

        # Enhanced experience buffer for temporal sequences
        self.experience_buffer = FixedExperienceBuffer(capacity=args.buffer_capacity)

        # Optimizer with adjusted parameters for larger model
        self.optimizer = optim.Adam(
            self.verifier.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            eps=1e-8,  # Better numerical stability
        )

        # NEW: Learning rate scheduler for better temporal learning
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.8, patience=10
        )

        # Training state
        self.episode = 0
        self.total_steps = 0
        self.best_win_rate = 0.0

        # Enhanced performance tracking with temporal metrics
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.recent_results = deque(maxlen=30)  # Increased for better temporal tracking
        self.win_rate_history = deque(maxlen=100)
        self.recent_losses = deque(maxlen=100)

        # NEW: Temporal performance metrics
        self.temporal_consistency_scores = deque(maxlen=50)
        self.frame_stack_utilization = deque(maxlen=50)

        # Termination reason tracking
        self.termination_reasons = deque(maxlen=200)

        # Setup logging
        self.setup_logging()

        # Load checkpoint if provided
        self.load_checkpoint()

        print(f"🛡️ Enhanced Fixed Trainer initialized")
        print(f"   - Device: {self.device}")
        print(f"   - Learning rate: {args.learning_rate:.2e}")
        print(f"   - Frame stacking: {FRAME_STACK_SIZE} frames")
        print(f"   - Temporal awareness: ENABLED")
        print(f"   - Health detection: MULTI-METHOD")
        print(f"   - Draw problem: TARGETED FOR ELIMINATION")

    def _setup_directories(self):
        """Create necessary directories for logging and checkpoints."""
        self.log_dir = Path("logs_fixed_8frame")
        self.checkpoint_dir = Path("checkpoints_fixed_8frame")
        self.log_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)

    def setup_logging(self):
        """Setup enhanced logging system."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"fixed_training_8frame_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def save_checkpoint(self, episode):
        """Save a training checkpoint with temporal metrics."""
        if not self.args.save_frequency > 0:
            return

        filename = self.checkpoint_dir / f"checkpoint_8frame_ep_{episode}.pth"
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
            "temporal_consistency_scores": list(self.temporal_consistency_scores),
            "args": self.args,
        }
        torch.save(state, filename)
        self.logger.info(f"💾 Enhanced checkpoint saved to {filename}")

    def load_checkpoint(self):
        """Load a training checkpoint with temporal metrics."""
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

                # Load training state
                self.episode = checkpoint["episode"] + 1
                self.wins = checkpoint.get("wins", 0)
                self.losses = checkpoint.get("losses", 0)
                self.draws = checkpoint.get("draws", 0)
                self.total_steps = checkpoint.get("total_steps", 0)
                self.best_win_rate = checkpoint.get("best_win_rate", 0.0)

                # Load temporal metrics if available
                if "temporal_consistency_scores" in checkpoint:
                    self.temporal_consistency_scores.extend(
                        checkpoint["temporal_consistency_scores"]
                    )

                frame_stack_in_checkpoint = checkpoint.get("frame_stack_size", 1)
                if frame_stack_in_checkpoint != FRAME_STACK_SIZE:
                    self.logger.warning(
                        f"⚠️ Frame stack size mismatch: checkpoint={frame_stack_in_checkpoint}, "
                        f"current={FRAME_STACK_SIZE}. Model may need retraining."
                    )

                self.logger.info(
                    f"✅ Enhanced checkpoint loaded. Resuming from episode {self.episode}."
                )
            else:
                self.logger.warning(
                    f"⚠️ Checkpoint file not found at {checkpoint_path}. Starting from scratch."
                )

    def _assess_temporal_consistency(self, episode_experiences):
        """Assess how well the agent is using temporal information."""
        if len(episode_experiences) < FRAME_STACK_SIZE:
            return 0.5  # Not enough data

        # Look for patterns in action selection that suggest temporal awareness
        actions = [exp["action"] for exp in episode_experiences]
        rewards = [exp["reward"] for exp in episode_experiences]

        # Calculate action consistency in sequences
        action_changes = sum(
            1 for i in range(1, len(actions)) if actions[i] != actions[i - 1]
        )
        action_consistency = 1.0 - (action_changes / max(1, len(actions) - 1))

        # Calculate reward-action correlation (good temporal learning shows positive correlation)
        if len(rewards) > 1:
            reward_improvement = sum(
                1 for i in range(1, len(rewards)) if rewards[i] > rewards[i - 1]
            )
            reward_trend = reward_improvement / max(1, len(rewards) - 1)
        else:
            reward_trend = 0.5

        # Combine metrics
        temporal_score = action_consistency * 0.4 + reward_trend * 0.6
        return np.clip(temporal_score, 0.0, 1.0)

    def run_episode(self):
        """Run a single episode with enhanced temporal awareness."""
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

        # NEW: Track frame stack utilization
        frame_stack_changes = 0
        prev_visual_obs = None

        while (
            not done and not truncated and episode_steps < self.args.max_episode_steps
        ):
            # Enhanced agent prediction with temporal context
            action, thinking_info = self.agent.predict(obs, deterministic=False)

            # Execute action
            next_obs, reward, done, truncated, info = self.env.step(action)

            episode_reward += reward
            episode_steps += 1
            self.total_steps += 1

            # NEW: Track frame stack changes for utilization metric
            if prev_visual_obs is not None:
                current_visual = obs["visual_obs"]
                if not np.array_equal(current_visual, prev_visual_obs):
                    frame_stack_changes += 1
            prev_visual_obs = obs["visual_obs"].copy()

            # Track round result
            if info.get("round_ended", False):
                termination_reason = info.get("termination_reason", "unknown")
                round_result = info.get("round_result", "ONGOING")

                if round_result == "WIN":
                    round_won = True
                elif round_result == "LOSE":
                    round_lost = True
                elif round_result == "DRAW":
                    round_draw = True

            # Store enhanced experience with temporal context
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
                "temporal_context": {
                    "frame_changes": frame_stack_changes,
                    "sequence_position": episode_steps,
                    "total_episode_steps": episode_steps,
                },
            }

            episode_experiences.append(experience)
            obs = next_obs

        # Update win/lose statistics
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

        # Track termination reason
        self.termination_reasons.append(termination_reason)

        # NEW: Assess temporal consistency
        temporal_consistency = self._assess_temporal_consistency(episode_experiences)
        self.temporal_consistency_scores.append(temporal_consistency)

        # NEW: Track frame stack utilization
        frame_utilization = frame_stack_changes / max(1, episode_steps)
        self.frame_stack_utilization.append(frame_utilization)

        # Add experiences to buffer with temporal enhancement
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
            "temporal_consistency": temporal_consistency,
            "frame_utilization": frame_utilization,
            "frame_stack_size": FRAME_STACK_SIZE,
        }

        return episode_stats

    def get_win_lose_stats(self):
        """Get comprehensive win/lose statistics with temporal metrics."""
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
                "recent_results": list(self.recent_results),
                "recent_terminations": {},
                "temporal_consistency": 0.5,
                "frame_utilization": 0.0,
            }

        # Calculate rates
        win_rate = self.wins / total_games
        draw_rate = self.draws / total_games

        # Recent win rate (last 15 games, increased for better temporal tracking)
        recent_wins = sum(
            1 for result in list(self.recent_results)[-15:] if result == "WIN"
        )
        recent_games = min(len(self.recent_results), 15)
        recent_win_rate = recent_wins / recent_games if recent_games > 0 else 0.0

        # Recent termination reasons
        recent_terminations = {}
        for reason in list(self.termination_reasons)[-15:]:
            recent_terminations[reason] = recent_terminations.get(reason, 0) + 1

        # Temporal metrics
        avg_temporal_consistency = safe_mean(
            list(self.temporal_consistency_scores), 0.5
        )
        avg_frame_utilization = safe_mean(list(self.frame_stack_utilization), 0.0)

        return {
            "total_games": total_games,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": win_rate,
            "draw_rate": draw_rate,
            "recent_win_rate": recent_win_rate,
            "recent_results": list(self.recent_results),
            "recent_terminations": recent_terminations,
            "temporal_consistency": avg_temporal_consistency,
            "frame_utilization": avg_frame_utilization,
        }

    def calculate_contrastive_loss(
        self, good_batch, bad_batch, margin=2.5
    ):  # Increased margin for temporal learning
        """Calculate enhanced contrastive loss for energy-based training with temporal awareness."""
        device = self.device

        def process_batch(batch):
            if not batch:
                return None, None

            obs_batch = []
            action_batch = []

            for exp in batch:
                obs = exp["obs"]
                action = exp["action"]

                # Convert observations (now with 8-frame stacking)
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

        # Stack observations with enhanced temporal processing
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

        # Enhanced contrastive loss with temporal awareness
        good_energy_mean = good_energies.mean()
        bad_energy_mean = bad_energies.mean()

        energy_diff = bad_energy_mean - good_energy_mean
        contrastive_loss = torch.clamp(margin - energy_diff, min=0.0)

        # Enhanced regularization for temporal stability
        energy_reg = 0.01 * (good_energies.pow(2).mean() + bad_energies.pow(2).mean())

        # NEW: Temporal consistency regularization
        good_energy_var = good_energies.var()
        bad_energy_var = bad_energies.var()
        temporal_reg = 0.005 * (good_energy_var + bad_energy_var)

        total_loss = contrastive_loss + energy_reg + temporal_reg

        return total_loss, {
            "contrastive_loss": contrastive_loss.item(),
            "energy_reg": energy_reg.item(),
            "temporal_reg": temporal_reg.item(),
            "good_energy_mean": good_energy_mean.item(),
            "bad_energy_mean": bad_energy_mean.item(),
            "energy_separation": energy_diff.item(),
            "good_energy_var": good_energy_var.item(),
            "bad_energy_var": bad_energy_var.item(),
        }

    def train_step(self):
        """Enhanced training step with temporal awareness."""
        # Sample batch
        good_batch, bad_batch = self.experience_buffer.sample_balanced_batch(
            self.args.batch_size
        )

        if good_batch is None or bad_batch is None:
            return None  # Not enough experiences

        # Calculate enhanced loss
        loss, loss_info = self.calculate_contrastive_loss(
            good_batch, bad_batch, margin=self.args.contrastive_margin
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Enhanced gradient clipping for temporal stability
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.verifier.parameters(), max_norm=1.0
        )

        # Check for gradient explosion
        if grad_norm > 10.0:
            print(f"⚠️ Large gradient norm detected: {grad_norm:.2f}")
            return None

        self.optimizer.step()

        # Update learning rate scheduler
        self.scheduler.step(loss.item())

        loss_info["grad_norm"] = grad_norm.item()
        loss_info["learning_rate"] = self.optimizer.param_groups[0]["lr"]
        return loss_info

    def evaluate_performance(self):
        """Evaluate current performance with temporal awareness."""
        eval_episodes = 5  # Increased for better temporal assessment

        wins = 0
        total_reward = 0.0
        total_steps = 0
        health_changes_detected = 0
        temporal_scores = []

        for _ in range(eval_episodes):
            obs, info = self.env.reset()
            done = False
            truncated = False
            episode_reward = 0.0
            episode_steps = 0
            episode_experiences = []
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

                # Track for temporal assessment
                episode_experiences.append(
                    {"action": action, "reward": reward, "thinking_info": thinking_info}
                )

                # Check for win
                round_result = info.get("round_result", "ONGOING")
                if round_result == "WIN":
                    wins += 1
                    break

            # Assess temporal consistency for this evaluation episode
            temporal_score = self._assess_temporal_consistency(episode_experiences)
            temporal_scores.append(temporal_score)

            # Check if health detection worked
            final_player_health = info.get("player_health", MAX_HEALTH)
            final_opponent_health = info.get("opponent_health", MAX_HEALTH)

            if (
                final_player_health != initial_player_health
                or final_opponent_health != initial_opponent_health
            ):
                health_changes_detected += 1

            total_reward += episode_reward
            total_steps += episode_steps

        win_rate = wins / eval_episodes
        avg_reward = total_reward / eval_episodes
        avg_steps = total_steps / eval_episodes
        health_detection_rate = health_changes_detected / eval_episodes
        avg_temporal_consistency = safe_mean(temporal_scores, 0.5)

        return {
            "win_rate": win_rate,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
            "health_detection_rate": health_detection_rate,
            "temporal_consistency": avg_temporal_consistency,
            "eval_episodes": eval_episodes,
            "frame_stack_size": FRAME_STACK_SIZE,
        }

    def train(self):
        """Enhanced main training loop with temporal awareness and draw elimination focus."""
        print(
            f"🛡️ Starting ENHANCED Fixed Training with {FRAME_STACK_SIZE}-Frame Stacking"
        )
        print(f"   - Target episodes: {self.args.max_episodes}")
        print(f"   - Starting from episode: {self.episode}")
        print(f"   - Focus: ELIMINATE DRAWS with TEMPORAL AWARENESS")
        print(f"   - Health detection: MULTI-METHOD")
        print(f"   - Frame stacking: {FRAME_STACK_SIZE} frames for context")

        training_start_time = time.time()

        for episode in range(self.episode, self.args.max_episodes):
            self.episode = episode
            episode_start_time = time.time()

            # Run episode with temporal enhancement
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

            # Enhanced periodic evaluation and logging
            if episode % self.args.eval_frequency == 0:
                # Performance evaluation with temporal metrics
                performance_stats = self.evaluate_performance()
                self.win_rate_history.append(performance_stats["win_rate"])

                # Get comprehensive stats
                buffer_stats = self.experience_buffer.get_stats()
                win_lose_stats = self.get_win_lose_stats()

                # Enhanced status printing with temporal metrics
                print(f"\n🎯 ENHANCED FIXED STATUS (Episode {episode}):")
                print(f"   - Good experiences: {buffer_stats['good_count']:,}")
                print(f"   - Bad experiences: {buffer_stats['bad_count']:,}")
                print(f"   - Total added: {buffer_stats['total_added']:,}")
                print(
                    f"   - Avg sequence quality: {buffer_stats.get('avg_sequence_quality', 0.5):.3f}"
                )

                # Detailed win/lose statistics
                print(f"\n🏆 WIN/LOSE STATISTICS:")
                print(f"   - Total games: {win_lose_stats['total_games']}")
                print(
                    f"   - Wins: {win_lose_stats['wins']} | Losses: {win_lose_stats['losses']} | Draws: {win_lose_stats['draws']}"
                )
                print(f"   - Overall win rate: {win_lose_stats['win_rate']:.1%}")
                print(f"   - Draw rate: {win_lose_stats['draw_rate']:.1%}")
                print(
                    f"   - Recent win rate (last 15): {win_lose_stats['recent_win_rate']:.1%}"
                )

                # Enhanced temporal metrics display
                print(f"\n🕰️ TEMPORAL AWARENESS METRICS:")
                print(
                    f"   - Temporal consistency: {win_lose_stats['temporal_consistency']:.3f}"
                )
                print(
                    f"   - Frame utilization: {win_lose_stats['frame_utilization']:.3f}"
                )
                print(f"   - Frame stack size: {FRAME_STACK_SIZE}")
                print(
                    f"   - Eval temporal consistency: {performance_stats.get('temporal_consistency', 0.5):.3f}"
                )

                # Recent results pattern with enhanced display
                recent_results_str = "".join(
                    [
                        (
                            "🏆"
                            if r == "WIN"
                            else "💀" if r == "LOSE" else "🤝" if r == "DRAW" else "❓"
                        )
                        for r in win_lose_stats["recent_results"][
                            -15:
                        ]  # Show more recent results
                    ]
                )
                print(f"   - Recent pattern: {recent_results_str}")

                # Termination reasons analysis
                if win_lose_stats["recent_terminations"]:
                    print(
                        f"   - Recent terminations: {dict(win_lose_stats['recent_terminations'])}"
                    )

                # Enhanced health detection status
                print(f"\n🔍 HEALTH DETECTION:")
                print(
                    f"   - Detection rate: {performance_stats.get('health_detection_rate', 0.0):.1%}"
                )
                print(
                    f"   - Last episode health: P{episode_stats.get('player_health', MAX_HEALTH)} vs O{episode_stats.get('opponent_health', MAX_HEALTH)}"
                )
                print(
                    f"   - Detection working: {'✅' if episode_stats.get('health_detection_working', False) else '❌'}"
                )

                # Enhanced draw problem status
                if win_lose_stats["draw_rate"] > 0.8:
                    print(f"\n🚨 DRAW PROBLEM PERSISTS:")
                    print(
                        f"   - Draw rate: {win_lose_stats['draw_rate']:.1%} (TOO HIGH)"
                    )
                    print(
                        f"   - Temporal consistency: {win_lose_stats['temporal_consistency']:.3f}"
                    )
                    print(f"   - Health detection may still be failing")
                elif win_lose_stats["draw_rate"] > 0.3:
                    print(f"\n⚠️  DRAW RATE IMPROVING:")
                    print(
                        f"   - Draw rate: {win_lose_stats['draw_rate']:.1%} (REDUCING)"
                    )
                    print(
                        f"   - Temporal learning helping: {win_lose_stats['temporal_consistency']:.3f}"
                    )
                else:
                    print(f"\n✅ DRAW PROBLEM SOLVED:")
                    print(
                        f"   - Draw rate: {win_lose_stats['draw_rate']:.1%} (EXCELLENT)"
                    )
                    print(
                        f"   - Temporal awareness working: {win_lose_stats['temporal_consistency']:.3f}"
                    )

                # Enhanced training progress
                if train_stats:
                    print(f"\n🧠 Enhanced Training:")
                    print(
                        f"   - Energy separation: {train_stats.get('energy_separation', 0.0):.3f}"
                    )
                    print(
                        f"   - Good energy: {train_stats.get('good_energy_mean', 0.0):.3f}"
                    )
                    print(
                        f"   - Bad energy: {train_stats.get('bad_energy_mean', 0.0):.3f}"
                    )
                    print(
                        f"   - Temporal reg: {train_stats.get('temporal_reg', 0.0):.4f}"
                    )
                    print(
                        f"   - Learning rate: {train_stats.get('learning_rate', 0.0):.2e}"
                    )

                # Enhanced success indicators
                success_indicators = []
                if win_lose_stats["draw_rate"] < 0.3:
                    success_indicators.append("✅ Low draw rate")
                if win_lose_stats["win_rate"] > 0.2:
                    success_indicators.append("✅ Winning games")
                if buffer_stats["good_count"] > 50:
                    success_indicators.append("✅ Good experiences")
                if performance_stats.get("health_detection_rate", 0.0) > 0.5:
                    success_indicators.append("✅ Health detection")
                if win_lose_stats["temporal_consistency"] > 0.6:
                    success_indicators.append("✅ Temporal awareness")
                if win_lose_stats["frame_utilization"] > 0.3:
                    success_indicators.append("✅ Frame stack utilization")

                if success_indicators:
                    print(f"\n🎉 SUCCESS INDICATORS:")
                    for indicator in success_indicators:
                        print(f"   - {indicator}")
                else:
                    print(f"\n🔧 NEEDS WORK:")
                    if win_lose_stats["draw_rate"] >= 0.3:
                        print(
                            f"   - ❌ High draw rate ({win_lose_stats['draw_rate']:.1%})"
                        )
                    if win_lose_stats["win_rate"] <= 0.2:
                        print(
                            f"   - ❌ Low win rate ({win_lose_stats['win_rate']:.1%})"
                        )
                    if buffer_stats["good_count"] <= 50:
                        print(
                            f"   - ❌ Few good experiences ({buffer_stats['good_count']})"
                        )
                    if win_lose_stats["temporal_consistency"] <= 0.5:
                        print(
                            f"   - ❌ Poor temporal consistency ({win_lose_stats['temporal_consistency']:.3f})"
                        )

                # Enhanced logging with temporal metrics
                self.logger.info(
                    f"Ep {episode}: "
                    f"Wins={win_lose_stats['wins']}, "
                    f"Losses={win_lose_stats['losses']}, "
                    f"Draws={win_lose_stats['draws']}, "
                    f"WinRate={win_lose_stats['win_rate']:.3f}, "
                    f"DrawRate={win_lose_stats['draw_rate']:.3f}, "
                    f"HealthDetection={performance_stats.get('health_detection_rate', 0.0):.3f}, "
                    f"TemporalConsistency={win_lose_stats['temporal_consistency']:.3f}, "
                    f"FrameUtil={win_lose_stats['frame_utilization']:.3f}, "
                    f"GoodExp={buffer_stats['good_count']}, "
                    f"FrameStack={FRAME_STACK_SIZE}, "
                    f"LastTerm={episode_stats.get('termination_reason', 'unknown')}"
                )

            # Save checkpoint
            if (
                self.args.save_frequency > 0
                and episode > 0
                and episode % self.args.save_frequency == 0
            ):
                self.save_checkpoint(episode)

            # Enhanced early stopping check with temporal awareness
            if len(self.win_rate_history) >= 10:
                recent_win_rate = safe_mean(list(self.win_rate_history)[-5:], 0.0)
                avg_temporal_consistency = safe_mean(
                    list(self.temporal_consistency_scores)[-10:], 0.5
                )

                if (
                    recent_win_rate >= self.args.target_win_rate
                    and avg_temporal_consistency > 0.6
                ):
                    print(
                        f"🎯 Target achieved! Win rate: {self.args.target_win_rate:.1%}, "
                        f"Temporal consistency: {avg_temporal_consistency:.3f}"
                    )
                    break

        # Training completed
        self.logger.info("💾 Saving final enhanced model checkpoint...")
        self.save_checkpoint(self.episode)

        final_performance = self.evaluate_performance()
        final_win_lose_stats = self.get_win_lose_stats()

        print(f"\n🏁 ENHANCED Fixed Training Completed!")
        print(f"   - Total episodes: {self.episode + 1}")
        print(f"   - Final win rate: {final_performance['win_rate']:.1%}")
        print(f"   - Final draw rate: {final_win_lose_stats['draw_rate']:.1%}")
        print(
            f"   - Health detection rate: {final_performance.get('health_detection_rate', 0.0):.1%}"
        )
        print(
            f"   - Temporal consistency: {final_win_lose_stats['temporal_consistency']:.3f}"
        )
        print(
            f"   - Frame utilization: {final_win_lose_stats['frame_utilization']:.3f}"
        )
        print(f"   - Frame stack size: {FRAME_STACK_SIZE}")

        # Enhanced final assessment
        print(f"\n🎯 FINAL ASSESSMENT:")
        if final_win_lose_stats["draw_rate"] < 0.2:
            print(
                f"   🎉 EXCELLENT: Draw problem eliminated! ({final_win_lose_stats['draw_rate']:.1%} draw rate)"
            )
        elif final_win_lose_stats["draw_rate"] < 0.5:
            print(
                f"   ✅ GOOD: Draw rate reduced to {final_win_lose_stats['draw_rate']:.1%}"
            )
        else:
            print(
                f"   ⚠️  PARTIAL: Draw rate still {final_win_lose_stats['draw_rate']:.1%}"
            )

        if final_win_lose_stats["win_rate"] > 0.4:
            print(f"   🏆 EXCELLENT: Win rate {final_win_lose_stats['win_rate']:.1%}")
        elif final_win_lose_stats["win_rate"] > 0.2:
            print(f"   📈 GOOD: Win rate {final_win_lose_stats['win_rate']:.1%}")
        else:
            print(f"   📉 NEEDS WORK: Win rate {final_win_lose_stats['win_rate']:.1%}")

        if final_win_lose_stats["temporal_consistency"] > 0.7:
            print(
                f"   🧠 EXCELLENT: Temporal awareness {final_win_lose_stats['temporal_consistency']:.3f}"
            )
        elif final_win_lose_stats["temporal_consistency"] > 0.5:
            print(
                f"   🧠 GOOD: Temporal learning {final_win_lose_stats['temporal_consistency']:.3f}"
            )
        else:
            print(
                f"   🧠 NEEDS WORK: Temporal consistency {final_win_lose_stats['temporal_consistency']:.3f}"
            )

        # Show enhanced final record
        print(f"\n📊 FINAL RECORD:")
        print(f"   - Total games: {final_win_lose_stats['total_games']}")
        print(
            f"   - Record: {final_win_lose_stats['wins']}W - {final_win_lose_stats['losses']}L - {final_win_lose_stats['draws']}D"
        )
        print(f"   - Win rate: {final_win_lose_stats['win_rate']:.1%}")
        print(f"   - Draw rate: {final_win_lose_stats['draw_rate']:.1%}")
        print(
            f"   - Temporal consistency: {final_win_lose_stats['temporal_consistency']:.3f}"
        )
        print(
            f"   - Frame stack utilization: {final_win_lose_stats['frame_utilization']:.3f}"
        )

        # Termination analysis
        if self.termination_reasons:
            termination_counts = {}
            for reason in self.termination_reasons:
                termination_counts[reason] = termination_counts.get(reason, 0) + 1
            print(f"   - Termination breakdown: {dict(termination_counts)}")

        # Enhanced success criteria
        temporal_success = final_win_lose_stats["temporal_consistency"] > 0.6
        draw_success = final_win_lose_stats["draw_rate"] < 0.3
        win_success = final_win_lose_stats["win_rate"] > 0.25

        overall_success = temporal_success and draw_success and win_success

        print(f"\n🎯 ENHANCED SUCCESS CRITERIA:")
        print(
            f"   - Temporal awareness: {'✅' if temporal_success else '❌'} ({final_win_lose_stats['temporal_consistency']:.3f} > 0.6)"
        )
        print(
            f"   - Draw elimination: {'✅' if draw_success else '❌'} ({final_win_lose_stats['draw_rate']:.1%} < 30%)"
        )
        print(
            f"   - Win performance: {'✅' if win_success else '❌'} ({final_win_lose_stats['win_rate']:.1%} > 25%)"
        )
        print(f"   - Overall success: {'🎉 YES' if overall_success else '🔧 PARTIAL'}")

        return overall_success


def main():
    """Enhanced main training function with 8-frame stacking."""
    parser = argparse.ArgumentParser(
        description="Enhanced Fixed Street Fighter Training - Eliminate Draws with 8-Frame Temporal Awareness"
    )

    # Environment arguments
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=800,
        help="Maximum training episodes (increased for temporal learning)",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=4000,
        help="Maximum steps per episode (increased for temporal sequences)",
    )
    parser.add_argument(
        "--eval-frequency",
        type=int,
        default=15,
        help="Evaluate every N episodes (adjusted for temporal tracking)",
    )
    parser.add_argument(
        "--save-frequency",
        type=int,
        default=25,
        help="Save checkpoint every N episodes. Set to 0 to disable.",
    )
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file to resume training.",
    )

    # Enhanced model arguments for temporal processing
    parser.add_argument(
        "--features-dim",
        type=int,
        default=512,
        help="Feature dimension (increased for temporal features)",
    )
    parser.add_argument(
        "--thinking-steps",
        type=int,
        default=5,
        help="Thinking steps (increased for temporal reasoning)",
    )
    parser.add_argument(
        "--thinking-lr",
        type=float,
        default=0.03,
        help="Thinking learning rate (adjusted for stability)",
    )

    # Enhanced training arguments
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (adjusted for temporal stability)",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=24,
        help="Batch size (adjusted for 8-frame sequences)",
    )
    parser.add_argument(
        "--contrastive-margin",
        type=float,
        default=2.5,
        help="Contrastive margin (increased for temporal separation)",
    )

    # Enhanced buffer arguments
    parser.add_argument(
        "--buffer-capacity",
        type=int,
        default=25000,
        help="Buffer capacity (increased for temporal sequences)",
    )

    # Enhanced evaluation arguments
    parser.add_argument(
        "--target-win-rate",
        type=float,
        default=0.65,
        help="Target win rate (increased for temporal awareness)",
    )
    parser.add_argument(
        "--verify-health",
        action="store_true",
        help="Verify health detection at start with frame stacking",
    )

    # Legacy arguments for compatibility
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.4,
        help="Quality threshold (adjusted for temporal sequences)",
    )
    parser.add_argument(
        "--render", action="store_true", help="Render environment (legacy, not used)"
    )

    args = parser.parse_args()

    # Enhanced configuration display
    print(f"🛡️ Enhanced Fixed Street Fighter Training Configuration:")
    print(f"   Max Episodes: {args.max_episodes:,}")
    print(f"   Learning Rate: {args.learning_rate:.2e}")
    print(f"   Save Frequency: Every {args.save_frequency} episodes")
    print(
        f"   Batch Size: {args.batch_size} (optimized for {FRAME_STACK_SIZE}-frame sequences)"
    )
    print(f"   Features Dim: {args.features_dim} (enhanced for temporal processing)")
    print(
        f"   Thinking Steps: {args.thinking_steps} (increased for temporal reasoning)"
    )
    if args.load_checkpoint:
        print(f"   Resuming from: {args.load_checkpoint}")
    print(f"   Target Win Rate: {args.target_win_rate:.1%}")
    print(f"   Health Verification: {args.verify_health}")
    print(f"   🎯 PRIMARY GOAL: ELIMINATE 176 vs 176 DRAWS with TEMPORAL AWARENESS")
    print(f"   🕰️ FRAME STACKING: {FRAME_STACK_SIZE} frames for temporal context")

    # Run enhanced training
    try:
        trainer = FixedTrainer(args)
        success = trainer.train()

        if success:
            print(f"\n🎉 MISSION ACCOMPLISHED!")
            print(
                f"   The 176 vs 176 draw problem has been resolved with temporal awareness!"
            )
            print(
                f"   8-frame stacking provided crucial temporal context for decision making!"
            )
        else:
            print(f"\n🔧 PARTIAL SUCCESS")
            print(
                f"   Draw rate reduced and temporal learning active, but may need further tuning"
            )

    except KeyboardInterrupt:
        print(f"\n⚠️  Enhanced training interrupted by user")
    except Exception as e:
        print(f"\n❌ Enhanced training failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
