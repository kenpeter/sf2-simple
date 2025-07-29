#!/usr/bin/env python3
"""
🚀 ENHANCED TRAINING - RGB Version with Transformer Context Sequence (FIXED)
Key Fixes:
1. Fixed gradient flow issue in train_step
2. Fixed win/loss classification logic
3. Improved health detection reliability
4. Better termination reason tracking
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
import torch.nn.functional as F

try:
    from wrapper import (
        make_enhanced_env,
        verify_health_detection,
        SimpleVerifier,
        ContextTransformer,
        AggressiveAgent,
        SimpleCNN,
        safe_mean,
        safe_std,
        safe_divide,
        MAX_FIGHT_STEPS,
        MAX_HEALTH,
        FRAME_STACK_SIZE,
        SCREEN_WIDTH,
        SCREEN_HEIGHT,
        CONTEXT_SEQUENCE_DIM,
    )

    print("✅ Successfully imported enhanced RGB wrapper components with Transformer")
    print(f"✅ RGB processing with {SCREEN_WIDTH}x{SCREEN_HEIGHT} images: ACTIVE")
    print(f"✅ Transformer context sequence: ACTIVE")
except ImportError as e:
    print(f"❌ Failed to import enhanced wrapper: {e}")
    print("Make sure wrapper.py is in the same directory")
    exit(1)


class ReservoirExperienceBuffer:
    def __init__(self, capacity=30000):
        self.capacity = capacity
        self.good_experiences = []
        self.bad_experiences = []
        self.total_added = 0
        self.good_reservoir_size = capacity // 2
        self.bad_reservoir_size = capacity // 2
        self.sequence_quality_tracker = deque(maxlen=1000)
        self.diversity_scores = deque(maxlen=500)

    def add_experience(self, experience, reward, win_result):
        self.total_added += 1
        temporal_quality = self._assess_temporal_quality(experience)
        diversity_score = self._assess_diversity(experience)
        self.sequence_quality_tracker.append(temporal_quality)
        self.diversity_scores.append(diversity_score)
        is_good_experience = (
            win_result == "WIN"
            or (reward > 0.2 and temporal_quality > 0.4)
            or (diversity_score > 0.7 and temporal_quality > 0.3)
        )
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
        if len(reservoir) < max_size:
            reservoir.append(experience)
        else:
            random_index = random.randint(0, len(reservoir) - 1)
            reservoir[random_index] = experience

    def _assess_temporal_quality(self, experience):
        reward = experience.get("reward", 0.0)
        thinking_info = experience.get("thinking_info", {})
        is_exploration = thinking_info.get("exploration", False)
        energy_improvement = thinking_info.get("energy_improvement", False)
        final_energy = thinking_info.get("final_energy", 0.0)
        quality = 0.5
        if reward > 0:
            quality += min(reward * 0.4, 0.4)
        elif reward < 0:
            quality -= min(abs(reward) * 0.3, 0.3)
        if energy_improvement:
            quality += 0.2
        if final_energy < 0:
            quality += 0.1
        if is_exploration:
            quality += 0.1
        return np.clip(quality, 0.0, 1.0)

    def _assess_diversity(self, experience):
        action = experience.get("action", 0)
        reward = experience.get("reward", 0.0)
        thinking_info = experience.get("thinking_info", {})
        diversity = 0.5
        if hasattr(self, "action_counts"):
            total_actions = sum(self.action_counts.values())
            action_frequency = self.action_counts.get(action, 1) / max(total_actions, 1)
            diversity += (1.0 - action_frequency) * 0.3
        else:
            self.action_counts = {}
        self.action_counts[action] = self.action_counts.get(action, 0) + 1
        if thinking_info.get("exploration", False):
            diversity += 0.2
        if abs(reward) > 1.0:
            diversity += 0.1
        return np.clip(diversity, 0.0, 1.0)

    def sample_balanced_batch(self, batch_size):
        if (
            len(self.good_experiences) < batch_size // 4
            or len(self.bad_experiences) < batch_size // 4
        ):
            return None, None
        good_count = batch_size // 2
        bad_count = batch_size // 2
        good_batch = self._sample_with_diversity_bias(self.good_experiences, good_count)
        bad_batch = self._sample_with_diversity_bias(self.bad_experiences, bad_count)
        return good_batch, bad_batch

    def _sample_with_diversity_bias(self, experience_list, count):
        if len(experience_list) <= count:
            return experience_list[:]
        weights = []
        for exp in experience_list:
            action = exp.get("action", 0)
            thinking_info = exp.get("thinking_info", {})
            weight = 1.0
            if hasattr(self, "action_counts"):
                total_actions = sum(self.action_counts.values())
                action_frequency = self.action_counts.get(action, 1) / max(
                    total_actions, 1
                )
                weight *= 2.0 - action_frequency
            if thinking_info.get("exploration", False):
                weight *= 1.5
            weights.append(weight)
        weights = np.array(weights)
        weights = weights / weights.sum()
        indices = np.random.choice(
            len(experience_list), size=count, replace=False, p=weights
        )
        return [experience_list[i] for i in indices]

    def get_stats(self):
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
            "image_format": "RGB",
            "image_size": f"{SCREEN_WIDTH}x{SCREEN_HEIGHT}",
        }


class EnhancedTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_directories()
        print(f"🚀 Initializing ENHANCED RGB environment with Transformer...")
        self.env = make_enhanced_env()
        if args.verify_health:
            if not verify_health_detection(self.env):
                print("⚠️ System verification failed, but continuing anyway...")
        obs, _ = self.env.reset()
        visual_shape = obs["visual_obs"].shape
        print(f"🎨 RGB Visual format verified: {visual_shape}")
        print(
            f"   - Expected: {3 * FRAME_STACK_SIZE} channels (RGB * {FRAME_STACK_SIZE} frames)"
        )
        print(f"   - Image size: {SCREEN_WIDTH} x {SCREEN_HEIGHT}")
        print(f"🧠 Initializing enhanced RGB models with Transformer...")
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
        self.experience_buffer = ReservoirExperienceBuffer(
            capacity=args.buffer_capacity
        )
        self.initial_learning_rate = args.learning_rate
        self.optimizer = optim.Adam(
            self.verifier.parameters(),
            lr=self.initial_learning_rate,
            weight_decay=args.weight_decay,
            eps=1e-8,
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.7, patience=8
        )
        self.lr_reboot_threshold = 0.02
        self.performance_history = deque(maxlen=20)
        self.last_reboot_episode = 0
        self.reboot_count = 0
        self.episode = 0
        self.total_steps = 0
        self.best_win_rate = 0.0
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.recent_results = deque(maxlen=40)
        self.win_rate_history = deque(maxlen=100)
        self.recent_rewards = deque(maxlen=100)  # Fixed: was recent_losses
        self.timeout_wins = 0
        self.fast_wins = 0
        self.combo_count_history = deque(maxlen=50)
        self.speed_history = deque(maxlen=50)
        self.setup_logging()
        self.load_checkpoint()
        print(f"🚀 Enhanced RGB Trainer initialized with Transformer")
        print(f"   - Device: {self.device}")
        print(f"   - Learning rate: {args.learning_rate:.2e} (with reboots)")
        print(f"   - Weight decay: {args.weight_decay:.2e}")
        print(f"   - Aggressive exploration: {self.agent.epsilon:.1%}")
        print(f"   - Reservoir sampling: ENABLED")
        print(f"   - Plateau detection: ACTIVE")
        print(f"   - Transformer context sequence: ENABLED")
        print(f"   - RGB processing: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")

    def _setup_directories(self):
        self.log_dir = Path("logs")
        self.checkpoint_dir = Path("checkpoints")
        self.log_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)

    def setup_logging(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"enhanced_rgb_training_{timestamp}.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def detect_learning_plateau(self):
        if len(self.performance_history) < 15:
            return False
        recent_performance = list(self.performance_history)[-10:]
        performance_std = safe_std(recent_performance, 0.0)
        is_stagnant = performance_std < self.lr_reboot_threshold
        early_performance = safe_mean(recent_performance[:5], 0.0)
        late_performance = safe_mean(recent_performance[-5:], 0.0)
        no_improvement = late_performance <= early_performance + 0.01
        should_reboot = (
            is_stagnant
            and no_improvement
            and self.episode - self.last_reboot_episode > 50
        )
        if should_reboot:
            self.logger.info(
                f"🔄 Plateau detected: std={performance_std:.4f}, "
                f"improvement={late_performance-early_performance:.4f}"
            )
        return should_reboot

    def reboot_learning_rate(self):
        self.reboot_count += 1
        self.last_reboot_episode = self.episode
        new_lr = self.initial_learning_rate * (1.2**self.reboot_count)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.7, patience=8
        )
        self.agent.epsilon = min(0.4, self.agent.epsilon * 2.0)
        self.logger.info(f"🚀 LEARNING RATE REBOOT #{self.reboot_count}")
        self.logger.info(f"   - New LR: {new_lr:.2e}")
        self.logger.info(f"   - Boosted exploration: {self.agent.epsilon:.1%}")
        print(f"🚀 LEARNING RATE REBOOT #{self.reboot_count}!")
        print(f"   - Learning rate reset to: {new_lr:.2e}")
        print(f"   - Exploration boosted to: {self.agent.epsilon:.1%}")
        print(f"   - Breaking out of plateau!")

    def save_checkpoint(self, episode):
        if not self.args.save_frequency > 0:
            return
        filename = self.checkpoint_dir / f"enhanced_rgb_checkpoint_ep_{episode}.pth"
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
            "image_format": "RGB",
            "screen_width": SCREEN_WIDTH,
            "screen_height": SCREEN_HEIGHT,
            "args": self.args,
        }
        torch.save(state, filename)
        self.logger.info(f"💾 Enhanced RGB checkpoint saved to {filename}")

    def load_checkpoint(self):
        if self.args.load_checkpoint:
            checkpoint_path = self.args.load_checkpoint
            if os.path.exists(checkpoint_path):
                self.logger.info(
                    f"🔄 Loading enhanced RGB checkpoint from {checkpoint_path}..."
                )
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.verifier.load_state_dict(checkpoint["verifier_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if "scheduler_state_dict" in checkpoint:
                    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
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
                if "agent_epsilon" in checkpoint:
                    self.agent.epsilon = checkpoint["agent_epsilon"]
                checkpoint_format = checkpoint.get("image_format", "unknown")
                checkpoint_width = checkpoint.get("screen_width", "unknown")
                checkpoint_height = checkpoint.get("screen_height", "unknown")
                print(f"   📸 Checkpoint image format: {checkpoint_format}")
                print(
                    f"   📏 Checkpoint resolution: {checkpoint_width}x{checkpoint_height}"
                )
                print(f"   📏 Current resolution: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
                self.logger.info(
                    f"✅ Enhanced RGB checkpoint loaded. Resuming from episode {self.episode}."
                )
            else:
                self.logger.warning(f"⚠️ Checkpoint file not found. Starting fresh.")

    def run_episode(self):
        obs, info = self.env.reset()
        done = False
        truncated = False
        episode_reward = 0.0
        episode_steps = 0
        episode_experiences = []
        round_won = False
        round_lost = False
        round_draw = False
        max_combo_length = 0
        total_damage_dealt = 0.0
        is_fast_win = False

        # Track health for better win detection
        initial_player_health = info.get("agent_hp", MAX_HEALTH)
        initial_opponent_health = info.get("enemy_hp", MAX_HEALTH)

        while (
            not done and not truncated and episode_steps < self.args.max_episode_steps
        ):
            action, thinking_info = self.agent.predict(obs, deterministic=False)
            next_obs, reward, done, truncated, info = self.env.step(action)
            episode_reward += reward
            episode_steps += 1
            self.total_steps += 1

            final_player_health = info.get("agent_hp", 0)
            final_opponent_health = info.get("enemy_hp", 0)

            # FIXED: Better win/loss detection logic
            if final_player_health > 0 and final_opponent_health <= 0:
                # Player won - opponent is defeated
                round_won = True
                if episode_steps < MAX_FIGHT_STEPS * 0.5:
                    is_fast_win = True
                    self.fast_wins += 1
            elif final_player_health <= 0 and final_opponent_health > 0:
                # Player lost - player is defeated
                round_lost = True
            else:
                # Draw case
                round_draw = True

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
                "image_format": "RGB",
                "image_size": f"{SCREEN_WIDTH}x{SCREEN_HEIGHT}",
                "enhanced_context": {
                    "is_exploration": thinking_info.get("exploration", False),
                    "episode_progress": episode_steps / MAX_FIGHT_STEPS,
                },
            }
            episode_experiences.append(experience)
            obs = next_obs

        # FIXED: Proper result classification
        if round_won:
            self.wins += 1
            self.recent_results.append("WIN")
            win_result = "WIN"
        elif round_lost:
            self.losses += 1
            self.recent_results.append("LOSE")
            win_result = "LOSE"
        else:  # round_draw or fallback
            self.draws += 1
            self.recent_results.append("DRAW")
            win_result = "DRAW"

        # Add experiences to buffer
        for experience in episode_experiences:
            self.experience_buffer.add_experience(
                experience, episode_reward, win_result
            )

        # Update tracking
        self.performance_history.append(episode_reward / max(1, episode_steps))
        self.speed_history.append(episode_steps / MAX_FIGHT_STEPS)

        total_matches = self.wins + self.losses + self.draws
        win_rate = safe_divide(self.wins, total_matches, 0.0)
        self.win_rate_history.append(win_rate)
        self.recent_rewards.append(episode_reward)

        return {
            "episode_reward": episode_reward,
            "episode_steps": episode_steps,
            "win_result": win_result,
            "total_damage_dealt": total_damage_dealt,
            "is_fast_win": is_fast_win,
        }

    def train_step(self):
        good_batch, bad_batch = self.experience_buffer.sample_balanced_batch(
            self.args.batch_size
        )
        if good_batch is None or bad_batch is None:
            return None

        batch = good_batch + bad_batch
        random.shuffle(batch)

        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []

        for exp in batch:
            observations.append(exp["obs"])
            actions.append(exp["action"])
            rewards.append(exp["reward"])
            next_observations.append(exp["next_obs"])
            dones.append(exp["done"])

        device = self.device

        visual_obs = torch.tensor(
            np.stack([obs["visual_obs"] for obs in observations]),
            dtype=torch.float32,
            device=device,
        )
        vector_obs = torch.tensor(
            np.stack([obs["vector_obs"] for obs in observations]),
            dtype=torch.float32,
            device=device,
        )
        context_sequences = torch.tensor(
            np.stack([obs["context_sequence"] for obs in observations]),
            dtype=torch.float32,
            device=device,
        )

        actions = torch.tensor(actions, dtype=torch.long, device=device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

        next_visual_obs = torch.tensor(
            np.stack([obs["visual_obs"] for obs in next_observations]),
            dtype=torch.float32,
            device=device,
        )
        next_vector_obs = torch.tensor(
            np.stack([obs["vector_obs"] for obs in next_observations]),
            dtype=torch.float32,
            device=device,
        )
        next_context_sequences = torch.tensor(
            np.stack([obs["context_sequence"] for obs in next_observations]),
            dtype=torch.float32,
            device=device,
        )

        dones = torch.tensor(dones, dtype=torch.float32, device=device)

        batch_size = visual_obs.shape[0]
        action_one_hot = torch.zeros(batch_size, self.env.action_space.n, device=device)
        action_one_hot.scatter_(1, actions.unsqueeze(1), 1.0)

        current_obs = {
            "visual_obs": visual_obs,
            "vector_obs": vector_obs,
            "context_sequence": context_sequences,
        }
        next_obs = {
            "visual_obs": next_visual_obs,
            "vector_obs": next_vector_obs,
            "context_sequence": next_context_sequences,
        }

        self.optimizer.zero_grad()

        # Current energy
        current_energy = self.verifier(current_obs, action_one_hot)

        # FIXED: Target energy calculation with proper gradient handling
        # Step 1: Initialize candidate action outside no_grad context
        next_candidate_action = (
            torch.randn(batch_size, self.env.action_space.n, device=device) * 0.01
        )

        # Step 2: Optimize the candidate action (this needs gradients)
        best_next_candidate = next_candidate_action.clone()
        best_energy = float("inf")

        for step in range(self.args.thinking_steps):
            # Create a variable that requires gradients
            next_action_var = next_candidate_action.clone().requires_grad_(True)
            next_action_probs = F.softmax(next_action_var, dim=-1)

            # Calculate energy (this creates the computation graph)
            energy_val = self.verifier(next_obs, next_action_probs)

            # Track best solution
            with torch.no_grad():
                current_energy_val = energy_val.mean().item()
                if current_energy_val < best_energy:
                    best_energy = current_energy_val
                    best_next_candidate = next_action_var.clone().detach()

            # Calculate gradients (now the computation graph exists)
            grad = torch.autograd.grad(
                outputs=energy_val.sum(),
                inputs=next_action_var,
                create_graph=False,
                retain_graph=False,
            )[0]

            # Update candidate action (no grad needed here)
            with torch.no_grad():
                step_size = self.args.thinking_lr * (0.85**step)
                next_candidate_action = next_candidate_action - step_size * grad

        # Step 3: Calculate final target energy with no_grad
        with torch.no_grad():
            best_next_action_probs = F.softmax(best_next_candidate, dim=-1)
            next_energy = self.verifier(next_obs, best_next_action_probs)

            # Calculate target energy
            target_energy = (
                rewards.unsqueeze(-1)
                + (1 - dones.unsqueeze(-1)) * self.args.gamma * next_energy
            )

        # Main loss
        loss = nn.functional.mse_loss(current_energy, target_energy)

        # Contrastive loss
        contrastive_loss = 0.0
        for _ in range(3):
            negative_actions = torch.randint(
                0, self.env.action_space.n, (batch_size,), device=device
            )
            negative_one_hot = torch.zeros(
                batch_size, self.env.action_space.n, device=device
            )
            negative_one_hot.scatter_(1, negative_actions.unsqueeze(1), 1.0)
            negative_energy = self.verifier(current_obs, negative_one_hot)
            contrastive_loss += torch.mean(
                nn.functional.relu(
                    current_energy - negative_energy + self.args.contrastive_margin
                )
            )

        total_loss = loss + self.args.contrastive_weight * contrastive_loss

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.verifier.parameters(), self.args.max_grad_norm
        )
        self.optimizer.step()
        self.scheduler.step(total_loss.item())

        return {
            "loss": total_loss.item(),
            "energy_loss": loss.item(),
            "contrastive_loss": contrastive_loss.item(),
        }

    def train(self):
        print(f"🎮 Starting ENHANCED RGB training with Transformer...")
        print(f"   - Total episodes: {self.args.num_episodes}")
        print(f"   - Batch size: {self.args.batch_size}")
        print(f"   - Initial learning rate: {self.args.learning_rate:.2e}")
        print(f"   - Weight decay: {self.args.weight_decay:.2e}")
        print(f"   - Initial exploration: {self.agent.epsilon:.1%}")
        print(f"   - Frame stacking: {FRAME_STACK_SIZE} frames")
        print(f"   - Image format: RGB ({SCREEN_WIDTH}x{SCREEN_HEIGHT})")
        print(f"   - Transformer context sequence: ENABLED")

        start_time = time.time()

        for episode in range(self.episode, self.args.num_episodes):
            self.episode = episode
            episode_info = self.run_episode()

            episode_reward = episode_info["episode_reward"]
            episode_steps = episode_info["episode_steps"]
            win_result = episode_info["win_result"]

            total_matches = self.wins + self.losses + self.draws
            win_rate = safe_divide(self.wins, total_matches, 0.0)

            if win_rate > self.best_win_rate:
                self.best_win_rate = win_rate
                self.save_checkpoint(episode)

            # Training step
            if (
                episode % self.args.train_frequency == 0
                and self.experience_buffer.total_added >= self.args.batch_size
            ):
                train_info = self.train_step()
                if train_info:
                    self.logger.info(
                        f"Episode {episode}: Loss={train_info['loss']:.4f}, "
                        f"EnergyLoss={train_info['energy_loss']:.4f}, "
                        f"ContrastiveLoss={train_info['contrastive_loss']:.4f}"
                    )

            # Logging
            if episode % self.args.log_frequency == 0:
                buffer_stats = self.experience_buffer.get_stats()
                agent_stats = self.agent.get_thinking_stats()
                avg_reward = safe_mean(list(self.recent_rewards), 0.0)
                avg_speed = safe_mean(list(self.speed_history), 1.0)
                win_rate = safe_mean(list(self.win_rate_history), 0.0)
                current_lr = self.optimizer.param_groups[0]["lr"]
                elapsed_time = (time.time() - start_time) / 3600

                print(f"\n📊 Episode {episode} Summary (RGB with Transformer):")
                print(f"   - Reward: {episode_reward:.2f}, Steps: {episode_steps}")
                print(
                    f"   - Result: {win_result}, WinRate: {win_rate:.1%} (Best: {self.best_win_rate:.1%})"
                )
                print(f"   - W/L/D: {self.wins}/{self.losses}/{self.draws}")
                print(f"   - Speed: {avg_speed:.2f}x")
                print(
                    f"   - Exploration: {agent_stats['exploration_rate']:.1%}, Success: {agent_stats['success_rate']:.1%}"
                )
                print(
                    f"   - Buffer: {buffer_stats['total_size']} (Good: {buffer_stats['good_count']}, Bad: {buffer_stats['bad_count']})"
                )
                print(
                    f"   - Diversity: {buffer_stats['avg_diversity']:.2f}, ActionDiversity: {buffer_stats['action_diversity']:.2f}"
                )
                print(
                    f"   - LearningRate: {current_lr:.2e}, Reboots: {self.reboot_count}"
                )
                print(
                    f"   - Timeouts: {self.timeout_wins}/{total_matches}, FastWins: {self.fast_wins}/{total_matches}"
                )
                print(f"   - Elapsed: {elapsed_time:.2f} hours")
                print(f"   - Image format: RGB ({SCREEN_WIDTH}x{SCREEN_HEIGHT})")

                self.logger.info(
                    f"Episode {episode}: Reward={episode_reward:.2f}, Steps={episode_steps}, "
                    f"Result={win_result}, WinRate={win_rate:.1%}, BestWinRate={self.best_win_rate:.1%}, "
                    f"AvgSpeed={avg_speed:.2f}, "
                    f"Exploration={agent_stats['exploration_rate']:.1%}, "
                    f"BufferSize={buffer_stats['total_size']}, "
                    f"GoodRatio={buffer_stats['good_ratio']:.2f}, "
                    f"Diversity={buffer_stats['avg_diversity']:.2f}, "
                    f"ActionDiversity={buffer_stats['action_diversity']:.2f}, "
                    f"LearningRate={current_lr:.2e}, Reboots={self.reboot_count}, "
                    f"Elapsed={elapsed_time:.2f} hours"
                )

            # Check for plateau and reboot if needed
            if (
                episode - self.last_reboot_episode > 50
                and self.detect_learning_plateau()
            ):
                self.reboot_learning_rate()

            # Save checkpoint
            if episode % self.args.save_frequency == 0:
                self.save_checkpoint(episode)

        self.env.close()
        print(f"\n🏁 Training completed!")
        print(f"   - Total episodes: {self.episode}")
        print(f"   - Final win rate: {win_rate:.1%}")
        print(f"   - Best win rate: {self.best_win_rate:.1%}")
        print(f"   - Total steps: {self.total_steps}")
        print(f"   - Fast wins: {self.fast_wins}, Timeouts: {self.timeout_wins}")
        print(f"   - Image format: RGB ({SCREEN_WIDTH}x{SCREEN_HEIGHT})")
        print(f"   - Transformer context sequence: ENABLED")
        self.logger.info(
            f"Training completed: Episodes={self.episode}, WinRate={win_rate:.1%}, BestWinRate={self.best_win_rate:.1%}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced RGB Street Fighter Training with Transformer (FIXED)"
    )
    parser.add_argument(
        "--num_episodes", type=int, default=1000, help="Number of episodes to train"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Initial learning rate"
    )
    parser.add_argument(
        "--features_dim", type=int, default=256, help="Feature dimension for SimpleCNN"
    )
    parser.add_argument(
        "--thinking_steps",
        type=int,
        default=6,
        help="Number of thinking steps for agent",
    )
    parser.add_argument(
        "--thinking_lr",
        type=float,
        default=0.025,
        help="Learning rate for thinking steps",
    )
    parser.add_argument(
        "--buffer_capacity", type=int, default=30000, help="Experience buffer capacity"
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--contrastive_margin", type=float, default=1.0, help="Contrastive loss margin"
    )
    parser.add_argument(
        "--contrastive_weight",
        type=float,
        default=0.5,
        help="Weight for contrastive loss",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping",
    )
    parser.add_argument(
        "--train_frequency", type=int, default=4, help="Train every N episodes"
    )
    parser.add_argument(
        "--log_frequency", type=int, default=10, help="Log every N episodes"
    )
    parser.add_argument(
        "--save_frequency",
        type=int,
        default=100,
        help="Save checkpoint every N episodes",
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=MAX_FIGHT_STEPS,
        help="Max steps per episode",
    )
    parser.add_argument(
        "--verify_health",
        action="store_true",
        help="Verify health detection before training",
    )
    parser.add_argument(
        "--load_checkpoint", type=str, default=None, help="Path to checkpoint to load"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-5,
        help="Weight decay for Adam optimizer",
    )

    args = parser.parse_args()
    trainer = EnhancedTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
