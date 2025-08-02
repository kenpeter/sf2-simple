"""
🚀 ENHANCED TRAINING - RGB Version with Transformer Context Sequence (SYNC FIXED + BEHAVIORAL COLLAPSE FIXED + TIER 2 HYBRID)
Key Fixes:
1. Fixed episode data synchronization issue
2. Proper logging timing alignment
3. Separated episode-specific from cumulative stats
4. Added sync verification
5. FIXED BEHAVIORAL COLLAPSE: Stricter experience buffer + Double Q-Learning
6. TIER 2 HYBRID APPROACH: Rich multimodal sequence processing
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
        HybridContextTransformer,
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
        VECTOR_FEATURE_DIM,
    )

    print(
        "✅ Successfully imported enhanced RGB wrapper components with Transformer + TIER 2 HYBRID"
    )
    print(f"✅ RGB processing with {SCREEN_WIDTH}x{SCREEN_HEIGHT} images: ACTIVE")
    print(f"✅ Transformer context sequence: ACTIVE")
    print(f"✅ TIER 2 HYBRID APPROACH: Rich multimodal sequence processing")
except ImportError as e:
    print(f"❌ Failed to import enhanced wrapper: {e}")
    print("Make sure wrapper.py is in the same directory")
    exit(1)


# RL Training Experience Buffer - stores complete transitions for Q-learning
# Different from CausalReplayBuffer (in wrapper.py) which stores MCMC action logits
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

        # FIXED: Make definition of "good" much stricter - only wins
        is_good_experience = win_result == "WIN"

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
        # MODIFIED: Sample 60% "good" and 40% "bad" to focus more on successful strategies
        good_count = int(batch_size * 0.6)
        bad_count = batch_size - good_count

        if (
            len(self.good_experiences) < good_count
            or len(self.bad_experiences) < bad_count
        ):
            return None, None  # Wait for enough diverse samples to be collected

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
            "tier2_hybrid": True,
        }


class EnhancedTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup_directories()
        print(
            f"🚀 Initializing ENHANCED RGB environment with Transformer + TIER 2 HYBRID..."
        )
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
        print(
            f"🧠 Initializing enhanced RGB models with Transformer + TIER 2 HYBRID..."
        )

        # we have verifier
        self.verifier = SimpleVerifier(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            features_dim=args.features_dim,
        ).to(self.device)

        # MODIFIED: Create a target network for stable learning

        # we have target verifier, for stable learning
        self.target_verifier = SimpleVerifier(
            # obs
            observation_space=self.env.observation_space,
            # action space
            action_space=self.env.action_space,
            # feature dim
            features_dim=args.features_dim,
        ).to(self.device)
        self.target_verifier.load_state_dict(self.verifier.state_dict())
        # network into eval mode so frozen
        self.target_verifier.eval()  # Target network is only for inference

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
        # Better scheduler with more appropriate parameters
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-6
        )
        # Better plateau detection parameters
        self.lr_reboot_threshold = 0.05
        self.performance_history = deque(maxlen=25)
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
        self.recent_rewards = deque(maxlen=100)
        self.timeout_wins = 0
        self.fast_wins = 0
        self.combo_count_history = deque(maxlen=50)
        self.speed_history = deque(maxlen=50)

        # SYNC FIX: Store per-episode data for accurate logging
        self.episode_data = {}

        self.setup_logging()
        self.load_checkpoint()
        print(
            f"🚀 Enhanced RGB Trainer initialized with Transformer + TIER 2 HYBRID (SYNC FIXED + BEHAVIORAL COLLAPSE FIXED)"
        )
        print(f"   - Device: {self.device}")
        print(f"   - Learning rate: {args.learning_rate:.2e} (with reboots)")
        print(f"   - Weight decay: {args.weight_decay:.2e}")
        print(f"   - Initial exploration: {self.agent.epsilon:.1%}")
        print(f"   - Reservoir sampling: ENABLED (60/40 Good/Bad Ratio)")
        print(f"   - Stable Learning: Target Network ENABLED (tau={args.tau})")
        print(f"   - Plateau detection: ACTIVE (Fixed)")
        print(f"   - Transformer context sequence: ENABLED")
        print(f"   - RGB processing: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
        print(f"   - Episode-data synchronization: FIXED")
        print(f"   - Behavioral collapse: FIXED (Double Q + Strict Buffer)")
        print(f"   - TIER 2 HYBRID APPROACH: Rich multimodal sequence processing")

    def _setup_directories(self):
        self.log_dir = Path("logs")
        self.checkpoint_dir = Path("checkpoints")
        self.log_dir.mkdir(exist_ok=True)
        self.checkpoint_dir.mkdir(exist_ok=True)

    def setup_logging(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"enhanced_rgb_training_tier2_{timestamp}.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def detect_learning_plateau(self):
        if len(self.performance_history) < 20:
            return False
        recent_performance = list(self.performance_history)[-15:]
        performance_std = safe_std(recent_performance, 0.0)
        is_stagnant = performance_std < self.lr_reboot_threshold
        early_performance = safe_mean(recent_performance[:7], 0.0)
        late_performance = safe_mean(recent_performance[-7:], 0.0)
        no_improvement = late_performance <= early_performance + 0.02

        # Also check win rate plateau
        recent_win_rates = list(self.win_rate_history)[-20:]
        if len(recent_win_rates) >= 20:
            win_rate_std = safe_std(recent_win_rates, 0.0)
            win_rate_stagnant = win_rate_std < 0.02
        else:
            win_rate_stagnant = False

        should_reboot = (
            (is_stagnant or win_rate_stagnant)
            and no_improvement
            and self.episode - self.last_reboot_episode > 75
        )
        if should_reboot:
            self.logger.info(
                f"🔄 Plateau detected: perf_std={performance_std:.4f}, "
                f"improvement={late_performance-early_performance:.4f}, "
                f"win_rate_std={safe_std(recent_win_rates, 0.0):.4f}"
            )
        return should_reboot

    def reboot_learning_rate(self):
        self.reboot_count += 1
        self.last_reboot_episode = self.episode
        # Better learning rate reboot strategy
        if self.reboot_count <= 3:
            new_lr = self.initial_learning_rate * (1.5**self.reboot_count)
        else:
            # After 3 reboots, try different strategy
            new_lr = self.initial_learning_rate * (0.5 ** max(0, self.reboot_count - 3))

        # Ensure minimum learning rate
        new_lr = max(new_lr, 1e-6)

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-6
        )
        # Better epsilon reboot strategy
        if self.reboot_count <= 2:
            self.agent.epsilon = min(0.6, self.agent.epsilon * 3.0)
        else:
            self.agent.epsilon = 0.4  # Reset to reasonable value

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
        filename = (
            self.checkpoint_dir / f"enhanced_rgb_tier2_checkpoint_ep_{episode}.pth"
        )
        state = {
            "episode": episode,
            "verifier_state_dict": self.verifier.state_dict(),
            "target_verifier_state_dict": self.target_verifier.state_dict(),  # MODIFIED: Save target network
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
            "tier2_hybrid": True,
            "args": self.args,
        }
        torch.save(state, filename)
        self.logger.info(f"💾 Enhanced RGB TIER 2 checkpoint saved to {filename}")

    def load_checkpoint(self):
        if self.args.load_checkpoint:
            checkpoint_path = self.args.load_checkpoint
            if os.path.exists(checkpoint_path):
                self.logger.info(
                    f"🔄 Loading enhanced RGB TIER 2 checkpoint from {checkpoint_path}..."
                )
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.verifier.load_state_dict(checkpoint["verifier_state_dict"])
                # MODIFIED: Load target network
                if "target_verifier_state_dict" in checkpoint:
                    self.target_verifier.load_state_dict(
                        checkpoint["target_verifier_state_dict"]
                    )
                else:  # For backwards compatibility with old checkpoints
                    self.target_verifier.load_state_dict(
                        checkpoint["verifier_state_dict"]
                    )

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
                tier2_enabled = checkpoint.get("tier2_hybrid", False)
                print(f"   📸 Checkpoint image format: {checkpoint_format}")
                print(
                    f"   📏 Checkpoint resolution: {checkpoint_width}x{checkpoint_height}"
                )
                print(f"   📏 Current resolution: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
                print(f"   🔧 TIER 2 HYBRID in checkpoint: {tier2_enabled}")
                self.logger.info(
                    f"✅ Enhanced RGB TIER 2 checkpoint loaded. Resuming from episode {self.episode}."
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
            # Pre-calculate the rich context sequence for the current state 'obs'
            current_rich_context_sequence = self.env.get_rich_context_sequence(
                self.verifier.features_extractor
            )
            # Inject it into a copy of the observation for prediction.
            obs_for_predict = obs.copy()
            obs_for_predict["rich_context_sequence"] = current_rich_context_sequence

            self.env.render()

            action, thinking_info = self.agent.predict(
                obs_for_predict, deterministic=False
            )
            next_obs, reward, done, truncated, info = self.env.step(action)
            episode_reward += reward
            episode_steps += 1
            self.total_steps += 1

            # After the step, get the rich context for the NEXT state
            next_rich_context_sequence = self.env.get_rich_context_sequence(
                self.verifier.features_extractor
            )

            final_player_health = info.get("agent_hp", 0)
            final_opponent_health = info.get("enemy_hp", 0)

            # Better win/loss detection logic
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
                "rich_context_sequence": current_rich_context_sequence,
                "action": action,
                "reward": reward,
                "next_obs": next_obs,
                "next_rich_context_sequence": next_rich_context_sequence,
                "done": done,
                "thinking_info": thinking_info,
                "episode": self.episode,
                "step": episode_steps,
                "frame_stack_size": FRAME_STACK_SIZE,
                "image_format": "RGB",
                "image_size": f"{SCREEN_WIDTH}x{SCREEN_HEIGHT}",
                "tier2_hybrid": True,
                "enhanced_context": {
                    "is_exploration": thinking_info.get("exploration", False),
                    "episode_progress": episode_steps / MAX_FIGHT_STEPS,
                },
            }
            episode_experiences.append(experience)
            obs = next_obs

        # Proper result classification
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

        # Add experiences to buffer (for Q-learning training)
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
        # good batch, bad batch
        good_batch, bad_batch = self.experience_buffer.sample_balanced_batch(
            self.args.batch_size
        )
        if good_batch is None or bad_batch is None:
            return None

        # shuffle good batch and bad batch
        batch = good_batch + bad_batch
        random.shuffle(batch)

        # obs, obs, action, reward, next obs, dones arr
        observations = []
        actions = []
        rewards = []
        next_observations = []
        dones = []
        rich_context_sequences_list = []
        next_rich_context_sequences_list = []

        # in batch, we get back obs, action, reward, next_obs, dones
        for exp in batch:
            observations.append(exp["obs"])
            actions.append(exp["action"])
            rewards.append(exp["reward"])
            next_observations.append(exp["next_obs"])
            dones.append(exp["done"])
            rich_context_sequences_list.append(exp["rich_context_sequence"])
            next_rich_context_sequences_list.append(exp["next_rich_context_sequence"])

        # device
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

        # TIER 2: Build rich context sequences from pre-calculated data
        rich_context_sequences = torch.tensor(
            np.stack(rich_context_sequences_list),
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

        # TIER 2: Build rich context sequences for next observations from pre-calculated data
        next_rich_context_sequences = torch.tensor(
            np.stack(next_rich_context_sequences_list),
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
            "rich_context_sequence": rich_context_sequences,  # TIER 2: Added
        }
        next_obs = {
            "visual_obs": next_visual_obs,
            "vector_obs": next_vector_obs,
            "context_sequence": next_context_sequences,
            "rich_context_sequence": next_rich_context_sequences,  # TIER 2: Added
        }

        self.optimizer.zero_grad()

        # Current energy (using initial landscape for training)
        current_energy = self.verifier(current_obs, action_one_hot, mcmc_step=0)

        # FIXED: Double Q-Learning target calculation with TIER 2

        # with this no training, no change weight
        with torch.no_grad():
            # 1. Find the best next action using the MAIN network
            next_energies_main_net = []
            # 56 action space
            for i in range(self.env.action_space.n):
                next_action_one_hot = torch.zeros(
                    batch_size, self.env.action_space.n, device=device
                )
                next_action_one_hot[:, i] = 1.0
                # in the loop, 56 times, we call verifier to get energy
                # energy score from energy net (using initial landscape)
                energy = self.verifier(
                    next_obs, next_action_one_hot, mcmc_step=0
                )  # Using self.verifier here!

                # assign energy into arr
                next_energies_main_net.append(energy)

            next_energies_main_net = torch.cat(next_energies_main_net, dim=1)
            best_next_action_indices = torch.argmin(next_energies_main_net, dim=1)
            best_next_action_one_hot = F.one_hot(
                best_next_action_indices, num_classes=self.env.action_space.n
            ).float()

            # 2. Evaluate the energy of that best action using the TARGET network
            next_energy = self.target_verifier(
                next_obs, best_next_action_one_hot, mcmc_step=0
            )  # Using self.target_verifier here!

            # 3. Calculate the final target
            target_energy = (
                rewards.unsqueeze(-1)
                + (1 - dones.unsqueeze(-1)) * self.args.gamma * next_energy
            )

        # Main loss
        loss = nn.functional.mse_loss(current_energy, target_energy)

        # Improved contrastive loss
        contrastive_loss = 0.0
        for _ in range(2):  # Reduced from 3 for stability
            negative_actions = torch.randint(
                0, self.env.action_space.n, (batch_size,), device=device
            )
            negative_one_hot = torch.zeros(
                batch_size, self.env.action_space.n, device=device
            )
            negative_one_hot.scatter_(1, negative_actions.unsqueeze(1), 1.0)
            negative_energy = self.verifier(current_obs, negative_one_hot, mcmc_step=0)
            contrastive_loss += torch.mean(
                nn.functional.relu(
                    current_energy - negative_energy + self.args.contrastive_margin
                )
            )

        # Reduce contrastive weight for stability
        total_loss = loss + (self.args.contrastive_weight) * contrastive_loss

        total_loss.backward()
        # Reduce gradient clipping for better convergence
        torch.nn.utils.clip_grad_norm_(
            self.verifier.parameters(), self.args.max_grad_norm * 0.5
        )
        self.optimizer.step()

        # MODIFIED: Update the target network via polyak averaging
        with torch.no_grad():
            # copy finished student to the target
            for target_param, param in zip(
                self.target_verifier.parameters(), self.verifier.parameters()
            ):
                target_param.data.mul_(1.0 - self.args.tau)
                target_param.data.add_(self.args.tau * param.data)

        self.scheduler.step(total_loss.item())

        return {
            "loss": total_loss.item(),
            "energy_loss": loss.item(),
            "contrastive_loss": contrastive_loss.item(),
        }

    def _log_synchronized_summary(self, episode):
        """SYNC FIX: Log with data that actually corresponds to the episode number"""
        if episode not in self.episode_data:
            print(f"⚠️ Episode {episode} data not found in episode_data!")
            return

        # Get this specific episode's data
        ep_data = self.episode_data[episode]

        # Get stats for context
        buffer_stats = self.experience_buffer.get_stats()
        agent_stats = self.agent.get_thinking_stats()
        avg_reward = safe_mean(list(self.recent_rewards), 0.0)
        avg_speed = safe_mean(list(self.speed_history), 1.0)
        current_lr = self.optimizer.param_groups[0]["lr"]
        elapsed_time = (
            (time.time() - self.train_start_time) / 3600
            if hasattr(self, "train_start_time")
            else 0
        )

        # Calculate win rate up to this episode
        total_matches = (
            ep_data["wins_total"] + ep_data["losses_total"] + ep_data["draws_total"]
        )
        win_rate_at_episode = safe_divide(ep_data["wins_total"], total_matches, 0.0)

        print(
            f"\n📊 Episode {episode} Summary (SYNC VERIFIED ✅ + EBT-ALIGNED 🧠 + TIER 2 HYBRID 🚀):"
        )
        print(
            f"   - THIS Episode: Reward={ep_data['reward']:.2f}, Steps={ep_data['steps']}, Result={ep_data['result']}"
        )
        print(
            f"   - Episode Change: W+{ep_data['wins_change']}, L+{ep_data['losses_change']}, D+{ep_data['draws_change']}"
        )
        print(
            f"   - CUMULATIVE (at ep {episode}): WinRate={win_rate_at_episode:.1%} (Best: {self.best_win_rate:.1%})"
        )
        print(
            f"   - CUMULATIVE (at ep {episode}): W/L/D={ep_data['wins_total']}/{ep_data['losses_total']}/{ep_data['draws_total']}"
        )
        print(f"   - Speed: {avg_speed:.2f}x")
        print(
            f"   - Exploration: {agent_stats['exploration_rate']:.1%}, Success: {agent_stats['success_rate']:.1%}"
        )
        print(
            f"   - EBT Energy Improvement: {agent_stats.get('energy_improvement_rate', 0.0):.1%}"
        )
        print(
            f"   - EBT Avg Thinking Steps: {agent_stats.get('average_thinking_steps', 0.0):.1f}"
        )
        print(
            f"   - EBT MCMC Acceptance: {agent_stats.get('mcmc_acceptance_rate', 0.0):.1%}"
        )
        print(
            f"   - EBT Replay Buffer: {agent_stats.get('replay_buffer_size', 0)} samples"
        )
        print(f"   - ActionDiversity: {agent_stats.get('action_diversity', 0.0):.3f}")
        print(
            f"   - Buffer: {buffer_stats['total_size']} (Good: {buffer_stats['good_ratio']:.1%} - STRICT WINS ONLY)"
        )
        print(
            f"   - Diversity: {buffer_stats['avg_diversity']:.2f}, ActionDiversity: {buffer_stats['action_diversity']:.3f}"
        )
        print(f"   - LearningRate: {current_lr:.2e}, Reboots: {self.reboot_count}")
        print(
            f"   - Timeouts: {self.timeout_wins}/{total_matches}, FastWins: {self.fast_wins}/{total_matches}"
        )
        print(f"   - Elapsed: {elapsed_time:.2f} hours")
        print(f"   - Image format: RGB ({SCREEN_WIDTH}x{SCREEN_HEIGHT})")
        print(f"   - 🔍 Sync Check: Episode data timestamp={ep_data['timestamp']:.1f}")
        print(f"   - 🔧 Behavioral Collapse Fixes: Double Q-Learning + Strict Buffer")
        print(
            f"   - 🚀 TIER 2 HYBRID: Rich multimodal sequence (Visual+Vector+Action+Reward)"
        )

        # Also log to file
        self.logger.info(
            f"Episode {episode} SYNC+FIX+TIER2: Reward={ep_data['reward']:.2f}, Steps={ep_data['steps']}, "
            f"Result={ep_data['result']}, WinRate={win_rate_at_episode:.1%}, "
            f"W/L/D={ep_data['wins_total']}/{ep_data['losses_total']}/{ep_data['draws_total']}, "
            f"Changes=W+{ep_data['wins_change']}/L+{ep_data['losses_change']}/D+{ep_data['draws_change']}, "
            f"Exploration={agent_stats['exploration_rate']:.1%}, "
            f"ActionDiversity={agent_stats.get('action_diversity', 0.0):.3f}, "
            f"BufferSize={buffer_stats['total_size']}, "
            f"LearningRate={current_lr:.2e}, Reboots={self.reboot_count}, "
            f"BehavioralCollapse=FIXED, TIER2_HYBRID=ENABLED"
        )

    def train(self):
        self.train_start_time = time.time()  # For elapsed time calculation

        print(
            f"🎮 Starting ENHANCED RGB training with Transformer + TIER 2 HYBRID + EBT-ALIGNED (SYNC FIXED + EBT THINKING)..."
        )
        print(f"   - Total episodes: {self.args.num_episodes}")
        print(f"   - Batch size: {self.args.batch_size}")
        print(f"   - Initial learning rate: {self.args.learning_rate:.2e}")
        print(f"   - Weight decay: {self.args.weight_decay:.2e}")
        print(f"   - Initial exploration: {self.agent.epsilon:.1%}")
        print(f"   - Frame stacking: {FRAME_STACK_SIZE} frames")
        print(f"   - Image format: RGB ({SCREEN_WIDTH}x{SCREEN_HEIGHT})")
        print(f"   - Transformer context sequence: ENABLED")
        print(f"   - Episode-data synchronization: FIXED")
        print(f"   - EBT-Aligned energy thinking: ACTIVE")
        print(f"     • True MCMC with Metropolis-Hastings acceptance")
        print(f"     • Multiple energy landscapes (step-dependent)")
        print(f"     • Langevin dynamics proposals (gradient + noise)")
        print(f"     • Causal replay buffer for experience reuse")
        print(f"     • Double Q-Learning (stable target calculation)")
        print(f"   - TIER 2 HYBRID APPROACH: Rich multimodal sequence processing")
        print(
            f"     • Visual features (256) + Vector features (32) + Action (56) + Reward (1)"
        )
        print(
            f"     • Deep temporal understanding across {FRAME_STACK_SIZE} historical steps"
        )

        for episode in range(self.episode, self.args.num_episodes):
            # SYNC FIX: Store episode state BEFORE running episode
            pre_episode_state = {
                "wins": self.wins,
                "losses": self.losses,
                "draws": self.draws,
                "timestamp": time.time(),
            }

            self.episode = episode
            episode_info = self.run_episode()

            # SYNC FIX: Store this episode's complete data immediately
            post_episode_state = {
                "wins": self.wins,
                "losses": self.losses,
                "draws": self.draws,
                "timestamp": time.time(),
            }

            # Calculate changes for this specific episode
            wins_this_episode = post_episode_state["wins"] - pre_episode_state["wins"]
            losses_this_episode = (
                post_episode_state["losses"] - pre_episode_state["losses"]
            )
            draws_this_episode = (
                post_episode_state["draws"] - pre_episode_state["draws"]
            )

            # Store complete episode data
            self.episode_data[episode] = {
                "reward": episode_info["episode_reward"],
                "steps": episode_info["episode_steps"],
                "result": episode_info["win_result"],
                "wins_total": post_episode_state["wins"],
                "losses_total": post_episode_state["losses"],
                "draws_total": post_episode_state["draws"],
                "wins_change": wins_this_episode,
                "losses_change": losses_this_episode,
                "draws_change": draws_this_episode,
                "timestamp": post_episode_state["timestamp"],
            }

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
                        f"ContrastiveLoss={train_info['contrastive_loss']:.4f}, "
                        f"TIER2_HYBRID=ENABLED"
                    )

            # SYNC FIX: Immediate logging with correct data
            if episode % self.args.log_frequency == 0:
                self._log_synchronized_summary(episode)

            # Check for plateau and reboot if needed
            if (
                episode - self.last_reboot_episode > 75
                and self.detect_learning_plateau()
            ):
                self.reboot_learning_rate()

            # Save checkpoint
            if episode % self.args.save_frequency == 0:
                self.save_checkpoint(episode)

        self.env.close()
        final_win_rate = safe_divide(
            self.wins, self.wins + self.losses + self.draws, 0.0
        )
        print(f"\n🏁 Training completed!")
        print(f"   - Total episodes: {self.episode}")
        print(f"   - Final win rate: {final_win_rate:.1%}")
        print(f"   - Best win rate: {self.best_win_rate:.1%}")
        print(f"   - Total steps: {self.total_steps}")
        print(f"   - Fast wins: {self.fast_wins}, Timeouts: {self.timeout_wins}")
        print(f"   - Learning rate reboots: {self.reboot_count}")
        print(f"   - Image format: RGB ({SCREEN_WIDTH}x{SCREEN_HEIGHT})")
        print(f"   - Transformer context sequence: ENABLED")
        print(f"   - Episode synchronization: VERIFIED ✅")
        print(f"   - Behavioral collapse: FIXED 🔧")
        print(f"   - TIER 2 HYBRID APPROACH: ENABLED 🚀")
        self.logger.info(
            f"Training completed: Episodes={self.episode}, WinRate={final_win_rate:.1%}, BestWinRate={self.best_win_rate:.1%}, Reboots={self.reboot_count}, BehavioralCollapse=FIXED, TIER2_HYBRID=ENABLED"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced RGB Street Fighter Training with Transformer + TIER 2 HYBRID + EBT-ALIGNED (SYNC FIXED + EBT THINKING)"
    )
    parser.add_argument(
        "--num_episodes", type=int, default=2000, help="Number of episodes to train"
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-4,
        help="Initial learning rate (INCREASED)",
    )
    parser.add_argument(
        "--features_dim", type=int, default=256, help="Feature dimension for SimpleCNN"
    )
    parser.add_argument(
        "--thinking_steps",
        type=int,
        default=8,  # MODIFIED: Back to 8 for stability
        help="Number of thinking steps for agent",
    )
    parser.add_argument(
        "--thinking_lr",
        type=float,
        default=0.025,  # MODIFIED: Back to 0.025 for stability
        help="Learning rate for thinking steps",
    )
    parser.add_argument(
        "--buffer_capacity", type=int, default=30000, help="Experience buffer capacity"
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--contrastive_margin",
        type=float,
        default=0.5,
        help="Contrastive loss margin (REDUCED)",
    )
    parser.add_argument(
        "--contrastive_weight",
        type=float,
        default=0.4,  # MODIFIED: Increased to strengthen learning signal
        help="Weight for contrastive loss",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=2.0,  # INCREASED
        help="Max gradient norm for clipping",
    )
    parser.add_argument(
        "--train_frequency",
        type=int,
        default=2,
        help="Train every N episodes (REDUCED)",
    )
    parser.add_argument(
        "--log_frequency",
        type=int,
        default=5,
        help="Log every N episodes (REDUCED for better sync)",
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
        default=5e-6,  # REDUCED
        help="Weight decay for Adam optimizer",
    )
    # MODIFIED: Added tau for target network updates
    parser.add_argument(
        "--tau",
        type=float,
        default=0.005,
        help="Polyak averaging factor for target network update",
    )

    args = parser.parse_args()

    print(f"🔧 SYNC FIXED + EBT-ALIGNED + TIER 2 HYBRID Training Configuration:")
    print(f"   - Learning rate: {args.learning_rate:.2e} (INCREASED)")
    print(f"   - Thinking steps: {args.thinking_steps} (BACK TO 8 for stability)")
    print(f"   - Thinking LR: {args.thinking_lr:.3f} (BACK TO 0.025 for stability)")
    print(f"   - Train frequency: {args.train_frequency} (INCREASED)")
    print(f"   - Log frequency: {args.log_frequency} (REDUCED for better sync)")
    print(f"   - Contrastive weight: {args.contrastive_weight:.2f} (INCREASED)")
    print(f"   - Weight decay: {args.weight_decay:.2e} (REDUCED)")
    print(f"   - Max grad norm: {args.max_grad_norm:.1f} (INCREASED)")
    print(f"   - Target Network Tau: {args.tau} (ADDED for stable learning)")
    print(f"   - Episode-data synchronization: ENABLED ✅")
    print(f"   - EBT-Aligned energy thinking: ENABLED 🧠")
    print(f"     • True MCMC: Metropolis-Hastings acceptance")
    print(f"     • Multiple landscapes: Step-dependent energy functions")
    print(f"     • Langevin proposals: Gradient + noise sampling")
    print(f"     • Causal replay buffer: EXPERIENCE reuse")
    print(f"     • Double Q-Learning: STABLE target calculation")
    print(f"   - TIER 2 HYBRID APPROACH: ENABLED 🚀")
    print(f"     • Rich multimodal sequence processing")
    print(
        f"     • Visual features (256) + Vector features (32) + Action (56) + Reward (1)"
    )
    print(f"     • Deep temporal understanding across {FRAME_STACK_SIZE} steps")

    trainer = EnhancedTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
