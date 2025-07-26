#!/usr/bin/env python3
"""
🚀 FINAL FIXED OPTIMIZED EBT TRAINING - High Performance Energy-Based Learning
Final fix for the list/int modulo error
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
import threading
import cv2
import traceback

# Import the optimized wrapper components with FIXED AGENT
from wrapper import (
    make_optimized_sf_env,
    verify_ebt_energy_flow,
    OptimizedStreetFighterVerifier,
    FixedOptimizedEnergyBasedAgent,  # Use the FIXED version
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
    VECTOR_FEATURE_DIM,
    OPTIMIZED_ACTIONS,
    SCREEN_WIDTH,
    SCREEN_HEIGHT,
    FRAME_STACK_SIZE,
)


class OptimizedEBTTrainer:
    """🚀 High-performance trainer with optimized EBT integration - FINAL FIXED VERSION."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_active = True

        # Enable optimizations
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        print(f"🎮 Initializing optimized Street Fighter environment...")
        self.env = make_optimized_sf_env()

        # NEW: Setup for daemon rendering thread
        self.render_frame = None
        self.render_lock = threading.Lock()
        self.render_thread = None
        if self.args.render:
            self.render_thread = threading.Thread(target=self._render_loop, daemon=True)
            print("   - Asynchronous rendering: ENABLED")

        # Initialize optimized verifier and agent
        self.verifier = OptimizedStreetFighterVerifier(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            features_dim=args.features_dim,
            use_ebt=args.use_ebt,
        ).to(self.device)

        # Verify energy flow
        if not verify_ebt_energy_flow(
            self.verifier, self.env.observation_space, self.env.action_space
        ):
            raise RuntimeError("EBT energy flow verification failed!")

        self.agent = FixedOptimizedEnergyBasedAgent(  # Use FIXED version
            verifier=self.verifier,
            thinking_steps=args.thinking_steps,
            thinking_lr=args.thinking_lr,
            noise_scale=args.noise_scale,
            use_ebt_thinking=args.use_ebt_thinking,
        )

        # Initialize managers
        self.policy_memory = PolicyMemoryManager(
            performance_drop_threshold=args.performance_drop_threshold,
            averaging_weight=args.averaging_weight,
        )

        self.experience_buffer = EBTEnhancedExperienceBuffer(
            capacity=args.buffer_capacity,
            quality_threshold=args.quality_threshold,
            golden_buffer_capacity=args.golden_buffer_capacity,
        )

        self.stability_manager = EnhancedEnergyStabilityManager(
            initial_lr=args.learning_rate,
            thinking_lr=args.thinking_lr,
            policy_memory_manager=self.policy_memory,
        )

        self.checkpoint_manager = EBTEnhancedCheckpointManager(
            checkpoint_dir=args.checkpoint_dir
        )

        # Optimized optimizer with different learning rates for components
        optimizer_params = []

        for name, param in self.verifier.named_parameters():
            if "ebt" in name and args.use_ebt:
                # Higher learning rate for EBT components initially
                optimizer_params.append(
                    {
                        "params": param,
                        "lr": args.learning_rate * args.ebt_lr_multiplier,
                        "weight_decay": args.weight_decay * 0.5,
                    }
                )
            else:
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

        # Enable mixed precision for performance
        if torch.cuda.is_available():
            try:
                # Use the newer API if available
                self.scaler = torch.amp.GradScaler("cuda")
            except:
                # Fallback to older API
                self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        # Training state - FIXED: Ensure all counters are integers
        self.episode = 0
        self.total_steps = 0
        self.best_win_rate = 0.0

        # Performance tracking with shorter windows for faster adaptation
        self.win_rate_history = deque(maxlen=int(args.win_rate_window))
        self.energy_quality_history = deque(maxlen=25)
        self.ebt_performance_history = deque(maxlen=25)
        self.reward_history = deque(maxlen=100)
        self.last_checkpoint_episode = 0

        # Performance monitoring
        self.episode_times = deque(maxlen=10)
        self.training_start_time = time.time()

        self.setup_logging()

        print(f"🚀 Optimized EBT Trainer initialized")
        print(f"   - Device: {self.device}")
        print(f"   - Mixed precision: {'ENABLED' if self.scaler else 'DISABLED'}")
        print(f"   - Learning rate: {args.learning_rate:.2e}")
        print(f"   - Action space: {len(OPTIMIZED_ACTIONS)} actions")
        print(f"   - Screen size: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
        print(f"   - Frame Stack: {FRAME_STACK_SIZE} frames")

    def setup_logging(self):
        """Setup optimized logging system."""
        log_dir = Path("logs_optimized")
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"optimized_training_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def _render_loop(self):
        """A daemon thread function for rendering the game screen without blocking training."""
        window_name = "Street Fighter II - Render Thread"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        while self.training_active:
            frame_to_render = None
            with self.render_lock:
                if self.render_frame is not None:
                    # Make a copy to avoid race conditions after lock is released
                    frame_to_render = self.render_frame.copy()

            if frame_to_render is not None:
                # Retro env returns RGB, OpenCV uses BGR
                frame_bgr = cv2.cvtColor(frame_to_render, cv2.COLOR_RGB2BGR)
                cv2.imshow(window_name, frame_bgr)

            # Check for quit key, and sleep to target ~30 FPS
            if cv2.waitKey(33) & 0xFF == ord("q"):
                self.training_active = False
                break

        cv2.destroyWindow(window_name)

    def calculate_enhanced_experience_quality(
        self, reward, reward_breakdown, episode_stats, thinking_info=None
    ):
        """Enhanced quality calculation optimized for Street Fighter learning."""
        base_quality = 0.5

        # Reward component (normalized and capped)
        reward_component = np.tanh(reward / 5.0) * 0.25  # Smoother scaling

        # Combat effectiveness (most important for SF)
        win_component = 0.0
        if "round_won" in reward_breakdown:
            win_component = 0.5  # Strong positive signal
        elif "round_lost" in reward_breakdown:
            win_component = -0.2  # Moderate negative signal

        # Damage effectiveness
        damage_component = 0.0
        if "damage_dealt" in reward_breakdown:
            damage_ratio = (
                reward_breakdown["damage_dealt"] / 2.0
            )  # Normalize by max expected
            damage_component = min(damage_ratio, 0.3)

        # Health management
        health_component = reward_breakdown.get("health_advantage", 0.0) * 0.1

        # Action engagement (prevents passive play)
        action_component = 0.0
        if "action_bonus" in reward_breakdown:
            action_component = 0.05
        elif "idle_penalty" in reward_breakdown:
            action_component = -0.05

        # EBT optimization quality
        ebt_component = 0.0
        if thinking_info and self.args.use_ebt:
            if thinking_info.get("optimization_successful", False):
                ebt_component += 0.1
            if thinking_info.get("ebt_success", False):
                ebt_component += 0.05

            # Reward meaningful energy improvements
            energy_improvement = thinking_info.get("energy_improvement", 0.0)
            if energy_improvement > 0.001:
                ebt_component += min(energy_improvement * 10.0, 0.1)

        # Episode context bonus
        episode_component = 0.0
        if episode_stats:
            if episode_stats.get("won", False):
                episode_component = 0.15
            elif episode_stats.get("damage_ratio", 0) > 1.5:
                episode_component = 0.05

        quality_score = (
            base_quality
            + reward_component
            + win_component
            + damage_component
            + health_component
            + action_component
            + ebt_component
            + episode_component
        )

        return np.clip(quality_score, 0.0, 1.0)

    def run_episode(self):
        """Optimized episode running with performance monitoring - FINAL FIXED VERSION."""
        episode_start_time = time.time()

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
        action_bonuses = 0

        # EBT-specific tracking
        ebt_successes = 0
        ebt_failures = 0
        total_energy_improvement = 0.0

        while (
            not done
            and not truncated
            and episode_steps < self.args.max_episode_steps
            and self.training_active
        ):
            # Get EBT sequence context from environment
            sequence_context = None
            if self.args.use_ebt_thinking:
                try:
                    sequence_context = self.env.get_ebt_sequence(self.device)
                except Exception as e:
                    if self.args.debug:
                        print(f"⚠️ EBT sequence error: {e}")
                    sequence_context = None

            # Agent prediction with EBT
            action, thinking_info = self.agent.predict(
                obs, deterministic=False, sequence_context=sequence_context
            )

            # Execute action
            next_obs, reward, done, truncated, info = self.env.step(action)

            # NEW: Update shared frame for rendering thread
            if self.args.render:
                raw_frame = self.env.get_last_raw_obs()
                if raw_frame is not None:
                    with self.render_lock:
                        self.render_frame = raw_frame

            # Update EBT sequence tracker with energy score
            if hasattr(self.env, "feature_tracker") and thinking_info:
                energy_score = thinking_info.get("final_energy", 0.0)
                # FIXED: Check if we have any steps recorded before accessing
                if (
                    hasattr(self.env.feature_tracker, "ebt_tracker")
                    and hasattr(self.env.feature_tracker.ebt_tracker, "energy_sequence")
                    and len(self.env.feature_tracker.ebt_tracker.energy_sequence) > 0
                ):
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

            if "action_bonus" in reward_breakdown:
                action_bonuses += 1

            if "round_won" in reward_breakdown:
                round_won = True

            # Track EBT performance
            if thinking_info.get("ebt_success", True):
                ebt_successes += 1
            else:
                ebt_failures += 1

            total_energy_improvement += thinking_info.get("energy_improvement", 0.0)

            # Calculate experience quality with enhanced factors
            episode_stats = {
                "won": round_won,
                "damage_ratio": safe_divide(
                    damage_dealt_total, damage_taken_total + 1e-6, 1.0
                ),
                "action_engagement": action_bonuses / max(episode_steps, 1),
            }

            quality_score = self.calculate_enhanced_experience_quality(
                reward, reward_breakdown, episode_stats, thinking_info
            )

            # Store experience with EBT sequence
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
                "reward_breakdown": reward_breakdown,
            }

            # Add EBT sequence to experience if available
            ebt_sequence = None
            if sequence_context is not None:
                ebt_sequence = sequence_context.detach().cpu().numpy()

            episode_experiences.append((experience, quality_score, ebt_sequence))
            obs = next_obs

        # Episode completed - calculate final stats
        episode_time = time.time() - episode_start_time
        self.episode_times.append(episode_time)

        episode_stats_final = {
            "won": round_won,
            "damage_ratio": safe_divide(
                damage_dealt_total, damage_taken_total + 1e-6, 1.0
            ),
            "reward": episode_reward,
            "steps": episode_steps,
            "duration": episode_time,
            "ebt_successes": ebt_successes,
            "ebt_failures": ebt_failures,
            "ebt_success_rate": safe_divide(
                ebt_successes, ebt_successes + ebt_failures, 1.0
            ),
            "avg_energy_improvement": safe_divide(
                total_energy_improvement, episode_steps, 0.0
            ),
            "action_engagement": action_bonuses / max(episode_steps, 1),
        }

        # Add experiences to buffer
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

    def calculate_optimized_contrastive_loss(
        self, good_batch, bad_batch, sequence_batch=None, margin=1.5
    ):
        """Optimized contrastive loss with mixed precision and EBT integration - FINAL FIXED VERSION."""
        device = self.device

        def process_batch_efficiently(batch):
            if not batch:
                return None, None, None

            obs_batch = []
            action_batch = []
            sequence_batch_processed = []

            for exp in batch:
                obs = exp["obs"]
                action = exp["action"]

                # Efficient observation conversion
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

                # Efficient action one-hot encoding
                action_one_hot = torch.zeros(len(OPTIMIZED_ACTIONS))
                action_one_hot[action] = 1.0

                # Handle EBT sequence efficiently
                ebt_sequence = exp.get("ebt_sequence", None)
                if ebt_sequence is not None and ebt_sequence.size > 0:
                    sequence_tensor = torch.from_numpy(ebt_sequence).float()
                    if sequence_tensor.dim() == 2:
                        sequence_tensor = sequence_tensor.unsqueeze(0)
                    elif sequence_tensor.dim() == 3 and sequence_tensor.shape[0] != 1:
                        sequence_tensor = sequence_tensor[:1]

                    # Ensure correct dimensions
                    if sequence_tensor.shape[-1] != VECTOR_FEATURE_DIM:
                        current_features = sequence_tensor.shape[-1]
                        if current_features < VECTOR_FEATURE_DIM:
                            padding = torch.zeros(
                                sequence_tensor.shape[0],
                                sequence_tensor.shape[1],
                                VECTOR_FEATURE_DIM - current_features,
                            )
                            sequence_tensor = torch.cat(
                                [sequence_tensor, padding], dim=-1
                            )
                        else:
                            sequence_tensor = sequence_tensor[:, :, :VECTOR_FEATURE_DIM]
                else:
                    sequence_tensor = torch.zeros(
                        1, EBT_SEQUENCE_LENGTH, VECTOR_FEATURE_DIM
                    )

                obs_batch.append(obs_tensor)
                action_batch.append(action_one_hot)
                sequence_batch_processed.append(sequence_tensor)

            return obs_batch, action_batch, sequence_batch_processed

        # Process batches
        good_obs, good_actions, good_sequences = process_batch_efficiently(good_batch)
        bad_obs, bad_actions, bad_sequences = process_batch_efficiently(bad_batch)

        if good_obs is None or bad_obs is None:
            return torch.tensor(0.0, device=device), {}

        # Stack observations efficiently
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
            try:
                good_seq_processed = [
                    seq.squeeze(0) if seq.dim() == 3 and seq.shape[0] == 1 else seq
                    for seq in good_sequences
                ]
                bad_seq_processed = [
                    seq.squeeze(0) if seq.dim() == 3 and seq.shape[0] == 1 else seq
                    for seq in bad_sequences
                ]

                good_sequences_stacked = torch.stack(good_seq_processed).to(device)
                bad_sequences_stacked = torch.stack(bad_seq_processed).to(device)

                # Verify shapes
                expected_shape = (
                    len(good_batch),
                    EBT_SEQUENCE_LENGTH,
                    VECTOR_FEATURE_DIM,
                )
                if good_sequences_stacked.shape != expected_shape:
                    good_sequences_stacked = None
                    bad_sequences_stacked = None

            except Exception as e:
                if self.args.debug:
                    print(f"⚠️ EBT sequence processing failed: {e}")
                good_sequences_stacked = None
                bad_sequences_stacked = None

        # Calculate energies with mixed precision
        if self.scaler:
            try:
                with torch.amp.autocast("cuda"):
                    good_energies = self.verifier(
                        good_obs_stacked, good_actions_stacked, good_sequences_stacked
                    )
                    bad_energies = self.verifier(
                        bad_obs_stacked, bad_actions_stacked, bad_sequences_stacked
                    )
            except:
                # Fallback to older API
                with torch.cuda.amp.autocast():
                    good_energies = self.verifier(
                        good_obs_stacked, good_actions_stacked, good_sequences_stacked
                    )
                    bad_energies = self.verifier(
                        bad_obs_stacked, bad_actions_stacked, bad_sequences_stacked
                    )
        else:
            good_energies = self.verifier(
                good_obs_stacked, good_actions_stacked, good_sequences_stacked
            )
            bad_energies = self.verifier(
                bad_obs_stacked, bad_actions_stacked, bad_sequences_stacked
            )

        # Enhanced contrastive loss with adaptive margin
        good_energy_mean = good_energies.mean()
        bad_energy_mean = bad_energies.mean()

        # Dynamic margin based on energy separation
        current_separation = abs((bad_energy_mean - good_energy_mean).item())
        adaptive_margin = max(
            margin * 0.5,
            min(margin * 2.0, margin + (margin - current_separation) * 0.1),
        )

        # We want good energies to be lower (more negative) than bad energies
        energy_diff = bad_energy_mean - good_energy_mean
        contrastive_loss = torch.clamp(adaptive_margin - energy_diff, min=0.0)

        # Regularization terms
        energy_reg = 0.005 * (good_energies.pow(2).mean() + bad_energies.pow(2).mean())

        # EBT-specific regularization
        ebt_reg = torch.tensor(0.0, device=device)
        if (
            self.args.use_ebt
            and good_sequences_stacked is not None
            and hasattr(self.verifier, "ebt")
        ):
            try:
                with torch.no_grad():
                    good_ebt_result = self.verifier.ebt(good_sequences_stacked)
                    bad_ebt_result = self.verifier.ebt(bad_sequences_stacked)

                    if (
                        isinstance(good_ebt_result, dict)
                        and "sequence_energies" in good_ebt_result
                    ):
                        good_ebt_energies = good_ebt_result["sequence_energies"]
                        bad_ebt_energies = bad_ebt_result["sequence_energies"]
                        ebt_reg = 0.002 * (
                            good_ebt_energies.pow(2).mean()
                            + bad_ebt_energies.pow(2).mean()
                        )

            except Exception as e:
                if self.args.debug:
                    print(f"⚠️ EBT regularization failed: {e}")

        total_loss = contrastive_loss + energy_reg + ebt_reg

        loss_info = {
            "contrastive_loss": contrastive_loss.item(),
            "energy_reg": energy_reg.item(),
            "ebt_reg": ebt_reg.item(),
            "good_energy_mean": good_energy_mean.item(),
            "bad_energy_mean": bad_energy_mean.item(),
            "energy_separation": energy_diff.item(),
            "adaptive_margin": adaptive_margin,
            "used_ebt_sequences": good_sequences_stacked is not None,
        }

        return total_loss, loss_info

    def train_step(self):
        """Optimized training step with mixed precision - FINAL FIXED VERSION."""
        # Sample balanced batch
        batch_result = self.experience_buffer.sample_enhanced_balanced_batch(
            self.args.batch_size, golden_ratio=0.1, sequence_ratio=0.15
        )

        if batch_result[0] is None or batch_result[1] is None:
            return None

        good_batch, bad_batch, golden_batch, sequence_batch = batch_result

        # Calculate loss with mixed precision
        if self.scaler:
            try:
                with torch.amp.autocast("cuda"):
                    loss, loss_info = self.calculate_optimized_contrastive_loss(
                        good_batch,
                        bad_batch,
                        sequence_batch,
                        margin=self.args.contrastive_margin,
                    )
            except:
                # Fallback to older API
                with torch.cuda.amp.autocast():
                    loss, loss_info = self.calculate_optimized_contrastive_loss(
                        good_batch,
                        bad_batch,
                        sequence_batch,
                        margin=self.args.contrastive_margin,
                    )
        else:
            loss, loss_info = self.calculate_optimized_contrastive_loss(
                good_batch,
                bad_batch,
                sequence_batch,
                margin=self.args.contrastive_margin,
            )

        # Get current learning rates
        current_lr, current_thinking_lr = self.stability_manager.get_current_lrs()

        # Update optimizer learning rates
        for param_group in self.optimizer.param_groups:
            base_lr = param_group.get("lr", current_lr)
            if base_lr > current_lr * 1.1:  # This is an EBT parameter group
                param_group["lr"] = current_lr * self.args.ebt_lr_multiplier
            else:
                param_group["lr"] = current_lr

        self.agent.current_thinking_lr = current_thinking_lr

        # Backward pass with mixed precision
        self.optimizer.zero_grad()

        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)

            # Gradient clipping
            total_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.verifier.parameters(), max_norm=0.5
            )

            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            total_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.verifier.parameters(), max_norm=0.5
            )
            self.optimizer.step()

        # Enhanced loss info
        loss_info.update(
            {
                "total_grad_norm": (
                    total_grad_norm.item()
                    if isinstance(total_grad_norm, torch.Tensor)
                    else total_grad_norm
                ),
                "learning_rate": current_lr,
                "thinking_lr": current_thinking_lr,
                "ebt_lr": (
                    current_lr * self.args.ebt_lr_multiplier
                    if self.args.use_ebt
                    else 0.0
                ),
                "sequence_batch_size": len(sequence_batch) if sequence_batch else 0,
                "golden_batch_size": len(golden_batch) if golden_batch else 0,
            }
        )

        return loss_info

    def evaluate_performance(self):
        """Optimized evaluation with performance monitoring - FINAL FIXED VERSION."""
        eval_episodes = min(3, max(1, self.episode // 200))  # Fewer episodes for speed

        wins = 0
        total_reward = 0.0
        total_steps = 0

        # EBT-specific metrics
        total_ebt_successes = 0
        total_ebt_attempts = 0
        total_energy_improvements = 0.0

        eval_start_time = time.time()

        for eval_ep in range(eval_episodes):
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

                # Deterministic prediction for evaluation
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

        eval_time = time.time() - eval_start_time

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
            "eval_time": eval_time,
            "ebt_success_rate": ebt_success_rate,
            "avg_energy_improvement": avg_energy_improvement,
            "ebt_attempts": total_ebt_attempts,
        }

    def handle_policy_memory_operations(self, performance_stats, train_stats):
        """Enhanced policy memory operations with optimized thresholds - FINAL FIXED VERSION."""
        current_win_rate = performance_stats["win_rate"]

        # Find base learning rate
        current_lr = None
        for param_group in self.optimizer.param_groups:
            if param_group.get("lr", 0) <= self.args.learning_rate * 1.1:
                current_lr = param_group["lr"]
                break

        if current_lr is None:
            current_lr = self.optimizer.param_groups[0]["lr"]

        # Update policy memory
        performance_improved, performance_drop = self.policy_memory.update_performance(
            current_win_rate, self.episode, self.verifier.state_dict(), current_lr
        )

        # Update experience buffer win rate
        self.experience_buffer.update_win_rate(current_win_rate)

        policy_memory_action_taken = False

        # Handle performance improvement
        if performance_improved:
            print(f"🏆 NEW PEAK PERFORMANCE: {current_win_rate:.1%}!")

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

            if self.policy_memory.should_perform_averaging(self.episode):
                print(f"🔄 Performing checkpoint averaging...")
                averaging_success = self.policy_memory.perform_checkpoint_averaging(
                    self.verifier
                )

                if averaging_success:
                    print(f"✅ Checkpoint averaging successful")
                    policy_memory_action_taken = True

                    # Reduce learning rate
                    if self.policy_memory.should_reduce_lr():
                        new_lr = self.policy_memory.get_reduced_lr(current_lr)

                        for param_group in self.optimizer.param_groups:
                            old_lr = param_group["lr"]
                            if old_lr > current_lr * 1.1:  # EBT parameter group
                                param_group["lr"] = new_lr * self.args.ebt_lr_multiplier
                            else:
                                param_group["lr"] = new_lr

                        self.stability_manager.current_lr = new_lr
                        print(f"📉 Learning rates reduced - Base: {new_lr:.2e}")

        return policy_memory_action_taken

    def train(self):
        """Optimized main training loop - FINAL FIXED VERSION."""
        print(f"\n🚀 Starting Optimized EBT Training")
        print(f"   - Target episodes: {self.args.max_episodes}")
        print(f"   - Action space: {len(OPTIMIZED_ACTIONS)} actions")
        print(f"   - Mixed precision: {'ENABLED' if self.scaler else 'DISABLED'}")

        # Start the rendering thread if enabled
        if self.render_thread:
            self.render_thread.start()

        # Training metrics
        episode_rewards = deque(maxlen=50)
        recent_losses = deque(maxlen=25)
        training_start_time = time.time()

        # FINAL FIX: Ensure all arguments are proper integers/floats before the loop
        max_episodes = int(self.args.max_episodes)
        eval_frequency = int(self.args.eval_frequency)
        checkpoint_frequency = int(self.args.checkpoint_frequency)

        print(
            f"🔍 DEBUG: Loop parameters - max_episodes: {max_episodes}, eval_frequency: {eval_frequency}"
        )

        for episode_idx in range(max_episodes):
            if not self.training_active:
                print("🛑 Training loop terminated by render window closure.")
                break

            # FINAL FIX: Ensure episode is always an integer
            self.episode = int(episode_idx)
            episode_start_time = time.time()

            # Run episode
            episode_stats = self.run_episode()
            episode_rewards.append(episode_stats["reward"])
            self.reward_history.append(episode_stats["reward"])

            # Track EBT performance
            ebt_success_rate = episode_stats.get("ebt_success_rate", 1.0)
            self.ebt_performance_history.append(ebt_success_rate)

            # Training step
            train_stats = self.train_step()
            if train_stats:
                recent_losses.append(train_stats.get("contrastive_loss", 0.0))
            else:
                train_stats = {}

            # FINAL FIX: Use integer modulo operations with explicit type checking
            if self.episode % eval_frequency == 0:
                # Performance evaluation
                performance_stats = self.evaluate_performance()

                # FINAL FIX: Ensure win_rate is a scalar float, not a list
                win_rate = performance_stats["win_rate"]
                if isinstance(win_rate, (list, np.ndarray)):
                    win_rate = float(win_rate[0]) if len(win_rate) > 0 else 0.0
                else:
                    win_rate = float(win_rate)
                performance_stats["win_rate"] = win_rate

                self.win_rate_history.append(win_rate)

                # Calculate energy quality metrics
                energy_separation = train_stats.get("energy_separation", 0.0)
                if isinstance(energy_separation, (list, np.ndarray)):
                    energy_separation = (
                        float(energy_separation[0])
                        if len(energy_separation) > 0
                        else 0.0
                    )
                else:
                    energy_separation = float(energy_separation)

                energy_quality = abs(energy_separation) * 10.0
                self.energy_quality_history.append(energy_quality)

                # Policy memory operations
                policy_memory_action = self.handle_policy_memory_operations(
                    performance_stats, train_stats
                )

                # Stability management
                early_stop_rate = safe_divide(
                    self.agent.thinking_stats.get("early_stops", 0),
                    self.agent.thinking_stats.get("total_predictions", 1),
                    0.0,
                )
                avg_ebt_success = safe_mean(list(self.ebt_performance_history), 1.0)

                stability_emergency = self.stability_manager.update_metrics(
                    win_rate,
                    energy_quality,
                    energy_separation,
                    early_stop_rate,
                    avg_ebt_success,
                )

                # Handle stability emergency
                if stability_emergency and not policy_memory_action:
                    print(f"🚨 Stability emergency - adjusting learning rates!")
                    new_lr, new_thinking_lr = self.stability_manager.get_current_lrs()

                    for param_group in self.optimizer.param_groups:
                        if param_group.get("lr", 0) > new_lr * 1.1:
                            param_group["lr"] = new_lr * self.args.ebt_lr_multiplier
                        else:
                            param_group["lr"] = new_lr

                    self.agent.current_thinking_lr = new_thinking_lr

                # FINAL FIX: Checkpoint saving with integer operations
                checkpoint_due = (
                    self.episode - self.last_checkpoint_episode
                ) >= checkpoint_frequency

                if checkpoint_due or win_rate > self.best_win_rate:

                    ebt_stats = {
                        "ebt_success_rate": performance_stats.get(
                            "ebt_success_rate", 1.0
                        ),
                        "avg_energy_improvement": performance_stats.get(
                            "avg_energy_improvement", 0.0
                        ),
                        "ebt_enabled": self.args.use_ebt,
                        "ebt_thinking_enabled": self.args.use_ebt_thinking,
                        "episode": self.episode,
                    }

                    self.checkpoint_manager.save_checkpoint(
                        self.verifier,
                        self.agent,
                        self.episode,
                        win_rate,
                        energy_quality,
                        policy_memory_stats=self.policy_memory.get_stats(),
                        ebt_stats=ebt_stats,
                    )
                    self.last_checkpoint_episode = self.episode

                    if win_rate > self.best_win_rate:
                        self.best_win_rate = win_rate

                # FINAL FIX: Adjust experience buffer threshold with integer operations
                if self.episode % 20 == 0:
                    self.experience_buffer.adjust_threshold(episode_number=self.episode)

                # Performance reporting
                buffer_stats = self.experience_buffer.get_stats()
                thinking_stats = self.agent.get_thinking_stats()
                current_lr = self.optimizer.param_groups[0]["lr"]

                # Calculate performance metrics
                avg_episode_time = safe_mean(list(self.episode_times), 1.0)
                episodes_per_hour = 3600 / max(avg_episode_time, 1.0)
                avg_reward = safe_mean(list(self.reward_history)[-50:], 0.0)

                # FINAL FIX: Ensure all values are properly converted to scalars
                ebt_success_rate_display = performance_stats.get(
                    "ebt_success_rate", 1.0
                )
                if isinstance(ebt_success_rate_display, (list, np.ndarray)):
                    ebt_success_rate_display = (
                        float(ebt_success_rate_display[0])
                        if len(ebt_success_rate_display) > 0
                        else 1.0
                    )
                else:
                    ebt_success_rate_display = float(ebt_success_rate_display)

                avg_energy_improvement_display = performance_stats.get(
                    "avg_energy_improvement", 0.0
                )
                if isinstance(avg_energy_improvement_display, (list, np.ndarray)):
                    avg_energy_improvement_display = (
                        float(avg_energy_improvement_display[0])
                        if len(avg_energy_improvement_display) > 0
                        else 0.0
                    )
                else:
                    avg_energy_improvement_display = float(
                        avg_energy_improvement_display
                    )

                thinking_success_rate = thinking_stats.get("success_rate", 1.0)
                if isinstance(thinking_success_rate, (list, np.ndarray)):
                    thinking_success_rate = (
                        float(thinking_success_rate[0])
                        if len(thinking_success_rate) > 0
                        else 1.0
                    )
                else:
                    thinking_success_rate = float(thinking_success_rate)

                contrastive_loss_display = train_stats.get("contrastive_loss", 0.0)
                if isinstance(contrastive_loss_display, (list, np.ndarray)):
                    contrastive_loss_display = (
                        float(contrastive_loss_display[0])
                        if len(contrastive_loss_display) > 0
                        else 0.0
                    )
                else:
                    contrastive_loss_display = float(contrastive_loss_display)

                print(f"\n🚀 OPTIMIZED STATUS (Episode {self.episode}):")
                print(f"   🎯 Performance:")
                print(f"      - Win rate: {win_rate:.1%}")
                print(f"      - Avg reward (50ep): {float(avg_reward):.2f}")
                print(f"      - Energy separation: {energy_separation:.4f}")
                print(f"      - Episodes/hour: {float(episodes_per_hour):.1f}")

                print(f"   🤖 Experience Buffer:")
                print(f"      - Good: {buffer_stats['good_count']:,}")
                print(f"      - Bad: {buffer_stats['bad_count']:,}")
                print(f"      - Sequences: {buffer_stats['sequence_count']:,}")
                print(f"      - Golden: {buffer_stats['golden_buffer']['size']:,}")

                print(f"   🧠 EBT Performance:")
                print(f"      - EBT success rate: {ebt_success_rate_display:.1%}")
                print(
                    f"      - Energy improvement: {avg_energy_improvement_display:.4f}"
                )

                print(f"   🔧 Training:")
                print(f"      - Thinking success: {thinking_success_rate:.1%}")
                print(f"      - Learning rate: {float(current_lr):.2e}")

                if train_stats:
                    print(f"   📊 Loss Info:")
                    print(f"      - Contrastive: {contrastive_loss_display:.4f}")
                    print(f"      - Energy sep: {energy_separation:.4f}")
                    print(
                        f"      - EBT sequences: {train_stats.get('used_ebt_sequences', False)}"
                    )

                self.logger.info(
                    f"E{self.episode}: Win={win_rate:.1%}, "
                    f"R={float(avg_reward):.2f}, Sep={energy_separation:.4f}, "
                    f"EBT={ebt_success_rate_display:.1%}, "
                    f"EPS/h={float(episodes_per_hour):.1f}"
                )

        self.training_active = False  # Signal render thread to exit
        # Training completed
        training_time = time.time() - training_start_time

        print(f"\n🎉 Optimized Training Completed!")
        print(f"   - Episodes: {max_episodes}")
        print(f"   - Training time: {training_time/3600:.1f} hours")
        print(f"   - Best win rate: {self.best_win_rate:.1%}")
        print(f"   - Avg episodes/hour: {max_episodes / (training_time/3600):.1f}")

        # Save final checkpoint
        final_performance = self.evaluate_performance()
        final_win_rate = final_performance["win_rate"]
        if isinstance(final_win_rate, (list, np.ndarray)):
            final_win_rate = (
                float(final_win_rate[0]) if len(final_win_rate) > 0 else 0.0
            )
        else:
            final_win_rate = float(final_win_rate)

        self.checkpoint_manager.save_checkpoint(
            self.verifier,
            self.agent,
            max_episodes,
            final_win_rate,
            0.0,
            ebt_stats={
                "training_completed": True,
                "final_win_rate": final_win_rate,
                "training_time_hours": training_time / 3600,
            },
        )


def parse_arguments():
    """Enhanced argument parsing with optimized defaults - FINAL FIXED VERSION."""
    parser = argparse.ArgumentParser(
        description="🚀 Optimized EBT Street Fighter Training"
    )

    # Environment and training
    parser.add_argument(
        "--max_episodes", type=int, default=2000, help="Maximum training episodes"
    )
    parser.add_argument(
        "--max_episode_steps",
        type=int,
        default=MAX_FIGHT_STEPS,
        help="Max steps per episode",
    )
    parser.add_argument(
        "--eval_frequency", type=int, default=20, help="Evaluation frequency"
    )
    parser.add_argument(
        "--checkpoint_frequency", type=int, default=50, help="Checkpoint frequency"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=True,
        help="Enable rendering in a separate, non-blocking window.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Model architecture (optimized defaults)
    parser.add_argument(
        "--features_dim", type=int, default=128, help="Feature dimension"
    )
    parser.add_argument(
        "--thinking_steps", type=int, default=2, help="Energy thinking steps"
    )
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension")

    # EBT parameters (optimized)
    parser.add_argument(
        "--use_ebt", action="store_true", default=True, help="Enable EBT"
    )
    parser.add_argument(
        "--use_ebt_thinking",
        action="store_true",
        default=True,
        help="Enable EBT thinking",
    )
    parser.add_argument(
        "--ebt_lr_multiplier",
        type=float,
        default=1.2,
        help="EBT learning rate multiplier",
    )
    parser.add_argument(
        "--ebt_num_heads", type=int, default=4, help="EBT attention heads"
    )
    parser.add_argument(
        "--ebt_num_layers", type=int, default=3, help="EBT transformer layers"
    )

    # Learning parameters (optimized)
    parser.add_argument(
        "--learning_rate", type=float, default=5e-4, help="Learning rate"
    )
    parser.add_argument(
        "--thinking_lr", type=float, default=3e-2, help="Thinking learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--contrastive_margin", type=float, default=1.5, help="Contrastive margin"
    )
    parser.add_argument(
        "--grad_clip", type=float, default=0.5, help="Gradient clipping"
    )

    # Experience buffer (optimized)
    parser.add_argument(
        "--buffer_capacity", type=int, default=20000, help="Buffer capacity"
    )
    parser.add_argument(
        "--golden_buffer_capacity", type=int, default=500, help="Golden buffer capacity"
    )
    parser.add_argument(
        "--quality_threshold", type=float, default=0.45, help="Quality threshold"
    )
    parser.add_argument(
        "--min_buffer_size", type=int, default=500, help="Min buffer size"
    )

    # Energy-based parameters
    parser.add_argument(
        "--noise_scale", type=float, default=0.01, help="Energy noise scale"
    )
    parser.add_argument(
        "--energy_reg", type=float, default=0.005, help="Energy regularization"
    )
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature")

    # Policy memory (optimized)
    parser.add_argument(
        "--performance_drop_threshold",
        type=float,
        default=0.1,
        help="Performance drop threshold",
    )
    parser.add_argument(
        "--averaging_weight", type=float, default=0.7, help="Averaging weight"
    )
    parser.add_argument(
        "--win_rate_window", type=int, default=30, help="Win rate window"
    )
    parser.add_argument(
        "--enable_policy_memory",
        action="store_true",
        default=True,
        help="Enable policy memory",
    )

    # Paths and logging
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints_optimized",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--log_dir", type=str, default="logs_optimized", help="Log directory"
    )
    parser.add_argument(
        "--save_frequency", type=int, default=100, help="Save frequency"
    )
    parser.add_argument("--log_frequency", type=int, default=5, help="Log frequency")

    # Evaluation
    parser.add_argument(
        "--eval_episodes", type=int, default=3, help="Evaluation episodes"
    )
    parser.add_argument(
        "--eval_deterministic",
        action="store_true",
        default=True,
        help="Deterministic eval",
    )

    # Debug and experimental
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--profile", action="store_true", help="Enable profiling")
    parser.add_argument(
        "--mixed_precision",
        action="store_true",
        default=True,
        help="Mixed precision training",
    )

    # Resume training
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )
    parser.add_argument("--load_best", action="store_true", help="Load best checkpoint")

    return parser.parse_args()


def main():
    """Optimized main function with performance monitoring - FINAL FIXED VERSION."""
    # Parse arguments
    args = parse_arguments()

    # FINAL FIX: Explicitly validate and convert arguments to ensure proper types
    print("🔍 FINAL FIX: Validating argument types...")

    # Ensure all integer arguments are proper integers
    args.max_episodes = int(args.max_episodes)
    args.eval_frequency = int(args.eval_frequency)
    args.checkpoint_frequency = int(args.checkpoint_frequency)
    args.batch_size = int(args.batch_size)
    args.thinking_steps = int(args.thinking_steps)
    args.features_dim = int(args.features_dim)

    # Ensure all float arguments are proper floats
    args.learning_rate = float(args.learning_rate)
    args.thinking_lr = float(args.thinking_lr)
    args.quality_threshold = float(args.quality_threshold)
    args.ebt_lr_multiplier = float(args.ebt_lr_multiplier)

    print(f"✅ FINAL FIX: Arguments validated:")
    print(f"   - max_episodes: {args.max_episodes} ({type(args.max_episodes)})")
    print(f"   - eval_frequency: {args.eval_frequency} ({type(args.eval_frequency)})")
    print(
        f"   - checkpoint_frequency: {args.checkpoint_frequency} ({type(args.checkpoint_frequency)})"
    )

    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Set up device and optimizations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")

    if torch.cuda.is_available():
        print(f"🔧 GPU: {torch.cuda.get_device_name()}")
        print(
            f"🔧 CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )

        # Enable optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        if args.mixed_precision:
            print(f"🔧 Mixed precision: ENABLED")

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Configuration display
    print(f"\n🚀 OPTIMIZED STREET FIGHTER RL TRAINING:")
    print(f"   🎮 Environment: Street Fighter II (Optimized)")
    print(
        f"   🎥 Rendering: {'ENABLED (daemon thread)' if args.render else 'DISABLED'}"
    )
    print(f"   🎲 Random Seed: {args.seed}")
    print(f"   🏆 Target Episodes: {args.max_episodes:,}")
    print(f"   🧠 Features Dim: {args.features_dim}")
    print(f"   🔄 Thinking Steps: {args.thinking_steps}")
    print(f"   ⚡ EBT: {'ENABLED' if args.use_ebt else 'DISABLED'}")
    print(f"   🎯 EBT Thinking: {'ENABLED' if args.use_ebt_thinking else 'DISABLED'}")
    print(f"   📚 Learning Rate: {args.learning_rate:.2e}")
    print(f"   📚 EBT LR Multiplier: {args.ebt_lr_multiplier}")
    print(f"   🎲 Batch Size: {args.batch_size}")
    print(f"   📊 Quality Threshold: {args.quality_threshold}")
    print(f"   💾 Buffer Capacity: {args.buffer_capacity:,}")
    print(f"   🏅 Golden Buffer: {args.golden_buffer_capacity:,}")
    print(f"   🎮 Action Space: {len(OPTIMIZED_ACTIONS)} actions")
    print(
        f"   📺 Screen Size: {SCREEN_WIDTH}x{SCREEN_HEIGHT} x {FRAME_STACK_SIZE} frames"
    )
    print(f"   📁 Checkpoint Dir: {args.checkpoint_dir}")

    if args.debug:
        print(f"   🐛 Debug Mode: ENABLED")

    print(f"\n🚀 KEY OPTIMIZATIONS:")
    print(f"   - Asynchronous rendering in a daemon thread (no slowdown)")
    print(f"   - 8-frame stacking for enhanced motion detection")
    print(
        f"   - Re-added Hurricane Kicks to action space: {len(OPTIMIZED_ACTIONS)} actions"
    )
    print(f"   - Dense reward signals for faster learning")
    print(f"   - Mixed precision training for 2x speedup")
    print(f"   - Pre-allocated memory buffers")
    print(f"   - Optimized CNN architecture")
    print(f"   - Enhanced experience quality scoring")
    print(f"   - Fixed type errors and stability issues")
    print(f"   - FINAL FIX: Explicit type validation for all arguments")

    trainer = None
    try:
        # Initialize and run trainer
        trainer = OptimizedEBTTrainer(args)

        # Resume from checkpoint if specified
        if args.resume:
            print(f"📂 Resuming training from: {args.resume}")
            checkpoint_data = trainer.checkpoint_manager.load_checkpoint(
                args.resume, trainer.verifier, trainer.agent
            )
            if checkpoint_data:
                trainer.episode = checkpoint_data.get("episode", 0)
                trainer.best_win_rate = checkpoint_data.get("win_rate", 0.0)
                print(f"✅ Resumed from episode {trainer.episode}")

        # Start training
        print(f"\n🚀 Starting optimized training...")
        start_time = time.time()

        trainer.train()

        total_time = time.time() - start_time
        print(f"\n✅ Training completed successfully!")
        print(f"   - Total time: {total_time/3600:.1f} hours")
        print(f"   - Episodes per hour: {args.max_episodes / (total_time/3600):.1f}")
        print(f"   - Best win rate achieved: {trainer.best_win_rate:.1%}")

    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted by user")
        if trainer:
            trainer.training_active = False  # Signal render thread to stop
        print(f"💾 Checkpoints saved in: {args.checkpoint_dir}")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        if trainer:
            trainer.training_active = False  # Signal render thread to stop
        # --- FIX: Provide full traceback in debug mode ---
        if args.debug:
            traceback.print_exc()
        else:
            print(f"💡 Use --debug flag for detailed error information")
    finally:
        if trainer and trainer.training_active:
            trainer.training_active = False  # Ensure thread is signaled to stop
        # Give the render thread a moment to close
        time.sleep(0.5)
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"🔚 Training session ended")


if __name__ == "__main__":
    main()
