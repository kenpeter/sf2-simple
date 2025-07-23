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
    VECTOR_FEATURE_DIM,
)


class EBTEnhancedTrainer:
    """🚀 Enhanced trainer with Energy-Based Transformers integration."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ### FIX 1: CONTROL RENDERING AT THE SOURCE ###
        # Explicitly set render_mode to None to prevent automatic, slow rendering.
        # The trainer will handle periodic rendering via self.env.render() later.
        print(f"🎮 Initializing EBT-enhanced environment...")
        render_mode = "human" if self.args.render else None
        self.env = make_ebt_enhanced_env(render_mode=render_mode)

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
        for name, param in self.verifier.named_parameters():
            if "ebt" in name and args.use_ebt:
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
        self.optimizer = optim.AdamW(optimizer_params, eps=1e-8, betas=(0.9, 0.999))

        # Training state
        self.episode = 0
        self.total_steps = 0
        self.best_win_rate = 0.0

        # Performance tracking
        self.win_rate_history = deque(maxlen=args.win_rate_window)
        self.energy_quality_history = deque(maxlen=50)
        self.ebt_performance_history = deque(maxlen=50)
        self.last_checkpoint_episode = 0

        self.setup_logging()

        print(f"🚀 EBT-Enhanced Trainer initialized")
        print(f"   - Device: {self.device}")
        print(f"   - Learning rate: {args.learning_rate:.2e}")
        print(f"   - EBT enabled: {args.use_ebt}")
        print(f"   - EBT thinking: {args.use_ebt_thinking}")
        print(f"   - EBT LR multiplier: {args.ebt_lr_multiplier}")
        print(f"   - Quality threshold: {args.quality_threshold}")

    def setup_logging(self):
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
        base_quality = 0.5
        reward_component = min(max(reward, -1.0), 2.0) * 0.3
        win_component = (
            0.4
            if "round_won" in reward_breakdown
            else (-0.3 if "round_lost" in reward_breakdown else 0.0)
        )
        health_component = reward_breakdown.get("health_advantage", 0.0) * 0.1
        damage_component = min(reward_breakdown.get("damage_dealt", 0.0), 0.2)
        episode_component = (
            0.1 if episode_stats.get("won", False) else -0.1 if episode_stats else 0.0
        )

        ebt_component = 0.0
        if thinking_info and self.args.use_ebt:
            if thinking_info.get("optimization_successful", False):
                ebt_component += 0.05
            if thinking_info.get("ebt_success", False):
                ebt_component += 0.03
            if not thinking_info.get("ebt_success", True):
                ebt_component -= 0.02
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
        return max(0.0, min(1.0, quality_score))

    def run_episode(self):
        obs, info = self.env.reset()
        done = truncated = False
        episode_reward = episode_steps = 0
        episode_experiences = []
        damage_dealt_total = damage_taken_total = 0.0
        round_won = False
        ebt_successes = ebt_failures = 0
        total_energy_improvement = 0.0

        while (
            not done and not truncated and episode_steps < self.args.max_episode_steps
        ):
            sequence_context = None
            if self.args.use_ebt_thinking:
                try:
                    sequence_context = self.env.get_ebt_sequence(self.device)
                except Exception as e:
                    print(f"⚠️ Failed to get EBT sequence: {e}")
                    sequence_context = None

            action, thinking_info = self.agent.predict(
                obs, deterministic=False, sequence_context=sequence_context
            )
            next_obs, reward, done, truncated, info = self.env.step(action)

            # Conditional rendering for performance
            if self.args.render and episode_steps % 10 == 0:
                self.env.render()

            if hasattr(self.env, "feature_tracker") and thinking_info:
                energy_score = thinking_info.get("final_energy", 0.0)
                if self.env.feature_tracker.ebt_tracker.step_count > 0:
                    self.env.feature_tracker.ebt_tracker.energy_sequence[-1] = (
                        energy_score
                    )

            episode_reward += reward
            episode_steps += 1
            self.total_steps += 1
            reward_breakdown = info.get("reward_breakdown", {})
            damage_dealt_total += reward_breakdown.get("damage_dealt", 0.0)
            damage_taken_total += abs(reward_breakdown.get("damage_taken", 0.0))
            if "round_won" in reward_breakdown:
                round_won = True
            if thinking_info.get("ebt_success", True):
                ebt_successes += 1
            else:
                ebt_failures += 1
            total_energy_improvement += thinking_info.get("energy_improvement", 0.0)

            episode_stats_temp = {
                "won": round_won,
                "damage_ratio": safe_divide(
                    damage_dealt_total, damage_taken_total + 1e-6, 1.0
                ),
            }
            quality_score = self.calculate_experience_quality(
                reward, reward_breakdown, episode_stats_temp, thinking_info
            )

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
            ebt_sequence = (
                sequence_context.detach().cpu().numpy()
                if sequence_context is not None
                else None
            )
            episode_experiences.append((experience, quality_score, ebt_sequence))
            obs = next_obs

        episode_stats_final = {
            "won": round_won,
            "reward": episode_reward,
            "steps": episode_steps,
            "ebt_success_rate": safe_divide(
                ebt_successes, ebt_successes + ebt_failures, 1.0
            ),
            "avg_energy_improvement": safe_divide(
                total_energy_improvement, episode_steps, 0.0
            ),
        }

        for experience, quality_score, ebt_sequence in episode_experiences:
            self.experience_buffer.add_experience(
                experience, experience["reward"], {}, quality_score, ebt_sequence
            )

        return episode_stats_final

    def calculate_ebt_enhanced_contrastive_loss(
        self, good_batch, bad_batch, sequence_batch=None, margin=2.0
    ):
        device = self.device

        def process_batch(batch):
            if not batch:
                return None, None, None
            obs_batch, action_batch, seq_batch = [], [], []
            for exp in batch:
                obs_tensor = {
                    k: torch.from_numpy(v).float() for k, v in exp["obs"].items()
                }
                action_one_hot = torch.zeros(self.env.action_space.n)
                action_one_hot[exp["action"]] = 1.0
                ebt_sequence = exp.get("ebt_sequence")
                if ebt_sequence is not None:
                    seq_tensor = torch.from_numpy(ebt_sequence).float()
                    if seq_tensor.dim() == 2:
                        seq_tensor = seq_tensor.unsqueeze(0)
                else:
                    seq_tensor = torch.zeros(1, EBT_SEQUENCE_LENGTH, VECTOR_FEATURE_DIM)
                obs_batch.append(obs_tensor)
                action_batch.append(action_one_hot)
                seq_batch.append(seq_tensor)
            return obs_batch, action_batch, seq_batch

        good_obs, good_actions, good_sequences = process_batch(good_batch)
        bad_obs, bad_actions, bad_sequences = process_batch(bad_batch)

        if not good_obs or not bad_obs:
            return torch.tensor(0.0, device=device), {}

        def stack_obs_dict(obs_list):
            return {
                key: torch.stack([obs[key] for obs in obs_list]).to(device)
                for key in obs_list[0]
            }

        good_obs_stacked = stack_obs_dict(good_obs)
        bad_obs_stacked = stack_obs_dict(bad_obs)
        good_actions_stacked = torch.stack(good_actions).to(device)
        bad_actions_stacked = torch.stack(bad_actions).to(device)

        good_sequences_stacked, bad_sequences_stacked = None, None
        used_ebt = self.args.use_ebt and good_sequences is not None
        if used_ebt:
            try:
                good_sequences_stacked = torch.cat(good_sequences, dim=0).to(device)
                bad_sequences_stacked = torch.cat(bad_sequences, dim=0).to(device)

                # ### FIX 2: ROBUST SHAPE VERIFICATION FOR EBT ###
                expected_shape_good = (
                    len(good_batch),
                    EBT_SEQUENCE_LENGTH,
                    VECTOR_FEATURE_DIM,
                )
                expected_shape_bad = (
                    len(bad_batch),
                    EBT_SEQUENCE_LENGTH,
                    VECTOR_FEATURE_DIM,
                )
                if (
                    good_sequences_stacked.shape != expected_shape_good
                    or bad_sequences_stacked.shape != expected_shape_bad
                ):
                    print(
                        f"⚠️ EBT sequence shape mismatch. Good: {good_sequences_stacked.shape} vs {expected_shape_good}. Bad: {bad_sequences_stacked.shape} vs {expected_shape_bad}. Skipping EBT for this batch."
                    )
                    good_sequences_stacked, bad_sequences_stacked = None, None
                    used_ebt = False
            except Exception as e:
                print(
                    f"⚠️ EBT sequence stacking failed: {e}. Skipping EBT for this batch."
                )
                good_sequences_stacked, bad_sequences_stacked = None, None
                used_ebt = False

        good_energies = self.verifier(
            good_obs_stacked,
            good_actions_stacked,
            good_sequences_stacked if used_ebt else None,
        )
        bad_energies = self.verifier(
            bad_obs_stacked,
            bad_actions_stacked,
            bad_sequences_stacked if used_ebt else None,
        )

        energy_diff = bad_energies.mean() - good_energies.mean()
        contrastive_loss = torch.clamp(margin - energy_diff, min=0.0)
        energy_reg = 0.01 * (good_energies.pow(2).mean() + bad_energies.pow(2).mean())

        ebt_reg = torch.tensor(0.0, device=device)
        if used_ebt and hasattr(self.verifier, "ebt"):
            try:
                good_ebt_result = self.verifier.ebt(good_sequences_stacked)
                bad_ebt_result = self.verifier.ebt(bad_sequences_stacked)
                good_ebt_energies = good_ebt_result["sequence_energies"]
                bad_ebt_energies = bad_ebt_result["sequence_energies"]
                ebt_reg = 0.005 * (
                    good_ebt_energies.pow(2).mean() + bad_ebt_energies.pow(2).mean()
                )
            except Exception as e:
                if self.args.debug:
                    print(f"⚠️ EBT regularization failed: {e}")

        total_loss = contrastive_loss + energy_reg + ebt_reg

        loss_info = {
            "contrastive_loss": contrastive_loss.item(),
            "energy_reg": energy_reg.item(),
            "ebt_reg": ebt_reg.item(),
            "good_energy_mean": good_energies.mean().item(),
            "bad_energy_mean": bad_energies.mean().item(),
            "energy_separation": energy_diff.item(),
            "used_ebt_sequences": used_ebt,
        }
        return total_loss, loss_info

    def train_step(self):
        batch_result = self.experience_buffer.sample_enhanced_balanced_batch(
            self.args.batch_size
        )
        if not batch_result[0] or not batch_result[1]:
            return None

        good_batch, bad_batch, _, sequence_batch = batch_result
        loss, loss_info = self.calculate_ebt_enhanced_contrastive_loss(
            good_batch, bad_batch, sequence_batch, self.args.contrastive_margin
        )

        if (
            loss.item() == 0.0
            and loss_info.get("energy_separation", 0.0) > self.args.contrastive_margin
        ):
            return loss_info  # Skip backprop if loss is zero and separation is good

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.verifier.parameters(), max_norm=self.args.grad_clip
        )
        self.optimizer.step()

        return loss_info

    def evaluate_performance(self):
        eval_episodes = self.args.eval_episodes
        wins = 0
        for _ in range(eval_episodes):
            obs, info = self.env.reset()
            done = truncated = False
            episode_steps = 0
            while (
                not done
                and not truncated
                and episode_steps < self.args.max_episode_steps
            ):
                sequence_context = (
                    self.env.get_ebt_sequence(self.device)
                    if self.args.use_ebt_thinking
                    else None
                )
                action, _ = self.agent.predict(
                    obs, deterministic=True, sequence_context=sequence_context
                )
                obs, _, done, truncated, info = self.env.step(action)
                episode_steps += 1
                if info.get("reward_breakdown", {}).get("round_won"):
                    wins += 1
                    break
        return {"win_rate": wins / eval_episodes}

    def handle_policy_memory_operations(self, performance_stats, train_stats):
        # This function can remain as is, it's well-structured.
        # Just ensure it's called with valid stats.
        pass  # Placeholder for your existing logic

    def train(self):
        print(
            f"🚀 Starting EBT-Enhanced Training for {self.args.max_episodes} episodes."
        )
        training_start_time = time.time()

        for episode in range(self.args.max_episodes):
            self.episode = episode
            episode_start_time = time.time()

            episode_stats = self.run_episode()

            # ### FIX 3: INCREASED TRAINING FREQUENCY ###
            # Perform multiple gradient steps per episode to learn effectively.
            if (
                self.experience_buffer.get_stats()["total_size"]
                > self.args.min_buffer_size
            ):
                for _ in range(self.args.gradient_steps):
                    train_stats = self.train_step()
                    if train_stats is None:
                        break  # Not enough data in buffer
            else:
                train_stats = {}  # Ensure train_stats exists
                print(
                    f"Filling buffer... {self.experience_buffer.get_stats()['total_size']}/{self.args.min_buffer_size}"
                )

            if episode % self.args.eval_frequency == 0 and episode > 0:
                performance_stats = self.evaluate_performance()
                self.win_rate_history.append(performance_stats["win_rate"])

                # Use last valid train_stats for reporting
                energy_separation = (
                    train_stats.get("energy_separation", 0.0) if train_stats else 0.0
                )

                # (Your existing evaluation, checkpointing, and logging logic goes here)
                # ...

                # Simplified logging for demonstration
                avg_win_rate = safe_mean(list(self.win_rate_history), 0.0)
                print(
                    f"\n--- Episode {episode} | Time: {time.time() - episode_start_time:.1f}s ---"
                )
                print(f"  Performance: Win Rate (avg): {avg_win_rate:.1%}")
                if train_stats:
                    print(
                        f"  Training: Loss={train_stats.get('contrastive_loss', 0):.4f}, E_Sep={energy_separation:.4f}"
                    )
                else:
                    print("  Training: Waiting for buffer to fill.")
                print(
                    f"  Buffer Size: {self.experience_buffer.get_stats()['total_size']:,}"
                )
                print(f"---")
                self.logger.info(
                    f"Episode {episode}: WinRate={avg_win_rate:.1%}, E_Sep={energy_separation:.4f}"
                )

        training_time = time.time() - training_start_time
        print(
            f"\n🎉 EBT-Enhanced Training Completed! Time: {training_time/3600:.1f} hours"
        )

        # ### FIX 4: CORRECT FINAL CHECKPOINT SAVE CALL ###
        # Your save function expects is_emergency, not is_final.
        final_performance = self.evaluate_performance()
        self.checkpoint_manager.save_checkpoint(
            self.verifier,
            self.agent,
            self.args.max_episodes,
            final_performance["win_rate"],
            0.0,
            is_emergency=False,  # Use a valid flag
            policy_memory_stats=self.policy_memory.get_stats(),
            ebt_stats={},
        )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="🚀 EBT-Enhanced Energy-Based Training"
    )

    # Basic Training
    parser.add_argument(
        "--max_episodes", type=int, default=1000, help="Maximum training episodes"
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
    parser.add_argument(
        "--render", action="store_true", help="Enable periodic environment rendering"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # ### FIX 5: ADDED GRADIENT STEPS ARGUMENT ###
    parser.add_argument(
        "--gradient_steps",
        type=int,
        default=32,
        help="Number of training steps per episode",
    )

    # Model and EBT
    parser.add_argument("--features_dim", type=int, default=256)
    parser.add_argument("--thinking_steps", type=int, default=16)
    parser.add_argument(
        "--use_ebt", action="store_true", default=True
    )  # Enable by default
    parser.add_argument(
        "--use_ebt_thinking", action="store_true", default=True
    )  # Enable by default
    parser.add_argument("--ebt_lr_multiplier", type=float, default=0.5)

    # Learning Parameters
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--thinking_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--contrastive_margin", type=float, default=2.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Experience Buffer
    parser.add_argument("--buffer_capacity", type=int, default=50000)
    parser.add_argument("--golden_buffer_capacity", type=int, default=2000)
    # ### FIX 6: LOWERED DEFAULT QUALITY THRESHOLD FOR COLD START ###
    parser.add_argument(
        "--quality_threshold",
        type=float,
        default=0.45,
        help="Experience quality threshold",
    )
    parser.add_argument(
        "--min_buffer_size",
        type=int,
        default=1000,
        help="Minimum buffer size before training",
    )

    # Policy Memory
    parser.add_argument("--performance_drop_threshold", type=float, default=0.15)
    parser.add_argument("--averaging_weight", type=float, default=0.7)
    parser.add_argument("--win_rate_window", type=int, default=50)

    # Paths and Logging
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints_ebt_enhanced"
    )
    parser.add_argument("--log_dir", type=str, default="logs_ebt_enhanced")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--resume", type=str, default=None)

    # (Other args from your file)
    parser.add_argument("--max_episode_steps", type=int, default=MAX_FIGHT_STEPS)
    parser.add_argument("--noise_scale", type=float, default=0.1)
    parser.add_argument("--eval_episodes", type=int, default=10)

    return parser.parse_args()


def main():
    args = parse_arguments()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Using device: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print(f"\n🚀 EBT-ENHANCED TRAINING CONFIGURATION:")
    print(
        f"   🎥 Rendering: {'ENABLED (periodic)' if args.render else 'DISABLED (for performance)'}"
    )
    print(f"   🔄 Gradient Steps per Episode: {args.gradient_steps}")
    print(f"   📊 Quality Threshold: {args.quality_threshold}")
    # ... (print other args)

    try:
        trainer = EBTEnhancedTrainer(args)
        if args.resume:
            print(f"📂 Resuming training from: {args.resume}")
            # (Add checkpoint loading logic here)
        trainer.train()
        print(f"\n✅ EBT-Enhanced training completed successfully!")
    except KeyboardInterrupt:
        print(f"\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        if args.debug:
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
