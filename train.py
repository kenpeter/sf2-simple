#!/usr/bin/env python3
"""
train.py - FINAL ENERGY-BASED TRANSFORMER TRAINING FOR STREET FIGHTER
COMPLETE FIXES IMPLEMENTED:
- DiversityExperienceBuffer with skill-level bucketing
- Enhanced contrastive loss with separation and diversity enforcement
- Energy collapse detection and automatic network reset
- Balanced good/bad sampling across training phases
- Emergency restoration protocols
- Comprehensive monitoring and adaptive learning
"""

import os
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from collections import deque
import logging
import time
from pathlib import Path
from tqdm import tqdm
import random

# Import FINAL components from the fixed wrapper.py
from wrapper import (
    EnergyBasedStreetFighterVerifier,
    StabilizedEnergyBasedAgent,
    make_stabilized_env,
    verify_stabilized_energy_flow,
    EnergyStabilityManager,
    DiversityExperienceBuffer,
    CheckpointManager,
    VECTOR_FEATURE_DIM,
    safe_mean,
    safe_std,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def calculate_energy_metrics(training_stats: dict) -> tuple[float, float, float]:
    """Calculate energy landscape quality and separation."""
    positive_energy = training_stats.get("positive_energy", 0.0)
    negative_energy = training_stats.get("negative_energy", 0.0)
    positive_history = training_stats.get("positive_energy_hist", [])
    negative_history = training_stats.get("negative_energy_hist", [])

    # 1. Energy Separation: Higher is better
    energy_separation = negative_energy - positive_energy

    # 2. Energy Quality: A composite score
    quality = 0.0
    if len(positive_history) > 10 and len(negative_history) > 10:
        # Scale separation (target > 1.0 is good)
        separation_score = np.clip(energy_separation, 0, 2.0) * 25  # Max 50 points

        # Consistency (lower std is better)
        pos_consistency = (1.0 / (1.0 + safe_std(positive_history, 1.0))) * 25
        neg_consistency = (1.0 / (1.0 + safe_std(negative_history, 1.0))) * 25
        consistency_score = pos_consistency + neg_consistency  # Max 50 points

        quality = separation_score + consistency_score

    return energy_separation, quality, training_stats.get("energy_loss", 0.0)


class EnhancedEnergyBasedTrainer:
    """
    FINAL: Enhanced Energy-Based Trainer with collapse prevention.
    Integrates with DiversityExperienceBuffer and implements advanced contrastive learning.
    """

    def __init__(
        self,
        verifier: EnergyBasedStreetFighterVerifier,
        experience_buffer: DiversityExperienceBuffer,
        initial_lr: float = 1e-4,
        contrastive_margin: float = 2.0,
        batch_size: int = 32,
        device: str = "auto",
    ):
        self.verifier = verifier
        self.experience_buffer = experience_buffer
        self.contrastive_margin = contrastive_margin
        self.batch_size = batch_size

        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.verifier.to(self.device)

        self.optimizer = optim.Adam(self.verifier.parameters(), lr=initial_lr)

        self.training_stats = {
            "energy_loss": 0.0,
            "positive_energy": 0.0,
            "negative_energy": 0.0,
            "gradient_norm": 0.0,
            "energy_separation": 0.0,
            "diversity_loss": 0.0,
            "separation_loss": 0.0,
            "positive_energy_hist": deque(maxlen=100),
            "negative_energy_hist": deque(maxlen=100),
        }
        print(f"✅ EnhancedEnergyBasedTrainer initialized with Diversity Buffer")

    def train_step(self) -> dict:
        """Perform ENHANCED training step with collapse prevention."""
        # Sample balanced batch from diversity buffer
        good_experiences, bad_experiences = (
            self.experience_buffer.sample_balanced_batch(self.batch_size)
        )

        if good_experiences is None or bad_experiences is None:
            return {"updated": False, "reason": "insufficient_data"}

        # Ensure equal batch sizes for stable contrastive learning
        min_samples = min(len(good_experiences), len(bad_experiences))
        if min_samples < 8:  # Minimum viable batch size
            return {"updated": False, "reason": "batch_too_small"}

        # Balance the datasets
        good_experiences = good_experiences[:min_samples]
        bad_experiences = bad_experiences[:min_samples]

        # Prepare observation batches
        try:
            good_obs = self._prepare_obs_batch(
                [exp["observations"] for exp in good_experiences]
            )
            good_actions = self._prepare_action_batch(
                [exp["action"] for exp in good_experiences]
            )
            bad_obs = self._prepare_obs_batch(
                [exp["observations"] for exp in bad_experiences]
            )
            bad_actions = self._prepare_action_batch(
                [exp["action"] for exp in bad_experiences]
            )
        except Exception as e:
            print(f"⚠️  Batch preparation failed: {e}")
            return {"updated": False, "reason": "batch_preparation_failed"}

        # Calculate energies
        good_energies = self.verifier(good_obs, good_actions)
        bad_energies = self.verifier(bad_obs, bad_actions)

        # ENHANCED contrastive loss with separation and diversity enforcement
        energy_diff = good_energies - bad_energies  # Should be negative (good < bad)
        contrastive_loss = F.relu(
            energy_diff + self.contrastive_margin
        )  # Penalize when good >= bad - margin

        # Separation enforcement: Ensure minimum separation
        mean_separation = bad_energies.mean() - good_energies.mean()
        separation_loss = F.relu(self.contrastive_margin - mean_separation)

        # Diversity enforcement: Prevent all energies from becoming identical
        good_energy_var = torch.var(good_energies)
        bad_energy_var = torch.var(bad_energies)
        diversity_loss = 0.1 * (
            F.relu(0.1 - good_energy_var) + F.relu(0.1 - bad_energy_var)
        )

        # Combined energy loss
        energy_loss = contrastive_loss.mean() + 0.5 * separation_loss + diversity_loss

        # Enhanced regularization
        energy_regularization = 0.01 * (
            good_energies.pow(2).mean() + bad_energies.pow(2).mean()
        )

        total_loss = energy_loss + energy_regularization

        # Enhanced gradient monitoring and clipping
        self.optimizer.zero_grad()
        total_loss.backward()

        # Check for gradient issues before clipping
        total_norm = 0
        param_count = 0
        for p in self.verifier.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        total_norm = total_norm ** (1.0 / 2)

        if total_norm > 10.0 or not np.isfinite(total_norm):
            print(f"🚨 Gradient explosion detected: {total_norm:.3f}, skipping update")
            return {
                "updated": False,
                "reason": "gradient_explosion",
                "energy_loss": energy_loss.item(),
                "positive_energy": good_energies.mean().item(),
                "negative_energy": bad_energies.mean().item(),
                "energy_separation": mean_separation.item(),
            }

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.verifier.parameters(), max_norm=1.0
        )
        self.optimizer.step()

        # Update comprehensive statistics
        positive_energy_val = good_energies.mean().item()
        negative_energy_val = bad_energies.mean().item()
        energy_separation = negative_energy_val - positive_energy_val

        self.training_stats.update(
            {
                "energy_loss": energy_loss.item(),
                "positive_energy": positive_energy_val,
                "negative_energy": negative_energy_val,
                "gradient_norm": gradient_norm.item(),
                "energy_separation": energy_separation,
                "total_norm": total_norm,
                "diversity_loss": diversity_loss.item(),
                "separation_loss": separation_loss.item(),
                "contrastive_loss": contrastive_loss.mean().item(),
            }
        )

        self.training_stats["positive_energy_hist"].append(positive_energy_val)
        self.training_stats["negative_energy_hist"].append(negative_energy_val)

        # Debug output for energy collapse monitoring
        if abs(energy_separation) < 0.01:
            print(f"⚠️  Energy separation very low: {energy_separation:.6f}")
            print(
                f"   Good energy: {positive_energy_val:.6f}, Bad energy: {negative_energy_val:.6f}"
            )
            print(f"   Contrastive loss: {contrastive_loss.mean().item():.6f}")
            print(f"   Separation loss: {separation_loss.item():.6f}")

        return {"updated": True, **self.training_stats}

    def _prepare_obs_batch(self, obs_list: list) -> dict:
        """Prepare a batch of observations for the verifier."""
        try:
            batch_obs = {}
            for key in obs_list[0].keys():
                tensors = []
                for obs in obs_list:
                    if isinstance(obs[key], torch.Tensor):
                        tensors.append(obs[key])
                    else:
                        tensors.append(torch.from_numpy(obs[key]).float())
                batch_obs[key] = torch.stack(tensors).to(self.device)
            return batch_obs
        except Exception as e:
            print(f"⚠️  Error preparing observation batch: {e}")
            raise

    def _prepare_action_batch(self, action_list: list) -> torch.Tensor:
        """Prepare a batch of actions for the verifier."""
        try:
            action_tensors = []
            for action in action_list:
                if isinstance(action, int):
                    # Convert discrete action to one-hot
                    action_onehot = torch.zeros(self.verifier.action_dim)
                    action_onehot[action] = 1.0
                    action_tensors.append(action_onehot)
                elif isinstance(action, torch.Tensor):
                    action_tensors.append(action)
                else:
                    action_tensors.append(torch.from_numpy(action).float())

            return torch.stack(action_tensors).to(self.device)
        except Exception as e:
            print(f"⚠️  Error preparing action batch: {e}")
            raise

    def set_learning_rate(self, new_lr: float):
        """Update the optimizer's learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
        print(f"🔧 Optimizer LR updated to: {new_lr:.2e}")

    def check_energy_collapse(self) -> bool:
        """Check if energy landscape has collapsed."""
        if len(self.training_stats["positive_energy_hist"]) < 10:
            return False

        recent_pos = list(self.training_stats["positive_energy_hist"])[-10:]
        recent_neg = list(self.training_stats["negative_energy_hist"])[-10:]

        if len(recent_pos) == 0 or len(recent_neg) == 0:
            return False

        avg_separation = np.mean([n - p for n, p in zip(recent_neg, recent_pos)])
        energy_variance = np.var(recent_pos + recent_neg)

        # Collapse indicators
        separation_collapsed = abs(avg_separation) < 0.01
        variance_collapsed = energy_variance < 1e-6

        if separation_collapsed or variance_collapsed:
            print(f"🚨 ENERGY COLLAPSE DETECTED:")
            print(f"   Separation: {avg_separation:.6f}")
            print(f"   Variance: {energy_variance:.6f}")
            return True

        return False

    def reset_energy_network(self):
        """🚨 EMERGENCY: Reset the energy network to break collapse."""
        print("🚨 RESETTING ENERGY NETWORK - BREAKING ENERGY COLLAPSE")

        # Re-initialize the energy network part of the verifier
        for name, module in self.verifier.named_modules():
            if "energy_net" in name and isinstance(module, torch.nn.Linear):
                if hasattr(module, "weight"):
                    torch.nn.init.xavier_uniform_(module.weight, gain=0.01)
                    print(f"   Reset {name}")
                if hasattr(module, "bias") and module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)

        # Clear energy history to reset statistics
        self.training_stats["positive_energy_hist"].clear()
        self.training_stats["negative_energy_hist"].clear()

        # Reset optimizer state for energy network
        energy_params = []
        for name, param in self.verifier.named_parameters():
            if "energy_net" in name:
                energy_params.append(param)

        if energy_params:
            # Create a new optimizer state for these parameters
            for group in self.optimizer.param_groups:
                for p in group["params"]:
                    if p in energy_params and p in self.optimizer.state:
                        del self.optimizer.state[p]

        print("   ✅ Energy network reset complete")


def run_enhanced_training_episode(
    env,
    agent: StabilizedEnergyBasedAgent,
    trainer: EnhancedEnergyBasedTrainer,
    max_steps: int = 18000,
) -> dict:
    """Run enhanced training episode with diversity-based experience collection."""
    obs, info = env.reset()
    total_reward = 0.0
    episode_length = 0
    done, truncated = False, False

    episode_thinking_stats = []
    episode_train_stats = []
    step_count = 0

    for step in range(max_steps):
        # Convert observations to tensors for the agent
        obs_tensor = {k: torch.from_numpy(v).float() for k, v in obs.items()}

        # Agent predicts action
        action, thinking_info = agent.predict(obs_tensor, deterministic=False)
        episode_thinking_stats.append(thinking_info)

        # Environment step
        next_obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        episode_length += 1
        step_count += 1

        obs = next_obs
        if done or truncated:
            break

    # Training is handled automatically by the environment's episode callback
    # which processes experiences through the diversity buffer

    # Multiple training steps per episode for better learning
    training_updates = 0
    for _ in range(
        min(10, max(1, episode_length // 100))
    ):  # Adaptive training frequency
        train_result = trainer.train_step()
        if train_result.get("updated"):
            episode_train_stats.append(train_result)
            training_updates += 1

    # Aggregate stats for the episode
    win = info.get("wins", 0) > 0

    final_stats = {
        "total_reward": total_reward,
        "episode_length": episode_length,
        "win": win,
        "thinking_stats": agent.get_thinking_stats(),
        "training_stats": (
            trainer.training_stats
            if not episode_train_stats
            else episode_train_stats[-1]
        ),
        "final_info": info,
        "training_updates": training_updates,
        "buffer_stats": trainer.experience_buffer.get_stats(),
    }
    return final_stats


def main():
    parser = argparse.ArgumentParser(
        description="FINAL EBT Training for Street Fighter with Collapse Prevention"
    )
    parser.add_argument("--total-episodes", type=int, default=50000)
    parser.add_argument(
        "--save-freq",
        type=int,
        default=50,
        help="Save best model checkpoint every X episodes",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--device", type=str, default="auto")

    parser.add_argument(
        "--lr",
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Initial learning rate",
    )
    parser.add_argument(
        "--thinking-lr", type=float, default=0.1, help="Initial thinking learning rate"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--contrastive-margin", type=float, default=2.0)
    parser.add_argument("--thinking-steps", type=int, default=3)
    args = parser.parse_args()

    print("🛡️ FINAL ENERGY-BASED TRANSFORMER TRAINING 🛡️")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  - Initial LR: {args.lr}")
    print(f"  - Initial Thinking LR: {args.thinking_lr}")
    print(f"  - Contrastive Margin: {args.contrastive_margin}")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - Feature System: Final ({VECTOR_FEATURE_DIM} dims)")
    print(f"  - Energy Collapse Prevention: ✅ ACTIVE")
    print(f"  - Diversity Buffer: ✅ ACTIVE")
    print("=" * 60)

    # Setup device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.device == "auto" else "cpu"
    )
    print(f"🔧 Using device: {device}")

    # Initialize Environment
    render_mode = "human" if args.render else None
    env = make_stabilized_env(render_mode=render_mode)

    # Initialize Core Components
    verifier = EnergyBasedStreetFighterVerifier(env.observation_space, env.action_space)
    agent = StabilizedEnergyBasedAgent(
        verifier, thinking_steps=args.thinking_steps, thinking_lr=args.thinking_lr
    )

    # Initialize Enhanced Systems
    stability_manager = EnergyStabilityManager(
        initial_lr=args.lr, thinking_lr=args.thinking_lr
    )

    # ENHANCED: Diversity Experience Buffer
    experience_buffer = DiversityExperienceBuffer(capacity=30000, quality_threshold=0.5)

    checkpoint_manager = CheckpointManager(checkpoint_dir="checkpoints")

    # ENHANCED: Trainer with collapse prevention
    trainer = EnhancedEnergyBasedTrainer(
        verifier,
        experience_buffer,
        initial_lr=args.lr,
        contrastive_margin=args.contrastive_margin,
        batch_size=args.batch_size,
        device=device,
    )

    # Set up environment callback for diversity buffer
    def episode_callback(episode_data, win_rate):
        """Callback to process episode data for diversity buffer."""
        experience_buffer.add_episode_experiences(episode_data, win_rate)

    env.set_episode_callback(episode_callback)

    # Resume from checkpoint if provided
    start_episode = 0
    if args.resume and os.path.exists(args.resume):
        print(f"📂 Resuming from checkpoint: {args.resume}")
        data = checkpoint_manager.load_checkpoint(Path(args.resume), verifier, agent)
        if data:
            start_episode = data.get("episode", 0)
            print(f"   ✅ Resumed from episode {start_episode}")
        else:
            print(f"   ❌ Failed to load checkpoint. Starting fresh.")

    # Verify energy flow
    print("\n🔬 Verifying energy flow...")
    if not verify_stabilized_energy_flow(verifier, env, device):
        print("🚨 CRITICAL: Energy flow verification failed. Exiting.")
        return
    print("   ✅ Energy flow is STABLE.")

    # Main Training Loop
    print(f"\n🚀 Starting FINAL Energy-Based Training...")
    print(f"🎯 Target: Prevent energy collapse through diversity and monitoring")

    pbar = tqdm(
        range(start_episode, args.total_episodes),
        initial=start_episode,
        total=args.total_episodes,
        desc="Episodes",
    )

    for episode in pbar:
        # Run enhanced training episode
        episode_data = run_enhanced_training_episode(env, agent, trainer)

        # Extract and calculate metrics
        win_rate = safe_mean(list(stability_manager.win_rate_window), 0.5)
        early_stop_rate = episode_data["thinking_stats"].get("early_stop_rate", 0.0)
        energy_separation, energy_quality, energy_loss = calculate_energy_metrics(
            episode_data["training_stats"]
        )

        # Update stability manager and check for collapse
        is_emergency = stability_manager.update_metrics(
            win_rate=episode_data["win"],
            energy_quality=energy_quality,
            energy_separation=energy_separation,
            early_stop_rate=early_stop_rate,
        )

        # 🚨 Enhanced Emergency Reset Protocol 🚨
        if is_emergency:
            print("🚨 EMERGENCY PROTOCOL TRIGGERED!")

            # Check for severe energy collapse first
            if trainer.check_energy_collapse():
                print("🚨 SEVERE ENERGY COLLAPSE - RESETTING ENERGY NETWORK")
                trainer.reset_energy_network()
                experience_buffer.emergency_purge(keep_ratio=0.1)  # Aggressive purge
            else:
                # Try to restore best model first
                restored_data = checkpoint_manager.emergency_restore(verifier, agent)
                if restored_data:
                    print(
                        f"   ✅ Model restored from episode {restored_data['episode']}"
                    )
                else:
                    print(
                        "   ❌ No checkpoint to restore from. Resetting energy network."
                    )
                    trainer.reset_energy_network()

                # Purge experience buffer
                experience_buffer.emergency_purge(keep_ratio=0.2)

            # Apply new learning rates
            new_lr, new_thinking_lr = stability_manager.get_current_lrs()
            trainer.set_learning_rate(new_lr)
            agent.current_thinking_lr = new_thinking_lr

        # Checkpoint saving logic
        if episode % args.save_freq == 0:
            current_win_rate = safe_mean(list(stability_manager.win_rate_window), 0.0)
            if stability_manager.should_save_checkpoint(current_win_rate):
                checkpoint_manager.save_checkpoint(
                    verifier, agent, episode, current_win_rate, energy_quality
                )

        # Check for recovery
        stability_manager.recovery_check(
            safe_mean(list(stability_manager.win_rate_window), 0.0)
        )

        # Enhanced progress bar with more metrics
        buffer_stats = experience_buffer.get_stats()
        energy_sep = energy_separation if not np.isnan(energy_separation) else 0.0

        pbar.set_postfix(
            {
                "WR": f"{win_rate:.2f}",
                "EQ": f"{energy_quality:.1f}",
                "Sep": f"{energy_sep:.3f}",
                "Loss": f"{energy_loss:.3f}",
                "LR": f"{trainer.optimizer.param_groups[0]['lr']:.1e}",
                "Buf": f"{buffer_stats['total_size']}",
                "G/B": f"{buffer_stats['good_count']}/{buffer_stats['bad_count']}",
                "Mode": "EMERGENCY" if stability_manager.emergency_mode else "Normal",
            }
        )

        # Detailed logging every 100 episodes
        if episode % 100 == 0 and episode > 0:
            print(f"\n📊 Episode {episode} Detailed Stats:")
            print(f"   🏆 Win Rate: {win_rate:.3f}")
            print(f"   ⚡ Energy Separation: {energy_sep:.3f}")
            print(f"   📈 Energy Quality: {energy_quality:.1f}")
            print(
                f"   🎯 Buffer Balance: {buffer_stats['good_count']}/{buffer_stats['bad_count']} (good/bad)"
            )
            print(
                f"   🧠 Thinking Success Rate: {episode_data['thinking_stats'].get('success_rate', 0.0):.3f}"
            )
            print(f"   🔄 Training Updates: {episode_data.get('training_updates', 0)}")

            # Buffer distribution
            for skill in ["beginner", "intermediate", "advanced"]:
                good_count = buffer_stats.get(f"{skill}_good", 0)
                bad_count = buffer_stats.get(f"{skill}_bad", 0)
                if good_count > 0 or bad_count > 0:
                    print(
                        f"   📚 {skill.capitalize()}: {good_count}/{bad_count} (good/bad)"
                    )

    print("\n🎉 FINAL Training completed!")
    print(f"💾 Checkpoints saved in 'checkpoints' directory")
    print(f"📊 Final buffer stats: {experience_buffer.get_stats()}")
    env.close()


if __name__ == "__main__":
    main()
