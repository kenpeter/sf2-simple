#!/usr/bin/env python3
"""
train.py - STABILIZED ENERGY-BASED TRANSFORMER TRAINING FOR STREET FIGHTER
ENHANCED WITH ZIP SAVING FUNCTIONALITY
FIXES IMPLEMENTED:
- Integrates EnergyStabilityManager to prevent landscape collapse.
- Implements Emergency Reset Protocol (restore best model, purge buffer).
- Uses quality-controlled ExperienceBuffer.
- Employs CheckpointManager for robust model saving and restoration.
- ADDED: Periodic ZIP file creation containing wrapper.py and train.py
- Provides detailed, actionable logging for monitoring training stability.
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
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm

# Import STABILIZED components from the fixed wrapper.py
from wrapper import (
    EnergyBasedStreetFighterVerifier,
    StabilizedEnergyBasedAgent,
    make_stabilized_env,
    verify_stabilized_energy_flow,
    EnergyStabilityManager,
    ExperienceBuffer,
    CheckpointManager,
    VECTOR_FEATURE_DIM,
    ENHANCED_VECTOR_FEATURE_DIM,
    BAIT_PUNISH_AVAILABLE,
    safe_mean,
    safe_std,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def create_zip_package(output_dir="zip_saves", episode=None, win_rate=None):
    """
    Create a ZIP package containing wrapper.py and train.py files.

    Args:
        output_dir: Directory to save ZIP files
        episode: Current episode number (optional)
        win_rate: Current win rate (optional)

    Returns:
        Path to created ZIP file or None if failed
    """
    try:
        # Create output directory
        zip_dir = Path(output_dir)
        zip_dir.mkdir(exist_ok=True)

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if episode is not None and win_rate is not None:
            zip_filename = f"ebt_training_ep{episode}_wr{win_rate:.3f}_{timestamp}.zip"
        else:
            zip_filename = f"ebt_training_{timestamp}.zip"

        zip_path = zip_dir / zip_filename

        # Create ZIP file
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            # Add wrapper.py if it exists
            if Path("wrapper.py").exists():
                zipf.write("wrapper.py", "wrapper.py")
                print(f"   📄 Added wrapper.py to ZIP")
            else:
                print(f"   ⚠️  wrapper.py not found, skipping")

            # Add train.py if it exists
            if Path("train.py").exists():
                zipf.write("train.py", "train.py")
                print(f"   📄 Added train.py to ZIP")
            else:
                print(f"   ⚠️  train.py not found, skipping")

            # Add any additional Python files that might be relevant
            for py_file in ["bait_punish_system.py", "utils.py"]:
                if Path(py_file).exists():
                    zipf.write(py_file, py_file)
                    print(f"   📄 Added {py_file} to ZIP")

            # Add a README with training info
            readme_content = f"""# Stabilized Energy-Based Transformer Training Package

## Training Session Information
- Timestamp: {timestamp}
- Episode: {episode if episode is not None else 'N/A'}
- Win Rate: {win_rate:.3f if win_rate is not None else 'N/A'}
- Feature Dimension: {VECTOR_FEATURE_DIM}
- Bait-Punish Available: {BAIT_PUNISH_AVAILABLE}

## Files Included
- wrapper.py: Complete EBT implementation with stability controls
- train.py: Training script with emergency protocols
- Additional support files (if present)

## Usage
1. Ensure you have all required dependencies installed
2. Run: python train.py --total-episodes 50000
3. Monitor training stability through the detailed logging output

## Key Features
- 🛡️ Energy landscape collapse prevention
- 🎯 Quality-controlled experience buffer
- 🚨 Emergency reset protocols
- 📊 Integrated oscillation tracking
- 💾 Automatic checkpoint management
"""

            zipf.writestr("README.md", readme_content)
            print(f"   📄 Added README.md to ZIP")

        file_size = zip_path.stat().st_size / (1024 * 1024)  # Size in MB
        print(f"💾 ZIP package created: {zip_filename} ({file_size:.2f} MB)")

        return zip_path

    except Exception as e:
        print(f"❌ Failed to create ZIP package: {e}")
        return None


def cleanup_old_zips(zip_dir="zip_saves", keep_count=5):
    """
    Clean up old ZIP files, keeping only the most recent ones.

    Args:
        zip_dir: Directory containing ZIP files
        keep_count: Number of recent ZIP files to keep
    """
    try:
        zip_path = Path(zip_dir)
        if not zip_path.exists():
            return

        # Get all ZIP files sorted by modification time (newest first)
        zip_files = sorted(
            zip_path.glob("*.zip"), key=lambda x: x.stat().st_mtime, reverse=True
        )

        # Remove old files
        for old_zip in zip_files[keep_count:]:
            old_zip.unlink()
            print(f"🗑️  Removed old ZIP: {old_zip.name}")

        print(f"📁 Kept {min(len(zip_files), keep_count)} most recent ZIP files")

    except Exception as e:
        print(f"⚠️  Error cleaning up old ZIPs: {e}")


def calculate_energy_metrics(training_stats: dict) -> tuple[float, float, float]:
    """Calculate energy landscape quality and separation."""
    positive_energy = training_stats.get("positive_energy", 0.0)
    negative_energy = training_stats.get("negative_energy", 0.0)
    positive_history = training_stats.get("positive_energy_hist", [])
    negative_history = training_stats.get("negative_energy_hist", [])

    # 1. Energy Separation: Higher is better.
    energy_separation = negative_energy - positive_energy

    # 2. Energy Quality: A composite score.
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


class EnergyBasedTrainer:
    """
    Manages the contrastive training of the Energy-Based Verifier.
    Integrates with the quality-controlled ExperienceBuffer.
    """

    def __init__(
        self,
        verifier: EnergyBasedStreetFighterVerifier,
        experience_buffer: ExperienceBuffer,
        initial_lr: float = 1e-4,
        contrastive_margin: float = 1.0,
        batch_size: int = 32,
        device: str = "auto",
    ):
        self.verifier = verifier
        self.experience_buffer = experience_buffer
        self.contrastive_margin = contrastive_margin
        self.batch_size = batch_size
        self.device = device
        self.verifier.to(self.device)

        self.optimizer = optim.Adam(self.verifier.parameters(), lr=initial_lr)

        self.training_stats = {
            "energy_loss": 0.0,
            "positive_energy": 0.0,
            "negative_energy": 0.0,
            "gradient_norm": 0.0,
            "positive_energy_hist": deque(maxlen=100),
            "negative_energy_hist": deque(maxlen=100),
        }
        print(f"✅ EnergyBasedTrainer initialized with Quality-Controlled Buffer")

    def train_step(self) -> dict:
        """Perform one training step using contrastive energy learning."""
        if self.experience_buffer.get_stats()["size"] < self.batch_size:
            return {"updated": False}

        batch, qualities = self.experience_buffer.sample_batch(
            self.batch_size, prioritize_quality=True
        )

        good_experiences = [exp for exp, q in zip(batch, qualities) if exp["is_good"]]
        bad_experiences = [
            exp for exp, q in zip(batch, qualities) if not exp["is_good"]
        ]

        if not good_experiences or not bad_experiences:
            return {"updated": False}

        # Prepare batches
        good_obs = self._prepare_obs_batch(
            [exp["observations"] for exp in good_experiences]
        )
        good_actions = torch.stack([exp["action"] for exp in good_experiences]).to(
            self.device
        )
        bad_obs = self._prepare_obs_batch(
            [exp["observations"] for exp in bad_experiences]
        )
        bad_actions = torch.stack([exp["action"] for exp in bad_experiences]).to(
            self.device
        )

        # Calculate energies
        good_energies = self.verifier(good_obs, good_actions)
        bad_energies = self.verifier(bad_obs, bad_actions)

        # Contrastive loss
        contrastive_loss = F.relu(
            good_energies - bad_energies + self.contrastive_margin
        )
        energy_loss = contrastive_loss.mean()

        # Regularization to keep energies from exploding
        energy_regularization = 0.01 * (
            good_energies.pow(2).mean() + bad_energies.pow(2).mean()
        )
        total_loss = energy_loss + energy_regularization

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.verifier.parameters(), max_norm=1.0
        )
        self.optimizer.step()

        # Update stats
        self.training_stats.update(
            {
                "energy_loss": energy_loss.item(),
                "positive_energy": good_energies.mean().item(),
                "negative_energy": bad_energies.mean().item(),
                "gradient_norm": gradient_norm.item(),
            }
        )
        self.training_stats["positive_energy_hist"].append(good_energies.mean().item())
        self.training_stats["negative_energy_hist"].append(bad_energies.mean().item())

        return {"updated": True, **self.training_stats}

    def _prepare_obs_batch(self, obs_list: list) -> dict:
        """Prepare a batch of observations for the verifier."""
        batch_obs = {}
        for key in obs_list[0].keys():
            batch_obs[key] = torch.stack([obs[key] for obs in obs_list]).to(self.device)
        return batch_obs

    def set_learning_rate(self, new_lr: float):
        """Update the optimizer's learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = new_lr
        print(f"🔧 Optimizer LR updated to: {new_lr:.2e}")


def run_training_episode(
    env,
    agent: StabilizedEnergyBasedAgent,
    trainer: EnergyBasedTrainer,
    experience_buffer: ExperienceBuffer,
    max_steps: int = 18000,
    train_freq: int = 10,
) -> dict:
    """Run one training episode and collect detailed statistics."""
    obs, info = env.reset()
    total_reward = 0.0
    episode_length = 0
    done, truncated = False, False

    episode_thinking_stats = []
    episode_train_stats = []

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

        # Calculate experience quality (simple reward-based proxy)
        quality_score = np.clip(
            (reward + 0.1) / 0.2, 0.1, 1.0
        )  # Normalize reward to 0.1-1.0 range

        # Add experience to buffer
        obs_tensor = {k: torch.from_numpy(v).float() for k, v in obs.items()}
        action_onehot = torch.zeros(agent.action_dim)
        action_onehot[action] = 1.0

        experience_buffer.add_experience(
            experience={
                "observations": obs_tensor,
                "action": action_onehot,
                "is_good": reward > 0.0,
            },
            quality_score=quality_score,
        )

        # Train periodically
        if step % train_freq == 0:
            train_result = trainer.train_step()
            if train_result.get("updated"):
                episode_train_stats.append(train_result)

        obs = next_obs
        if done or truncated:
            break

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
    }
    return final_stats


def main():
    parser = argparse.ArgumentParser(
        description="Stabilized EBT Training for Street Fighter with ZIP Saving"
    )
    parser.add_argument("--total-episodes", type=int, default=50000)
    parser.add_argument(
        "--save-freq",
        type=int,
        default=50,
        help="Save best model checkpoint if new best is found, check every X episodes.",
    )
    parser.add_argument(
        "--zip-freq",
        type=int,
        default=500,
        help="Create ZIP package every X episodes.",
    )
    parser.add_argument(
        "--keep-zips",
        type=int,
        default=5,
        help="Number of recent ZIP files to keep.",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from."
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--device", type=str, default="auto")

    # --- FIX: Accept both --lr and --learning-rate ---
    parser.add_argument(
        "--lr",
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Initial learning rate.",
    )

    parser.add_argument(
        "--thinking-lr", type=float, default=0.1, help="Initial thinking learning rate."
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--contrastive-margin", type=float, default=2.0)
    parser.add_argument("--thinking-steps", type=int, default=3)
    args = parser.parse_args()

    print("🛡️ STABILIZED ENERGY-BASED TRANSFORMER TRAINING 🛡️")
    print("📦 WITH PERIODIC ZIP SAVING ENABLED 📦")
    print("=" * 50)
    print(f"Hyperparameters:")
    print(f"  - Initial LR: {args.lr}")
    print(f"  - Initial Thinking LR: {args.thinking_lr}")
    print(f"  - Contrastive Margin: {args.contrastive_margin}")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - ZIP Frequency: Every {args.zip_freq} episodes")
    print(f"  - Keep ZIPs: {args.keep_zips} most recent")
    print(
        f"Feature System: {'Enhanced' if BAIT_PUNISH_AVAILABLE else 'Base'} ({VECTOR_FEATURE_DIM} dims)"
    )
    print("=" * 50)

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

    # Initialize Stability and Data Management Systems
    stability_manager = EnergyStabilityManager(
        initial_lr=args.lr, thinking_lr=args.thinking_lr
    )
    experience_buffer = ExperienceBuffer(capacity=20000, quality_threshold=0.5)
    checkpoint_manager = CheckpointManager(checkpoint_dir="checkpoints")

    trainer = EnergyBasedTrainer(
        verifier,
        experience_buffer,
        initial_lr=args.lr,
        contrastive_margin=args.contrastive_margin,
        batch_size=args.batch_size,
        device=device,
    )

    # Resume from checkpoint if provided
    start_episode = 0
    if args.resume and os.path.exists(args.resume):
        print(f"📂 Resuming from checkpoint: {args.resume}")
        data = checkpoint_manager.load_checkpoint(args.resume, verifier, agent)
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

    # Create initial ZIP package
    print("\n📦 Creating initial ZIP package...")
    create_zip_package(episode=start_episode, win_rate=0.0)

    # Main Training Loop
    print("\n🚀 Starting Stabilized Training...")
    pbar = tqdm(
        range(start_episode, args.total_episodes),
        initial=start_episode,
        total=args.total_episodes,
        desc="Episodes",
    )

    for episode in pbar:
        # Run one episode
        episode_data = run_training_episode(
            env, agent, trainer, experience_buffer, train_freq=10
        )

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

        # 🚨 Emergency Reset Protocol 🚨
        if is_emergency:
            print("🚨 EMERGENCY PROTOCOL TRIGGERED! Restoring to best state.")
            # 1. Restore best model
            restored_data = checkpoint_manager.emergency_restore(verifier, agent)
            if restored_data:
                print(f"   ✅ Model restored from episode {restored_data['episode']}")
            else:
                print(
                    "   ❌ No checkpoint to restore from. Continuing with reduced LR."
                )

            # 2. Purge experience buffer
            experience_buffer.emergency_purge(keep_ratio=0.2)

            # 3. Apply new learning rates
            new_lr, new_thinking_lr = stability_manager.get_current_lrs()
            trainer.set_learning_rate(new_lr)
            agent.current_thinking_lr = new_thinking_lr

            # 4. Create emergency ZIP package
            print("📦 Creating emergency ZIP package...")
            create_zip_package(episode=episode, win_rate=win_rate)

        # Checkpoint saving logic
        if episode % args.save_freq == 0:
            current_win_rate = safe_mean(list(stability_manager.win_rate_window), 0.0)
            if stability_manager.should_save_checkpoint(current_win_rate):
                checkpoint_manager.save_checkpoint(
                    verifier, agent, episode, current_win_rate, energy_quality
                )

        # 📦 ZIP Package Creation Logic 📦
        if episode % args.zip_freq == 0 and episode > 0:
            print(f"\n📦 Creating ZIP package for episode {episode}...")
            current_win_rate = safe_mean(list(stability_manager.win_rate_window), 0.0)
            zip_path = create_zip_package(episode=episode, win_rate=current_win_rate)

            if zip_path:
                # Clean up old ZIP files
                cleanup_old_zips(keep_count=args.keep_zips)
                print(f"   ✅ ZIP package saved and old files cleaned up")
            else:
                print(f"   ❌ Failed to create ZIP package")

        # Check for recovery
        stability_manager.recovery_check(
            safe_mean(list(stability_manager.win_rate_window), 0.0)
        )

        # Update progress bar
        pbar.set_postfix(
            {
                "WR": f"{win_rate:.2f}",
                "EQ": f"{energy_quality:.1f}",
                "Loss": f"{energy_loss:.3f}",
                "LR": f"{trainer.optimizer.param_groups[0]['lr']:.1e}",
                "Mode": "EMERGENCY" if stability_manager.emergency_mode else "Normal",
            }
        )

    # Create final ZIP package
    print("\n📦 Creating final ZIP package...")
    final_win_rate = safe_mean(list(stability_manager.win_rate_window), 0.0)
    create_zip_package(episode=args.total_episodes, win_rate=final_win_rate)
    cleanup_old_zips(keep_count=args.keep_zips)

    print("\n🎉 Training finished!")
    print(f"📦 All ZIP packages saved in 'zip_saves' directory")
    env.close()


if __name__ == "__main__":
    main()
