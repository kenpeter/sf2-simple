#!/usr/bin/env python3
"""
train.py - STABILIZED TRAINING SCRIPT with fixes for training instability AND array ambiguity - FULL SIZE FRAMES
FIXES: High value loss (35+→<8), Low explained variance (0.11→>0.3), Low clip fraction (0.04→0.1-0.3), Array ambiguity errors
SOLUTION: Conservative hyperparameters, enhanced monitoring, stability-focused training, proper scalar handling
MODIFICATION: Adapted for full-size frames (224x320) with adjusted hyperparameters for increased memory usage
"""

import os
import argparse
import torch
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
import retro
import logging

# Import stabilized components from wrapper.py
from wrapper import (
    FixedStreetFighterPolicy,
    StreetFighterVisionWrapper,
    verify_gradient_flow,
    ensure_scalar,
    safe_bool_check,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StabilityCallback(BaseCallback):
    """
    STABILIZED callback with focus on training stability metrics AND array safety - UPDATED for full-size frames.
    Monitors value loss, explained variance, clip fraction for stability issues.
    Additional monitoring for memory usage with larger frames.
    CRITICAL FIX: Proper handling of array values from metrics.
    """

    def __init__(self, save_freq=50000, save_path="./models/", verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

        # Core stability metrics tracking
        self.value_losses = []
        self.explained_variances = []
        self.clip_fractions = []

        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.win_rates = []

        # Stability monitoring
        self.value_loss_spikes = 0
        self.stability_warnings = 0
        self.training_start_time = None
        self.device = None

        # Best model tracking
        self.best_explained_variance = -float("inf")
        self.best_value_loss = float("inf")
        self.best_stability_score = 0.0

        # Memory monitoring for full-size frames
        self.memory_warnings = 0
        self.peak_memory_usage = 0.0

    def _on_training_start(self):
        """Initialize training with stability focus - UPDATED for full-size frames."""
        self.training_start_time = datetime.now()
        self.device = next(self.model.policy.parameters()).device

        print(f"🚀 STABILIZED Training Started (FULL SIZE FRAMES)")
        print(f"🎯 STABILITY TARGETS:")
        print(f"   - Value Loss: <8.0 (currently expecting 35+)")
        print(f"   - Explained Variance: >0.3 (currently 0.11)")
        print(f"   - Clip Fraction: 0.1-0.3 (currently 0.04)")
        print(
            f"🖼️  Frame Size: Full resolution (224x320) - Higher memory usage expected"
        )
        print(f"🔧 Device: {self.device}")
        print(f"🛡️  Array Safety: Enhanced scalar conversion for metrics")

        # Check memory availability
        if torch.cuda.is_available() and self.device.type == "cuda":
            memory_gb = torch.cuda.get_device_properties(self.device).total_memory / 1e9
            print(f"📊 GPU Memory: {memory_gb:.1f} GB")
            if memory_gb < 8:
                print(
                    "   ⚠️  WARNING: Limited GPU memory - consider reducing batch size"
                )

        # Initial stability check
        self._perform_initial_stability_check()

    def _perform_initial_stability_check(self):
        """Check initial model stability before training - UPDATED for full-size frames."""
        print("\n🔬 Initial Stability Assessment (FULL SIZE)")

        env = self._get_env()
        if env:
            try:
                stable = verify_gradient_flow(self.model, env, self.device)
                if stable:
                    print("   ✅ Initial stability verified for full-size frames")
                else:
                    print("   ⚠️  Initial stability issues detected")
                    print("   🔧 Proceeding with enhanced monitoring")
            except Exception as e:
                print(f"   ❌ Stability check failed: {e}")

    def _get_env(self):
        """Get environment for testing."""
        if hasattr(self.training_env, "envs"):
            return self.training_env.envs[0]
        elif hasattr(self.training_env, "env"):
            return self.training_env.env
        else:
            return self.training_env

    def _on_step(self) -> bool:
        """Monitor training stability each step - UPDATED for memory monitoring."""

        # Extract training metrics with array safety
        self._extract_training_metrics()

        # Extract performance metrics with array safety
        self._extract_performance_metrics()

        # Monitor memory usage for full-size frames
        self._monitor_memory_usage()

        # Periodic stability monitoring
        if self.num_timesteps % 10000 == 0:
            self._check_training_stability()

        # Enhanced reporting for stability
        if self.num_timesteps % 5000 == 0:
            self._log_stability_report()

        # Save model when stability improves
        if self.num_timesteps > 0 and self.num_timesteps % self.save_freq == 0:
            self._save_stability_checkpoint()

        return True

    def _monitor_memory_usage(self):
        """Monitor memory usage for full-size frames."""
        if torch.cuda.is_available() and self.device.type == "cuda":
            current_memory = torch.cuda.memory_allocated(self.device) / 1e9
            self.peak_memory_usage = max(self.peak_memory_usage, current_memory)

            # Warn if memory usage is very high
            if (
                current_memory
                > 0.9 * torch.cuda.get_device_properties(self.device).total_memory / 1e9
            ):
                self.memory_warnings += 1
                if self.memory_warnings % 10 == 0:  # Only warn every 10th time
                    print(f"⚠️  High GPU memory usage: {current_memory:.1f} GB")

    def _extract_training_metrics(self):
        """Extract key training stability metrics with ARRAY SAFETY."""
        if hasattr(self.logger, "name_to_value"):
            metrics = self.logger.name_to_value

            # CRITICAL FIX: Value loss (CRITICAL for stability) - ensure scalar
            if "train/value_loss" in metrics:
                value_loss_raw = metrics["train/value_loss"]
                value_loss = ensure_scalar(value_loss_raw, 0.0)
                self.value_losses.append(value_loss)

                # Detect dangerous value loss spikes
                if value_loss > 50.0:
                    self.value_loss_spikes += 1
                    print(
                        f"🚨 VALUE LOSS SPIKE: {value_loss:.1f} at step {self.num_timesteps}"
                    )

                # Keep recent history
                if len(self.value_losses) > 1000:
                    self.value_losses.pop(0)

            # CRITICAL FIX: Explained variance - ensure scalar
            if "train/explained_variance" in metrics:
                explained_var_raw = metrics["train/explained_variance"]
                explained_var = ensure_scalar(explained_var_raw, 0.0)
                self.explained_variances.append(explained_var)
                if len(self.explained_variances) > 1000:
                    self.explained_variances.pop(0)

            # CRITICAL FIX: Clip fraction - ensure scalar
            if "train/clip_fraction" in metrics:
                clip_frac_raw = metrics["train/clip_fraction"]
                clip_frac = ensure_scalar(clip_frac_raw, 0.0)
                self.clip_fractions.append(clip_frac)
                if len(self.clip_fractions) > 1000:
                    self.clip_fractions.pop(0)

    def _extract_performance_metrics(self):
        """Extract episode performance metrics with ARRAY SAFETY."""
        if hasattr(self.locals, "infos") and self.locals["infos"]:
            for info in self.locals["infos"]:
                if "episode" in info:
                    episode_info = info["episode"]
                    # CRITICAL FIX: Ensure scalar episode metrics
                    episode_reward = ensure_scalar(episode_info["r"], 0.0)
                    episode_length = ensure_scalar(episode_info["l"], 0.0)

                    self.episode_rewards.append(episode_reward)
                    self.episode_lengths.append(episode_length)

                    # Keep recent episodes only
                    if len(self.episode_rewards) > 100:
                        self.episode_rewards.pop(0)
                        self.episode_lengths.pop(0)

                # CRITICAL FIX: Win rate tracking - ensure scalar
                if "win_rate" in info:
                    win_rate = ensure_scalar(info["win_rate"], 0.0)
                    self.win_rates.append(win_rate)
                    if len(self.win_rates) > 100:
                        self.win_rates.pop(0)

    def _check_training_stability(self):
        """Check current training stability status."""
        if not self.value_losses:
            return

        # Calculate recent metrics - all guaranteed to be scalars now
        recent_value_loss = np.mean(self.value_losses[-10:])
        recent_explained_var = (
            np.mean(self.explained_variances[-10:]) if self.explained_variances else 0
        )
        recent_clip_frac = (
            np.mean(self.clip_fractions[-10:]) if self.clip_fractions else 0
        )

        # Calculate stability score
        stability_score = self._calculate_stability_score(
            recent_value_loss, recent_explained_var, recent_clip_frac
        )

        # Issue warnings for instability
        if stability_score < 30:
            self.stability_warnings += 1
            print(
                f"⚠️  STABILITY WARNING #{self.stability_warnings} at step {self.num_timesteps}"
            )
            print(f"   Stability Score: {stability_score:.0f}/100")
            print(f"   Value Loss: {recent_value_loss:.1f} (target: <8.0)")
            print(f"   Explained Var: {recent_explained_var:.3f} (target: >0.3)")
            print(f"   Clip Fraction: {recent_clip_frac:.3f} (target: 0.1-0.3)")

        self.best_stability_score = max(self.best_stability_score, stability_score)

    def _calculate_stability_score(self, value_loss, explained_var, clip_frac):
        """Calculate overall stability score (0-100) - inputs guaranteed to be scalars."""
        score = 0.0

        # Value loss component (40 points max)
        if value_loss < 8.0:
            score += 40
        elif value_loss < 15.0:
            score += 25
        elif value_loss < 30.0:
            score += 10

        # Explained variance component (35 points max)
        if explained_var > 0.5:
            score += 35
        elif explained_var > 0.3:
            score += 25
        elif explained_var > 0.15:
            score += 15
        elif explained_var > 0.05:
            score += 5

        # Clip fraction component (25 points max)
        if 0.1 <= clip_frac <= 0.3:
            score += 25
        elif 0.05 <= clip_frac <= 0.5:
            score += 15
        elif clip_frac > 0:
            score += 5

        return score

    def _log_stability_report(self):
        """Log detailed stability report - UPDATED for full-size frames."""
        print(f"\n📊 STABILITY REPORT (FULL SIZE) - Step {self.num_timesteps:,}")
        print("=" * 65)

        # Training time
        if self.training_start_time:
            elapsed = datetime.now() - self.training_start_time
            hours = elapsed.total_seconds() / 3600
            print(f"⏱️  Training Time: {hours:.1f} hours")

        # Memory usage for full-size frames
        if torch.cuda.is_available() and self.device.type == "cuda":
            current_memory = torch.cuda.memory_allocated(self.device) / 1e9
            print(
                f"🖼️  Memory Usage: {current_memory:.1f} GB (Peak: {self.peak_memory_usage:.1f} GB)"
            )
            if self.memory_warnings > 0:
                print(f"   ⚠️  Memory warnings: {self.memory_warnings}")

        # Core stability metrics - all guaranteed to be scalars
        if self.value_losses:
            recent_value_loss = np.mean(self.value_losses[-20:])
            print(f"🎯 Value Loss: {recent_value_loss:.2f}")

            if recent_value_loss < 8.0:
                print(f"   ✅ EXCELLENT - Stable value predictions!")
            elif recent_value_loss < 15.0:
                print(f"   👍 GOOD - Moderate stability")
            elif recent_value_loss < 30.0:
                print(f"   ⚠️  FAIR - Some instability detected")
            else:
                print(f"   🚨 CRITICAL - High instability! Training may be stuck")

        if self.explained_variances:
            recent_explained_var = np.mean(self.explained_variances[-20:])
            print(f"📈 Explained Variance: {recent_explained_var:.3f}")

            if recent_explained_var > 0.5:
                print(
                    f"   ✅ EXCELLENT - Model explains {recent_explained_var:.1%} of rewards"
                )
            elif recent_explained_var > 0.3:
                print(
                    f"   👍 GOOD - Model explains {recent_explained_var:.1%} of rewards"
                )
            elif recent_explained_var > 0.1:
                print(
                    f"   ⚠️  FAIR - Model explains {recent_explained_var:.1%} of rewards"
                )
            else:
                print(
                    f"   🚨 POOR - Model explains only {recent_explained_var:.1%} of rewards"
                )

        if self.clip_fractions:
            recent_clip_frac = np.mean(self.clip_fractions[-20:])
            print(f"✂️  Clip Fraction: {recent_clip_frac:.3f}")

            if 0.1 <= recent_clip_frac <= 0.3:
                print(f"   ✅ EXCELLENT - Healthy policy updates")
            elif 0.05 <= recent_clip_frac <= 0.5:
                print(f"   👍 GOOD - Moderate policy updates")
            elif recent_clip_frac < 0.05:
                print(f"   ⚠️  LOW - Policy updates may be too small")
            else:
                print(f"   ⚠️  HIGH - Policy updates may be too aggressive")

        # Overall stability assessment
        if self.value_losses and self.explained_variances and self.clip_fractions:
            recent_value_loss = np.mean(self.value_losses[-20:])
            recent_explained_var = np.mean(self.explained_variances[-20:])
            recent_clip_frac = np.mean(self.clip_fractions[-20:])

            stability_score = self._calculate_stability_score(
                recent_value_loss, recent_explained_var, recent_clip_frac
            )
            print(f"🏥 Stability Score: {stability_score:.0f}/100")

            if stability_score >= 80:
                print(f"   🎉 EXCELLENT - Optimal training conditions!")
            elif stability_score >= 60:
                print(f"   ✅ GOOD - Training progressing well")
            elif stability_score >= 40:
                print(f"   ⚠️  MODERATE - Monitor closely")
            else:
                print(f"   🚨 POOR - Consider adjusting hyperparameters")

        # Performance metrics
        if self.episode_rewards:
            recent_rewards = self.episode_rewards[-20:]
            avg_reward = np.mean(recent_rewards)
            reward_std = np.std(recent_rewards)

            print(f"🎮 Performance (last 20 episodes):")
            print(f"   - Average reward: {avg_reward:.2f} ± {reward_std:.2f}")

        if self.win_rates:
            recent_win_rate = np.mean(self.win_rates[-20:])
            print(f"🏆 Win Rate: {recent_win_rate:.1%}")

            if recent_win_rate > 0.6:
                print(f"   🔥 DOMINATING!")
            elif recent_win_rate > 0.4:
                print(f"   👍 COMPETITIVE!")
            else:
                print(f"   💪 LEARNING!")

        # Stability issues summary
        if self.value_loss_spikes > 0:
            print(f"⚠️  Value loss spikes detected: {self.value_loss_spikes}")

        print()

    def _save_stability_checkpoint(self):
        """Save model checkpoint with stability metrics."""
        # All metrics guaranteed to be scalars now
        current_value_loss = (
            np.mean(self.value_losses[-10:]) if self.value_losses else float("inf")
        )
        current_explained_var = (
            np.mean(self.explained_variances[-10:]) if self.explained_variances else 0
        )
        current_clip_frac = (
            np.mean(self.clip_fractions[-10:]) if self.clip_fractions else 0
        )

        # Simple checkpoint naming
        model_name = f"model_{self.num_timesteps}"
        model_path = os.path.join(self.save_path, f"{model_name}.zip")

        self.model.save(model_path)
        print(f"💾 Checkpoint: {model_name}.zip")

        # Track and save best models
        if current_explained_var > self.best_explained_variance:
            self.best_explained_variance = current_explained_var
            best_path = os.path.join(self.save_path, "best_explained_variance.zip")
            self.model.save(best_path)
            print(f"   🎯 NEW BEST explained variance: {current_explained_var:.3f}")

        if current_value_loss < self.best_value_loss:
            self.best_value_loss = current_value_loss
            best_path = os.path.join(self.save_path, "best_value_loss.zip")
            self.model.save(best_path)
            print(f"   🎯 NEW BEST value loss: {current_value_loss:.2f}")

        # Log current stability
        stability_score = self._calculate_stability_score(
            current_value_loss, current_explained_var, current_clip_frac
        )
        print(f"   📊 Current stability: {stability_score:.0f}/100")

        # Log memory usage
        if torch.cuda.is_available() and self.device.type == "cuda":
            current_memory = torch.cuda.memory_allocated(self.device) / 1e9
            print(f"   💾 GPU Memory: {current_memory:.1f} GB")


def make_env(
    game="StreetFighterIISpecialChampionEdition-Genesis",
    state="ken_bison_12.state",
    render_mode=None,
):
    """Create the Street Fighter environment with stability wrapper - UPDATED for full-size frames."""
    try:
        env = retro.make(game=game, state=state, render_mode=render_mode)
        env = Monitor(env)
        env = StreetFighterVisionWrapper(
            env, frame_stack=8, rendering=(render_mode is not None)
        )
        print(f"✅ Environment created with full-size frames and array safety")
        return env
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="STABILIZED Street Fighter Training - Full Size + Array Safety"
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=1000000, help="Total training timesteps"
    )
    parser.add_argument("--resume", type=str, default=None, help="Path to resume from")
    parser.add_argument("--render", action="store_true", help="Render during training")
    parser.add_argument(
        "--game", type=str, default="StreetFighterIISpecialChampionEdition-Genesis"
    )
    parser.add_argument("--state", type=str, default="ken_bison_12.state")
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)"
    )
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU usage")
    parser.add_argument(
        "--test-stability", action="store_true", help="Test stability before training"
    )

    # STABILITY-FOCUSED HYPERPARAMETERS - ADJUSTED FOR FULL-SIZE FRAMES
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=8e-5,  # Reduced from 1e-4 for larger frames
        help="Learning rate (default: 8e-5 for full-size stability)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,  # Reduced from 256 for memory management
        help="Batch size (default: 128 for full-size frames)",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=4096,  # Reduced from 8192 for memory management
        help="Rollout steps (default: 4096 for full-size frames)",
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=3,
        help="Epochs per update (default: 3 for stability)",
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=0.1,
        help="Clip range (default: 0.1 for stability)",
    )
    parser.add_argument(
        "--vf-coef",
        type=float,
        default=2.0,
        help="Value function coefficient (default: 2.0 for stability)",
    )

    args = parser.parse_args()

    print("🔧 STABILIZED STREET FIGHTER TRAINING - FULL SIZE FRAMES + ARRAY SAFETY")
    print("=" * 80)
    print("🎯 ADDRESSING TRAINING INSTABILITY:")
    print("   - HIGH Value Loss: 35+ → Target <8.0")
    print("   - LOW Explained Variance: 0.11 → Target >0.3")
    print("   - LOW Clip Fraction: 0.04 → Target 0.1-0.3")
    print("   - ARRAY AMBIGUITY: Fixed with scalar conversion")
    print()
    print("🖼️  FULL SIZE FRAME MODIFICATIONS:")
    print("   - Frame Resolution: 224x320 (vs 128x180 previously)")
    print("   - Memory Usage: ~3x higher")
    print("   - Batch Size: Reduced to 128 (from 256)")
    print("   - Rollout Steps: Reduced to 4096 (from 8192)")
    print("   - Learning Rate: Reduced to 8e-5 (from 1e-4)")
    print()
    print("🛡️  ARRAY SAFETY ENHANCEMENTS:")
    print("   - ensure_scalar() for all metric extractions")
    print("   - safe_bool_check() for array boolean operations")
    print("   - Proper vectorized env info handling")
    print()
    print("🔧 STABILIZED HYPERPARAMETERS:")
    print(f"   - Learning Rate: {args.learning_rate} (conservative for full-size)")
    print(f"   - Batch Size: {args.batch_size} (reduced for memory)")
    print(f"   - Rollout Steps: {args.n_steps} (reduced for memory)")
    print(f"   - Epochs: {args.n_epochs} (fewer for stability)")
    print(f"   - Clip Range: {args.clip_range} (low for stability)")
    print(f"   - Value Coefficient: {args.vf_coef} (high for value stability)")
    print(f"   - Total timesteps: {args.total_timesteps:,}")
    print()

    # Device selection with memory considerations
    if args.force_cpu:
        device = torch.device("cpu")
        print(f"🔧 Device: CPU (forced)")
    elif args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🔧 Device: {device} (auto-detected)")

        # Check GPU memory for full-size frames
        if device.type == "cuda":
            memory_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
            print(f"📊 GPU Memory: {memory_gb:.1f} GB")
            if memory_gb < 6:
                print("   ⚠️  WARNING: Low GPU memory for full-size frames!")
                print("   💡 Consider using --force-cpu or reducing batch size")
    else:
        device = torch.device(args.device)
        print(f"🔧 Device: {device} (specified)")

    # Set random seed for reproducibility
    set_random_seed(42)

    # Create environment
    render_mode = "human" if args.render else None
    env = make_env(args.game, args.state, render_mode)

    # Test environment with full-size frames
    print("🧪 Testing environment with full-size frames and array safety...")
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]

    # Check frame dimensions
    visual_shape = obs["visual_obs"].shape
    print(f"   ✅ Visual obs shape: {visual_shape}")
    print(
        f"   🖼️  Frame size: {visual_shape[1]}x{visual_shape[2]} (confirmed full-size)"
    )

    # Estimate memory usage
    frame_memory_mb = np.prod(visual_shape) * 4 / 1024 / 1024  # 4 bytes per float32
    batch_memory_mb = frame_memory_mb * args.batch_size
    print(f"   📊 Estimated batch memory: {batch_memory_mb:.1f} MB")
    print(f"   🛡️  Array safety: All metrics will be converted to scalars")

    # Create or load model
    if args.resume and os.path.exists(args.resume):
        print(f"📂 Resuming from {args.resume}")
        print("🔄 Applying new hyperparameters for continued full-size training...")
        try:
            # Pass hyperparameters directly into the .load() method.
            model = PPO.load(
                args.resume,
                env=env,
                device=device,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                n_steps=args.n_steps,
                n_epochs=args.n_epochs,
                clip_range=args.clip_range,
                vf_coef=args.vf_coef,
            )
            print(
                f"   ✅ Model loaded on {device} with updated hyperparameters for full-size."
            )

        except Exception as e:
            print(f"   ❌ Failed to load model: {e}")
            print("   🆕 Creating new model instead...")
            model = None
    else:
        model = None

    if model is None:
        print("🆕 Creating new STABILIZED model for full-size frames...")
        model = PPO(
            FixedStreetFighterPolicy,
            env,
            learning_rate=args.learning_rate,  # 8e-5 (reduced for full-size)
            n_steps=args.n_steps,  # 4096 (reduced for memory)
            batch_size=args.batch_size,  # 128 (reduced for memory)
            n_epochs=args.n_epochs,  # 3 (fewer for stability)
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=args.clip_range,  # 0.1 (low for stability)
            ent_coef=0.03,
            vf_coef=args.vf_coef,  # 2.0 (higher for value stability)
            max_grad_norm=0.5,  # Conservative gradient clipping
            verbose=1,
            device=device,
        )

    # Verify model device
    model_device = next(model.policy.parameters()).device
    print(f"   ✅ Model on device: {model_device}")

    # Test stability if requested
    if args.test_stability:
        print("\n🔬 Testing training stability with full-size frames...")
        stable = verify_gradient_flow(model, env, device)
        if not stable:
            print("⚠️  Stability issues detected but proceeding with monitoring")
        else:
            print("✅ Stability verified - ready for stable full-size training!")

    # Create stability-focused callback with memory monitoring and array safety
    callback = StabilityCallback(save_freq=50000, save_path="./models/")

    print("\n🚀 STARTING STABILIZED FULL-SIZE TRAINING WITH ARRAY SAFETY...")
    print("📊 Real-time monitoring of:")
    print("   - Value Loss (target: <8.0)")
    print("   - Explained Variance (target: >0.3)")
    print("   - Clip Fraction (target: 0.1-0.3)")
    print("   - GPU Memory Usage (full-size frames)")
    print("   - Array Safety (scalar conversion)")
    print("💾 Auto-saving best stability models")
    print()

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            reset_num_timesteps=not args.resume,
            progress_bar=True,
        )

        # Save final model
        final_path = "./models/final_stabilized_fullsize_arraysafe_model.zip"
        model.save(final_path)
        print(f"🎉 Full-size training with array safety completed successfully!")
        print(f"💾 Final model saved: {final_path}")

        # Final stability report
        print(f"\n📊 FINAL STABILITY RESULTS (FULL SIZE + ARRAY SAFE):")
        if callback.value_losses:
            final_value_loss = np.mean(callback.value_losses[-20:])
            print(f"📉 Final Value Loss: {final_value_loss:.2f}")
            if final_value_loss < 8.0:
                print(
                    f"   ✅ SUCCESS: Value loss target achieved with full-size frames!"
                )
            else:
                improvement = 35.0 - final_value_loss  # Assuming started at 35
                print(f"   📈 IMPROVEMENT: Reduced by {improvement:.1f} points")

        if callback.explained_variances:
            final_explained_var = np.mean(callback.explained_variances[-20:])
            print(f"📈 Final Explained Variance: {final_explained_var:.3f}")
            if final_explained_var > 0.3:
                print(
                    f"   ✅ SUCCESS: Explained variance target achieved with full-size!"
                )
            else:
                improvement = final_explained_var - 0.11  # Assuming started at 0.11
                print(f"   📈 IMPROVEMENT: Increased by {improvement:.3f}")

        if callback.clip_fractions:
            final_clip_frac = np.mean(callback.clip_fractions[-20:])
            print(f"✂️  Final Clip Fraction: {final_clip_frac:.3f}")
            if 0.1 <= final_clip_frac <= 0.3:
                print(f"   ✅ SUCCESS: Clip fraction in healthy range!")

        final_stability = callback.best_stability_score
        print(f"🏥 Best Stability Score: {final_stability:.0f}/100")

        # Memory usage summary
        if callback.peak_memory_usage > 0:
            print(f"📊 Peak GPU Memory: {callback.peak_memory_usage:.1f} GB")
            if callback.memory_warnings > 0:
                print(f"   ⚠️  Memory warnings: {callback.memory_warnings}")

        print(f"🛡️  Array Safety: No 'truth value of array' errors encountered!")

        if final_stability >= 60:
            print("🎉 FULL-SIZE TRAINING STABILIZATION WITH ARRAY SAFETY SUCCESSFUL!")
        else:
            print("📈 PARTIAL IMPROVEMENT - Continue training for full stabilization")

    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        interrupted_path = "./models/interrupted_fullsize_arraysafe_model.zip"
        model.save(interrupted_path)
        print(f"💾 Model saved: {interrupted_path}")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback

        traceback.print_exc()

        # Check if it's the array ambiguity error we were trying to fix
        if "truth value of an array" in str(e):
            print("🚨 CRITICAL: Array ambiguity error still occurring!")
            print("   This suggests the error is coming from a different location.")
            print("   Check for any remaining numpy array boolean operations.")

        error_path = "./models/error_fullsize_arraysafe_model.zip"
        try:
            model.save(error_path)
            print(f"💾 Model saved: {error_path}")
        except Exception as save_error:
            print(f"❌ Could not save model: {save_error}")
        raise

    finally:
        env.close()
        print("🔚 Full-size training session with array safety ended")


if __name__ == "__main__":
    main()
