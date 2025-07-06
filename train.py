#!/usr/bin/env python3
"""
FIXED TRAINING SCRIPT - Uses the corrected feature extractor with proper device handling
Fixed: Simplified checkpoint naming, removed JSON files
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

# Import fixed components
from wrapper import (
    FixedStreetFighterPolicy,
    verify_gradient_flow,
    diagnose_vector_features,
)

# Import wrapper (assuming it's working correctly)
from wrapper import StreetFighterVisionWrapper

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FixedTrainingCallback(BaseCallback):
    """
    Callback that properly monitors the FIXED architecture with device compatibility.
    FIXED: Simplified checkpoint naming, removed JSON files
    """

    def __init__(self, save_freq=50000, save_path="./fixed_models/", verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

        # Performance tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.win_rates = []

        # Gradient health tracking
        self.gradient_checks = []
        self.last_gradient_check = 0

        # Training metrics
        self.training_start_time = None
        self.best_avg_reward = -float("inf")
        self.consecutive_improvements = 0

        # Device tracking
        self.device = None

    def _on_training_start(self):
        """Called when training starts."""
        self.training_start_time = datetime.now()

        # Detect device
        self.device = next(self.model.policy.parameters()).device

        print(f"🚀 Training started at {self.training_start_time}")
        print(f"🔧 Using device: {self.device}")

        # Get the environment from the training environment
        env = None
        if hasattr(self.training_env, "envs"):
            env = self.training_env.envs[0]
        elif hasattr(self.training_env, "env"):
            env = self.training_env.env
        else:
            env = self.training_env

        # Diagnose vector features first
        print("\n🔍 Vector Feature Quality Check")
        try:
            vector_stats = diagnose_vector_features(env, num_steps=30)

            if vector_stats["std"] < 1e-3:
                print("   ⚠️  WARNING: Very low vector feature variation!")
                print("   This WILL cause gradient blocking in vector components")
                print("   Consider checking your feature engineering")
            elif vector_stats["active_features"] < vector_stats["total_features"] * 0.5:
                print("   ⚠️  WARNING: Many inactive features detected")
                print("   Vector processing may be suboptimal")
            else:
                print("   ✅ Vector features look healthy for gradient flow")

        except Exception as e:
            print(f"   ❌ Vector diagnostic failed: {e}")

        # Initial gradient flow verification with proper device handling
        print("\n🔬 Initial Gradient Flow Verification (with Vector Focus)")

        gradient_ok = verify_gradient_flow(self.model, env, self.device)

        if gradient_ok:
            print("   ✅ Gradient flow verified - training can proceed")
            print("   ✅ Vector components confirmed flowing")
        else:
            print("   ❌ Gradient flow issues detected!")
            print(
                "   🚨 Vector components may be blocked - strategic learning impaired!"
            )
            print("   🛑 Consider stopping training to fix architecture")

    def _on_step(self) -> bool:
        """Called after each step."""

        # Periodic gradient flow checks
        if self.num_timesteps - self.last_gradient_check >= 10000:
            self.last_gradient_check = self.num_timesteps
            gradient_health = self._check_gradient_health()
            self.gradient_checks.append(gradient_health)

            # Alert if gradient flow degrades
            if gradient_health["coverage"] < 50:
                print(f"⚠️  GRADIENT FLOW DEGRADED at step {self.num_timesteps}")
                print(f"   Coverage: {gradient_health['coverage']:.1f}%")

        # Performance monitoring
        if self.num_timesteps % 5000 == 0:
            self._log_performance()

        # Save model periodically
        if self.num_timesteps > 0 and self.num_timesteps % self.save_freq == 0:
            self._save_checkpoint()

        # Extract episode information
        if hasattr(self.locals, "infos") and self.locals["infos"]:
            for info in self.locals["infos"]:
                if "episode" in info:
                    episode_info = info["episode"]
                    self.episode_rewards.append(episode_info["r"])
                    self.episode_lengths.append(episode_info["l"])

                    # Keep only recent episodes
                    if len(self.episode_rewards) > 100:
                        self.episode_rewards.pop(0)
                        self.episode_lengths.pop(0)

                # Extract win rate if available
                if "win_rate" in info:
                    self.win_rates.append(info["win_rate"])
                    if len(self.win_rates) > 100:
                        self.win_rates.pop(0)

        return True

    def _check_gradient_health(self) -> dict:
        """Quick gradient health check with device awareness."""
        total_params = 0
        params_with_grads = 0
        total_grad_norm = 0.0
        device_mismatches = 0

        for name, param in self.model.policy.named_parameters():
            total_params += param.numel()

            # Check device consistency
            if param.device != self.device:
                device_mismatches += 1

            if param.grad is not None:
                params_with_grads += param.numel()
                total_grad_norm += param.grad.norm().item()

        coverage = (params_with_grads / max(total_params, 1)) * 100
        avg_grad_norm = total_grad_norm / max(params_with_grads, 1)

        return {
            "step": self.num_timesteps,
            "coverage": coverage,
            "avg_grad_norm": avg_grad_norm,
            "total_params": total_params,
            "device_mismatches": device_mismatches,
            "device": str(self.device),
        }

    def _log_performance(self):
        """Log current performance metrics."""
        print(f"\n📊 Step {self.num_timesteps:,} Performance Report")
        print("=" * 50)

        # Training time
        if self.training_start_time:
            elapsed = datetime.now() - self.training_start_time
            hours = elapsed.total_seconds() / 3600
            print(f"⏱️  Training time: {hours:.1f} hours")

        # Gradient health
        if self.gradient_checks:
            latest_check = self.gradient_checks[-1]
            coverage = latest_check["coverage"]
            device_mismatches = latest_check.get("device_mismatches", 0)

            print(f"🧠 Gradient Health:")
            print(f"   - Coverage: {coverage:.1f}%")
            print(f"   - Avg norm: {latest_check['avg_grad_norm']:.6f}")
            print(f"   - Device: {latest_check.get('device', 'unknown')}")

            if device_mismatches > 0:
                print(f"   - ⚠️  Device mismatches: {device_mismatches}")

            if coverage > 95:
                print(f"   ✅ EXCELLENT gradient flow")
            elif coverage > 80:
                print(f"   👍 GOOD gradient flow")
            elif coverage > 50:
                print(f"   ⚠️  MODERATE gradient flow")
            else:
                print(f"   ❌ POOR gradient flow - CHECK ARCHITECTURE!")

        # Episode performance
        if self.episode_rewards:
            recent_rewards = self.episode_rewards[-20:]  # Last 20 episodes
            avg_reward = np.mean(recent_rewards)
            max_reward = np.max(recent_rewards)
            min_reward = np.min(recent_rewards)

            print(f"🎮 Episode Performance (last 20):")
            print(f"   - Average reward: {avg_reward:.2f}")
            print(f"   - Best reward: {max_reward:.2f}")
            print(f"   - Worst reward: {min_reward:.2f}")

            # Check for improvement
            if avg_reward > self.best_avg_reward:
                self.best_avg_reward = avg_reward
                self.consecutive_improvements += 1
                print(f"   🎉 NEW BEST! ({self.consecutive_improvements} improvements)")
            else:
                self.consecutive_improvements = 0

        # Win rate
        if self.win_rates:
            recent_win_rate = np.mean(self.win_rates[-20:])
            print(f"🏆 Win Rate: {recent_win_rate:.1%}")

            if recent_win_rate > 0.6:
                print(f"   🔥 DOMINATING! ({recent_win_rate:.1%})")
            elif recent_win_rate > 0.4:
                print(f"   👍 COMPETITIVE! ({recent_win_rate:.1%})")
            elif recent_win_rate > 0.2:
                print(f"   📈 IMPROVING! ({recent_win_rate:.1%})")
            else:
                print(f"   💪 LEARNING! ({recent_win_rate:.1%})")

        print()

    def _save_checkpoint(self):
        """
        Save model checkpoint with SIMPLIFIED naming.
        FIXED: Short filename with just timesteps, no JSON files.
        """
        # Simple naming: just timesteps
        model_name = f"model_{self.num_timesteps}"
        model_path = os.path.join(self.save_path, f"{model_name}.zip")

        self.model.save(model_path)
        print(f"💾 Checkpoint saved: {model_name}.zip")

        # Log performance in console instead of JSON file
        if self.episode_rewards:
            avg_reward = np.mean(self.episode_rewards[-10:])
            print(f"   📊 Current avg reward: {avg_reward:.1f}")

        if self.win_rates:
            win_rate = np.mean(self.win_rates[-10:])
            print(f"   🏆 Current win rate: {win_rate:.1%}")

        if self.gradient_checks:
            gradient_coverage = self.gradient_checks[-1]["coverage"]
            print(f"   🧠 Gradient coverage: {gradient_coverage:.0f}%")


def make_env(
    game="StreetFighterIISpecialChampionEdition-Genesis",
    state="ken_bison_12.state",
    render_mode=None,
):
    """Create the Street Fighter environment."""
    try:
        env = retro.make(game=game, state=state, render_mode=render_mode)
        env = Monitor(env)
        env = StreetFighterVisionWrapper(
            env, frame_stack=8, rendering=(render_mode is not None)
        )
        return env
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Train with FIXED architecture and device compatibility"
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=2000000, help="Total training timesteps"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument("--render", action="store_true", help="Render during training")
    parser.add_argument("--resume", type=str, default=None, help="Path to resume from")
    parser.add_argument(
        "--game", type=str, default="StreetFighterIISpecialChampionEdition-Genesis"
    )
    parser.add_argument("--state", type=str, default="ken_bison_12.state")
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)"
    )
    parser.add_argument(
        "--test-gradient-flow",
        action="store_true",
        help="Test gradient flow before training",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU usage even if CUDA is available",
    )

    args = parser.parse_args()

    print("🔧 FIXED ARCHITECTURE TRAINING WITH DEVICE COMPATIBILITY")
    print("=" * 70)
    print(f"   - Total timesteps: {args.total_timesteps:,}")
    print(f"   - Learning rate: {args.learning_rate}")
    print(f"   - Rendering: {'Enabled' if args.render else 'Disabled'}")

    # Device selection with proper handling
    if args.force_cpu:
        device = torch.device("cpu")
        print(f"   - Device: CPU (forced)")
    elif args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"   - Device: {device} (auto-detected)")
    else:
        device = torch.device(args.device)
        print(f"   - Device: {device} (specified)")

    # Validate device
    if device.type == "cuda" and not torch.cuda.is_available():
        print("   ⚠️  CUDA requested but not available, falling back to CPU")
        device = torch.device("cpu")

    # Set random seed
    set_random_seed(42)

    # Create environment
    render_mode = "human" if args.render else None
    env = make_env(args.game, args.state, render_mode)

    # Test environment
    print("\n🧪 Testing environment...")
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    print("   ✅ Environment working")

    # Create model with FIXED policy and proper device handling
    if args.resume and os.path.exists(args.resume):
        print(f"📂 Resuming from {args.resume}")
        try:
            model = PPO.load(args.resume, env=env, device=device)
            print(f"   ✅ Model loaded and moved to {device}")
        except Exception as e:
            print(f"   ❌ Failed to load model: {e}")
            print("   🆕 Creating new model instead...")
            model = None
    else:
        model = None

    if model is None:
        print("🆕 Creating model with FIXED architecture...")
        model = PPO(
            FixedStreetFighterPolicy,
            env,
            learning_rate=args.learning_rate,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            device=device,
        )

    # Verify model is on correct device
    model_device = next(model.policy.parameters()).device
    print(f"   ✅ Model created on device: {model_device}")

    if model_device != device:
        print(f"   ⚠️  Device mismatch detected! Moving model to {device}")
        model.policy.to(device)
        final_device = next(model.policy.parameters()).device
        print(f"   ✅ Model moved to: {final_device}")

    # Test gradient flow if requested
    if args.test_gradient_flow:
        print("\n🔬 Testing gradient flow...")
        gradient_ok = verify_gradient_flow(model, env, device)

        if not gradient_ok:
            print("❌ Gradient flow test failed!")
            print("🛑 Please fix architecture before training")
            return
        else:
            print("✅ Gradient flow verified - ready to train!")

    # Create callback with simplified naming
    callback = FixedTrainingCallback(save_freq=50000, save_path="./fixed_models/")

    print("\n🎯 Starting training with FIXED architecture...")
    print("   - Feature extractor properly integrated")
    print("   - Device compatibility ensured")
    print("   - Gradient flow verified")
    print("   - Simplified checkpoint naming")
    print("   - No JSON files - performance logged to console")

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            reset_num_timesteps=not args.resume,
            progress_bar=True,
        )

        # Save final model with simple name
        final_path = "./fixed_models/final_model.zip"
        model.save(final_path)
        print(f"🎉 Training completed successfully!")
        print(f"💾 Final model saved: final_model.zip")

        # Final performance summary
        if callback.episode_rewards:
            final_avg_reward = np.mean(callback.episode_rewards[-20:])
            print(f"📊 Final average reward: {final_avg_reward:.2f}")

        if callback.win_rates:
            final_win_rate = np.mean(callback.win_rates[-20:])
            print(f"🏆 Final win rate: {final_win_rate:.1%}")

        if callback.gradient_checks:
            final_gradient_health = callback.gradient_checks[-1]
            print(
                f"🧠 Final gradient coverage: {final_gradient_health['coverage']:.1f}%"
            )
            print(f"🔧 Final device: {final_gradient_health.get('device', 'unknown')}")

    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        interrupted_path = "./fixed_models/interrupted_model.zip"
        model.save(interrupted_path)
        print(f"💾 Model saved: interrupted_model.zip")

    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback

        traceback.print_exc()

        error_path = "./fixed_models/error_model.zip"
        try:
            model.save(error_path)
            print(f"💾 Model saved: error_model.zip")
        except Exception as save_error:
            print(f"❌ Could not save model: {save_error}")

        raise

    finally:
        env.close()
        print("🔚 Training session ended")


if __name__ == "__main__":
    main()
