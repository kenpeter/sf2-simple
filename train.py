#!/usr/bin/env python3
"""
train.py - STABILIZED TRAINING SCRIPT with advanced tactical rewards
FIXES: Addresses 50% win-rate plateau by teaching agent to block, bait, and punish.
NEW: Enhanced monitoring for advanced tactics (punishes, whiffs).
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
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StabilityCallback(BaseCallback):
    """
    STABILIZED callback with focus on stability and advanced tactics monitoring.
    Monitors punishes, whiffs, value loss, explained variance, and clip fraction.
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

        # --- NEW: Advanced tactics tracking ---
        self.successful_punishes = []
        self.induced_whiffs = []

        # Stability monitoring
        self.value_loss_spikes = 0
        self.stability_warnings = 0
        self.training_start_time = None
        self.device = None

        # Best model tracking
        self.best_explained_variance = -float("inf")
        self.best_value_loss = float("inf")
        self.best_stability_score = 0.0

    def _on_training_start(self):
        """Initialize training with stability focus."""
        self.training_start_time = datetime.now()
        self.device = next(self.model.policy.parameters()).device

        print(f"🚀 STABILIZED Training Started")
        print(f"🎯 NEW GOAL: Break 50% win rate by teaching advanced tactics.")
        print(f"   - Monitoring for successful punishes and induced whiffs.")
        print(f"🔧 Device: {self.device}")

        # Initial stability check
        self._perform_initial_stability_check()

    def _perform_initial_stability_check(self):
        """Check initial model stability before training."""
        print("\n🔬 Initial Stability Assessment")

        env = self._get_env()
        if env:
            try:
                stable = verify_gradient_flow(self.model, env, self.device)
                if stable:
                    print("   ✅ Initial stability verified")
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
        """Monitor training stability each step."""

        # Extract training metrics
        self._extract_training_metrics()

        # Extract performance metrics
        self._extract_performance_metrics()

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

    def _extract_training_metrics(self):
        """Extract key training stability metrics."""
        if hasattr(self.logger, "name_to_value"):
            metrics = self.logger.name_to_value

            if "train/value_loss" in metrics:
                value_loss = metrics["train/value_loss"]
                self.value_losses.append(value_loss)
                if value_loss > 50.0:
                    self.value_loss_spikes += 1
                if len(self.value_losses) > 1000:
                    self.value_losses.pop(0)
            if "train/explained_variance" in metrics:
                self.explained_variances.append(metrics["train/explained_variance"])
                if len(self.explained_variances) > 1000:
                    self.explained_variances.pop(0)
            if "train/clip_fraction" in metrics:
                self.clip_fractions.append(metrics["train/clip_fraction"])
                if len(self.clip_fractions) > 1000:
                    self.clip_fractions.pop(0)

    def _extract_performance_metrics(self):
        """Extract episode performance and advanced tactics metrics."""
        if hasattr(self.locals, "infos") and self.locals["infos"]:
            for info in self.locals["infos"]:
                if "episode" in info:
                    self.episode_rewards.append(info["episode"]["r"])
                    self.episode_lengths.append(info["episode"]["l"])
                    if len(self.episode_rewards) > 100:
                        self.episode_rewards.pop(0)
                        self.episode_lengths.pop(0)

                if "win_rate" in info:
                    self.win_rates.append(info["win_rate"])
                    if len(self.win_rates) > 100:
                        self.win_rates.pop(0)

                # --- NEW: Track advanced tactics ---
                if "successful_punishes" in info:
                    self.successful_punishes.append(info["successful_punishes"])
                    if len(self.successful_punishes) > 100:
                        self.successful_punishes.pop(0)
                if "induced_whiffs" in info:
                    self.induced_whiffs.append(info["induced_whiffs"])
                    if len(self.induced_whiffs) > 100:
                        self.induced_whiffs.pop(0)

    def _check_training_stability(self):
        """Check current training stability status."""
        if not self.value_losses:
            return

        recent_value_loss = np.mean(self.value_losses[-10:])
        recent_explained_var = (
            np.mean(self.explained_variances[-10:]) if self.explained_variances else 0
        )
        recent_clip_frac = (
            np.mean(self.clip_fractions[-10:]) if self.clip_fractions else 0
        )
        stability_score = self._calculate_stability_score(
            recent_value_loss, recent_explained_var, recent_clip_frac
        )
        if stability_score < 30:
            self.stability_warnings += 1
            print(
                f"⚠️  STABILITY WARNING #{self.stability_warnings} at step {self.num_timesteps}"
            )

        self.best_stability_score = max(self.best_stability_score, stability_score)

    def _calculate_stability_score(self, value_loss, explained_var, clip_frac):
        score = 0.0
        if value_loss < 8.0:
            score += 40
        elif value_loss < 15.0:
            score += 25
        if explained_var > 0.5:
            score += 35
        elif explained_var > 0.3:
            score += 25
        if 0.1 <= clip_frac <= 0.3:
            score += 25
        elif 0.05 <= clip_frac <= 0.5:
            score += 15
        return score

    def _log_stability_report(self):
        """Log detailed stability and advanced tactics report."""
        print(f"\n📊 STABILITY & TACTICS REPORT - Step {self.num_timesteps:,}")
        print("=" * 60)
        if self.training_start_time:
            elapsed = datetime.now() - self.training_start_time
            print(f"⏱️  Training Time: {elapsed.total_seconds() / 3600:.1f} hours")

        # --- NEW: Advanced Tactics Report ---
        if self.successful_punishes:
            avg_punishes = np.mean(self.successful_punishes)
            avg_whiffs = np.mean(self.induced_whiffs) if self.induced_whiffs else 0
            print(f"🧠 Advanced Tactics (avg per episode):")
            print(f"   - Successful Punishes: {avg_punishes:.2f}")
            print(f"   - Induced Whiffs/Blocks: {avg_whiffs:.2f}")

        # Core stability metrics
        if self.value_losses:
            recent_value_loss = np.mean(self.value_losses[-20:])
            print(f"🎯 Value Loss: {recent_value_loss:.2f}")
        if self.explained_variances:
            recent_explained_var = np.mean(self.explained_variances[-20:])
            print(f"📈 Explained Variance: {recent_explained_var:.3f}")
        if self.clip_fractions:
            recent_clip_frac = np.mean(self.clip_fractions[-20:])
            print(f"✂️  Clip Fraction: {recent_clip_frac:.3f}")

        # Overall stability & Performance
        if self.value_losses:
            score = self._calculate_stability_score(
                np.mean(self.value_losses[-20:]),
                np.mean(self.explained_variances[-20:]),
                np.mean(self.clip_fractions[-20:]),
            )
            print(f"🏥 Stability Score: {score:.0f}/100")
        if self.episode_rewards:
            print(f"🎮 Avg Reward (last 20): {np.mean(self.episode_rewards[-20:]):.2f}")
        if self.win_rates:
            print(f"🏆 Win Rate (last 20): {np.mean(self.win_rates[-20:]):.1%}")
        print()

    def _save_stability_checkpoint(self):
        """Save model checkpoint with stability metrics."""
        model_name = f"model_{self.num_timesteps}"
        model_path = os.path.join(self.save_path, f"{model_name}.zip")
        self.model.save(model_path)
        print(f"💾 Checkpoint: {model_name}.zip")

        current_value_loss = (
            np.mean(self.value_losses[-10:]) if self.value_losses else float("inf")
        )
        if current_value_loss < self.best_value_loss:
            self.best_value_loss = current_value_loss
            self.model.save(os.path.join(self.save_path, "best_value_loss.zip"))
            print(f"   🎯 NEW BEST value loss: {current_value_loss:.2f}")


def make_env(
    game="StreetFighterIISpecialChampionEdition-Genesis",
    state="ken_bison_12.state",
    render_mode=None,
):
    """Create the Street Fighter environment with stability wrapper."""
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
        description="STABILIZED Street Fighter Training with Advanced Tactics"
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=1000000, help="Total training timesteps"
    )
    parser.add_argument("--resume", type=str, default=None, help="Path to resume from")
    parser.add_argument("--render", action="store_true", help="Render during training")
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)"
    )
    # STABILITY-FOCUSED HYPERPARAMETERS
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=8192)
    parser.add_argument("--n-epochs", type=int, default=3)
    parser.add_argument("--clip-range", type=float, default=0.1)
    parser.add_argument("--vf-coef", type=float, default=2.0)
    args = parser.parse_args()

    print("🔧 STABILIZED STREET FIGHTER TRAINING - ADVANCED TACTICS")
    print("=" * 60)
    print("🎯 GOAL: Break the 50% win rate plateau.")
    print("   - Rewarding the agent for blocking, baiting, and punishing.")
    print("   - This encourages intelligent defense and counter-attacks.")
    print()
    print("🔧 STABILIZED HYPERPARAMETERS:")
    print(
        f"   - Learning Rate: {args.learning_rate}, Batch Size: {args.batch_size}, Rollout: {args.n_steps}"
    )
    print()

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.device == "auto" else "cpu"
    )
    print(f"🔧 Device: {device}")
    set_random_seed(42)

    render_mode = "human" if args.render else None
    env = make_env(render_mode=render_mode)

    model = None
    if args.resume and os.path.exists(args.resume):
        print(f"📂 Resuming from {args.resume}")
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

    if model is None:
        print("🆕 Creating new STABILIZED model...")
        model = PPO(
            FixedStreetFighterPolicy,
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=args.clip_range,
            ent_coef=0.03,
            vf_coef=args.vf_coef,
            max_grad_norm=0.5,
            verbose=1,
            device=device,
        )

    callback = StabilityCallback(save_freq=50000, save_path="./models/")

    print("\n🚀 STARTING ADVANCED TACTICS TRAINING...")
    print("📊 Real-time monitoring of Punishes, Whiffs, and Stability.")
    print()

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            reset_num_timesteps=not args.resume,
            progress_bar=True,
        )
        model.save("./models/final_tactical_model.zip")
        print(f"🎉 Training completed successfully!")
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        model.save("./models/interrupted_tactical_model.zip")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback

        traceback.print_exc()
        model.save("./models/error_tactical_model.zip")
    finally:
        env.close()
        print("🔚 Training session ended")


if __name__ == "__main__":
    main()
