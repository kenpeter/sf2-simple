"""
train.py - UPDATED TRAINING SCRIPT FOR ADVANCED STREET FIGHTER AI
COMPATIBLE: Works with the new AdvancedStreetFighterPolicy and complete wrapper
FEATURES: Advanced monitoring for baiting, blocking, and tactical gameplay
UPDATES: Full frame size support (320x224), adjusted monitoring for pixel changes
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

# Import new components from the advanced wrapper
from wrapper import (
    AdvancedStreetFighterPolicy,  # Updated import
    StreetFighterVisionWrapper,
    verify_gradient_flow,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedTacticsCallback(BaseCallback):
    """Advanced callback for monitoring Street Fighter tactics and stability - UPDATED FOR FULL FRAME"""

    def __init__(self, save_freq=25000, save_path="./models/", verbose=1):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

        # Core training metrics
        self.value_losses = []
        self.policy_losses = []
        self.explained_variances = []
        self.clip_fractions = []
        self.episode_rewards = []

        # Advanced tactical metrics
        self.win_rates = []
        self.bait_success_rates = []
        self.punish_success_rates = []
        self.whiff_punishes = []
        self.frame_perfect_blocks = []
        self.optimal_spacing_ratios = []

        # Stability monitoring
        self.value_loss_threshold = 20.0
        self.stability_warnings = 0
        self.training_start_time = None
        self.best_win_rate = 0.0
        self.best_tactical_score = 0.0
        self.consecutive_stable_steps = 0
        self.required_stable_steps = 15000

        # Performance tracking
        self.performance_history = {
            "win_rate": [],
            "tactical_effectiveness": [],
            "learning_progress": [],
        }

    def _on_training_start(self):
        """Initialize training with advanced monitoring"""
        self.training_start_time = datetime.now()
        print(f"🥊 ADVANCED STREET FIGHTER TRAINING STARTED (FULL FRAME)")
        print("=" * 60)
        print(f"🎯 Goals:")
        print(f"   - Learn proper baiting techniques")
        print(f"   - Master blocking and punishment")
        print(f"   - Develop advanced spacing control")
        print(f"   - Achieve stable value loss < {self.value_loss_threshold}")
        print(f"🔧 Advanced Features:")
        print(f"   - Full frame size: 320x224 (no scaling)")
        print(f"   - 42-dimensional feature space")
        print(f"   - Research-based baiting system")
        print(f"   - Frame-accurate blocking detection")
        print(f"   - Move-specific punishment tracking")
        print(f"   - Updated pixel values for baiting ranges")
        print()

        # Initial stability check
        self._perform_initial_checks()

    def _perform_initial_checks(self):
        """Perform initial system checks"""
        print("🔬 Initial System Checks (Full Frame)")
        print("-" * 30)

        env = self._get_env()
        if env:
            try:
                stable = verify_gradient_flow(self.model, env)
                if stable:
                    print("   ✅ Gradient flow stable")
                else:
                    print("   ⚠️  Gradient flow issues detected")
                    
                # Check frame size
                obs = env.reset()[0]
                visual_shape = obs["visual_obs"].shape
                print(f"   📺 Frame size: {visual_shape[1]}x{visual_shape[2]} (Full Frame)")
                
                if visual_shape[1] == 224 and visual_shape[2] == 320:
                    print("   ✅ Full frame size confirmed")
                else:
                    print("   ⚠️  Unexpected frame size")
                    
            except Exception as e:
                print(f"   ❌ System check failed: {e}")
        print()

    def _get_env(self):
        """Get environment for testing"""
        if hasattr(self.training_env, "envs"):
            return self.training_env.envs[0]
        elif hasattr(self.training_env, "env"):
            return self.training_env.env
        else:
            return self.training_env

    def _on_step(self) -> bool:
        """Monitor each training step"""

        # Extract training metrics
        self._extract_training_metrics()

        # Extract tactical performance metrics
        self._extract_tactical_metrics()

        # Stability checks every 1000 steps
        if self.num_timesteps % 1000 == 0:
            if not self._check_training_stability():
                print("🚨 CRITICAL: Training unstable - stopping!")
                return False

        # Detailed reporting every 10000 steps
        if self.num_timesteps % 10000 == 0:
            self._log_advanced_report()

        # Save model checkpoints
        if self.num_timesteps > 0 and self.num_timesteps % self.save_freq == 0:
            self._save_advanced_checkpoint()

        return True

    def _extract_training_metrics(self):
        """Extract core training metrics"""
        if hasattr(self.logger, "name_to_value"):
            metrics = self.logger.name_to_value

            # Core training metrics
            if "train/value_loss" in metrics:
                value_loss = metrics["train/value_loss"]
                self.value_losses.append(value_loss)
                if len(self.value_losses) > 100:
                    self.value_losses.pop(0)

            if "train/policy_loss" in metrics:
                policy_loss = metrics["train/policy_loss"]
                self.policy_losses.append(policy_loss)
                if len(self.policy_losses) > 100:
                    self.policy_losses.pop(0)

            if "train/explained_variance" in metrics:
                explained_var = metrics["train/explained_variance"]
                self.explained_variances.append(explained_var)
                if len(self.explained_variances) > 100:
                    self.explained_variances.pop(0)

            if "train/clip_fraction" in metrics:
                clip_frac = metrics["train/clip_fraction"]
                self.clip_fractions.append(clip_frac)
                if len(self.clip_fractions) > 100:
                    self.clip_fractions.pop(0)

    def _extract_tactical_metrics(self):
        """Extract advanced tactical metrics"""
        if hasattr(self.locals, "infos") and self.locals["infos"]:
            for info in self.locals["infos"]:
                # Basic performance
                if "episode" in info:
                    self.episode_rewards.append(info["episode"]["r"])
                    if len(self.episode_rewards) > 100:
                        self.episode_rewards.pop(0)

                if "win_rate" in info:
                    win_rate = info["win_rate"]
                    self.win_rates.append(win_rate)
                    if len(self.win_rates) > 50:
                        self.win_rates.pop(0)

                    if win_rate > self.best_win_rate:
                        self.best_win_rate = win_rate

                # Advanced tactical metrics
                if "bait_success_rate" in info:
                    self.bait_success_rates.append(info["bait_success_rate"])
                    if len(self.bait_success_rates) > 50:
                        self.bait_success_rates.pop(0)

                if "punish_success_rate" in info:
                    self.punish_success_rates.append(info["punish_success_rate"])
                    if len(self.punish_success_rates) > 50:
                        self.punish_success_rates.pop(0)

                if "whiff_punishes" in info:
                    self.whiff_punishes.append(info["whiff_punishes"])
                    if len(self.whiff_punishes) > 50:
                        self.whiff_punishes.pop(0)

                if "frame_perfect_blocks" in info:
                    self.frame_perfect_blocks.append(info["frame_perfect_blocks"])
                    if len(self.frame_perfect_blocks) > 50:
                        self.frame_perfect_blocks.pop(0)

                if "optimal_spacing_ratio" in info:
                    self.optimal_spacing_ratios.append(info["optimal_spacing_ratio"])
                    if len(self.optimal_spacing_ratios) > 50:
                        self.optimal_spacing_ratios.pop(0)

    def _check_training_stability(self) -> bool:
        """Check if training remains stable"""
        if not self.value_losses:
            return True

        recent_value_loss = np.mean(self.value_losses[-5:])

        # Critical stability check
        if recent_value_loss > self.value_loss_threshold:
            self.stability_warnings += 1
            print(
                f"⚠️  STABILITY WARNING #{self.stability_warnings}: Value loss = {recent_value_loss:.2f}"
            )

            if self.stability_warnings >= 3:
                print(
                    "🚨 CRITICAL: Multiple stability warnings - training may be diverging"
                )
                return False

            self.consecutive_stable_steps = 0
        else:
            self.consecutive_stable_steps += 1000

        # Check for catastrophic failure
        if recent_value_loss > self.value_loss_threshold * 3:
            print(
                f"🚨 CRITICAL: Value loss catastrophically high ({recent_value_loss:.2f}) - stopping"
            )
            return False

        return True

    def _log_advanced_report(self):
        """Generate comprehensive training report"""
        print(f"\n🥊 ADVANCED TACTICAL REPORT (Full Frame) - Step {self.num_timesteps:,}")
        print("=" * 60)

        if self.training_start_time:
            elapsed = datetime.now() - self.training_start_time
            print(f"⏱️  Training Time: {elapsed.total_seconds() / 3600:.1f} hours")

        # Core training stability
        print(f"\n📊 TRAINING STABILITY:")
        if self.value_losses:
            recent_value_loss = np.mean(self.value_losses[-10:])
            print(f"   🎯 Value Loss: {recent_value_loss:.3f}", end="")
            if recent_value_loss < 5.0:
                print(" ✅ EXCELLENT")
            elif recent_value_loss < self.value_loss_threshold:
                print(" ✅ STABLE")
            else:
                print(" ⚠️  UNSTABLE")

        if self.explained_variances:
            recent_explained_var = np.mean(self.explained_variances[-10:])
            print(f"   📈 Explained Variance: {recent_explained_var:.3f}")

        if self.clip_fractions:
            recent_clip_frac = np.mean(self.clip_fractions[-10:])
            print(f"   ✂️  Clip Fraction: {recent_clip_frac:.3f}")

        # Performance metrics
        print(f"\n🎮 COMBAT PERFORMANCE:")
        if self.win_rates:
            recent_win_rate = np.mean(self.win_rates[-10:])
            print(
                f"   🏆 Win Rate: {recent_win_rate:.1%} (Best: {self.best_win_rate:.1%})"
            )

        if self.episode_rewards:
            recent_reward = np.mean(self.episode_rewards[-20:])
            print(f"   📈 Avg Episode Reward: {recent_reward:.3f}")

        # Advanced tactical analysis
        print(f"\n🥋 TACTICAL MASTERY (Updated Ranges):")

        if self.bait_success_rates:
            recent_bait_rate = np.mean(self.bait_success_rates[-10:])
            print(f"   🎯 Baiting Success: {recent_bait_rate:.1%}", end="")
            if recent_bait_rate > 0.4:
                print(" ✅ EXCELLENT")
            elif recent_bait_rate > 0.2:
                print(" 🔄 LEARNING")
            else:
                print(" 📚 DEVELOPING")

        if self.punish_success_rates:
            recent_punish_rate = np.mean(self.punish_success_rates[-10:])
            print(f"   🛡️  Punishment Success: {recent_punish_rate:.1%}", end="")
            if recent_punish_rate > 0.5:
                print(" ✅ EXCELLENT")
            elif recent_punish_rate > 0.3:
                print(" 🔄 LEARNING")
            else:
                print(" 📚 DEVELOPING")

        if self.whiff_punishes:
            total_whiff_punishes = sum(self.whiff_punishes[-10:])
            print(f"   ⚔️  Whiff Punishes: {total_whiff_punishes}")

        if self.frame_perfect_blocks:
            total_perfect_blocks = sum(self.frame_perfect_blocks[-10:])
            print(f"   🛡️  Perfect Blocks: {total_perfect_blocks}")

        if self.optimal_spacing_ratios:
            recent_spacing = np.mean(self.optimal_spacing_ratios[-10:])
            print(f"   📏 Optimal Spacing: {recent_spacing:.1%}")

        # Calculate tactical effectiveness score
        tactical_score = self._calculate_tactical_effectiveness()
        print(f"\n🎖️  TACTICAL EFFECTIVENESS SCORE: {tactical_score:.1f}/10.0")

        if tactical_score > self.best_tactical_score:
            self.best_tactical_score = tactical_score
            print(f"   🌟 NEW BEST TACTICAL SCORE!")

        # Learning progress analysis
        self._analyze_learning_progress()

        print(
            f"\n🔒 Stability: {self.consecutive_stable_steps} consecutive stable steps"
        )
        print()

    def _calculate_tactical_effectiveness(self) -> float:
        """Calculate overall tactical effectiveness score (0-10)"""
        score = 0.0
        components = 0

        # Win rate component (0-3 points)
        if self.win_rates:
            win_rate = np.mean(self.win_rates[-10:])
            score += min(3.0, win_rate * 6)  # 50% win rate = 3 points
            components += 1

        # Baiting component (0-2 points)
        if self.bait_success_rates:
            bait_rate = np.mean(self.bait_success_rates[-10:])
            score += min(2.0, bait_rate * 5)  # 40% bait success = 2 points
            components += 1

        # Punishment component (0-2 points)
        if self.punish_success_rates:
            punish_rate = np.mean(self.punish_success_rates[-10:])
            score += min(2.0, punish_rate * 4)  # 50% punish success = 2 points
            components += 1

        # Spacing component (0-1.5 points)
        if self.optimal_spacing_ratios:
            spacing_ratio = np.mean(self.optimal_spacing_ratios[-10:])
            score += min(1.5, spacing_ratio * 2.5)  # 60% optimal spacing = 1.5 points
            components += 1

        # Advanced techniques component (0-1.5 points)
        if self.whiff_punishes and self.frame_perfect_blocks:
            whiff_count = np.mean(self.whiff_punishes[-10:])
            block_count = np.mean(self.frame_perfect_blocks[-10:])
            advanced_score = min(1.5, (whiff_count + block_count) * 0.1)
            score += advanced_score
            components += 1

        return score if components > 0 else 0.0

    def _analyze_learning_progress(self):
        """Analyze learning progress over time"""
        if len(self.win_rates) > 20:
            early_performance = np.mean(self.win_rates[:10])
            recent_performance = np.mean(self.win_rates[-10:])
            improvement = recent_performance - early_performance

            print(f"\n📈 LEARNING PROGRESS:")
            print(f"   Early Win Rate: {early_performance:.1%}")
            print(f"   Recent Win Rate: {recent_performance:.1%}")

            if improvement > 0.1:
                print(f"   🚀 SIGNIFICANT IMPROVEMENT: +{improvement:.1%}")
            elif improvement > 0.05:
                print(f"   📈 GOOD PROGRESS: +{improvement:.1%}")
            elif improvement > 0.01:
                print(f"   📊 STEADY PROGRESS: +{improvement:.1%}")
            elif improvement > -0.05:
                print(f"   🔄 STABLE PERFORMANCE: {improvement:+.1%}")
            else:
                print(f"   📉 DECLINING: {improvement:+.1%}")

    def _save_advanced_checkpoint(self):
        """Save model with advanced metadata"""
        model_name = f"sf_advanced_fullframe_{self.num_timesteps}"
        model_path = os.path.join(self.save_path, f"{model_name}.zip")

        try:
            self.model.save(model_path)
            print(f"💾 Checkpoint: {model_name}.zip")

            # Save best tactical model
            tactical_score = self._calculate_tactical_effectiveness()
            if (
                tactical_score > 6.0
                and self.value_losses
                and np.mean(self.value_losses[-5:]) < 10.0
            ):

                best_path = os.path.join(self.save_path, "best_tactical_fullframe_model.zip")
                self.model.save(best_path)
                print(
                    f"   🌟 Saved as best tactical model (score: {tactical_score:.1f})"
                )

            # Save stable model
            if self.consecutive_stable_steps > 20000 and self.best_win_rate > 0.3:

                stable_path = os.path.join(self.save_path, "stable_advanced_fullframe_model.zip")
                self.model.save(stable_path)
                print(f"   🛡️  Saved as stable model")

        except Exception as e:
            print(f"❌ Failed to save checkpoint: {e}")


def make_advanced_env(
    game="StreetFighterIISpecialChampionEdition-Genesis",
    state="ken_bison_12.state",
    render_mode=None,
):
    """Create advanced Street Fighter environment with full frame support"""
    try:
        env = retro.make(game=game, state=state, render_mode=render_mode)
        env = Monitor(env)
        env = StreetFighterVisionWrapper(
            env,
            frame_stack=4,
            rendering=(render_mode is not None),
        )
        
        # Verify full frame size
        obs = env.reset()[0]
        visual_shape = obs["visual_obs"].shape
        print(f"🎮 Environment created with frame size: {visual_shape[1]}x{visual_shape[2]}")
        
        return env
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Advanced Street Fighter Training (Full Frame)")
    parser.add_argument(
        "--total-timesteps", type=int, default=1000000, help="Total training timesteps"
    )
    parser.add_argument("--resume", type=str, default=None, help="Path to resume from")
    parser.add_argument("--render", action="store_true", help="Render during training")
    parser.add_argument(
        "--device", type=str, default="auto", help="Device (auto, cpu, cuda)"
    )

    # Advanced hyperparameters
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--n-steps", type=int, default=2048, help="Steps per rollout")
    parser.add_argument("--n-epochs", type=int, default=4, help="Training epochs")
    parser.add_argument("--clip-range", type=float, default=0.2, help="PPO clip range")
    parser.add_argument(
        "--vf-coef", type=float, default=0.5, help="Value function coefficient"
    )

    args = parser.parse_args()

    print("🥊 ADVANCED STREET FIGHTER AI TRAINING (FULL FRAME)")
    print("=" * 50)
    print("🎯 Training Goals:")
    print("   - Master baiting techniques with updated ranges")
    print("   - Perfect blocking and punishment")
    print("   - Develop advanced spacing (320x224 full frame)")
    print("   - Achieve consistent wins")
    print()
    print("🔧 Technical Updates:")
    print("   - Full frame size: 320x224 (no downscaling)")
    print("   - Updated psycho crusher bait range: 90-140 pixels")
    print("   - Updated head stomp bait range: 45-90 pixels")
    print("   - Enhanced CNN for full frame processing")
    print()
    print("🔧 Hyperparameters:")
    print(f"   - Learning Rate: {args.learning_rate}")
    print(f"   - Batch Size: {args.batch_size}")
    print(f"   - Rollout Steps: {args.n_steps}")
    print(f"   - Training Epochs: {args.n_epochs}")
    print(f"   - Clip Range: {args.clip_range}")
    print(f"   - Value Coefficient: {args.vf_coef}")
    print()

    # Setup device
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    )
    print(f"🔧 Device: {device}")
    set_random_seed(42)

    # Create environment
    render_mode = "human" if args.render else None
    env = make_advanced_env(render_mode=render_mode)
    print("✅ Advanced environment created successfully")

    # Create or load model
    model = None
    if args.resume and os.path.exists(args.resume):
        print(f"📂 Resuming from {args.resume}")
        try:
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
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print("🔄 Creating new model instead...")
            model = None

    if model is None:
        print("🆕 Creating new advanced model (full frame)...")
        model = PPO(
            AdvancedStreetFighterPolicy,  # Updated policy with full frame support
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=args.clip_range,
            ent_coef=0.01,
            vf_coef=args.vf_coef,
            max_grad_norm=0.5,
            verbose=1,
            device=device,
        )
        print("✅ New advanced model created")

    # Pre-training stability test
    print("\n🔬 Pre-training system verification (full frame)...")
    try:
        obs = env.reset()[0]
        obs_tensor = {}
        for key, value in obs.items():
            obs_tensor[key] = torch.from_numpy(value).unsqueeze(0).float().to(device)

        with torch.no_grad():
            actions, values, log_probs = model.policy(obs_tensor)

        print(f"✅ System verification passed")
        print(f"   Initial value: {values.item():.3f}")
        print(f"   Feature dimensions verified: {obs['vector_obs'].shape}")
        print(f"   Visual dimensions verified: {obs['visual_obs'].shape}")

        # Verify frame size
        visual_shape = obs['visual_obs'].shape
        if visual_shape[1] == 224 and visual_shape[2] == 320:
            print(f"   ✅ Full frame size confirmed: {visual_shape[1]}x{visual_shape[2]}")
        else:
            print(f"   ⚠️  Unexpected frame size: {visual_shape[1]}x{visual_shape[2]}")

    except Exception as e:
        print(f"❌ System verification failed: {e}")
        return

    # Create advanced callback
    callback = AdvancedTacticsCallback(save_freq=50000, save_path="./models/")

    print("\n🚀 STARTING ADVANCED TRAINING (FULL FRAME)...")
    print("📊 Monitoring: Baiting, Blocking, Spacing, Win Rate")
    print("🛡️  Safety: Auto-stop on training instability")
    print("🎯 Tactical: Updated pixel ranges for all moves")
    print()

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            reset_num_timesteps=not args.resume,
            progress_bar=True,
        )

        # Save final model
        final_path = "./models/final_advanced_sf_fullframe_model.zip"
        model.save(final_path)
        print(f"🎉 Advanced training completed successfully!")
        print(f"💾 Final model saved: {final_path}")

        # Final comprehensive report
        if callback.value_losses:
            final_value_loss = np.mean(callback.value_losses[-10:])
            print(f"📊 Final value loss: {final_value_loss:.3f}")

        if callback.win_rates:
            final_win_rate = callback.best_win_rate
            print(f"🏆 Best win rate: {final_win_rate:.1%}")

        final_tactical_score = callback._calculate_tactical_effectiveness()
        print(f"🎖️  Final tactical score: {final_tactical_score:.1f}/10.0")

        # Frame size summary
        obs = env.reset()[0]
        visual_shape = obs["visual_obs"].shape
        print(f"📺 Training completed with frame size: {visual_shape[1]}x{visual_shape[2]}")

    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        model.save("./models/interrupted_advanced_fullframe_model.zip")
        print("💾 Progress saved")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback

        traceback.print_exc()

        # Try to save model even after error
        try:
            model.save("./models/error_advanced_fullframe_model.zip")
            print("💾 Model saved despite error")
        except:
            print("❌ Could not save model")

    finally:
        env.close()
        print("🔚 Advanced training session ended")


if __name__ == "__main__":
    main()