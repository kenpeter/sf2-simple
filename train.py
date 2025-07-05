#!/usr/bin/env python3
"""
train.py - REFACTORED for a SINGLE ENVIRONMENT with FIXED observation space handling.
KEY FIXES:
1. CORRECTED Policy to "MultiInputPolicy" to match the Dict observation space.
2. REMOVED all vectorized environment logic (DummyVecEnv).
3. SIMPLIFIED the training loop and callback to work with a single env instance.
4. HARD-CODED --n-envs to 1 to prevent misuse.
5. ENSURED full compatibility with the --render flag.
6. FIXED gradient monitoring and parameter counting.
7. IMPROVED error handling and diagnostics.
"""
import os
import argparse
import torch
import numpy as np
import json
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
import retro
import logging
import gymnasium as gym

# Import the fixed wrapper and components
from wrapper import (
    StreetFighterVisionWrapper,
    StreetFighterCrossAttentionCNN,
    monitor_gradients,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SingleEnvCallback(BaseCallback):
    """Enhanced callback adapted for a single environment with better diagnostics."""

    def __init__(
        self,
        save_freq=100000,
        save_path="./enhanced_oscillation_trained_models/",
        verbose=1,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        self.last_episode_count = 0
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # Log gradients periodically
        if self.num_timesteps % 5000 == 0:
            monitor_gradients(self.model, self.num_timesteps)

        # Log performance stats
        if self.num_timesteps % 10000 == 0:
            self._log_performance_stats()

        # Save model periodically
        if self.num_timesteps > 0 and self.num_timesteps % self.save_freq == 0:
            self._save_model()

        # Check for episode completion
        if hasattr(self.locals, "infos") and self.locals["infos"]:
            for info in self.locals["infos"]:
                if "episode" in info:
                    episode_info = info["episode"]
                    self.episode_rewards.append(episode_info["r"])
                    self.episode_lengths.append(episode_info["l"])

                    # Keep only last 100 episodes
                    if len(self.episode_rewards) > 100:
                        self.episode_rewards.pop(0)
                        self.episode_lengths.pop(0)

        return True

    def _log_performance_stats(self):
        """Enhanced performance logging with episode statistics."""
        print(f"\n--- 📊 Step {self.num_timesteps} - Performance Analysis ---")

        # Episode statistics
        if self.episode_rewards:
            avg_reward = np.mean(self.episode_rewards[-10:])  # Last 10 episodes
            avg_length = np.mean(self.episode_lengths[-10:])
            print(
                f"  🎮 Recent Episodes - Avg Reward: {avg_reward:.2f}, Avg Length: {avg_length:.0f}"
            )

        # Get wrapper stats if available
        if hasattr(self.training_env, "stats"):
            stats = self.training_env.stats
            win_rate = stats.get("win_rate", 0.0)
            wins = stats.get("wins", 0)
            losses = stats.get("losses", 0)
            total_games = stats.get("total_games", 0)

            print(
                f"  🎯 Win Rate: {win_rate:.1%} ({wins}W/{losses}L over {total_games} games)"
            )

            avg_damage = stats.get("avg_damage_per_round", 0)
            max_combo = stats.get("max_combo", 0)
            avg_freq = stats.get("player_oscillation_frequency", 0)
            avg_space_control = stats.get("space_control_score", 0)

            print(
                f"  ⚡ Avg Damage/Round: {avg_damage:.1f} | 🔥 Max Combo: {max_combo}"
            )
            print(
                f"  🌊 Avg Oscillation Freq: {avg_freq:.3f} Hz | 🎯 Avg Space Control: {avg_space_control:.3f}"
            )

        # Model statistics
        if hasattr(self.model, "policy"):
            total_params = sum(p.numel() for p in self.model.policy.parameters())
            trainable_params = sum(
                p.numel() for p in self.model.policy.parameters() if p.requires_grad
            )
            print(
                f"  🧠 Model: {trainable_params:,} trainable / {total_params:,} total parameters"
            )

    def _save_model(self):
        """Enhanced model saving with comprehensive stats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if not hasattr(self.training_env, "stats"):
            model_path = os.path.join(
                self.save_path, f"sf2_ppo_{self.num_timesteps}_steps_{timestamp}.zip"
            )
            self.model.save(model_path)
            print(f"💾 Model saved (no stats available): {model_path}")
            return

        stats = self.training_env.stats
        win_rate = stats.get("win_rate", 0.0)
        total_games = stats.get("total_games", 0)
        avg_freq = stats.get("player_oscillation_frequency", 0.0)
        avg_reward = (
            np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0.0
        )

        model_name = (
            f"sf2_ppo_{self.num_timesteps}_steps_"
            f"winrate_{win_rate:.1%}_"
            f"games_{total_games}_"
            f"freq_{avg_freq:.2f}Hz_"
            f"reward_{avg_reward:.1f}_"
            f"{timestamp}"
        )
        model_path = os.path.join(self.save_path, f"{model_name}.zip")
        self.model.save(model_path)
        print(f"💾 Model saved: {model_path}")

        # Save training statistics
        stats_path = os.path.join(self.save_path, f"stats_{timestamp}.json")
        training_stats = {
            "timesteps": self.num_timesteps,
            "win_rate": win_rate,
            "total_games": total_games,
            "avg_oscillation_frequency": avg_freq,
            "recent_episode_rewards": (
                self.episode_rewards[-10:] if self.episode_rewards else []
            ),
            "recent_episode_lengths": (
                self.episode_lengths[-10:] if self.episode_lengths else []
            ),
        }
        with open(stats_path, "w") as f:
            json.dump(training_stats, f, indent=2)


def make_env(
    game="StreetFighterIISpecialChampionEdition-Genesis",
    state="ken_bison_12.state",
    render_mode=None,
):
    """Creates and wraps a single instance of the Street Fighter environment."""
    try:
        env = retro.make(game=game, state=state, render_mode=render_mode)
        env = Monitor(env)
        env = StreetFighterVisionWrapper(
            env, frame_stack=8, rendering=(render_mode is not None)
        )

        # Verify observation space
        obs_space = env.observation_space
        print(f"📏 Environment observation space: {obs_space}")
        if isinstance(obs_space, gym.spaces.Dict):
            for key, space in obs_space.spaces.items():
                print(f"   - {key}: {space}")

        return env
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Train Street Fighter AI with enhanced cross-attention"
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=3000000, help="Total training timesteps"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the game during training"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to model to resume training from"
    )
    parser.add_argument(
        "--game",
        type=str,
        default="StreetFighterIISpecialChampionEdition-Genesis",
        help="Game to train on",
    )
    parser.add_argument(
        "--state", type=str, default="ken_bison_12.state", help="Game state file"
    )
    parser.add_argument(
        "--save-freq", type=int, default=100000, help="Model save frequency"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)"
    )
    args = parser.parse_args()

    print("🚀 Starting Street Fighter Training (SINGLE ENVIRONMENT MODE - FIXED)")
    print(f"   - Total timesteps: {args.total_timesteps:,}")
    print(f"   - Learning rate: {args.learning_rate}")
    print(f"   - Device: {args.device}")
    print(f"   - Rendering: {'Enabled' if args.render else 'Disabled'}")

    set_random_seed(42)

    # Create environment
    render_mode = "human" if args.render else None
    env = make_env(args.game, args.state, render_mode)

    # Test environment
    print("\n🧪 Testing environment...")
    try:
        obs = env.reset()
        print(f"   - Reset successful, observation type: {type(obs)}")
        if isinstance(obs, tuple):
            obs = obs[0]  # Handle new gym API

        if isinstance(obs, dict):
            print("   - Dict observation detected:")
            for key, value in obs.items():
                print(
                    f"     - {key}: {value.shape if hasattr(value, 'shape') else type(value)}"
                )

        # Test a step
        action = env.action_space.sample()
        step_result = env.step(action)
        print(f"   - Step successful, result length: {len(step_result)}")

    except Exception as e:
        print(f"   ❌ Environment test failed: {e}")
        raise

    # Configure policy
    policy_kwargs = dict(
        features_extractor_class=StreetFighterCrossAttentionCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.ReLU,
    )

    # Create or load model
    if args.resume and os.path.exists(args.resume):
        print(f"📂 Resuming training from {args.resume}")
        try:
            model = PPO.load(
                args.resume,
                env=env,
                custom_objects={
                    "policy_kwargs": policy_kwargs,
                    "learning_rate": args.learning_rate,
                },
                device=args.device,
            )
            print("   ✅ Model loaded successfully")
        except Exception as e:
            print(f"   ❌ Failed to load model: {e}")
            print("   🆕 Creating new model instead...")
            model = None
    else:
        model = None

    if model is None:
        print(f"🆕 Creating new model with learning rate: {args.learning_rate}")
        try:
            model = PPO(
                "MultiInputPolicy",  # FIXED: Use MultiInputPolicy for Dict observation space
                env,
                learning_rate=args.learning_rate,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.98,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                policy_kwargs=policy_kwargs,
                verbose=1,
                device=args.device,
            )
            print("   ✅ Model created successfully")
        except Exception as e:
            print(f"   ❌ Failed to create model: {e}")
            raise

    # Verify model architecture
    print("\n🏗️ Model Architecture:")
    total_params = sum(p.numel() for p in model.policy.parameters())
    trainable_params = sum(
        p.numel() for p in model.policy.parameters() if p.requires_grad
    )
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")
    print(f"   - Policy type: {type(model.policy).__name__}")

    # Create callback
    callback = SingleEnvCallback(
        save_freq=args.save_freq,
        save_path="./enhanced_oscillation_trained_models/",
        verbose=1,
    )

    print("\n🎯 Starting training...")
    if args.render:
        print("   - RENDERING ENABLED. Training will be much slower.")
    print("   - Using single environment with fixed observation space.")
    print("   - Press Ctrl+C to stop training and save the model.")

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            reset_num_timesteps=not args.resume,
            progress_bar=True,
        )

        # Save final model
        final_model_path = (
            "./enhanced_oscillation_trained_models/final_model_single_env_fixed.zip"
        )
        model.save(final_model_path)
        print(f"🎉 Training completed! Final model saved: {final_model_path}")

    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user.")
        interrupted_model_path = "./enhanced_oscillation_trained_models/interrupted_model_single_env_fixed.zip"
        model.save(interrupted_model_path)
        print(f"💾 Model saved before exit: {interrupted_model_path}")

    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        error_model_path = (
            "./enhanced_oscillation_trained_models/error_model_single_env_fixed.zip"
        )
        model.save(error_model_path)
        print(f"💾 Model saved after error: {error_model_path}")
        raise

    finally:
        print("\n🧹 Cleaning up...")
        env.close()
        print("   ✅ Environment closed.")


if __name__ == "__main__":
    main()
