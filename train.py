#!/usr/bin/env python3
"""
train.py - REFACTORED for a SINGLE ENVIRONMENT to support rendering.
KEY FIXES:
1. CORRECTED Policy to "MultiInputPolicy" to match the Dict observation space.
2. REMOVED all vectorized environment logic (DummyVecEnv).
3. SIMPLIFIED the training loop and callback to work with a single env instance.
4. HARD-CODED --n-envs to 1 to prevent misuse.
5. ENSURED full compatibility with the --render flag.
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
    """Callback adapted for a single environment."""

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

    def _on_step(self) -> bool:
        # Log gradients periodically
        if self.num_timesteps % 5000 == 0:
            monitor_gradients(self.model, self.num_timesteps)

        # Log performance stats from the Monitor wrapper's info buffer
        if self.num_timesteps % 10000 == 0:
            self._log_performance_stats()

        # Save model periodically
        if self.num_timesteps > 0 and self.num_timesteps % self.save_freq == 0:
            self._save_model()
        return True

    def _log_performance_stats(self):
        # In a single-env setup with Monitor, episode stats are in `info['episode']` when an episode ends
        # For periodic logging, we access the wrapper's internal stats directly.
        if hasattr(self.training_env, "stats"):
            stats = self.training_env.stats
            win_rate = stats.get("win_rate", 0.0)
            wins = stats.get("wins", 0)
            losses = stats.get("losses", 0)
            total_games = stats.get("total_games", 0)

            print(f"\n--- 📊 Step {self.num_timesteps} - Performance Analysis ---")
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

    def _save_model(self):
        if not hasattr(self.training_env, "stats"):
            model_path = os.path.join(
                self.save_path, f"sf2_ppo_{self.num_timesteps}_steps.zip"
            )
            self.model.save(model_path)
            print(f"💾 Model saved (no stats available): {model_path}")
            return

        stats = self.training_env.stats
        win_rate = stats.get("win_rate", 0.0)
        total_games = stats.get("total_games", 0)
        avg_freq = stats.get("player_oscillation_frequency", 0.0)

        model_name = (
            f"sf2_ppo_{self.num_timesteps}_steps_"
            f"winrate_{win_rate:.1%}_"
            f"games_{total_games}_"
            f"freq_{avg_freq:.2f}Hz"
        )
        model_path = os.path.join(self.save_path, f"{model_name}.zip")
        self.model.save(model_path)
        print(f"💾 Model saved: {model_path}")


def make_env(
    game="StreetFighterIISpecialChampionEdition-Genesis",
    state="ken_bison_12.state",
    render_mode=None,
):
    """Creates and wraps a single instance of the Street Fighter environment."""
    env = retro.make(game=game, state=state, render_mode=render_mode)
    env = Monitor(env)
    env = StreetFighterVisionWrapper(
        env, frame_stack=8, rendering=(render_mode is not None)
    )
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=3000000)
    parser.add_argument("--learning-rate", type=float, default=4e-4)
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the game during training. Slows down training significantly.",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to model to resume training from"
    )
    parser.add_argument(
        "--game", type=str, default="StreetFighterIISpecialChampionEdition-Genesis"
    )
    parser.add_argument("--state", type=str, default="ken_bison_12.state")
    args = parser.parse_args()

    print("🚀 Starting Street Fighter Training (SINGLE ENVIRONMENT MODE)")
    set_random_seed(42)

    render_mode = "human" if args.render else None
    env = make_env(args.game, args.state, render_mode)

    policy_kwargs = dict(
        features_extractor_class=StreetFighterCrossAttentionCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.ReLU,
    )

    if args.resume and os.path.exists(args.resume):
        print(f"📂 Resuming training from {args.resume}")
        model = PPO.load(
            args.resume,
            env=env,
            custom_objects={
                "policy_kwargs": policy_kwargs,
                "learning_rate": args.learning_rate,
            },
        )
    else:
        print(f"🆕 Creating new model with learning rate: {args.learning_rate}")
        # ===================================================================
        # THE FIX IS HERE: Changed "CnnPolicy" to "MultiInputPolicy"
        # ===================================================================
        model = PPO(
            "MultiInputPolicy",  # <-- THIS IS THE FIX
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
            device="auto",
        )

    callback = SingleEnvCallback(
        save_freq=100000, save_path="./enhanced_oscillation_trained_models/"
    )

    print("\n🎯 Starting training...")
    if args.render:
        print("   - RENDERING ENABLED. Training will be much slower.")
    print("   - Using single environment.")

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            reset_num_timesteps=not args.resume,
        )
        model.save("./enhanced_oscillation_trained_models/final_model_single_env.zip")
        print("🎉 Training completed!")

    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user.")
        model.save(
            "./enhanced_oscillation_trained_models/interrupted_model_single_env.zip"
        )
        print("💾 Model saved before exit.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
