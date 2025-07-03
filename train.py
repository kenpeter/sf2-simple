#!/usr/bin/env python3
"""
Enhanced CUDA-Optimized Training Script for Street Fighter II with Cross-Attention
Key improvements:
1. Cross-attention Vision Transformer integration
2. Better learning rate scheduling for complex attention mechanisms
3. Enhanced reward system with combo bonuses
4. Improved action space for more effective gameplay
5. Better training hyperparameters based on cross-attention analysis
6. Enhanced curriculum learning approach
7. File-based logging with attention analysis
"""

import os
import sys
import argparse
import time
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime

import retro
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# Import the enhanced wrapper with cross-attention
from wrapper import StreetFighterVisionWrapper, StreetFighterCrossAttentionCNN

# Create directories
os.makedirs("logs", exist_ok=True)
os.makedirs("analysis_data", exist_ok=True)


def setup_cuda_optimization():
    """Setup CUDA for optimal performance"""
    if not torch.cuda.is_available():
        print("⚠️ CUDA not available! Running on CPU. This will be very slow.")
        return torch.device("cpu")

    # Set CUDA device
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)

    # Enable optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    # Set memory allocation strategy
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    print(f"🚀 CUDA Setup Complete:")
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   cuDNN Enabled: {torch.backends.cudnn.enabled}")
    print(f"   cuDNN Benchmark: {torch.backends.cudnn.benchmark}")

    return device


class EnhancedCrossAttentionTrainingCallback(BaseCallback):
    """Enhanced callback with cross-attention training monitoring and analysis"""

    def __init__(self, enable_vision_transformer=True, verbose=0):
        super().__init__(verbose)
        self.enable_vision_transformer = enable_vision_transformer
        self.last_log_time = time.time()
        self.log_interval = 120  # Log every 2 minutes
        self.win_rate_history = []
        self.performance_milestones = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        self.achieved_milestones = set()

    def _on_step(self):
        current_time = time.time()
        if current_time - self.last_log_time > self.log_interval:
            self.last_log_time = current_time
            self._log_enhanced_cross_attention_stats()
            self._check_performance_milestones()
        return True

    def _log_enhanced_cross_attention_stats(self):
        print(f"\n🎯 Cross-Attention Training Step: {self.num_timesteps:,}")

        if hasattr(self.training_env, "get_attr"):
            all_stats = self.training_env.get_attr("stats")
            if not all_stats:
                return
            stats = all_stats[0]

            wrapper_env = self.training_env.envs[0].env
            win_rate = wrapper_env.wins / max(wrapper_env.total_rounds, 1)
            self.win_rate_history.append(win_rate)

            print(
                f"  🏆 Win Rate: {win_rate:.1%} ({wrapper_env.wins}/{wrapper_env.total_rounds})"
            )
            if stats.get("max_combo", 0) > 0:
                print(f"  🔥 Max Combo: {stats.get('max_combo', 0)}")

            if stats.get("cross_attention_ready", False):
                print(
                    f"  🎯 Attention (Vis/Str/Btn): {stats.get('visual_attention_weight', 0.0):.3f} / "
                    f"{stats.get('strategy_attention_weight', 0.0):.3f} / "
                    f"{stats.get('button_attention_weight', 0.0):.3f}"
                )

    def _check_performance_milestones(self):
        if not self.win_rate_history:
            return
        current_win_rate = self.win_rate_history[-1]
        for milestone in self.performance_milestones:
            if (
                milestone not in self.achieved_milestones
                and current_win_rate >= milestone
            ):
                self.achieved_milestones.add(milestone)
                print(f"🎉 CROSS-ATTENTION MILESTONE: {milestone:.0%} Win Rate!")
                if hasattr(self.model, "save"):
                    path = f"enhanced_trained_models/cross_attention_milestone_{milestone:.0%}.zip"
                    self.model.save(path)
                    print(f"💾 Milestone model saved to {path}")


def create_learning_rate_schedule(initial_lr=3e-4):
    """Create a learning rate schedule optimized for complex policies."""

    def schedule(progress):
        if progress < 0.2:
            return initial_lr
        elif progress < 0.6:
            return initial_lr * 0.5
        elif progress < 0.9:
            return initial_lr * 0.25
        else:
            return initial_lr * 0.1

    return schedule


def main():
    device = setup_cuda_optimization()

    parser = argparse.ArgumentParser(
        description="Enhanced Street Fighter II Training with Cross-Attention"
    )
    parser.add_argument("--total-timesteps", type=int, default=20_000_000)
    parser.add_argument(
        "--num-envs", type=int, default=1
    )  # Using DummyVecEnv, >1 may have issues
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--no-vision-transformer", action="store_true")
    # --- CHANGE IS HERE ---
    parser.add_argument("--state", type=str, default="ken_bison_12.state")
    # --- END OF CHANGE ---
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=4096)
    args = parser.parse_args()

    game = "StreetFighterIISpecialChampionEdition-Genesis"
    enable_vision_transformer = not args.no_vision_transformer
    save_dir = "enhanced_trained_models"
    os.makedirs(save_dir, exist_ok=True)

    state_file_name = args.state
    # Check for existence using an absolute path to give a clear error.
    state_file_path = os.path.abspath(state_file_name)
    if not os.path.exists(state_file_path):
        print(f"❌ State file not found at: {state_file_path}")
        print(
            "   Please make sure the .state file is in the current directory or provide a full path."
        )
        sys.exit(1)

    print(f"🎯 Training with Cross-Attention: {enable_vision_transformer}")
    print(f"💾 State file: {state_file_path}")

    def make_env():
        env = retro.make(
            game=game,
            # Pass the simple file name. retro will find it if it's in the CWD.
            state=state_file_name,
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            render_mode="human" if args.render else None,
        )
        env = StreetFighterVisionWrapper(
            env,
            reset_round=True,
            rendering=args.render,
            max_episode_steps=8000,
            enable_vision_transformer=enable_vision_transformer,
            log_transformer_predictions=True,
        )
        env = Monitor(env)
        env.reset(seed=0)
        return env

    env = DummyVecEnv([make_env for _ in range(args.num_envs)])

    policy_kwargs = dict(
        features_extractor_class=StreetFighterCrossAttentionCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[512, 256], vf=[512, 256]),
        activation_fn=nn.ReLU,
    )

    if args.resume and os.path.exists(args.resume):
        print(f"📂 Resuming training from {args.resume}")
        model = PPO.load(args.resume, env=env, device=device)
        model.learning_rate = create_learning_rate_schedule(args.learning_rate)
    else:
        print("🧠 Creating a new PPO model")
        model = PPO(
            "CnnPolicy",
            env,
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=1,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=10,
            gamma=0.99,
            learning_rate=create_learning_rate_schedule(args.learning_rate),
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            gae_lambda=0.95,
            tensorboard_log="enhanced_logs",
        )

    if enable_vision_transformer:
        print("💉 Injecting cross-attention feature extractor into wrapper...")
        try:
            feature_extractor = model.policy.features_extractor
            # Pass the VecEnv to the feature extractor
            feature_extractor.inject_cross_attention_components(
                None, env
            )  # Pass VecEnv
            # Inject components into each individual environment
            for i in range(args.num_envs):
                env.envs[i].env.inject_feature_extractor(feature_extractor)
            print("✅ Injection successful.")
        except Exception as e:
            print(f"⚠️ Feature extractor injection failed: {e}")

    checkpoint_callback = CheckpointCallback(
        save_freq=max(100000 // args.num_envs, 1),
        save_path=save_dir,
        name_prefix="cross_attention_ppo_sf2",
    )
    enhanced_callback = EnhancedCrossAttentionTrainingCallback(
        enable_vision_transformer
    )

    print("🏋️ Starting training...")
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback, enhanced_callback],
            reset_num_timesteps=not bool(args.resume),
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("⏹️ Training interrupted.")
    finally:
        final_model_path = os.path.join(save_dir, "ppo_sf2_final.zip")
        model.save(final_model_path)
        print(f"💾 Final model saved to: {final_model_path}")
        env.close()


if __name__ == "__main__":
    main()
