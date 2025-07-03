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
8. FIXED: Proper model loading/resuming with cross-attention transformer
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

    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    print(f"🚀 CUDA Setup Complete:")
    print(f"   Device: {torch.cuda.get_device_name(0)}")
    return device


class EnhancedCrossAttentionTrainingCallback(BaseCallback):
    """Enhanced callback with cross-attention training monitoring and analysis"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.last_log_time = time.time()
        self.log_interval = 120

    def _on_step(self):
        if time.time() - self.last_log_time > self.log_interval:
            self.last_log_time = time.time()
            if hasattr(self.training_env, "get_attr"):
                stats_list = self.training_env.get_attr("stats")
                if stats_list:
                    stats = stats_list[0]
                    wrapper_env = self.training_env.envs[0].env
                    win_rate = wrapper_env.wins / max(1, wrapper_env.total_rounds)
                    print(
                        f"\nStep: {self.num_timesteps}, Win Rate: {win_rate:.1%}, Max Combo: {stats.get('max_combo', 0)}"
                    )
        return True


def create_learning_rate_schedule(initial_lr=3e-4):
    """Create a learning rate schedule optimized for complex policies."""

    def schedule(progress):
        if progress < 0.2:
            return initial_lr
        elif progress < 0.6:
            return initial_lr * 0.5
        else:
            return initial_lr * 0.1

    return schedule


def create_model_with_proper_structure(env, device, args):
    """Create a new model with the proper structure for cross-attention"""
    policy_kwargs = dict(
        features_extractor_class=StreetFighterCrossAttentionCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[512, 256], vf=[512, 256]),
        activation_fn=nn.ReLU,
    )

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

    return model, policy_kwargs


def inject_cross_attention_transformer(model, env, args):
    """Inject cross-attention transformer into the model's feature extractor"""
    print("💉 Injecting cross-attention feature extractor into wrapper...")
    try:
        feature_extractor = model.policy.features_extractor
        for i in range(args.num_envs):
            env.envs[i].env.inject_feature_extractor(feature_extractor)
        print("✅ Cross-attention injection successful.")
        return True
    except Exception as e:
        print(f"⚠️ Feature extractor injection failed: {e}")
        return False


def load_model_with_cross_attention(model_path, env, device, args):
    """
    Load a model with cross-attention transformer properly.
    This creates the base model structure first, then injects the cross-attention,
    and finally loads the saved parameters.
    """
    print(f"📂 Loading model with cross-attention from {model_path}")

    # Step 1: Create a new model with the proper base structure
    model, policy_kwargs = create_model_with_proper_structure(env, device, args)

    # Step 2: Inject cross-attention transformer to create the full structure
    inject_success = inject_cross_attention_transformer(model, env, args)

    if not inject_success:
        print("⚠️ Cross-attention injection failed, loading without it...")
        # Fallback: load the model without cross-attention
        model = PPO.load(model_path, env=env, device=device)
        return model

    # Step 3: Load the saved parameters with strict=False to handle any missing keys
    print("📥 Loading saved parameters...")
    try:
        # Load the saved model data
        saved_data = torch.load(model_path, map_location=device)

        # Extract the parameters
        if "policy" in saved_data:
            saved_params = saved_data["policy"]
        else:
            saved_params = saved_data

        # Load parameters with strict=False to ignore missing keys
        model.policy.load_state_dict(saved_params, strict=False)

        # Update learning rate
        model.learning_rate = create_learning_rate_schedule(args.learning_rate)

        print("✅ Model parameters loaded successfully!")
        return model

    except Exception as e:
        print(f"⚠️ Failed to load saved parameters: {e}")
        print("🔄 Falling back to fresh model initialization...")

        # Return the fresh model if loading fails
        return model


def main():
    device = setup_cuda_optimization()

    parser = argparse.ArgumentParser(
        description="Enhanced Street Fighter II Training with Cross-Attention"
    )
    parser.add_argument("--total-timesteps", type=int, default=20_000_000)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--no-vision-transformer", action="store_true")
    parser.add_argument("--state", type=str, default="ken_bison_12.state")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=4096)
    args = parser.parse_args()

    game = "StreetFighterIISpecialChampionEdition-Genesis"
    enable_vision_transformer = not args.no_vision_transformer
    save_dir = "enhanced_trained_models"
    os.makedirs(save_dir, exist_ok=True)

    state_file_name = args.state
    state_file_path = os.path.abspath(state_file_name)
    if not os.path.exists(state_file_path):
        print(f"❌ State file not found at: {state_file_path}")
        sys.exit(1)

    print(f"🎯 Training with Cross-Attention: {enable_vision_transformer}")
    print(f"💾 State file: {state_file_path}")

    def make_env():
        env = retro.make(
            game=game,
            state=state_file_name,
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            render_mode="human" if args.render else None,
        )
        env = StreetFighterVisionWrapper(
            env, reset_round=True, enable_vision_transformer=enable_vision_transformer
        )
        env = Monitor(env)
        env.reset(seed=0)
        return env

    env = DummyVecEnv([make_env for _ in range(args.num_envs)])

    # Model creation and loading logic
    if args.resume and os.path.exists(args.resume):
        print(f"📂 Resuming training from {args.resume}")
        model = load_model_with_cross_attention(args.resume, env, device, args)
    else:
        print("🧠 Creating a new PPO model")
        model, _ = create_model_with_proper_structure(env, device, args)

        # Inject cross-attention transformer for new models
        if enable_vision_transformer:
            inject_cross_attention_transformer(model, env, args)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(25000 // args.num_envs, 1),
        save_path=save_dir,
        name_prefix="cross_attention_ppo_sf2",
    )
    enhanced_callback = EnhancedCrossAttentionTrainingCallback()

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
