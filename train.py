#!/usr/bin/env python3
"""
Enhanced CUDA-Optimized Training Script for Street Fighter II with Oscillation-Based Positioning
Key improvements:
1. Oscillation tracking and analysis for footsies gameplay
2. Enhanced neutral game state detection
3. Spatial control and range management features
4. Whiff punishment and baiting mechanics integration
5. Better training hyperparameters based on oscillation analysis
6. Enhanced curriculum learning with spatial awareness
7. File-based logging with oscillation pattern analysis
8. FIXED: Proper model loading/resuming with oscillation-enhanced transformer
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

# Import the enhanced wrapper with oscillation analysis
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


class EnhancedOscillationTrainingCallback(BaseCallback):
    """Enhanced callback with oscillation training monitoring and analysis"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.last_log_time = time.time()
        self.log_interval = 120
        self.oscillation_log_file = os.path.join(
            "analysis_data",
            f"oscillation_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        )

    def _on_step(self):
        if time.time() - self.last_log_time > self.log_interval:
            self.last_log_time = time.time()
            if hasattr(self.training_env, "get_attr"):
                stats_list = self.training_env.get_attr("stats")
                if stats_list:
                    stats = stats_list[0]

                    # Get wrapper env correctly
                    wrapper_env = None
                    if (
                        hasattr(self.training_env, "envs")
                        and len(self.training_env.envs) > 0
                    ):
                        env = self.training_env.envs[0]
                        wrapper_env = env.env if hasattr(env, "env") else env

                    win_rate = 0.0
                    if (
                        wrapper_env
                        and hasattr(wrapper_env, "wins")
                        and hasattr(wrapper_env, "total_rounds")
                    ):
                        win_rate = wrapper_env.wins / max(1, wrapper_env.total_rounds)

                    # Enhanced logging with oscillation metrics
                    log_message = (
                        f"\n🎯 Training Progress (Step: {self.num_timesteps})\n"
                        f"   Win Rate: {win_rate:.1%}\n"
                        f"   Max Combo: {stats.get('max_combo', 0)}\n"
                        f"   Player Oscillation Frequency: {stats.get('player_oscillation_frequency', 0.0):.2f} per second\n"
                        f"   Space Control Score: {stats.get('space_control_score', 0.0):.2f}\n"
                        f"   Neutral Game Duration: {stats.get('neutral_game_duration', 0)} frames\n"
                        f"   Whiff Bait Attempts: {stats.get('whiff_bait_attempts', 0)}\n"
                        f"   Cross-Attention Weights:\n"
                        f"     Visual: {stats.get('visual_attention_weight', 0.0):.3f}\n"
                        f"     Strategy: {stats.get('strategy_attention_weight', 0.0):.3f}\n"
                        f"     Oscillation: {stats.get('oscillation_attention_weight', 0.0):.3f}\n"
                        f"     Button: {stats.get('button_attention_weight', 0.0):.3f}"
                    )
                    print(log_message)

                    # Log to file for analysis
                    try:
                        with open(self.oscillation_log_file, "a") as f:
                            f.write(
                                f"{datetime.now().isoformat()},{self.num_timesteps},{win_rate:.4f},"
                                f"{stats.get('player_oscillation_frequency', 0.0):.4f},"
                                f"{stats.get('space_control_score', 0.0):.4f},"
                                f"{stats.get('neutral_game_duration', 0)},"
                                f"{stats.get('whiff_bait_attempts', 0)},"
                                f"{stats.get('oscillation_attention_weight', 0.0):.4f}\n"
                            )
                    except Exception as e:
                        print(f"⚠️ Failed to log oscillation data: {e}")
        return True


def create_oscillation_aware_learning_rate_schedule(initial_lr=3e-4):
    """Create a learning rate schedule optimized for oscillation analysis."""

    def schedule(progress):
        # Start with higher learning rate for oscillation pattern recognition
        if progress < 0.1:
            return initial_lr * 1.2  # Boost for early oscillation learning
        elif progress < 0.3:
            return initial_lr
        elif progress < 0.7:
            return initial_lr * 0.7  # Stabilize oscillation patterns
        else:
            return initial_lr * 0.3  # Fine-tune spatial control

    return schedule


def create_model_with_oscillation_structure(env, device, args):
    """Create a new model with the proper structure for oscillation-enhanced cross-attention"""
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
        learning_rate=create_oscillation_aware_learning_rate_schedule(
            args.learning_rate
        ),
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        gae_lambda=0.95,
        tensorboard_log="enhanced_oscillation_logs",
    )

    return model, policy_kwargs


def inject_oscillation_cross_attention_transformer(model, env, args):
    """Inject oscillation-enhanced cross-attention transformer into the model's feature extractor"""
    print(
        "💉 Injecting oscillation-enhanced cross-attention feature extractor into wrapper..."
    )
    try:
        feature_extractor = model.policy.features_extractor
        for i in range(args.num_envs):
            wrapper = env.envs[i].env if hasattr(env.envs[i], "env") else env.envs[i]
            wrapper.inject_feature_extractor(feature_extractor)
        print("✅ Oscillation-enhanced cross-attention injection successful.")
        return True
    except Exception as e:
        print(f"⚠️ Feature extractor injection failed: {e}")
        return False


def load_model_with_oscillation_cross_attention(model_path, env, device, args):
    """
    Load a model with oscillation-enhanced cross-attention transformer properly.
    This creates the base model structure first, then injects the oscillation cross-attention,
    and finally loads the saved parameters.
    """
    print(f"📂 Loading model with oscillation cross-attention from {model_path}")

    # Step 1: Create a new model with the proper base structure
    model, policy_kwargs = create_model_with_oscillation_structure(env, device, args)

    # Step 2: Inject oscillation cross-attention transformer to create the full structure
    inject_success = inject_oscillation_cross_attention_transformer(model, env, args)

    if not inject_success:
        print("⚠️ Oscillation cross-attention injection failed, loading without it...")
        # Fallback: load the model without oscillation cross-attention
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
        model.learning_rate = create_oscillation_aware_learning_rate_schedule(
            args.learning_rate
        )

        print("✅ Model parameters loaded successfully!")
        return model

    except Exception as e:
        print(f"⚠️ Failed to load saved parameters: {e}")
        print("🔄 Falling back to fresh model initialization...")

        # Return the fresh model if loading fails
        return model


def validate_oscillation_features(env):
    """Validate that oscillation features are working correctly"""
    print("🔍 Validating oscillation features...")
    try:
        obs, info = env.reset()
        for i in range(100):  # Test for 100 steps
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)

            if i % 50 == 0:  # Check every 50 steps
                stats = info.get("stats", {})
                oscillation_freq = stats.get("player_oscillation_frequency", 0.0)
                space_control = stats.get("space_control_score", 0.0)
                print(
                    f"   Step {i}: Oscillation freq: {oscillation_freq:.3f}, Space control: {space_control:.3f}"
                )

            if done or truncated:
                obs, info = env.reset()

        print("✅ Oscillation feature validation successful!")
        return True

    except Exception as e:
        print(f"❌ Oscillation feature validation failed: {e}")
        return False


def main():
    device = setup_cuda_optimization()

    parser = argparse.ArgumentParser(
        description="Enhanced Street Fighter II Training with Oscillation-Based Positioning"
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
    parser.add_argument(
        "--validate-oscillation",
        action="store_true",
        help="Run oscillation feature validation",
    )
    args = parser.parse_args()

    game = "StreetFighterIISpecialChampionEdition-Genesis"
    enable_vision_transformer = not args.no_vision_transformer
    save_dir = "enhanced_oscillation_trained_models"
    os.makedirs(save_dir, exist_ok=True)

    state_file_name = args.state
    state_file_path = os.path.abspath(state_file_name)
    if not os.path.exists(state_file_path):
        print(f"❌ State file not found at: {state_file_path}")
        sys.exit(1)

    print(
        f"🎯 Training with Oscillation-Enhanced Cross-Attention: {enable_vision_transformer}"
    )
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
        return env

    env = DummyVecEnv([make_env for _ in range(args.num_envs)])

    # Optional oscillation feature validation
    if args.validate_oscillation:
        print("🧪 Running oscillation feature validation...")
        test_env = make_env()
        if not validate_oscillation_features(test_env):
            print("❌ Oscillation validation failed. Exiting.")
            sys.exit(1)
        test_env.close()
        print("✅ Oscillation validation passed!")

    # Model creation and loading logic
    if args.resume and os.path.exists(args.resume):
        print(f"📂 Resuming training from {args.resume}")
        model = load_model_with_oscillation_cross_attention(
            args.resume, env, device, args
        )
    else:
        print("🧠 Creating a new PPO model with oscillation analysis")
        model, _ = create_model_with_oscillation_structure(env, device, args)

        # Inject oscillation cross-attention transformer for new models
        if enable_vision_transformer:
            inject_oscillation_cross_attention_transformer(model, env, args)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(25000 // args.num_envs, 1),
        save_path=save_dir,
        name_prefix="oscillation_cross_attention_ppo_sf2",
    )
    enhanced_callback = EnhancedOscillationTrainingCallback()

    print("🏋️ Starting training with oscillation analysis...")
    print("📊 Monitoring:")
    print("   - Player oscillation frequency (movements per second)")
    print("   - Space control scores (-1.0 to 1.0)")
    print("   - Neutral game duration tracking")
    print("   - Whiff bait attempt detection")
    print("   - Cross-attention weight distribution")

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
        final_model_path = os.path.join(save_dir, "ppo_sf2_oscillation_final.zip")
        model.save(final_model_path)
        print(f"💾 Final oscillation-enhanced model saved to: {final_model_path}")

        # Save oscillation analysis summary
        try:
            stats_list = env.get_attr("stats")
            if stats_list:
                stats = stats_list[0]
                summary_file = os.path.join(
                    "analysis_data",
                    f"oscillation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                )
                with open(summary_file, "w") as f:
                    f.write("=== Oscillation Analysis Summary ===\n")
                    f.write(f"Total Training Steps: {args.total_timesteps}\n")
                    f.write(
                        f"Final Player Oscillation Frequency: {stats.get('player_oscillation_frequency', 0.0):.4f} per second\n"
                    )
                    f.write(
                        f"Final Space Control Score: {stats.get('space_control_score', 0.0):.4f}\n"
                    )
                    f.write(
                        f"Final Neutral Game Duration: {stats.get('neutral_game_duration', 0)} frames\n"
                    )
                    f.write(
                        f"Total Whiff Bait Attempts: {stats.get('whiff_bait_attempts', 0)}\n"
                    )
                    f.write(f"Cross-Attention Weight Distribution:\n")
                    f.write(
                        f"  Visual: {stats.get('visual_attention_weight', 0.0):.4f}\n"
                    )
                    f.write(
                        f"  Strategy: {stats.get('strategy_attention_weight', 0.0):.4f}\n"
                    )
                    f.write(
                        f"  Oscillation: {stats.get('oscillation_attention_weight', 0.0):.4f}\n"
                    )
                    f.write(
                        f"  Button: {stats.get('button_attention_weight', 0.0):.4f}\n"
                    )
                    f.write(f"\n=== Oscillation Insights ===\n")
                    f.write(
                        f"The oscillation attention weight of {stats.get('oscillation_attention_weight', 0.0):.4f} indicates how much the AI\n"
                    )
                    f.write(
                        f"is focusing on movement patterns and spatial control during decision making.\n"
                    )
                    f.write(
                        f"Higher oscillation attention suggests the AI has learned to prioritize footsies\n"
                    )
                    f.write(
                        f"and neutral game positioning, which is crucial for high-level fighting game play.\n"
                    )
                print(f"📊 Oscillation analysis summary saved to: {summary_file}")
        except Exception as e:
            print(f"⚠️ Failed to save oscillation summary: {e}")

        env.close()


if __name__ == "__main__":
    main()
