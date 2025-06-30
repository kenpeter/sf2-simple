#!/usr/bin/env python3
"""
train.py - Enhanced Training Script for Street Fighter II with Relative Position Vision Pipeline
Uses 8-frame RGB stacking + OpenCV + CNN + Enhanced Vision Transformer with relative positioning
"""

import os
import sys
import argparse
import time
import torch
import torch.nn as nn

import retro
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

# Import the enhanced wrapper
from wrapper import StreetFighterVisionWrapper, StreetFighterEnhancedCNN


class EnhancedVisionPipelineCallback(BaseCallback):
    """Enhanced callback to monitor vision pipeline training progress with relative positioning"""

    def __init__(self, enable_vision_transformer=True, verbose=0):
        super(EnhancedVisionPipelineCallback, self).__init__(verbose)
        self.enable_vision_transformer = enable_vision_transformer
        self.last_log_time = time.time()
        self.log_interval = 120  # Log every 2 minutes

    def _on_step(self) -> bool:
        current_time = time.time()
        if current_time - self.last_log_time > self.log_interval:
            self.last_log_time = current_time
            self._log_enhanced_training_stats()
        return True

    def _log_enhanced_training_stats(self):
        """Log comprehensive enhanced training statistics"""
        try:
            print(
                f"\n--- 📊 Enhanced Street Fighter Vision Pipeline @ Step {self.num_timesteps:,} ---"
            )

            # Get environment statistics from first environment
            if hasattr(self.training_env, "get_attr"):
                all_stats = self.training_env.get_attr("stats")
                if all_stats and len(all_stats) > 0:
                    stats = all_stats[0]

                    # Motion detection statistics
                    motion_count = stats.get("motion_detected_count", 0)
                    print(f"   🔍 Motion Detection: {motion_count:,} detections")

                    # Enhanced vision transformer statistics
                    if self.enable_vision_transformer:
                        vt_ready = stats.get("vision_transformer_ready", False)
                        predictions = stats.get("predictions_made", 0)

                        if vt_ready:
                            print(
                                f"   🧠 Enhanced Vision Transformer: Ready ({predictions:,} predictions)"
                            )

                            # Enhanced tactical predictions
                            avg_attack = stats.get("avg_attack_timing", 0.0)
                            avg_defend = stats.get("avg_defend_timing", 0.0)
                            avg_aggression = stats.get("avg_aggression_level", 0.0)
                            avg_positioning = stats.get("avg_positioning_score", 0.0)

                            print(f"   ⚔️  Avg Attack Timing: {avg_attack:.3f}")
                            print(f"   🛡️  Avg Defend Timing: {avg_defend:.3f}")
                            print(f"   🔥 Avg Aggression Level: {avg_aggression:.3f}")
                            print(f"   📍 Avg Positioning Score: {avg_positioning:.3f}")

                        else:
                            print("   🧠 Enhanced Vision Transformer: Initializing...")
                    else:
                        print("   🧠 Enhanced Vision Transformer: Disabled")

                    # Detection rates
                    total_frames = self.num_timesteps
                    if total_frames > 0:
                        motion_rate = motion_count / total_frames * 1000
                        prediction_rate = (
                            predictions / total_frames * 1000 if predictions > 0 else 0
                        )
                        print(f"   📈 Motion Rate: {motion_rate:.2f} per 1000 frames")
                        print(
                            f"   📈 Prediction Rate: {prediction_rate:.2f} per 1000 frames"
                        )

            # System stats
            self._log_system_stats()
            print("   🎯 Features: Health + Score + Relative Position (19D momentum)")
            print("   📐 Using agent-enemy position differences for tactical analysis")
            print("--------------------------------------------------")

        except Exception as e:
            print(f"   ⚠️ Logging error: {e}")

    def _log_system_stats(self):
        """Log system and training parameters"""
        try:
            # Memory usage
            if torch.cuda.is_available():
                vram_alloc = torch.cuda.memory_allocated() / (1024**3)
                vram_cached = torch.cuda.memory_reserved() / (1024**3)
                print(
                    f"   💾 VRAM: {vram_alloc:.2f}GB allocated / {vram_cached:.2f}GB cached"
                )

            # Learning rate
            if hasattr(self.model, "learning_rate"):
                lr = self.model.learning_rate
                if callable(lr):
                    progress = getattr(self.model, "_current_progress_remaining", 1.0)
                    lr = lr(progress)
                print(f"   📈 Learning Rate: {lr:.2e}")

        except Exception as e:
            print(f"   ⚠️ System stats error: {e}")


def make_enhanced_env(
    game,
    state,
    seed=0,
    rendering=False,
    enable_vision_transformer=True,
    defend_actions=None,
):
    """Create environment with enhanced vision wrapper using relative positioning"""

    def _init():
        env = retro.make(
            game=game,
            state=state,
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            render_mode="human" if rendering else None,
        )

        env = StreetFighterVisionWrapper(
            env,
            reset_round=True,
            rendering=rendering,
            max_episode_steps=5000,
            frame_stack=8,  # 8 RGB frames
            enable_vision_transformer=enable_vision_transformer,
            defend_action_indices=defend_actions,
        )

        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


def linear_schedule(initial_value, final_value=0.0):
    """Linear learning rate scheduler"""

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler


def main():
    parser = argparse.ArgumentParser(
        description="Train Enhanced Street Fighter II Vision Pipeline Agent with Relative Positioning"
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=10000000, help="Total timesteps to train"
    )
    parser.add_argument(
        "--num-envs", type=int, default=8, help="Number of parallel environments"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2.5e-4, help="Learning rate"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from saved model path"
    )
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument(
        "--no-vision-transformer",
        action="store_true",
        help="Disable Enhanced Vision Transformer (OpenCV + CNN only)",
    )
    parser.add_argument(
        "--use-original-state",
        action="store_true",
        help="Use original state file instead of absolute path",
    )
    parser.add_argument(
        "--defend-actions",
        type=str,
        default="4,5,6",
        help="Comma-separated defend action indices (default: 4,5,6)",
    )
    parser.add_argument(
        "--list-states", action="store_true", help="List available states and exit"
    )

    args = parser.parse_args()

    # List available states if requested
    if args.list_states:
        try:
            game = "StreetFighterIISpecialChampionEdition-Genesis"
            states = retro.data.list_states(game)
            print(f"📋 Available states for {game}:")
            for state in states:
                print(f"   - {state}")
        except Exception as e:
            print(f"❌ Could not list states: {e}")
        return

    game = "StreetFighterIISpecialChampionEdition-Genesis"
    enable_vision_transformer = not args.no_vision_transformer

    # Parse defend action indices
    defend_actions = [int(x.strip()) for x in args.defend_actions.split(",")]
    print(f"   🛡️  Defend actions: {defend_actions}")

    # Handle state file path
    if args.use_original_state:
        state_file = "ken_bison_12.state"
        print("🎮 Using original state file: ken_bison_12.state")
    else:
        state_file = os.path.abspath("ken_bison_12.state")
        if not os.path.exists("ken_bison_12.state"):
            print(f"❌ State file not found: ken_bison_12.state")
            print("🔍 Current directory files:")
            for f in os.listdir("."):
                if f.endswith(".state"):
                    print(f"   - {f}")
            print("💡 Try using --use-original-state flag or --list-states")
            return
        print(f"🎮 Using absolute state path: {state_file}")

    save_dir = "trained_models_enhanced_vision"
    os.makedirs(save_dir, exist_ok=True)

    # Print enhanced configuration
    mode_name = (
        "Enhanced Vision Pipeline (Relative Positioning)"
        if enable_vision_transformer
        else "OpenCV + CNN"
    )
    print(f"🚀 Street Fighter II {mode_name} Training")
    print(f"   Total timesteps: {args.total_timesteps:,}")
    print(f"   Environments: {args.num_envs}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Frame stack: 8 RGB frames (24 channels)")
    print(f"   Resolution: 180×128")
    print(f"   State file: {state_file}")
    print(f"   Enhanced Features:")
    print(f"     - Health momentum tracking")
    print(f"     - Score momentum tracking")
    print(f"     - Relative positioning (agent_x - enemy_x, agent_y - enemy_y)")
    print(f"     - Distance and approach/retreat analysis")
    print(
        f"     - Enhanced tactical predictions (attack/defend/aggression/positioning)"
    )
    print(
        f"   Vision Transformer: {'Enabled' if enable_vision_transformer else 'Disabled'}"
    )
    if args.resume:
        print(f"   Resuming from: {args.resume}")

    # Create single environment
    print(f"🔧 Creating enhanced environment...")
    try:
        env = DummyVecEnv(
            [
                make_enhanced_env(
                    game,
                    state=state_file,
                    seed=0,
                    rendering=args.render,
                    enable_vision_transformer=enable_vision_transformer,
                    defend_actions=defend_actions,
                )
            ]
        )
        print("✅ Enhanced environment created successfully")
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        print("💡 Try using --use-original-state flag or check state file path")
        return

    # Enhanced policy configuration for vision pipeline
    policy_kwargs = dict(
        features_extractor_class=StreetFighterEnhancedCNN,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(
            pi=[512, 256, 128],  # Larger policy network for enhanced features
            vf=[512, 256, 128],  # Larger value network
        ),
        activation_fn=nn.ReLU,
    )

    # Create or load model
    if args.resume and os.path.exists(args.resume):
        print(f"📂 Loading model from: {args.resume}")
        model = PPO.load(args.resume, env=env, device="cuda")
        print("✅ Model loaded, resuming training")
    else:
        print(f"🧠 Creating new PPO model with {mode_name}")

        # Learning rate schedule
        lr_schedule = linear_schedule(args.learning_rate, args.learning_rate * 0.1)

        model = PPO(
            "CnnPolicy",
            env,
            policy_kwargs=policy_kwargs,
            device="cuda",
            verbose=1,
            n_steps=2048,  # Steps per environment per update
            batch_size=64,  # Smaller batch for complex model
            n_epochs=4,
            gamma=0.99,
            learning_rate=lr_schedule,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            gae_lambda=0.95,
            tensorboard_log="logs_enhanced_vision",
        )

    # Inject enhanced vision components into environment
    print("💉 Injecting CNN feature extractor into enhanced vision pipeline...")
    try:
        # Get the CNN feature extractor from the model
        feature_extractor = model.policy.features_extractor

        # For DummyVecEnv with single environment, inject directly
        wrapper_env = env.envs[
            0
        ].env  # DummyVecEnv -> Monitor -> StreetFighterVisionWrapper
        if hasattr(wrapper_env, "inject_feature_extractor"):
            wrapper_env.inject_feature_extractor(feature_extractor)
            print("✅ Enhanced feature extractor injected successfully")
        else:
            print(
                f"⚠️ Could not find inject_feature_extractor method on {type(wrapper_env)}"
            )

    except Exception as e:
        print(f"⚠️ Feature extractor injection failed: {e}")
        print("Training will continue with base CNN only")

    # Setup enhanced callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,  # Save every 100k steps (single env)
        save_path=save_dir,
        name_prefix="ppo_sf2_enhanced_vision",
    )

    enhanced_vision_callback = EnhancedVisionPipelineCallback(
        enable_vision_transformer=enable_vision_transformer
    )

    # Training
    start_time = time.time()
    print(f"🏋️ Starting {mode_name} training for {args.total_timesteps:,} timesteps")
    print(f"🎯 Enhanced features: Health + Score + Relative Position momentum tracking")

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback, enhanced_vision_callback],
            reset_num_timesteps=not bool(args.resume),
        )

        training_time = time.time() - start_time
        print(f"🎉 Enhanced training completed in {training_time/3600:.1f} hours!")

    except KeyboardInterrupt:
        print(f"⏹️ Training interrupted")
        training_time = time.time() - start_time
        print(f"Training time: {training_time/3600:.1f} hours")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        return

    finally:
        env.close()

    # Always save final model
    final_model_path = os.path.join(save_dir, "ppo_sf2_enhanced_vision_final.zip")
    model.save(final_model_path)
    print(f"💾 Final enhanced model saved to: {final_model_path}")

    print("✅ Enhanced training complete!")
    print(f"🎮 Test with: python eval.py --model-path {final_model_path}")
    print(f"🔄 Resume with: python train.py --resume {final_model_path}")
    print(f"📊 Enhanced features used: Health + Score + Relative Position (18D)")
    print(f"🎯 Tactical predictions: Attack/Defend/Aggression/Positioning + Movement")


if __name__ == "__main__":
    main()
