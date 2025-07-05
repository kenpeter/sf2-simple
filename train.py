#!/usr/bin/env python3
"""
train.py - FIXED Training Script with Proper Model Loading
KEY FIXES APPLIED:
1. Fixed model loading to handle cross-attention parameter mismatches
2. Added proper parameter filtering for compatible loading
3. Enhanced error handling for model architecture differences
4. Improved logging for debugging parameter loading issues
5. Added fallback mechanisms for partial model loading
"""

import os
import argparse
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
import retro
import logging
from datetime import datetime
import json
import zipfile
import tempfile

# Import the fixed wrapper
from wrapper import (
    StreetFighterVisionWrapper,
    StreetFighterCrossAttentionCNN,
    monitor_gradients,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FixedTrainingCallback(BaseCallback):
    """Enhanced callback with gradient monitoring and proper attention weight logging"""

    def __init__(
        self,
        save_freq=100000,
        save_path="./enhanced_oscillation_trained_models/",
        verbose=1,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_mean_reward = -np.inf

        # Enhanced logging for cross-attention
        self.attention_weights_history = []
        self.oscillation_frequency_history = []
        self.gradient_norms_history = []

        # Create save directory
        os.makedirs(save_path, exist_ok=True)

        # Create detailed log file
        self.log_file = os.path.join(
            save_path, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

    def _on_step(self) -> bool:
        """Enhanced step logging with gradient monitoring"""

        # Monitor gradients every 5000 steps
        if self.num_timesteps % 5000 == 0:
            monitor_gradients(self.model, self.num_timesteps)

        # Enhanced logging every 1000 steps
        if self.num_timesteps % 1000 == 0:
            self._log_enhanced_stats()

        # Save model periodically
        if self.num_timesteps % self.save_freq == 0:
            self._save_model()

        return True

    def _log_enhanced_stats(self):
        """Log enhanced statistics including attention weights and oscillation frequency"""
        try:
            # Get environment statistics
            if hasattr(self.training_env, "envs"):
                env_stats = {}
                attention_weights = {}
                oscillation_stats = {}

                for i, env in enumerate(self.training_env.envs):
                    wrapper = env.env if hasattr(env, "env") else env

                    if hasattr(wrapper, "stats"):
                        stats = wrapper.stats

                        # Collect attention weights
                        attention_weights[f"env_{i}"] = {
                            "visual": stats.get("visual_attention_weight", 0.0),
                            "strategy": stats.get("strategy_attention_weight", 0.0),
                            "oscillation": stats.get(
                                "oscillation_attention_weight", 0.0
                            ),
                            "button": stats.get("button_attention_weight", 0.0),
                        }

                        # Collect oscillation statistics
                        oscillation_stats[f"env_{i}"] = {
                            "frequency": stats.get("player_oscillation_frequency", 0.0),
                            "space_control": stats.get("space_control_score", 0.0),
                            "neutral_game_duration": stats.get(
                                "neutral_game_duration", 0.0
                            ),
                            "whiff_bait_attempts": stats.get("whiff_bait_attempts", 0),
                        }

                        # Regular stats
                        for key, value in stats.items():
                            if key not in env_stats:
                                env_stats[key] = []
                            env_stats[key].append(value)

                # Log attention weight analysis
                if attention_weights:
                    avg_attention = {}
                    for weight_type in ["visual", "strategy", "oscillation", "button"]:
                        weights = [
                            env_weights[weight_type]
                            for env_weights in attention_weights.values()
                        ]
                        avg_attention[weight_type] = np.mean(weights)

                    print(f"\n🎯 Step {self.num_timesteps} - Attention Analysis:")
                    print(f"   Visual: {avg_attention['visual']:.4f}")
                    print(f"   Strategy: {avg_attention['strategy']:.4f}")
                    print(f"   Oscillation: {avg_attention['oscillation']:.4f}")
                    print(f"   Button: {avg_attention['button']:.4f}")

                    # Check if attention weights are learning (not identical)
                    weight_values = list(avg_attention.values())
                    weight_variance = np.var(weight_values)

                    if weight_variance < 1e-6:
                        print(
                            f"   ⚠️  WARNING: Attention weights are identical! Variance: {weight_variance:.8f}"
                        )
                        print(f"   🔧 Cross-attention may not be learning properly")
                    else:
                        print(
                            f"   ✅ Attention weights are diverse! Variance: {weight_variance:.6f}"
                        )

                    self.attention_weights_history.append(
                        {
                            "step": self.num_timesteps,
                            "weights": avg_attention,
                            "variance": weight_variance,
                        }
                    )

                # Log oscillation frequency analysis
                if oscillation_stats:
                    avg_oscillation = {}
                    for key in [
                        "frequency",
                        "space_control",
                        "neutral_game_duration",
                        "whiff_bait_attempts",
                    ]:
                        values = [
                            env_stats[key] for env_stats in oscillation_stats.values()
                        ]
                        avg_oscillation[key] = np.mean(values)

                    print(f"\n🌊 Oscillation Analysis:")
                    print(f"   Frequency: {avg_oscillation['frequency']:.3f} Hz")
                    print(f"   Space Control: {avg_oscillation['space_control']:.3f}")
                    print(
                        f"   Neutral Game: {avg_oscillation['neutral_game_duration']:.1f} frames"
                    )
                    print(
                        f"   Whiff Baits: {avg_oscillation['whiff_bait_attempts']:.0f}"
                    )

                    # Check if frequency is in the expected range
                    freq = avg_oscillation["frequency"]
                    if freq < 0.1:
                        print(f"   ⚠️  WARNING: Very low oscillation frequency!")
                    elif 1.0 <= freq <= 3.0:
                        print(f"   ✅ Oscillation frequency in optimal range (1-3 Hz)")
                    elif freq > 5.0:
                        print(
                            f"   ⚠️  WARNING: Very high oscillation frequency (possible noise)"
                        )

                    self.oscillation_frequency_history.append(
                        {
                            "step": self.num_timesteps,
                            "frequency": freq,
                            "space_control": avg_oscillation["space_control"],
                        }
                    )

                # Log average statistics
                if env_stats:
                    avg_stats = {k: np.mean(v) for k, v in env_stats.items()}

                    print(f"\n📊 Step {self.num_timesteps} - Training Stats:")
                    print(
                        f"   Predictions Made: {avg_stats.get('predictions_made', 0):.0f}"
                    )
                    print(
                        f"   Cross-Attention Ready: {avg_stats.get('cross_attention_ready', False)}"
                    )

                # Save detailed log
                self._save_detailed_log()

        except Exception as e:
            logger.error(f"Error in enhanced stats logging: {e}")

    def _save_detailed_log(self):
        """Save detailed training log with attention and oscillation data"""
        try:
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "step": self.num_timesteps,
                "attention_weights_history": self.attention_weights_history[
                    -10:
                ],  # Last 10 entries
                "oscillation_frequency_history": self.oscillation_frequency_history[
                    -10:
                ],
            }

            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_data) + "\n")

        except Exception as e:
            logger.error(f"Error saving detailed log: {e}")

    def _save_model(self):
        """Save model with enhanced naming"""
        try:
            # Enhanced model naming with frequency info
            avg_freq = 0.0
            if self.oscillation_frequency_history:
                avg_freq = self.oscillation_frequency_history[-1]["frequency"]

            model_name = f"oscillation_cross_attention_ppo_sf2_{self.num_timesteps}_steps_freq_{avg_freq:.2f}Hz"
            model_path = os.path.join(self.save_path, f"{model_name}.zip")

            self.model.save(model_path)
            print(f"💾 Model saved: {model_path}")

            # Save attention analysis
            analysis_path = os.path.join(self.save_path, f"{model_name}_analysis.json")
            analysis_data = {
                "step": self.num_timesteps,
                "attention_weights_history": self.attention_weights_history,
                "oscillation_frequency_history": self.oscillation_frequency_history,
                "learning_rate": self.model.learning_rate,
            }

            with open(analysis_path, "w") as f:
                json.dump(analysis_data, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving model: {e}")


def make_env(
    game="StreetFighterIISpecialChampionEdition-Genesis",
    state="ken_bison_12.state",
    render_mode=None,
):
    """Create Street Fighter environment with fixed wrapper"""

    def _init():
        env = retro.make(game=game, state=state, render_mode=render_mode)
        env = Monitor(env)
        env = StreetFighterVisionWrapper(
            env,
            reset_round=True,
            rendering=render_mode is not None,
            max_episode_steps=5000,
            frame_stack=8,
            enable_vision_transformer=True,
            log_transformer_predictions=True,
        )
        return env

    return _init


def filter_state_dict_for_loading(saved_state_dict, current_model_state_dict):
    """
    FIXED: Filter saved state dict to only include parameters that are compatible
    with the current model architecture
    """
    compatible_params = {}
    skipped_params = []

    print(f"🔍 Filtering state dict for compatible parameters...")
    print(f"   Saved model has {len(saved_state_dict)} parameters")
    print(f"   Current model has {len(current_model_state_dict)} parameters")

    for key, value in saved_state_dict.items():
        if key in current_model_state_dict:
            if current_model_state_dict[key].shape == value.shape:
                compatible_params[key] = value
            else:
                skipped_params.append(
                    f"{key} (shape mismatch: saved={value.shape}, current={current_model_state_dict[key].shape})"
                )
        else:
            skipped_params.append(f"{key} (not found in current model)")

    print(f"   ✅ Compatible parameters: {len(compatible_params)}")
    print(f"   ⚠️  Skipped parameters: {len(skipped_params)}")

    if len(skipped_params) > 0:
        print(f"   📋 First 10 skipped parameters:")
        for param in skipped_params[:10]:
            print(f"      - {param}")

    return compatible_params, skipped_params


def load_model_with_cross_attention(model_path, env, learning_rate, policy_kwargs):
    """
    FIXED: Load model with proper cross-attention parameter handling and filtering
    """
    try:
        print(f"📂 Loading model from {model_path}")
        print(f"🔍 Analyzing saved model structure...")

        # Step 1: Create a new model with the target architecture
        print("🔧 Creating new model architecture...")
        new_model = PPO(
            "CnnPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=None,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device="auto",
        )

        # Step 2: Inject cross-attention components into the new model
        print("🔗 Injecting cross-attention components...")
        for i, env_instance in enumerate(env.envs):
            wrapper = env_instance.env if hasattr(env_instance, "env") else env_instance
            if hasattr(wrapper, "inject_feature_extractor"):
                wrapper.inject_feature_extractor(new_model.policy.features_extractor)
                print(f"   ✅ Environment {i+1} injected successfully")

        # Step 3: Load the saved model and extract its state dict
        print("📂 Loading saved model parameters...")

        # Load the entire saved model first to get its state dict
        try:
            saved_model = PPO.load(model_path, device="auto")
            saved_state_dict = saved_model.policy.state_dict()
            print(f"   ✅ Successfully loaded saved model state dict")
        except Exception as e:
            print(f"   ❌ Failed to load saved model: {e}")
            print(f"   🔧 Attempting alternative loading method...")

            # Alternative: Load just the zip file and extract the policy state dict
            import zipfile
            import tempfile

            with zipfile.ZipFile(model_path, "r") as zip_file:
                with tempfile.TemporaryDirectory() as tmp_dir:
                    zip_file.extractall(tmp_dir)

                    # Try to load the policy state dict directly
                    policy_path = os.path.join(tmp_dir, "policy.pth")
                    if os.path.exists(policy_path):
                        saved_state_dict = torch.load(policy_path, map_location="cpu")
                        print(f"   ✅ Successfully loaded policy state dict from zip")
                    else:
                        print(f"   ❌ Could not find policy.pth in saved model")
                        raise e

        # Step 4: Get the current model's state dict
        current_state_dict = new_model.policy.state_dict()

        # Step 5: Filter compatible parameters
        compatible_params, skipped_params = filter_state_dict_for_loading(
            saved_state_dict, current_state_dict
        )

        # Step 6: Load the compatible parameters
        print("🔄 Loading compatible parameters...")

        # Use strict=False to allow partial loading
        load_result = new_model.policy.load_state_dict(compatible_params, strict=False)

        if hasattr(load_result, "missing_keys") and load_result.missing_keys:
            print(
                f"   ⚠️  Missing keys (will be randomly initialized): {len(load_result.missing_keys)}"
            )
            for key in load_result.missing_keys[:5]:  # Show first 5
                print(f"      - {key}")

        if hasattr(load_result, "unexpected_keys") and load_result.unexpected_keys:
            print(
                f"   ⚠️  Unexpected keys (ignored): {len(load_result.unexpected_keys)}"
            )

        print(f"✅ Model loaded successfully!")
        print(f"   📊 Loaded {len(compatible_params)} compatible parameters")
        print(f"   ⚠️  Skipped {len(skipped_params)} incompatible parameters")
        print(f"   🎯 Learning rate set to: {learning_rate}")

        # Step 7: Copy other important attributes if possible
        try:
            if hasattr(saved_model, "num_timesteps"):
                new_model.num_timesteps = saved_model.num_timesteps
                print(f"   📈 Restored timesteps: {new_model.num_timesteps}")
        except Exception as e:
            print(f"   ⚠️  Could not restore timesteps: {e}")

        return new_model

    except Exception as e:
        print(f"❌ Error in model loading: {e}")
        logger.error(f"Model loading error: {e}", exc_info=True)
        raise e


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=3000000)
    parser.add_argument(
        "--learning-rate", type=float, default=4e-3
    )  # Use the learning rate from command line
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to model to resume training from"
    )
    parser.add_argument(
        "--game", type=str, default="StreetFighterIISpecialChampionEdition-Genesis"
    )
    parser.add_argument("--state", type=str, default="ken_bison_12.state")

    args = parser.parse_args()

    # Handle retro multi-environment limitation
    if args.render and args.n_envs > 1:
        print("⚠️  WARNING: Retro doesn't support multiple environments with rendering!")
        print("   🔧 Automatically setting n_envs=1 for rendering mode")
        args.n_envs = 1

    # Enhanced logging
    print("🚀 Starting FIXED Street Fighter Training with Enhanced Cross-Attention!")
    print(f"   Total timesteps: {args.total_timesteps:,}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Number of environments: {args.n_envs}")
    print(f"   Rendering: {args.render}")
    if args.resume:
        print(f"   Resuming from: {args.resume}")

    # Set random seed
    set_random_seed(42)

    # Create environments with proper handling of render mode
    render_mode = "human" if args.render else None

    # Always use DummyVecEnv for retro environments to avoid multi-process issues
    env = DummyVecEnv(
        [make_env(args.game, args.state, render_mode) for _ in range(args.n_envs)]
    )

    # Enhanced PPO configuration for cross-attention learning
    policy_kwargs = dict(
        features_extractor_class=StreetFighterCrossAttentionCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.ReLU,
    )

    # FIXED: Create or resume model with enhanced parameter handling
    if args.resume and os.path.exists(args.resume):
        print(f"📂 Loading model from {args.resume}")
        model = load_model_with_cross_attention(
            args.resume, env, args.learning_rate, policy_kwargs
        )
    else:
        print(f"🆕 Creating new model with learning rate: {args.learning_rate}")
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            target_kl=None,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device="auto",
        )

        # Inject cross-attention components for new models
        print("🔗 Injecting cross-attention components...")
        for i, env_instance in enumerate(env.envs):
            wrapper = env_instance.env if hasattr(env_instance, "env") else env_instance
            if hasattr(wrapper, "inject_feature_extractor"):
                wrapper.inject_feature_extractor(model.policy.features_extractor)
                print(f"   ✅ Environment {i+1} injected successfully")

    # Enhanced callback with gradient monitoring
    callback = FixedTrainingCallback(
        save_freq=100000, save_path="./enhanced_oscillation_trained_models/", verbose=1
    )

    # Enhanced training with gradient monitoring
    print("\n🎯 Starting training with enhanced cross-attention learning...")
    print("🔍 Gradient monitoring enabled every 5000 steps")
    print("📊 Attention weight analysis every 1000 steps")
    print("🌊 Oscillation frequency tracking enabled")

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            log_interval=10,
            reset_num_timesteps=False if args.resume else True,
        )

        # Final model save with analysis
        final_model_path = "./enhanced_oscillation_trained_models/final_oscillation_cross_attention_ppo_sf2.zip"
        model.save(final_model_path)
        print(f"🎉 Training completed! Final model saved: {final_model_path}")

        # Save final analysis
        final_analysis = {
            "total_timesteps": args.total_timesteps,
            "learning_rate": args.learning_rate,
            "attention_weights_history": callback.attention_weights_history,
            "oscillation_frequency_history": callback.oscillation_frequency_history,
            "training_completed": datetime.now().isoformat(),
        }

        analysis_path = (
            "./enhanced_oscillation_trained_models/final_training_analysis.json"
        )
        with open(analysis_path, "w") as f:
            json.dump(final_analysis, f, indent=2)

        print(f"📊 Final analysis saved: {analysis_path}")

    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        model.save("./enhanced_oscillation_trained_models/interrupted_model.zip")
        print("💾 Model saved before exit")

    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        logger.error(f"Training error: {e}", exc_info=True)
        model.save("./enhanced_oscillation_trained_models/error_model.zip")
        print("💾 Model saved before exit")

    finally:
        env.close()


if __name__ == "__main__":
    main()
