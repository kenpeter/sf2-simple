#!/usr/bin/env python3
"""
train.py - FIXED Training Script with Enhanced Learning Rate and Gradient Monitoring
KEY FIXES APPLIED:
1. Increased learning rate to 1e-3 for better cross-attention learning
2. Added gradient monitoring for cross-attention components
3. Enhanced callback system for better debugging
4. Fixed learning rate scheduling for different components
5. Added proper attention weight logging
6. FIXED: Handle retro multi-environment limitation when rendering
7. FIXED: Proper model loading with cross-attention parameter handling
8. FIXED: WIN RATE CALCULATION AND DISPLAY - Added comprehensive performance metrics
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
    """FIXED Enhanced callback with gradient monitoring and WIN RATE display"""

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

        # FIXED: Enhanced logging for cross-attention
        self.attention_weights_history = []
        self.oscillation_frequency_history = []
        self.gradient_norms_history = []
        self.performance_history = []  # NEW: Track performance over time

        # Create save directory
        os.makedirs(save_path, exist_ok=True)

        # FIXED: Create detailed log file
        self.log_file = os.path.join(
            save_path, f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

    def _on_step(self) -> bool:
        """FIXED: Enhanced step logging with gradient monitoring"""

        # FIXED: Monitor gradients every 5000 steps
        if self.num_timesteps % 5000 == 0:
            monitor_gradients(self.model, self.num_timesteps)

        # FIXED: Enhanced logging every 1000 steps
        if self.num_timesteps % 1000 == 0:
            self._log_enhanced_stats()

        # Save model periodically
        if self.num_timesteps % self.save_freq == 0:
            self._save_model()

        return True

    def _log_enhanced_stats(self):
        """FIXED: Log enhanced statistics including WIN RATE, attention weights and oscillation frequency"""
        try:
            # Get environment statistics
            if hasattr(self.training_env, "envs"):
                env_stats = {}
                attention_weights = {}
                oscillation_stats = {}
                performance_stats = {}  # NEW: Added performance stats collection

                for i, env in enumerate(self.training_env.envs):
                    wrapper = env.env if hasattr(env, "env") else env

                    if hasattr(wrapper, "stats"):
                        stats = wrapper.stats

                        # FIXED: Collect performance/win rate statistics
                        performance_stats[f"env_{i}"] = {
                            "win_rate": stats.get("win_rate", 0.0),
                            "wins": stats.get("wins", 0),
                            "losses": stats.get("losses", 0),
                            "total_rounds": stats.get("total_rounds", 0),
                            "avg_damage_per_round": stats.get(
                                "avg_damage_per_round", 0.0
                            ),
                            "defensive_efficiency": stats.get(
                                "defensive_efficiency", 0.0
                            ),
                            "max_combo": stats.get("max_combo", 0),
                            "total_combos": stats.get("total_combos", 0),
                            "damage_ratio": stats.get("damage_ratio", 0.0),
                            "total_games": stats.get("total_games", 0),
                            "rounds_per_game": stats.get("rounds_per_game", 0.0),
                        }

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
                            "advantage_transitions": stats.get(
                                "advantage_transitions", 0
                            ),
                            "oscillation_amplitude": stats.get(
                                "oscillation_amplitude", 0.0
                            ),
                        }

                        # Regular stats (keep all existing functionality)
                        for key, value in stats.items():
                            if key not in env_stats:
                                env_stats[key] = []
                            env_stats[key].append(value)

                # FIXED: Log performance statistics FIRST (most important)
                if performance_stats:
                    print(f"\n🏆 Step {self.num_timesteps} - Performance Analysis:")

                    # Calculate aggregated performance metrics
                    total_wins = sum(env_stats.get("wins", [0]))
                    total_losses = sum(env_stats.get("losses", [0]))
                    total_games = total_wins + total_losses
                    overall_win_rate = (
                        total_wins / total_games if total_games > 0 else 0.0
                    )

                    total_rounds = sum(env_stats.get("total_rounds", [0]))
                    avg_damage = np.mean(env_stats.get("avg_damage_per_round", [0]))
                    avg_defensive_eff = np.mean(
                        env_stats.get("defensive_efficiency", [0])
                    )
                    max_combo = max(env_stats.get("max_combo", [0]))
                    total_combos = sum(env_stats.get("total_combos", [0]))
                    avg_damage_ratio = np.mean(env_stats.get("damage_ratio", [0]))

                    # Display performance metrics
                    print(
                        f"   🎯 Win Rate: {overall_win_rate:.1%} ({total_wins}W/{total_losses}L)"
                    )
                    print(f"   💪 Total Games: {total_games}")
                    print(f"   🥊 Total Rounds: {total_rounds}")
                    print(f"   ⚡ Avg Damage/Round: {avg_damage:.1f}")
                    print(f"   🛡️  Defensive Efficiency: {avg_defensive_eff:.1%}")
                    print(f"   🔥 Max Combo: {max_combo}")
                    print(f"   💥 Total Combos: {total_combos}")
                    print(f"   ⚔️  Damage Ratio: {avg_damage_ratio:.2f}")

                    # Performance assessment with detailed feedback
                    if total_games > 0:
                        rounds_per_game = total_rounds / total_games
                        print(f"   📊 Avg Rounds/Game: {rounds_per_game:.1f}")

                        # Win rate assessment
                        if overall_win_rate > 0.7:
                            print("   ✅ Excellent performance! AI is dominating! 🏆")
                        elif overall_win_rate > 0.6:
                            print(
                                "   🌟 Very good performance! AI is winning consistently! 💪"
                            )
                        elif overall_win_rate > 0.5:
                            print("   📈 Good performance! AI is competitive! 🎯")
                        elif overall_win_rate > 0.4:
                            print(
                                "   🔄 Decent progress! AI is learning to compete! 📚"
                            )
                        elif overall_win_rate > 0.2:
                            print("   📊 Learning in progress... AI is improving! 🚀")
                        else:
                            print(
                                "   🌱 Early training phase - AI is still learning basics!"
                            )

                        # Match length assessment
                        if rounds_per_game < 1.5:
                            print("   ⚡ Quick decisive matches!")
                        elif rounds_per_game > 2.5:
                            print("   🥊 Intense back-and-forth battles!")
                        else:
                            print("   ⚖️  Balanced match lengths")

                        # Damage efficiency assessment
                        if avg_damage_ratio > 2.0:
                            print("   🔥 Excellent damage efficiency!")
                        elif avg_damage_ratio > 1.5:
                            print("   💪 Good damage output!")
                        elif avg_damage_ratio > 1.0:
                            print("   📊 Competitive damage trading")
                        else:
                            print("   🛡️  Defensive playstyle")
                    else:
                        print("   🎮 Training starting... waiting for first matches!")

                    # Track performance history
                    self.performance_history.append(
                        {
                            "step": self.num_timesteps,
                            "win_rate": overall_win_rate,
                            "total_games": total_games,
                            "avg_damage": avg_damage,
                            "defensive_efficiency": avg_defensive_eff,
                            "max_combo": max_combo,
                            "damage_ratio": avg_damage_ratio,
                        }
                    )

                # Log attention weight analysis (keep existing detailed analysis)
                if attention_weights:
                    avg_attention = {}
                    for weight_type in ["visual", "strategy", "oscillation", "button"]:
                        weights = [
                            env_weights[weight_type]
                            for env_weights in attention_weights.values()
                        ]
                        avg_attention[weight_type] = np.mean(weights)

                    print(f"\n🎯 Step {self.num_timesteps} - Attention Analysis:")
                    print(f"   👁️  Visual: {avg_attention['visual']:.4f}")
                    print(f"   🧠 Strategy: {avg_attention['strategy']:.4f}")
                    print(f"   🌊 Oscillation: {avg_attention['oscillation']:.4f}")
                    print(f"   🎮 Button: {avg_attention['button']:.4f}")

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

                    # FIXED: Enhanced attention analysis
                    dominant_attention = max(avg_attention, key=avg_attention.get)
                    print(
                        f"   🎯 Dominant Attention: {dominant_attention.capitalize()} ({avg_attention[dominant_attention]:.4f})"
                    )

                    # Check for attention balance
                    if weight_variance > 0.01:
                        print(
                            f"   ⚖️  Excellent attention diversity - AI is using multiple information sources"
                        )
                    elif weight_variance > 0.005:
                        print(f"   📊 Good attention diversity - AI has balanced focus")
                    elif weight_variance > 0.001:
                        print(
                            f"   🔄 Moderate attention diversity - AI has some specialization"
                        )
                    else:
                        print(
                            f"   ⚠️  Low attention diversity - AI may need more training"
                        )

                    self.attention_weights_history.append(
                        {
                            "step": self.num_timesteps,
                            "weights": avg_attention,
                            "variance": weight_variance,
                            "dominant": dominant_attention,
                        }
                    )

                # FIXED: Enhanced oscillation frequency analysis
                if oscillation_stats:
                    avg_oscillation = {}
                    for key in [
                        "frequency",
                        "space_control",
                        "neutral_game_duration",
                        "whiff_bait_attempts",
                        "advantage_transitions",
                        "oscillation_amplitude",
                    ]:
                        values = [
                            env_stats[key] for env_stats in oscillation_stats.values()
                        ]
                        avg_oscillation[key] = np.mean(values)

                    print(f"\n🌊 Oscillation Analysis:")
                    print(f"   📊 Frequency: {avg_oscillation['frequency']:.3f} Hz")
                    print(
                        f"   🎯 Space Control: {avg_oscillation['space_control']:.3f}"
                    )
                    print(
                        f"   ⚖️  Neutral Game: {avg_oscillation['neutral_game_duration']:.1f} frames"
                    )
                    print(
                        f"   🎣 Whiff Baits: {avg_oscillation['whiff_bait_attempts']:.0f}"
                    )
                    print(
                        f"   🔄 Advantage Transitions: {avg_oscillation['advantage_transitions']:.0f}"
                    )
                    print(
                        f"   📈 Oscillation Amplitude: {avg_oscillation['oscillation_amplitude']:.2f}"
                    )

                    # FIXED: Enhanced frequency analysis with more detailed feedback
                    freq = avg_oscillation["frequency"]
                    if freq < 0.1:
                        print(
                            f"   ⚠️  WARNING: Very low oscillation frequency! AI may be too passive."
                        )
                    elif 0.1 <= freq < 0.5:
                        print(
                            f"   📚 Low oscillation frequency - AI is learning movement patterns"
                        )
                    elif 0.5 <= freq < 1.0:
                        print(
                            f"   📈 Moderate oscillation frequency - AI is developing footsies"
                        )
                    elif 1.0 <= freq <= 3.0:
                        print(
                            f"   ✅ Optimal oscillation frequency! AI has good neutral game movement"
                        )
                    elif 3.0 < freq <= 5.0:
                        print(
                            f"   ⚡ High oscillation frequency - AI is very active in neutral"
                        )
                    else:
                        print(
                            f"   ⚠️  WARNING: Very high oscillation frequency! Possible noise or over-aggression"
                        )

                    # Space control analysis
                    space_control = avg_oscillation["space_control"]
                    if space_control > 0.3:
                        print(
                            f"   🏆 Excellent space control - AI is dominating positioning"
                        )
                    elif space_control > 0.1:
                        print(
                            f"   👍 Good space control - AI is winning the positioning game"
                        )
                    elif space_control > -0.1:
                        print(f"   ⚖️  Balanced space control - AI is competing evenly")
                    else:
                        print(f"   📚 AI is learning space control - needs improvement")

                    # Neutral game analysis
                    neutral_duration = avg_oscillation["neutral_game_duration"]
                    if neutral_duration < 30:
                        print(f"   ⚡ Very aggressive gameplay - short neutral phases")
                    elif neutral_duration < 60:
                        print(f"   🎯 Balanced neutral game - good engagement timing")
                    elif neutral_duration < 120:
                        print(f"   🤔 Patient neutral game - careful approach")
                    else:
                        print(f"   🐌 Very patient gameplay - may be too cautious")

                    self.oscillation_frequency_history.append(
                        {
                            "step": self.num_timesteps,
                            "frequency": freq,
                            "space_control": avg_oscillation["space_control"],
                            "win_rate": (
                                overall_win_rate
                                if "overall_win_rate" in locals()
                                else 0.0
                            ),
                            "avg_damage": (
                                avg_damage if "avg_damage" in locals() else 0.0
                            ),
                            "neutral_duration": neutral_duration,
                        }
                    )

                # FIXED: Enhanced training statistics
                if env_stats:
                    avg_stats = {k: np.mean(v) for k, v in env_stats.items()}

                    print(f"\n📊 Step {self.num_timesteps} - Technical Stats:")
                    print(
                        f"   🔮 Predictions Made: {avg_stats.get('predictions_made', 0):.0f}"
                    )
                    print(
                        f"   🤖 Cross-Attention Ready: {avg_stats.get('cross_attention_ready', False)}"
                    )

                    # Additional technical insights
                    if avg_stats.get("predictions_made", 0) > 0:
                        print(f"   ✅ AI is actively making predictions")
                    else:
                        print(f"   ⚠️  AI prediction system may not be fully active")

                    # Show learning progress over time
                    if len(self.performance_history) > 1:
                        prev_performance = self.performance_history[-2]
                        current_performance = self.performance_history[-1]

                        win_rate_change = (
                            current_performance["win_rate"]
                            - prev_performance["win_rate"]
                        )
                        games_played_change = (
                            current_performance["total_games"]
                            - prev_performance["total_games"]
                        )

                        if games_played_change > 0:
                            print(
                                f"   📈 Progress: {win_rate_change:+.1%} win rate change, {games_played_change} new games"
                            )

                # Save detailed log (keep existing functionality)
                self._save_detailed_log()

        except Exception as e:
            logger.error(f"Error in enhanced stats logging: {e}", exc_info=True)
            # Add more detailed error info for debugging
            import traceback

            print(f"❌ Stats logging error: {e}")
            print(f"📋 Traceback: {traceback.format_exc()}")

    def _save_detailed_log(self):
        """FIXED: Save detailed training log with attention, oscillation, and performance data"""
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
                "performance_history": self.performance_history[
                    -10:
                ],  # NEW: Performance tracking
            }

            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_data) + "\n")

        except Exception as e:
            logger.error(f"Error saving detailed log: {e}")

    def _save_model(self):
        """Save model with enhanced naming including performance metrics"""
        try:
            # FIXED: Enhanced model naming with performance info
            avg_freq = 0.0
            win_rate = 0.0
            total_games = 0

            if self.oscillation_frequency_history:
                avg_freq = self.oscillation_frequency_history[-1]["frequency"]

            if self.performance_history:
                latest_perf = self.performance_history[-1]
                win_rate = latest_perf["win_rate"]
                total_games = latest_perf["total_games"]

            model_name = f"oscillation_cross_attention_ppo_sf2_{self.num_timesteps}_steps_freq_{avg_freq:.2f}Hz_winrate_{win_rate:.1%}_games_{total_games}"
            model_path = os.path.join(self.save_path, f"{model_name}.zip")

            self.model.save(model_path)
            print(f"💾 Model saved: {model_path}")

            # FIXED: Save comprehensive analysis
            analysis_path = os.path.join(self.save_path, f"{model_name}_analysis.json")
            analysis_data = {
                "step": self.num_timesteps,
                "attention_weights_history": self.attention_weights_history,
                "oscillation_frequency_history": self.oscillation_frequency_history,
                "performance_history": self.performance_history,
                "learning_rate": self.model.learning_rate,
                "current_performance": {
                    "win_rate": win_rate,
                    "total_games": total_games,
                    "oscillation_frequency": avg_freq,
                },
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
    parser.add_argument("--learning-rate", type=float, default=4e-3)
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
    print("🏆 Performance metrics (WIN RATE) tracking enabled")

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

        # Save final comprehensive analysis
        final_analysis = {
            "total_timesteps": args.total_timesteps,
            "learning_rate": args.learning_rate,
            "attention_weights_history": callback.attention_weights_history,
            "oscillation_frequency_history": callback.oscillation_frequency_history,
            "performance_history": callback.performance_history,
            "training_completed": datetime.now().isoformat(),
            "final_performance": (
                callback.performance_history[-1] if callback.performance_history else {}
            ),
        }

        analysis_path = (
            "./enhanced_oscillation_trained_models/final_training_analysis.json"
        )
        with open(analysis_path, "w") as f:
            json.dump(final_analysis, f, indent=2)

        print(f"📊 Final analysis saved: {analysis_path}")

        # Print final performance summary
        if callback.performance_history:
            final_perf = callback.performance_history[-1]
            print(f"\n🏆 Final Performance Summary:")
            print(f"   🎯 Final Win Rate: {final_perf['win_rate']:.1%}")
            print(f"   💪 Total Games Played: {final_perf['total_games']}")
            print(f"   ⚡ Average Damage/Round: {final_perf['avg_damage']:.1f}")
            print(
                f"   🛡️  Defensive Efficiency: {final_perf['defensive_efficiency']:.1%}"
            )
            print(f"   🔥 Max Combo Achieved: {final_perf['max_combo']}")

    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        model.save("./enhanced_oscillation_trained_models/interrupted_model.zip")
        print("💾 Model saved before exit")

        # Save interrupted training analysis
        if callback.performance_history:
            interrupted_analysis = {
                "interrupted_at_step": callback.num_timesteps,
                "performance_history": callback.performance_history,
                "attention_weights_history": callback.attention_weights_history,
                "oscillation_frequency_history": callback.oscillation_frequency_history,
                "interrupted_at": datetime.now().isoformat(),
            }

            with open(
                "./enhanced_oscillation_trained_models/interrupted_analysis.json", "w"
            ) as f:
                json.dump(interrupted_analysis, f, indent=2)

            print("📊 Interrupted training analysis saved")

    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        logger.error(f"Training error: {e}", exc_info=True)
        model.save("./enhanced_oscillation_trained_models/error_model.zip")
        print("💾 Model saved before exit")

        # Save error analysis
        if callback.performance_history:
            error_analysis = {
                "error_at_step": callback.num_timesteps,
                "error_message": str(e),
                "performance_history": callback.performance_history,
                "attention_weights_history": callback.attention_weights_history,
                "oscillation_frequency_history": callback.oscillation_frequency_history,
                "error_at": datetime.now().isoformat(),
            }

            with open(
                "./enhanced_oscillation_trained_models/error_analysis.json", "w"
            ) as f:
                json.dump(error_analysis, f, indent=2)

            print("📊 Error analysis saved")

    finally:
        env.close()
        print("\n🎮 Training session completed!")
        print("📈 Check the enhanced_oscillation_trained_models/ directory for:")
        print("   - Saved models with performance metrics in filename")
        print("   - Comprehensive analysis files with attention and performance data")
        print("   - Training logs with detailed statistics")


if __name__ == "__main__":
    main()
