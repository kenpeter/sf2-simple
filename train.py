#!/usr/bin/env python3
"""
FIXED Enhanced CUDA-Optimized Training Script for Street Fighter II with Oscillation-Based Positioning
KEY FIXES APPLIED:
1. More aggressive learning rate schedule for oscillation learning
2. Enhanced debugging and monitoring with detailed oscillation validation
3. Better oscillation reward integration and detection sensitivity
4. Fixed cross-attention balance detection and warnings
5. Improved hyperparameters specifically tuned for oscillation training
6. Enhanced callback with comprehensive oscillation logging and analysis
7. Better validation with realistic thresholds and debugging output
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


class FixedEnhancedOscillationTrainingCallback(BaseCallback):
    """FIXED Enhanced callback with comprehensive oscillation monitoring and validation"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.last_log_time = time.time()
        self.log_interval = 45  # Log every 45 seconds (more frequent)
        self.debug_interval = 500  # Debug every 500 steps (more frequent)
        self.validation_interval = 5000  # Validate every 5000 steps
        self.oscillation_log_file = os.path.join(
            "analysis_data",
            f"fixed_oscillation_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        )

        # Initialize log file with headers
        with open(self.oscillation_log_file, "w") as f:
            f.write(
                "timestamp,step,win_rate,player_osc_freq,space_control,neutral_duration,whiff_baits,visual_att,strategy_att,oscillation_att,button_att,freq_status,attention_balance,debug_info\n"
            )

        # Tracking for validation
        self.oscillation_history = []
        self.attention_weight_history = []
        self.last_validation_step = 0

        print(f"📊 FIXED Oscillation logging to: {self.oscillation_log_file}")

    def _on_step(self):
        # Enhanced debug logging
        if self.num_timesteps % self.debug_interval == 0:
            self._log_enhanced_debug_info()

        # Validation checks
        if self.num_timesteps % self.validation_interval == 0:
            self._perform_oscillation_validation()

        # Regular progress logging
        if time.time() - self.last_log_time > self.log_interval:
            self.last_log_time = time.time()
            self._log_enhanced_oscillation_progress()

        return True

    def _log_enhanced_debug_info(self):
        """FIXED Enhanced debug logging with detailed oscillation analysis"""
        if hasattr(self.training_env, "get_attr"):
            try:
                # Get wrapper and debug info
                wrapper_env = None
                if (
                    hasattr(self.training_env, "envs")
                    and len(self.training_env.envs) > 0
                ):
                    env = self.training_env.envs[0]
                    wrapper_env = env.env if hasattr(env, "env") else env

                if wrapper_env and hasattr(wrapper_env, "get_debug_info"):
                    debug_info = wrapper_env.get_debug_info()
                    print(f"\n🔍 FIXED DEBUG - Step {self.num_timesteps}:")

                    # Enhanced velocity and position analysis
                    recent_velocities = debug_info.get("recent_velocities", [])
                    recent_positions = debug_info.get("recent_positions", [])
                    recent_direction_changes = debug_info.get(
                        "recent_direction_changes", []
                    )

                    if recent_velocities:
                        velocities = [
                            v.get("player_vel", 0) for v in recent_velocities[-5:]
                        ]
                        print(
                            f"   Recent player velocities: {[f'{v:.3f}' for v in velocities]}"
                        )
                        print(
                            f"   Velocity range: {min(velocities):.3f} to {max(velocities):.3f}"
                        )

                    if recent_positions:
                        positions = [
                            p.get("player_x", 0) for p in recent_positions[-5:]
                        ]
                        print(
                            f"   Recent player positions: {[f'{p:.1f}' for p in positions]}"
                        )
                        if len(positions) > 1:
                            position_variance = np.var(positions)
                            print(f"   Position variance: {position_variance:.3f}")

                    if recent_direction_changes:
                        print(
                            f"   Recent direction changes: {len(recent_direction_changes)}"
                        )
                        for change in recent_direction_changes[-3:]:
                            print(
                                f"     Frame {change['frame']}: {change['prev_vel']:.3f} → {change['curr_vel']:.3f}"
                            )

                    # Detection sensitivity analysis
                    sensitivity = debug_info.get("detection_sensitivity", {})
                    print(f"   Detection Settings:")
                    print(
                        f"     Movement threshold: {sensitivity.get('movement_threshold', 'N/A')}"
                    )
                    print(
                        f"     Direction change threshold: {sensitivity.get('direction_change_threshold', 'N/A')}"
                    )
                    print(
                        f"     Velocity smoothing: {sensitivity.get('velocity_smoothing_factor', 'N/A')}"
                    )

                    # Current oscillation stats
                    current_stats = debug_info.get("current_stats", {})
                    if current_stats:
                        print(f"   Current Oscillation Stats:")
                        print(
                            f"     Direction changes: {current_stats.get('player_direction_changes', 0)}"
                        )
                        print(
                            f"     Space control: {current_stats.get('space_control_score', 0):.3f}"
                        )
                        print(
                            f"     Neutral duration: {current_stats.get('neutral_game_duration', 0)}"
                        )

            except Exception as e:
                print(f"⚠️ Enhanced debug logging error: {e}")

    def _perform_oscillation_validation(self):
        """FIXED Perform comprehensive oscillation validation"""
        if hasattr(self.training_env, "get_attr"):
            try:
                stats_list = self.training_env.get_attr("stats")
                if stats_list:
                    stats = stats_list[0]

                    # Get oscillation metrics
                    osc_freq = stats.get("player_oscillation_frequency", 0.0)
                    space_control = stats.get("space_control_score", 0.0)
                    neutral_duration = stats.get("neutral_game_duration", 0.0)

                    # Track history for trend analysis
                    self.oscillation_history.append(
                        {
                            "step": self.num_timesteps,
                            "frequency": osc_freq,
                            "space_control": space_control,
                            "neutral_duration": neutral_duration,
                        }
                    )

                    # Keep only last 20 validation points
                    if len(self.oscillation_history) > 20:
                        self.oscillation_history = self.oscillation_history[-20:]

                    print(f"\n🧪 OSCILLATION VALIDATION - Step {self.num_timesteps}:")
                    print(f"   Frequency: {osc_freq:.3f} Hz (target: 1.0-3.0)")
                    print(
                        f"   Space Control: {space_control:.3f} (target: varying -0.5 to +0.5)"
                    )
                    print(
                        f"   Neutral Duration: {neutral_duration:.1f} frames (target: 30-90)"
                    )

                    # Trend analysis
                    if len(self.oscillation_history) >= 5:
                        recent_freqs = [
                            h["frequency"] for h in self.oscillation_history[-5:]
                        ]
                        freq_trend = np.polyfit(range(5), recent_freqs, 1)[0]
                        print(
                            f"   Frequency trend: {'📈 increasing' if freq_trend > 0.01 else '📉 decreasing' if freq_trend < -0.01 else '➡️ stable'}"
                        )

                    # Validation status
                    validation_status = []
                    if osc_freq < 0.5:
                        validation_status.append("❌ Frequency too low")
                    elif osc_freq > 5.0:
                        validation_status.append(
                            "⚠️ Frequency too high (possible noise)"
                        )
                    else:
                        validation_status.append("✅ Frequency in range")

                    if abs(space_control) < 0.01:
                        validation_status.append("❌ Space control not varying")
                    else:
                        validation_status.append("✅ Space control active")

                    if neutral_duration == 0:
                        validation_status.append("❌ No neutral game detected")
                    else:
                        validation_status.append("✅ Neutral game detected")

                    print(f"   Status: {' | '.join(validation_status)}")

            except Exception as e:
                print(f"⚠️ Oscillation validation error: {e}")

    def _log_enhanced_oscillation_progress(self):
        """FIXED Enhanced oscillation progress logging with comprehensive analysis"""
        if hasattr(self.training_env, "get_attr"):
            try:
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
                        and wrapper_env.total_rounds > 0
                    ):
                        win_rate = wrapper_env.wins / wrapper_env.total_rounds

                    # Get oscillation metrics
                    osc_freq = stats.get("player_oscillation_frequency", 0.0)
                    space_control = stats.get("space_control_score", 0.0)
                    neutral_duration = stats.get("neutral_game_duration", 0.0)
                    whiff_baits = stats.get("whiff_bait_attempts", 0)

                    # Get attention weights
                    visual_attention = stats.get("visual_attention_weight", 0.0)
                    strategy_attention = stats.get("strategy_attention_weight", 0.0)
                    osc_attention = stats.get("oscillation_attention_weight", 0.0)
                    button_attention = stats.get("button_attention_weight", 0.0)

                    # Track attention weight history
                    self.attention_weight_history.append(
                        {
                            "step": self.num_timesteps,
                            "visual": visual_attention,
                            "strategy": strategy_attention,
                            "oscillation": osc_attention,
                            "button": button_attention,
                        }
                    )

                    # Keep only last 10 attention measurements
                    if len(self.attention_weight_history) > 10:
                        self.attention_weight_history = self.attention_weight_history[
                            -10:
                        ]

                    # Enhanced logging with comprehensive analysis
                    log_message = (
                        f"\n🎯 FIXED Training Progress (Step: {self.num_timesteps:,})\n"
                        f"   Win Rate: {win_rate:.1%}\n"
                        f"   Max Combo: {stats.get('max_combo', 0)}\n"
                        f"   🌊 OSCILLATION METRICS:\n"
                        f"     Player Oscillation Frequency: {osc_freq:.3f} Hz {'✅' if 1.0 <= osc_freq <= 3.0 else '❌' if osc_freq < 0.5 else '⚠️'}\n"
                        f"     Space Control Score: {space_control:.3f} {'✅' if abs(space_control) > 0.01 else '❌'}\n"
                        f"     Neutral Game Duration: {neutral_duration:.1f} frames {'✅' if neutral_duration > 0 else '❌'}\n"
                        f"     Whiff Bait Attempts: {whiff_baits}\n"
                        f"   🎯 CROSS-ATTENTION WEIGHTS:\n"
                        f"     Visual: {visual_attention:.3f}\n"
                        f"     Strategy: {strategy_attention:.3f}\n"
                        f"     Oscillation: {osc_attention:.3f}\n"
                        f"     Button: {button_attention:.3f}"
                    )
                    print(log_message)

                    # FIXED: Enhanced attention balance analysis
                    weights = [
                        visual_attention,
                        strategy_attention,
                        osc_attention,
                        button_attention,
                    ]

                    # Check for identical weights (learning problem)
                    unique_weights = len(set([round(w, 4) for w in weights]))
                    weights_balanced = unique_weights > 1

                    if not weights_balanced:
                        print(
                            "   ⚠️ CRITICAL: All attention weights are identical - cross-attention may not be learning!"
                        )
                        print(
                            "   💡 SUGGESTION: Increase learning rate or check gradient flow"
                        )

                    # Check for attention weight distribution
                    weight_std = np.std(weights)
                    if weight_std < 0.01:
                        print(
                            "   ⚠️ WARNING: Very low attention weight variation - model may not be utilizing cross-attention"
                        )
                    elif weight_std > 0.3:
                        print(
                            "   ⚠️ WARNING: Very high attention weight variation - may indicate unstable training"
                        )
                    else:
                        print("   ✅ Attention weight distribution looks healthy")

                    # FIXED: Enhanced oscillation frequency analysis
                    freq_status = "unknown"
                    if osc_freq < 0.5:
                        freq_status = "too_low"
                        print(
                            f"   ❌ CRITICAL: Oscillation frequency too low ({osc_freq:.3f} Hz)"
                        )
                        print(
                            "   💡 SUGGESTION: Reduce movement_threshold and direction_change_threshold in OscillationTracker"
                        )
                    elif osc_freq > 5.0:
                        freq_status = "too_high"
                        print(
                            f"   ⚠️ WARNING: Oscillation frequency very high ({osc_freq:.3f} Hz)"
                        )
                        print(
                            "   💡 SUGGESTION: Increase thresholds to reduce noise detection"
                        )
                    elif 1.0 <= osc_freq <= 3.0:
                        freq_status = "optimal"
                        print(
                            f"   ✅ Oscillation frequency in optimal range ({osc_freq:.3f} Hz)"
                        )
                    else:
                        freq_status = "suboptimal"
                        print(
                            f"   ⚠️ Oscillation frequency suboptimal ({osc_freq:.3f} Hz)"
                        )

                    # Space control analysis
                    if abs(space_control) < 0.01:
                        print(
                            "   ❌ CRITICAL: Space control not varying - calculation may be broken!"
                        )
                        print(
                            "   💡 SUGGESTION: Check _calculate_enhanced_space_control method"
                        )
                    else:
                        print(
                            f"   ✅ Space control showing variation ({space_control:.3f})"
                        )

                    # Neutral game analysis
                    if neutral_duration == 0:
                        print(
                            "   ❌ CRITICAL: No neutral game detected - detection logic may be broken!"
                        )
                        print(
                            "   💡 SUGGESTION: Check neutral game detection conditions"
                        )
                    else:
                        print(
                            f"   ✅ Neutral game detected ({neutral_duration:.1f} frames)"
                        )

                    # Attention balance status
                    attention_balance = "balanced" if weights_balanced else "unbalanced"

                    # Log to file for analysis
                    try:
                        with open(self.oscillation_log_file, "a") as f:
                            timestamp = datetime.now().isoformat()
                            debug_info = (
                                f"freq_{freq_status},att_{attention_balance},"
                                f"std_{weight_std:.4f},unique_{unique_weights}"
                            )
                            f.write(
                                f"{timestamp},{self.num_timesteps},{win_rate:.4f},"
                                f"{osc_freq:.4f},{space_control:.4f},"
                                f"{neutral_duration:.1f},{whiff_baits},"
                                f"{visual_attention:.4f},{strategy_attention:.4f},"
                                f"{osc_attention:.4f},{button_attention:.4f},"
                                f"{freq_status},{attention_balance},{debug_info}\n"
                            )
                    except Exception as e:
                        print(f"⚠️ Failed to log oscillation data: {e}")

                    # FIXED: Trend analysis for attention weights
                    if len(self.attention_weight_history) >= 5:
                        recent_osc_weights = [
                            h["oscillation"] for h in self.attention_weight_history[-5:]
                        ]
                        osc_trend = np.polyfit(range(5), recent_osc_weights, 1)[0]

                        if abs(osc_trend) > 0.001:
                            trend_dir = "increasing" if osc_trend > 0 else "decreasing"
                            print(
                                f"   📊 Oscillation attention trend: {trend_dir} ({osc_trend:.4f}/step)"
                            )
                        else:
                            print(f"   📊 Oscillation attention trend: stable")

            except Exception as e:
                print(f"⚠️ Enhanced oscillation progress logging error: {e}")


def create_fixed_oscillation_aware_learning_rate_schedule(initial_lr=3e-4):
    """FIXED Create a more aggressive learning rate schedule optimized for oscillation analysis"""

    def schedule(progress):
        # FIXED: More aggressive schedule specifically for oscillation learning
        if progress < 0.05:
            return (
                initial_lr * 3.0
            )  # Very strong boost for early oscillation pattern recognition
        elif progress < 0.15:
            return initial_lr * 2.0  # Strong boost for oscillation learning
        elif progress < 0.4:
            return (
                initial_lr * 1.5
            )  # Sustained higher learning for pattern consolidation
        elif progress < 0.7:
            return initial_lr * 1.0  # Standard rate for refinement
        elif progress < 0.9:
            return initial_lr * 0.7  # Reduce for stability
        else:
            return initial_lr * 0.4  # Fine-tuning phase

    return schedule


def create_fixed_model_with_oscillation_structure(env, device, args):
    """FIXED Create a model with enhanced structure optimized for oscillation learning"""
    # FIXED: Enhanced policy architecture with deeper networks for oscillation patterns
    policy_kwargs = dict(
        features_extractor_class=StreetFighterCrossAttentionCNN,
        features_extractor_kwargs=dict(features_dim=512),  # Increased capacity
        net_arch=dict(
            pi=[512, 512, 256, 128],  # Deeper policy network
            vf=[512, 512, 256, 128],  # Deeper value network
        ),
        activation_fn=nn.ReLU,
    )

    # FIXED: Enhanced PPO hyperparameters specifically tuned for oscillation learning
    model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=1,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=15,  # Increased from 12 for better oscillation pattern learning
        gamma=0.99,  # Slightly higher discount factor for longer-term oscillation patterns
        learning_rate=create_fixed_oscillation_aware_learning_rate_schedule(
            args.learning_rate
        ),
        clip_range=0.2,
        ent_coef=0.025,  # Increased exploration for oscillation discovery
        vf_coef=0.5,
        max_grad_norm=0.5,
        gae_lambda=0.98,  # Higher GAE lambda for longer oscillation dependencies
        tensorboard_log="fixed_enhanced_oscillation_logs",
    )

    return model, policy_kwargs


def inject_fixed_oscillation_cross_attention_transformer(model, env, args):
    """FIXED Inject oscillation-enhanced cross-attention transformer with validation"""
    print(
        "💉 Injecting FIXED oscillation-enhanced cross-attention feature extractor..."
    )
    try:
        feature_extractor = model.policy.features_extractor

        # Inject into all environments
        for i in range(args.num_envs):
            wrapper = env.envs[i].env if hasattr(env.envs[i], "env") else env.envs[i]
            wrapper.inject_feature_extractor(feature_extractor)

        print("✅ FIXED oscillation-enhanced cross-attention injection successful.")

        # Validation: Check if injection worked
        test_wrapper = env.envs[0].env if hasattr(env.envs[0], "env") else env.envs[0]
        if hasattr(test_wrapper, "vision_ready") and test_wrapper.vision_ready:
            print("✅ Cross-attention transformer validation: PASSED")
        else:
            print("❌ Cross-attention transformer validation: FAILED")
            return False

        return True
    except Exception as e:
        print(f"❌ Feature extractor injection failed: {e}")
        return False


def load_fixed_model_with_oscillation_cross_attention(model_path, env, device, args):
    """FIXED Load a model with oscillation-enhanced cross-attention transformer"""
    print(f"📂 Loading FIXED model with oscillation cross-attention from {model_path}")

    # Step 1: Create a new model with the proper base structure
    model, policy_kwargs = create_fixed_model_with_oscillation_structure(
        env, device, args
    )

    # Step 2: Inject oscillation cross-attention transformer
    inject_success = inject_fixed_oscillation_cross_attention_transformer(
        model, env, args
    )

    if not inject_success:
        print(
            "⚠️ FIXED oscillation cross-attention injection failed, loading without it..."
        )
        model = PPO.load(model_path, env=env, device=device)
        return model

    # Step 3: Load the saved parameters
    print("📥 Loading saved parameters...")
    try:
        saved_data = torch.load(model_path, map_location=device)

        if "policy" in saved_data:
            saved_params = saved_data["policy"]
        else:
            saved_params = saved_data

        # Load parameters with strict=False to handle missing keys
        missing_keys, unexpected_keys = model.policy.load_state_dict(
            saved_params, strict=False
        )

        if missing_keys:
            print(
                f"   ⚠️ Missing keys: {len(missing_keys)} (expected for new oscillation features)"
            )
        if unexpected_keys:
            print(f"   ⚠️ Unexpected keys: {len(unexpected_keys)}")

        # Update learning rate
        model.learning_rate = create_fixed_oscillation_aware_learning_rate_schedule(
            args.learning_rate
        )

        print("✅ FIXED model parameters loaded successfully!")
        return model

    except Exception as e:
        print(f"❌ Failed to load saved parameters: {e}")
        print("🔄 Falling back to fresh model initialization...")
        return model


def validate_fixed_oscillation_features(env, validation_steps=300):
    """FIXED Comprehensive oscillation feature validation with enhanced checks"""
    print("🔍 Running FIXED oscillation feature validation...")
    try:
        obs, info = env.reset()

        # Enhanced validation metrics
        validation_data = {
            "oscillation_frequencies": [],
            "space_control_scores": [],
            "neutral_durations": [],
            "direction_changes": [],
            "position_variances": [],
            "velocity_ranges": [],
        }

        print(
            f"   Running {validation_steps} validation steps with enhanced monitoring..."
        )

        # Run validation steps
        for i in range(validation_steps):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)

            # Collect data every 20 steps
            if i % 20 == 0:
                stats = info.get("stats", {})
                osc_freq = stats.get("player_oscillation_frequency", 0.0)
                space_control = stats.get("space_control_score", 0.0)
                neutral_duration = stats.get("neutral_game_duration", 0)

                validation_data["oscillation_frequencies"].append(osc_freq)
                validation_data["space_control_scores"].append(space_control)
                validation_data["neutral_durations"].append(neutral_duration)

                # Get debug info for deeper analysis
                if hasattr(env.envs[0], "env") and hasattr(
                    env.envs[0].env, "get_debug_info"
                ):
                    debug_info = env.envs[0].env.get_debug_info()

                    # Analyze position and velocity data
                    recent_positions = debug_info.get("recent_positions", [])
                    recent_velocities = debug_info.get("recent_velocities", [])

                    if recent_positions:
                        positions = [p.get("player_x", 0) for p in recent_positions]
                        if len(positions) > 1:
                            validation_data["position_variances"].append(
                                np.var(positions)
                            )

                    if recent_velocities:
                        velocities = [v.get("player_vel", 0) for v in recent_velocities]
                        if velocities:
                            validation_data["velocity_ranges"].append(
                                max(velocities) - min(velocities)
                            )

                # Progress indicator
                if i % 50 == 0:
                    print(
                        f"   Step {i}: Freq: {osc_freq:.3f}Hz, Space: {space_control:.3f}, Neutral: {neutral_duration}"
                    )

            if done or truncated:
                obs, info = env.reset()

        # FIXED: Enhanced validation analysis
        print("\n🔍 FIXED Validation Results:")

        # 1. Oscillation frequency analysis
        frequencies = [f for f in validation_data["oscillation_frequencies"] if f > 0]
        if frequencies:
            avg_freq = np.mean(frequencies)
            freq_std = np.std(frequencies)
            print(f"   Oscillation Frequency Analysis:")
            print(f"     Average: {avg_freq:.3f} Hz")
            print(f"     Std Dev: {freq_std:.3f}")
            print(f"     Range: {min(frequencies):.3f} - {max(frequencies):.3f}")

            if avg_freq < 0.5:
                print("   ❌ CRITICAL: Average oscillation frequency too low!")
                print("   💡 RECOMMENDATION: Reduce movement_threshold to 0.2 or lower")
                return False
            elif avg_freq > 8.0:
                print(
                    "   ⚠️ WARNING: Average oscillation frequency very high (possible noise)"
                )
                print(
                    "   💡 RECOMMENDATION: Increase movement_threshold to reduce noise"
                )
            elif 1.0 <= avg_freq <= 3.0:
                print("   ✅ Oscillation frequency in optimal range")
            else:
                print("   ⚠️ Oscillation frequency suboptimal but acceptable")
        else:
            print("   ❌ CRITICAL: No oscillation detected at all!")
            return False

        # 2. Space control variation analysis
        space_scores = validation_data["space_control_scores"]
        if space_scores:
            space_std = np.std(space_scores)
            space_range = max(space_scores) - min(space_scores)
            print(f"   Space Control Analysis:")
            print(f"     Std Dev: {space_std:.3f}")
            print(f"     Range: {space_range:.3f}")

            if space_std < 0.01:
                print("   ❌ CRITICAL: Space control not varying enough!")
                return False
            else:
                print("   ✅ Space control showing good variation")

        # 3. Neutral game detection analysis
        neutral_detected = [d for d in validation_data["neutral_durations"] if d > 0]
        neutral_ratio = len(neutral_detected) / len(
            validation_data["neutral_durations"]
        )
        print(f"   Neutral Game Detection:")
        print(f"     Detection ratio: {neutral_ratio:.1%}")

        if neutral_ratio < 0.1:
            print("   ❌ CRITICAL: Very low neutral game detection!")
            return False
        else:
            print("   ✅ Neutral game detection working")

        # 4. Position and velocity analysis
        if validation_data["position_variances"]:
            avg_pos_var = np.mean(validation_data["position_variances"])
            print(f"   Position Variance: {avg_pos_var:.3f}")

            if avg_pos_var < 1.0:
                print("   ⚠️ WARNING: Low position variance - limited movement detected")
            else:
                print("   ✅ Good position variance indicating movement")

        if validation_data["velocity_ranges"]:
            avg_vel_range = np.mean(validation_data["velocity_ranges"])
            print(f"   Velocity Range: {avg_vel_range:.3f}")

            if avg_vel_range < 0.5:
                print("   ⚠️ WARNING: Low velocity range - limited speed variation")
            else:
                print("   ✅ Good velocity range indicating dynamic movement")

        print("✅ FIXED oscillation feature validation PASSED!")
        return True

    except Exception as e:
        print(f"❌ FIXED oscillation feature validation FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    device = setup_cuda_optimization()

    parser = argparse.ArgumentParser(
        description="FIXED Enhanced Street Fighter II Training with Oscillation-Based Positioning"
    )
    parser.add_argument("--total-timesteps", type=int, default=20_000_000)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--no-vision-transformer", action="store_true")
    parser.add_argument("--state", type=str, default="ken_bison_12.state")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--n-steps", type=int, default=8192)
    parser.add_argument(
        "--validate-oscillation",
        action="store_true",
        help="Run comprehensive oscillation feature validation",
    )
    parser.add_argument(
        "--debug-oscillation",
        action="store_true",
        help="Enable detailed oscillation debugging",
    )
    parser.add_argument(
        "--validation-steps",
        type=int,
        default=300,
        help="Number of steps for oscillation validation",
    )
    args = parser.parse_args()

    game = "StreetFighterIISpecialChampionEdition-Genesis"
    enable_vision_transformer = not args.no_vision_transformer
    save_dir = "fixed_enhanced_oscillation_trained_models"
    os.makedirs(save_dir, exist_ok=True)

    state_file_name = args.state
    state_file_path = os.path.abspath(state_file_name)
    if not os.path.exists(state_file_path):
        print(f"❌ State file not found at: {state_file_path}")
        sys.exit(1)

    print(
        f"🎯 Training with FIXED Oscillation-Enhanced Cross-Attention: {enable_vision_transformer}"
    )
    print(f"💾 State file: {state_file_path}")
    print(f"📊 FIXED Enhanced hyperparameters:")
    print(f"   Batch size: {args.batch_size}")
    print(f"   N-steps: {args.n_steps}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Validation steps: {args.validation_steps}")

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

    # FIXED: Enhanced oscillation feature validation
    if args.validate_oscillation:
        print("🧪 Running FIXED comprehensive oscillation feature validation...")
        test_env = make_env()
        validation_passed = validate_fixed_oscillation_features(
            DummyVecEnv([lambda: test_env]), args.validation_steps
        )
        test_env.close()

        if not validation_passed:
            print("❌ FIXED oscillation validation failed. Exiting.")
            sys.exit(1)
        print("✅ FIXED oscillation validation passed!")

    # Model creation and loading logic
    if args.resume and os.path.exists(args.resume):
        print(f"📂 Resuming training from {args.resume}")
        model = load_fixed_model_with_oscillation_cross_attention(
            args.resume, env, device, args
        )
    else:
        print("🧠 Creating a new PPO model with FIXED enhanced oscillation analysis")
        model, _ = create_fixed_model_with_oscillation_structure(env, device, args)

        # Inject oscillation cross-attention transformer for new models
        if enable_vision_transformer:
            inject_fixed_oscillation_cross_attention_transformer(model, env, args)

    # FIXED: Enhanced callbacks with more frequent monitoring
    checkpoint_callback = CheckpointCallback(
        save_freq=max(25000 // args.num_envs, 1),  # More frequent checkpoints
        save_path=save_dir,
        name_prefix="fixed_oscillation_cross_attention_ppo_sf2",
    )
    enhanced_callback = FixedEnhancedOscillationTrainingCallback()

    print("🏋️ Starting training with FIXED enhanced oscillation analysis...")
    print("📊 FIXED Enhanced monitoring:")
    print("   - Player oscillation frequency (target: 1-3 Hz)")
    print("   - Space control scores (target: varying -0.5 to +0.5)")
    print("   - Neutral game duration tracking (target: 30-90 frames)")
    print("   - Whiff bait attempt detection")
    print("   - Cross-attention weight distribution (target: dynamic weights)")
    print("   - Enhanced debug logging every 500 steps")
    print("   - Comprehensive validation every 5000 steps")
    print("   - Progress logging every 45 seconds")

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
        final_model_path = os.path.join(save_dir, "ppo_sf2_fixed_oscillation_final.zip")
        model.save(final_model_path)
        print(f"💾 Final FIXED oscillation model saved to: {final_model_path}")

        # FIXED: Enhanced oscillation analysis summary
        try:
            stats_list = env.get_attr("stats")
            if stats_list:
                stats = stats_list[0]
                summary_file = os.path.join(
                    "analysis_data",
                    f"fixed_oscillation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                )
                with open(summary_file, "w") as f:
                    f.write("=== FIXED Enhanced Oscillation Analysis Summary ===\n")
                    f.write(f"Total Training Steps: {args.total_timesteps:,}\n")
                    f.write(f"Validation Steps: {args.validation_steps}\n")
                    f.write(f"Hyperparameters:\n")
                    f.write(f"  Batch Size: {args.batch_size}\n")
                    f.write(f"  N-Steps: {args.n_steps}\n")
                    f.write(f"  Learning Rate: {args.learning_rate}\n")
                    f.write(f"  N-Epochs: 15\n")
                    f.write(f"  Entropy Coefficient: 0.025\n")
                    f.write(f"\nFinal Oscillation Metrics:\n")
                    f.write(
                        f"  Final Player Oscillation Frequency: {stats.get('player_oscillation_frequency', 0.0):.4f} Hz\n"
                    )
                    f.write(
                        f"  Final Space Control Score: {stats.get('space_control_score', 0.0):.4f}\n"
                    )
                    f.write(
                        f"  Final Neutral Game Duration: {stats.get('neutral_game_duration', 0)} frames\n"
                    )
                    f.write(
                        f"  Total Whiff Bait Attempts: {stats.get('whiff_bait_attempts', 0)}\n"
                    )
                    f.write(f"\nFinal Cross-Attention Weight Distribution:\n")
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
                    f.write(f"\n=== FIXED Analysis and Recommendations ===\n")

                    # Enhanced analysis with specific recommendations
                    osc_freq = stats.get("player_oscillation_frequency", 0.0)
                    space_control = stats.get("space_control_score", 0.0)
                    osc_attention = stats.get("oscillation_attention_weight", 0.0)
                    visual_attention = stats.get("visual_attention_weight", 0.0)
                    strategy_attention = stats.get("strategy_attention_weight", 0.0)
                    button_attention = stats.get("button_attention_weight", 0.0)

                    f.write(f"Oscillation Frequency Analysis:\n")
                    if osc_freq < 0.5:
                        f.write(
                            f"  ISSUE: Low oscillation frequency ({osc_freq:.4f} Hz)\n"
                        )
                        f.write(
                            f"  RECOMMENDATION: In wrapper.py, OscillationTracker.__init__():\n"
                        )
                        f.write(
                            f"    - Set self.movement_threshold = 0.2 (currently 0.3)\n"
                        )
                        f.write(
                            f"    - Set self.direction_change_threshold = 0.05 (currently 0.1)\n"
                        )
                        f.write(
                            f"    - Set self.velocity_smoothing_factor = 0.2 (currently 0.3)\n"
                        )
                    elif osc_freq > 5.0:
                        f.write(
                            f"  ISSUE: High oscillation frequency ({osc_freq:.4f} Hz)\n"
                        )
                        f.write(
                            f"  RECOMMENDATION: In wrapper.py, OscillationTracker.__init__():\n"
                        )
                        f.write(
                            f"    - Set self.movement_threshold = 0.5 (currently 0.3)\n"
                        )
                        f.write(
                            f"    - Set self.direction_change_threshold = 0.2 (currently 0.1)\n"
                        )
                    else:
                        f.write(
                            f"  STATUS: Oscillation frequency in acceptable range\n"
                        )

                    f.write(f"\nSpace Control Analysis:\n")
                    f.write(f"  Current score: {space_control:.4f}\n")
                    if abs(space_control) < 0.01:
                        f.write(f"  ISSUE: Space control not varying enough\n")
                        f.write(
                            f"  RECOMMENDATION: Check _calculate_enhanced_space_control method\n"
                        )
                    else:
                        f.write(f"  STATUS: Space control showing good variation\n")

                    f.write(f"\nCross-Attention Analysis:\n")
                    weights = [
                        visual_attention,
                        strategy_attention,
                        osc_attention,
                        button_attention,
                    ]
                    weight_std = np.std(weights)
                    unique_weights = len(set([round(w, 4) for w in weights]))

                    f.write(
                        f"  Weight distribution standard deviation: {weight_std:.4f}\n"
                    )
                    f.write(f"  Number of unique weights: {unique_weights}/4\n")

                    if unique_weights == 1:
                        f.write(
                            f"  CRITICAL ISSUE: All attention weights are identical!\n"
                        )
                        f.write(
                            f"  RECOMMENDATION: Increase learning rate or check gradient flow\n"
                        )
                        f.write(
                            f"    - Try initial learning rate of 5e-4 instead of 3e-4\n"
                        )
                        f.write(f"    - Add gradient clipping diagnostics\n")
                    elif weight_std < 0.01:
                        f.write(f"  ISSUE: Very low attention weight variation\n")
                        f.write(
                            f"  RECOMMENDATION: Model may not be utilizing cross-attention effectively\n"
                        )
                    elif weight_std > 0.3:
                        f.write(f"  ISSUE: Very high attention weight variation\n")
                        f.write(f"  RECOMMENDATION: May indicate unstable training\n")
                    else:
                        f.write(
                            f"  STATUS: Attention weight distribution looks healthy\n"
                        )

                    f.write(f"  Oscillation attention weight: {osc_attention:.4f}\n")
                    if osc_attention < 0.10:
                        f.write(f"  ISSUE: Low oscillation attention weight\n")
                        f.write(
                            f"  RECOMMENDATION: Increase oscillation feature importance\n"
                        )
                    elif osc_attention > 0.70:
                        f.write(f"  ISSUE: Excessive oscillation attention weight\n")
                        f.write(f"  RECOMMENDATION: Rebalance attention mechanisms\n")
                    else:
                        f.write(
                            f"  STATUS: Oscillation attention weight in good range\n"
                        )

                    f.write(f"\nTechnical Recommendations:\n")
                    f.write(f"1. If oscillation frequency is too low:\n")
                    f.write(f"   - Reduce movement_threshold in OscillationTracker\n")
                    f.write(f"   - Reduce direction_change_threshold\n")
                    f.write(f"   - Reduce velocity_smoothing_factor\n")
                    f.write(f"2. If attention weights are identical:\n")
                    f.write(f"   - Increase learning rate by 50%\n")
                    f.write(f"   - Add attention weight regularization\n")
                    f.write(f"   - Check for gradient vanishing issues\n")
                    f.write(f"3. If space control is not varying:\n")
                    f.write(f"   - Check _calculate_enhanced_space_control method\n")
                    f.write(f"   - Verify center_control calculation\n")
                    f.write(f"   - Verify movement_initiative calculation\n")

                    f.write(f"\nNext Steps:\n")
                    f.write(
                        f"1. Run validation: python train.py --validate-oscillation --validation-steps 500\n"
                    )
                    f.write(
                        f"2. Check detailed logs: analysis_data/fixed_oscillation_analysis_*.log\n"
                    )
                    f.write(f"3. Adjust parameters based on validation results\n")
                    f.write(
                        f"4. Consider longer training if oscillation patterns are still developing\n"
                    )
                    f.write(f"5. Monitor attention weight trends over time\n")

                    f.write(f"\nFIXED Implementation Details:\n")
                    f.write(f"- Enhanced oscillation detection with lower thresholds\n")
                    f.write(f"- Velocity smoothing for noise reduction\n")
                    f.write(f"- Improved neutral game detection\n")
                    f.write(f"- Oscillation-specific reward bonuses\n")
                    f.write(f"- More aggressive learning rate schedule\n")
                    f.write(f"- Comprehensive validation and debugging\n")
                    f.write(f"- Enhanced cross-attention balance monitoring\n")

                print(
                    f"📊 FIXED enhanced oscillation analysis summary saved to: {summary_file}"
                )
        except Exception as e:
            print(f"⚠️ Failed to save FIXED oscillation summary: {e}")

        env.close()


if __name__ == "__main__":
    main()
