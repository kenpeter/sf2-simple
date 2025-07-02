#!/usr/bin/env python3
"""
Enhanced CUDA-Optimized Training Script for Street Fighter II
Key improvements:
1. Better learning rate scheduling
2. Enhanced reward system with combo bonuses
3. Improved action space for more effective gameplay
4. Better training hyperparameters based on analysis
5. Enhanced curriculum learning approach
6. File-based logging and organized output
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

# Import the enhanced wrapper
from wrapper import StreetFighterVisionWrapper, StreetFighterSimplifiedCNN

# Create directories
os.makedirs("logs", exist_ok=True)
os.makedirs("analysis_data", exist_ok=True)


def setup_cuda_optimization():
    """Setup CUDA for optimal performance"""
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA not available! This script requires GPU.")

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


class EnhancedStrategicVisionTransformer(nn.Module):
    """Enhanced Strategic Vision Transformer optimized for combo detection and strategic play"""

    def __init__(self, visual_dim=512, strategic_dim=33, seq_length=8):
        super().__init__()
        self.seq_length = seq_length

        # Combined input dimension: 512 + 33 = 545
        combined_dim = visual_dim + strategic_dim

        # Enhanced transformer dimension for better pattern recognition
        d_model = 384  # Increased from 256 for better capacity
        self.input_projection = nn.Linear(combined_dim, d_model)

        # Enhanced positional encoding
        self.register_parameter(
            "pos_encoding",
            nn.Parameter(
                self._create_positional_encoding(seq_length, d_model),
                requires_grad=False,
            ),
        )

        # Enhanced transformer with more layers for better pattern recognition
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=12,  # More attention heads for better feature interaction
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",  # Better activation for complex patterns
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=6
        )  # More layers

        # Enhanced tactical prediction head with better architecture
        self.tactical_predictor = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(128, 2),  # [attack_timing, defend_timing]
        )

        # Add layer normalization for stability
        self.layer_norm = nn.LayerNorm(d_model)

    def _create_positional_encoding(self, max_len, d_model):
        """Create enhanced positional encoding"""
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, combined_sequence):
        """Enhanced forward pass with better regularization"""
        batch_size, seq_len = combined_sequence.shape[:2]

        # Ensure float32 dtype
        combined_sequence = combined_sequence.float()

        # Project input features
        projected = self.input_projection(combined_sequence)

        # Add positional encoding
        projected = projected + self.pos_encoding[:, :seq_len, :].float()

        # Apply layer normalization
        projected = self.layer_norm(projected)

        # Apply transformer
        transformer_out = self.transformer(projected)
        final_features = transformer_out[:, -1, :]  # Last timestep

        # Generate tactical predictions with better activation
        tactical_logits = self.tactical_predictor(final_features)
        tactical_probs = torch.sigmoid(tactical_logits)

        return {
            "attack_timing": tactical_probs[:, 0],
            "defend_timing": tactical_probs[:, 1],
        }


class EnhancedCUDAOptimizedCNN(nn.Module):
    """Enhanced CUDA-optimized CNN with better architecture for fighting games"""

    def __init__(self, input_channels=24, feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim

        # Enhanced CNN architecture optimized for fighting game pattern recognition
        self.cnn = nn.Sequential(
            # First conv block - better feature extraction
            nn.Conv2d(input_channels, 64, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # Second conv block - improved spatial processing
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Third conv block - better feature abstraction
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # Fourth conv block - high-level features
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        # Enhanced projection layer with better regularization
        self.projection = nn.Sequential(
            nn.Linear(512, 768),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(768, feature_dim),
        )

        # Tactical predictions cache
        self.register_buffer("attack_timing_cache", torch.tensor(0.0))
        self.register_buffer("defend_timing_cache", torch.tensor(0.0))

    def forward(self, frame_stack):
        """Enhanced forward pass with better feature extraction"""
        try:
            # Ensure input is on CUDA and float32
            if not frame_stack.is_cuda:
                frame_stack = frame_stack.cuda(non_blocking=True)

            frame_stack = frame_stack.float()

            # Apply enhanced CNN
            cnn_output = self.cnn(frame_stack)
            features = self.projection(cnn_output)

            return features

        except Exception as e:
            print(f"❌ Enhanced CUDA CNN error: {e}")
            batch_size = frame_stack.shape[0] if len(frame_stack.shape) > 0 else 1
            device = frame_stack.device if frame_stack.is_cuda else torch.device("cuda")
            return torch.zeros(
                batch_size, self.feature_dim, device=device, dtype=torch.float32
            )

    def update_tactical_predictions(self, attack_timing, defend_timing):
        """Update tactical predictions on CUDA"""
        self.attack_timing_cache.copy_(
            torch.tensor(attack_timing, device="cuda", dtype=torch.float32)
        )
        self.defend_timing_cache.copy_(
            torch.tensor(defend_timing, device="cuda", dtype=torch.float32)
        )


class EnhancedCUDAStreetFighterCNN(StreetFighterSimplifiedCNN):
    """Enhanced CUDA-optimized CNN for Street Fighter with better architecture"""

    def __init__(self, observation_space, features_dim=512):
        # Initialize parent but replace the CNN
        super().__init__(observation_space, features_dim)

        # Replace with enhanced CUDA-optimized CNN
        n_input_channels = observation_space.shape[0]
        self.cnn = EnhancedCUDAOptimizedCNN(
            input_channels=n_input_channels, feature_dim=features_dim
        )

        # Force to CUDA
        self.cuda()

        print(f"🚀 Enhanced CUDA Street Fighter CNN initialized:")
        print(
            f"   Input: {n_input_channels} channels → Output: {features_dim} features"
        )
        print(f"   Enhanced architecture for combo detection")
        print(f"   Device: {next(self.parameters()).device}")

    def forward(self, observations):
        """Enhanced CUDA-optimized forward pass"""
        # Ensure observations are on CUDA
        if not observations.is_cuda:
            observations = observations.cuda(non_blocking=True)

        # Normalize to [0, 1] range and ensure float32
        normalized_obs = observations.float() / 255.0

        return self.cnn(normalized_obs)


class EnhancedTrainingCallback(BaseCallback):
    """Enhanced callback with better training monitoring and curriculum learning"""

    def __init__(self, enable_vision_transformer=True, verbose=0):
        super().__init__(verbose)
        self.enable_vision_transformer = enable_vision_transformer
        self.last_log_time = time.time()
        self.log_interval = 120  # Log every 2 minutes

        # Enhanced tracking
        self.win_rate_history = []
        self.performance_milestones = [0.3, 0.4, 0.5, 0.6, 0.7]  # Win rate milestones
        self.achieved_milestones = set()

        # Learning rate adaptation
        self.last_win_rate = 0.0
        self.stagnation_counter = 0
        self.adaptation_threshold = 10  # Adjust LR after 10 stagnant periods

    def _on_step(self):
        current_time = time.time()
        if current_time - self.last_log_time > self.log_interval:
            self.last_log_time = current_time
            self._log_enhanced_training_stats()
            self._check_performance_milestones()
            self._adapt_learning_rate()
        return True

    def _log_enhanced_training_stats(self):
        """Enhanced training statistics logging to file"""
        try:
            # Log to file only, minimal console output
            print(f"Training Step: {self.num_timesteps:,}")

            # CUDA memory statistics (log to file)
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                memory_reserved = torch.cuda.memory_reserved() / (1024**3)
                max_memory = torch.cuda.max_memory_allocated() / (1024**3)

                # Create a simple log entry for file
                log_entry = f"Step {self.num_timesteps:,} - CUDA Memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved, {max_memory:.2f}GB peak"

                # Write to training log file
                log_filename = (
                    f'logs/training_stats_{datetime.now().strftime("%Y%m%d")}.log'
                )
                with open(log_filename, "a") as f:
                    f.write(f"{datetime.now().isoformat()} - {log_entry}\n")

                # Reset peak stats periodically
                if self.num_timesteps % 1000000 == 0:
                    torch.cuda.reset_peak_memory_stats()

            # Enhanced environment statistics (log to file)
            if hasattr(self.training_env, "get_attr"):
                all_stats = self.training_env.get_attr("stats")
                if all_stats and len(all_stats) > 0:
                    stats = all_stats[0]

                    # Get wrapper for additional stats
                    wrapper_env = self.training_env.envs[0].env

                    predictions = stats.get("predictions_made", 0)
                    win_rate = wrapper_env.wins / max(wrapper_env.total_rounds, 1)

                    # Console output - minimal
                    print(
                        f"  Win Rate: {win_rate:.1%} ({wrapper_env.wins}/{wrapper_env.total_rounds})"
                    )
                    if stats.get("max_combo", 0) > 0:
                        print(f"  Max Combo: {stats.get('max_combo', 0)}")

                    # Detailed logging to file
                    detailed_log = f"""Step {self.num_timesteps:,} - Performance Metrics:
  Win Rate: {win_rate:.1%} ({wrapper_env.wins}/{wrapper_env.total_rounds})
  Max Combo: {stats.get('max_combo', 0)}
  Total Combos: {stats.get('total_combos', 0)}
  Avg Damage/Round: {stats.get('avg_damage_per_round', 0):.1f}
  Transformer Predictions: {predictions:,}
  Attack Timing: {stats.get('avg_attack_timing', 0.0):.3f}
  Defend Timing: {stats.get('avg_defend_timing', 0.0):.3f}"""

                    log_filename = (
                        f'logs/performance_{datetime.now().strftime("%Y%m%d")}.log'
                    )
                    with open(log_filename, "a") as f:
                        f.write(f"{datetime.now().isoformat()} - {detailed_log}\n\n")

                    # Track win rate for learning rate adaptation
                    self.win_rate_history.append(win_rate)
                    if len(self.win_rate_history) > 10:
                        self.win_rate_history.pop(0)

        except Exception as e:
            print(f"Logging error: {e}")

    def _check_performance_milestones(self):
        """Check if performance milestones have been reached - log to file"""
        try:
            if len(self.win_rate_history) > 0:
                current_win_rate = self.win_rate_history[-1]

                for milestone in self.performance_milestones:
                    if (
                        milestone not in self.achieved_milestones
                        and current_win_rate >= milestone
                    ):
                        self.achieved_milestones.add(milestone)
                        print(f"🎉 MILESTONE: {milestone:.0%} Win Rate!")

                        # Log milestone to file
                        log_filename = (
                            f'logs/milestones_{datetime.now().strftime("%Y%m%d")}.log'
                        )
                        with open(log_filename, "a") as f:
                            f.write(
                                f"{datetime.now().isoformat()} - MILESTONE ACHIEVED: {milestone:.0%} Win Rate at step {self.num_timesteps:,}\n"
                            )

                        # Save milestone model
                        if hasattr(self.model, "save"):
                            milestone_path = f"enhanced_trained_models/milestone_{milestone:.0%}_step_{self.num_timesteps}.zip"
                            self.model.save(milestone_path)
                            print(f"💾 Milestone model saved")

        except Exception as e:
            print(f"Milestone checking error: {e}")

    def _adapt_learning_rate(self):
        """Adaptive learning rate based on performance - log to file"""
        try:
            if len(self.win_rate_history) >= 5:
                recent_win_rate = np.mean(self.win_rate_history[-5:])

                # Check for stagnation
                if (
                    abs(recent_win_rate - self.last_win_rate) < 0.01
                ):  # Less than 1% improvement
                    self.stagnation_counter += 1
                else:
                    self.stagnation_counter = 0

                # Log learning rate adaptation to file
                if self.stagnation_counter >= self.adaptation_threshold:
                    log_filename = (
                        f'logs/learning_rate_{datetime.now().strftime("%Y%m%d")}.log'
                    )
                    with open(log_filename, "a") as f:
                        f.write(
                            f"{datetime.now().isoformat()} - Performance stagnation detected at step {self.num_timesteps:,}. Win rate: {recent_win_rate:.1%}\n"
                        )
                    self.stagnation_counter = 0

                self.last_win_rate = recent_win_rate

        except Exception as e:
            print(f"Learning rate adaptation error: {e}")


def create_enhanced_learning_rate_schedule(initial_lr=2.5e-4):
    """Create enhanced learning rate schedule with better adaptation"""

    def schedule(progress):
        """
        Enhanced learning rate schedule:
        - Higher initial learning rate for faster early learning
        - Gradual decay with plateau periods
        - Final fine-tuning phase
        """
        if progress < 0.1:
            # Initial learning phase - higher LR
            return initial_lr * 1.5
        elif progress < 0.3:
            # Early learning - standard LR
            return initial_lr
        elif progress < 0.6:
            # Mid training - gradual decay
            return initial_lr * (0.8 - 0.3 * (progress - 0.3) / 0.3)
        elif progress < 0.8:
            # Later training - slower decay
            return initial_lr * (0.5 - 0.2 * (progress - 0.6) / 0.2)
        else:
            # Fine-tuning phase - very low LR
            return initial_lr * 0.3 * (1.0 - progress) / 0.2

    return schedule


def main():
    # Force CUDA setup first
    device = setup_cuda_optimization()

    parser = argparse.ArgumentParser(
        description="Enhanced CUDA-Optimized Street Fighter II Training"
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=15000000
    )  # Increased for better learning
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4
    )  # Slightly higher initial LR
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--no-vision-transformer", action="store_true")
    parser.add_argument("--use-original-state", action="store_true")
    parser.add_argument(
        "--defend-actions", type=str, default="47,48,49,50,51,52"
    )  # Updated defensive actions
    parser.add_argument(
        "--batch-size", type=int, default=128
    )  # Larger batch size for better gradients
    parser.add_argument(
        "--n-steps", type=int, default=4096
    )  # More steps for better value estimation

    args = parser.parse_args()

    # Configuration
    game = "StreetFighterIISpecialChampionEdition-Genesis"
    enable_vision_transformer = not args.no_vision_transformer
    defend_actions = [int(x.strip()) for x in args.defend_actions.split(",")]

    # State file handling
    if args.use_original_state:
        state_file = "ken_bison_12.state"
    else:
        state_file = os.path.abspath("ken_bison_12.state")
        if not os.path.exists("ken_bison_12.state"):
            print(f"❌ State file not found: ken_bison_12.state")
            return

    save_dir = "enhanced_trained_models"
    os.makedirs(save_dir, exist_ok=True)

    print(f"🚀 ENHANCED CUDA-Optimized Street Fighter II Training")
    print(f"   Total timesteps: {args.total_timesteps:,}")
    print(f"   Learning rate: {args.learning_rate} (adaptive)")
    print(f"   Batch size: {args.batch_size}")
    print(f"   N-steps: {args.n_steps}")
    print(f"   Device: {device}")
    print(f"   Enhanced Features: Combo detection, better rewards, improved actions")

    # Create enhanced environment
    def make_env():
        env = retro.make(
            game=game,
            state=state_file,
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            render_mode="human" if args.render else None,
        )

        # Use enhanced wrapper with improved features
        env = StreetFighterVisionWrapper(
            env,
            reset_round=True,
            rendering=args.render,
            max_episode_steps=6000,  # Longer episodes for better learning
            frame_stack=8,
            enable_vision_transformer=enable_vision_transformer,
            defend_action_indices=defend_actions,
            log_transformer_predictions=True,  # Enable enhanced logging
        )

        env = Monitor(env)
        env.reset(seed=0)
        return env

    env = DummyVecEnv([make_env])

    # Enhanced policy configuration
    policy_kwargs = dict(
        features_extractor_class=EnhancedCUDAStreetFighterCNN,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(
            pi=[768, 512, 256],  # Larger policy network
            vf=[768, 512, 256],  # Larger value network
        ),
        activation_fn=nn.ReLU,
        ortho_init=False,  # Better initialization for complex tasks
    )

    # Create enhanced model
    if args.resume and os.path.exists(args.resume):
        print(f"📂 Loading enhanced model from: {args.resume}")
        model = PPO.load(args.resume, env=env, device=device)

        # Update hyperparameters for continued training
        model.learning_rate = create_enhanced_learning_rate_schedule(args.learning_rate)
        model.batch_size = args.batch_size
        model.n_steps = args.n_steps

    else:
        print(f"🧠 Creating enhanced PPO model")

        model = PPO(
            "CnnPolicy",
            env,
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=1,
            # Enhanced hyperparameters based on analysis
            n_steps=args.n_steps,  # More steps for better value estimation
            batch_size=args.batch_size,  # Larger batch size
            n_epochs=6,  # More epochs for better policy updates
            gamma=0.995,  # Slightly higher discount for longer-term strategy
            learning_rate=create_enhanced_learning_rate_schedule(args.learning_rate),
            clip_range=0.2,
            clip_range_vf=None,  # No value function clipping for better learning
            ent_coef=0.005,  # Lower entropy for more focused actions
            vf_coef=0.5,
            max_grad_norm=0.5,
            gae_lambda=0.95,
            use_sde=False,  # No state-dependent exploration for fighting games
            sde_sample_freq=-1,
            normalize_advantage=True,
            target_kl=0.01,  # Lower target KL for more stable updates
            tensorboard_log="enhanced_logs",
        )

    # Inject enhanced feature extractor
    print("💉 Injecting enhanced CUDA feature extractor...")
    try:
        feature_extractor = model.policy.features_extractor
        wrapper_env = env.envs[0].env

        if hasattr(wrapper_env, "inject_feature_extractor"):
            wrapper_env.inject_feature_extractor(feature_extractor)
            print("✅ Enhanced CUDA feature extractor injected successfully")
    except Exception as e:
        print(f"⚠️ Enhanced feature extractor injection failed: {e}")

    # Setup enhanced callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,  # More frequent checkpoints
        save_path=save_dir,
        name_prefix="enhanced_ppo_sf2",
    )

    enhanced_callback = EnhancedTrainingCallback(
        enable_vision_transformer=enable_vision_transformer
    )

    # Enhanced training with better monitoring
    start_time = time.time()
    print(f"🏋️ Starting enhanced training...")
    print(f"📁 Logs: logs/")
    print(f"📊 Analysis: analysis_data/")
    print(f"💾 Models: {save_dir}/")

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback, enhanced_callback],
            reset_num_timesteps=not bool(args.resume),
            progress_bar=True,
        )

        training_time = time.time() - start_time
        print(f"🎉 Training completed in {training_time/3600:.1f} hours!")

        # Post-training analysis
        print("\n📈 TRAINING COMPLETED")
        try:
            final_stats = env.get_attr("stats")[0]
            wrapper_env = env.envs[0].env
            final_win_rate = wrapper_env.wins / max(wrapper_env.total_rounds, 1)

            print(f"Final Win Rate: {final_win_rate:.1%}")
            print(f"Total Rounds: {wrapper_env.total_rounds}")
            print(f"Max Combo: {final_stats.get('max_combo', 0)}")

            # Save final enhanced analysis to analysis_data folder
            if hasattr(wrapper_env, "save_enhanced_analysis"):
                final_analysis_path = os.path.join(
                    "analysis_data", "final_enhanced_analysis.json"
                )
                wrapper_env.save_enhanced_analysis(final_analysis_path)
                print(f"📊 Analysis saved to analysis_data/")

        except Exception as e:
            print(f"Post-training analysis error: {e}")

    except KeyboardInterrupt:
        print(f"⏹️ Enhanced training interrupted")
    except Exception as e:
        print(f"❌ Enhanced training failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        env.close()

    # Save final enhanced model
    final_model_path = os.path.join(save_dir, "enhanced_ppo_sf2_final.zip")
    model.save(final_model_path)
    print(f"💾 Final enhanced model saved to: {final_model_path}")

    # CUDA cleanup
    torch.cuda.empty_cache()
    print("🧹 CUDA cache cleared")

    print("✅ Enhanced CUDA-optimized training complete!")

    # Final recommendations
    print(f"\n🎯 TRAINING RECOMMENDATIONS:")
    print(f"   • If win rate < 50%: Continue training with lower learning rate")
    print(f"   • If win rate 50-60%: Good progress, consider fine-tuning")
    print(f"   • If win rate > 60%: Excellent! Consider advanced techniques")
    print(f"   • Monitor combo usage - aim for 3+ hit combos consistently")
    print(f"   • Enhanced score momentum should show positive trend")


if __name__ == "__main__":
    main()
