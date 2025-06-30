#!/usr/bin/env python3
"""
CUDA-Optimized Training Script for Street Fighter II with Enhanced Vision Pipeline
Ensures all operations run on GPU for maximum performance
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


# Force CUDA usage and optimize
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


class CUDAOptimizedVisionTransformer(nn.Module):
    """CUDA-optimized Vision Transformer with consistent dtype handling"""

    def __init__(self, visual_dim=512, opencv_dim=2, momentum_dim=18, seq_length=8):
        super().__init__()
        self.seq_length = seq_length

        # Combined input dimension
        combined_dim = visual_dim + opencv_dim + momentum_dim

        # Transformer dimension
        d_model = 256
        self.input_projection = nn.Linear(combined_dim, d_model)

        # Positional encoding as parameter for CUDA efficiency
        self.register_parameter(
            "pos_encoding",
            nn.Parameter(
                self._create_positional_encoding(seq_length, d_model),
                requires_grad=False,
            ),
        )

        # Standard transformer (avoid potential mixed precision issues)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
            activation="relu",  # More stable than GELU for dtype consistency
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)

        # Tactical prediction heads with consistent dtypes
        self.tactical_predictor = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 4),
        )

        self.position_predictor = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3),
        )

    def _create_positional_encoding(self, max_len, d_model):
        """Create positional encoding optimized for CUDA"""
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # [1, seq_len, d_model]

    def forward(self, combined_sequence):
        """Forward pass with CUDA optimizations and dtype consistency"""
        batch_size, seq_len = combined_sequence.shape[:2]

        # Ensure float32 dtype
        combined_sequence = combined_sequence.float()

        # Project input features
        projected = self.input_projection(combined_sequence)

        # Add positional encoding (CUDA-optimized)
        projected = projected + self.pos_encoding[:, :seq_len, :].float()

        # Apply transformer
        transformer_out = self.transformer(projected)
        final_features = transformer_out[:, -1, :]  # Last timestep

        # Generate predictions
        tactical_logits = self.tactical_predictor(final_features)
        tactical_probs = torch.sigmoid(tactical_logits)

        position_logits = self.position_predictor(final_features)
        position_probs = torch.softmax(position_logits, dim=-1)

        return {
            "attack_timing": tactical_probs[:, 0],
            "defend_timing": tactical_probs[:, 1],
            "aggression_level": tactical_probs[:, 2],
            "positioning_score": tactical_probs[:, 3],
            "should_advance": position_probs[:, 0],
            "should_retreat": position_probs[:, 1],
            "should_maintain": position_probs[:, 2],
        }


class CUDAOptimizedCNN(nn.Module):
    """CUDA-optimized CNN with consistent dtype handling"""

    def __init__(self, input_channels=24, feature_dim=512):
        super().__init__()
        self.feature_dim = feature_dim

        # Optimized CNN architecture - all float32 to avoid dtype issues
        self.cnn = nn.Sequential(
            # First conv block
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm2d(32),  # Back to BatchNorm for stability
            nn.ReLU(),
            # Second conv block
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        # Projection layer
        self.projection = nn.Linear(256, feature_dim)

        # Tactical predictions cache
        self.register_buffer("attack_timing_cache", torch.tensor(0.0))
        self.register_buffer("defend_timing_cache", torch.tensor(0.0))

    def forward(self, frame_stack):
        """Forward pass with CUDA optimizations and consistent dtypes"""
        try:
            # Ensure input is on CUDA and float32
            if not frame_stack.is_cuda:
                frame_stack = frame_stack.cuda(non_blocking=True)

            # Ensure float32 dtype
            frame_stack = frame_stack.float()

            # Apply CNN
            cnn_output = self.cnn(frame_stack)
            features = self.projection(cnn_output)

            return features

        except Exception as e:
            print(f"❌ CUDA CNN error: {e}")
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


class CUDAStreetFighterCNN(StreetFighterEnhancedCNN):
    """CUDA-optimized version of Street Fighter CNN for stable-baselines3"""

    def __init__(self, observation_space, features_dim=512):
        # Initialize parent but replace the CNN
        super().__init__(observation_space, features_dim)

        # Replace with CUDA-optimized CNN
        n_input_channels = observation_space.shape[0]
        self.cnn = CUDAOptimizedCNN(
            input_channels=n_input_channels, feature_dim=features_dim
        )

        # Force to CUDA
        self.cuda()

        print(f"🚀 CUDA-Optimized Street Fighter CNN initialized:")
        print(
            f"   Input: {n_input_channels} channels → Output: {features_dim} features"
        )
        print(f"   Device: {next(self.parameters()).device}")

    def forward(self, observations):
        """CUDA-optimized forward pass with proper dtype handling"""
        # Ensure observations are on CUDA
        if not observations.is_cuda:
            observations = observations.cuda(non_blocking=True)

        # Normalize to [0, 1] range and ensure float32
        normalized_obs = observations.float() / 255.0

        # Forward pass without autocast to avoid dtype issues
        return self.cnn(normalized_obs)


class CUDAOptimizedVisionPipelineCallback(BaseCallback):
    """CUDA-optimized callback with GPU memory monitoring"""

    def __init__(self, enable_vision_transformer=True, verbose=0):
        super().__init__(verbose)
        self.enable_vision_transformer = enable_vision_transformer
        self.last_log_time = time.time()
        self.log_interval = 120

    def _on_step(self):
        current_time = time.time()
        if current_time - self.last_log_time > self.log_interval:
            self.last_log_time = current_time
            self._log_cuda_training_stats()
        return True

    def _log_cuda_training_stats(self):
        """Log CUDA-optimized training statistics"""
        try:
            print(
                f"\n--- 🚀 CUDA Street Fighter Vision Pipeline @ Step {self.num_timesteps:,} ---"
            )

            # CUDA memory statistics
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / (1024**3)
                memory_reserved = torch.cuda.memory_reserved() / (1024**3)
                max_memory = torch.cuda.max_memory_allocated() / (1024**3)

                print(f"   💾 CUDA Memory:")
                print(f"      Allocated: {memory_allocated:.2f}GB")
                print(f"      Reserved: {memory_reserved:.2f}GB")
                print(f"      Peak: {max_memory:.2f}GB")

                # Reset peak stats periodically
                if self.num_timesteps % 1000000 == 0:
                    torch.cuda.reset_peak_memory_stats()

            # Environment statistics
            if hasattr(self.training_env, "get_attr"):
                all_stats = self.training_env.get_attr("stats")
                if all_stats and len(all_stats) > 0:
                    stats = all_stats[0]

                    motion_count = stats.get("motion_detected_count", 0)
                    predictions = stats.get("predictions_made", 0)

                    print(f"   🔍 Motion Detections: {motion_count:,}")

                    if self.enable_vision_transformer:
                        vt_ready = stats.get("vision_transformer_ready", False)
                        if vt_ready:
                            print(
                                f"   🧠 CUDA Vision Transformer: Active ({predictions:,} predictions)"
                            )

                            # Tactical statistics
                            avg_attack = stats.get("avg_attack_timing", 0.0)
                            avg_defend = stats.get("avg_defend_timing", 0.0)
                            avg_aggression = stats.get("avg_aggression_level", 0.0)
                            avg_positioning = stats.get("avg_positioning_score", 0.0)

                            print(f"   ⚔️  Attack Timing: {avg_attack:.3f}")
                            print(f"   🛡️  Defend Timing: {avg_defend:.3f}")
                            print(f"   🔥 Aggression: {avg_aggression:.3f}")
                            print(f"   📍 Positioning: {avg_positioning:.3f}")

            # Learning rate
            if hasattr(self.model, "learning_rate"):
                lr = self.model.learning_rate
                if callable(lr):
                    progress = getattr(self.model, "_current_progress_remaining", 1.0)
                    lr = lr(progress)
                print(f"   📈 Learning Rate: {lr:.2e}")

            print("--------------------------------------------------")

        except Exception as e:
            print(f"   ⚠️ CUDA logging error: {e}")


def main():
    # Force CUDA setup first
    device = setup_cuda_optimization()

    parser = argparse.ArgumentParser(
        description="CUDA-Optimized Street Fighter II Enhanced Vision Training"
    )
    parser.add_argument("--total-timesteps", type=int, default=10000000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2.5e-4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--no-vision-transformer", action="store_true")
    parser.add_argument("--use-original-state", action="store_true")
    parser.add_argument("--defend-actions", type=str, default="4,5,6")
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable mixed precision training (experimental)",
    )

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

    save_dir = "trained_models_cuda_enhanced"
    os.makedirs(save_dir, exist_ok=True)

    print(f"🚀 CUDA-Optimized Street Fighter II Training")
    print(f"   Total timesteps: {args.total_timesteps:,}")
    print(f"   Learning rate: {args.learning_rate}")
    print(f"   Mixed precision: {args.mixed_precision}")
    print(f"   Device: {device}")

    # Create environment with enhanced wrapper
    def make_env():
        env = retro.make(
            game=game,
            state=state_file,
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            render_mode="human" if args.render else None,
        )

        env = StreetFighterVisionWrapper(
            env,
            reset_round=True,
            rendering=args.render,
            max_episode_steps=5000,
            frame_stack=8,
            enable_vision_transformer=enable_vision_transformer,
            defend_action_indices=defend_actions,
        )

        env = Monitor(env)
        env.reset(seed=0)
        return env

    env = DummyVecEnv([make_env])

    # CUDA-optimized policy configuration
    policy_kwargs = dict(
        features_extractor_class=CUDAStreetFighterCNN,
        features_extractor_kwargs=dict(features_dim=512),
        net_arch=dict(
            pi=[512, 256, 128],
            vf=[512, 256, 128],
        ),
        activation_fn=nn.ReLU,  # More stable than GELU for dtype consistency
    )

    # Create model with CUDA optimizations
    if args.resume and os.path.exists(args.resume):
        print(f"📂 Loading CUDA model from: {args.resume}")
        model = PPO.load(args.resume, env=env, device=device)
    else:
        print(f"🧠 Creating CUDA-optimized PPO model")

        # Learning rate schedule
        def lr_schedule(progress):
            return args.learning_rate * (1.0 - 0.9 * progress)

        model = PPO(
            "CnnPolicy",
            env,
            policy_kwargs=policy_kwargs,
            device=device,
            verbose=1,
            n_steps=2048,
            batch_size=64,
            n_epochs=4,
            gamma=0.99,
            learning_rate=lr_schedule,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            gae_lambda=0.95,
            tensorboard_log="logs_cuda_enhanced",
        )

    # Inject CUDA-optimized feature extractor
    print("💉 Injecting CUDA feature extractor...")
    try:
        feature_extractor = model.policy.features_extractor
        wrapper_env = env.envs[0].env

        if hasattr(wrapper_env, "inject_feature_extractor"):
            wrapper_env.inject_feature_extractor(feature_extractor)
            print("✅ CUDA feature extractor injected successfully")
    except Exception as e:
        print(f"⚠️ Feature extractor injection failed: {e}")

    # Setup callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=save_dir,
        name_prefix="ppo_sf2_cuda_enhanced",
    )

    cuda_callback = CUDAOptimizedVisionPipelineCallback(
        enable_vision_transformer=enable_vision_transformer
    )

    # Training with CUDA optimizations
    start_time = time.time()
    print(f"🏋️ Starting CUDA-optimized training...")
    print(f"💡 Note: Mixed precision disabled to avoid dtype issues")

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback, cuda_callback],
            reset_num_timesteps=not bool(args.resume),
        )

        training_time = time.time() - start_time
        print(f"🎉 CUDA training completed in {training_time/3600:.1f} hours!")

    except KeyboardInterrupt:
        print(f"⏹️ Training interrupted")
    except Exception as e:
        print(f"❌ Training failed: {e}")
    finally:
        env.close()

    # Save final model
    final_model_path = os.path.join(save_dir, "ppo_sf2_cuda_enhanced_final.zip")
    model.save(final_model_path)
    print(f"💾 Final CUDA model saved to: {final_model_path}")

    # CUDA memory cleanup
    torch.cuda.empty_cache()
    print("🧹 CUDA cache cleared")

    print("✅ CUDA-optimized training complete!")


if __name__ == "__main__":
    main()
