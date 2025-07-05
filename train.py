#!/usr/bin/env python3
"""
train_ultra_simple.py - Training with ultra-simplified architecture for guaranteed gradient flow
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
import gymnasium as gym

# Import the ultra-simplified components
from wrapper import (
    StreetFighterVisionWrapper,
    StreetFighterUltraSimpleCNN,
    monitor_gradients,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UltraSimpleCallback(BaseCallback):
    """Callback optimized for ultra-simplified architecture."""

    def __init__(
        self,
        save_freq=50000,  # Save more frequently
        save_path="./ultra_simple_models/",
        verbose=1,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        self.episode_rewards = []
        self.episode_lengths = []
        self.gradient_health_history = []

    def _on_step(self) -> bool:
        # Monitor gradients more frequently due to simplified architecture
        if self.num_timesteps % 2500 == 0:  # Every 2500 steps
            gradient_health = self._analyze_gradient_health()
            self.gradient_health_history.append(gradient_health)

            if self.num_timesteps % 5000 == 0:  # Full monitoring every 5000 steps
                monitor_gradients(self.model, self.num_timesteps)

        # Log performance stats
        if self.num_timesteps % 10000 == 0:
            self._log_performance_stats()

        # Save model more frequently
        if self.num_timesteps > 0 and self.num_timesteps % self.save_freq == 0:
            self._save_model()

        # Check for episode completion
        if hasattr(self.locals, "infos") and self.locals["infos"]:
            for info in self.locals["infos"]:
                if "episode" in info:
                    episode_info = info["episode"]
                    self.episode_rewards.append(episode_info["r"])
                    self.episode_lengths.append(episode_info["l"])

                    # Keep only last 100 episodes
                    if len(self.episode_rewards) > 100:
                        self.episode_rewards.pop(0)
                        self.episode_lengths.pop(0)

        return True

    def _analyze_gradient_health(self) -> dict:
        """Quick gradient health check."""
        if not hasattr(self.model, "policy"):
            return {}

        total_params = 0
        params_with_grads = 0
        total_grad_norm = 0.0

        for param in self.model.policy.parameters():
            total_params += param.numel()
            if param.grad is not None:
                params_with_grads += param.numel()
                total_grad_norm += param.grad.norm().item()

        return {
            "total_params": total_params,
            "params_with_grads": params_with_grads,
            "gradient_coverage": (params_with_grads / max(total_params, 1)) * 100,
            "avg_grad_norm": total_grad_norm / max(params_with_grads, 1),
        }

    def _log_performance_stats(self):
        """Enhanced performance logging."""
        print(
            f"\n--- 📊 Step {self.num_timesteps} - Ultra-Simple Architecture Performance ---"
        )

        # Gradient health
        if self.gradient_health_history:
            latest_health = self.gradient_health_history[-1]
            coverage = latest_health.get("gradient_coverage", 0)

            print(f"  🧠 Gradient Health:")
            print(f"     - Total parameters: {latest_health.get('total_params', 0):,}")
            print(f"     - Gradient coverage: {coverage:.1f}%")
            print(
                f"     - Avg gradient norm: {latest_health.get('avg_grad_norm', 0):.6f}"
            )

            if coverage > 95:
                print(f"     ✅ EXCELLENT: {coverage:.1f}% gradient coverage!")
            elif coverage > 80:
                print(f"     👍 GOOD: {coverage:.1f}% gradient coverage!")
            else:
                print(f"     ⚠️ WARNING: Only {coverage:.1f}% gradient coverage!")

        # Episode statistics
        if self.episode_rewards:
            recent_rewards = self.episode_rewards[-10:]
            avg_reward = np.mean(recent_rewards)
            max_reward = np.max(recent_rewards)
            min_reward = np.min(recent_rewards)

            print(f"  🎮 Recent Episodes:")
            print(f"     - Avg reward: {avg_reward:.2f}")
            print(f"     - Best reward: {max_reward:.2f}")
            print(f"     - Worst reward: {min_reward:.2f}")

        # Environment stats
        if hasattr(self.training_env, "stats"):
            stats = self.training_env.stats
            win_rate = stats.get("win_rate", 0.0)
            wins = stats.get("wins", 0)
            losses = stats.get("losses", 0)

            print(f"  🎯 Fight Stats: {win_rate:.1%} win rate ({wins}W/{losses}L)")

    def _save_model(self):
        """Save model with gradient health info."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Get stats
        gradient_health = "unknown"
        if self.gradient_health_history:
            latest = self.gradient_health_history[-1]
            coverage = latest.get("gradient_coverage", 0)
            gradient_health = f"grad_{coverage:.0f}pct"

        avg_reward = (
            np.mean(self.episode_rewards[-10:]) if self.episode_rewards else 0.0
        )

        model_name = (
            f"ultra_simple_{self.num_timesteps}_steps_"
            f"{gradient_health}_"
            f"reward_{avg_reward:.1f}_"
            f"{timestamp}"
        )

        model_path = os.path.join(self.save_path, f"{model_name}.zip")
        self.model.save(model_path)
        print(f"💾 Model saved: {model_path}")


def make_env(
    game="StreetFighterIISpecialChampionEdition-Genesis",
    state="ken_bison_12.state",
    render_mode=None,
):
    """Create environment for ultra-simple architecture."""
    try:
        env = retro.make(game=game, state=state, render_mode=render_mode)
        env = Monitor(env)
        env = StreetFighterVisionWrapper(
            env, frame_stack=8, rendering=(render_mode is not None)
        )
        return env
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Train with ultra-simplified architecture"
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=2000000, help="Total training timesteps"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2.5e-4, help="Learning rate"
    )
    parser.add_argument("--render", action="store_true", help="Render during training")
    parser.add_argument("--resume", type=str, default=None, help="Path to resume from")
    parser.add_argument(
        "--game", type=str, default="StreetFighterIISpecialChampionEdition-Genesis"
    )
    parser.add_argument("--state", type=str, default="ken_bison_12.state")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    print("🚀 Ultra-Simplified Architecture Training")
    print("=" * 60)
    print(f"   - Total timesteps: {args.total_timesteps:,}")
    print(f"   - Learning rate: {args.learning_rate}")
    print(f"   - Device: {args.device}")
    print(f"   - Rendering: {'Enabled' if args.render else 'Disabled'}")

    set_random_seed(42)

    # Create environment
    render_mode = "human" if args.render else None
    env = make_env(args.game, args.state, render_mode)

    # Test environment
    print("\n🧪 Testing environment...")
    try:
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        print("   ✅ Environment test passed")
    except Exception as e:
        print(f"   ❌ Environment test failed: {e}")
        raise

    # Configure ultra-simple policy
    policy_kwargs = dict(
        features_extractor_class=StreetFighterUltraSimpleCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[128, 64], vf=[128, 64]),  # Very simple networks
        activation_fn=torch.nn.ReLU,
        ortho_init=True,
    )

    # Create model
    if args.resume and os.path.exists(args.resume):
        print(f"📂 Resuming from {args.resume}")
        model = PPO.load(args.resume, env=env, device=args.device)
    else:
        print("🆕 Creating ultra-simplified model...")
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=1024,  # Smaller steps for faster iteration
            batch_size=32,  # Smaller batch size
            n_epochs=5,  # Fewer epochs
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            verbose=1,
            device=args.device,
        )

    # Verify model
    print("\n🏗️ Model Verification:")
    total_params = sum(p.numel() for p in model.policy.parameters())
    print(f"   - Total parameters: {total_params:,}")

    # Test gradient flow
    print("\n🔬 Testing gradient flow...")
    try:
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        # Convert numpy arrays to tensors for the policy
        obs_tensor = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                obs_tensor[key] = torch.from_numpy(value).unsqueeze(0).to(model.device)
            else:
                obs_tensor[key] = torch.tensor(value).unsqueeze(0).to(model.device)

        # Test forward pass
        with torch.no_grad():
            actions, values, log_probs = model.policy(obs_tensor)

        print(f"   ✅ Forward pass successful")

        # Test backward pass (create new tensors with gradients)
        obs_tensor_grad = {}
        for key, value in obs.items():
            if isinstance(value, np.ndarray):
                obs_tensor_grad[key] = (
                    torch.from_numpy(value).unsqueeze(0).float().to(model.device)
                )
            else:
                obs_tensor_grad[key] = (
                    torch.tensor(value).unsqueeze(0).float().to(model.device)
                )

        # Forward pass with gradients
        actions, values, log_probs = model.policy(obs_tensor_grad)

        # Create loss and backward pass
        loss = values.sum() + log_probs.sum()
        model.policy.zero_grad()
        loss.backward()

        grad_count = sum(1 for p in model.policy.parameters() if p.grad is not None)
        total_param_tensors = sum(1 for p in model.policy.parameters())
        coverage = (grad_count / total_param_tensors) * 100

        print(f"   ✅ Backward pass successful")
        print(
            f"   - Gradient coverage: {coverage:.1f}% ({grad_count}/{total_param_tensors})"
        )

        if coverage < 90:
            print(f"   ⚠️ WARNING: Only {coverage:.1f}% gradient coverage!")
        else:
            print(f"   ✅ EXCELLENT: {coverage:.1f}% gradient coverage!")

    except Exception as e:
        print(f"   ❌ Gradient flow test failed: {e}")
        import traceback

        traceback.print_exc()
        raise

    # Create callback
    callback = UltraSimpleCallback(
        save_freq=50000,
        save_path="./ultra_simple_models/",
    )

    print("\n🎯 Starting training with ultra-simplified architecture...")
    print("   - Optimized for fast iteration and guaranteed gradient flow")
    print("   - Frequent saves and monitoring")

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            reset_num_timesteps=not args.resume,
            progress_bar=True,
        )

        final_path = "./ultra_simple_models/final_ultra_simple.zip"
        model.save(final_path)
        print(f"🎉 Training completed! Final model: {final_path}")

    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted")
        interrupted_path = "./ultra_simple_models/interrupted_ultra_simple.zip"
        model.save(interrupted_path)
        print(f"💾 Model saved: {interrupted_path}")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        error_path = "./ultra_simple_models/error_ultra_simple.zip"
        model.save(error_path)
        print(f"💾 Model saved: {error_path}")
        raise

    finally:
        env.close()


if __name__ == "__main__":
    main()
