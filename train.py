#!/usr/bin/env python3
"""
🥊 Street Fighter RL Training - Based on nicknochnack/StreetFighterRL
Simple PPO training using Stable Baselines3
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from wrapper import StreetFighter
import argparse

# Create directories
os.makedirs("train", exist_ok=True)
os.makedirs("logs", exist_ok=True)

print("🥊 Street Fighter RL Training - Simple PPO Implementation")


class TrainAndLoggingCallback(BaseCallback):
    """
    Custom callback for training and logging
    """

    def __init__(self, check_freq, save_path, resume_model_name=None, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.resume_model_name = resume_model_name

        # Win rate tracking
        self.matches_played = 0
        self.matches_won = 0

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        # Check for match completion (episode done) and track wins
        if hasattr(self, "locals") and "infos" in self.locals:
            infos = self.locals["infos"]
            for info in infos:
                if info and "agent_won" in info:  # Episode finished with win/loss info
                    self.matches_played += 1
                    if info["agent_won"]:  # Agent won the match
                        self.matches_won += 1

        if self.n_calls % self.check_freq == 0:
            # Always overwrite the same checkpoint file
            model_path = os.path.join(self.save_path, "checkpoint")
            self.model.save(model_path)

            # Calculate and display win rate
            if self.matches_played > 0:
                win_rate = (self.matches_won / self.matches_played) * 100
                print(
                    f"Model saved at step {self.n_calls} | Win Rate: {win_rate:.1f}% ({self.matches_won}/{self.matches_played})"
                )
            else:
                print(f"Model saved at step {self.n_calls}")
        return True


def make_env(rank=0):
    """
    Create a single environment instance (for parallel training)
    """

    def _init():
        env = StreetFighter()
        env = Monitor(env)
        return env

    return _init


def make_vec_env(n_envs=4, frame_stack=1024, use_subprocess=True):
    """
    Create vectorized environment with multiple parallel environments
    Args:
        n_envs: Number of parallel environments (default: 4)
        frame_stack: Number of frames to stack (default: 1024)
        use_subprocess: Use SubprocVecEnv for better CPU parallelization (default: True)
    """
    # Create multiple environments in parallel
    # SubprocVecEnv runs each env in a separate process for true parallelization
    if use_subprocess and n_envs > 1:
        env = SubprocVecEnv([make_env(i) for i in range(n_envs)], start_method="fork")
    else:
        env = DummyVecEnv([make_env(i) for i in range(n_envs)])

    # Frame stack
    env = VecFrameStack(env, frame_stack, channels_order="last")
    return env


def train_model(args):
    """
    Train the PPO model
    """
    print("🚀 Starting PPO training...")

    # Create vectorized environment with multiple parallel environments
    env = make_vec_env(n_envs=args.n_envs, frame_stack=args.frame_stack)
    print(f"Created {args.n_envs} parallel environments")
    print(f"Environment observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Model parameters
    model_params = {
        "n_steps": args.n_steps,
        # what is this gamma?
        "gamma": args.gamma,
        "learning_rate": args.learning_rate,
        "clip_range": args.clip_range,
        # what is gae lambda?
        "gae_lambda": args.gae_lambda,
        "ent_coef": 0.01,  # Encourage exploration to prevent blocking
        "verbose": 1,
        "tensorboard_log": args.log_dir,
        "device": "cuda",
    }

    print(f"Model parameters: {model_params}")

    # Create model
    if args.resume:
        print(f"📂 Loading model from: {args.resume}")
        model = PPO.load(args.resume, env=env)
        print("✅ Model loaded successfully!")
    else:
        model = PPO("CnnPolicy", env, **model_params)

    # Create callback
    resume_model_name = None
    if args.resume:
        # Extract model name from resume path (e.g., "best_model_940000" from "train/best_model_940000.zip")
        resume_model_name = os.path.splitext(os.path.basename(args.resume))[0]

    # it is a call back func
    callback = TrainAndLoggingCallback(
        check_freq=args.save_freq,
        save_path=args.save_dir,
        resume_model_name=resume_model_name,
    )

    # Train the model
    print(f"Training for {args.total_timesteps:,} timesteps...")
    # model will use this callback for log
    # the func learn will call this callback inside
    model.learn(total_timesteps=args.total_timesteps, callback=callback)

    # Save final model (overwrite checkpoint)
    final_model_path = os.path.join(args.save_dir, "checkpoint")
    model.save(final_model_path)
    print(f"✅ Final model saved to: {final_model_path}")

    return model


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description="Street Fighter RL Training")

    # Training parameters
    parser.add_argument(
        "--episodes_per_env",
        type=int,
        default=1000,
        help="Training episodes per environment",
    )
    parser.add_argument(
        "--avg_episode_length",
        type=int,
        default=500,
        help="Average episode length (for timestep calculation)",
    )
    parser.add_argument(
        "--n_steps", type=int, default=2048, help="Number of steps per update"
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument(
        "--learning_rate", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument(
        "--clip_range", type=float, default=0.2, help="Clip range for PPO"
    )
    parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE lambda")

    # Environment parameters
    parser.add_argument(
        "--n_envs",
        type=int,
        default=64,
        help="Number of parallel environments using SubprocVecEnv",
    )
    parser.add_argument(
        "--frame_stack",
        type=int,
        default=4,
        help="Number of frames to stack (4 is standard for Atari)",
    )

    # Callback parameters
    parser.add_argument(
        "--save_freq", type=int, default=10000, help="Save model every N steps"
    )
    parser.add_argument(
        "--save_dir", type=str, default="train", help="Directory to save models"
    )
    parser.add_argument(
        "--log_dir", type=str, default="logs", help="Directory for tensorboard logs"
    )

    # Resume training
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to model checkpoint to resume from",
    )

    args = parser.parse_args()

    # Calculate total episodes and timesteps
    args.total_episodes = args.episodes_per_env * args.n_envs
    args.total_timesteps = args.total_episodes * args.avg_episode_length

    print(f"Configuration: {vars(args)}")

    print(f"\n📊 Training Statistics:")
    print(f"   Episodes per environment: {args.episodes_per_env:,}")
    print(f"   Parallel environments: {args.n_envs} (SubprocVecEnv)")
    print(f"   Total episodes (all envs): {args.total_episodes:,}")
    print(f"   Average episode length: {args.avg_episode_length} steps")
    print(f"   Total timesteps: {args.total_timesteps:,}")
    print(f"   Frame stack: {args.frame_stack}")
    print(f"\n💡 Using SubprocVecEnv: Each environment runs in a separate process")
    print(f"   {args.n_envs} parallel Genesis emulators for maximum CPU utilization\n")

    train_model(args)

    print("🏁 Complete!")


if __name__ == "__main__":
    main()
