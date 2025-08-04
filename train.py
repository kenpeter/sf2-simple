#!/usr/bin/env python3
"""
🥊 Street Fighter RL Training - Based on nicknochnack/StreetFighterRL
Simple PPO training using Stable Baselines3
"""

import os
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
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
            if self.resume_model_name:
                # Extract the previous number from resume model name and add current steps
                import re

                match = re.search(r"_(\d+)$", self.resume_model_name)
                if match:
                    previous_steps = int(match.group(1))
                    total_steps = previous_steps + self.n_calls
                    base_name = re.sub(r"_\d+$", "", self.resume_model_name)
                    model_path = os.path.join(
                        self.save_path, "{}_{}".format(base_name, total_steps)
                    )
                else:
                    model_path = os.path.join(
                        self.save_path,
                        "{}_{}".format(self.resume_model_name, self.n_calls),
                    )
            else:
                model_path = os.path.join(
                    self.save_path, "best_model_{}".format(self.n_calls)
                )
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


def make_env():
    """
    Create environment with frame stacking
    """
    # env street figher
    env = StreetFighter()
    # monitor
    env = Monitor(env)
    # vec env
    env = DummyVecEnv([lambda: env])
    # frame stack 4
    env = VecFrameStack(env, 1024, channels_order="last")
    # return env
    return env


def optimize_ppo(trial):
    """
    Optuna optimization function for PPO hyperparameters
    """
    return {
        "n_steps": trial.suggest_int("n_steps", 2048, 8192),
        "gamma": trial.suggest_float("gamma", 0.8, 0.9999, log=True),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
        "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 0.99),
    }


def train_model(args):
    """
    Train the PPO model
    """
    print("🚀 Starting PPO training...")

    # Create environment
    env = make_env()
    print(f"Environment created with observation space: {env.observation_space}")
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

    # Save final model
    final_model_path = os.path.join(args.save_dir, "final_model")
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")

    return model


def test_model(args):
    """
    Test a trained model
    """
    print("🧪 Testing trained model...")

    # Create environment
    env = make_env()

    # Load model
    model = PPO.load(args.model_path)
    print(f"Model loaded from: {args.model_path}")

    # Test for specified episodes
    for episode in range(args.test_episodes):
        obs = env.reset()
        total_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward

            if args.render:
                env.render()

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    env.close()


def optimize_hyperparameters(args):
    """
    Optimize hyperparameters using Optuna
    """
    if optuna is None:
        print("❌ Optuna not installed. Install with: pip install optuna")
        return None

    print("🔧 Starting hyperparameter optimization...")

    def objective(trial):
        # Get hyperparameters
        # auto param
        params = optimize_ppo(trial)

        # Create environment
        env = make_env()

        # Create model with trial parameters
        # pass dynamic param to ppo
        model = PPO("CnnPolicy", env, **params, verbose=0)

        # Train for a shorter period for optimization
        model.learn(total_timesteps=args.optim_timesteps)

        # Evaluate the model
        obs = env.reset()
        total_reward = 0
        for _ in range(100):  # Short evaluation
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            if done:
                obs = env.reset()

        env.close()
        return total_reward

    # Create study and optimize
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.n_trials)

    print("Best hyperparameters:", study.best_params)
    return study.best_params


def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser(description="Street Fighter RL Training")

    # Training mode
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test", "optimize"],
        help="Mode: train, test, or optimize",
    )

    # Training parameters
    parser.add_argument(
        "--total_timesteps", type=int, default=1000000, help="Total training timesteps"
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

    # Testing parameters
    parser.add_argument(
        "--model_path",
        type=str,
        default="train/final_model.zip",
        help="Path to trained model for testing",
    )
    parser.add_argument(
        "--test_episodes", type=int, default=5, help="Number of episodes to test"
    )
    parser.add_argument("--render", action="store_true", help="Render during testing")

    # Optimization parameters
    parser.add_argument(
        "--n_trials", type=int, default=10, help="Number of optimization trials"
    )
    parser.add_argument(
        "--optim_timesteps",
        type=int,
        default=50000,
        help="Timesteps per optimization trial",
    )

    # Resume training
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to model checkpoint to resume from",
    )

    args = parser.parse_args()

    print(f"Mode: {args.mode}")
    print(f"Configuration: {vars(args)}")

    if args.mode == "train":
        train_model(args)
    elif args.mode == "test":
        test_model(args)
    elif args.mode == "optimize":
        best_params = optimize_hyperparameters(args)
        print("Optimization complete. Best parameters:", best_params)

    print("🏁 Complete!")


if __name__ == "__main__":
    main()
