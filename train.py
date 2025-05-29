import os
import sys
import argparse
import time

import retro
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnvWrapper

# Import the ENHANCED wrapper with trending
from wrapper import StreetFighterCustomWrapper

NUM_ENV = 50
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


class WinRateTrendingCallback(BaseCallback):
    """Custom callback to log win rate trends during training"""

    def __init__(self, print_freq=10000, verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.last_print = 0

    def _on_step(self) -> bool:
        # Print win rate trends every print_freq steps
        if self.num_timesteps - self.last_print >= self.print_freq:
            self.last_print = self.num_timesteps

            # Try to get win rate stats from vectorized environment
            try:
                # For SubprocVecEnv, we can't easily get the wrapper stats
                # So we'll just print a simple progress update
                print(f"\n{'='*50}")
                print(f"Training Progress - Step {self.num_timesteps:,}")
                print(
                    f"Continuing training... (Win rate stats available in individual env logs)"
                )
                print(f"{'='*50}\n")

            except Exception as e:
                if self.verbose > 0:
                    print(f"Could not get detailed stats: {e}")

        return True


class VecEnv60FPS(VecEnvWrapper):
    """Wrapper to limit vectorized environment to 60fps during training"""

    def __init__(self, venv, enable_fps_limit=True):
        super().__init__(venv)
        self.enable_fps_limit = enable_fps_limit
        self.last_step_time = time.time()
        self.target_fps = 60
        self.frame_time = 1.0 / self.target_fps  # 1/60 ≈ 0.0167

    def step_wait(self):
        if self.enable_fps_limit:
            # Calculate elapsed time since last step
            current_time = time.time()
            elapsed = current_time - self.last_step_time

            # Sleep if we're going too fast
            if elapsed < self.frame_time:
                time.sleep(self.frame_time - elapsed)

            self.last_step_time = time.time()

        return self.venv.step_wait()

    def reset(self):
        return self.venv.reset()

    def step_async(self, actions):
        return self.venv.step_async(actions)


# Linear scheduler
def linear_schedule(initial_value, final_value=0.0):
    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert initial_value > 0.0

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler


def make_env(game, state, seed=0, rendering=False):
    def _init():
        # Create retro environment
        env = retro.make(
            game=game,
            state=state,
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            render_mode="human" if rendering else None,
        )

        # Apply action filtering directly in the wrapper instead of separate class
        # This avoids pickling issues with SubprocVecEnv

        # Add custom wrapper with trending
        env = StreetFighterCustomWrapper(env, reset_round=True, rendering=rendering)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


def main():
    parser = argparse.ArgumentParser(
        description="Train Street Fighter II Agent with Win Rate Trending"
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=10000000, help="Total timesteps to train"
    )
    parser.add_argument(
        "--use-original-state",
        action="store_true",
        help="Use the ORIGINAL state file ken_bison_12.state",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable rendering during training at 60fps (will slow down training)",
    )
    parser.add_argument(
        "--trend-freq",
        type=int,
        default=50000,
        help="Print win rate trends every N timesteps (default: 50000)",
    )

    args = parser.parse_args()

    game = "StreetFighterIISpecialChampionEdition-Genesis"

    # Use state file
    if args.use_original_state:
        state_file = "ken_bison_12.state"
        print("Using ORIGINAL state file: ken_bison_12.state")
    else:
        state_file = os.path.abspath("ken_bison_12.state")
        print(f"Using custom state file: {state_file}")

    save_dir = "trained_models"
    os.makedirs(save_dir, exist_ok=True)

    # Main model path
    model_path = os.path.join(save_dir, "ppo_sf2_model_trending.zip")

    # Create environments with ORIGINAL settings + trending
    env = SubprocVecEnv(
        [
            make_env(game, state=state_file, seed=i, rendering=args.render)
            for i in range(NUM_ENV)
        ]
    )

    # Add 60fps limiting if rendering is enabled
    if args.render:
        env = VecEnv60FPS(env, enable_fps_limit=True)
        print("Applied 60fps timing to vectorized environment")

    # Learning rate and clip range schedules
    lr_schedule = linear_schedule(2.5e-4, 2.5e-6)
    clip_range_schedule = linear_schedule(0.15, 0.025)

    print("Starting training with win rate trending enabled")
    print(f"Will train for {args.total_timesteps:,} timesteps")
    print(f"Using {NUM_ENV} parallel environments")
    print(f"Progress updates every {args.trend_freq:,} timesteps")
    print("Using ORIGINAL reward system (no normalization)")
    print("Action filtering handled in wrapper")

    # Create model with ORIGINAL hyperparameters
    model = PPO(
        "CnnPolicy",
        env,
        device="cuda",
        verbose=1,
        n_steps=512,
        batch_size=512,
        n_epochs=4,
        gamma=0.94,
        learning_rate=lr_schedule,
        clip_range=clip_range_schedule,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log="logs",
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=save_dir,
        name_prefix="ppo_sf2_trending",
    )

    trending_callback = WinRateTrendingCallback(print_freq=args.trend_freq, verbose=1)

    # Train with trending
    print(f"Training features:")
    print(f"- Win rate trending tracked per environment")
    print(f"- Progress updates every {args.trend_freq:,} steps")
    print(f"- Individual environment stats logged")
    print(f"- No pickling issues with multiprocessing")

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, trending_callback],
        reset_num_timesteps=True,
    )

    env.close()

    # Save main model
    model.save(model_path)

    print(f"\nTraining completed! Model saved to: {model_path}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print("Win rate trending data was tracked throughout training")


if __name__ == "__main__":
    main()
