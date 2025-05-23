import os
import sys
import json
import argparse

import retro
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage

from wrapper import StreetFighterCustomWrapper

NUM_ENV = 16
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


def save_progress(total_timesteps, save_dir):
    """Save current total timesteps"""
    progress_file = os.path.join(save_dir, "progress.json")
    with open(progress_file, "w") as f:
        json.dump({"total_timesteps": total_timesteps}, f)


def load_progress(save_dir):
    """Load previous total timesteps"""
    progress_file = os.path.join(save_dir, "progress.json")
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            return json.load(f).get("total_timesteps", 0)
    return 0


# Linear scheduler
def linear_schedule(initial_value, final_value=0.0):
    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)
        assert initial_value > 0.0

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler


class FilteredActionWrapper(gym.Wrapper):
    """Wrapper to filter out START and MODE buttons to prevent cheating"""

    def __init__(self, env):
        super(FilteredActionWrapper, self).__init__(env)
        # Street Fighter II MultiBinary actions:
        # [B, Y, SELECT, START, UP, DOWN, LEFT, RIGHT, A, X, L, R]
        # We want to disable START (index 3) and SELECT (index 2)
        # SELECT is often used as MODE button in some versions
        self.disabled_buttons = [2, 3]  # SELECT and START

    def step(self, action):
        # Filter out disabled buttons by setting them to 0
        if hasattr(action, "__len__") and len(action) >= 4:
            filtered_action = action.copy() if hasattr(action, "copy") else list(action)
            for button_idx in self.disabled_buttons:
                if button_idx < len(filtered_action):
                    filtered_action[button_idx] = 0
            return self.env.step(filtered_action)
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


def make_env(game, state, seed=0):
    def _init():
        if state and os.path.exists(state):
            state_path = os.path.abspath(state)
        else:
            state_path = state

        env = retro.make(
            game=game,
            state=state_path,
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
        )

        # Add action filtering to prevent cheating
        env = FilteredActionWrapper(env)

        # Add custom wrapper (reset_round=True to ensure proper game flow)
        env = StreetFighterCustomWrapper(env, reset_round=True, rendering=False)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


def main():
    parser = argparse.ArgumentParser(description="Train Street Fighter II Agent")
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from saved model"
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=1000000, help="Total timesteps to train"
    )

    args = parser.parse_args()

    game = "StreetFighterIISpecialChampionEdition-Genesis"
    state_file = os.path.abspath("ken_bison_12.state")
    save_dir = "trained_models"
    os.makedirs(save_dir, exist_ok=True)

    # Main model path (always the same)
    model_path = os.path.join(save_dir, "ppo_sf2_model.zip")

    # Create environments with VecTransposeImage for proper image handling
    env = SubprocVecEnv(
        [make_env(game, state=state_file, seed=i) for i in range(NUM_ENV)]
    )
    env = VecTransposeImage(env)  # Add this for proper image preprocessing

    # Learning rate and clip range schedules
    lr_schedule = linear_schedule(2.5e-4, 2.5e-6)
    clip_range_schedule = linear_schedule(0.15, 0.025)

    # Load previous progress
    completed_timesteps = 0
    if args.resume and os.path.exists(model_path):
        completed_timesteps = load_progress(save_dir)
        print(f"Resuming from {completed_timesteps:,} timesteps")
    else:
        print("Starting new training")

    # Calculate remaining timesteps
    remaining_timesteps = max(0, args.total_timesteps - completed_timesteps)

    if remaining_timesteps == 0:
        print(f"Training completed! Total: {completed_timesteps:,} timesteps")
        return

    print(f"Will train for {remaining_timesteps:,} more timesteps")
    print(
        "Action filtering enabled - START and SELECT buttons disabled to prevent cheating"
    )

    # Create or load model
    if args.resume and os.path.exists(model_path):
        custom_objects = {
            "learning_rate": lr_schedule,
            "clip_range": clip_range_schedule,
            "n_steps": 512,
        }
        model = PPO.load(
            model_path, env=env, device="cuda", custom_objects=custom_objects
        )
    else:
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
            tensorboard_log="logs",
        )

    # Checkpoint callback (separate from main model)
    checkpoint_callback = CheckpointCallback(
        save_freq=31250,  # checkpoint_interval * num_envs = total_steps_per_checkpoint
        save_path=save_dir,
        name_prefix="ppo_ryu",
    )

    # Train
    original_stdout = sys.stdout
    log_file_path = os.path.join(save_dir, "training_log.txt")
    with open(log_file_path, "a" if args.resume else "w") as log_file:
        sys.stdout = log_file

        model.learn(
            total_timesteps=remaining_timesteps,
            callback=[checkpoint_callback],
            reset_num_timesteps=False,  # Keep timestep counter
        )

        env.close()

    sys.stdout = original_stdout

    # Save main model and progress
    model.save(model_path)
    save_progress(completed_timesteps + remaining_timesteps, save_dir)

    print(f"Training completed! Model saved to: {model_path}")
    print(f"Total timesteps: {completed_timesteps + remaining_timesteps:,}")


if __name__ == "__main__":
    main()
