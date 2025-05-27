import os
import sys
import argparse
import time

import retro
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnvWrapper

# Import the FIXED wrapper
from wrapper import StreetFighterCustomWrapper

NUM_ENV = 64
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


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


class FilteredActionWrapper(gym.Wrapper):
    """Wrapper to filter out START and MODE buttons to prevent cheating"""

    def __init__(self, env):
        super(FilteredActionWrapper, self).__init__(env)
        # Street Fighter II MultiBinary actions:
        # [B, Y, SELECT, START, UP, DOWN, LEFT, RIGHT, A, X, L, R]
        # We want to disable START (index 3) and SELECT (index 2)
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

        # Add action filtering to prevent cheating
        env = FilteredActionWrapper(env)

        # Add custom wrapper (FPS limiting handled at vec env level)
        env = StreetFighterCustomWrapper(env, reset_round=True, rendering=rendering)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


def main():
    parser = argparse.ArgumentParser(
        description="Train Street Fighter II Agent (ORIGINAL)"
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=10000000, help="Total timesteps to train"
    )
    parser.add_argument(
        "--use-original-state",
        action="store_true",
        help="Use the ORIGINAL state file ken_bison_12.state instead of ken_bison_12.state",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable rendering during training at 60fps (will slow down training)",
    )

    args = parser.parse_args()

    game = "StreetFighterIISpecialChampionEdition-Genesis"

    # CRITICAL: Use the ORIGINAL state file for better performance
    if args.use_original_state:
        state_file = "ken_bison_12.state"  # ORIGINAL state
        print("Using ORIGINAL state file: ken_bison_12.state")
    else:
        state_file = os.path.abspath("ken_bison_12.state")
        print(f"Using custom state file: {state_file}")

    save_dir = "trained_models"
    os.makedirs(save_dir, exist_ok=True)

    # Main model path
    model_path = os.path.join(save_dir, "ppo_sf2_model_original.zip")

    # Create environments with ORIGINAL settings
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

    # IMPORTANT: Remove VecTransposeImage if it's causing issues
    # The original doesn't seem to emphasize this
    # env = VecTransposeImage(env)

    # ORIGINAL Learning rate and clip range schedules
    lr_schedule = linear_schedule(2.5e-4, 2.5e-6)  # Same as original
    clip_range_schedule = linear_schedule(0.15, 0.025)

    print("Starting new training with ORIGINAL parameters")
    print(f"Will train for {args.total_timesteps:,} timesteps")
    print(f"Using {NUM_ENV} parallel environments")
    print("Using ORIGINAL reward system (no normalization)")
    print("Action filtering enabled - START and SELECT buttons disabled")
    print(
        f"Rendering: {'Enabled at 60fps' if args.render else 'Disabled (faster training)'}"
    )
    if args.render:
        print("Training will maintain 60fps timing for consistent experience")

    # Create model with ORIGINAL hyperparameters
    model = PPO(
        "CnnPolicy",
        env,
        device="cuda",
        verbose=1,
        n_steps=512,  # Same as original
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

    # Checkpoint callback - adjusted for 64 environments
    checkpoint_callback = CheckpointCallback(
        save_freq=50000,  # 781 * 64 ≈ 50000 (checkpoint every ~781 training steps)
        save_path=save_dir,
        name_prefix="ppo_sf2_original",
    )

    # Train with proper logging
    original_stdout = sys.stdout
    log_file_path = os.path.join(save_dir, "training_log_original.txt")

    print(f"Starting training with ORIGINAL reward system:")
    print(f"- Reward coefficient: 3 (integer)")
    print(f"- Full HP: 176")
    print(f"- NO normalization factor")
    print(f"- Health conditions: < 0 (not <= 0)")
    print(f"- Environments: {NUM_ENV} parallel")
    print(
        f"- Training speed: {'60 FPS (consistent with eval)' if args.render else 'Maximum (no rendering)'}"
    )

    with open(log_file_path, "w") as log_file:
        sys.stdout = log_file

        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback],
            reset_num_timesteps=True,
        )

        env.close()

    sys.stdout = original_stdout

    # Save main model
    model.save(model_path)

    print(f"Training completed! Model saved to: {model_path}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print("Used ORIGINAL implementation parameters")


if __name__ == "__main__":
    main()
