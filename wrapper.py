#!/usr/bin/env python3
"""
Complete Street Fighter II RL Training Script with All Fixes Applied
Combines wrapper + training + debugging in a single file

Major fixes:
- Fixed frame stacking logic (was critically broken)
- Better health extraction from multiple sources
- Longer episodes (5000 steps)
- Improved reward structure
- Better hyperparameters
- Comprehensive debugging

Usage:
python sf2_trainer_fixed.py --total-timesteps 10000000 --debug
"""

import os
import sys
import argparse
import time
import collections
import numpy as np

import retro
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnvWrapper


class StreetFighterCustomWrapper(gym.Wrapper):
    """Fixed Street Fighter wrapper with proper frame stacking and health extraction"""

    def __init__(self, env, reset_round=True, rendering=False, max_episode_steps=5000):
        super(StreetFighterCustomWrapper, self).__init__(env)
        self.env = env

        # Action filtering to prevent cheating
        self.disabled_buttons = [2, 3]  # SELECT and START

        # Frame stacking - FIXED configuration
        self.num_frames = 4  # Reduced from 9 for stability
        self.frame_stack = collections.deque(maxlen=self.num_frames)

        # Health tracking
        self.full_hp = 176
        self.prev_player_health = self.full_hp
        self.prev_opponent_health = self.full_hp

        # Win/Loss tracking
        self.wins = 0
        self.losses = 0
        self.total_rounds = 0
        self.current_round_active = True

        # Episode management - EXTENDED length
        self.max_episode_steps = max_episode_steps
        self.episode_steps = 0
        self.total_timesteps = 0

        # Win rate trending
        self.win_rate_history = []
        self.trend_window_size = 50
        self.rounds_per_trend_update = 5

        # FIXED observation space - now uses 4 frames * 3 channels = 12 channels
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(100, 128, 12), dtype=np.uint8
        )

        self.reset_round = reset_round
        self.rendering = rendering

        # Health extraction debugging
        self.debug_health = False
        self.health_extraction_attempts = 0
        self.successful_health_extractions = 0

        print(f"🚀 StreetFighterCustomWrapper initialized (FIXED VERSION):")
        print(f"   📊 Observation shape: {self.observation_space.shape}")
        print(f"   ⏱️  Max episode steps: {self.max_episode_steps}")
        print(f"   🎯 Frame stacking: {self.num_frames} frames")
        print(f"   🚫 Action filtering: SELECT/START disabled")
        print(f"   📈 Win rate tracking enabled")

    def _stack_observation(self):
        """FIXED frame stacking - properly concatenate recent frames"""
        if len(self.frame_stack) < self.num_frames:
            # Initialize with copies of the first frame if we don't have enough
            while len(self.frame_stack) < self.num_frames:
                if len(self.frame_stack) > 0:
                    self.frame_stack.append(self.frame_stack[-1].copy())
                else:
                    # Create a black frame as placeholder
                    self.frame_stack.append(np.zeros((100, 128, 3), dtype=np.uint8))

        # Concatenate the frames along the channel dimension
        # This creates shape (100, 128, 12) from 4 frames of (100, 128, 3)
        stacked = np.concatenate(list(self.frame_stack), axis=-1)
        return stacked

    def _extract_health_from_info(self, info):
        """Extract health and game state using the provided memory map"""
        self.health_extraction_attempts += 1

        player_health = self.full_hp
        opponent_health = self.full_hp
        extraction_successful = False

        # Use the specific keys from your memory map
        try:
            if "agent_hp" in info and info["agent_hp"] is not None:
                player_health = int(info["agent_hp"])
                extraction_successful = True

            if "enemy_hp" in info and info["enemy_hp"] is not None:
                opponent_health = int(info["enemy_hp"])
                extraction_successful = True

        except (ValueError, TypeError) as e:
            print(f"⚠️  Health extraction error: {e}")

        # Debug output for first few attempts
        if (
            self.health_extraction_attempts <= 5
            or self.health_extraction_attempts % 1000 == 0
        ):
            print(
                f"🔍 Health extraction #{self.health_extraction_attempts}: "
                f"Player={player_health}, Opponent={opponent_health}, "
                f"Success={extraction_successful}"
            )

            if not extraction_successful and self.health_extraction_attempts <= 3:
                print(f"   Available info keys: {list(info.keys())}")
                # Show specific keys we're looking for
                for key in [
                    "agent_hp",
                    "enemy_hp",
                    "agent_x",
                    "agent_y",
                    "enemy_x",
                    "enemy_y",
                ]:
                    if key in info:
                        print(f"   {key}: {info[key]}")

        if extraction_successful:
            self.successful_health_extractions += 1

        return player_health, opponent_health

    def _calculate_reward(self, curr_player_health, curr_opponent_health, info=None):
        """Simple damage-based reward: +1 damage dealt, -1 damage received"""
        custom_reward = 0.0
        custom_done = False

        # Episode length management
        if self.episode_steps >= self.max_episode_steps:
            custom_done = True
            return custom_reward, custom_done

        # Win/Loss tracking (for statistics only, no reward)
        round_ended = False
        if curr_player_health <= 0 or curr_opponent_health <= 0:
            if self.current_round_active:
                self.total_rounds += 1
                round_ended = True

                if curr_opponent_health <= 0 and curr_player_health > 0:
                    self.wins += 1
                    print(
                        f"🏆 VICTORY! Win #{self.wins}/{self.total_rounds} "
                        f"(Rate: {self.get_win_rate():.1%})"
                    )
                elif curr_player_health <= 0 and curr_opponent_health > 0:
                    self.losses += 1
                    print(
                        f"💀 DEFEAT! Loss #{self.losses}/{self.total_rounds} "
                        f"(Rate: {self.get_win_rate():.1%})"
                    )
                else:
                    print(f"⚡ DOUBLE KO! Round #{self.total_rounds}")

                self.current_round_active = False
                self._update_win_rate_trend()

        # Simple damage-based rewards: +1 per damage dealt, -1 per damage received
        damage_dealt = max(0, self.prev_opponent_health - curr_opponent_health)
        damage_received = max(0, self.prev_player_health - curr_player_health)

        custom_reward = damage_dealt - damage_received

        # Update health tracking
        self.prev_player_health = curr_player_health
        self.prev_opponent_health = curr_opponent_health

        # Force reset after win/loss if reset_round is True
        if self.reset_round and round_ended:
            custom_done = True

        return custom_reward, custom_done

    def _update_win_rate_trend(self):
        """Update win rate trending data"""
        if (
            self.total_rounds
            >= len(self.win_rate_history) * self.rounds_per_trend_update
        ):
            current_win_rate = self.get_win_rate()
            self.win_rate_history.append(current_win_rate)

            if len(self.win_rate_history) > self.trend_window_size:
                self.win_rate_history.pop(0)

    def get_win_rate(self):
        """Calculate current win rate"""
        return self.wins / max(1, self.total_rounds)

    def get_win_stats(self, info=None):
        """Get comprehensive win statistics including game state"""
        base_stats = {
            "wins": self.wins,
            "losses": self.losses,
            "total_rounds": self.total_rounds,
            "win_rate": self.get_win_rate(),
            "episode_steps": self.episode_steps,
            "health_extraction_success_rate": (
                self.successful_health_extractions
                / max(1, self.health_extraction_attempts)
            ),
        }

        # Add current game state information if available
        if info:
            game_state = {}
            for key in [
                "agent_hp",
                "enemy_hp",
                "agent_x",
                "agent_y",
                "enemy_x",
                "enemy_y",
                "agent_victories",
                "enemy_victories",
                "round_countdown",
                "score",
            ]:
                if key in info:
                    game_state[key] = info[key]

            if game_state:
                base_stats["current_game_state"] = game_state

        return base_stats

    def reset(self, **kwargs):
        """Reset environment with proper initialization"""
        # Get initial observation
        if hasattr(self.env, "reset"):
            result = self.env.reset(**kwargs)
            if isinstance(result, tuple):
                observation, info = result
            else:
                observation = result
                info = {}
        else:
            observation = self.env.reset()
            info = {}

        # Reset tracking variables
        self.prev_player_health = self.full_hp
        self.prev_opponent_health = self.full_hp
        self.episode_steps = 0
        self.current_round_active = True

        # Initialize frame stack with downsampled observation
        self.frame_stack.clear()
        downsampled = observation[::2, ::2, :]  # Downsample to (100, 128, 3)

        for _ in range(self.num_frames):
            self.frame_stack.append(downsampled.copy())

        stacked_obs = self._stack_observation()

        # Add stats to info
        info.update(self.get_win_stats(info))

        return stacked_obs, info

    def step(self, action):
        """Enhanced step function with all fixes"""
        # Handle action filtering and conversion
        if isinstance(self.env.action_space, gym.spaces.MultiBinary):
            if isinstance(action, np.ndarray):
                action = action.astype(int)
            elif isinstance(action, (list, tuple)):
                action = np.array(action, dtype=int)
            else:
                # Convert single integer to MultiBinary
                binary_action = np.zeros(self.env.action_space.n, dtype=int)
                if 0 <= action < self.env.action_space.n:
                    binary_action[action] = 1
                action = binary_action

        # Apply action filtering
        if hasattr(action, "__len__") and len(action) >= 4:
            filtered_action = action.copy() if hasattr(action, "copy") else list(action)
            for button_idx in self.disabled_buttons:
                if button_idx < len(filtered_action):
                    filtered_action[button_idx] = 0
            action = filtered_action

        # Execute step
        observation, reward, done, truncated, info = self.env.step(action)

        # Extract health information
        curr_player_health, curr_opponent_health = self._extract_health_from_info(info)

        # Calculate custom reward (now passing info for additional game state)
        custom_reward, custom_done = self._calculate_reward(
            curr_player_health, curr_opponent_health, info
        )

        # Override done signal if needed
        if custom_done:
            done = custom_done

        # Update frame stack
        downsampled = observation[::2, ::2, :]
        self.frame_stack.append(downsampled)
        stacked_obs = self._stack_observation()

        # Update counters
        self.episode_steps += 1
        self.total_timesteps += 1

        # Update info
        info.update(self.get_win_stats(info))

        return stacked_obs, custom_reward, done, truncated, info


class VecEnv60FPS(VecEnvWrapper):
    """Wrapper to limit vectorized environment to reasonable FPS"""

    def __init__(self, venv, target_fps=60, enable_fps_limit=True):
        super().__init__(venv)
        self.enable_fps_limit = enable_fps_limit
        self.target_fps = target_fps
        self.frame_time = 1.0 / self.target_fps
        self.last_step_time = time.time()

    def step_wait(self):
        if self.enable_fps_limit:
            current_time = time.time()
            elapsed = current_time - self.last_step_time

            if elapsed < self.frame_time:
                time.sleep(self.frame_time - elapsed)

            self.last_step_time = time.time()

        return self.venv.step_wait()

    def step_async(self, actions):
        return self.venv.step_async(actions)

    def reset(self):
        return self.venv.reset()


class EnhancedCallback(BaseCallback):
    """Enhanced callback with detailed logging and win rate tracking"""

    def __init__(self, print_freq=25000, save_freq=100000, verbose=1):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.save_freq = save_freq
        self.last_print = 0
        self.last_save = 0

    def _on_step(self) -> bool:
        # Print progress updates
        if self.num_timesteps - self.last_print >= self.print_freq:
            self.last_print = self.num_timesteps

            print(f"\n{'='*60}")
            print(f"🚀 Training Progress - Step {self.num_timesteps:,}")
            print(
                f"📊 FPS: {self.model.get_env().num_envs * self.print_freq / (time.time() - getattr(self, 'last_time', time.time())):.1f}"
            )
            print(f"📈 Learning rate: {self.model.learning_rate:.2e}")
            print(f"🎯 Clip range: {self.model.clip_range:.3f}")

            # Try to get some environment stats
            try:
                # This won't work perfectly with SubprocVecEnv, but we'll try
                print(f"📋 Check individual environment logs for win rates")
            except:
                pass

            print(f"{'='*60}\n")
            self.last_time = time.time()

        return True


def linear_schedule(initial_value, final_value=0.0):
    """Linear learning rate/clip range schedule"""
    if isinstance(initial_value, str):
        initial_value = float(initial_value)
        final_value = float(final_value)

    def scheduler(progress):
        return final_value + progress * (initial_value - final_value)

    return scheduler


def make_env(game, state, seed=0, rendering=False):
    """Create environment with fixed wrapper"""

    def _init():
        env = retro.make(
            game=game,
            state=state,
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
            render_mode="human" if rendering else None,
        )

        # Apply fixed wrapper
        env = StreetFighterCustomWrapper(
            env,
            reset_round=True,
            rendering=rendering,
            max_episode_steps=5000,  # Increased episode length
        )

        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


def debug_environment(game, state):
    """Debug the environment setup before training"""
    print("\n🔍 DEBUGGING ENVIRONMENT SETUP...")

    try:
        # Test basic environment
        env_fn = make_env(game, state, rendering=False)
        env = env_fn()

        print("✅ Environment created successfully")
        print(f"   Action space: {env.action_space}")
        print(f"   Observation space: {env.observation_space}")

        # Test reset
        obs, info = env.reset()
        print(f"✅ Reset successful - obs shape: {obs.shape}")

        # Test a few steps
        total_reward = 0
        for i in range(20):
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward

            if reward != 0:
                print(f"   Step {i}: reward = {reward:.2f}")

            if done:
                print(f"   Episode ended at step {i}")
                break

        print(f"✅ Environment test completed - total reward: {total_reward:.2f}")

        # Check health extraction success
        stats = env.get_win_stats()
        print(
            f"   Health extraction success rate: {stats['health_extraction_success_rate']:.1%}"
        )

        env.close()
        return True

    except Exception as e:
        print(f"❌ Environment debug failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Fixed Street Fighter II RL Training")
    parser.add_argument("--total-timesteps", type=int, default=10000000)
    parser.add_argument(
        "--num-envs", type=int, default=32, help="Number of parallel environments"
    )
    parser.add_argument("--state-file", type=str, default="ken_bison_12.state")
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--debug", action="store_true", help="Run environment debugging first"
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--save-dir", type=str, default="trained_models_fixed")

    args = parser.parse_args()

    game = "StreetFighterIISpecialChampionEdition-Genesis"

    # Handle state file path
    if os.path.isfile(args.state_file):
        state_file = os.path.abspath(args.state_file)
        print(f"📁 Using custom state file: {state_file}")
    else:
        state_file = args.state_file
        print(f"📁 Using state: {state_file}")

    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Debug environment if requested
    if args.debug:
        if not debug_environment(game, state_file):
            print("❌ Environment debugging failed. Please fix issues before training.")
            return
        print("✅ Environment debugging passed!\n")

    print(f"🚀 Starting FIXED Street Fighter II Training")
    print(f"   📊 Total timesteps: {args.total_timesteps:,}")
    print(f"   🔄 Parallel environments: {args.num_envs}")
    print(f"   📈 Learning rate: {args.learning_rate}")
    print(f"   💾 Save directory: {args.save_dir}")
    print(f"   🎮 State file: {state_file}")

    # Create vectorized environment
    env = SubprocVecEnv(
        [
            make_env(game, state_file, seed=i, rendering=args.render)
            for i in range(args.num_envs)
        ]
    )

    # Add FPS limiting if rendering
    if args.render:
        env = VecEnv60FPS(env, enable_fps_limit=True)
        print("🎮 Applied 60fps limiting for rendering")

    # Improved learning schedules
    lr_schedule = linear_schedule(args.learning_rate, args.learning_rate * 0.1)
    clip_schedule = linear_schedule(0.2, 0.05)

    # Create model with improved hyperparameters
    print("🧠 Creating PPO model with optimized hyperparameters...")
    model = PPO(
        "CnnPolicy",
        env,
        device="cuda",
        verbose=1,
        n_steps=1024,  # Increased buffer size
        batch_size=256,  # Smaller batches for stability
        n_epochs=8,  # More training epochs
        gamma=0.995,  # Higher discount for long-term rewards
        learning_rate=lr_schedule,
        clip_range=clip_schedule,
        ent_coef=0.01,  # Exploration coefficient
        vf_coef=0.8,  # Value function weight
        max_grad_norm=0.5,  # Gradient clipping
        gae_lambda=0.95,  # GAE parameter
        tensorboard_log="logs",
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=100000 // args.num_envs,  # Adjust for number of envs
        save_path=args.save_dir,
        name_prefix="ppo_sf2_fixed",
    )

    progress_callback = EnhancedCallback(print_freq=50000, verbose=1)

    # Train the model
    print("🏋️ Starting training with all fixes applied...")
    print("📊 Monitor progress with: tensorboard --logdir logs")

    start_time = time.time()

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback, progress_callback],
            reset_num_timesteps=True,
        )

        training_time = time.time() - start_time
        print(f"\n🎉 Training completed in {training_time/3600:.1f} hours!")

    except KeyboardInterrupt:
        print(f"\n⏹️ Training interrupted by user")
        training_time = time.time() - start_time
        print(f"   Training time: {training_time/3600:.1f} hours")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return

    finally:
        env.close()

    # Save final model
    final_model_path = os.path.join(args.save_dir, "ppo_sf2_final_fixed.zip")
    model.save(final_model_path)
    print(f"💾 Final model saved: {final_model_path}")

    print("\n✅ Training session complete!")
    print("🎮 Test your model with the evaluation script")


if __name__ == "__main__":
    main()
