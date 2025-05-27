import os
import argparse
import time

import retro
import gymnasium as gym
from stable_baselines3 import PPO

# Import the wrapper
from wrapper import StreetFighterCustomWrapper


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


def create_eval_env(game, state):
    """Create evaluation environment with rendering enabled"""
    # Create retro environment
    env = retro.make(
        game=game,
        state=state,
        use_restricted_actions=retro.Actions.FILTERED,
        obs_type=retro.Observations.IMAGE,
        render_mode="human",  # Enable human-visible rendering
    )

    # Add action filtering to prevent cheating
    env = FilteredActionWrapper(env)

    # Add custom wrapper with rendering enabled and reset_round=False for full matches
    env = StreetFighterCustomWrapper(env, reset_round=False, rendering=True)

    return env


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained Street Fighter II Agent"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="trained_models/ppo_sf2_model_original.zip",
        help="Path to the trained model",
    )
    parser.add_argument(
        "--state-file",
        type=str,
        default="ken_bison_12.state",
        help="State file to use",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to run",
    )

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return

    game = "StreetFighterIISpecialChampionEdition-Genesis"

    print(f"Loading model from: {args.model_path}")
    print(f"Using state file: {args.state_file}")
    print(f"Will run {args.episodes} episodes")
    print("Running at 60 FPS for smooth gameplay")
    print("\nPress Ctrl+C to quit at any time")
    print("=" * 50)

    # Create evaluation environment
    env = create_eval_env(game, args.state_file)

    # Load the trained model
    try:
        model = PPO.load(args.model_path, device="cuda")
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    try:
        for episode in range(args.episodes):
            print(f"\n--- Episode {episode + 1}/{args.episodes} ---")

            obs, info = env.reset()
            episode_reward = 0
            step_count = 0

            print("Starting new match... Watch the game window!")

            while True:
                # Get action from the trained model
                action, _states = model.predict(obs, deterministic=True)

                # Take step in environment
                obs, reward, terminated, truncated, info = env.step(action)

                episode_reward += reward
                step_count += 1

                # Maintain 60 FPS for smooth viewing
                time.sleep(0.0167)  # 60 FPS timing

                # Check if episode is done
                if terminated or truncated:
                    break

                # Optional: Add some info display every 5 seconds at 60fps
                if step_count % 300 == 0:  # Every 5 seconds at 60fps (300 steps)
                    player_hp = info.get("agent_hp", "?")
                    enemy_hp = info.get("enemy_hp", "?")
                    print(
                        f"Step {step_count}: Player HP: {player_hp}, Enemy HP: {enemy_hp}"
                    )

            print(f"Episode {episode + 1} finished!")
            print(f"Total reward: {episode_reward:.1f}")
            print(f"Steps taken: {step_count}")

            # Get final health values
            player_hp = info.get("agent_hp", "?")
            enemy_hp = info.get("enemy_hp", "?")
            print(f"Final - Player HP: {player_hp}, Enemy HP: {enemy_hp}")

            # Determine winner
            if player_hp <= 0:
                print("🔴 AI Lost this round")
            elif enemy_hp <= 0:
                print("🟢 AI Won this round")
            else:
                print("⚪ Round ended without knockout")

            # Pause between episodes
            if episode < args.episodes - 1:
                print("\nWaiting 3 seconds before next episode...")
                time.sleep(3)

    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
    except Exception as e:
        print(f"\nError during evaluation: {e}")
    finally:
        env.close()
        print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
