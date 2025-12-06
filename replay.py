#!/usr/bin/env python3
"""
🥊 Street Fighter RL Model Replay
Load and visualize a trained model playing Street Fighter
"""

import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from wrapper import StreetFighter
import retro


def replay_model(model_path, num_episodes=5, render=True, frame_stack=4):
    """
    Replay a trained model

    Args:
        model_path: Path to the trained model (.zip file)
        num_episodes: Number of episodes to play
        render: Whether to show the game UI
        frame_stack: Number of frames to stack (must match training, default: 4)
    """
    print(f"🎮 Loading model from: {model_path}")

    if render:
        print("🎮 Rendering enabled - you'll see the game!")

    print("⚠️  Note: Each round is a separate episode (matching training).")
    print("    The environment will be recreated after each round to prevent 'best of 3' mode.\n")

    # Load the trained model first (without env)
    model = PPO.load(model_path)
    print("✅ Model loaded successfully!")

    # Play episodes (each episode = one round, matching training)
    # We recreate the environment each time to avoid "best of 3" continuation
    for episode in range(num_episodes):
        print(f"\n🥊 Round {episode + 1}/{num_episodes}")

        # Create fresh environment for each episode
        def make_env():
            env = StreetFighter()
            # Override render mode if requested
            if render:
                env.game.close()
                game = retro.make(
                    "StreetFighterIISpecialChampionEdition-Genesis",
                    state="ken_bison_12.state",
                    use_restricted_actions=retro.Actions.FILTERED,
                    render_mode="human",
                )
                from discretizer import StreetFighter2Discretizer
                env.game = StreetFighter2Discretizer(game)
            env = Monitor(env)
            return env

        # Create vectorized environment with frame stacking (must match training)
        env = DummyVecEnv([make_env])
        env = VecFrameStack(env, frame_stack, channels_order="last")

        obs = env.reset()

        total_reward = 0
        steps = 0
        done = False
        info_dict = {}

        while not done:
            # Predict action using the trained model
            action, _states = model.predict(obs, deterministic=True)

            # Take action (vectorized env returns arrays)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]  # Extract scalar from array
            done = done[0]  # Extract scalar from array
            info_dict = info[0] if isinstance(info, list) else info
            steps += 1

            # Debug: print HP every 100 steps
            if steps % 100 == 0:
                print(f"  Step {steps}: Agent HP={info_dict.get('agent_hp', '?')}, Enemy HP={info_dict.get('enemy_hp', '?')}, Done={done}")

            # Render if enabled
            if render:
                env.render()

        # Print round results (matching training: one round per episode)
        agent_hp = info_dict.get("agent_hp", 0)
        enemy_hp = info_dict.get("enemy_hp", 0)
        agent_won = info_dict.get("agent_won", False)

        print(f"   Steps: {steps}")
        print(f"   Total Reward: {total_reward:.4f}")
        print(f"   Agent HP: {agent_hp}")
        print(f"   Enemy HP: {enemy_hp}")
        print(f"   Result: {'🏆 WIN!' if agent_won else '💀 LOSS'}")

        # Close environment after each episode to prevent "best of 3" continuation
        env.close()

    print("\n🏁 Replay complete!")


def main():
    parser = argparse.ArgumentParser(description="Replay a trained Street Fighter RL model")

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model (.zip file)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to play (default: 5)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable game rendering (headless mode)",
    )
    parser.add_argument(
        "--frame_stack",
        type=int,
        default=4,
        help="Number of frames to stack (must match training, default: 4)",
    )

    args = parser.parse_args()

    replay_model(
        model_path=args.model_path,
        num_episodes=args.episodes,
        render=not args.no_render,
        frame_stack=args.frame_stack,
    )


if __name__ == "__main__":
    main()
