#!/usr/bin/env python3
"""
🥊 Street Fighter RL Model Replay
Load and visualize a trained model playing Street Fighter
"""

import argparse
from stable_baselines3 import PPO
from wrapper import StreetFighter
import retro


def replay_model(model_path, num_episodes=5, render=True):
    """
    Replay a trained model

    Args:
        model_path: Path to the trained model (.zip file)
        num_episodes: Number of episodes to play
        render: Whether to show the game UI
    """
    print(f"🎮 Loading model from: {model_path}")

    # Load the trained model
    model = PPO.load(model_path)
    print("✅ Model loaded successfully!")

    # Create environment with rendering enabled
    env = StreetFighter()

    # Override render mode if requested
    if render:
        # Recreate the game with rendering enabled
        env.game.close()
        game = retro.make(
            "StreetFighterIISpecialChampionEdition-Genesis",
            state="ken_bison_12.state",
            use_restricted_actions=retro.Actions.FILTERED,
            render_mode="human",
        )
        from discretizer import StreetFighter2Discretizer
        env.game = StreetFighter2Discretizer(game)
        print("🎮 Rendering enabled - you'll see the game!")

    # Play episodes
    for episode in range(num_episodes):
        print(f"\n🥊 Episode {episode + 1}/{num_episodes}")

        obs, info = env.reset()

        total_reward = 0
        steps = 0
        done = False

        while not done:
            # Predict action using the trained model
            action, _states = model.predict(obs, deterministic=True)

            # Take action
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1

            # Render if enabled
            if render:
                env.render()

        # Print episode results
        agent_hp = info.get("agent_hp", 0)
        enemy_hp = info.get("enemy_hp", 0)
        agent_won = info.get("agent_won", False)

        print(f"   Steps: {steps}")
        print(f"   Total Reward: {total_reward:.4f}")
        print(f"   Agent HP: {agent_hp}")
        print(f"   Enemy HP: {enemy_hp}")
        print(f"   Result: {'🏆 WIN!' if agent_won else '💀 LOSS'}")

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

    args = parser.parse_args()

    replay_model(
        model_path=args.model_path,
        num_episodes=args.episodes,
        render=not args.no_render,
    )


if __name__ == "__main__":
    main()
