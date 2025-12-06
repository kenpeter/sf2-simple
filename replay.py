#!/usr/bin/env python3
"""
🥊 Street Fighter RL Model Replay
Load and visualize a trained model playing Street Fighter
"""

import argparse
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env.wrappers.atari_wrappers import FrameStack
from wrapper import StreetFighter
import retro


def replay_model(checkpoint_path, num_episodes=5, render=True, frame_stack=4):
    """
    Replay a trained model

    Args:
        checkpoint_path: Path to RLlib checkpoint directory
        num_episodes: Number of episodes to play
        render: Whether to show the game UI
        frame_stack: Number of frames to stack (must match training, default: 4)
    """
    print(f"🎮 Loading model from: {checkpoint_path}")

    if render:
        print("🎮 Rendering enabled - you'll see the game!")

    # Load the trained RLlib algorithm
    algo = PPO.from_checkpoint(checkpoint_path)
    print("✅ Model loaded successfully!")

    wins = 0

    # Play episodes
    for episode in range(num_episodes):
        print(f"\n🥊 Round {episode + 1}/{num_episodes}")

        # Create environment for this episode
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

        # Apply frame stacking
        env = FrameStack(env, frame_stack)

        obs, _ = env.reset()

        total_reward = 0
        steps = 0
        done = False
        info_dict = {}

        while not done:
            # Get action from trained model
            action = algo.compute_single_action(obs, explore=False)

            # Take action
            obs, reward, done, truncated, info_dict = env.step(action)
            total_reward += reward
            steps += 1

            # Debug: print HP every 100 steps
            if steps % 100 == 0:
                print(f"  Step {steps}: Agent HP={info_dict.get('agent_hp', '?')}, Enemy HP={info_dict.get('enemy_hp', '?')}")

            # Render if enabled
            if render:
                env.render()

            if done or truncated:
                break

        # Print round results
        agent_hp = info_dict.get("agent_hp", 0)
        enemy_hp = info_dict.get("enemy_hp", 0)
        agent_won = info_dict.get("agent_won", False)

        if agent_won:
            wins += 1

        print(f"   Steps: {steps}")
        print(f"   Total Reward: {total_reward:.4f}")
        print(f"   Agent HP: {agent_hp}")
        print(f"   Enemy HP: {enemy_hp}")
        print(f"   Result: {'🏆 WIN!' if agent_won else '💀 LOSS'}")

        env.close()

    print(f"\n🏁 Replay complete! Wins: {wins}/{num_episodes} ({wins/num_episodes*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Replay a trained Street Fighter RL model")

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to RLlib checkpoint directory (e.g., ~/ray_results/PPO_xxx/checkpoint_000100)",
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
        "--frame-stack",
        type=int,
        default=4,
        help="Number of frames to stack (must match training, default: 4)",
    )

    args = parser.parse_args()

    replay_model(
        checkpoint_path=args.checkpoint,
        num_episodes=args.episodes,
        render=not args.no_render,
        frame_stack=args.frame_stack,
    )


if __name__ == "__main__":
    main()
