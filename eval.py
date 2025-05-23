import os
import argparse
import numpy as np
import retro
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from wrapper import StreetFighterCustomWrapper


def evaluate_agent(model, env, episodes=10, render=True):
    """Evaluate the trained agent"""
    episode_rewards = []
    episode_lengths = []
    wins = 0
    losses = 0

    for episode in range(episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        print(f"Episode {episode + 1}/{episodes}")

        # Press different button at start for randomization
        random_start_action = np.random.randint(0, 12)  # Random action from 0-11
        obs, _, _, _ = env.step([random_start_action])

        while not done:
            # Get action from model - use stochastic for more variety
            action, _states = model.predict(obs, deterministic=True)

            # Log the action for debugging
            print(f"Step {episode_length}: Action = {action}")

            # Step environment - handle both 4 and 5 return values
            step_result = env.step(action)

            if len(step_result) == 5:
                # New gymnasium API: obs, reward, terminated, truncated, info
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                # Old gym API: obs, reward, done, info
                obs, reward, done, info = step_result

            # Handle vectorized reward (convert to scalar)
            if isinstance(reward, np.ndarray):
                episode_reward += reward[0]
            else:
                episode_reward += reward
            episode_length += 1

            if render:
                env.render()

            # Check game outcome from info if available
            if done and hasattr(info, "__len__") and len(info) > 0:
                # Handle vectorized env info
                game_info = info[0] if isinstance(info, (list, tuple)) else info

                if isinstance(game_info, dict):
                    agent_hp = game_info.get("agent_hp", 176)
                    enemy_hp = game_info.get("enemy_hp", 176)

                    print(f"Final HP - Agent: {agent_hp}, Enemy: {enemy_hp}")

                    if agent_hp > enemy_hp:
                        wins += 1
                        print("Result: WIN!")
                    elif enemy_hp > agent_hp:
                        losses += 1
                        print("Result: LOSS!")
                    else:
                        print("Result: DRAW!")

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        print(f"Episode reward: {episode_reward:.2f}")
        print(f"Episode length: {episode_length}")
        print("-" * 50)

    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)

    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Episodes: {episodes}")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean episode length: {mean_length:.1f}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Win rate: {wins/episodes*100:.1f}%")

    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_length": mean_length,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / episodes,
    }


def make_eval_env(game, state=None):
    """Create evaluation environment"""

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
        # Remove rendering=True to match training environment
        env = StreetFighterCustomWrapper(env, rendering=True, reset_round=True)
        return env

    return _init


def main():
    parser = argparse.ArgumentParser(description="Evaluate Street Fighter II Agent")
    parser.add_argument(
        "--model",
        type=str,
        default="trained_models/ppo_ryu_4500000_steps.zip",
        help="Path to trained model",
    )
    parser.add_argument(
        "--episodes", type=int, default=10, help="Number of episodes to evaluate"
    )
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument(
        "--state", type=str, default="ken_bison_12.state", help="Game state file"
    )

    args = parser.parse_args()

    game = "StreetFighterIISpecialChampionEdition-Genesis"

    print("Street Fighter II Agent Evaluation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Game: {game}")
    print(f"State: {args.state}")
    print(f"Episodes: {args.episodes}")
    print(f"Render: {not args.no_render}")

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return

    # Create environment
    env = DummyVecEnv([make_eval_env(game, args.state)])
    env = VecTransposeImage(env)

    # Load model
    print(f"Loading model from: {args.model}")
    try:
        model = PPO.load(args.model, env=env)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Evaluate
    print(f"\nStarting evaluation for {args.episodes} episodes...")
    print("=" * 60)

    results = evaluate_agent(
        model=model, env=env, episodes=args.episodes, render=not args.no_render
    )

    env.close()


if __name__ == "__main__":
    main()
