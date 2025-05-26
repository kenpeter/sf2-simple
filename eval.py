import os
import argparse
import numpy as np
import retro
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecTransposeImage,
)
from stable_baselines3.common.monitor import Monitor

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


def make_env(game, state, seed=0):
    """EXACT copy of training environment function"""

    def _init():
        # Handle state file path properly
        if state and os.path.exists(state):
            state_path = os.path.abspath(state)
        else:
            state_path = state

        # Create retro environment
        env = retro.make(
            game=game,
            state=state_path,
            use_restricted_actions=retro.Actions.FILTERED,
            obs_type=retro.Observations.IMAGE,
        )

        # Add action filtering to prevent cheating
        env = FilteredActionWrapper(env)

        # Add ORIGINAL custom wrapper (no extra parameters!)
        env = StreetFighterCustomWrapper(env, reset_round=True, rendering=False)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


def evaluate_agent(model, env, episodes=10, render=False):
    """Evaluate the agent"""
    episode_rewards = []
    wins = 0
    losses = 0
    draws = 0

    for episode in range(episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        print(f"Episode {episode + 1}/{episodes}")

        while not done:
            # Use deterministic evaluation
            action, _states = model.predict(obs, deterministic=True)

            # Ensure action is properly formatted for MultiBinary
            if hasattr(action, "flatten"):
                action = action.flatten()
            action = action.astype(int)

            # Step environment
            step_result = env.step(action)

            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result

            # Handle vectorized reward
            if isinstance(reward, np.ndarray):
                episode_reward += reward[0]
            else:
                episode_reward += reward
            episode_length += 1

            if render:
                env.render()

            # Check game outcome from info if available
            if done and hasattr(info, "__len__") and len(info) > 0:
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
                        draws += 1
                        print("Result: DRAW!")

        episode_rewards.append(episode_reward)

        print(f"Episode reward: {episode_reward:.2f}")
        print(f"Episode length: {episode_length}")
        print("-" * 50)

    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean([len(ep) for ep in [episode_rewards]])

    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Episodes: {episodes}")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Wins: {wins}")
    print(f"Losses: {losses}")
    print(f"Draws: {draws}")
    print(f"Win rate: {wins/episodes*100:.1f}%")
    print(f"Loss rate: {losses/episodes*100:.1f}%")

    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / episodes,
    }


def create_training_matched_env(game, state):
    """Create environment that exactly matches training setup"""
    # Use DummyVecEnv for single-environment evaluation (faster than SubprocVecEnv)
    env = DummyVecEnv([make_env(game, state=state, seed=0)])

    # CRITICAL: Add VecTransposeImage to match training
    # Training shows "Wrapping the env in a VecTransposeImage" so we must do the same
    env = VecTransposeImage(env)

    return env


def debug_environment_setup(model_path, game, state):
    """Debug environment setup to verify it matches training"""
    print(f"\n{'='*60}")
    print("DEBUGGING ENVIRONMENT SETUP")
    print(f"{'='*60}")

    # Create environment exactly like training
    env = create_training_matched_env(game, state)

    # Load model
    model = PPO.load(model_path, env=env)

    # Check environment properties
    obs = env.reset()
    print(f"Environment observation shape: {obs.shape}")
    print(f"Environment observation space: {env.observation_space}")
    print(f"Model observation space: {model.observation_space}")
    print(f"Action space: {env.action_space}")

    # Test prediction
    action, _ = model.predict(obs, deterministic=True)
    print(f"Action shape: {action.shape}")
    print(f"Sample action: {action.flatten()}")

    # Test one step
    obs, reward, done, info = env.step(action)
    print(f"Step successful - reward: {reward}, done: {done}")

    env.close()

    print("✓ Environment setup matches training expectations")


def compare_models(model_paths, game, state, episodes=5):
    """Compare performance of different models"""
    print(f"\n{'='*60}")
    print("COMPARING MODEL PERFORMANCE")
    print(f"{'='*60}")

    results = {}

    for model_path in model_paths:
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            continue

        print(f"\nTesting model: {os.path.basename(model_path)}")
        print("-" * 40)

        try:
            env = create_training_matched_env(game, state)
            model = PPO.load(model_path, env=env)
            result = evaluate_agent(model, env, episodes=episodes)
            results[os.path.basename(model_path)] = result
            env.close()
        except Exception as e:
            print(f"Error testing {model_path}: {e}")

    # Summary
    print(f"\n{'='*60}")
    print("MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    for model_name, result in results.items():
        print(
            f"{model_name}: {result['win_rate']*100:.1f}% win rate ({result['wins']}/{episodes} wins)"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Street Fighter II Agent Evaluation (FIXED - Matches Training)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="trained_models/ppo_sf2_original_7000000_steps.zip",
        help="Path to trained model",
    )
    parser.add_argument(
        "--episodes", type=int, default=10, help="Number of episodes to evaluate"
    )
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument(
        "--state", type=str, default="ken_bison_12.state", help="Game state file"
    )
    parser.add_argument("--debug", action="store_true", help="Debug environment setup")
    parser.add_argument(
        "--compare", action="store_true", help="Compare all available models"
    )
    parser.add_argument(
        "--latest", action="store_true", help="Use the latest model automatically"
    )

    args = parser.parse_args()

    game = "StreetFighterIISpecialChampionEdition-Genesis"

    print("Street Fighter II Agent Evaluation (FIXED VERSION)")
    print("=" * 60)
    print("This evaluation EXACTLY matches the training setup:")
    print("- DummyVecEnv (single environment for evaluation)")
    print("- VecTransposeImage (channels-first observations)")
    print("- FilteredActionWrapper (START/SELECT disabled)")
    print("- Monitor wrapper")
    print("- Same reward system")
    print("=" * 60)

    # Auto-select latest model if requested
    if args.latest:
        model_dir = "trained_models"
        if os.path.exists(model_dir):
            models = [
                f
                for f in os.listdir(model_dir)
                if f.endswith(".zip") and "ppo_sf2" in f
            ]
            if models:
                # Sort by modification time and get latest
                models.sort(key=lambda x: os.path.getmtime(os.path.join(model_dir, x)))
                args.model = os.path.join(model_dir, models[-1])
                print(f"Auto-selected latest model: {args.model}")

    print(f"Model: {args.model}")
    print(f"State: {args.state}")
    print(f"Episodes: {args.episodes}")
    print(f"Render: {not args.no_render}")

    # Check if state file exists
    if not os.path.exists(args.state):
        print(f"Error: State file not found: {args.state}")
        print("Available state files:")
        for f in os.listdir("."):
            if f.endswith(".state"):
                print(f"  {f}")
        return

    if args.debug:
        debug_environment_setup(args.model, game, args.state)
        return

    if args.compare:
        # Find all available models
        model_dir = "trained_models"
        if os.path.exists(model_dir):
            models = [
                os.path.join(model_dir, f)
                for f in os.listdir(model_dir)
                if f.endswith(".zip") and "ppo_sf2" in f
            ]
            models.sort(key=lambda x: os.path.getmtime(x))  # Sort by creation time

            if models:
                compare_models(
                    models[-5:], game, args.state, episodes=args.episodes
                )  # Test last 5 models
            else:
                print("No models found for comparison")
        return

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        print("Available models:")
        model_dir = "trained_models"
        if os.path.exists(model_dir):
            for f in os.listdir(model_dir):
                if f.endswith(".zip"):
                    print(f"  {f}")
        return

    # Create environment that exactly matches training
    print(f"Creating environment with state: {os.path.abspath(args.state)}")
    env = create_training_matched_env(game, args.state)

    # Load model
    print(f"Loading model from: {args.model}")
    try:
        model = PPO.load(args.model, env=env)
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        env.close()
        return

    # Evaluate
    print(f"\nStarting evaluation for {args.episodes} episodes...")
    print("Using EXACT training environment setup")
    print("=" * 60)

    results = evaluate_agent(
        model=model,
        env=env,
        episodes=args.episodes,
        render=not args.no_render,
    )

    env.close()

    # Final summary
    print(f"\nFINAL SUMMARY:")
    print(f"Model performance: {results['win_rate']*100:.1f}% win rate")
    if results["win_rate"] > 0.5:
        print("🎉 Great performance! Agent is winning more than 50% of fights.")
    elif results["win_rate"] > 0.2:
        print("👍 Decent performance! Agent is learning to fight.")
    elif results["win_rate"] > 0.05:
        print("📈 Some progress! Agent occasionally wins fights.")
    else:
        print("⚠️  Poor performance. Agent may need more training or debugging.")


if __name__ == "__main__":
    main()
