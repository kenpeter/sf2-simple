import os
import argparse
import numpy as np
import retro
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

from wrapper import StreetFighterCustomWrapper


class VariedActionWrapper:
    """Wrapper to add controlled randomness to actions"""

    def __init__(self, model, exploration_rate=0.15, action_noise=0.1):
        self.model = model
        self.exploration_rate = exploration_rate  # Chance to take random action
        self.action_noise = action_noise  # Chance to modify predicted action

    def predict(self, obs, deterministic=False):
        # Get model's predicted action
        action, states = self.model.predict(obs, deterministic=deterministic)

        # Add some randomness based on exploration_rate
        if np.random.random() < self.exploration_rate:
            # Take a completely random action occasionally
            # For MultiBinary action space (12 buttons)
            random_action = np.random.randint(0, 2, size=12).astype(int)
            return random_action, states

        # Add noise to the predicted action
        if not deterministic and np.random.random() < self.action_noise:
            # For MultiBinary actions, flip some bits randomly
            noisy_action = action.copy()
            # Ensure it's a proper numpy array
            if hasattr(noisy_action, "flatten"):
                noisy_action = noisy_action.flatten()

            for i in range(len(noisy_action)):
                if np.random.random() < 0.2:  # 20% chance to flip each bit
                    noisy_action[i] = 1 - noisy_action[i]
            return noisy_action.astype(int), states

        return action, states


def random_starting_sequence(env, num_actions=None):
    """Execute a random sequence of actions at the start"""
    if num_actions is None:
        num_actions = np.random.randint(5, 20)  # Random between 5-20 actions

    print(f"Executing {num_actions} random starting actions...")

    for i in range(num_actions):
        # Create random MultiBinary action (12 buttons)
        if np.random.random() < 0.7:  # 70% chance for action
            # Create a random button combination
            action = np.random.randint(0, 2, size=12).astype(int)

            # Sometimes just press single buttons for more realistic actions
            if np.random.random() < 0.5:
                action = np.zeros(12, dtype=int)
                # Press 1-3 random buttons
                num_buttons = np.random.randint(1, 4)
                buttons_to_press = np.random.choice(12, size=num_buttons, replace=False)
                action[buttons_to_press] = 1
        else:
            # No-op action (all zeros)
            action = np.zeros(12, dtype=int)

        # Step the environment
        step_result = env.step(action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, reward, done, info = step_result

        if done:
            print("Game ended during random sequence, resetting...")
            obs = env.reset()
            break

    return obs


def evaluate_agent(model, env, episodes=10, render=True, varied_gameplay=True):
    """Evaluate the trained agent with optional gameplay variation"""
    episode_rewards = []
    episode_lengths = []
    wins = 0
    losses = 0

    # Create varied action wrapper if requested
    if varied_gameplay:
        # Different exploration rates for different episodes
        exploration_rates = [0.1, 0.15, 0.2, 0.25, 0.3]
        action_noise_rates = [0.05, 0.1, 0.15, 0.2]

    for episode in range(episodes):
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        print(f"Episode {episode + 1}/{episodes}")

        # Create episode-specific varied wrapper
        if varied_gameplay:
            exploration_rate = np.random.choice(exploration_rates)
            action_noise = np.random.choice(action_noise_rates)
            varied_model = VariedActionWrapper(model, exploration_rate, action_noise)
            print(
                f"Using exploration_rate: {exploration_rate:.2f}, action_noise: {action_noise:.2f}"
            )
        else:
            varied_model = model

        # Random starting sequence to create different game states
        if varied_gameplay:
            obs = random_starting_sequence(env.envs[0], np.random.randint(3, 15))

        # Vary deterministic behavior throughout episode
        deterministic_phases = [True, False] if varied_gameplay else [True]
        phase_length = np.random.randint(50, 150) if varied_gameplay else float("inf")
        current_phase = 0
        steps_in_phase = 0

        while not done:
            # Switch between deterministic and stochastic phases
            if varied_gameplay and steps_in_phase >= phase_length:
                current_phase = (current_phase + 1) % len(deterministic_phases)
                steps_in_phase = 0
                phase_length = np.random.randint(30, 100)
                print(
                    f"Switching to {'deterministic' if deterministic_phases[current_phase] else 'stochastic'} phase"
                )

            use_deterministic = (
                deterministic_phases[current_phase] if varied_gameplay else True
            )

            # Get action from model (with or without variation)
            action, _states = varied_model.predict(obs, deterministic=use_deterministic)

            # Ensure action is properly formatted for MultiBinary
            if hasattr(action, "flatten"):
                action = action.flatten()
            action = action.astype(int)

            # Occasionally log actions for debugging
            if episode_length % 50 == 0:
                print(f"Step {episode_length}: Action = {action}")

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
            steps_in_phase += 1

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


def make_eval_env(game, state=None, add_randomness=True):
    """Create evaluation environment with FIXED wrapper parameters"""

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

        # Use wrapper parameters that match your current wrapper.py
        env = StreetFighterCustomWrapper(
            env,
            rendering=True,
            reset_round=True,
            # Note: reward_coeff and full_hp are handled internally in your wrapper
        )
        return env

    return _init


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Street Fighter II Agent (FIXED)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="trained_models/ppo_sf2_original_4500000_steps.zip",  # Updated default
        help="Path to trained model",
    )
    parser.add_argument(
        "--episodes", type=int, default=4, help="Number of episodes to evaluate"
    )
    parser.add_argument("--no-render", action="store_true", help="Disable rendering")
    parser.add_argument(
        "--state", type=str, default="ken_bison_12.state", help="Game state file"
    )
    parser.add_argument(
        "--no-variation",
        action="store_true",
        help="Disable gameplay variation (use deterministic evaluation)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible varied gameplay",
    )

    args = parser.parse_args()

    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Using random seed: {args.seed}")

    game = "StreetFighterIISpecialChampionEdition-Genesis"
    varied_gameplay = not args.no_variation

    print("Street Fighter II Agent Evaluation (FIXED WRAPPER)")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Game: {game}")
    print(f"State: {args.state}")
    print(f"Episodes: {args.episodes}")
    print(f"Render: {not args.no_render}")
    print(f"Varied Gameplay: {varied_gameplay}")
    print("Using FIXED reward system (no normalization, reward_coeff=3)")

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return

    # Create environment with FIXED wrapper
    env = DummyVecEnv([make_eval_env(game, args.state, add_randomness=varied_gameplay)])
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
    if varied_gameplay:
        print(f"\nStarting VARIED evaluation for {args.episodes} episodes...")
        print("Each episode will have different:")
        print("- Random starting sequence")
        print("- Different exploration rates")
        print("- Mixed deterministic/stochastic phases")
        print("- Action noise variations")
        print("- Using FIXED reward system")
    else:
        print(f"\nStarting STANDARD evaluation for {args.episodes} episodes...")
        print("- Using FIXED reward system")

    print("=" * 60)

    results = evaluate_agent(
        model=model,
        env=env,
        episodes=args.episodes,
        render=not args.no_render,
        varied_gameplay=varied_gameplay,
    )

    env.close()


if __name__ == "__main__":
    main()
