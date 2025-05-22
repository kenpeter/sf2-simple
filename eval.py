import os
import time
import numpy as np
import retro
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from wrapper import StreetFighterCustomWrapper


def evaluate_agent(
    model_path, game, state=None, num_episodes=5, render=True, record_video=False
):
    """
    Evaluate a trained agent and display performance metrics

    Args:
        model_path: Path to the trained model
        game: Game name for retro
        state: State file to use (optional)
        num_episodes: Number of episodes to evaluate
        render: Whether to render the game
        record_video: Whether to record evaluation videos
    """

    print(f"Loading model from: {model_path}")

    # Create environment
    env = retro.make(
        game=game,
        state=state,
        use_restricted_actions=retro.Actions.FILTERED,
        obs_type=retro.Observations.IMAGE,
        record=record_video,
    )

    # Wrap the environment
    env = StreetFighterCustomWrapper(env, rendering=render)
    env = Monitor(env)

    # Load the trained model
    try:
        model = PPO.load(model_path, env=env)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    wins = 0
    losses = 0
    total_damage_dealt = 0
    total_damage_taken = 0

    print(f"\nStarting evaluation for {num_episodes} episodes...")
    print("=" * 60)

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        # Track HP at start
        initial_agent_hp = info.get("agent_hp", 176)
        initial_enemy_hp = info.get("enemy_hp", 176)

        print(f"\nEpisode {episode + 1}/{num_episodes}")
        print(f"Starting HP - Agent: {initial_agent_hp}, Enemy: {initial_enemy_hp}")

        start_time = time.time()

        while not done:
            # Get action from model
            action, _states = model.predict(obs, deterministic=True)

            # Take step in environment
            obs, reward, done, info = env.step(action)

            episode_reward += reward
            episode_length += 1

            # Optional: print current HP every few seconds
            if episode_length % 60 == 0:  # Every ~1 second at 60fps
                current_agent_hp = info.get("agent_hp", "N/A")
                current_enemy_hp = info.get("enemy_hp", "N/A")
                print(
                    f"  Step {episode_length}: Agent HP: {current_agent_hp}, Enemy HP: {current_enemy_hp}, Reward: {reward:.4f}"
                )

            if render:
                time.sleep(0.016)  # ~60 FPS

        # Episode finished
        end_time = time.time()
        episode_duration = end_time - start_time

        # Get final HP values
        final_agent_hp = info.get("agent_hp", initial_agent_hp)
        final_enemy_hp = info.get("enemy_hp", initial_enemy_hp)

        # Calculate damage
        damage_dealt = initial_enemy_hp - final_enemy_hp
        damage_taken = initial_agent_hp - final_agent_hp

        # Determine winner
        if final_agent_hp <= 0:
            result = "LOSS"
            losses += 1
        elif final_enemy_hp <= 0:
            result = "WIN"
            wins += 1
        else:
            result = "DRAW"

        # Store metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        total_damage_dealt += max(0, damage_dealt)
        total_damage_taken += max(0, damage_taken)

        # Print episode summary
        print(f"Episode {episode + 1} Complete!")
        print(f"  Result: {result}")
        print(f"  Final HP - Agent: {final_agent_hp}, Enemy: {final_enemy_hp}")
        print(f"  Damage Dealt: {damage_dealt}, Damage Taken: {damage_taken}")
        print(f"  Episode Reward: {episode_reward:.4f}")
        print(f"  Episode Length: {episode_length} steps ({episode_duration:.1f}s)")
        print("-" * 40)

    env.close()

    # Calculate and display final statistics
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(f"Total Episodes: {num_episodes}")
    print(f"Wins: {wins} ({wins/num_episodes*100:.1f}%)")
    print(f"Losses: {losses} ({losses/num_episodes*100:.1f}%)")
    print(f"Draws: {num_episodes - wins - losses}")

    print(f"\nReward Statistics:")
    print(
        f"  Average Reward: {np.mean(episode_rewards):.4f} ± {np.std(episode_rewards):.4f}"
    )
    print(f"  Max Reward: {np.max(episode_rewards):.4f}")
    print(f"  Min Reward: {np.min(episode_rewards):.4f}")

    print(f"\nEpisode Length Statistics:")
    print(
        f"  Average Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f} steps"
    )
    print(f"  Max Length: {np.max(episode_lengths)} steps")
    print(f"  Min Length: {np.min(episode_lengths)} steps")

    print(f"\nDamage Statistics:")
    print(f"  Total Damage Dealt: {total_damage_dealt}")
    print(f"  Total Damage Taken: {total_damage_taken}")
    print(f"  Average Damage Dealt per Episode: {total_damage_dealt/num_episodes:.1f}")
    print(f"  Average Damage Taken per Episode: {total_damage_taken/num_episodes:.1f}")

    if total_damage_taken > 0:
        damage_ratio = total_damage_dealt / total_damage_taken
        print(f"  Damage Ratio (Dealt/Taken): {damage_ratio:.2f}")

    print("=" * 60)

    return {
        "wins": wins,
        "losses": losses,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "total_damage_dealt": total_damage_dealt,
        "total_damage_taken": total_damage_taken,
        "win_rate": wins / num_episodes,
    }


def main():
    # Configuration
    game = "StreetFighterIISpecialChampionEdition-Genesis"

    # Model path - adjust this to your trained model
    model_path = "trained_models/ppo_sf2_ryu_final.zip"

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Model not found at: {model_path}")
        print("Available models in trained_models/:")
        if os.path.exists("trained_models"):
            for file in os.listdir("trained_models"):
                if file.endswith(".zip"):
                    print(f"  - {file}")
        else:
            print("  No trained_models directory found")
        return

    # Evaluation settings
    state = None  # Set to your state file if you have one
    num_episodes = 10
    render = True
    record_video = False

    print("Street Fighter II Agent Evaluation")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Game: {game}")
    print(f"State: {state if state else 'Default'}")
    print(f"Episodes: {num_episodes}")
    print(f"Render: {render}")
    print(f"Record Video: {record_video}")

    # Run evaluation
    results = evaluate_agent(
        model_path=model_path,
        game=game,
        state=state,
        num_episodes=num_episodes,
        render=render,
        record_video=record_video,
    )

    # Save results to file
    import json

    results_file = f"evaluation_results_{int(time.time())}.json"

    # Convert numpy arrays to lists for JSON serialization
    results_for_json = {
        "wins": results["wins"],
        "losses": results["losses"],
        "win_rate": results["win_rate"],
        "total_damage_dealt": results["total_damage_dealt"],
        "total_damage_taken": results["total_damage_taken"],
        "episode_rewards": [float(r) for r in results["episode_rewards"]],
        "episode_lengths": [int(l) for l in results["episode_lengths"]],
        "model_path": model_path,
        "game": game,
        "state": state,
        "timestamp": time.time(),
    }

    with open(results_file, "w") as f:
        json.dump(results_for_json, f, indent=2)

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
