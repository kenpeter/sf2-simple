#!/usr/bin/env python3
"""
Evaluation script for trained Street Fighter II agent with discrete actions
Compatible with StreetFighterVisionWrapper and strategic vision pipeline
"""

import os
import argparse
import time
import numpy as np
import torch

import retro
import gymnasium as gym
from stable_baselines3 import PPO

# Import the correct wrapper and CNN from training setup
from wrapper import StreetFighterVisionWrapper, StreetFighterSimplifiedCNN


def create_eval_env(game, state, enable_vision_transformer=True, defend_actions=None):
    """Create evaluation environment matching training setup exactly"""

    # Handle state file path consistently with train.py
    if os.path.isfile(state):
        state_file = os.path.abspath(state)
        print(f"📁 Using custom state file: {state_file}")
    else:
        if state.endswith(".state"):
            state_file = state[:-6]  # Remove .state extension for built-in states
            print(f"🎮 Using built-in state: {state_file}")
        else:
            state_file = state
            print(f"🎮 Using state: {state_file}")

    # Create retro environment with exact training settings
    env = retro.make(
        game=game,
        state=state_file,
        use_restricted_actions=retro.Actions.FILTERED,  # 12 buttons filtered
        obs_type=retro.Observations.IMAGE,
        render_mode="human",  # Enable rendering for human observation
    )

    # Apply StreetFighterVisionWrapper with exact training configuration
    env = StreetFighterVisionWrapper(
        env,
        reset_round=True,
        rendering=True,
        max_episode_steps=5000,
        frame_stack=8,  # 8 RGB frames
        enable_vision_transformer=enable_vision_transformer,
        defend_action_indices=defend_actions or [54, 55, 56],  # Default block actions
        log_transformer_predictions=False,  # Disable logging during eval
    )

    print(f"✅ Evaluation environment created:")
    print(f"   Observation space: {env.observation_space.shape}")
    print(f"   Action space: Discrete({env.action_space.n}) actions")
    print(
        f"   Vision Transformer: {'Enabled' if enable_vision_transformer else 'Disabled'}"
    )

    return env


def display_action_info(action, discrete_actions):
    """Display human-readable action information"""
    try:
        action_name = discrete_actions.get_action_name(action)
        return action_name
    except:
        return f"Action_{action}"


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained Street Fighter II Agent with Discrete Actions"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="trained_models_cuda_discrete/ppo_sf2_cuda_discrete_final.zip",
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
    parser.add_argument(
        "--use-built-in-state",
        action="store_true",
        help="Use built-in state (removes .state extension)",
    )
    parser.add_argument(
        "--no-vision-transformer",
        action="store_true",
        help="Disable vision transformer during evaluation",
    )
    parser.add_argument(
        "--defend-actions",
        type=str,
        default="54,55,56",
        help="Comma-separated list of defend action indices",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Frames per second for rendering (default: 60)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions (no exploration)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run model on (cuda/cpu)"
    )

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model file not found at {args.model_path}")

        # Check common directories
        for dirname in ["trained_models", "trained_models_cuda_discrete", "."]:
            if os.path.exists(dirname):
                print(f"\nAvailable models in {dirname}/:")
                for f in sorted(os.listdir(dirname)):
                    if f.endswith(".zip"):
                        print(f"   - {f}")
        return

    game = "StreetFighterIISpecialChampionEdition-Genesis"
    enable_vision_transformer = not args.no_vision_transformer
    defend_actions = [int(x.strip()) for x in args.defend_actions.split(",")]

    # Handle state file properly
    if args.use_built_in_state:
        state_file = (
            args.state_file[:-6]
            if args.state_file.endswith(".state")
            else args.state_file
        )
    else:
        state_file = args.state_file

    print(f"🚀 Street Fighter II Discrete Action Evaluation")
    print(f"🤖 Model: {args.model_path}")
    print(f"🎮 State: {state_file}")
    print(f"🔄 Episodes: {args.episodes}")
    print(
        f"🧠 Vision Transformer: {'Enabled' if enable_vision_transformer else 'Disabled'}"
    )
    print(f"🛡️  Defend Actions: {defend_actions}")
    print(f"🎬 FPS: {args.fps}")
    print(f"🎯 Deterministic: {args.deterministic}")
    print(f"💻 Device: {args.device}")
    print("=" * 60)

    # Create evaluation environment
    try:
        env = create_eval_env(
            game,
            state_file,
            enable_vision_transformer=enable_vision_transformer,
            defend_actions=defend_actions,
        )
        print("✅ Environment created successfully!")
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        print(
            "\n💡 Try using --use-built-in-state flag if you're using a built-in state"
        )
        import traceback

        traceback.print_exc()
        return

    # Load the trained model
    try:
        print("🧠 Loading model...")
        device = (
            args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
        )
        model = PPO.load(args.model_path, device=device)
        print(f"✅ Model loaded on {device}!")

        # Check observation space compatibility
        model_shape = model.observation_space.shape
        env_shape = env.observation_space.shape

        print(f"🔍 Model expects: {model_shape}")
        print(f"🔍 Environment provides: {env_shape}")

        if model_shape != env_shape:
            print("⚠️  Observation shapes differ! This may cause issues.")
            return
        else:
            print("✅ Observation shapes match perfectly!")

        # Check action space compatibility
        model_actions = model.action_space.n
        env_actions = env.action_space.n

        print(f"🎮 Model actions: {model_actions}")
        print(f"🎮 Environment actions: {env_actions}")

        if model_actions != env_actions:
            print("⚠️  Action spaces differ! This may cause issues.")
            return
        else:
            print("✅ Action spaces match perfectly!")

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback

        traceback.print_exc()
        return

    # Inject feature extractor for vision transformer
    if enable_vision_transformer:
        try:
            print("💉 Injecting feature extractor for vision transformer...")
            feature_extractor = model.policy.features_extractor

            if hasattr(env, "inject_feature_extractor"):
                env.inject_feature_extractor(feature_extractor)
                print("✅ Feature extractor injected successfully!")
            else:
                print("⚠️  Environment doesn't support feature extractor injection")

        except Exception as e:
            print(f"⚠️  Feature extractor injection failed: {e}")

    # Run evaluation episodes
    try:
        total_wins = 0
        total_losses = 0
        total_draws = 0
        total_reward = 0.0
        frame_time = 1.0 / args.fps  # Calculate frame timing

        for episode in range(args.episodes):
            print(f"\n🥊 --- Episode {episode + 1}/{args.episodes} ---")

            obs, info = env.reset()
            episode_reward = 0
            step_count = 0
            last_action_name = "IDLE"

            print("🎬 Starting new match... Watch the game window!")
            start_time = time.time()

            while True:
                # Get action from the trained model
                action, _states = model.predict(obs, deterministic=args.deterministic)

                # Display action info occasionally
                if hasattr(env, "discrete_actions"):
                    action_name = display_action_info(action, env.discrete_actions)
                    if action_name != last_action_name:
                        print(f"   🎮 Action: {action_name}")
                        last_action_name = action_name

                # Take step in environment
                obs, reward, terminated, truncated, info = env.step(action)

                episode_reward += reward
                step_count += 1

                # Maintain target FPS for smooth viewing
                time.sleep(frame_time)

                # Check if episode is done
                if terminated or truncated:
                    break

                # Display info every 5 seconds
                if step_count % (args.fps * 5) == 0:  # Every 5 seconds
                    elapsed = time.time() - start_time
                    player_hp = info.get("agent_hp", "?")
                    enemy_hp = info.get("enemy_hp", "?")
                    print(
                        f"   ⏱️  {elapsed:.1f}s - Player: {player_hp} HP, Enemy: {enemy_hp} HP"
                    )

                    # Show vision transformer predictions if available
                    if enable_vision_transformer and hasattr(
                        env, "current_attack_timing"
                    ):
                        attack_timing = getattr(env, "current_attack_timing", 0)
                        defend_timing = getattr(env, "current_defend_timing", 0)
                        print(
                            f"   🧠 AI Analysis - Attack: {attack_timing:.3f}, Defend: {defend_timing:.3f}"
                        )

            # Episode finished
            total_reward += episode_reward
            print(f"🏁 Episode {episode + 1} finished!")
            print(f"   Duration: {time.time() - start_time:.1f} seconds")
            print(f"   Total reward: {episode_reward:.1f}")
            print(f"   Steps taken: {step_count}")

            # Get final health values and determine winner
            player_hp = info.get("agent_hp", 0)
            enemy_hp = info.get("enemy_hp", 0)
            print(f"   Final - Player HP: {player_hp}, Enemy HP: {enemy_hp}")

            if player_hp <= 0 and enemy_hp > 0:
                print("   🔴 AI Lost this round")
                total_losses += 1
            elif enemy_hp <= 0 and player_hp > 0:
                print("   🟢 AI Won this round")
                total_wins += 1
            else:
                print("   ⚪ Draw or timeout")
                total_draws += 1

            # Show win rate so far
            total_games = total_wins + total_losses + total_draws
            if total_games > 0:
                win_rate = (total_wins / total_games) * 100
                print(
                    f"   📊 Win rate so far: {win_rate:.1f}% ({total_wins}/{total_games})"
                )

            # Pause between episodes
            if episode < args.episodes - 1:
                print("\n⏳ Waiting 3 seconds before next episode...")
                time.sleep(3)

        # Final statistics
        total_games = total_wins + total_losses + total_draws
        win_rate = (total_wins / total_games) * 100 if total_games > 0 else 0
        avg_reward = total_reward / args.episodes

        print(f"\n📊 === FINAL EVALUATION RESULTS ===")
        print(f"   Games Played: {total_games}")
        print(f"   Wins: {total_wins}")
        print(f"   Losses: {total_losses}")
        print(f"   Draws: {total_draws}")
        print(f"   Win Rate: {win_rate:.1f}%")
        print(f"   Average Reward: {avg_reward:.2f}")
        print(f"   Model: {os.path.basename(args.model_path)}")
        print("=" * 40)

        # Performance assessment
        if win_rate >= 80:
            print("🏆 EXCELLENT performance!")
        elif win_rate >= 60:
            print("🥈 GOOD performance!")
        elif win_rate >= 40:
            print("🥉 AVERAGE performance")
        else:
            print("📚 Needs more training")

    except KeyboardInterrupt:
        print("\n\n⏹️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        print("💡 Check that your model and wrapper configurations are compatible")
        import traceback

        traceback.print_exc()
    finally:
        env.close()
        print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
