#!/usr/bin/env python3
"""
eval.py - Evaluation script for trained Street Fighter models
Load a trained model and watch it play for multiple rounds with performance statistics
"""

import os
import argparse
import torch
import numpy as np
import time
from datetime import datetime
from stable_baselines3 import PPO
import retro
import cv2

# Import components from wrapper
from wrapper import (
    StreetFighterVisionWrapper,
    FixedStreetFighterPolicy,
    verify_gradient_flow,
)


def make_eval_env(
    game="StreetFighterIISpecialChampionEdition-Genesis",
    state="ken_bison_12.state",
    render_mode="human",
):
    """Create evaluation environment with rendering enabled."""
    try:
        env = retro.make(game=game, state=state, render_mode=render_mode)
        env = StreetFighterVisionWrapper(env, frame_stack=8, rendering=True)
        return env
    except Exception as e:
        print(f"❌ Error creating environment: {e}")
        raise


class EvaluationStats:
    """Track and display evaluation statistics."""

    def __init__(self):
        self.reset_stats()

    def reset_stats(self):
        """Reset all statistics."""
        self.games_played = 0
        self.rounds_played = 0
        self.wins = 0
        self.losses = 0
        self.total_reward = 0
        self.total_damage_dealt = 0
        self.total_damage_received = 0
        self.round_rewards = []
        self.round_lengths = []
        self.max_combo_this_session = 0
        self.win_streak = 0
        self.current_streak = 0
        self.best_win_streak = 0
        self.session_start_time = datetime.now()

    def update_round_end(self, player_won, reward, round_length, info):
        """Update stats when a round ends."""
        self.rounds_played += 1
        self.total_reward += reward
        self.round_rewards.append(reward)
        self.round_lengths.append(round_length)

        # Update combo stats
        max_combo = info.get("max_combo", 0)
        self.max_combo_this_session = max(self.max_combo_this_session, max_combo)

        # Update damage stats
        damage_dealt = info.get("total_damage_dealt", 0)
        damage_received = info.get("total_damage_received", 0)
        self.total_damage_dealt += damage_dealt
        self.total_damage_received += damage_received

        if player_won:
            self.wins += 1
            self.current_streak += 1
            self.best_win_streak = max(self.best_win_streak, self.current_streak)
        else:
            self.losses += 1
            self.current_streak = 0

    def update_game_end(self):
        """Update stats when a game ends."""
        self.games_played += 1

    def get_win_rate(self):
        """Calculate current win rate."""
        total_games = self.wins + self.losses
        return (self.wins / total_games * 100) if total_games > 0 else 0

    def get_avg_reward(self):
        """Calculate average reward per round."""
        return np.mean(self.round_rewards) if self.round_rewards else 0

    def get_damage_ratio(self):
        """Calculate damage dealt vs received ratio."""
        return self.total_damage_dealt / max(1, self.total_damage_received)

    def print_current_stats(self):
        """Print current session statistics."""
        elapsed = datetime.now() - self.session_start_time
        elapsed_minutes = elapsed.total_seconds() / 60

        print(f"\n📊 CURRENT SESSION STATS")
        print(f"=" * 50)
        print(f"⏱️  Session time: {elapsed_minutes:.1f} minutes")
        print(f"🎮 Games played: {self.games_played}")
        print(f"🥊 Rounds played: {self.rounds_played}")
        print(f"🏆 Record: {self.wins}W - {self.losses}L")
        print(f"📈 Win rate: {self.get_win_rate():.1f}%")
        print(f"🔥 Current streak: {self.current_streak}")
        print(f"⭐ Best streak: {self.best_win_streak}")
        print(f"💰 Avg reward: {self.get_avg_reward():.1f}")
        print(f"🥷 Max combo: {self.max_combo_this_session}")
        print(f"⚔️  Damage ratio: {self.get_damage_ratio():.2f}")

        if self.round_lengths:
            avg_round_length = np.mean(self.round_lengths)
            print(f"⏰ Avg round length: {avg_round_length:.1f} steps")


def evaluate_model(
    model_path,
    num_rounds=10,
    render=True,
    verbose=True,
    game="StreetFighterIISpecialChampionEdition-Genesis",
    state="ken_bison_12.state",
    device="auto",
):
    """
    Evaluate a trained model for multiple rounds.

    Args:
        model_path: Path to the saved model (.zip file)
        num_rounds: Number of rounds to play (-1 for infinite)
        render: Whether to show the game visually
        verbose: Whether to print detailed stats
        game: Retro game name
        state: Game state to load
        device: Device to run on (auto, cpu, cuda)
    """

    print(f"🎮 STREET FIGHTER AI EVALUATION")
    print(f"=" * 50)
    print(f"📁 Model: {os.path.basename(model_path)}")
    print(f"🎯 Target rounds: {num_rounds if num_rounds > 0 else 'Infinite'}")
    print(f"👁️  Rendering: {'Enabled' if render else 'Disabled'}")

    # Device setup
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"🔧 Device: {device}")

    # Create environment
    render_mode = "human" if render else None
    env = make_eval_env(game, state, render_mode)

    # Load model
    print(f"\n📂 Loading model...")
    try:
        model = PPO.load(model_path, env=env, device=device)
        print(f"✅ Model loaded successfully")

        # Verify model device
        model_device = next(model.policy.parameters()).device
        print(f"🔧 Model device: {model_device}")

    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Optional gradient flow check
    print(f"\n🔬 Quick gradient flow check...")
    try:
        gradient_ok = verify_gradient_flow(model, env, device)
        if gradient_ok:
            print("✅ Model architecture looks healthy")
        else:
            print("⚠️  Some gradient flow issues detected (model may still work)")
    except Exception as e:
        print(f"⚠️  Gradient check failed: {e}")

    # Initialize statistics
    stats = EvaluationStats()

    print(f"\n🚀 Starting evaluation...")
    print(f"   Press Ctrl+C to stop early")
    print(f"   Stats will be shown every few rounds")

    try:
        rounds_completed = 0
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]

        round_start_time = time.time()
        step_count = 0

        while True:
            # Check if we've completed target rounds
            if num_rounds > 0 and rounds_completed >= num_rounds:
                print(f"\n🎯 Target of {num_rounds} rounds completed!")
                break

            # Get action from model
            action, _states = model.predict(obs, deterministic=True)

            # Take step
            obs, reward, done, truncated, info = env.step(action)
            step_count += 1

            # Render if enabled
            if render:
                env.render()
                # Small delay to make it watchable
                time.sleep(0.016)  # ~60 FPS

            # Check for round end
            if done or truncated:
                round_end_time = time.time()
                round_duration = round_end_time - round_start_time

                # Determine who won
                player_health = info.get("agent_hp", 0)
                opponent_health = info.get("enemy_hp", 0)
                player_won = player_health > 0 and opponent_health <= 0

                # Update statistics
                stats.update_round_end(player_won, reward, step_count, info)
                rounds_completed += 1

                # Print round result
                if verbose:
                    result = "WON! 🏆" if player_won else "LOST 💀"
                    win_rate = stats.get_win_rate()
                    print(
                        f"Round {rounds_completed}: {result} | "
                        f"Reward: {reward:.1f} | "
                        f"Steps: {step_count} | "
                        f"Win Rate: {win_rate:.1f}%"
                    )

                # Show detailed stats every 5 rounds
                if rounds_completed % 5 == 0:
                    stats.print_current_stats()

                # Reset for next round
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
                round_start_time = time.time()
                step_count = 0
                stats.update_game_end()

    except KeyboardInterrupt:
        print(f"\n⏹️  Evaluation stopped by user after {rounds_completed} rounds")

    except Exception as e:
        print(f"\n❌ Evaluation error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Final statistics
        print(f"\n" + "=" * 60)
        print(f"🏁 FINAL EVALUATION RESULTS")
        print(f"=" * 60)
        stats.print_current_stats()

        # Performance assessment
        win_rate = stats.get_win_rate()
        if win_rate >= 80:
            print(f"\n🌟 EXCELLENT performance! ({win_rate:.1f}% win rate)")
        elif win_rate >= 60:
            print(f"\n🔥 STRONG performance! ({win_rate:.1f}% win rate)")
        elif win_rate >= 40:
            print(f"\n👍 DECENT performance! ({win_rate:.1f}% win rate)")
        elif win_rate >= 20:
            print(f"\n📈 LEARNING performance! ({win_rate:.1f}% win rate)")
        else:
            print(f"\n💪 EARLY STAGE performance! ({win_rate:.1f}% win rate)")

        # Close environment
        env.close()
        print(f"\n🔚 Evaluation completed")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Street Fighter model"
    )
    parser.add_argument("model_path", help="Path to the model file (.zip)")
    parser.add_argument(
        "--rounds",
        type=int,
        default=10,
        help="Number of rounds to play (-1 for infinite)",
    )
    parser.add_argument(
        "--no-render", action="store_true", help="Disable visual rendering"
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument(
        "--game",
        type=str,
        default="StreetFighterIISpecialChampionEdition-Genesis",
        help="Retro game name",
    )
    parser.add_argument(
        "--state", type=str, default="ken_bison_12.state", help="Game state to load"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)"
    )
    parser.add_argument(
        "--fast", action="store_true", help="Run at maximum speed (no rendering delay)"
    )

    args = parser.parse_args()

    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"❌ Model file not found: {args.model_path}")
        return

    if not args.model_path.endswith(".zip"):
        print(f"❌ Model file should be a .zip file")
        return

    # Set rendering
    render = not args.no_render
    verbose = not args.quiet

    # Run evaluation
    evaluate_model(
        model_path=args.model_path,
        num_rounds=args.rounds,
        render=render,
        verbose=verbose,
        game=args.game,
        state=args.state,
        device=args.device,
    )


if __name__ == "__main__":
    main()
