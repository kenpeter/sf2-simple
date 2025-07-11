#!/usr/bin/env python3
"""
Clean Multi-Round Training Script for Street Fighter AI
Handles matches with up to 3 rounds each
"""

import argparse
import numpy as np
import torch
import random
from tqdm import tqdm
import os
import glob

# Import from wrapper.py
from wrapper import (
    make_multi_round_env,
    EnergyBasedStreetFighterVerifier,
    StabilizedEnergyBasedAgent,
    EnergyStabilityManager,
    ExperienceBuffer,
    CheckpointManager,
    EnergyBasedTrainer,
    calculate_match_win_rate,
    format_match_result,
    calculate_experience_quality,
    create_experience_tuple,
    safe_mean,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Street Fighter AI (Multi-Round)"
    )
    parser.add_argument(
        "--total-matches", type=int, default=500, help="Total training matches"
    )
    parser.add_argument(
        "--max-rounds", type=int, default=3, help="Maximum rounds per match"
    )
    parser.add_argument(
        "--save-freq", type=int, default=50, help="Save checkpoint frequency"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--thinking-steps", type=int, default=3, help="Number of thinking steps"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Training batch size"
    )
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument(
        "--load-checkpoint", type=str, help="Path to checkpoint to load"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from latest checkpoint"
    )
    return parser.parse_args()


def find_latest_checkpoint(checkpoint_dir="checkpoints"):
    """Find the latest checkpoint file."""
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "model_ep*.zip"))
    if not checkpoint_files:
        return None

    episodes = []
    for file in checkpoint_files:
        try:
            basename = os.path.basename(file)
            episode_str = basename.replace("model_ep", "").replace(".zip", "")
            episodes.append((int(episode_str), file))
        except ValueError:
            continue

    if not episodes:
        return None

    episodes.sort(key=lambda x: x[0])
    return episodes[-1][1]


def main(args):
    # Setup
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"🔩 Using device: {device}")
    print(f"🥊 Training Configuration:")
    print(f"   Total Matches: {args.total_matches}")
    print(f"   Max Rounds per Match: {args.max_rounds}")
    print(f"   First to win 2 rounds wins the match")
    print(f"   Save frequency: Every {args.save_freq} matches")

    # Initialize environment and models
    render_mode = "human" if args.render else None
    env = make_multi_round_env(render_mode=render_mode, max_rounds=args.max_rounds)

    verifier = EnergyBasedStreetFighterVerifier(
        observation_space=env.observation_space,
        action_space=env.action_space,
        features_dim=256,
    ).to(device)

    agent = StabilizedEnergyBasedAgent(
        verifier=verifier,
        thinking_steps=args.thinking_steps,
        thinking_lr=0.05,
        noise_scale=0.01,
    )

    # Initialize training components
    stability_manager = EnergyStabilityManager(
        initial_lr=args.learning_rate, thinking_lr=0.05
    )
    experience_buffer = ExperienceBuffer(capacity=50000, quality_threshold=0.6)
    checkpoint_manager = CheckpointManager()
    trainer = EnergyBasedTrainer(
        verifier=verifier,
        agent=agent,
        device=device,
        lr=args.learning_rate,
        batch_size=args.batch_size,
    )

    # Handle checkpoints
    start_match = 1
    if args.resume:
        latest_checkpoint = find_latest_checkpoint()
        if latest_checkpoint:
            print(f"🔄 Resume mode: Found latest checkpoint {latest_checkpoint}")
            data = checkpoint_manager.load_checkpoint(
                latest_checkpoint, verifier, agent
            )
            start_match = data.get("episode", 1) + 1
            loaded_win_rate = data.get("win_rate", 0.0)
            print(
                f"📂 Resumed from match {data.get('episode', 0)} (Win Rate: {loaded_win_rate:.3f})"
            )
        else:
            print("🔄 Resume mode: No checkpoints found, starting fresh training")
    elif args.load_checkpoint:
        if os.path.exists(args.load_checkpoint):
            data = checkpoint_manager.load_checkpoint(
                args.load_checkpoint, verifier, agent
            )
            start_match = data.get("episode", 1) + 1
            loaded_win_rate = data.get("win_rate", 0.0)
            print(
                f"📂 Loaded checkpoint from match {data.get('episode', 0)} (Win Rate: {loaded_win_rate:.3f})"
            )
        else:
            print(f"⚠️  Checkpoint file not found: {args.load_checkpoint}")

    # Training metrics
    match_rewards = []
    match_win_rates = []
    energy_qualities = []
    early_stop_rates = []
    match_results = []

    print("========== 🚀 Starting Multi-Round Training ==========")
    print("🥊 Each match = Best of 3 rounds")
    print("🏆 Win Rate = Matches won / Total matches")

    try:
        for match_num in range(start_match, args.total_matches + 1):
            obs, info = env.reset()
            match_reward = 0
            steps = 0
            early_stops = 0
            max_steps = 54000  # Increased for multi-round matches

            print(f"\n🥊 Match {match_num}/{args.total_matches} Starting")

            # Match loop
            with tqdm(
                total=max_steps,
                desc=f"Match {match_num}/{args.total_matches}",
                leave=False,
            ) as pbar:

                while steps < max_steps and not info.get("match_finished", False):
                    # Agent prediction
                    action, agent_info = agent.predict(obs, deterministic=False)

                    # Environment step
                    next_obs, reward, done, truncated, info = env.step(action)

                    # Update tracking
                    match_reward += reward
                    steps += 1

                    # Show round completions
                    if info.get("round_finished", False) and info.get("round_winner"):
                        round_num = info.get("current_round", 1)
                        round_winner = info.get("round_winner")
                        player_rounds = info.get("player_rounds_won", 0)
                        opponent_rounds = info.get("opponent_rounds_won", 0)

                        result_emoji = (
                            "🏆"
                            if round_winner == "player"
                            else "💀" if round_winner == "opponent" else "🤝"
                        )
                        pbar.set_description(
                            f"Match {match_num} - Round: {result_emoji} {round_winner.upper()} | Score: {player_rounds}-{opponent_rounds}"
                        )

                    # Check for early termination
                    if done or truncated:
                        if steps < max_steps * 0.6:  # Early termination
                            early_stops += 1

                        # If match is finished, break
                        if info.get("match_finished", False):
                            break

                    obs = next_obs
                    pbar.update(1)

            # Process match results
            match_info = {
                "match_num": match_num,
                "match_winner": info.get("match_winner", "unknown"),
                "player_rounds_won": info.get("player_rounds_won", 0),
                "opponent_rounds_won": info.get("opponent_rounds_won", 0),
                "round_results": info.get("round_results", []),
                "total_reward": match_reward,
                "total_steps": steps,
                "match_finished": info.get("match_finished", False),
            }
            match_results.append(match_info)

            # Display match result
            match_result_str = format_match_result(info)
            print(f"🏁 Match {match_num}: {match_result_str}")
            print(f"   Reward: {match_reward:.1f} | Steps: {steps}")

            # Add experience to buffer
            if steps > 0:
                quality = calculate_experience_quality(info, match_reward)
                alt_action = np.random.randint(0, env.action_space.n)

                if match_reward >= 0:
                    good_action, bad_action = action, alt_action
                else:
                    good_action, bad_action = alt_action, action

                experience = create_experience_tuple(obs, good_action, bad_action)
                experience_buffer.add_experience(experience, quality)

            # Training step
            if len(experience_buffer.buffer) >= args.batch_size:
                batch = experience_buffer.sample_batch(args.batch_size)
                if batch:
                    try:
                        loss, energy_sep, energy_qual = trainer.train_step(batch)
                        energy_qualities.append(energy_qual)
                    except Exception as e:
                        print(f"⚠️  Training step failed: {e}")
                        energy_qualities.append(0.0)
            else:
                energy_qualities.append(0.0)

            # Calculate metrics
            match_rewards.append(match_reward)
            win_rate = calculate_match_win_rate(info.get("match_winner", "draw"))
            match_win_rates.append(win_rate)

            early_stop_rate = early_stops / max(steps, 1)
            early_stop_rates.append(early_stop_rate)

            # Get buffer stats
            buffer_stats = experience_buffer.get_stats()

            # Update stability manager
            current_energy_quality = energy_qualities[-1] if energy_qualities else 0.0
            current_energy_separation = 0.1  # Placeholder

            emergency_triggered = stability_manager.update_metrics(
                win_rate=win_rate,
                energy_quality=current_energy_quality,
                energy_separation=current_energy_separation,
                early_stop_rate=early_stop_rate,
            )

            # Handle emergency protocol
            if emergency_triggered:
                print(f"🚨 EMERGENCY PROTOCOL TRIGGERED at match {match_num}")
                print(
                    f"   Reason: {stability_manager.consecutive_poor_episodes} consecutive poor episodes"
                )

                new_lr, new_thinking_lr = stability_manager.get_current_lrs()
                trainer.update_learning_rate(new_lr)
                agent.current_thinking_lr = new_thinking_lr
                print(
                    f"   🔧 Learning rates reduced: {new_lr:.2e} / {new_thinking_lr:.2e}"
                )

                old_buffer_size = len(experience_buffer.buffer)
                experience_buffer.emergency_purge(keep_ratio=0.3)
                new_buffer_size = len(experience_buffer.buffer)
                print(
                    f"   🧹 Buffer purged: {old_buffer_size} → {new_buffer_size} experiences"
                )

                current_acceptance = buffer_stats.get("acceptance_rate", 0.5)
                experience_buffer.adapt_quality_threshold(current_acceptance)
                print(
                    f"   🎯 Quality threshold adjusted to: {experience_buffer.quality_threshold:.2f}"
                )
            else:
                current_acceptance = buffer_stats.get("acceptance_rate", 0.5)
                experience_buffer.adapt_quality_threshold(current_acceptance)

            # Periodic reporting
            if match_num % 10 == 0:
                recent_rewards = (
                    match_rewards[-10:] if len(match_rewards) >= 10 else match_rewards
                )
                recent_win_rate = safe_mean(
                    (
                        match_win_rates[-10:]
                        if len(match_win_rates) >= 10
                        else match_win_rates
                    ),
                    0.0,
                )
                recent_energy_quality = safe_mean(
                    (
                        energy_qualities[-10:]
                        if len(energy_qualities) >= 10
                        else energy_qualities
                    ),
                    0.0,
                )

                # Calculate match statistics
                recent_results = (
                    match_results[-10:] if len(match_results) >= 10 else match_results
                )
                wins = sum(1 for r in recent_results if r["match_winner"] == "player")
                losses = sum(
                    1 for r in recent_results if r["match_winner"] == "opponent"
                )
                draws = sum(1 for r in recent_results if r["match_winner"] == "draw")

                print(f"\n📊 Match {match_num} Summary:")
                print(
                    f"   Average Reward (last 10): {safe_mean(recent_rewards, 0.0):.2f}"
                )
                print(
                    f"   Match Win Rate (last 10): {recent_win_rate:.2f} ({wins}W-{losses}L-{draws}D)"
                )
                print(f"   Energy Quality: {recent_energy_quality:.2f}")
                print(f"   Buffer Size: {buffer_stats['size']}")
                print(
                    f"   Buffer Acceptance Rate: {buffer_stats.get('acceptance_rate', 0.0):.2f}"
                )

                # Show recent match results
                print("   📋 Last 5 Matches:")
                for r in recent_results[-5:]:
                    result_emoji = (
                        "🏆"
                        if r["match_winner"] == "player"
                        else "💀" if r["match_winner"] == "opponent" else "🤝"
                    )
                    rounds_str = "-".join(
                        [
                            (
                                "🏆"
                                if rr == "player"
                                else "💀" if rr == "opponent" else "🤝"
                            )
                            for rr in r["round_results"]
                        ]
                    )
                    print(
                        f"      Match {r['match_num']}: {result_emoji} {r['match_winner'].upper()} {r['player_rounds_won']}-{r['opponent_rounds_won']} [{rounds_str}]"
                    )

                # Emergency status
                emergency_status = (
                    "🚨 YES" if stability_manager.emergency_mode else "✅ NO"
                )
                print(f"   Emergency Mode: {emergency_status}")

                if stability_manager.emergency_mode:
                    current_lr, current_thinking_lr = (
                        stability_manager.get_current_lrs()
                    )
                    print(
                        f"   🔧 Reduced LR: {current_lr:.2e} (Thinking: {current_thinking_lr:.2e})"
                    )

                if stability_manager.consecutive_poor_episodes > 0:
                    print(
                        f"   ⚠️ Poor Episodes Streak: {stability_manager.consecutive_poor_episodes}/5"
                    )

                # Quality threshold
                quality_threshold = experience_buffer.quality_threshold
                print(f"   🎯 Quality Threshold: {quality_threshold:.2f}")

                # Collapse risk assessment
                collapse_risk = 0
                if recent_win_rate < 0.2:
                    collapse_risk += 1
                if recent_energy_quality < 10.0:
                    collapse_risk += 1
                if buffer_stats.get("acceptance_rate", 1.0) < 0.3:
                    collapse_risk += 1

                if collapse_risk >= 2:
                    print(f"   🚨 COLLAPSE RISK: HIGH ({collapse_risk}/3 indicators)")
                elif collapse_risk == 1:
                    print(f"   ⚠️ COLLAPSE RISK: MEDIUM ({collapse_risk}/3 indicators)")
                else:
                    print(f"   ✅ COLLAPSE RISK: LOW ({collapse_risk}/3 indicators)")

            # Save checkpoint
            if match_num % args.save_freq == 0:
                current_win_rate = safe_mean(
                    (
                        match_win_rates[-20:]
                        if len(match_win_rates) >= 20
                        else match_win_rates
                    ),
                    0.0,
                )
                current_energy_quality = safe_mean(
                    (
                        energy_qualities[-20:]
                        if len(energy_qualities) >= 20
                        else energy_qualities
                    ),
                    0.0,
                )

                checkpoint_manager.save_checkpoint(
                    verifier=verifier,
                    agent=agent,
                    episode=match_num,
                    win_rate=current_win_rate,
                    energy_quality=current_energy_quality,
                )

                # Show overall statistics at checkpoint
                total_wins = sum(
                    1 for r in match_results if r["match_winner"] == "player"
                )
                total_losses = sum(
                    1 for r in match_results if r["match_winner"] == "opponent"
                )
                total_draws = sum(
                    1 for r in match_results if r["match_winner"] == "draw"
                )
                overall_win_rate = (
                    total_wins / len(match_results) if match_results else 0.0
                )

                print(f"💾 Checkpoint saved at match {match_num}")
                print(
                    f"   📈 Overall Record: {total_wins}W-{total_losses}L-{total_draws}D (Win Rate: {overall_win_rate:.3f})"
                )

                # Round-level statistics
                total_player_rounds = sum(r["player_rounds_won"] for r in match_results)
                total_opponent_rounds = sum(
                    r["opponent_rounds_won"] for r in match_results
                )
                total_rounds_played = total_player_rounds + total_opponent_rounds
                round_win_rate = (
                    total_player_rounds / total_rounds_played
                    if total_rounds_played > 0
                    else 0.0
                )

                print(
                    f"   🥊 Round Stats: {total_player_rounds}W-{total_opponent_rounds}L (Round Win Rate: {round_win_rate:.3f})"
                )

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        env.close()

        # Final statistics
        if match_results:
            total_wins = sum(1 for r in match_results if r["match_winner"] == "player")
            total_losses = sum(
                1 for r in match_results if r["match_winner"] == "opponent"
            )
            total_draws = sum(1 for r in match_results if r["match_winner"] == "draw")
            overall_win_rate = total_wins / len(match_results)

            # Round statistics
            total_player_rounds = sum(r["player_rounds_won"] for r in match_results)
            total_opponent_rounds = sum(r["opponent_rounds_won"] for r in match_results)
            total_rounds_played = total_player_rounds + total_opponent_rounds
            round_win_rate = (
                total_player_rounds / total_rounds_played
                if total_rounds_played > 0
                else 0.0
            )

            print(f"\n🏁 Training completed")
            print(f"📊 Final Match Statistics:")
            print(f"   Total Matches: {len(match_results)}")
            print(f"   Wins: {total_wins} ({total_wins/len(match_results)*100:.1f}%)")
            print(
                f"   Losses: {total_losses} ({total_losses/len(match_results)*100:.1f}%)"
            )
            print(
                f"   Draws: {total_draws} ({total_draws/len(match_results)*100:.1f}%)"
            )
            print(f"   Final Match Win Rate: {overall_win_rate:.3f}")

            print(f"\n🥊 Final Round Statistics:")
            print(f"   Total Rounds Played: {total_rounds_played}")
            print(
                f"   Player Rounds Won: {total_player_rounds} ({round_win_rate*100:.1f}%)"
            )
            print(
                f"   Opponent Rounds Won: {total_opponent_rounds} ({(1-round_win_rate)*100:.1f}%)"
            )
            print(f"   Round Win Rate: {round_win_rate:.3f}")

            # Show match type distribution
            decisive_wins = sum(
                1
                for r in match_results
                if r["match_winner"] == "player"
                and r["player_rounds_won"] == 2
                and r["opponent_rounds_won"] == 0
            )
            close_wins = sum(
                1
                for r in match_results
                if r["match_winner"] == "player" and r["opponent_rounds_won"] > 0
            )

            print(f"\n🎯 Match Type Analysis:")
            print(f"   Decisive Wins (2-0): {decisive_wins}")
            print(f"   Close Wins (2-1): {close_wins}")

            if total_wins > 0:
                print(
                    f"   Dominance Rate: {decisive_wins/total_wins*100:.1f}% of wins were decisive"
                )


if __name__ == "__main__":
    args = parse_args()

    # Print training mode
    if args.resume:
        print("🔄 Training mode: RESUME from latest checkpoint")
    elif args.load_checkpoint:
        print(f"📂 Training mode: LOAD from {args.load_checkpoint}")
    else:
        print("🆕 Training mode: NEW training session")

    main(args)
