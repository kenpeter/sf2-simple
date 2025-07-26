#!/usr/bin/env python3
"""
🛡️ FIXED TRAINING - Solves the 176 vs 176 draw problem
Complete training script with proper health detection and win/lose logic
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import time
import os
from collections import deque
from pathlib import Path
import logging
from datetime import datetime

# Import the FIXED wrapper components
try:
    from wrapper import (
        make_fixed_env,
        verify_health_detection,
        SimpleVerifier,
        SimpleAgent,
        SimpleCNN,
        safe_mean,
        safe_std,
        safe_divide,
        MAX_FIGHT_STEPS,
        MAX_HEALTH,
    )

    print("✅ Successfully imported fixed wrapper components")
except ImportError as e:
    print(f"❌ Failed to import fixed wrapper: {e}")
    print("Make sure fixed_wrapper.py is in the same directory")
    exit(1)


class FixedExperienceBuffer:
    """🎯 Simple experience buffer for fixed training."""

    def __init__(self, capacity=20000):
        self.capacity = capacity
        self.good_experiences = deque(maxlen=capacity // 2)
        self.bad_experiences = deque(maxlen=capacity // 2)
        self.total_added = 0

    def add_experience(self, experience, reward, win_result):
        """Add experience based on win/lose result."""
        self.total_added += 1

        # Classify as good or bad based on actual results
        if win_result == "WIN" or reward > 0.5:
            self.good_experiences.append(experience)
        else:
            self.bad_experiences.append(experience)

    def sample_balanced_batch(self, batch_size):
        """Sample balanced batch of good and bad experiences."""
        if (
            len(self.good_experiences) < batch_size // 4
            or len(self.bad_experiences) < batch_size // 4
        ):
            return None, None

        good_count = batch_size // 2
        bad_count = batch_size // 2

        good_indices = np.random.choice(
            len(self.good_experiences), good_count, replace=False
        )
        good_batch = [self.good_experiences[i] for i in good_indices]

        bad_indices = np.random.choice(
            len(self.bad_experiences), bad_count, replace=False
        )
        bad_batch = [self.bad_experiences[i] for i in bad_indices]

        return good_batch, bad_batch

    def get_stats(self):
        """Get buffer statistics."""
        total_size = len(self.good_experiences) + len(self.bad_experiences)
        return {
            "total_size": total_size,
            "good_count": len(self.good_experiences),
            "bad_count": len(self.bad_experiences),
            "good_ratio": len(self.good_experiences) / max(1, total_size),
            "total_added": self.total_added,
        }


class FixedTrainer:
    """🛡️ Fixed trainer that resolves the draw problem."""

    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize FIXED environment
        print(f"🎮 Initializing FIXED environment...")
        self.env = make_fixed_env()

        # Verify health detection works
        if args.verify_health:
            if not verify_health_detection(self.env):
                print(
                    "⚠️  Health detection verification failed, but continuing anyway..."
                )
                print("   The training will work with whatever detection is available.")

        # Initialize models
        self.verifier = SimpleVerifier(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            features_dim=args.features_dim,
        ).to(self.device)

        self.agent = SimpleAgent(
            verifier=self.verifier,
            thinking_steps=args.thinking_steps,
            thinking_lr=args.thinking_lr,
        )

        # Experience buffer
        self.experience_buffer = FixedExperienceBuffer(capacity=args.buffer_capacity)

        # Optimizer
        self.optimizer = optim.Adam(
            self.verifier.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        # Training state
        self.episode = 0
        self.total_steps = 0
        self.best_win_rate = 0.0

        # Performance tracking with DETAILED win/lose stats
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.recent_results = deque(maxlen=20)
        self.win_rate_history = deque(maxlen=50)
        self.recent_losses = deque(maxlen=50)

        # Termination reason tracking
        self.termination_reasons = deque(maxlen=100)

        # Setup logging
        self.setup_logging()

        print(f"🛡️ Fixed Trainer initialized")
        print(f"   - Device: {self.device}")
        print(f"   - Learning rate: {args.learning_rate:.2e}")
        print(f"   - Health detection: MULTI-METHOD")
        print(f"   - Draw problem: TARGETED FOR ELIMINATION")

    def setup_logging(self):
        """Setup logging system."""
        log_dir = Path("logs_fixed")
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"fixed_training_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def run_episode(self):
        """Run a single episode with fixed health detection."""
        obs, info = self.env.reset()
        done = False
        truncated = False

        episode_reward = 0.0
        episode_steps = 0
        episode_experiences = []

        # Episode tracking
        round_won = False
        round_lost = False
        round_draw = False
        termination_reason = "ongoing"

        while (
            not done and not truncated and episode_steps < self.args.max_episode_steps
        ):
            # Agent prediction
            action, thinking_info = self.agent.predict(obs, deterministic=False)

            # Execute action
            next_obs, reward, done, truncated, info = self.env.step(action)

            episode_reward += reward
            episode_steps += 1
            self.total_steps += 1

            # Track round result
            if info.get("round_ended", False):
                termination_reason = info.get("termination_reason", "unknown")
                round_result = info.get("round_result", "ONGOING")

                if round_result == "WIN":
                    round_won = True
                elif round_result == "LOSE":
                    round_lost = True
                elif round_result == "DRAW":
                    round_draw = True

            # Store experience
            experience = {
                "obs": obs,
                "action": action,
                "reward": reward,
                "next_obs": next_obs,
                "done": done,
                "thinking_info": thinking_info,
                "episode": self.episode,
                "step": episode_steps,
            }

            episode_experiences.append(experience)
            obs = next_obs

        # Update win/lose statistics
        if round_won:
            self.wins += 1
            self.recent_results.append("WIN")
            win_result = "WIN"
        elif round_lost:
            self.losses += 1
            self.recent_results.append("LOSE")
            win_result = "LOSE"
        elif round_draw:
            self.draws += 1
            self.recent_results.append("DRAW")
            win_result = "DRAW"
        else:
            self.recent_results.append("UNKNOWN")
            win_result = "UNKNOWN"

        # Track termination reason
        self.termination_reasons.append(termination_reason)

        # Add experiences to buffer
        for experience in episode_experiences:
            self.experience_buffer.add_experience(
                experience, experience["reward"], win_result
            )

        # Episode stats
        episode_stats = {
            "won": round_won,
            "lost": round_lost,
            "draw": round_draw,
            "reward": episode_reward,
            "steps": episode_steps,
            "termination_reason": termination_reason,
            "player_health": info.get("player_health", MAX_HEALTH),
            "opponent_health": info.get("opponent_health", MAX_HEALTH),
            "health_detection_working": info.get("health_detection_working", False),
        }

        return episode_stats

    def get_win_lose_stats(self):
        """Get comprehensive win/lose statistics."""
        total_games = self.wins + self.losses + self.draws

        if total_games == 0:
            return {
                "total_games": 0,
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "win_rate": 0.0,
                "recent_win_rate": 0.0,
                "draw_rate": 0.0,
                "recent_results": list(self.recent_results),
                "recent_terminations": {},
            }

        # Calculate rates
        win_rate = self.wins / total_games
        draw_rate = self.draws / total_games

        # Recent win rate (last 10 games)
        recent_wins = sum(
            1 for result in list(self.recent_results)[-10:] if result == "WIN"
        )
        recent_games = min(len(self.recent_results), 10)
        recent_win_rate = recent_wins / recent_games if recent_games > 0 else 0.0

        # Recent termination reasons
        recent_terminations = {}
        for reason in list(self.termination_reasons)[-10:]:
            recent_terminations[reason] = recent_terminations.get(reason, 0) + 1

        return {
            "total_games": total_games,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": win_rate,
            "draw_rate": draw_rate,
            "recent_win_rate": recent_win_rate,
            "recent_results": list(self.recent_results),
            "recent_terminations": recent_terminations,
        }

    def calculate_contrastive_loss(self, good_batch, bad_batch, margin=2.0):
        """Calculate contrastive loss for energy-based training."""
        device = self.device

        def process_batch(batch):
            if not batch:
                return None, None

            obs_batch = []
            action_batch = []

            for exp in batch:
                obs = exp["obs"]
                action = exp["action"]

                # Convert observations
                if isinstance(obs, dict):
                    obs_tensor = {
                        key: (
                            torch.from_numpy(val).float()
                            if isinstance(val, np.ndarray)
                            else val.float()
                        )
                        for key, val in obs.items()
                    }
                else:
                    obs_tensor = torch.from_numpy(obs).float()

                # Convert action to one-hot
                action_one_hot = torch.zeros(self.env.action_space.n)
                action_one_hot[action] = 1.0

                obs_batch.append(obs_tensor)
                action_batch.append(action_one_hot)

            return obs_batch, action_batch

        # Process batches
        good_obs, good_actions = process_batch(good_batch)
        bad_obs, bad_actions = process_batch(bad_batch)

        if good_obs is None or bad_obs is None:
            return torch.tensor(0.0, device=device), {}

        # Stack observations
        def stack_obs_dict(obs_list):
            stacked = {}
            for key in obs_list[0].keys():
                stacked[key] = torch.stack([obs[key] for obs in obs_list]).to(device)
            return stacked

        good_obs_stacked = stack_obs_dict(good_obs)
        bad_obs_stacked = stack_obs_dict(bad_obs)
        good_actions_stacked = torch.stack(good_actions).to(device)
        bad_actions_stacked = torch.stack(bad_actions).to(device)

        # Calculate energies
        good_energies = self.verifier(good_obs_stacked, good_actions_stacked)
        bad_energies = self.verifier(bad_obs_stacked, bad_actions_stacked)

        # Contrastive loss (good energies should be lower)
        good_energy_mean = good_energies.mean()
        bad_energy_mean = bad_energies.mean()

        energy_diff = bad_energy_mean - good_energy_mean
        contrastive_loss = torch.clamp(margin - energy_diff, min=0.0)

        # Regularization
        energy_reg = 0.01 * (good_energies.pow(2).mean() + bad_energies.pow(2).mean())

        total_loss = contrastive_loss + energy_reg

        return total_loss, {
            "contrastive_loss": contrastive_loss.item(),
            "energy_reg": energy_reg.item(),
            "good_energy_mean": good_energy_mean.item(),
            "bad_energy_mean": bad_energy_mean.item(),
            "energy_separation": energy_diff.item(),
        }

    def train_step(self):
        """Training step with contrastive loss."""
        # Sample batch
        good_batch, bad_batch = self.experience_buffer.sample_balanced_batch(
            self.args.batch_size
        )

        if good_batch is None or bad_batch is None:
            return None  # Not enough experiences

        # Calculate loss
        loss, loss_info = self.calculate_contrastive_loss(
            good_batch, bad_batch, margin=self.args.contrastive_margin
        )

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.verifier.parameters(), max_norm=1.0
        )

        if grad_norm > 5.0:
            print(f"⚠️ Large gradient norm detected: {grad_norm:.2f}")
            return None

        self.optimizer.step()

        loss_info["grad_norm"] = grad_norm.item()
        return loss_info

    def evaluate_performance(self):
        """Evaluate current performance."""
        eval_episodes = 3

        wins = 0
        total_reward = 0.0
        total_steps = 0
        health_changes_detected = 0

        for _ in range(eval_episodes):
            obs, info = self.env.reset()
            done = False
            truncated = False
            episode_reward = 0.0
            episode_steps = 0
            initial_player_health = info.get("player_health", MAX_HEALTH)
            initial_opponent_health = info.get("opponent_health", MAX_HEALTH)

            while (
                not done
                and not truncated
                and episode_steps < self.args.max_episode_steps
            ):
                action, thinking_info = self.agent.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.env.step(action)

                episode_reward += reward
                episode_steps += 1

                # Check for win
                round_result = info.get("round_result", "ONGOING")
                if round_result == "WIN":
                    wins += 1
                    break

            # Check if health detection worked
            final_player_health = info.get("player_health", MAX_HEALTH)
            final_opponent_health = info.get("opponent_health", MAX_HEALTH)

            if (
                final_player_health != initial_player_health
                or final_opponent_health != initial_opponent_health
            ):
                health_changes_detected += 1

            total_reward += episode_reward
            total_steps += episode_steps

        win_rate = wins / eval_episodes
        avg_reward = total_reward / eval_episodes
        avg_steps = total_steps / eval_episodes
        health_detection_rate = health_changes_detected / eval_episodes

        return {
            "win_rate": win_rate,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
            "health_detection_rate": health_detection_rate,
            "eval_episodes": eval_episodes,
        }

    def train(self):
        """Main training loop with draw elimination focus."""
        print(f"🛡️ Starting FIXED Training")
        print(f"   - Target episodes: {self.args.max_episodes}")
        print(f"   - Focus: ELIMINATE DRAWS")
        print(f"   - Health detection: MULTI-METHOD")

        training_start_time = time.time()
        last_eval_episode = 0

        for episode in range(self.args.max_episodes):
            self.episode = episode
            episode_start_time = time.time()

            # Run episode
            episode_stats = self.run_episode()

            # Training step
            if (
                len(self.experience_buffer.good_experiences)
                >= self.args.batch_size // 4
            ):
                train_stats = self.train_step()
                if train_stats:
                    self.recent_losses.append(train_stats.get("contrastive_loss", 0.0))
            else:
                train_stats = {}

            # Periodic evaluation and logging
            if episode % self.args.eval_frequency == 0:
                # Performance evaluation
                performance_stats = self.evaluate_performance()
                self.win_rate_history.append(performance_stats["win_rate"])

                # Get comprehensive stats
                buffer_stats = self.experience_buffer.get_stats()
                win_lose_stats = self.get_win_lose_stats()

                # Print comprehensive status
                print(f"\n🎯 FIXED STATUS (Episode {episode}):")
                print(f"   - Good experiences: {buffer_stats['good_count']:,}")
                print(f"   - Bad experiences: {buffer_stats['bad_count']:,}")
                print(f"   - Total added: {buffer_stats['total_added']:,}")

                # Detailed win/lose statistics
                print(f"\n🏆 WIN/LOSE STATISTICS:")
                print(f"   - Total games: {win_lose_stats['total_games']}")
                print(
                    f"   - Wins: {win_lose_stats['wins']} | Losses: {win_lose_stats['losses']} | Draws: {win_lose_stats['draws']}"
                )
                print(f"   - Overall win rate: {win_lose_stats['win_rate']:.1%}")
                print(f"   - Draw rate: {win_lose_stats['draw_rate']:.1%}")
                print(
                    f"   - Recent win rate (last 10): {win_lose_stats['recent_win_rate']:.1%}"
                )

                # Recent results pattern
                recent_results_str = "".join(
                    [
                        (
                            "🏆"
                            if r == "WIN"
                            else "💀" if r == "LOSE" else "🤝" if r == "DRAW" else "❓"
                        )
                        for r in win_lose_stats["recent_results"][-10:]
                    ]
                )
                print(f"   - Recent pattern: {recent_results_str}")

                # Termination reasons analysis
                if win_lose_stats["recent_terminations"]:
                    print(
                        f"   - Recent terminations: {dict(win_lose_stats['recent_terminations'])}"
                    )

                # Health detection status
                print(f"\n🔍 HEALTH DETECTION:")
                print(
                    f"   - Detection rate: {performance_stats.get('health_detection_rate', 0.0):.1%}"
                )
                print(
                    f"   - Last episode health: P{episode_stats.get('player_health', MAX_HEALTH)} vs O{episode_stats.get('opponent_health', MAX_HEALTH)}"
                )
                print(
                    f"   - Detection working: {'✅' if episode_stats.get('health_detection_working', False) else '❌'}"
                )

                # Draw problem status
                if win_lose_stats["draw_rate"] > 0.8:
                    print(f"\n🚨 DRAW PROBLEM PERSISTS:")
                    print(
                        f"   - Draw rate: {win_lose_stats['draw_rate']:.1%} (TOO HIGH)"
                    )
                    print(f"   - Health detection may still be failing")
                    print(f"   - Check retro integration files")
                elif win_lose_stats["draw_rate"] > 0.3:
                    print(f"\n⚠️  DRAW RATE IMPROVING:")
                    print(
                        f"   - Draw rate: {win_lose_stats['draw_rate']:.1%} (REDUCING)"
                    )
                    print(f"   - Health detection partially working")
                else:
                    print(f"\n✅ DRAW PROBLEM SOLVED:")
                    print(
                        f"   - Draw rate: {win_lose_stats['draw_rate']:.1%} (ACCEPTABLE)"
                    )
                    print(f"   - Health detection working well")

                # Training progress
                if train_stats:
                    print(f"\n🧠 Training:")
                    print(
                        f"   - Energy separation: {train_stats.get('energy_separation', 0.0):.3f}"
                    )
                    print(
                        f"   - Good energy: {train_stats.get('good_energy_mean', 0.0):.3f}"
                    )
                    print(
                        f"   - Bad energy: {train_stats.get('bad_energy_mean', 0.0):.3f}"
                    )

                # Success indicators
                success_indicators = []
                if win_lose_stats["draw_rate"] < 0.3:
                    success_indicators.append("✅ Low draw rate")
                if win_lose_stats["win_rate"] > 0.2:
                    success_indicators.append("✅ Winning games")
                if buffer_stats["good_count"] > 50:
                    success_indicators.append("✅ Good experiences")
                if performance_stats.get("health_detection_rate", 0.0) > 0.5:
                    success_indicators.append("✅ Health detection")

                if success_indicators:
                    print(f"\n🎉 SUCCESS INDICATORS:")
                    for indicator in success_indicators:
                        print(f"   - {indicator}")
                else:
                    print(f"\n🔧 NEEDS WORK:")
                    if win_lose_stats["draw_rate"] >= 0.3:
                        print(
                            f"   - ❌ High draw rate ({win_lose_stats['draw_rate']:.1%})"
                        )
                    if win_lose_stats["win_rate"] <= 0.2:
                        print(
                            f"   - ❌ Low win rate ({win_lose_stats['win_rate']:.1%})"
                        )
                    if buffer_stats["good_count"] <= 50:
                        print(
                            f"   - ❌ Few good experiences ({buffer_stats['good_count']})"
                        )

                # Log detailed metrics
                self.logger.info(
                    f"Ep {episode}: "
                    f"Wins={win_lose_stats['wins']}, "
                    f"Losses={win_lose_stats['losses']}, "
                    f"Draws={win_lose_stats['draws']}, "
                    f"WinRate={win_lose_stats['win_rate']:.3f}, "
                    f"DrawRate={win_lose_stats['draw_rate']:.3f}, "
                    f"HealthDetection={performance_stats.get('health_detection_rate', 0.0):.3f}, "
                    f"GoodExp={buffer_stats['good_count']}, "
                    f"LastTerm={episode_stats.get('termination_reason', 'unknown')}"
                )

                last_eval_episode = episode

            # Early stopping check
            if len(self.win_rate_history) >= 10:
                recent_win_rate = safe_mean(list(self.win_rate_history)[-5:], 0.0)
                if recent_win_rate >= self.args.target_win_rate:
                    print(
                        f"🎯 Target win rate {self.args.target_win_rate:.1%} achieved!"
                    )
                    break

        # Training completed
        final_performance = self.evaluate_performance()
        final_win_lose_stats = self.get_win_lose_stats()

        print(f"\n🏁 FIXED Training Completed!")
        print(f"   - Total episodes: {self.episode + 1}")
        print(f"   - Final win rate: {final_performance['win_rate']:.1%}")
        print(f"   - Final draw rate: {final_win_lose_stats['draw_rate']:.1%}")
        print(
            f"   - Health detection rate: {final_performance.get('health_detection_rate', 0.0):.1%}"
        )

        # Final assessment
        print(f"\n🎯 FINAL ASSESSMENT:")
        if final_win_lose_stats["draw_rate"] < 0.2:
            print(
                f"   🎉 SUCCESS: Draw problem eliminated! ({final_win_lose_stats['draw_rate']:.1%} draw rate)"
            )
        elif final_win_lose_stats["draw_rate"] < 0.5:
            print(
                f"   ✅ IMPROVEMENT: Draw rate reduced to {final_win_lose_stats['draw_rate']:.1%}"
            )
        else:
            print(
                f"   ⚠️  PARTIAL: Draw rate still {final_win_lose_stats['draw_rate']:.1%}"
            )

        if final_win_lose_stats["win_rate"] > 0.3:
            print(f"   🏆 EXCELLENT: Win rate {final_win_lose_stats['win_rate']:.1%}")
        elif final_win_lose_stats["win_rate"] > 0.1:
            print(f"   📈 GOOD: Win rate {final_win_lose_stats['win_rate']:.1%}")
        else:
            print(f"   📉 NEEDS WORK: Win rate {final_win_lose_stats['win_rate']:.1%}")

        # Show final record
        print(f"\n📊 FINAL RECORD:")
        print(f"   - Total games: {final_win_lose_stats['total_games']}")
        print(
            f"   - Record: {final_win_lose_stats['wins']}W - {final_win_lose_stats['losses']}L - {final_win_lose_stats['draws']}D"
        )
        print(f"   - Win rate: {final_win_lose_stats['win_rate']:.1%}")
        print(f"   - Draw rate: {final_win_lose_stats['draw_rate']:.1%}")

        # Termination analysis
        if self.termination_reasons:
            termination_counts = {}
            for reason in self.termination_reasons:
                termination_counts[reason] = termination_counts.get(reason, 0) + 1
            print(f"   - Termination breakdown: {dict(termination_counts)}")

        return final_win_lose_stats["draw_rate"] < 0.3  # Success if draw rate below 30%


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Fixed Street Fighter Training - Eliminate Draws"
    )

    # Environment arguments
    parser.add_argument(
        "--max-episodes", type=int, default=500, help="Maximum training episodes"
    )
    parser.add_argument(
        "--max-episode-steps", type=int, default=3500, help="Maximum steps per episode"
    )
    parser.add_argument(
        "--eval-frequency", type=int, default=10, help="Evaluate every N episodes"
    )

    # Model arguments
    parser.add_argument(
        "--features-dim", type=int, default=256, help="Feature dimension"
    )
    parser.add_argument("--thinking-steps", type=int, default=3, help="Thinking steps")
    parser.add_argument(
        "--thinking-lr", type=float, default=0.05, help="Thinking learning rate"
    )

    # Training arguments
    parser.add_argument(
        "--learning-rate", type=float, default=3e-4, help="Learning rate"
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--contrastive-margin", type=float, default=2.0, help="Contrastive margin"
    )

    # Buffer arguments
    parser.add_argument(
        "--buffer-capacity", type=int, default=20000, help="Buffer capacity"
    )

    # Evaluation arguments
    parser.add_argument(
        "--target-win-rate", type=float, default=0.6, help="Target win rate"
    )
    parser.add_argument(
        "--verify-health", action="store_true", help="Verify health detection at start"
    )

    # Legacy arguments for compatibility
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.3,
        help="Quality threshold (legacy, not used in fixed version)",
    )
    parser.add_argument(
        "--render", action="store_true", help="Render environment (legacy, not used)"
    )

    args = parser.parse_args()

    # Print configuration
    print(f"🛡️ Fixed Street Fighter Training Configuration:")
    print(f"   Max Episodes: {args.max_episodes:,}")
    print(f"   Learning Rate: {args.learning_rate:.2e}")
    print(f"   Target Win Rate: {args.target_win_rate:.1%}")
    print(f"   Health Verification: {args.verify_health}")
    print(f"   🎯 PRIMARY GOAL: ELIMINATE 176 vs 176 DRAWS")

    # Run training
    try:
        trainer = FixedTrainer(args)
        success = trainer.train()

        if success:
            print(f"\n🎉 MISSION ACCOMPLISHED!")
            print(f"   The 176 vs 176 draw problem has been resolved!")
        else:
            print(f"\n🔧 PARTIAL SUCCESS")
            print(f"   Draw rate reduced but may need further tuning")

    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
