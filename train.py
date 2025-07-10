import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
from tqdm import tqdm
from pathlib import Path

# Import all necessary components from the single, unified wrapper.py
from wrapper import (
    make_stabilized_env,
    EnergyBasedStreetFighterVerifier,
    StabilizedEnergyBasedAgent,
    EnergyStabilityManager,
    ExperienceBuffer,
    CheckpointManager,
    EnergyBasedTrainer,
)


def find_latest_checkpoint(checkpoint_dir="checkpoints"):
    """Finds the most recent checkpoint file in the given directory."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = list(checkpoint_dir.glob("*.pt"))
    if not checkpoints:
        return None
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    return latest_checkpoint


def main(args):
    """
    Main training function for the Stabilized Energy-Based Transformer.
    """
    # --- 1. Setup and Initialization ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔩 Using device: {device}")

    # Create the stabilized environment
    env = make_stabilized_env(render_mode="human" if args.render else None)
    obs_space = env.observation_space
    action_space = env.action_space

    # Instantiate the core models
    verifier = EnergyBasedStreetFighterVerifier(obs_space, action_space).to(device)
    agent = StabilizedEnergyBasedAgent(
        verifier=verifier,
        thinking_steps=args.thinking_steps,
        thinking_lr=args.thinking_lr,
        noise_scale=0.01,
    )

    # Instantiate the stability and management tools
    stability_manager = EnergyStabilityManager(
        initial_lr=args.learning_rate, thinking_lr=args.thinking_lr
    )
    experience_buffer = ExperienceBuffer(
        capacity=args.buffer_size, quality_threshold=0.3
    )
    checkpoint_manager = CheckpointManager(checkpoint_dir="checkpoints")
    trainer = EnergyBasedTrainer(
        verifier=verifier,
        agent=agent,
        device=device,
        lr=args.learning_rate,
        batch_size=args.batch_size,
    )

    # --- 2. Checkpoint Loading Logic (with --resume) ---
    start_episode = 0
    checkpoint_to_load = None

    if args.resume:
        print("🔄 Resume flag detected. Searching for the latest checkpoint...")
        checkpoint_to_load = find_latest_checkpoint()
        if checkpoint_to_load:
            print(f"   Found latest checkpoint: {checkpoint_to_load.name}")
        else:
            print("   ⚠️ No checkpoints found to resume from. Starting a fresh run.")
    elif args.load_checkpoint:
        checkpoint_to_load = Path(args.load_checkpoint)
        if not checkpoint_to_load.exists():
            print(
                f"   ⚠️ Specified checkpoint not found at {checkpoint_to_load}. Starting fresh."
            )
            checkpoint_to_load = None

    if checkpoint_to_load:
        print(f"💾 Loading checkpoint from: {checkpoint_to_load}")
        checkpoint_data = checkpoint_manager.load_checkpoint(
            checkpoint_to_load, verifier, agent
        )
        if checkpoint_data:
            start_episode = checkpoint_data.get("episode", 0) + 1
            print(f"🔄 Resuming training from episode {start_episode}")
    else:
        print(" Bắt đầu một khóa đào tạo mới.")

    # --- 3. Main Training Loop ---
    print("\n" + "=" * 10 + " 🚀 Starting Training " + "=" * 10)
    episode_losses = deque(maxlen=100)
    episode_energy_qualities = deque(maxlen=100)
    episode_energy_separations = deque(maxlen=100)

    for episode in range(start_episode, args.total_episodes):
        obs, info = env.reset()
        done, truncated = False, False
        episode_reward = 0
        steps = 0

        pbar = tqdm(
            total=env.max_episode_steps,
            desc=f"Ep {episode+1}/{args.total_episodes}",
            unit="step",
        )

        while not done and not truncated:
            with torch.no_grad():
                obs_tensor = {
                    k: torch.from_numpy(v).unsqueeze(0).to(device)
                    for k, v in obs.items()
                }
                action, _ = agent.predict(obs_tensor, deterministic=False)

            next_obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            pbar.update(1)

            good_action_one_hot = F.one_hot(
                torch.tensor(action), num_classes=action_space.n
            ).numpy()
            bad_action = action_space.sample()
            while bad_action == action:
                bad_action = action_space.sample()
            bad_action_one_hot = F.one_hot(
                torch.tensor(bad_action), num_classes=action_space.n
            ).numpy()

            quality_score = max(0, reward + 1.0)
            experience_buffer.add_experience(
                (obs, good_action_one_hot, bad_action_one_hot), quality_score
            )
            obs = next_obs

            if len(experience_buffer.buffer) > args.batch_size:
                batch, _ = experience_buffer.sample_batch(args.batch_size)
                loss, es, eq = trainer.train_step(batch)
                episode_losses.append(loss)
                episode_energy_separations.append(es)
                episode_energy_qualities.append(eq)

        pbar.close()

        # --- 4. End-of-Episode Management ---
        win_rate = info.get("win_rate", 0.0)
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        avg_eq = np.mean(episode_energy_qualities) if episode_energy_qualities else 0.0
        avg_es = (
            np.mean(episode_energy_separations) if episode_energy_separations else 0.0
        )

        thinking_stats = agent.get_thinking_stats()
        early_stop_rate = thinking_stats.get("early_stop_rate", 0.0)

        print(
            f"\n--- Episode {episode + 1} Summary --- "
            f"Win Rate: {win_rate:.2%} | Steps: {steps} | "
            f"Loss: {avg_loss:.4f} | EQ: {avg_eq:.2f} | ES: {avg_es:.3f} | "
            f"Buffer: {len(experience_buffer.buffer)}/{experience_buffer.capacity}"
        )

        was_emergency = stability_manager.update_metrics(
            win_rate, avg_eq, avg_es, early_stop_rate
        )

        if was_emergency:
            print("🚨 Emergency Detected! Restoring and adjusting...")
            checkpoint_manager.emergency_restore(verifier, agent)
            experience_buffer.emergency_purge(keep_ratio=0.25)
            new_lr, new_thinking_lr = stability_manager.get_current_lrs()
            trainer.update_learning_rate(new_lr)
            agent.current_thinking_lr = new_thinking_lr
        else:
            stability_manager.recovery_check(win_rate)

        buffer_stats = experience_buffer.get_stats()
        experience_buffer.adapt_quality_threshold(buffer_stats["acceptance_rate"])

        if (episode + 1) % args.save_freq == 0:
            checkpoint_manager.save_checkpoint(
                verifier, agent, episode, win_rate, avg_eq
            )

    print("\n" + "=" * 10 + " ✅ Training Complete " + "=" * 10)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Stabilized Energy-Based Transformer for Street Fighter"
    )
    parser.add_argument(
        "--total-episodes",
        type=int,
        default=1000,
        help="Total number of episodes to train for.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-5,
        help="Initial learning rate for the optimizer.",
    )
    parser.add_argument(
        "--thinking-steps",
        type=int,
        default=2,
        help="Number of optimization steps in the agent's thinking process.",
    )
    parser.add_argument(
        "--thinking-lr",
        type=float,
        default=0.05,
        help="Learning rate for the agent's thinking process.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for training."
    )
    parser.add_argument(
        "--buffer-size",
        type=int,
        default=10000,
        help="Size of the experience replay buffer.",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=25,
        help="How often to save a checkpoint (in episodes).",
    )

    # --- NEW & UPDATED ARGUMENTS ---
    parser.add_argument(
        "--load-checkpoint",
        type=str,
        default=None,
        help="Path to a specific checkpoint to load.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint in the checkpoints directory.",
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the game during training."
    )

    args = parser.parse_args()
    main(args)
